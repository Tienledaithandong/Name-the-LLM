"""Class-specific intercepts for balanced accuracy (plan §7 L196).

Strict separation. For every outer fold of the frozen prompt-grouped splits the
18 class biases are tuned on **inner** grouped out-of-fold scores, produced by
models fit only on inner-training rows. Outer-validation rows never enter the
search, no test row is read, and nothing here is transductive.

Three arms are compared on the same outer rows:

``no_bias``          the adopted ``0.5 z(char) + 0.5 z(style)`` blend (reference)
``prior_log_offset`` analytic ``-log(class prior)`` from outer-train labels only;
                     a heuristic reference, not a calibrated posterior
``tuned``            coordinate ascent on inner-OOF balanced accuracy, then a
                     shrinkage factor chosen on the same inner OOF

    python experiments/class_bias.py --data-dir data --seeds 42 13

Writes ``artifacts/class_bias/results.json``. The official-only fallback is the
subject: the pool-MIX candidate's char member is fit with external rows, so a
bias tuned here must be re-measured on that arm before it could be adopted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.constants import LABELS, N_CLASSES  # noqa: E402
from src.data import (  # noqa: E402
    indices_to_labels,
    labels_to_indices,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.metrics import balanced_accuracy, mean_ci, paired_difference  # noqa: E402
from src.model import estimator_from_config  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402
from src.validation import make_splits, validate_splits  # noqa: E402

WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
BIAS_GRID = tuple(float(v) for v in np.linspace(-1.5, 1.5, 13))
SHRINK_GRID = (0.0, 0.25, 0.5, 1.0)
SEARCH_PASSES = 3
SEED_SOURCES = {
    42: ("artifacts/splits/splits_official_verified_s42.json", "artifacts/cv_identity/official"),
    13: ("artifacts/splits/splits_official_verified_s13.json", "artifacts/cv_identity/official_s13"),
}


def blend_scores(char: np.ndarray, style: np.ndarray) -> np.ndarray:
    return WEIGHTS["char_tfidf_logreg"] * normalize_scores(char) + WEIGHTS["style_only"] * normalize_scores(style)


def prior_log_offset(y: np.ndarray) -> np.ndarray:
    """``-log(prior)`` over the 18 labels; absent classes get offset 0."""
    support = np.bincount(np.asarray(y, dtype=np.int64), minlength=N_CLASSES).astype(np.float64)
    prior = np.where(support > 0, support / max(support.sum(), 1.0), 1.0)
    offset = -np.log(prior)
    offset[support == 0] = 0.0
    return offset - offset.mean()


def _ba(truth: np.ndarray, scores: np.ndarray) -> float:
    return balanced_accuracy(truth, indices_to_labels(scores.argmax(axis=1)))


def tune_bias(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    grid: tuple[float, ...] = BIAS_GRID,
    shrinks: tuple[float, ...] = SHRINK_GRID,
    passes: int = SEARCH_PASSES,
) -> tuple[np.ndarray, dict]:
    """Coordinate ascent on balanced accuracy, then shrink toward zero.

    Ties prefer the smaller absolute bias (and the smaller shrinkage), so a
    direction that does not improve the inner score leaves the bias at zero.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if scores.shape != (y.size, N_CLASSES):
        raise ValueError(f"expected ({y.size}, {N_CLASSES}) scores, got {scores.shape}")
    truth = indices_to_labels(y)
    bias = np.zeros(N_CLASSES, dtype=np.float64)
    base = _ba(truth, scores)
    for _ in range(passes):
        for c in range(N_CLASSES):
            best_value, best_score = bias[c], _ba(truth, scores + bias)
            for value in grid:
                if value == bias[c]:
                    continue
                trial = bias.copy()
                trial[c] = value
                score = _ba(truth, scores + trial)
                if score > best_score + 1e-12 or (
                    abs(score - best_score) <= 1e-12 and abs(value) < abs(best_value)
                ):
                    best_value, best_score = value, score
            bias[c] = best_value
    tuned = _ba(truth, scores + bias)
    shrink_scores = {float(s): _ba(truth, scores + s * bias) for s in shrinks}
    shrink = min(shrinks, key=lambda s: (-shrink_scores[float(s)], s))
    return bias * shrink, {
        "inner_balanced_accuracy_no_bias": base,
        "inner_balanced_accuracy_tuned": tuned,
        "inner_balanced_accuracy_shrunk": shrink_scores[float(shrink)],
        "inner_shrink_scores": {str(k): v for k, v in shrink_scores.items()},
        "shrinkage": float(shrink),
        "bias_l2_before_shrinkage": float(np.linalg.norm(bias)),
    }


def inner_oof(
    df,
    train_idx: np.ndarray,
    configs: list[dict],
    *,
    model_seed: int,
    inner_splits: int,
    inner_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Blended inner-OOF scores for ``train_idx`` rows, in local order."""
    sub = df.iloc[train_idx].reset_index(drop=True)
    inner, method = make_splits(sub, n_splits=inner_splits, seed=inner_seed)
    validate_splits(sub, inner)
    y_local = labels_to_indices(sub["model"].tolist())
    texts = sub["text"].tolist()
    groups = sub["prompt_id"].astype(str).to_numpy()
    out = np.full((len(sub), N_CLASSES), np.nan, dtype=np.float64)
    folds = []
    for index, (tr, va) in enumerate(inner):
        scores = {}
        for cfg in configs:
            model = estimator_from_config(cfg, seed=model_seed)
            model.fit([texts[i] for i in tr], y_local[tr], groups=groups[tr])
            scores[cfg["name"]] = model.decision_function([texts[i] for i in va])
        out[va] = blend_scores(scores["char_tfidf_logreg"], scores["style_only"])
        folds.append({"inner_fold": index, "n_train": int(tr.size), "n_valid": int(va.size)})
    if not np.isfinite(out).all():
        raise RuntimeError("inner folds do not cover every outer-training row")
    return out, [{"inner_method": method, "inner_seed": inner_seed, "folds": folds}]


def run_seed(
    df,
    seed: int,
    splits_path: Path,
    oof_dir: Path,
    configs: list[dict],
    *,
    model_seed: int,
    inner_splits: int,
    inner_seed: int,
) -> dict:
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(df)
    if splits.get("ordered_dataset_sha256") != fingerprint:
        raise SystemExit(f"[abort] seed {seed}: split fingerprint does not match the local train frame")
    if int(splits.get("seed", -1)) != seed:
        raise SystemExit(f"[abort] {splits_path} is not the seed-{seed} split file")
    members = list(WEIGHTS)
    oof = load_oof(oof_dir, members)
    folds_full = [
        (np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int)) for f in splits["folds"]
    ]
    n_rows = int(oof[members[0]]["y"].size)
    if n_rows != len(df):
        raise SystemExit(f"[abort] seed {seed}: OOF has {n_rows} rows but train has {len(df)}")
    expected = fold_assignments(n_rows, folds_full)
    for name in members:
        item = oof[name]
        if item["ordered_dataset_sha256"] != fingerprint:
            raise SystemExit(f"[abort] seed {seed} {name}: OOF fingerprint differs from the splits")
        if not np.array_equal(item["validation_fold_ids"], expected):
            raise SystemExit(f"[abort] seed {seed} {name}: OOF fold assignments differ from the splits")
    y = oof[members[0]]["y"]
    truth = indices_to_labels(y)
    outer_scores = blend_scores(oof["char_tfidf_logreg"]["scores"], oof["style_only"]["scores"])

    arms = ("no_bias", "prior_log_offset", "tuned")
    pred = {arm: np.full(n_rows, -1, dtype=np.int64) for arm in arms}
    per_fold = []
    started = time.perf_counter()
    for index, (train_idx, valid_idx) in enumerate(folds_full):
        if np.intersect1d(train_idx, valid_idx).size:
            raise RuntimeError("outer fold trains on its own validation rows")
        inner, inner_info = inner_oof(
            df, train_idx, configs, model_seed=model_seed,
            inner_splits=inner_splits, inner_seed=inner_seed + index,
        )
        bias, info = tune_bias(inner, y[train_idx])
        offset = prior_log_offset(y[train_idx])
        local_truth = truth[valid_idx]
        block = outer_scores[valid_idx]
        chosen = {"no_bias": np.zeros(N_CLASSES), "prior_log_offset": offset, "tuned": bias}
        fold_report = {
            "outer_fold": index,
            "n_train": int(train_idx.size),
            "n_valid": int(valid_idx.size),
            "inner": inner_info[0],
            "tuning": info,
            "tuned_bias": {LABELS[c]: float(bias[c]) for c in range(N_CLASSES)},
            "prior_log_offset_l2": float(np.linalg.norm(offset)),
            "balanced_accuracy": {},
        }
        for arm in arms:
            scores = block + chosen[arm]
            pred[arm][valid_idx] = scores.argmax(axis=1)
            fold_report["balanced_accuracy"][arm] = _ba(local_truth, scores)
        per_fold.append(fold_report)
        print(
            f"[seed {seed} outer {index}] no_bias {fold_report['balanced_accuracy']['no_bias']:.4f} "
            f"prior {fold_report['balanced_accuracy']['prior_log_offset']:.4f} "
            f"tuned {fold_report['balanced_accuracy']['tuned']:.4f} "
            f"(shrink {info['shrinkage']}, inner gain "
            f"{info['inner_balanced_accuracy_shrunk'] - info['inner_balanced_accuracy_no_bias']:+.4f})",
            flush=True,
        )
    if any((pred[arm] < 0).any() for arm in arms):
        raise RuntimeError("outer predictions do not cover every row")

    results = {}
    base_folds = [f["balanced_accuracy"]["no_bias"] for f in per_fold]
    for arm in arms:
        fold_scores = [f["balanced_accuracy"][arm] for f in per_fold]
        entry = {
            "pooled_balanced_accuracy": balanced_accuracy(truth, indices_to_labels(pred[arm])),
            "fold_scores": fold_scores,
            "fold_ci": mean_ci(fold_scores),
        }
        if arm != "no_bias":
            entry["paired_vs_no_bias"] = mean_ci(paired_difference(base_folds, fold_scores))
        results[arm] = entry
    inner_gain = float(np.mean([
        f["tuning"]["inner_balanced_accuracy_shrunk"] - f["tuning"]["inner_balanced_accuracy_no_bias"]
        for f in per_fold
    ]))
    outer_gain = results["tuned"]["paired_vs_no_bias"]["mean"]
    return {
        "split_seed": seed,
        "split_id": f"{splits['method']}-s{seed}-n{splits['n_splits']}",
        "splits": str(splits_path),
        "oof_dir": str(oof_dir),
        "ordered_dataset_sha256": fingerprint,
        "n_rows": n_rows,
        "arms": results,
        "folds": per_fold,
        "mean_inner_gain": inner_gain,
        "mean_outer_gain": outer_gain,
        "inner_minus_outer_gain": inner_gain - outer_gain,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def verdict(seeds: dict[int, dict]) -> dict:
    """Adopt only if the tuned arm beats the reference on every seed with CI > 0."""
    deltas = {
        seed: payload["arms"]["tuned"]["paired_vs_no_bias"] for seed, payload in seeds.items()
    }
    wins = all(d["mean"] > 0 and d["ci_low"] > 0 for d in deltas.values())
    return {
        "adopt_tuned_bias": bool(wins),
        "rule": "tuned arm must beat no_bias on every seed with a paired 95% CI excluding 0",
        "per_seed_paired_delta": {str(k): v for k, v in deltas.items()},
        "reason": (
            "every seed improves with a paired CI above 0"
            if wins
            else "at least one seed fails to improve with a paired CI above 0; keep the adopted blend unchanged"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--config", default="configs/submission.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 13])
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--inner-seed", type=int, default=212)
    ap.add_argument("--out", default="artifacts/class_bias/results.json")
    args = ap.parse_args()

    unknown = [s for s in args.seeds if s not in SEED_SOURCES]
    if unknown:
        raise SystemExit(f"[abort] no registered split/OOF source for seeds {unknown}")
    config = load_config(args.config)
    configs = config["models"]
    if [c["name"] for c in configs] != list(WEIGHTS):
        raise SystemExit(f"[abort] config must hold exactly {list(WEIGHTS)} in order")
    model_seed = int(config.get("seed", 42))
    df, _test = load_competition_data(args.data_dir)

    seeds: dict[int, dict] = {}
    for seed in args.seeds:
        splits_path, oof_dir = SEED_SOURCES[seed]
        seeds[seed] = run_seed(
            df, seed, Path(splits_path), Path(oof_dir), configs,
            model_seed=model_seed, inner_splits=args.inner_splits, inner_seed=args.inner_seed,
        )

    decision = verdict(seeds)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": env_snapshot(),
        "analysis": "class_specific_intercepts_nested_tuning",
        "plan_item": "§7 L196",
        "transductive": False,
        "test_rows_read": 0,
        "subject": "official-only fallback blend (char_tfidf_logreg + style_only, 0.5/0.5 row-z)",
        "blend_weights": WEIGHTS,
        "model_seed": model_seed,
        "config": args.config,
        "inner_splits": args.inner_splits,
        "inner_seed_base": args.inner_seed,
        "bias_grid": list(BIAS_GRID),
        "shrink_grid": list(SHRINK_GRID),
        "search_passes": SEARCH_PASSES,
        "separation": (
            "Per outer fold, biases are tuned only on inner grouped OOF from models fit on "
            "inner-training rows; outer-validation rows are scored once with the frozen bias. "
            "The prior arm uses outer-train label counts only."
        ),
        "seeds": {str(k): v for k, v in seeds.items()},
        "decision": decision,
        "caveats": [
            "The bias is tuned on inner models' score scale and transferred to the outer model's "
            "saved OOF scores; the two scales are close but not identical.",
            "Official train is near-balanced (support 149/150), so the prior arm is a near no-op by construction.",
            "The hidden test's class distribution is unknown; a bias tuned on train priors may not transfer.",
            "Measured on the official-only fallback. The pool-MIX candidate's char member is fit with "
            "external rows and must be re-measured before any bias could be adopted there.",
            "Prior development used all official prompts, so these folds are not an untouched final test (D024).",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print()
    for seed, payload_seed in seeds.items():
        print(f"seed {seed} ({payload_seed['split_id']}, {payload_seed['n_rows']} rows)")
        for arm, entry in payload_seed["arms"].items():
            delta = entry.get("paired_vs_no_bias")
            shown = "" if delta is None else f" | delta {delta['mean']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
            print(
                f"  {arm:17s} pooled BA {entry['pooled_balanced_accuracy']:.6f} "
                f"| fold mean {entry['fold_ci']['mean']:.4f} ± {entry['fold_ci']['std']:.4f}{shown}"
            )
        print(
            f"  inner gain {payload_seed['mean_inner_gain']:+.4f} vs outer gain "
            f"{payload_seed['mean_outer_gain']:+.4f} (optimism gap {payload_seed['inner_minus_outer_gain']:+.4f})"
        )
    print(f"\nadopt tuned bias: {decision['adopt_tuned_bias']} — {decision['reason']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()