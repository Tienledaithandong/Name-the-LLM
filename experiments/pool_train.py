"""Official-only vs official+pool training comparison (Phase 1b/1c).

For each model in a config, loads the official-only OOF produced by the harness
(``artifacts/cv/official/oof/<name>.npz``) and re-fits the model with the audited
clean Open Pool added to the TRAINING side of every grouped fold. Pool rows are
decontaminated per fold against that fold's official holdout
(``src.pool.decontaminate_fold``: exact dup, shared prompt, word TF-IDF cosine >
threshold fit on the holdout only). Validation stays on official rows only, so
the official+pool OOF is fold-aligned with the official-only OOF and the blend is
directly comparable. Reports pooled balanced accuracy, per-class recall and
within-provider errors, plus paired per-fold differences with a 95% Student-t CI.

Leakage guards (D019): only pool ``text``/``model``/``prompt_canon`` are read — the
upstream ``match_weight``/``style_score`` columns (fitted on ALL train prompts) are
NEVER used. Every learned transform (TF-IDF/SVD/style scaler/classifier) is fit on
the training side of each fold only. Pool rows never enter validation. No
transductive step. This is a training experiment, not a submission.

Run:
    python experiments/pool_train.py --config configs/gbdt.yaml --models gbdt_style
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

from src.config import load_config
from src.constants import ARTIFACTS_DIR, DEFAULT_REGISTRY, LABELS, N_CLASSES
from src.data import indices_to_labels, labels_to_indices, load_competition_data
from src.metrics import (
    aggregate_folds,
    evaluate,
    mean_ci,
    paired_difference,
    per_class_recall,
    within_provider_confusions,
)
from src.model import estimator_from_config
from src.pool import decontaminate_fold, load_pool
from src.repro import git_commit
from src.train import append_registry, describe
from src.validation import load_splits

ROOT = Path(__file__).resolve().parents[1]


def load_clean_pool(pool_path: str, clean_ids_path: str, *, strict: bool = False):
    pool = load_pool(pool_path, strict=strict)
    clean_ids = set(json.loads(Path(clean_ids_path).read_text(encoding="utf-8")))
    keep = pool["id"].astype(str).isin(clean_ids).to_numpy()
    pool = pool.loc[keep].reset_index(drop=True)
    return pool


def cap_pool_stratified(pool_df, cap: int, seed: int):
    """Subsample to <=cap rows preserving class proportions (CPU budget guard)."""
    if cap <= 0 or len(pool_df) <= cap:
        return pool_df
    rng = np.random.default_rng(seed)
    counts = pool_df["model"].value_counts()
    parts = []
    for label, n in counts.items():
        take = max(1, int(round(cap * n / len(pool_df))))
        idx = pool_df.index[pool_df["model"] == label].to_numpy()
        parts.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return pool_df.loc[np.sort(np.concatenate(parts))].reset_index(drop=True)


def official_oof(oof_dir: Path, name: str):
    data = np.load(oof_dir / f"{name}.npz", allow_pickle=True)
    scores = np.asarray(data["scores"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    labels = [str(v) for v in data["labels"]]
    if labels != list(LABELS):
        raise ValueError(f"{name}: OOF class order is not the official order")
    return scores, y


def fold_ba(scores: np.ndarray, y: np.ndarray, folds) -> list[float]:
    truth = indices_to_labels(y)
    out = []
    for _, valid_idx in folds:
        pred = indices_to_labels(scores[valid_idx].argmax(axis=1))
        out.append(evaluate(truth[valid_idx], pred)["balanced_accuracy"])
    return out


def pool_cv(train_df, pool_df, folds, raw, seed, cos_threshold, pool_cap=0):
    """Fit on official-train + decontaminated pool per fold; score official holdout."""
    texts = train_df["text"].tolist()
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y_idx = labels_to_indices(train_df["model"].tolist())
    y_labels = indices_to_labels(y_idx)

    scores = np.full((len(train_df), N_CLASSES), -1e9, dtype=np.float32)
    fold_metrics, decon = [], []
    fit_seconds = infer_seconds = 0.0

    for fold, (train_idx, valid_idx) in enumerate(folds):
        holdout = train_df.iloc[valid_idx]
        keep, stats = decontaminate_fold(pool_df, holdout, cos_threshold=cos_threshold)
        pool_keep = cap_pool_stratified(pool_df.loc[keep], pool_cap, seed + fold)
        stats["capped"] = int(len(pool_keep))
        decon.append({"fold": fold, **stats})
        pool_keep_texts = pool_keep["text"].tolist()
        pool_keep_y = labels_to_indices(pool_keep["model"].tolist())
        pool_keep_groups = pool_keep["prompt_canon"].astype(str).to_numpy()

        fit_texts = [texts[i] for i in train_idx] + pool_keep_texts
        fit_y = np.concatenate([y_idx[train_idx], pool_keep_y])
        fit_groups = np.concatenate([groups[train_idx], pool_keep_groups])

        estimator = estimator_from_config(raw, seed=seed)
        start = time.perf_counter()
        estimator.fit(fit_texts, fit_y, groups=fit_groups)
        fit_seconds += time.perf_counter() - start

        start = time.perf_counter()
        fold_scores = estimator.decision_function([texts[i] for i in valid_idx])
        infer_seconds += time.perf_counter() - start

        scores[valid_idx] = fold_scores
        pred = indices_to_labels(fold_scores.argmax(axis=1))
        fold_metrics.append(evaluate(y_labels[valid_idx], pred, model=f"{raw['name']}#fold{fold}"))
        print(
            f"  [fold {fold}] pool kept {stats['kept']} -> capped {stats['capped']} "
            f"(removed {stats['removed']}) | fit {fit_seconds:.1f}s | "
            f"BA {fold_metrics[-1]['balanced_accuracy']:.4f}",
            flush=True,
        )

    return scores, fold_metrics, decon, fit_seconds, infer_seconds


def compare(name, off_scores, off_y, pool_scores, folds, train_df):
    truth = indices_to_labels(off_y)
    off_pred = indices_to_labels(off_scores.argmax(axis=1))
    pool_pred = indices_to_labels(pool_scores.argmax(axis=1))
    off_eval = evaluate(truth, off_pred, model=f"{name}-official")
    pool_eval = evaluate(truth, pool_pred, model=f"{name}-pool")

    off_folds = fold_ba(off_scores, off_y, folds)
    pool_folds = fold_ba(pool_scores, off_y, folds)
    diffs = paired_difference(off_folds, pool_folds)
    ci = mean_ci(diffs)

    off_recall = per_class_recall(truth, off_pred)
    pool_recall = per_class_recall(truth, pool_pred)
    recall_delta = {
        k: (None if np.isnan(off_recall[k]) or np.isnan(pool_recall[k])
            else round(pool_recall[k] - off_recall[k], 4))
        for k in LABELS
    }
    return {
        "model": name,
        "official_only_ba": off_eval["balanced_accuracy"],
        "official_pool_ba": pool_eval["balanced_accuracy"],
        "ba_delta": pool_eval["balanced_accuracy"] - off_eval["balanced_accuracy"],
        "official_only_macro_f1": off_eval["macro_f1"],
        "official_pool_macro_f1": pool_eval["macro_f1"],
        "official_fold_ba": off_folds,
        "pool_fold_ba": pool_folds,
        "paired_fold_diff": list(diffs),
        "paired_diff_ci": ci,
        "official_per_class_recall": off_recall,
        "pool_per_class_recall": pool_recall,
        "per_class_recall_delta": recall_delta,
        "official_within_provider": within_provider_confusions(truth, off_pred),
        "pool_within_provider": within_provider_confusions(truth, pool_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/gbdt.yaml")
    parser.add_argument("--pool", default="artifacts/pool_audit/ntl_open_pool.parquet")
    parser.add_argument("--clean-ids", default="artifacts/pool_audit/clean_pool_ids.json")
    parser.add_argument("--oof-dir", default="artifacts/cv/official/oof")
    parser.add_argument("--splits", default="artifacts/splits/splits_official.json")
    parser.add_argument("--out", default="artifacts/pool_cv/official")
    parser.add_argument("--models", nargs="*", default=None, help="subset of model names")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-threshold", type=float, default=0.3)
    parser.add_argument("--pool-cap", type=int, default=0,
                        help="stratified cap on pool rows per fold (0=full); CPU budget guard")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_df, _ = load_competition_data(args.data_dir)
    folds, payload = load_splits(args.splits, train_df)
    split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
    pool_df = load_clean_pool(args.pool, args.clean_ids, strict=args.strict)
    print(f"official train {len(train_df)} rows | clean pool {len(pool_df)} rows | {split_id}",
          flush=True)

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["name"] in set(args.models)]
    if not models:
        raise SystemExit("no matching models in config")

    out_dir = Path(args.out)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)
    oof_dir = Path(args.oof_dir)
    commit = git_commit(ROOT)
    results = []

    for raw in models:
        name = str(raw["name"])
        print(f"\n=== {name} ({describe(raw)}) ===", flush=True)
        off_scores, off_y = official_oof(oof_dir, name)
        if not np.array_equal(off_y, labels_to_indices(train_df["model"].tolist())):
            raise ValueError(f"{name}: official OOF rows are not aligned with train")
        pool_scores, fold_metrics, decon, fit_s, infer_s = pool_cv(
            train_df, pool_df, folds, raw, args.seed, args.cos_threshold, args.pool_cap
        )
        cmp = compare(name, off_scores, off_y, pool_scores, folds, train_df)
        cmp["decontamination"] = decon
        cmp["fit_seconds"] = round(fit_s, 2)
        cmp["infer_seconds"] = round(infer_s, 2)
        results.append(cmp)

        np.savez_compressed(
            out_dir / "oof" / f"{name}.npz",
            scores=pool_scores, y=off_y, labels=np.asarray(LABELS, dtype=object),
        )
        agg = aggregate_folds(fold_metrics)
        ci = cmp["paired_diff_ci"]
        print(
            f"  official-only BA {cmp['official_only_ba']:.4f} -> +pool BA "
            f"{cmp['official_pool_ba']:.4f} (delta {cmp['ba_delta']:+.4f}) | "
            f"paired fold diff {ci['mean']:+.4f} +/- {ci['ci_half']:.4f} "
            f"[{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]",
            flush=True,
        )

        if not args.no_registry:
            append_registry(args.registry, {
                "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official_pool-{name}-s{args.seed}",
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": commit,
                "method": name,
                "features": describe(raw),
                "seed": args.seed,
                "split_id": split_id,
                "fold_scores": " ".join(f"{s:.4f}" for s in agg["fold_scores"]),
                "pooled_oof_balanced_accuracy": round(cmp["official_pool_ba"], 6),
                "macro_f1": round(float(cmp["official_pool_macro_f1"]), 6),
                "fit_seconds": cmp["fit_seconds"],
                "infer_seconds": cmp["infer_seconds"],
                "peak_ram_gb": "",
                "external_data_id": "open-pool-v15",
                "artifact_path": str(out_dir / "oof" / f"{name}.npz"),
                "public_lb_score": "",
                "notes": (f"official+pool (clean {len(pool_df)}, per-fold decon cos>{args.cos_threshold}, "
                          f"cap/fold={args.pool_cap or 'full'}); "
                          f"vs official-only {cmp['official_only_ba']:.4f}; "
                          f"paired fold diff {ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]"),
                "status": "done",
            })

    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    _write_report(out_dir / "comparison.md", results, len(pool_df), split_id, args.cos_threshold)
    print(f"\nwrote {out_dir / 'results.json'} and comparison.md", flush=True)


def _write_report(path, results, n_pool, split_id, cos_threshold) -> None:
    lines = [
        "# Official-only vs official+pool training comparison",
        "",
        f"- split: **{split_id}** (grouped folds; validation on official rows only)",
        f"- clean pool: **{n_pool}** rows (per-fold decontamination cos>{cos_threshold})",
        "- pool added to the TRAINING side only; match_weight/style_score never used (D019)",
        "",
        "| model | official-only BA | +pool BA | delta | paired fold diff [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        ci = r["paired_diff_ci"]
        lines.append(
            f"| `{r['model']}` | {r['official_only_ba']:.4f} | {r['official_pool_ba']:.4f} | "
            f"{r['ba_delta']:+.4f} | {ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}] |"
        )
    lines += ["", "## Per-class recall delta (+pool − official)", ""]
    for r in results:
        lines.append(f"### {r['model']}")
        lines.append("")
        lines.append("| label | official | +pool | delta |")
        lines.append("|---|---|---|---|")
        for k in LABELS:
            off = r["official_per_class_recall"][k]
            pool = r["pool_per_class_recall"][k]
            d = r["per_class_recall_delta"][k]
            off_s = "nan" if np.isnan(off) else f"{off:.3f}"
            pool_s = "nan" if np.isnan(pool) else f"{pool:.3f}"
            d_s = "n/a" if d is None else f"{d:+.3f}"
            lines.append(f"| {k} | {off_s} | {pool_s} | {d_s} |")
        wp_off = r["official_within_provider"]
        wp_pool = r["pool_within_provider"]
        lines += [
            "",
            f"within-provider errors: official {wp_off['within_provider_errors']}/{wp_off['errors']} "
            f"-> +pool {wp_pool['within_provider_errors']}/{wp_pool['errors']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()