"""Does choosing the blend weight beat the pre-specified 0.5/0.5? (D009 weights, never swept)

The adopted cap-40k pool-MIX candidate inherited ``0.5 z(char) + 0.5 z(style)`` from the
official-only blend. The char member got much stronger under the pool (0.3193 -> 0.4104)
while style stayed official-only, so the optimal mixing weight may have moved.

Two clearly separated outputs:

1. ``--curve-only`` (default) replays a weight grid on **saved, identity-verified OOF**.
   This is **descriptive sensitivity**, not a selection score: reading a peak off these
   curves and shipping it would be selection on the same OOF that reports the score.
2. ``--nested`` runs the honest protocol. For each grouped outer fold it refits both
   members on outer-train, selects the weight by **inner** grouped CV inside outer-train,
   and scores the untouched outer holdout with that weight and with the fixed 0.5. Only
   the paired outer-fold difference between those two recipes can support adoption.

The pool is screened once per outer fold against outer-train text only (D028-style,
fold-independent: the same keep-mask serves every inner fold), capped by class-proportional
subsampling, and is never scored. ``match_weight``/``style_score`` are never features (D019).
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
from src.constants import DEFAULT_REGISTRY, LABELS
from src.data import (indices_to_labels, labels_to_indices, load_competition_data,
                      ordered_dataset_fingerprint)
from src.ensemble import normalize_scores
from src.metrics import evaluate, mean_ci, paired_difference
from src.model import estimator_from_config
from src.pool import decontaminate_global, load_pool
from src.repro import git_commit
from src.train import append_registry
from src.validation import make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]

# Pre-specified before any nested run. Coarse on purpose: a fine grid invites the inner
# selector to chase noise, and the descriptive curve is flat between 0.5 and 0.7.
WEIGHT_GRID: tuple[float, ...] = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
FIXED_WEIGHT = 0.5


def cap_pool_stratified(pool_df, cap: int, seed: int):
    """Class-proportional subsample to <=cap rows (a validation-independent budget)."""
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


def load_identity_oof(path: Path):
    with np.load(path, allow_pickle=True) as d:
        required = {"scores", "y", "labels", "ordered_dataset_sha256", "validation_fold_ids"}
        missing = required - set(d.files)
        if missing:
            raise ValueError(f"{path}: OOF provenance missing {sorted(missing)}")
        scores = np.asarray(d["scores"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int64)
        labels = [str(v) for v in d["labels"]]
        fingerprint = str(np.asarray(d["ordered_dataset_sha256"]).item())
        folds = np.asarray(d["validation_fold_ids"], dtype=np.int64)
    if labels != list(LABELS):
        raise ValueError(f"{path}: class order is not the official order")
    return scores, y, fingerprint, folds


def blend_predictions(char_z: np.ndarray, style_z: np.ndarray, weight: float) -> np.ndarray:
    return (weight * char_z + (1.0 - weight) * style_z).argmax(axis=1)


def select_weight(char_z: np.ndarray, style_z: np.ndarray, y: np.ndarray,
                  grid: tuple[float, ...] = WEIGHT_GRID) -> tuple[float, list[dict]]:
    """Pick the weight with the best pooled balanced accuracy on INNER OOF only.

    Ties break toward the pre-specified 0.5, so a flat curve cannot drift the recipe.
    """
    truth = indices_to_labels(y)
    scored = []
    for w in grid:
        pred = blend_predictions(char_z, style_z, w)
        ba = float(evaluate(truth, indices_to_labels(pred))["balanced_accuracy"])
        scored.append({"weight": w, "inner_pooled_ba": ba})
    best = max(s["inner_pooled_ba"] for s in scored)
    tied = [s["weight"] for s in scored if s["inner_pooled_ba"] == best]
    chosen = min(tied, key=lambda w: (abs(w - FIXED_WEIGHT), w))
    return float(chosen), scored


def weight_curve(char_scores: np.ndarray, style_scores: np.ndarray, y: np.ndarray,
                 folds: np.ndarray, grid: tuple[float, ...]) -> list[dict]:
    """Descriptive sensitivity of the pooled/fold BA to the char weight."""
    char_z, style_z = normalize_scores(char_scores), normalize_scores(style_scores)
    truth = indices_to_labels(y)
    fold_ids = sorted(set(int(f) for f in folds))
    base_pred = blend_predictions(char_z, style_z, FIXED_WEIGHT)
    base_folds = [
        float(evaluate(truth[folds == k], indices_to_labels(base_pred[folds == k]))["balanced_accuracy"])
        for k in fold_ids
    ]
    rows = []
    for w in grid:
        pred = blend_predictions(char_z, style_z, w)
        fold_ba = [
            float(evaluate(truth[folds == k], indices_to_labels(pred[folds == k]))["balanced_accuracy"])
            for k in fold_ids
        ]
        ci = mean_ci(paired_difference(base_folds, fold_ba))
        rows.append({
            "char_weight": w,
            "pooled_ba": float(evaluate(truth, indices_to_labels(pred))["balanced_accuracy"]),
            "fold_ba": fold_ba,
            "paired_diff_vs_fixed_0.5": ci,
            "all_folds_non_negative": bool(all(d >= 0 for d in paired_difference(base_folds, fold_ba))),
        })
    return rows


def evaluate_nested_weight_selection(df, pool_df, configs, *, seed=42, cos_threshold=0.3,
                                     pool_cap=40000, outer_splits=3, outer_seed=121,
                                     inner_splits=3, inner_seed=71,
                                     grid: tuple[float, ...] = WEIGHT_GRID, verbose=True):
    """Outer-fold comparison of inner-selected weight vs the pre-specified 0.5."""
    names = [str(c["name"]) for c in configs]
    if names != ["char_tfidf_logreg", "style_only"]:
        raise ValueError(f"configs must be [char_tfidf_logreg, style_only], got {names}")

    y = labels_to_indices(df["model"].tolist())
    truth = indices_to_labels(y)
    folds, _ = make_splits(df, n_splits=outer_splits, seed=outer_seed)
    validate_splits(df, folds)

    sel_pred = np.full(len(df), -1, dtype=int)
    fix_pred = np.full(len(df), -1, dtype=int)
    reports = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        start = time.perf_counter()
        outer_train_df = df.iloc[train_idx].reset_index(drop=True)
        keep, gstats = decontaminate_global(pool_df, outer_train_df, cos_threshold=cos_threshold)
        clean_pool = pool_df.loc[keep].reset_index(drop=True)
        pool_keep = cap_pool_stratified(clean_pool, pool_cap, seed + fold)
        pool_texts = pool_keep["text"].tolist()
        pool_y = labels_to_indices(pool_keep["model"].tolist())
        pool_groups = pool_keep["prompt_canon"].astype(str).to_numpy()
        outer_y = y[train_idx]

        # --- inner selection: grouped folds inside outer-train, same pool keep-mask ---
        inner_folds, _ = make_splits(outer_train_df, n_splits=inner_splits, seed=inner_seed)
        validate_splits(outer_train_df, inner_folds)
        inner_char = np.full((len(outer_train_df), len(LABELS)), -1e9, dtype=np.float64)
        inner_style = np.full_like(inner_char, -1e9)
        inner_y = np.full(len(outer_train_df), -1, dtype=np.int64)
        for k, (itr, iva) in enumerate(inner_folds):
            texts = [outer_train_df["text"].iloc[i] for i in itr]
            groups = outer_train_df["prompt_id"].iloc[itr].astype(str).to_numpy()
            char = estimator_from_config(configs[0], seed=seed)
            char.fit(texts + pool_texts, np.concatenate([outer_y[itr], pool_y]),
                     groups=np.concatenate([groups, pool_groups]))
            style = estimator_from_config(configs[1], seed=seed)
            style.fit(texts, outer_y[itr], groups=groups)
            vtexts = [outer_train_df["text"].iloc[i] for i in iva]
            inner_char[iva] = char.decision_function(vtexts)
            inner_style[iva] = style.decision_function(vtexts)
            inner_y[iva] = outer_y[iva]
        if (inner_y < 0).any():
            raise RuntimeError("inner OOF does not cover every outer-train row")
        chosen, inner_curve = select_weight(normalize_scores(inner_char),
                                            normalize_scores(inner_style), inner_y, grid)

        # --- outer refit on all outer-train, score the untouched holdout ---
        train_texts = [df["text"].iloc[i] for i in train_idx]
        train_groups = df["prompt_id"].iloc[train_idx].astype(str).to_numpy()
        char = estimator_from_config(configs[0], seed=seed)
        char.fit(train_texts + pool_texts, np.concatenate([outer_y, pool_y]),
                 groups=np.concatenate([train_groups, pool_groups]))
        style = estimator_from_config(configs[1], seed=seed)
        style.fit(train_texts, outer_y, groups=train_groups)
        valid_texts = [df["text"].iloc[i] for i in valid_idx]
        char_z = normalize_scores(char.decision_function(valid_texts))
        style_z = normalize_scores(style.decision_function(valid_texts))
        sel_pred[valid_idx] = blend_predictions(char_z, style_z, chosen)
        fix_pred[valid_idx] = blend_predictions(char_z, style_z, FIXED_WEIGHT)

        sel_ba = float(evaluate(truth[valid_idx], indices_to_labels(sel_pred[valid_idx]))["balanced_accuracy"])
        fix_ba = float(evaluate(truth[valid_idx], indices_to_labels(fix_pred[valid_idx]))["balanced_accuracy"])
        elapsed = time.perf_counter() - start
        reports.append({
            "outer_fold": fold,
            "selected_char_weight": chosen,
            "inner_curve": inner_curve,
            "inner_selection_rows": int(len(outer_train_df)),
            "outer_holdout_rows": int(len(valid_idx)),
            "pool_screened_against": "outer_train_only",
            "pool_clean_rows": int(gstats["kept"]),
            "capped_rows": int(len(pool_keep)),
            "fit_seconds": round(elapsed, 2),
            "selected_outer_balanced_accuracy": sel_ba,
            "fixed_outer_balanced_accuracy": fix_ba,
        })
        if verbose:
            print(f"  [outer {fold}] inner-selected w={chosen:.2f} | fit {elapsed:.1f}s | "
                  f"fixed {fix_ba:.4f} -> selected {sel_ba:.4f}", flush=True)

    if (sel_pred < 0).any() or (fix_pred < 0).any():
        raise RuntimeError("outer predictions do not cover every row")
    sel_eval = evaluate(truth, indices_to_labels(sel_pred), model="nested_weight_selected")
    fix_eval = evaluate(truth, indices_to_labels(fix_pred), model="nested_weight_fixed")
    sel_folds = [r["selected_outer_balanced_accuracy"] for r in reports]
    fix_folds = [r["fixed_outer_balanced_accuracy"] for r in reports]
    return {
        "dataset": "official",
        "method": "nested outer-fold: inner-CV-selected char weight vs pre-specified 0.5",
        "ordered_dataset_sha256": ordered_dataset_fingerprint(df),
        "outer_splits": outer_splits, "outer_seed": outer_seed,
        "inner_splits": inner_splits, "inner_seed": inner_seed, "seed": seed,
        "cos_threshold": cos_threshold, "pool_cap_per_fold": pool_cap,
        "weight_grid": list(grid), "fixed_char_weight": FIXED_WEIGHT,
        "selection_metric": "pooled balanced accuracy on inner OOF; ties break toward 0.5",
        "selected_pooled_balanced_accuracy": sel_eval["balanced_accuracy"],
        "fixed_pooled_balanced_accuracy": fix_eval["balanced_accuracy"],
        "pooled_ba_delta": sel_eval["balanced_accuracy"] - fix_eval["balanced_accuracy"],
        "paired_outer_fold_diff": mean_ci(paired_difference(fix_folds, sel_folds)),
        "folds": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument("--pool", default="artifacts/pool_audit/ntl_open_pool.parquet")
    parser.add_argument("--out", type=Path, default=Path("artifacts/blend_weights/results.json"))
    parser.add_argument("--nested", action="store_true", help="run the honest nested selection protocol")
    parser.add_argument("--char-oof", type=Path,
                        default=Path("artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz"))
    parser.add_argument("--style-oof", type=Path,
                        default=Path("artifacts/cv_identity/official/oof/style_only.npz"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-threshold", type=float, default=0.3)
    parser.add_argument("--pool-cap", type=int, default=40000)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=121)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-seed", type=int, default=71)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    result: dict = {"mode": "nested" if args.nested else "descriptive_curve"}

    if not args.nested:
        char_scores, char_y, char_fp, char_folds = load_identity_oof(args.char_oof)
        style_scores, style_y, style_fp, style_folds = load_identity_oof(args.style_oof)
        if char_fp != style_fp:
            raise SystemExit("OOF fingerprints differ; the members are not fold-aligned")
        if not (np.array_equal(char_y, style_y) and np.array_equal(char_folds, style_folds)):
            raise SystemExit("OOF labels/fold ids differ; the members are not fold-aligned")
        grid = tuple(round(float(w), 2) for w in np.arange(0.0, 1.0001, 0.05))
        result.update({
            "dataset": "official",
            "ordered_dataset_sha256": char_fp,
            "n_rows": int(char_y.size),
            "char_oof": str(args.char_oof), "style_oof": str(args.style_oof),
            "warning": ("DESCRIPTIVE ONLY. These are saved OOF scores; a weight read off this "
                        "curve is selected on the same data that reports the score. Only the "
                        "--nested paired outer-fold result can support adoption."),
            "curve": weight_curve(char_scores, style_scores, char_y, char_folds, grid),
        })
        best = max(result["curve"], key=lambda r: r["pooled_ba"])
        fixed = next(r for r in result["curve"] if abs(r["char_weight"] - FIXED_WEIGHT) < 1e-9)
        print(f"[descriptive] fixed 0.5 pooled {fixed['pooled_ba']:.6f} | "
              f"curve peak w={best['char_weight']:.2f} pooled {best['pooled_ba']:.6f} "
              f"(NOT a selection score)", flush=True)
    else:
        configs = load_config(args.config)["models"]
        df, _ = load_competition_data(args.data_dir)
        pool_df = load_pool(args.pool, strict=args.strict)
        print(f"official {len(df)} rows | pool {len(pool_df)} rows | outer {args.outer_splits}"
              f"/s{args.outer_seed} inner {args.inner_splits}/s{args.inner_seed} | "
              f"cap {args.pool_cap} | grid {list(WEIGHT_GRID)}", flush=True)
        result.update(evaluate_nested_weight_selection(
            df, pool_df, configs, seed=args.seed, cos_threshold=args.cos_threshold,
            pool_cap=args.pool_cap, outer_splits=args.outer_splits, outer_seed=args.outer_seed,
            inner_splits=args.inner_splits, inner_seed=args.inner_seed,
        ))
        ci = result["paired_outer_fold_diff"]
        print(f"\n[nested] fixed {result['fixed_pooled_balanced_accuracy']:.6f} -> "
              f"selected {result['selected_pooled_balanced_accuracy']:.6f} "
              f"(delta {result['pooled_ba_delta']:+.6f}) | paired outer diff {ci['mean']:+.6f} "
              f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]", flush=True)

    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if not args.no_registry and args.nested:
        ci = result["paired_outer_fold_diff"]
        append_registry(args.registry, {
            "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official_pool_global-weight_selection",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "method": "nested_inner_weight_selection(char) + style official",
            "features": ("z(char_tfidf_logreg +GLOBAL pool cap40k) * w + z(style_only official) * (1-w); "
                         "w chosen by inner grouped CV, grid " + str(list(WEIGHT_GRID))),
            "seed": args.seed,
            "split_id": f"nested_group-s{args.outer_seed}-n{args.outer_splits}-inner{args.inner_splits}s{args.inner_seed}",
            "fold_scores": " ".join(f"{r['selected_outer_balanced_accuracy']:.4f}" for r in result["folds"]),
            "pooled_oof_balanced_accuracy": round(result["selected_pooled_balanced_accuracy"], 6),
            "macro_f1": "",
            "fit_seconds": result["elapsed_seconds"],
            "infer_seconds": "",
            "peak_ram_gb": "",
            "external_data_id": "open-pool-v15",
            "artifact_path": str(args.out),
            "public_lb_score": "",
            "notes": (f"Weight-selection gate for the cap-40k MIX candidate. Selected weights "
                      f"{[r['selected_char_weight'] for r in result['folds']]} vs pre-specified 0.5. "
                      f"fixed {result['fixed_pooled_balanced_accuracy']:.6f} -> selected "
                      f"{result['selected_pooled_balanced_accuracy']:.6f} (delta {result['pooled_ba_delta']:+.6f}); "
                      f"paired outer diff {ci['mean']:+.6f} [{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]. "
                      f"Pool screened against outer-train only, cap {args.pool_cap}/fold, training-only."),
            "status": "candidate" if ci["ci_low"] > 0 else "rejected",
        })


if __name__ == "__main__":
    main()