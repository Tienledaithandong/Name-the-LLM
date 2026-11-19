"""Non-transductive outer-fold evaluation of the pool-MIX submission recipe (D028/D035).

The candidate submission recipe is a FIXED 0.5/0.5 row-wise-z blend of
``char_tfidf_logreg`` trained on (official fold-train + globally-decontaminated Open
Pool) and ``style_only`` trained on official fold-train only. The blend weights are
pre-specified (identical to the 0.3760 fallback), never tuned on data, so there is no
inner selection to guard against; a grouped outer-fold evaluation is therefore an
honest, fold-independent estimate.

For each outer fold the script refits every member on that fold's outer-train rows,
screens the pool ONLY against outer-train text (the outer-holdout text never changes
which pool rows are eligible), caps the clean pool by class-proportional random
subsampling (a validation-independent compute budget), and scores the untouched outer
holdout. ``match_weight``/``style_score`` are never used (D019). It reports the same
recipe WITHOUT the pool (official-only) on identical folds as the paired baseline, so
the paired outer-fold diff isolates the pool contribution.
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
from src.data import indices_to_labels, labels_to_indices, load_competition_data, ordered_dataset_fingerprint
from src.ensemble import normalize_scores
from src.metrics import evaluate, mean_ci, paired_difference, per_class_recall
from src.model import estimator_from_config
from src.pool import decontaminate_global, load_pool
from src.repro import git_commit
from src.train import append_registry
from src.validation import make_splits, validate_splits

ROOT = Path(__file__).resolve().parents[1]


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


def evaluate_nested_pool_blend(df, pool_df, configs, *, seed=42, cos_threshold=0.3,
                               pool_cap=12000, outer_splits=3, outer_seed=121, verbose=True):
    """Return the fixed-weight pool-MIX recipe evaluated on grouped outer folds.

    ``configs`` must be ``[char_tfidf_logreg, style_only]`` in that order. The pool is
    screened against each outer fold's TRAIN text only, so outer-holdout passages never
    influence which pool rows are trained on (fold-independent, non-transductive).
    """
    names = [str(c["name"]) for c in configs]
    if names != ["char_tfidf_logreg", "style_only"]:
        raise ValueError(f"configs must be [char_tfidf_logreg, style_only], got {names}")

    y = labels_to_indices(df["model"].tolist())
    truth = indices_to_labels(y)
    folds, _ = make_splits(df, n_splits=outer_splits, seed=outer_seed)
    validate_splits(df, folds)

    mix_pred = np.full(len(df), -1, dtype=int)
    base_pred = np.full(len(df), -1, dtype=int)
    reports = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        if not set(df["prompt_id"].iloc[train_idx].astype(str)).isdisjoint(
                set(df["prompt_id"].iloc[valid_idx].astype(str))):
            raise ValueError(f"outer fold {fold} has a prompt on both sides")

        # Screen pool against THIS fold's outer-train text only (holdout never seen).
        outer_train_df = df.iloc[train_idx].reset_index(drop=True)
        keep, gstats = decontaminate_global(pool_df, outer_train_df, cos_threshold=cos_threshold)
        clean_pool = pool_df.loc[keep].reset_index(drop=True)
        pool_keep = cap_pool_stratified(clean_pool, pool_cap, seed + fold)
        pool_texts = pool_keep["text"].tolist()
        pool_y = labels_to_indices(pool_keep["model"].tolist())
        pool_groups = pool_keep["prompt_canon"].astype(str).to_numpy()

        train_texts = [df["text"].iloc[i] for i in train_idx]
        train_groups = df["prompt_id"].iloc[train_idx].astype(str).to_numpy()
        valid_texts = [df["text"].iloc[i] for i in valid_idx]

        start = time.perf_counter()
        char_pool = estimator_from_config(configs[0], seed=seed)
        char_pool.fit(train_texts + pool_texts, np.concatenate([y[train_idx], pool_y]),
                      groups=np.concatenate([train_groups, pool_groups]))
        style = estimator_from_config(configs[1], seed=seed)
        style.fit(train_texts, y[train_idx], groups=train_groups)
        char_off = estimator_from_config(configs[0], seed=seed)
        char_off.fit(train_texts, y[train_idx], groups=train_groups)
        elapsed = time.perf_counter() - start

        char_pool_z = normalize_scores(char_pool.decision_function(valid_texts))
        style_z = normalize_scores(style.decision_function(valid_texts))
        char_off_z = normalize_scores(char_off.decision_function(valid_texts))

        mix_pred[valid_idx] = (0.5 * char_pool_z + 0.5 * style_z).argmax(1)
        base_pred[valid_idx] = (0.5 * char_off_z + 0.5 * style_z).argmax(1)

        mix_ba = evaluate(truth[valid_idx], indices_to_labels(mix_pred[valid_idx]))["balanced_accuracy"]
        base_ba = evaluate(truth[valid_idx], indices_to_labels(base_pred[valid_idx]))["balanced_accuracy"]
        reports.append({
            "outer_fold": fold,
            "pool_screened_against": "outer_train_only",
            "pool_clean_rows": int(gstats["kept"]),
            "pool_exact_dup": int(gstats["exact_dup"]),
            "pool_near_dup": int(gstats["near_dup"]),
            "capped_rows": int(len(pool_keep)),
            "fit_seconds": round(elapsed, 2),
            "mix_outer_balanced_accuracy": float(mix_ba),
            "official_only_outer_balanced_accuracy": float(base_ba),
        })
        if verbose:
            print(f"  [outer {fold}] pool->{gstats['kept']} clean (exact {gstats['exact_dup']}"
                  f" near {gstats['near_dup']}) cap {len(pool_keep)} | fit {elapsed:.1f}s | "
                  f"official-only {base_ba:.4f} -> MIX {mix_ba:.4f}", flush=True)

    if (mix_pred < 0).any() or (base_pred < 0).any():
        raise RuntimeError("outer predictions do not cover every row")

    mix_eval = evaluate(truth, indices_to_labels(mix_pred), model="nested_pool_mix")
    base_eval = evaluate(truth, indices_to_labels(base_pred), model="nested_pool_official_only")
    mix_folds = [r["mix_outer_balanced_accuracy"] for r in reports]
    base_folds = [r["official_only_outer_balanced_accuracy"] for r in reports]
    ci = mean_ci(paired_difference(base_folds, mix_folds))
    mix_recall = per_class_recall(truth, indices_to_labels(mix_pred))
    base_recall = per_class_recall(truth, indices_to_labels(base_pred))
    return {
        "dataset": "official",
        "method": "nested outer-fold pool-MIX (fixed 0.5/0.5 z, D028 non-transductive)",
        "ordered_dataset_sha256": ordered_dataset_fingerprint(df),
        "outer_splits": outer_splits, "outer_seed": outer_seed, "seed": seed,
        "cos_threshold": cos_threshold, "pool_cap_per_fold": pool_cap,
        "fixed_char_weight": 0.5, "no_inner_selection": True,
        "mix_pooled_balanced_accuracy": mix_eval["balanced_accuracy"],
        "official_only_pooled_balanced_accuracy": base_eval["balanced_accuracy"],
        "mix_macro_f1": mix_eval["macro_f1"],
        "pooled_ba_delta": mix_eval["balanced_accuracy"] - base_eval["balanced_accuracy"],
        "paired_outer_fold_diff": ci,
        "mix_per_class_recall": mix_recall,
        "official_only_per_class_recall": base_recall,
        "per_class_recall_delta": {k: (None if np.isnan(base_recall[k]) or np.isnan(mix_recall[k])
                                       else round(mix_recall[k] - base_recall[k], 4)) for k in LABELS},
        "folds": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument("--pool", default="artifacts/pool_audit/ntl_open_pool.parquet")
    parser.add_argument("--out", type=Path, default=Path("artifacts/nested_pool_blend/results.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-threshold", type=float, default=0.3)
    parser.add_argument("--pool-cap", type=int, default=12000)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=121)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    configs = load_config(args.config)["models"]
    df, _ = load_competition_data(args.data_dir)
    pool_df = load_pool(args.pool, strict=args.strict)
    print(f"official {len(df)} rows | input pool {len(pool_df)} rows | outer {args.outer_splits}"
          f" seed {args.outer_seed} | cap {args.pool_cap}/fold", flush=True)
    started = time.perf_counter()
    result = evaluate_nested_pool_blend(
        df, pool_df, configs, seed=args.seed, cos_threshold=args.cos_threshold,
        pool_cap=args.pool_cap, outer_splits=args.outer_splits, outer_seed=args.outer_seed,
    )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)

    ci = result["paired_outer_fold_diff"]
    print(f"\n[outer-fold pooled] official-only {result['official_only_pooled_balanced_accuracy']:.4f}"
          f" -> MIX {result['mix_pooled_balanced_accuracy']:.4f} (delta {result['pooled_ba_delta']:+.4f})"
          f" | paired outer diff {ci['mean']:+.4f} +/- {ci['ci_half']:.4f} "
          f"[{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if not args.no_registry:
        append_registry(args.registry, {
            "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official_pool_global-nested_s{args.outer_seed}",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "method": "pool_global_blend_MIX_nested",
            "features": "0.5 z(char_tfidf_logreg +GLOBAL pool) + 0.5 z(style_only official) [outer-fold]",
            "seed": args.seed,
            "split_id": f"nested_group-s{args.outer_seed}-n{args.outer_splits}",
            "fold_scores": " ".join(f"{r['mix_outer_balanced_accuracy']:.4f}" for r in result["folds"]),
            "pooled_oof_balanced_accuracy": round(result["mix_pooled_balanced_accuracy"], 6),
            "macro_f1": round(float(result["mix_macro_f1"]), 6),
            "fit_seconds": result["elapsed_seconds"],
            "infer_seconds": "",
            "peak_ram_gb": "",
            "external_data_id": "open-pool-v15",
            "artifact_path": str(args.out),
            "public_lb_score": "",
            "notes": (f"D035 FINAL GATE for the pool-MIX candidate. Grouped outer-fold (n={args.outer_splits},"
                      f" seed {args.outer_seed}) eval of the FIXED 0.5/0.5 z recipe; pool screened against each"
                      f" outer-TRAIN only (holdout never influences eligibility, non-transductive D028), cap"
                      f" {args.pool_cap}/fold. official-only pooled {result['official_only_pooled_balanced_accuracy']:.4f}"
                      f" -> MIX {result['mix_pooled_balanced_accuracy']:.4f} (delta {result['pooled_ba_delta']:+.4f});"
                      f" paired outer diff {ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]."),
            "status": "candidate",
        })


if __name__ == "__main__":
    main()
