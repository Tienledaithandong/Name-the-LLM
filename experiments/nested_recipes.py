"""Can a THIRD member survive an honest nested gate? (H10, mandate priority 5)

D055 screened six cheap official-only arms against the cap-40k pool-MIX blend and
all of them lost. D056 then gave the word view the *same* pool advantage: solo
0.2697 -> 0.3549 (s42) and 0.2559 -> 0.3463 (s13), which made the pre-specified
0.4/0.4/0.2 recipe the first positive point estimate on both seeds (+0.0074 s42,
+0.0096 s13). Both paired CIs include zero and both were read off the reporting
OOF, so neither number can support adoption (D026).

This is the gate. Grouped OUTER folds; the recipe is chosen by INNER grouped CV
inside outer-train only; the untouched outer holdout is scored with both the
selected recipe and the pre-specified two-member base. Only the paired outer-fold
difference between those two is selection-free.

The pool is screened once per outer fold against outer-train text (D028-style and
fold-independent: one keep-mask serves every inner fold), capped by
class-proportional subsampling, and never scored. ``match_weight``/``style_score``
are never features (D019).
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

from experiments.blend_weights import cap_pool_stratified
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

MEMBERS: tuple[str, ...] = ("char_tfidf_logreg", "style_only", "word_tfidf_linearsvc")
POOL_MEMBERS = frozenset({"char_tfidf_logreg", "word_tfidf_linearsvc"})
BASE_RECIPE = "base"

# Pre-specified before any nested run, and identical to the descriptive screen's
# recipes so the gate tests the same objects the screen measured.
RECIPES: tuple[tuple[str, dict[str, float]], ...] = (
    (BASE_RECIPE, {"char_tfidf_logreg": 0.5, "style_only": 0.5}),
    ("thirds", {name: 1.0 / 3.0 for name in MEMBERS}),
    ("keep_ratio", {"char_tfidf_logreg": 0.4, "style_only": 0.4, "word_tfidf_linearsvc": 0.2}),
)


def recipe_predictions(zs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Argmax of a weighted sum of row-wise-z member scores."""
    total = np.zeros_like(next(iter(zs.values())))
    for name, weight in weights.items():
        if weight:
            total = total + weight * zs[name]
    return total.argmax(axis=1)


def select_recipe(zs: dict[str, np.ndarray], y: np.ndarray,
                  recipes: tuple[tuple[str, dict[str, float]], ...] = RECIPES) -> tuple[str, list[dict]]:
    """Pick the recipe with the best pooled BA on INNER OOF only; ties keep the base."""
    truth = indices_to_labels(y)
    curve = []
    for name, weights in recipes:
        pred = recipe_predictions(zs, weights)
        curve.append({"recipe": name, "weights": dict(weights),
                      "inner_pooled_ba": float(evaluate(truth, indices_to_labels(pred))["balanced_accuracy"])})
    best = max(r["inner_pooled_ba"] for r in curve)
    tied = [r["recipe"] for r in curve if r["inner_pooled_ba"] == best]
    chosen = BASE_RECIPE if BASE_RECIPE in tied else tied[0]
    return chosen, curve


def fit_members(configs: list[dict], texts: list[str], y: np.ndarray, groups: np.ndarray, *,
                pool_texts: list[str], pool_y: np.ndarray, pool_groups: np.ndarray,
                seed: int) -> dict[str, object]:
    """Fit every member; only POOL_MEMBERS see the (screened, capped) external rows."""
    fitted = {}
    for cfg in configs:
        name = str(cfg["name"])
        est = estimator_from_config(cfg, seed=seed)
        if name in POOL_MEMBERS:
            est.fit(texts + pool_texts, np.concatenate([y, pool_y]),
                    groups=np.concatenate([groups, pool_groups]))
        else:
            est.fit(texts, y, groups=groups)
        fitted[name] = est
    return fitted


def evaluate_nested_recipe_selection(df, pool_df, configs, *, seed=42, cos_threshold=0.3,
                                     pool_cap=40000, outer_splits=3, outer_seed=121,
                                     inner_splits=3, inner_seed=71,
                                     recipes: tuple[tuple[str, dict[str, float]], ...] = RECIPES,
                                     verbose=True) -> dict:
    """Outer-fold comparison of inner-selected recipe vs the pre-specified base."""
    names = [str(c["name"]) for c in configs]
    if names != list(MEMBERS):
        raise ValueError(f"configs must be {list(MEMBERS)}, got {names}")
    base_weights = dict(recipes[0][1])
    if recipes[0][0] != BASE_RECIPE:
        raise ValueError(f"the first recipe must be the pre-specified {BASE_RECIPE!r}")

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
        inner_scores = {name: np.full((len(outer_train_df), len(LABELS)), -1e9, dtype=np.float64)
                        for name in names}
        inner_y = np.full(len(outer_train_df), -1, dtype=np.int64)
        for itr, iva in inner_folds:
            texts = [outer_train_df["text"].iloc[i] for i in itr]
            groups = outer_train_df["prompt_id"].iloc[itr].astype(str).to_numpy()
            fitted = fit_members(configs, texts, outer_y[itr], groups, pool_texts=pool_texts,
                                 pool_y=pool_y, pool_groups=pool_groups, seed=seed)
            vtexts = [outer_train_df["text"].iloc[i] for i in iva]
            for name, est in fitted.items():
                inner_scores[name][iva] = est.decision_function(vtexts)
            inner_y[iva] = outer_y[iva]
        if (inner_y < 0).any():
            raise RuntimeError("inner OOF does not cover every outer-train row")
        inner_z = {name: normalize_scores(scores) for name, scores in inner_scores.items()}
        chosen, inner_curve = select_recipe(inner_z, inner_y, recipes)
        chosen_weights = next(w for n, w in recipes if n == chosen)

        # --- outer refit on all outer-train, score the untouched holdout ---
        train_texts = [df["text"].iloc[i] for i in train_idx]
        train_groups = df["prompt_id"].iloc[train_idx].astype(str).to_numpy()
        fitted = fit_members(configs, train_texts, outer_y, train_groups, pool_texts=pool_texts,
                             pool_y=pool_y, pool_groups=pool_groups, seed=seed)
        valid_texts = [df["text"].iloc[i] for i in valid_idx]
        valid_z = {name: normalize_scores(est.decision_function(valid_texts))
                   for name, est in fitted.items()}
        sel_pred[valid_idx] = recipe_predictions(valid_z, chosen_weights)
        fix_pred[valid_idx] = recipe_predictions(valid_z, base_weights)

        sel_ba = float(evaluate(truth[valid_idx], indices_to_labels(sel_pred[valid_idx]))["balanced_accuracy"])
        fix_ba = float(evaluate(truth[valid_idx], indices_to_labels(fix_pred[valid_idx]))["balanced_accuracy"])
        elapsed = time.perf_counter() - start
        reports.append({
            "outer_fold": fold,
            "selected_recipe": chosen,
            "selected_weights": dict(chosen_weights),
            "inner_curve": inner_curve,
            "inner_selection_rows": int(len(outer_train_df)),
            "outer_holdout_rows": int(len(valid_idx)),
            "pool_screened_against": "outer_train_only",
            "pool_clean_rows": int(gstats["kept"]),
            "capped_rows": int(len(pool_keep)),
            "fit_seconds": round(elapsed, 2),
            "selected_outer_balanced_accuracy": sel_ba,
            "base_outer_balanced_accuracy": fix_ba,
        })
        if verbose:
            print(f"  [outer {fold}] inner-selected {chosen} | fit {elapsed:.1f}s | "
                  f"base {fix_ba:.4f} -> selected {sel_ba:.4f}", flush=True)

    if (sel_pred < 0).any() or (fix_pred < 0).any():
        raise RuntimeError("outer predictions do not cover every row")
    sel_eval = evaluate(truth, indices_to_labels(sel_pred), model="nested_recipe_selected")
    fix_eval = evaluate(truth, indices_to_labels(fix_pred), model="nested_recipe_base")
    sel_folds = [r["selected_outer_balanced_accuracy"] for r in reports]
    fix_folds = [r["base_outer_balanced_accuracy"] for r in reports]
    return {
        "dataset": "official",
        "method": "nested outer-fold: inner-CV-selected recipe vs pre-specified two-member base",
        "ordered_dataset_sha256": ordered_dataset_fingerprint(df),
        "outer_splits": outer_splits, "outer_seed": outer_seed,
        "inner_splits": inner_splits, "inner_seed": inner_seed, "seed": seed,
        "cos_threshold": cos_threshold, "pool_cap_per_fold": pool_cap,
        "members": list(names), "pool_members": sorted(POOL_MEMBERS),
        "recipes": [{"recipe": n, "weights": dict(w)} for n, w in recipes],
        "base_recipe": BASE_RECIPE,
        "selection_metric": "pooled balanced accuracy on inner OOF; ties keep the base recipe",
        "selected_pooled_balanced_accuracy": sel_eval["balanced_accuracy"],
        "base_pooled_balanced_accuracy": fix_eval["balanced_accuracy"],
        "pooled_ba_delta": sel_eval["balanced_accuracy"] - fix_eval["balanced_accuracy"],
        "paired_outer_fold_diff": mean_ci(paired_difference(fix_folds, sel_folds)),
        "folds": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/nested_recipes.yaml")
    parser.add_argument("--pool", default="artifacts/pool_audit/ntl_open_pool.parquet")
    parser.add_argument("--out", type=Path, default=Path("artifacts/nested_recipes/results.json"))
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
    configs = load_config(args.config)["models"]
    df, _ = load_competition_data(args.data_dir)
    pool_df = load_pool(args.pool, strict=args.strict)
    print(f"official {len(df)} rows | pool {len(pool_df)} rows | outer {args.outer_splits}"
          f"/s{args.outer_seed} inner {args.inner_splits}/s{args.inner_seed} | cap {args.pool_cap} | "
          f"recipes {[n for n, _ in RECIPES]}", flush=True)
    result = evaluate_nested_recipe_selection(
        df, pool_df, configs, seed=args.seed, cos_threshold=args.cos_threshold,
        pool_cap=args.pool_cap, outer_splits=args.outer_splits, outer_seed=args.outer_seed,
        inner_splits=args.inner_splits, inner_seed=args.inner_seed,
    )
    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    ci = result["paired_outer_fold_diff"]
    print(f"\n[nested] base {result['base_pooled_balanced_accuracy']:.6f} -> "
          f"selected {result['selected_pooled_balanced_accuracy']:.6f} "
          f"(delta {result['pooled_ba_delta']:+.6f}) | paired outer diff {ci['mean']:+.6f} "
          f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if not args.no_registry:
        append_registry(args.registry, {
            "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official_pool_global-recipe_selection",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "method": "nested_inner_recipe_selection(char+pool, style, word+pool)",
            "features": ("row-z blend of char_tfidf_logreg(+GLOBAL pool cap40k), style_only(official) "
                         "and word_tfidf_linearsvc(+GLOBAL pool cap40k); recipe chosen by inner "
                         "grouped CV from " + str([n for n, _ in RECIPES])),
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
            "notes": (f"H10 third-member gate (D056). Selected recipes "
                      f"{[r['selected_recipe'] for r in result['folds']]} vs pre-specified base. "
                      f"base {result['base_pooled_balanced_accuracy']:.6f} -> selected "
                      f"{result['selected_pooled_balanced_accuracy']:.6f} "
                      f"(delta {result['pooled_ba_delta']:+.6f}); paired outer diff {ci['mean']:+.6f} "
                      f"[{ci['ci_low']:+.6f},{ci['ci_high']:+.6f}]. Pool screened against outer-train "
                      f"only, cap {args.pool_cap}/fold, training-only."),
            "status": "candidate" if ci["ci_low"] > 0 else "rejected",
        })


if __name__ == "__main__":
    main()