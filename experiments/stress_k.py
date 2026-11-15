"""Evaluation protocol: primary grouped OOF + k-subsampled paired-difference stress test.

PRIMARY (independent-passage models). The argmax prediction for a row does not
depend on how many sibling passages are scored alongside it, so the **full
prompt-grouped out-of-fold balanced accuracy over ALL rows** is the most
statistically efficient estimate and the MAIN comparison for the argmax blend
(and any per-passage model: DeBERTa, GBDT, ...). We report pooled OOF, per-fold
mean/std, per-class recall and within-provider confusions.

STRESS TEST (batch/group-dependent methods only). The hidden test is believed to
have ~1-3 passages per prompt (k~2), not the train set's k=18 (D017). Transductive
post-processing -- recovered-group Hungarian and damped Sinkhorn -- can exploit
batch structure that the test does not have. We therefore subsample the pooled
OOF to k=1,2,3,18 passages per prompt, repeatedly over several subsample seeds,
and report **paired differences** (method - argmax) on the SAME subsampled rows:
mean +/- sample std and a 95% Student-t CI across seeds. This measures whether a
batch method helps or hurts at a test-like k. It is INFERENCE-ONLY: these methods
are never part of the primary CV score (hard rule: no transductive steps in
reported scores). We report FIXED settings, not a best-of grid -- selecting the
best setting on the stress test would be tuning on it.

Holdout. The seed-7 holdout has been EXAMINED repeatedly (D015-D017) and is no
longer "untouched". A FRESH holdout (seed 2027) is reserved for final selection
and is NOT scored here; create/persist it with --reserve-fresh-holdout.

All local research on train data; no test.csv, no submission. Blend members and
0.5/0.5 weights are fixed (the adopted models); only the evaluation k varies.

Run:
    python experiments/stress_k.py --data-dir data
    python experiments/stress_k.py --data-dir data --reserve-fresh-holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import calibrate_predict
from src.constants import ARTIFACTS_DIR
from src.crossed import assign_one_per_model
from src.data import indices_to_labels, labels_to_indices, load_competition_data
from src.ensemble import normalize_scores
from src.features_embedding import FrozenEncoder
from src.grouping import cluster_embeddings
from src.metrics import (
    balanced_accuracy,
    mean_ci,
    paired_difference,
    per_class_recall,
    within_provider_confusions,
)
from src.model import estimator_from_config
from src.validation import make_holdout, make_splits, save_splits

CHAR = {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg",
        "C": 2.0, "max_features": 100000}
STYLE = {"name": "style_only", "features": [], "style": True, "classifier": "logreg",
         "C": 1.0}
MODEL_SEED = 42
# Fixed, pre-registered settings (NOT a best-of grid): the adopted damped Sinkhorn
# and a stronger variant, each reported separately as a paired difference.
SINKHORN_SETTINGS = [(0.3, 0.35), (0.6, 0.35)]
EXAMINED_HOLDOUT_SEED = 7
FRESH_HOLDOUT_SEED = 2027
FRESH_HOLDOUT_GROUPS = 25


def _bal(y, pred):
    return balanced_accuracy(indices_to_labels(y), indices_to_labels(pred))


def fit_blend_scores(train_texts, train_y, train_groups, eval_texts):
    blocks = []
    for cfg in (CHAR, STYLE):
        est = estimator_from_config(cfg, seed=MODEL_SEED)
        est.fit(train_texts, train_y, groups=train_groups)
        blocks.append(normalize_scores(est.decision_function(eval_texts)))
    return 0.5 * blocks[0] + 0.5 * blocks[1]


def pooled_oof(pairs, texts, y, groups):
    n = len(texts)
    scores = np.zeros((n, 18), dtype=np.float64)
    evaluated = np.zeros(n, dtype=bool)
    fit_seconds = 0.0
    for train_idx, eval_idx in pairs:
        t0 = time.perf_counter()
        scores[eval_idx] = fit_blend_scores(
            [texts[i] for i in train_idx], y[train_idx], groups[train_idx],
            [texts[i] for i in eval_idx],
        )
        fit_seconds += time.perf_counter() - t0
        evaluated[eval_idx] = True
    return scores, evaluated, fit_seconds


def primary_report(pairs, scores, evaluated, y):
    """Primary metric for the independent-passage argmax blend: full grouped OOF."""
    rows = np.flatnonzero(evaluated)
    truth = indices_to_labels(y[rows])
    pred = indices_to_labels(scores[rows].argmax(axis=1))
    fold_bas = [
        balanced_accuracy(indices_to_labels(y[np.asarray(ei)]),
                          indices_to_labels(scores[np.asarray(ei)].argmax(axis=1)))
        for _, ei in pairs
    ]
    return {
        "n_rows": int(rows.size),
        "pooled_oof_balanced_accuracy": balanced_accuracy(truth, pred),
        "fold_balanced_accuracy": [float(b) for b in fold_bas],
        "fold_mean": float(np.mean(fold_bas)),
        "fold_std": float(np.std(fold_bas, ddof=0)),
        "per_class_recall": per_class_recall(truth, pred),
        "within_provider": within_provider_confusions(truth, pred),
    }


def subsample_k(groups, evaluated, k, rng):
    rows = np.flatnonzero(evaluated)
    g = groups[rows]
    picked = []
    for grp in np.unique(g):
        idx = rows[g == grp]
        take = min(k, len(idx))
        picked.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(picked))


def stress_subset(scores, emb, y, idx):
    S, e, yy = scores[idx], emb[idx], y[idx]
    out = {"argmax": _bal(yy, S.argmax(axis=1))}
    lab = cluster_embeddings(e, distance_threshold=0.4)
    out["hungarian_rec"] = _bal(yy, assign_one_per_model(S, lab))
    for strength, mix in SINKHORN_SETTINGS:
        pred = calibrate_predict(S, strength=strength, target_mix=mix)
        out[f"sinkhorn_s{strength}_m{mix}"] = _bal(yy, pred)
    return out


def run_stress(pairs, texts, y, groups, emb, ks, sub_seeds):
    scores, evaluated, fit_seconds = pooled_oof(pairs, texts, y, groups)
    primary = primary_report(pairs, scores, evaluated, y)
    methods = ["hungarian_rec"] + [f"sinkhorn_s{s}_m{m}" for s, m in SINKHORN_SETTINGS]
    stress = {}
    for k in ks:
        per_seed = {m: [] for m in ["argmax"] + methods}
        nrows = []
        for ss in sub_seeds:
            rng = np.random.default_rng(1000 * k + ss)
            idx = subsample_k(groups, evaluated, k, rng)
            res = stress_subset(scores, emb, y, idx)
            nrows.append(len(idx))
            for m in per_seed:
                per_seed[m].append(res[m])
        entry = {"n_rows_mean": float(np.mean(nrows)),
                 "argmax": mean_ci(per_seed["argmax"])}
        for m in methods:
            diffs = paired_difference(per_seed["argmax"], per_seed[m])
            entry[m] = {"ba": mean_ci(per_seed[m]), "paired_diff": mean_ci(diffs)}
        stress[f"k{k}"] = entry
    return {"fit_seconds": round(fit_seconds, 1), "primary": primary, "stress": stress}


def reserve_fresh_holdout(train_df, out_dir):
    """Create and persist a FRESH holdout (seed 2027) WITHOUT scoring it.

    Reserved for final selection only; examining it now would burn the reserve.
    """
    tr, ho = make_holdout(train_df, n_groups=FRESH_HOLDOUT_GROUPS, seed=FRESH_HOLDOUT_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fresh_holdout_s{FRESH_HOLDOUT_SEED}.json"
    save_splits(path, train_df, [(tr, ho)], "holdout", FRESH_HOLDOUT_SEED)
    note = {
        "seed": FRESH_HOLDOUT_SEED,
        "n_groups": FRESH_HOLDOUT_GROUPS,
        "n_train": int(len(tr)),
        "n_holdout": int(len(ho)),
        "status": "RESERVED — not examined; score only at final selection",
        "replaces_examined_seed": EXAMINED_HOLDOUT_SEED,
    }
    (out_dir / "fresh_holdout_reservation.json").write_text(
        json.dumps(note, indent=1), encoding="utf-8")
    return note


def _fmt_ci(d):
    return f"{d['mean']:.4f}+/-{d['std']:.4f} [{d['ci_low']:+.4f},{d['ci_high']:+.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="cv_s42,holdout_s7")
    parser.add_argument("--ks", default="1,2,3,18")
    parser.add_argument("--sub-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--reserve-fresh-holdout", action="store_true",
                        help="persist a fresh seed-2027 holdout without scoring it")
    args = parser.parse_args()

    train_df, _ = load_competition_data(args.data_dir)
    out_dir = ARTIFACTS_DIR / "stress_k"

    if args.reserve_fresh_holdout:
        note = reserve_fresh_holdout(train_df, out_dir)
        print("Fresh holdout RESERVED (not scored):", json.dumps(note, indent=1))
        return

    groups = train_df["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(train_df["model"].tolist())
    texts = train_df["text"].tolist()
    ks = [int(k) for k in args.ks.split(",")]
    sub_seeds = [int(s) for s in args.sub_seeds.split(",")]

    encoder = FrozenEncoder(backend="sentence-transformers")
    emb = np.asarray(encoder.encode_cached(texts), dtype=np.float32)

    split_pairs = {}
    for name in args.splits.split(","):
        if name.startswith("cv_s"):
            folds, _ = make_splits(train_df, n_splits=5, seed=int(name[4:]))
            split_pairs[name] = folds
        elif name.startswith("holdout_s"):
            tr, ho = make_holdout(train_df, n_groups=25, seed=int(name[9:]))
            split_pairs[name] = [(tr, ho)]
        else:
            raise ValueError(f"unknown split {name}")

    results = {}
    for name, pairs in split_pairs.items():
        print(f"{name}: computing pooled OOF + stress...", flush=True)
        results[name] = run_stress(pairs, texts, y, groups, emb, ks, sub_seeds)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n=== PRIMARY: full prompt-grouped OOF (independent-passage argmax blend) ===\n")
    for name, r in results.items():
        p = r["primary"]
        wp = p["within_provider"]
        share = wp["within_provider_errors"] / wp["errors"] if wp["errors"] else 0.0
        print(f"{name}: pooled OOF BA = {p['pooled_oof_balanced_accuracy']:.4f} "
              f"({p['n_rows']} rows) | folds {p['fold_mean']:.4f}+/-{p['fold_std']:.4f} "
              f"| within-provider {share:.1%}")

    print("\n=== STRESS TEST (inference-only; paired diff vs argmax, mean±std [95% CI]) ===")
    print("Not a selection score; transductive methods are excluded from the primary CV.\n")
    for name, r in results.items():
        print(f"\n{name}  (fit {r['fit_seconds']}s, {len(sub_seeds)} subsample seeds)")
        for k in ks:
            b = r["stress"][f"k{k}"]
            print(f"  k={k:<2} rows~{b['n_rows_mean']:>5.0f}  argmax {_fmt_ci(b['argmax'])}")
            for m in ["hungarian_rec"] + [f"sinkhorn_s{s}_m{m2}" for s, m2 in SINKHORN_SETTINGS]:
                pd = b[m]["paired_diff"]
                print(f"        {m:<22} delta={pd['mean']:+.4f}+/-{pd['std']:.4f} "
                      f"[{pd['ci_low']:+.4f},{pd['ci_high']:+.4f}]")
    print(f"\nresults written to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()