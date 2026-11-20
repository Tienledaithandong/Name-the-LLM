"""Historical k-subsampled, transductive research helpers.

The executable command is retired under D024: transductive steps cannot be
used in reported scores, and the seed-2027 subset was part of earlier
full-OOF development. Historical functions remain only for reproducibility.
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
    """Retire the historical transductive reporting command under D024."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", default="cv_s42,holdout_s7")
    parser.add_argument("--ks", default="1,2,3,18")
    parser.add_argument("--sub-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--reserve-fresh-holdout", action="store_true")
    parser.parse_args()
    raise SystemExit(
        "[abort] D024: transductive methods are prohibited in reported scores; "
        "the seed-2027 subset is not an untouched final holdout. "
        "Use experiments/prepare_nested_splits.py for non-transductive evaluation."
    )


if __name__ == "__main__":
    main()