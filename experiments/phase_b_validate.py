"""Phase B validation: does the cluster->assign gain survive out of sample?

Refits the adopted blend (char_tfidf_logreg + style_only, 0.5/0.5) and tests the
full recovered-grouping pipeline on:

* the secondary prompt-held-out split (make_holdout, seed 7, 25 prompts) --
  untouched by model selection;
* alternative grouped-CV fold seeds (7 and 13) -- refit OOF, then cluster each
  fold's validation passages label-free and assign.

For each split we report argmax, oracle (known groups) and recovered assignment
(k = rows/18 and cosine threshold 0.4). All local research on train data; no
test.csv, no submission. Model fit seed is fixed at 42 (the adopted models); only
the split varies.

Run:
    python experiments/phase_b_validate.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ARTIFACTS_DIR
from src.crossed import assign_one_per_model
from src.data import indices_to_labels, labels_to_indices, load_competition_data
from src.ensemble import normalize_scores
from src.features_embedding import FrozenEncoder
from src.grouping import cluster_embeddings, grouping_purity
from src.metrics import balanced_accuracy
from src.model import estimator_from_config
from src.validation import make_holdout, make_splits

CHAR = {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg",
        "C": 2.0, "max_features": 100000}
STYLE = {"name": "style_only", "features": [], "style": True, "classifier": "logreg",
         "C": 1.0}
MODEL_SEED = 42
MODELS_PER_PROMPT = 18


def _bal(y, pred):
    return balanced_accuracy(indices_to_labels(y), indices_to_labels(pred))


def fit_blend_scores(train_texts, train_y, train_groups, eval_texts):
    blocks = []
    for cfg in (CHAR, STYLE):
        est = estimator_from_config(cfg, seed=MODEL_SEED)
        est.fit(train_texts, train_y, groups=train_groups)
        blocks.append(normalize_scores(est.decision_function(eval_texts)))
    return 0.5 * blocks[0] + 0.5 * blocks[1]


def run_split(pairs, texts, y, groups, emb):
    n = len(texts)
    pred = {k: np.full(n, -1, dtype=np.int64)
            for k in ("argmax", "oracle", "rec_k", "rec_thr")}
    evaluated = np.zeros(n, dtype=bool)
    pur_k, pur_thr = [], []
    fit_seconds = 0.0
    for train_idx, eval_idx in pairs:
        t0 = time.perf_counter()
        scores = fit_blend_scores(
            [texts[i] for i in train_idx], y[train_idx], groups[train_idx],
            [texts[i] for i in eval_idx],
        )
        fit_seconds += time.perf_counter() - t0
        e, g = emb[eval_idx], groups[eval_idx]
        k = max(2, int(round(len(eval_idx) / MODELS_PER_PROMPT)))
        lk = cluster_embeddings(e, n_clusters=k)
        lt = cluster_embeddings(e, distance_threshold=0.4)
        pred["argmax"][eval_idx] = scores.argmax(axis=1)
        pred["oracle"][eval_idx] = assign_one_per_model(scores, g)
        pred["rec_k"][eval_idx] = assign_one_per_model(scores, lk)
        pred["rec_thr"][eval_idx] = assign_one_per_model(scores, lt)
        evaluated[eval_idx] = True
        pur_k.append(grouping_purity(g, lk)["purity"])
        pur_thr.append(grouping_purity(g, lt)["purity"])
    rows = np.flatnonzero(evaluated)
    return {
        "n_evaluated": int(rows.size),
        "pooled": {key: _bal(y[rows], p[rows]) for key, p in pred.items()},
        "purity_k_mean": float(np.mean(pur_k)),
        "purity_thr_mean": float(np.mean(pur_thr)),
        "fit_seconds": round(fit_seconds, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seeds", default="7,13")
    args = parser.parse_args()

    train_df, _ = load_competition_data(args.data_dir)
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(train_df["model"].tolist())
    texts = train_df["text"].tolist()

    encoder = FrozenEncoder(backend="sentence-transformers")
    emb = np.asarray(encoder.encode_cached(texts), dtype=np.float32)

    results = {}

    tr_idx, ho_idx = make_holdout(train_df, n_groups=25, seed=7)
    print(f"holdout: {len(ho_idx)} rows / {len(np.unique(groups[ho_idx]))} prompts", flush=True)
    results["holdout_s7"] = run_split([(tr_idx, ho_idx)], texts, y, groups, emb)

    for seed in [int(s) for s in args.seeds.split(",")]:
        folds, _ = make_splits(train_df, n_splits=5, seed=seed)
        print(f"CV seed {seed}: refitting blend over {len(folds)} folds...", flush=True)
        results[f"cv_s{seed}"] = run_split(folds, texts, y, groups, emb)

    out_dir = ARTIFACTS_DIR / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n=== Phase B validation: recovered assignment out of sample ===\n")
    print(f"{'split':<12} {'argmax':>7} {'oracle':>7} {'rec_k':>7} {'rec_thr':>8} "
          f"{'gain_k':>7} {'purity_k':>8}")
    for name, r in results.items():
        p = r["pooled"]
        print(f"{name:<12} {p['argmax']:>7.4f} {p['oracle']:>7.4f} {p['rec_k']:>7.4f} "
              f"{p['rec_thr']:>8.4f} {p['rec_k'] - p['argmax']:>+7.4f} "
              f"{r['purity_k_mean']:>8.3f}")
    print(f"\nresults written to {out_dir / 'validation.json'}")


if __name__ == "__main__":
    main()