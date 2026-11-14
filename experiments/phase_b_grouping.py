"""Phase B: can we recover prompt groups well enough to capture the oracle gain?

Simulates the deployable test-time pipeline on grouped CV, using only local
train data (no test.csv, no submission):

* reuse the saved blend OOF scores (each fold's validation passages are scored
  by a model trained on the other folds -- exactly the test-time situation);
* embed the validation passages with frozen MiniLM (topic-sensitive);
* cluster them into prompt groups WITHOUT using prompt_id;
* measure cluster purity / ARI / strict recovery against the known prompts;
* apply within-group centering and one-per-model assignment on the RECOVERED
  groups and compare pooled balanced accuracy to argmax and to the oracle
  (known groups).

The gap between "recovered" and "oracle" is the cost of imperfect grouping.

Run:
    python experiments/phase_b_grouping.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ARTIFACTS_DIR, LABELS
from src.crossed import assign_one_per_model, center_scores_within_groups
from src.data import indices_to_labels, labels_to_indices, load_competition_data
from src.ensemble import load_oof, normalize_scores
from src.features_embedding import FrozenEncoder
from src.grouping import cluster_embeddings, grouping_purity
from src.metrics import balanced_accuracy
from src.validation import load_splits

BLEND_WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
MODELS_PER_PROMPT = 18  # full-factorial design (audit: 17-18)


def _bal(y, pred):
    return balanced_accuracy(indices_to_labels(y), indices_to_labels(pred))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.6,0.7,0.8",
        help="cosine distance thresholds to sweep",
    )
    args = parser.parse_args()

    train_df, _ = load_competition_data(args.data_dir)
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(train_df["model"].tolist())
    texts = train_df["text"].tolist()
    n = len(train_df)

    oof = load_oof(ARTIFACTS_DIR / "cv" / "official")
    blend = sum(
        normalize_scores(oof[name]["scores"]) * w for name, w in BLEND_WEIGHTS.items()
    )
    folds, _ = load_splits(ARTIFACTS_DIR / "splits" / "splits_official.json", train_df)

    print("encoding passages with MiniLM (cached)...", flush=True)
    t0 = time.perf_counter()
    encoder = FrozenEncoder(backend="sentence-transformers")
    emb = np.asarray(encoder.encode_cached(texts), dtype=np.float32)
    print(f"  embeddings {emb.shape} in {time.perf_counter() - t0:.1f}s", flush=True)

    # per-fold validation views
    fold_views = []
    for train_idx, valid_idx in folds:
        fold_views.append(
            {
                "idx": valid_idx,
                "scores": blend[valid_idx],
                "groups": groups[valid_idx],
                "emb": emb[valid_idx],
            }
        )

    configs = [("k_estimate", None)] + [
        (f"thr_{t}", float(t)) for t in args.thresholds.split(",")
    ]

    results = {}
    for name, threshold in configs:
        pooled = {
            key: np.full(n, -1, dtype=np.int64)
            for key in ("argmax", "oracle_assign", "rec_center", "rec_assign", "rec_center_assign")
        }
        purities, aris, recoveries, nclusters = [], [], [], []
        for view in fold_views:
            idx, scores, vgroups, vemb = (
                view["idx"], view["scores"], view["groups"], view["emb"]
            )
            if threshold is None:
                k = max(2, int(round(len(idx) / MODELS_PER_PROMPT)))
                labels = cluster_embeddings(vemb, n_clusters=k)
            else:
                labels = cluster_embeddings(vemb, distance_threshold=threshold)

            m = grouping_purity(vgroups, labels)
            purities.append(m["purity"])
            aris.append(m["adjusted_rand_index"])
            recoveries.append(m["recovered_prompt_fraction"])
            nclusters.append(m["n_clusters"])

            centered = center_scores_within_groups(scores, labels)
            pooled["argmax"][idx] = scores.argmax(axis=1)
            pooled["oracle_assign"][idx] = assign_one_per_model(scores, vgroups)
            pooled["rec_center"][idx] = centered.argmax(axis=1)
            pooled["rec_assign"][idx] = assign_one_per_model(scores, labels)
            pooled["rec_center_assign"][idx] = assign_one_per_model(centered, labels)

        results[name] = {
            "threshold": threshold,
            "purity_mean": float(np.mean(purities)),
            "ari_mean": float(np.mean(aris)),
            "recovered_prompt_fraction_mean": float(np.mean(recoveries)),
            "n_clusters_mean": float(np.mean(nclusters)),
            "pooled": {key: _bal(y, pred) for key, pred in pooled.items()},
        }

    out_dir = ARTIFACTS_DIR / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    print("\n=== Phase B: recovered prompt groups (MiniLM + agglomerative cosine) ===\n")
    print(
        f"{'config':<12} {'purity':>7} {'ARI':>6} {'recover':>8} {'#clus':>6} "
        f"{'argmax':>7} {'oracle':>7} {'recAsgn':>8} {'recC+A':>7}"
    )
    for name, r in results.items():
        p = r["pooled"]
        print(
            f"{name:<12} {r['purity_mean']:>7.3f} {r['ari_mean']:>6.3f} "
            f"{r['recovered_prompt_fraction_mean']:>8.3f} {r['n_clusters_mean']:>6.1f} "
            f"{p['argmax']:>7.4f} {p['oracle_assign']:>7.4f} "
            f"{p['rec_assign']:>8.4f} {p['rec_center_assign']:>7.4f}"
        )
    best = max(results, key=lambda k: results[k]["pooled"]["rec_assign"])
    bp = results[best]["pooled"]
    print(
        f"\nbest recovered-assignment config: {best} -> {bp['rec_assign']:.4f} "
        f"(argmax {bp['argmax']:.4f}, oracle {bp['oracle_assign']:.4f}); "
        f"captured {bp['rec_assign'] - bp['argmax']:+.4f} of "
        f"{bp['oracle_assign'] - bp['argmax']:+.4f} oracle gain"
    )
    print(f"\nresults written to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()