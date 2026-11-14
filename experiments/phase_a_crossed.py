"""Phase A oracle: how much does exploiting the crossed design help?

Uses KNOWN train prompt groups to measure the ceiling of two label-free
inference transforms on the saved OOF scores:

* A1-score  -- within-prompt score centering, then argmax.
* A2        -- one-passage-per-model assignment within prompt groups.
* A1+A2     -- centering followed by assignment.

Plus a deployable-consistent check:

* A1-feature -- retrain ``style_only`` on within-prompt-centered features
  (centered on both the train and validation sides of each grouped fold).

These are ORACLE / diagnostic numbers, not deployable CV scores: the deployable
version needs test-time prompt-group recovery (Phase B), since ``test.csv`` has
no ``prompt_id``.  Nothing here uses labels at inference time.

Run:
    python experiments/phase_a_crossed.py --data-dir data
    python experiments/phase_a_crossed.py --data-dir data --char-features
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ARTIFACTS_DIR, LABELS
from src.crossed import assign_one_per_model, center_scores_within_groups
from src.data import indices_to_labels, labels_to_indices, load_competition_data
from src.ensemble import load_oof, normalize_scores
from src.features_tfidf import TextMatrixBuilder
from src.metrics import balanced_accuracy
from src.validation import load_splits

BLEND_WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}


def _bal(y_idx: np.ndarray, pred_idx: np.ndarray) -> float:
    return balanced_accuracy(indices_to_labels(y_idx), indices_to_labels(pred_idx))


def center_features(matrix, groups) -> np.ndarray:
    """Densify (if needed) and subtract the per-group feature mean."""
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    M = np.asarray(matrix, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        M[idx] -= M[idx].mean(axis=0)
    return M


def score_experiments(blend: np.ndarray, singles: dict, y: np.ndarray, groups) -> dict:
    out: dict = {}

    def variants(name: str, scores: np.ndarray) -> None:
        base = _bal(y, scores.argmax(axis=1))
        centered = center_scores_within_groups(scores, groups)
        a1 = _bal(y, centered.argmax(axis=1))
        a2 = _bal(y, assign_one_per_model(scores, groups))
        both = _bal(y, assign_one_per_model(centered, groups))
        out[name] = {
            "argmax": base,
            "A1_score_centering": a1,
            "A2_assignment": a2,
            "A1_plus_A2": both,
        }

    variants("blend", blend)
    for name, scores in singles.items():
        variants(name, scores)
    return out


def feature_centering_cv(
    train_df, folds, y: np.ndarray, groups, kinds, style, C, max_features, seed=42
) -> dict:
    """Grouped CV with features centered within prompt groups on both sides."""
    texts = train_df["text"].tolist()
    scores = np.full((len(train_df), len(LABELS)), -1e9, dtype=np.float32)
    fold_scores = []
    start = time.perf_counter()
    for train_idx, valid_idx in folds:
        builder = TextMatrixBuilder(
            kinds=kinds, style=style, params={"max_features": max_features}
        )
        m_tr = builder.fit_transform([texts[i] for i in train_idx], groups=groups[train_idx])
        m_va = builder.transform([texts[i] for i in valid_idx])
        x_tr = center_features(m_tr, groups[train_idx])
        x_va = center_features(m_va, groups[valid_idx])
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=2000, random_state=seed
        )
        clf.fit(x_tr, y[train_idx])
        fold = clf.decision_function(x_va)
        scores[valid_idx] = fold
        fold_scores.append(_bal(y[valid_idx], fold.argmax(axis=1)))
    elapsed = time.perf_counter() - start
    return {
        "pooled": _bal(y, scores.argmax(axis=1)),
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "fit_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--char-features",
        action="store_true",
        help="also retrain char_tfidf_logreg on centered features (densifies 100k cols; heavy)",
    )
    args = parser.parse_args()

    train_df, _ = load_competition_data(args.data_dir)
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(train_df["model"].tolist())

    oof = load_oof(ARTIFACTS_DIR / "cv" / "official")
    normalized = {name: normalize_scores(oof[name]["scores"]) for name in BLEND_WEIGHTS}
    blend = sum(normalized[n] * w for n, w in BLEND_WEIGHTS.items())
    singles = {n: normalized[n] for n in BLEND_WEIGHTS}

    results: dict = {"score_experiments": score_experiments(blend, singles, y, groups)}

    folds, _ = load_splits(ARTIFACTS_DIR / "splits" / "splits_official.json", train_df)
    results["A1_feature_style"] = feature_centering_cv(
        train_df, folds, y, groups, kinds=[], style=True, C=1.0, max_features=100000
    )
    if args.char_features:
        results["A1_feature_char"] = feature_centering_cv(
            train_df, folds, y, groups, kinds=["char"], style=False, C=2.0, max_features=100000
        )

    out_dir = ARTIFACTS_DIR / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # ---- report ----------------------------------------------------------
    print("\n=== Phase A oracle (known train prompt groups; NOT deployable CV) ===\n")
    print(f"{'model':<20} {'argmax':>8} {'A1 center':>10} {'A2 assign':>10} {'A1+A2':>8}")
    for name, v in results["score_experiments"].items():
        print(
            f"{name:<20} {v['argmax']:>8.4f} {v['A1_score_centering']:>10.4f} "
            f"{v['A2_assignment']:>10.4f} {v['A1_plus_A2']:>8.4f}"
        )
    fs = results["A1_feature_style"]
    print(
        f"\nA1-feature style retrain: pooled {fs['pooled']:.4f} "
        f"(fold {fs['fold_mean']:.4f} +/- {fs['fold_std']:.4f}), {fs['fit_seconds']}s"
    )
    if "A1_feature_char" in results:
        fc = results["A1_feature_char"]
        print(
            f"A1-feature char retrain:  pooled {fc['pooled']:.4f} "
            f"(fold {fc['fold_mean']:.4f} +/- {fc['fold_std']:.4f}), {fc['fit_seconds']}s"
        )
    base = results["score_experiments"]["blend"]["argmax"]
    best_oracle = max(
        results["score_experiments"]["blend"][k]
        for k in ("A1_score_centering", "A2_assignment", "A1_plus_A2")
    )
    print(f"\nblend baseline {base:.4f} -> best oracle {best_oracle:.4f} "
          f"(delta {best_oracle - base:+.4f})")
    print(f"\nresults written to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()