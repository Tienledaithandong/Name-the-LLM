"""Sweep D010 rank and SVD width, and test the cell as a third blend member.

D058 measured one D010 cell (char TF-IDF -> 128 SVD -> project out the top 4
between-prompt directions) at 0.2563 pooled OOF, while the adopted blend of
`char_tfidf_logreg` and `style_only` scores 0.375971.  That cell is the only
place where removing prompt directions helped, so this sweep asks two questions:

1. does the cell improve as the SVD width and the removed rank change?
2. does adding it to the adopted blend lift the blend's pooled OOF at all?

The blend recipe is fixed in advance: equal weight on the two adopted members
and the candidate, i.e. ``(z_char + z_style + z_candidate) / 3``.  No weight is
searched on the OOF, so no blend weight is fitted to validation rows.  The
sweep over (svd, k) is a **screen**: choosing the best cell by pooled OOF is
selection on the validation rows, so the reported numbers are exploratory and a
nested inner-selection gate is required before anything is adopted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import (  # noqa: E402
    DEFAULT_REGISTRY,
    LABELS,
    N_CLASSES,
    RANDOM_STATE,
)
from src.data import (  # noqa: E402
    find_competition_file,
    indices_to_labels,
    labels_to_indices,
    ordered_dataset_fingerprint,
    validate_train_frame,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.features_tfidf import TextMatrixBuilder  # noqa: E402
from src.metrics import balanced_accuracy  # noqa: E402
from src.prompt_invariance import NuisanceSuppressor  # noqa: E402
from src.repro import git_commit  # noqa: E402
from src.train import append_registry  # noqa: E402
from src.validation import load_splits  # noqa: E402

SVD_WIDTHS = (64, 128, 256)
NUISANCE_KS = (2, 4, 8, 16)
MEMBERS = ("char_tfidf_logreg", "style_only")
ADOPTED_BA = 0.3759706686552324
SENTINEL = -1e9


def _expand(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.full((scores.shape[0], N_CLASSES), SENTINEL, dtype=np.float32)
    out[:, np.asarray(classes, dtype=np.int64)] = np.asarray(scores, dtype=np.float32)
    return out


def cell_oof(
    texts,
    y,
    groups,
    folds,
    svd: int,
    nuisance_k: int,
    c: float = 1.0,
) -> np.ndarray:
    """Out-of-fold scores for char TF-IDF -> SVD -> D010 projection -> logreg."""
    texts = list(texts)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    scores = np.full((len(texts), N_CLASSES), SENTINEL, dtype=np.float32)
    for train_idx, valid_idx in folds:
        builder = TextMatrixBuilder(kinds=("char",), svd=int(svd)).fit(
            [texts[i] for i in train_idx], groups=[groups[i] for i in train_idx]
        )
        train_matrix = np.asarray(
            builder.transform([texts[i] for i in train_idx]), dtype=np.float32
        )
        valid_matrix = np.asarray(
            builder.transform([texts[i] for i in valid_idx]), dtype=np.float32
        )
        if nuisance_k:
            suppressor = NuisanceSuppressor(
                n_components=int(nuisance_k), min_group_size=4
            ).fit(train_matrix, groups[train_idx])
            train_matrix = suppressor.transform(train_matrix)
            valid_matrix = suppressor.transform(valid_matrix)
        classifier = LogisticRegression(
            C=c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ).fit(train_matrix, y[train_idx])
        scores[valid_idx] = _expand(
            classifier.decision_function(valid_matrix), classifier.classes_
        )
    return scores


def blend_with_third(adopted: np.ndarray, third: np.ndarray) -> np.ndarray:
    """Equal thirds: two adopted members plus one candidate.

    ``adopted`` is already ``0.5*z_char + 0.5*z_style``, so this is
    ``(2*adopted + third) / 3`` and searches no weight.
    """
    return (2.0 * np.asarray(adopted, dtype=np.float64) + np.asarray(third, dtype=np.float64)) / 3.0


def _pooled(y: np.ndarray, scores: np.ndarray) -> float:
    return balanced_accuracy(
        indices_to_labels(y), indices_to_labels(np.asarray(scores).argmax(axis=1))
    )


def _fold_scores(y: np.ndarray, scores: np.ndarray, folds) -> list[float]:
    return [
        balanced_accuracy(
            indices_to_labels(y[valid_idx]),
            indices_to_labels(np.asarray(scores)[valid_idx].argmax(axis=1)),
        )
        for _, valid_idx in folds
    ]


def run_sweep(
    texts,
    y,
    groups,
    folds,
    adopted_scores: np.ndarray,
    svd_widths=SVD_WIDTHS,
    nuisance_ks=NUISANCE_KS,
) -> dict:
    """Score every (svd, k) cell alone and as an equal-weighted third member."""
    cells: dict[str, dict] = {}
    for svd in svd_widths:
        for k in nuisance_ks:
            key = f"svd{int(svd)}_k{int(k)}"
            raw = cell_oof(texts, y, groups, folds, svd, k)
            third = normalize_scores(raw)
            blended = blend_with_third(adopted_scores, third)
            cells[key] = {
                "svd": int(svd),
                "nuisance_k": int(k),
                "single_view": float(_pooled(y, raw)),
                "blend": float(_pooled(y, blended)),
                "blend_fold_scores": [float(v) for v in _fold_scores(y, blended, folds)],
                "single_view_fold_scores": [
                    float(v) for v in _fold_scores(y, third, folds)
                ],
            }
    return cells


def load_official_train(data_dir):
    import pandas as pd

    path = find_competition_file("train.csv", data_dir)
    frame = validate_train_frame(pd.read_csv(path))
    if set(frame["model"]) != set(LABELS):
        raise ValueError("expected support for all 18 official labels")
    return frame


def adopted_matrix(oof_dir: Path, y: np.ndarray, folds) -> np.ndarray:
    """Rebuild the adopted 0.5/0.5 blend from its registered members."""
    oof = load_oof(oof_dir, list(MEMBERS))
    reference = oof[MEMBERS[0]]
    if not np.array_equal(reference["y"], y):
        raise ValueError("saved OOF labels are not aligned with the training frame")
    normalized = {name: normalize_scores(oof[name]["scores"]) for name in MEMBERS}
    return 0.5 * normalized[MEMBERS[0]] + 0.5 * normalized[MEMBERS[1]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--splits", default="artifacts/splits/splits_official_verified_s42.json")
    parser.add_argument("--oof-dir", default="artifacts/cv_identity/official")
    parser.add_argument("--out", default="artifacts/residual/d010_sweep.json")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args(argv)

    frame = load_official_train(args.data_dir)
    folds, payload = load_splits(args.splits, frame)
    split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
    texts = frame["text"].tolist()
    groups = frame["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(frame["model"].tolist())

    adopted = adopted_matrix(Path(args.oof_dir), y, folds)
    adopted_ba = _pooled(y, adopted)
    if abs(adopted_ba - ADOPTED_BA) > 5e-7:
        raise SystemExit(
            f"adopted blend scored {adopted_ba:.6f} but the registered value is "
            f"{ADOPTED_BA:.6f}; refusing to report deltas against a different control"
        )

    started = time.perf_counter()
    cells = run_sweep(texts, y, groups, folds, adopted)
    elapsed = round(time.perf_counter() - started, 2)

    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "official",
        "split_id": split_id,
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "ordered_dataset_sha256": ordered_dataset_fingerprint(frame),
        "n_rows": int(len(frame)),
        "n_prompts": int(frame["prompt_id"].nunique()),
        "adopted": {
            "members": list(MEMBERS),
            "weights": [0.5, 0.5],
            "pooled": float(adopted_ba),
            "fold_scores": [float(v) for v in _fold_scores(y, adopted, folds)],
        },
        "recipe": "equal thirds (no weight search)",
        "status": "exploratory: (svd, k) chosen by pooled OOF is selection on validation",
        "cells": cells,
        "seconds": elapsed,
        "command": "python -m experiments.sweep_d010 " + " ".join(sys.argv[1:]),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    if not args.no_registry:
        commit = git_commit(Path(__file__).resolve().parents[1])
        for key, cell in cells.items():
            append_registry(
                args.registry,
                {
                    "experiment_id": (
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official-"
                        f"d010-{key}-s{RANDOM_STATE}"
                    ),
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": commit,
                    "method": f"blend_char+style+{key}",
                    "features": (
                        f"0.5 char_tfidf_logreg + 0.5 style_only + candidate "
                        f"char|svd{cell['svd']}|nuisance{cell['nuisance_k']}|logreg "
                        f"(equal thirds)"
                    ),
                    "seed": RANDOM_STATE,
                    "split_id": split_id,
                    "fold_scores": " ".join(
                        f"{v:.4f}" for v in cell["blend_fold_scores"]
                    ),
                    "pooled_oof_balanced_accuracy": round(cell["blend"], 6),
                    "macro_f1": "",
                    "fit_seconds": "",
                    "infer_seconds": "",
                    "peak_ram_gb": "",
                    "external_data_id": "none",
                    "artifact_path": str(out),
                    "public_lb_score": "",
                    "notes": (
                        f"Sweep screen, not a candidate: cell single-view "
                        f"{cell['single_view']:.6f}; adopted control {adopted_ba:.6f}; "
                        f"equal thirds weights, no OOF weight search; "
                        f"command: {result['command']}"
                    ),
                    "status": "exploratory",
                },
            )

    print(f"[d010] split {split_id} | adopted {adopted_ba:.4f} | {elapsed}s")
    print(f"{'cell':14s} {'single':>7s} {'blend':>7s} {'delta':>8s}")
    for key, cell in cells.items():
        print(
            f"{key:14s} {cell['single_view']:7.4f} {cell['blend']:7.4f} "
            f"{cell['blend'] - adopted_ba:+8.4f}"
        )
    best = max(cells, key=lambda k: cells[k]["blend"])
    print(f"[d010] best blend cell {best} at {cells[best]['blend']:.4f}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
