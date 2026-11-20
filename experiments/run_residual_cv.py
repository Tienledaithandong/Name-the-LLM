"""The {none, D010, learned-g} x {style, svd128, embedding} table (spec 2.1).

Approach A generalises our published negative result D010 (global projection of
the top between-prompt directions), so the first measurement must isolate
whether the *learned, per-passage* neutralizer adds anything over that
projection on the same representation.  Every cell therefore differs only in
the representation transform; the classifier is one fixed logistic regression.

Shrinkage is chosen inside each outer fold on its training side with an inner
prompt-grouped CV, then the chosen value is refit on the whole training side.
Validation rows never influence the choice and never contribute a prompt mean.

    python -m experiments.run_residual_cv \
        --splits artifacts/splits/splits_official_verified_s42.json \
        --out artifacts/residual/cv_s42.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

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
from src.features_tfidf import TextMatrixBuilder  # noqa: E402
from src.metrics import balanced_accuracy  # noqa: E402
from src.prompt_invariance import NuisanceSuppressor, centered_features  # noqa: E402
from src.repro import git_commit  # noqa: E402
from src.residual import PromptMeanNeutralizer  # noqa: E402
from src.train import append_registry  # noqa: E402
from src.validation import load_splits  # noqa: E402

MODES = ("none", "d010", "learned")
SPACES = ("style", "svd128", "embedding")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SENTINEL = -1e9

SPACE_KWARGS: dict[str, dict] = {
    "style": {"kinds": (), "style": True},
    "svd128": {"kinds": ("char",), "svd": 128},
    "embedding": {"kinds": (), "embedding": {"backend": "hashing", "dim": 384}},
}


def _dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _classifier() -> LogisticRegression:
    """One classifier for every cell, so only the representation differs."""
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def base_matrices(
    texts,
    groups,
    train_idx,
    valid_idx,
    space: str = "style",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the named representation on the training side and transform both."""
    kwargs = dict(SPACE_KWARGS[space])
    builder = TextMatrixBuilder(**kwargs).fit(
        [texts[i] for i in train_idx], groups=[groups[i] for i in train_idx]
    )
    return (
        _dense(builder.transform([texts[i] for i in train_idx])),
        _dense(builder.transform([texts[i] for i in valid_idx])),
    )


def _residualize(Xtr, Xva, groups_tr, lam: float, alpha: float):
    """Apply ``e - lam * g(e)``; ``lam=0`` is an exact no-op."""
    if not lam:
        return Xtr, Xva, None
    neutralizer = PromptMeanNeutralizer(alpha=alpha, shrinkage=lam).fit(Xtr, groups_tr)
    return neutralizer.transform(Xtr), neutralizer.transform(Xva), neutralizer


def _expand(scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Place a classifier's columns into the official 18-class order."""
    scores = np.asarray(scores, dtype=np.float32)
    n = scores.shape[0]
    out = np.full((n, N_CLASSES), SENTINEL, dtype=np.float32)
    out[:, np.asarray(classes, dtype=np.int64)] = scores
    return out


def _d010(Xtr, Xva, groups_tr, k: int):
    """D010: project out the top-k directions of between-prompt variation.

    Applied to the dense representation directly rather than through the
    builder, because D010 is defined on any dense space and the builder's
    suppression stage intentionally requires an SVD reduction.
    """
    if not k:
        return Xtr, Xva, None
    suppressor = NuisanceSuppressor(n_components=int(k), min_group_size=4).fit(
        Xtr, groups_tr
    )
    return suppressor.transform(Xtr), suppressor.transform(Xva), suppressor


def _score_cell(Xtr, ytr, Xva, groups_tr, lam: float, alpha: float) -> np.ndarray:
    Xtr, Xva, _ = _residualize(Xtr, Xva, groups_tr, lam, alpha)
    classifier = _classifier().fit(Xtr, ytr)
    return _expand(classifier.decision_function(Xva), classifier.classes_)


def select_lambda(
    texts,
    y,
    groups,
    lambdas=LAMBDAS,
    space: str = "style",
    alpha: float = 1.0,
    n_inner: int = 3,
) -> tuple[float, list[dict]]:
    """Choose shrinkage by inner prompt-grouped CV on these rows only."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    n_splits = min(int(n_inner), int(np.unique(groups).size))
    if n_splits < 2:
        return 0.0, []
    texts = list(texts)
    parts = []
    for inner_train, inner_valid in GroupKFold(n_splits=n_splits).split(
        np.zeros((len(texts), 1)), y, groups
    ):
        Xtr, Xva = base_matrices(texts, groups, inner_train, inner_valid, space=space)
        parts.append((Xtr, y[inner_train], Xva, y[inner_valid], groups[inner_train]))

    rows: list[dict] = []
    for lam in lambdas:
        fold_scores = []
        for Xtr, ytr, Xva, yva, groups_tr in parts:
            scores = _score_cell(Xtr, ytr, Xva, groups_tr, lam, alpha)
            fold_scores.append(
                balanced_accuracy(
                    indices_to_labels(yva), indices_to_labels(scores.argmax(axis=1))
                )
            )
        rows.append(
            {
                "lambda": float(lam),
                "inner_balanced_accuracy": float(np.mean(fold_scores)),
            }
        )
    best = max(rows, key=lambda r: (r["inner_balanced_accuracy"], -r["lambda"]))
    return float(best["lambda"]), rows


def _pool(scores: np.ndarray, y: np.ndarray) -> float:
    return balanced_accuracy(
        indices_to_labels(y), indices_to_labels(scores.argmax(axis=1))
    )


def run_residual_table(
    texts,
    y,
    groups,
    folds,
    spaces: tuple[str, ...] = SPACES,
    lambdas: tuple[float, ...] = LAMBDAS,
    nuisance_k: int = 4,
    alpha: float = 1.0,
    n_inner: int = 3,
) -> dict:
    """Run every cell on the given outer folds and report pooled OOF scores."""
    texts = list(texts)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    n = len(texts)
    cells: dict[str, dict] = {}
    oracle: dict[str, dict] = {}

    for space in spaces:
        pooled = {mode: np.full((n, N_CLASSES), SENTINEL, dtype=np.float32) for mode in MODES}
        fold_scores: dict[str, list[float]] = {mode: [] for mode in MODES}
        selection: list[dict] = []
        diagnostics: list[dict] = []
        oracle_scores = np.full((n, N_CLASSES), SENTINEL, dtype=np.float32)
        oracle_folds: list[float] = []

        for train_idx, valid_idx in folds:
            Xtr, Xva = base_matrices(texts, groups, train_idx, valid_idx, space=space)

            for mode in ("none", "learned"):
                if mode == "none":
                    lam = 0.0
                    scores = _score_cell(Xtr, y[train_idx], Xva, groups[train_idx], 0.0, alpha)
                else:
                    lam, rows = select_lambda(
                        [texts[i] for i in train_idx],
                        y[train_idx],
                        groups[train_idx],
                        lambdas=lambdas,
                        space=space,
                        alpha=alpha,
                        n_inner=n_inner,
                    )
                    selection.append({"fold": len(selection), "lambda": lam, "inner": rows})
                    Xtr_lam, Xva_lam, neutralizer = _residualize(
                        Xtr, Xva, groups[train_idx], lam, alpha
                    )
                    classifier = _classifier().fit(Xtr_lam, y[train_idx])
                    scores = _expand(
                        classifier.decision_function(Xva_lam), classifier.classes_
                    )
                    if neutralizer is not None:
                        diagnostics.append(
                            {
                                "fold": len(diagnostics),
                                "oof_r2": float(neutralizer.oof_r2_),
                                "predicted_norm_ratio": float(
                                    neutralizer.predicted_norm_ratio_
                                ),
                            }
                        )
                pooled[mode][valid_idx] = scores
                fold_scores[mode].append(
                    balanced_accuracy(
                        indices_to_labels(y[valid_idx]),
                        indices_to_labels(scores.argmax(axis=1)),
                    )
                )

            Xtr_d, Xva_d, _suppressor = _d010(
                Xtr, Xva, groups[train_idx], nuisance_k
            )
            scores_d = _score_cell(
                Xtr_d, y[train_idx], Xva_d, groups[train_idx], 0.0, alpha
            )
            pooled["d010"][valid_idx] = scores_d
            fold_scores["d010"].append(
                balanced_accuracy(
                    indices_to_labels(y[valid_idx]),
                    indices_to_labels(scores_d.argmax(axis=1)),
                )
            )

            # Oracle: centre by the *true* group, including validation groups.
            # Diagnostic ceiling only; never a candidate.
            oracle_scores[valid_idx] = _score_cell(
                centered_features(Xtr, groups[train_idx]),
                y[train_idx],
                centered_features(Xva, groups[valid_idx]),
                groups[train_idx],
                0.0,
                alpha,
            )
            oracle_folds.append(
                balanced_accuracy(
                    indices_to_labels(y[valid_idx]),
                    indices_to_labels(oracle_scores[valid_idx].argmax(axis=1)),
                )
            )

        for mode in MODES:
            selected = [row["lambda"] for row in selection] if mode == "learned" else None
            r2 = [row["oof_r2"] for row in diagnostics] if mode == "learned" else None
            cells[f"{space}|{mode}"] = {
                "space": space,
                "mode": mode,
                "pooled": float(_pool(pooled[mode], y)),
                "fold_scores": [float(v) for v in fold_scores[mode]],
                "selected_lambda": selected,
                "oof_r2": r2,
                "predicted_norm_ratio": (
                    [row["predicted_norm_ratio"] for row in diagnostics]
                    if mode == "learned"
                    else None
                ),
            }
        oracle[f"{space}|oracle"] = {
            "space": space,
            "pooled": float(_pool(oracle_scores, y)),
            "fold_scores": [float(v) for v in oracle_folds],
        }

    return {
        "cells": cells,
        "oracle": oracle,
        "lambdas": [float(v) for v in lambdas],
        "nuisance_k": int(nuisance_k),
    }


def load_official_train(data_dir):
    import pandas as pd

    path = find_competition_file("train.csv", data_dir)
    frame = validate_train_frame(pd.read_csv(path))
    if set(frame["model"]) != set(LABELS):
        raise ValueError("expected support for all 18 official labels")
    return frame


def _registry_row(cell: dict, command: str, split_id: str, commit: str) -> dict:
    selected = cell["selected_lambda"]
    return {
        "experiment_id": (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official-residual-"
            f"{cell['space']}-{cell['mode']}"
        ),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "method": f"residual_{cell['space']}_{cell['mode']}",
        "features": f"{cell['space']}|logreg|{cell['mode']}",
        "seed": RANDOM_STATE,
        "split_id": split_id,
        "fold_scores": " ".join(f"{v:.4f}" for v in cell["fold_scores"]),
        "pooled_oof_balanced_accuracy": round(cell["pooled"], 6),
        "macro_f1": "",
        "fit_seconds": "",
        "infer_seconds": "",
        "peak_ram_gb": "",
        "external_data_id": "none",
        "artifact_path": "",
        "public_lb_score": "",
        "notes": (
            f"lambdas {LAMBDAS}; selected {selected}; "
            f"oof_r2 {[round(v, 4) for v in (cell['oof_r2'] or [])]}; "
            f"command: {command}"
        ),
        "status": "done",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--splits", default="artifacts/splits/splits_official_verified_s42.json")
    parser.add_argument("--out", default="artifacts/residual/cv_s42.json")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--spaces", nargs="+", default=list(SPACES))
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(LAMBDAS))
    parser.add_argument("--nuisance-k", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args(argv)

    frame = load_official_train(args.data_dir)
    folds, payload = load_splits(args.splits, frame)
    split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
    texts = frame["text"].tolist()
    groups = frame["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(frame["model"].tolist())

    started = time.perf_counter()
    result = run_residual_table(
        texts,
        y,
        groups,
        folds,
        spaces=tuple(args.spaces),
        lambdas=tuple(args.lambdas),
        nuisance_k=args.nuisance_k,
        n_inner=args.inner_splits,
    )
    result.update(
        {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "official",
            "split_id": split_id,
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "ordered_dataset_sha256": ordered_dataset_fingerprint(frame),
            "n_rows": int(len(frame)),
            "n_prompts": int(frame["prompt_id"].nunique()),
            "command": "python -m experiments.run_residual_cv " + " ".join(sys.argv[1:]),
            "seconds": round(time.perf_counter() - started, 2),
            "baseline_reference": {
                "name": "char_tfidf_logreg+style_only 0.5/0.5",
                "pooled_oof_balanced_accuracy": 0.3759706686552324,
            },
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    if not args.no_registry:
        commit = git_commit(Path(__file__).resolve().parents[1])
        for cell in result["cells"].values():
            append_registry(args.registry, _registry_row(cell, result["command"], split_id, commit))

    print(f"[residual] split {split_id} | {len(frame)} rows | {result['seconds']}s")
    header = f"{'cell':22s} {'pooled':>8s} {'folds':>34s}"
    print(header)
    for key, cell in result["cells"].items():
        folds_text = " ".join(f"{v:.3f}" for v in cell["fold_scores"])
        print(f"{key:22s} {cell['pooled']:8.4f} {folds_text:>34s}")
    for key, cell in result["oracle"].items():
        print(f"{key:22s} {cell['pooled']:8.4f} (diagnostic ceiling)")
    print(f"[residual] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
