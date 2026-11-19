from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from experiments import pool_train
from experiments.pool_train import official_oof, save_pool_oof
from src.constants import LABELS
from src.data import labels_to_indices, ordered_dataset_fingerprint
from src.oof_identity import fold_assignments


def test_global_pool_clean_ids_are_rejected(tmp_path, monkeypatch):
    # D028 recomputes the clean pool globally from raw data; a prefiltered ID list
    # (whose provenance could depend on validation text) must fail closed rather than
    # silently change the training set.
    monkeypatch.setattr(sys, "argv", ["pool_train.py", "--clean-ids",
                                      str(tmp_path / "clean_pool_ids.json")])
    with pytest.raises(SystemExit, match="clean-ids is unsupported"):
        pool_train.main()


@pytest.fixture
def sample():
    frame = pd.DataFrame({
        "id": ["row-1", "row-2", "row-3", "row-4"],
        "prompt_id": ["p1", "p2", "p3", "p4"],
        "text": ["one text", "two text", "three text", "four text"],
        "model": [LABELS[0], LABELS[0], LABELS[1], LABELS[1]],
    })
    folds = [(np.array([2, 3]), np.array([0, 1])),
             (np.array([0, 1]), np.array([2, 3]))]
    return frame, folds


def save_oof(path, frame, folds, *, omit=(), fingerprint=None, assigned=None):
    fields = {
        "scores": np.zeros((len(frame), len(LABELS))),
        "y": labels_to_indices(frame["model"].tolist()),
        "labels": np.asarray(LABELS, dtype=object),
        "ordered_dataset_sha256": np.asarray(
            fingerprint if fingerprint is not None else ordered_dataset_fingerprint(frame)),
        "validation_fold_ids": (
            assigned if assigned is not None else fold_assignments(len(frame), folds)),
    }
    np.savez_compressed(path, **{key: value for key, value in fields.items() if key not in omit})


def test_official_oof_accepts_exact_data_and_folds(tmp_path, sample):
    frame, folds = sample
    save_oof(tmp_path / "model.npz", frame, folds)
    scores, y = official_oof(tmp_path, "model", frame, folds)
    assert scores.shape == (4, len(LABELS))
    assert len(y) == 4


def test_official_oof_rejects_legacy_or_reordered_data(tmp_path, sample):
    frame, folds = sample
    path = tmp_path / "model.npz"
    save_oof(path, frame, folds, omit=("ordered_dataset_sha256", "validation_fold_ids"))
    with pytest.raises(ValueError, match="provenance missing"):
        official_oof(tmp_path, "model", frame, folds)
    save_oof(path, frame, folds)
    reordered = frame.iloc[[1, 0, 2, 3]].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered dataset fingerprint"):
        official_oof(tmp_path, "model", reordered, folds)


def test_official_oof_rejects_different_validation_folds(tmp_path, sample):
    frame, folds = sample
    save_oof(tmp_path / "model.npz", frame, folds, assigned=np.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="validation folds"):
        official_oof(tmp_path, "model", frame, folds)


def test_pool_oof_round_trip_preserves_required_provenance(tmp_path, sample):
    frame, folds = sample
    scores = np.zeros((len(frame), len(LABELS)))
    labels = labels_to_indices(frame["model"].tolist())
    save_pool_oof(tmp_path / "model.npz", scores, labels, frame, folds)
    loaded_scores, loaded_labels = official_oof(tmp_path, "model", frame, folds)
    assert np.array_equal(scores, loaded_scores)
    assert np.array_equal(labels, loaded_labels)
    with pytest.raises(ValueError, match="scores or labels"):
        save_pool_oof(tmp_path / "invalid.npz", scores, labels[::-1], frame, folds)
