from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.blend_deberta import load_oof, verify_oof_alignment
from src.constants import LABELS, N_CLASSES
from src.data import ordered_dataset_fingerprint
from src.oof_identity import fold_assignments
from src.validation import LeakageError, load_splits, save_splits


def example_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "prompt_id": ["p1", "p1", "p2", "p2"],
        "text": ["one", "two", "three", "four"],
        "model": [LABELS[0], LABELS[1], LABELS[0], LABELS[1]],
    })


def test_ordered_fingerprint_detects_same_label_row_reorder_and_text_change():
    df = example_frame()
    original = ordered_dataset_fingerprint(df)
    assert ordered_dataset_fingerprint(df.iloc[[2, 1, 0, 3]]) != original
    modified = df.copy()
    modified.loc[0, "text"] += " changed"
    assert ordered_dataset_fingerprint(modified) != original


def test_kaggle_training_fingerprint_matches_local_implementation():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "deberta" / "deberta_train.py"
    spec = importlib.util.spec_from_file_location("deberta_training_identity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ordered_dataset_fingerprint(example_frame()) == ordered_dataset_fingerprint(example_frame())


def test_saved_splits_reject_reordered_rows_with_the_same_labels(tmp_path):
    df = example_frame()
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    path = save_splits(tmp_path / "splits.json", df, folds, "grouped", 42)
    load_splits(path, df)
    with pytest.raises(LeakageError, match="ordered dataset"):
        load_splits(path, df.iloc[[2, 1, 0, 3]].reset_index(drop=True))


def test_legacy_splits_reject_whole_prompt_group_reorder(tmp_path):
    df = example_frame()
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    path = save_splits(tmp_path / "splits.json", df, folds, "grouped", 42)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["ordered_dataset_sha256"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    # This permutation preserves the legacy, order-insensitive fingerprint and
    # group separation, while swapping which prompts these indices hold out.
    reordered = df.iloc[[2, 3, 0, 1]].reset_index(drop=True)
    with pytest.raises(LeakageError, match="ordered dataset fingerprint is missing"):
        load_splits(path, reordered)
    with pytest.raises(LeakageError, match="ordered dataset fingerprint is missing"):
        load_splits(path)


def test_saved_splits_recheck_prompt_groups_even_with_matching_fingerprint(tmp_path):
    df = example_frame()
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    path = save_splits(tmp_path / "splits.json", df, folds, "grouped", 42)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["folds"][0]["train"] = [1, 2, 3]
    payload["folds"][0]["valid"] = [0]
    payload["folds"][1]["train"] = [0]
    payload["folds"][1]["valid"] = [1, 2, 3]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LeakageError, match="leaks"):
        load_splits(path, df)


def test_fold_assignments_reject_overlap_and_missing_rows():
    folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    assert fold_assignments(4, folds).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        fold_assignments(4, [(np.array([0, 2, 3]), np.array([0, 1]))])
    with pytest.raises(ValueError):
        fold_assignments(4, folds[:1])


def test_blend_rejects_legacy_oof_and_mismatched_rows_or_folds(tmp_path):
    path = tmp_path / "legacy.npz"
    scores = np.zeros((4, N_CLASSES))
    y = np.array([0, 1, 0, 1])
    np.savez(path, scores=scores, y=y, labels=np.array(LABELS))
    with pytest.raises(ValueError, match="provenance missing"):
        load_oof(path)

    fold_ids = np.array([0, 0, 1, 1])
    base = (scores, y, "a" * 64, fold_ids)
    verify_oof_alignment([base, base], fold_ids, "a" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        verify_oof_alignment([base, (scores, y, "b" * 64, fold_ids)], fold_ids, "a" * 64)
    with pytest.raises(ValueError, match="fold"):
        verify_oof_alignment([base, (scores, y, "a" * 64, 1 - fold_ids)], fold_ids, "a" * 64)
    with pytest.raises(ValueError, match="differs from requested splits"):
        verify_oof_alignment([base, base], fold_ids, "b" * 64)
    with pytest.raises(ValueError, match="missing or invalid"):
        verify_oof_alignment([base, base], fold_ids, None)
