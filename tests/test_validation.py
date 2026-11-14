from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LABELS
from src.data import dataset_fingerprint
from src.validation import (
    LeakageError,
    fold_class_support,
    load_splits,
    make_holdout,
    make_splits,
    save_splits,
    validate_splits,
)


def test_splits_never_share_prompts(train_df, folds):
    prompts = train_df["prompt_id"].astype(str).to_numpy()
    covered = np.zeros(len(train_df), dtype=bool)
    for train_idx, valid_idx in folds:
        assert set(prompts[train_idx]).isdisjoint(set(prompts[valid_idx]))
        assert not covered[valid_idx].any()
        covered[valid_idx] = True
    assert covered.all()


def test_splits_are_deterministic(train_df):
    first, method_a = make_splits(train_df, n_splits=3, seed=5)
    second, method_b = make_splits(train_df, n_splits=3, seed=5)
    assert method_a == method_b
    for (tr_a, va_a), (tr_b, va_b) in zip(first, second, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


def test_more_folds_than_prompts_is_rejected(train_df):
    with pytest.raises(ValueError, match="distinct prompts"):
        make_splits(train_df, n_splits=train_df["prompt_id"].nunique() + 1)


def test_leakage_detector_catches_prompt_overlap():
    # Two rows per prompt so a prompt can straddle the fold boundary without
    # also sharing row indices (the row check would fire first).
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(8)],
            "prompt_id": [f"p{i // 2}" for i in range(8)],
            "text": ["sample text here"] * 8,
            "model": [LABELS[0], LABELS[0], LABELS[1], LABELS[1]] * 2,
        }
    )
    fold0 = (np.array([0, 2, 3]), np.array([1, 4, 5, 6, 7]))
    fold1 = (np.array([1, 4, 5, 6, 7]), np.array([0, 2, 3]))
    with pytest.raises(LeakageError, match="leaks"):
        validate_splits(frame, [fold0, fold1])


def test_leakage_detector_requires_full_coverage(train_df, folds):
    with pytest.raises(LeakageError):
        validate_splits(train_df, folds[:1])


def test_fold_class_support_covers_every_label(train_df, folds):
    support = fold_class_support(train_df, folds)
    assert len(support) == len(folds)
    for row in support:
        assert list(row) == list(LABELS)
        assert sum(row.values()) > 0


def test_splits_roundtrip_and_fingerprint_guard(train_df, folds, tmp_path):
    path = save_splits(tmp_path / "splits.json", train_df, folds, "stratified_group", 3)
    loaded, payload = load_splits(path, train_df)
    assert payload["dataset_fingerprint"] == dataset_fingerprint(train_df)
    assert payload["n_splits"] == len(folds)
    for (tr_a, va_a), (tr_b, va_b) in zip(folds, loaded, strict=True):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(va_a, va_b)

    changed = train_df.copy()
    changed.loc[0, "model"] = "kimi-k3" if changed.loc[0, "model"] != "kimi-k3" else "glm-5.3"
    with pytest.raises(LeakageError, match="does not match"):
        load_splits(path, changed)


def test_missing_split_file_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cli splits"):
        load_splits(tmp_path / "nope.json")


def test_holdout_is_prompt_disjoint(train_df):
    train_idx, hold_idx = make_holdout(train_df, n_groups=8, seed=1)
    prompts = train_df["prompt_id"].astype(str).to_numpy()
    assert len(np.intersect1d(train_idx, hold_idx)) == 0
    assert set(prompts[train_idx]).isdisjoint(set(prompts[hold_idx]))
    assert len(train_idx) + len(hold_idx) == len(train_df)


def test_holdout_rejects_consuming_every_prompt(train_df):
    with pytest.raises(ValueError, match="every prompt"):
        make_holdout(train_df, n_groups=train_df["prompt_id"].nunique())


def test_groupkfold_fallback_on_rare_classes():
    rows = [
        # class 0 appears under a single prompt only: only 1 group for 3 folds
        {"id": "a0", "prompt_id": "p0", "text": "text " * 5, "model": LABELS[0]},
        {"id": "b0", "prompt_id": "p0", "text": "text " * 5, "model": LABELS[1]},
        {"id": "b1", "prompt_id": "p1", "text": "text " * 5, "model": LABELS[1]},
        {"id": "c1", "prompt_id": "p1", "text": "text " * 5, "model": LABELS[2]},
        {"id": "c2", "prompt_id": "p2", "text": "text " * 5, "model": LABELS[2]},
        {"id": "d2", "prompt_id": "p2", "text": "text " * 5, "model": LABELS[3]},
    ]
    frame = pd.DataFrame(rows)
    folds, method = make_splits(frame, n_splits=3, seed=0)
    assert method == "group"
    assert len(folds) == 3
