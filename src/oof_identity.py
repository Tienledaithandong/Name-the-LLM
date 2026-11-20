"""Bind out-of-fold predictions to exact rows and validation folds."""

from __future__ import annotations

import numpy as np


def fold_assignments(n_rows: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Return each row's fold, rejecting missing, duplicated, or invalid splits."""
    if n_rows <= 0 or not folds:
        raise ValueError("OOF metadata requires rows and folds")
    assignments = np.full(n_rows, -1, dtype=np.int32)
    for index, (train, valid) in enumerate(folds):
        train = np.asarray(train, dtype=np.int64)
        valid = np.asarray(valid, dtype=np.int64)
        if train.ndim != 1 or valid.ndim != 1 or not train.size or not valid.size:
            raise ValueError("fold indices must be nonempty vectors")
        if (np.any(train < 0) or np.any(train >= n_rows) or
                np.any(valid < 0) or np.any(valid >= n_rows)):
            raise ValueError("fold index out of range")
        if np.unique(train).size != train.size or np.unique(valid).size != valid.size:
            raise ValueError("fold indices contain duplicates")
        if np.intersect1d(train, valid).size or train.size + valid.size != n_rows:
            raise ValueError("training indices must complement validation indices")
        if np.any(assignments[valid] != -1):
            raise ValueError("row appears in multiple validation folds")
        assignments[valid] = index
    if np.any(assignments < 0):
        raise ValueError("some rows have no validation fold")
    return assignments
