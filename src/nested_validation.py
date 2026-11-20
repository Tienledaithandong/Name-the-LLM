"""Prepare nested prompt-grouped splits without fitting a model or scoring rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation import make_splits, validate_splits


def make_nested_splits(
    df: pd.DataFrame, *, outer_splits: int = 3, inner_splits: int = 3,
    outer_seed: int = 121, inner_seed: int = 212,
) -> list[dict]:
    """Return global indices for outer folds and inner folds on outer training data.

    The caller must perform model and blend selection using only inner folds and
    run the selected procedure on each outer holdout once. Merely creating these
    splits does not make earlier development decisions independent.
    """
    outer, outer_method = make_splits(df, n_splits=outer_splits, seed=outer_seed)
    plans = []
    for index, (outer_train, outer_valid) in enumerate(outer):
        train_df = df.iloc[outer_train].reset_index(drop=True)
        inner, inner_method = make_splits(
            train_df, n_splits=inner_splits, seed=inner_seed + index
        )
        validate_splits(train_df, inner)
        global_inner = [(outer_train[tr], outer_train[va]) for tr, va in inner]
        if any(np.intersect1d(outer_valid, np.concatenate((tr, va))).size for tr, va in global_inner):
            raise ValueError("inner split overlaps outer validation rows")
        plans.append({
            "outer_fold": index, "outer_method": outer_method,
            "train": outer_train, "valid": outer_valid,
            "inner_method": inner_method, "inner": global_inner,
        })
    return plans
