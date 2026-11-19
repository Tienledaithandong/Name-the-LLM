"""Nested CV must isolate every held-out prompt from inner model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LABELS
from src.nested_validation import make_nested_splits


def test_inner_folds_never_use_outer_validation_prompts() -> None:
    df = pd.DataFrame([{"id": f"{p}-{m}", "text": "synthetic", "prompt_id": f"p{p}", "model": m}
                       for p in range(12) for m in LABELS[:2]])
    plans = make_nested_splits(df, outer_splits=3, inner_splits=3)
    for plan in plans:
        outer_train = set(plan["train"])
        outer_prompts = set(df.iloc[plan["valid"]]["prompt_id"])
        inner_valid = []
        for train, valid in plan["inner"]:
            assert set(train) | set(valid) == outer_train
            assert not outer_prompts.intersection(df.iloc[np.concatenate((train, valid))]["prompt_id"])
            assert set(df.iloc[train]["prompt_id"]).isdisjoint(df.iloc[valid]["prompt_id"])
            inner_valid.extend(valid.tolist())
        assert sorted(inner_valid) == sorted(outer_train)
