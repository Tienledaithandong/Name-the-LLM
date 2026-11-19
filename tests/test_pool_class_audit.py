"""Behavioral checks for aggregate-only external-pool diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.pool_class_audit import summarize_confusions, summarize_pool


def test_diagnostics_report_imbalance_and_confusions_without_rows() -> None:
    pool = pd.DataFrame({"model": ["a", "a", "b"], "gen_route": ["", "api", None]})
    summary = summarize_pool(pool, ["a", "b"])
    assert summary == {"a": {"rows": 2, "gen_route_missing": 1},
                       "b": {"rows": 1, "gen_route_missing": 1}}
    result = summarize_confusions(np.array([0, 0, 1]), np.array([[2, 1], [0, 3], [0, 2]]), ["a", "b"])
    assert result["a"] == {"support": 2, "recall": 0.5, "predicted_counts": {"a": 1, "b": 1}}
    assert result["b"]["recall"] == 1.0


def test_diagnostics_reject_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarize_confusions(np.array([0]), np.array([[np.nan]]), ["a"])
