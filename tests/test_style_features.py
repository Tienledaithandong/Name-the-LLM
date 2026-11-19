"""The per-feature style analysis must aggregate folds and classes correctly.

Plan §4 L142 asks for per-feature and per-class analyses without causal claims.
These tests pin ``experiments/style_features.py``: fold statistics keep the class
dimension, the overall profile averages over classes (multinomial coefficients
cancel, so the signed overall mean is not informative), and the markdown writers
emit well-formed tables including a class with no support.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "style_features.py"
_spec = importlib.util.spec_from_file_location("style_features", SCRIPT)
style_features = importlib.util.module_from_spec(_spec)
sys.modules["style_features"] = style_features
_spec.loader.exec_module(style_features)


def _stack() -> np.ndarray:
    # two folds, three classes, four features
    return np.array([
        [[1.0, -2.0, 0.0, 0.5], [-1.0, 2.0, 0.0, -0.5], [0.0, 0.0, 3.0, 0.0]],
        [[3.0, -4.0, 0.0, 1.5], [-3.0, 4.0, 0.0, -1.5], [0.0, 0.0, 1.0, 0.0]],
    ])


def test_aggregate_keeps_the_class_dimension() -> None:
    stats = style_features.aggregate(_stack())
    for key in ("mean", "mean_abs", "std", "sign_agreement"):
        assert stats[key].shape == (3, 4), key
    assert stats["mean"][0] == pytest.approx([2.0, -3.0, 0.0, 1.0])
    assert stats["mean_abs"][0] == pytest.approx([2.0, 3.0, 0.0, 1.0])
    assert stats["sign_agreement"][0] == pytest.approx([1.0, 1.0, 0.0, 1.0])
    assert stats["std"][2] == pytest.approx([0.0, 0.0, np.sqrt(2.0), 0.0])


def test_single_fold_has_zero_spread() -> None:
    stats = style_features.aggregate(_stack()[:1])
    assert np.all(stats["std"] == 0.0)


def test_slice_stats_selects_a_class_or_averages_them() -> None:
    stats = style_features.aggregate(_stack())
    assert style_features.slice_stats(stats, 1)["mean"] == pytest.approx([-2.0, 3.0, 0.0, -1.0])
    overall = style_features.slice_stats(stats, None)
    assert overall["mean_abs"] == pytest.approx(np.abs(stats["mean_abs"]).mean(axis=0))
    # multinomial-style coefficients centred across classes cancel in the signed mean
    assert overall["mean"] == pytest.approx([0.0, 0.0, 2.0 / 3.0, 0.0])


def test_feature_rows_follows_the_given_order_and_limit() -> None:
    stats = style_features.aggregate(_stack())
    overall = style_features.slice_stats(stats, None)
    order = np.argsort(-overall["mean_abs"], kind="stable")
    rows = style_features.feature_rows(["a", "b", "c", "d"], overall, order, 2)
    assert [r["feature"] for r in rows] == ["b", "a"]
    assert rows[0]["mean_abs_coefficient"] == pytest.approx(2.0)
    assert set(rows[0]) == {"feature", "mean_coefficient", "mean_abs_coefficient", "fold_std", "sign_agreement"}


@pytest.mark.parametrize("signed,ncols", [(True, 5), (False, 4)])
def test_markdown_table_is_well_formed(signed: bool, ncols: int) -> None:
    stats = style_features.aggregate(_stack())
    overall = style_features.slice_stats(stats, None)
    rows = style_features.feature_rows(["a", "b", "c", "d"], overall, np.arange(4), 4)
    table = style_features.markdown_table(rows, signed=signed)
    lines = table.splitlines()
    assert len(lines) == 6
    assert all(line.count("|") == ncols + 1 for line in lines), table


def test_per_class_markdown_marks_an_unsupported_class() -> None:
    payload = {
        "gpt-5.5": {
            "style_only_recall": 0.5,
            "top_positive": [{"feature": "style:rate_dash", "mean_coefficient": 1.25}],
            "top_negative": [{"feature": "style:rate_comma", "mean_coefficient": -0.75}],
        },
        "kimi-k3": {"style_only_recall": float("nan"), "top_positive": [], "top_negative": []},
    }
    table = style_features.per_class_markdown(payload)
    lines = table.splitlines()
    assert len(lines) == 4
    assert all(line.count("|") == 5 for line in lines)
    assert "`rate_dash` (+1.250)" in lines[2]
    assert "| kimi-k3 | — |" in lines[3]