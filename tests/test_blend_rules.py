"""Blend-rule transforms are scale-only variants of the adopted row-z rule.

Plan §7 asks for simple averaging, rank-based and normalized-margin blending to be
compared. These tests pin the transforms used by ``experiments/blend_rules.py``:
hand-computed ranks/min-max/softmax, neutral handling of a constant row and of the
absent-class sentinel, and exact agreement of the ``row_z_mean`` arm with
``src.ensemble.normalize_scores`` (so the comparison shares the adopted baseline).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from src.ensemble import SENTINEL, normalize_scores

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "blend_rules.py"
_spec = importlib.util.spec_from_file_location("blend_rules", SCRIPT)
blend_rules = importlib.util.module_from_spec(_spec)
sys.modules["blend_rules"] = blend_rules
_spec.loader.exec_module(blend_rules)


def test_rank_transform_matches_hand_computation():
    out = blend_rules.transform_rows(np.array([[3.0, 1.0, 2.0]]), "rank_mean")
    assert out[0] == pytest.approx([1.0, 0.0, 0.5])


def test_minmax_transform_matches_hand_computation():
    out = blend_rules.transform_rows(np.array([[2.0, 4.0, 6.0]]), "minmax_mean")
    assert out[0] == pytest.approx([0.0, 0.5, 1.0])


def test_softmax_transform_is_positive_ordered_and_sums_to_one():
    out = blend_rules.transform_rows(np.array([[1.0, 2.0, 3.0]]), "softmax_mean")
    assert out.sum() == pytest.approx(1.0)
    assert (out > 0).all()
    assert out[0, 0] < out[0, 1] < out[0, 2]


def test_constant_row_gets_the_neutral_value():
    row = np.array([[5.0, 5.0, 5.0]])
    assert blend_rules.transform_rows(row, "minmax_mean")[0] == pytest.approx([0.5] * 3)
    assert blend_rules.transform_rows(row, "rank_mean")[0] == pytest.approx([0.5] * 3)
    assert blend_rules.transform_rows(row, "softmax_mean")[0] == pytest.approx([1 / 3] * 3)
    assert blend_rules.transform_rows(row, "raw_mean") == pytest.approx(row)


def test_absent_class_sentinel_gets_the_neutral_value_and_stays_finite():
    row = np.array([[2.0, SENTINEL * 10, 1.0]])
    for rule, neutral in (("rank_mean", 0.5), ("minmax_mean", 0.5), ("softmax_mean", 0.5)):
        out = blend_rules.transform_rows(row, rule)
        assert np.isfinite(out).all()
        assert out[0, 1] == pytest.approx(neutral)
    z = blend_rules.transform_rows(row, "row_z_mean")
    assert np.isfinite(z).all()
    assert z[0, 1] == pytest.approx(0.0)


def test_all_absent_row_does_not_produce_nan():
    row = np.full((1, 3), SENTINEL * 10)
    for rule in blend_rules.RULES:
        assert np.isfinite(blend_rules.transform_rows(row, rule)).all()


def test_row_z_arm_is_exactly_the_adopted_normalization():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(7, 18))
    scores[0, 3] = SENTINEL * 10
    assert blend_rules.transform_rows(scores, "row_z_mean") == pytest.approx(
        normalize_scores(scores)
    )


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown blend rule"):
        blend_rules.transform_rows(np.zeros((1, 3)), "median_mean")


def test_blend_is_the_weighted_sum_of_the_member_transforms():
    rng = np.random.default_rng(1)
    members = {name: rng.normal(size=(5, 18)) for name in blend_rules.WEIGHTS}
    for rule in blend_rules.RULES:
        expected = sum(
            blend_rules.transform_rows(members[name], rule) * weight
            for name, weight in blend_rules.WEIGHTS.items()
        )
        assert blend_rules.blend(members, rule) == pytest.approx(expected)