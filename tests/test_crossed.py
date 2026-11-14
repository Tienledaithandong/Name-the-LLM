"""Unit tests for the crossed-design inference transforms (src/crossed.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.crossed import (
    CrossedError,
    assign_one_per_model,
    center_scores_within_groups,
)


def test_centering_zeroes_group_means():
    scores = np.array([[1.0, 3.0], [3.0, 1.0], [10.0, 20.0], [20.0, 10.0]])
    groups = np.array(["a", "a", "b", "b"])
    out = center_scores_within_groups(scores, groups)
    assert out.shape == scores.shape
    assert np.allclose(out[:2].mean(axis=0), 0.0)
    assert np.allclose(out[2:].mean(axis=0), 0.0)
    # within-group differences are preserved
    assert np.allclose(out[0] - out[1], scores[0] - scores[1])


def test_centering_singleton_group_is_zero():
    scores = np.array([[5.0, -2.0]])
    out = center_scores_within_groups(scores, np.array(["solo"]))
    assert np.allclose(out, 0.0)


def test_assignment_enforces_unique_models():
    # Both passages prefer class 0, but the global optimum is row0->1, row1->0
    # (4 + 4.5 = 8.5 beats 5 + 1 = 6), so assignment overrides row0's argmax.
    scores = np.array([[5.0, 4.0], [4.5, 1.0]])
    groups = np.array(["p", "p"])
    pred = assign_one_per_model(scores, groups)
    assert sorted(pred.tolist()) == [0, 1]
    assert pred.tolist() == [1, 0]


def test_assignment_maximises_total_score():
    scores = np.array([[2.0, 1.0], [1.0, 0.0]])
    pred = assign_one_per_model(scores, np.array(["p", "p"]))
    # optimal pairing is (0->0, 1->1) total 2.0 vs (0->1,1->0) total 2.0 tie;
    # both are optimal, just assert a valid bijection
    assert sorted(pred.tolist()) == [0, 1]


def test_assignment_surplus_rows_fall_back_to_argmax():
    # 3 rows, 2 classes: optimal assignment covers rows 0 and 2 (5 + 9 = 14),
    # leaving row 1 on its plain argmax (class 0).
    scores = np.array([[5.0, 0.0], [4.0, 0.0], [0.0, 9.0]])
    pred = assign_one_per_model(scores, np.array(["p", "p", "p"]))
    assert pred.tolist() == [0, 0, 1]


def test_assignment_singleton_matches_argmax():
    scores = np.array([[0.1, 0.9, 0.2]])
    pred = assign_one_per_model(scores, np.array(["solo"]))
    assert pred[0] == 1


def test_groups_are_independent():
    scores = np.array([[5.0, 4.0], [4.5, 1.0], [0.0, 9.0]])
    groups = np.array(["a", "a", "b"])
    pred = assign_one_per_model(scores, groups)
    assert pred[2] == 1  # singleton group b unaffected by group a


def test_shape_mismatch_raises():
    with pytest.raises(CrossedError):
        center_scores_within_groups(np.zeros((2, 3)), np.array(["a"]))
    with pytest.raises(CrossedError):
        assign_one_per_model(np.zeros((2, 3)), np.array(["a"]))


def test_non_finite_scores_raise_for_assignment():
    with pytest.raises(CrossedError):
        assign_one_per_model(np.array([[np.nan, 1.0]]), np.array(["a"]))