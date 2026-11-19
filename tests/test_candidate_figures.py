from __future__ import annotations

import numpy as np
import pytest

from experiments.candidate_figures import aggregate_costs, blend, check, fold_scores
from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels


def test_check_accepts_the_recorded_value_and_rejects_drift():
    check(0.4426746209296544, 0.4426746209296544, "pooled")
    check([0.44, 0.45], np.asarray([0.44, 0.45]), "folds")
    with pytest.raises(SystemExit, match="no longer reproduces"):
        check(0.4426746209296544, 0.4426, "pooled")
    with pytest.raises(SystemExit, match="no longer reproduces"):
        check([0.44, 0.45], [0.45, 0.44], "folds")
    with pytest.raises(SystemExit, match="no longer reproduces"):
        check([0.44, 0.45], [0.44], "folds")


def test_aggregate_costs_adds_seconds_and_takes_the_larger_peak():
    costs = {"a": (100.0, 0.2), "b": (40.0, 0.3)}
    assert aggregate_costs(costs, ("a", "b")) == (140.0, 0.3)


def test_aggregate_costs_propagates_an_unrecorded_peak():
    # The pool arm's fit RSS was never recorded; the blend must inherit "unknown"
    # rather than silently report the other member's peak as its own.
    costs = {"pool": (1600.0, None), "style": (40.0, 0.2)}
    seconds, peak = aggregate_costs(costs, ("pool", "style"))
    assert seconds == 1640.0
    assert peak is None


def test_blend_applies_the_fixed_half_half_row_z_rule():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, N_CLASSES))
    b = rng.normal(size=(8, N_CLASSES))
    out = blend({"char_tfidf_logreg": a, "style_only": b})
    assert np.allclose(out, 0.5 * a + 0.5 * b)


def test_fold_scores_uses_only_the_validation_rows_of_each_fold():
    truth = np.asarray(LABELS, dtype=object)[np.arange(20) % N_CLASSES]
    scores = np.full((20, N_CLASSES), -1e9)
    scores[np.arange(20), np.arange(20) % N_CLASSES] = 1.0
    folds = [np.arange(0, 10), np.arange(10, 20)]
    assert fold_scores(scores, truth, folds) == [1.0, 1.0]
    scores[10:, :] = -1e9
    scores[10:, 0] = 1.0
    perfect, broken = fold_scores(scores, truth, folds)
    assert perfect == 1.0
    assert broken < 1.0
    assert indices_to_labels(scores[10:].argmax(axis=1))[0] == LABELS[0]