from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    aggregate_folds,
    balanced_accuracy,
    class_support,
    confusion_matrix,
    evaluate,
    macro_f1,
    per_class_recall,
    within_provider_confusions,
)

LABELS3 = ("a", "b", "c")


def test_balanced_accuracy_matches_hand_computation():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = ["a", "b", "b", "b", "c"]
    # recalls: a=0.5, b=1.0, c=1.0
    assert balanced_accuracy(y_true, y_pred, LABELS3) == pytest.approx(2.5 / 3)
    recalls = per_class_recall(y_true, y_pred, LABELS3)
    assert recalls["a"] == pytest.approx(0.5)
    assert recalls["b"] == pytest.approx(1.0)


def test_absent_classes_are_skipped_not_scored_as_zero():
    y_true = ["a", "a", "b"]
    y_pred = ["a", "a", "a"]
    recalls = per_class_recall(y_true, y_pred, LABELS3)
    assert np.isnan(recalls["c"])
    # mean over a and b only: (1.0 + 0.0) / 2
    assert balanced_accuracy(y_true, y_pred, LABELS3) == pytest.approx(0.5)
    assert class_support(y_true, LABELS3)["c"] == 0


def test_perfect_predictions_score_one():
    y = ["a", "b", "c", "a", "b", "c"]
    assert balanced_accuracy(y, y, LABELS3) == pytest.approx(1.0)


def test_metric_requires_matching_lengths():
    with pytest.raises(ValueError, match="shape mismatch"):
        balanced_accuracy(["a", "b"], ["a"], LABELS3)


def test_metric_rejects_integer_indices():
    with pytest.raises(TypeError, match="label strings"):
        balanced_accuracy([0, 1], [0, 1], LABELS3)


def test_macro_f1_and_confusion_matrix():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    # a: p=1.0 r=0.5 -> f1=0.667 ; b: p=0.667 r=1.0 -> f1=0.8
    assert macro_f1(y_true, y_pred, LABELS3) == pytest.approx((2 / 3 + 0.8) / 2)
    matrix = confusion_matrix(y_true, y_pred, LABELS3)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4
    assert matrix[0, 0] == 1 and matrix[0, 1] == 1 and matrix[2].sum() == 0


def test_evaluate_reports_absent_classes_and_support():
    result = evaluate(["a", "b"], ["a", "a"], LABELS3, model="m")
    assert result["classes_absent"] == ["c"]
    assert result["n"] == 2
    assert result["support"]["c"] == 0
    assert 0.0 <= result["balanced_accuracy"] <= 1.0


def test_aggregate_folds_statistics():
    folds = [
        {"balanced_accuracy": 0.4},
        {"balanced_accuracy": 0.5},
        {"balanced_accuracy": 0.6},
    ]
    agg = aggregate_folds(folds)
    assert agg["n_folds"] == 3
    assert agg["mean"] == pytest.approx(0.5)
    assert agg["std"] == pytest.approx(np.std([0.4, 0.5, 0.6]))
    assert agg["min"] == pytest.approx(0.4) and agg["max"] == pytest.approx(0.6)


def test_within_provider_confusions_counts_same_provider_errors():
    y_true = ["gpt-5.5", "gpt-5.5", "kimi-k3"]
    y_pred = ["gpt-6-astra", "kimi-k3", "kimi-k3"]
    counts = within_provider_confusions(y_true, y_pred)
    assert counts["errors"] == 2
    assert counts["within_provider_errors"] == 1
    assert counts["cross_provider_errors"] == 1
