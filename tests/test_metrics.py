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
    mean_ci,
    paired_difference,
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


def test_paired_difference_is_elementwise_other_minus_base():
    base = [0.40, 0.50, 0.60]
    other = [0.42, 0.48, 0.65]
    diff = paired_difference(base, other)
    assert diff == pytest.approx([0.02, -0.02, 0.05])


def test_paired_difference_rejects_length_mismatch():
    with pytest.raises(ValueError, match="paired length mismatch"):
        paired_difference([0.1, 0.2], [0.1])


def test_mean_ci_single_value_collapses_to_point():
    summary = mean_ci([0.42])
    assert summary["n"] == 1
    assert summary["mean"] == pytest.approx(0.42)
    assert summary["std"] == 0.0
    assert summary["ci_low"] == pytest.approx(0.42)
    assert summary["ci_high"] == pytest.approx(0.42)


def test_mean_ci_brackets_mean_with_sample_std():
    values = [0.40, 0.42, 0.44, 0.46]
    summary = mean_ci(values)
    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(0.43)
    assert summary["std"] == pytest.approx(np.std(values, ddof=1))
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]
    assert summary["ci_half"] == pytest.approx(
        summary["ci_high"] - summary["mean"]
    )


def test_mean_ci_rejects_empty_and_bad_confidence():
    with pytest.raises(ValueError, match="no values"):
        mean_ci([])
    with pytest.raises(ValueError, match="confidence"):
        mean_ci([0.1, 0.2], confidence=1.5)
