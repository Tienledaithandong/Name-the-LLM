from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    aggregate_folds,
    balanced_accuracy,
    class_support,
    confusion_matrix,
    evaluate,
    length_band_metrics,
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


def test_length_band_metrics_puts_edge_lengths_in_the_upper_band():
    bands = length_band_metrics(
        ["a", "a", "b", "b"], ["a", "b", "b", "b"], [0, 50, 50, 100],
        edges=(0, 50, 100), labels=LABELS3,
    )
    assert [b["band"] for b in bands] == ["[0, 50)", "[50, 100)", "[100, inf)"]
    assert [b["n"] for b in bands] == [1, 2, 1]
    # band 0: a->a. band 1: a->b, b->b (recalls 0 and 1). band 2: b->b.
    assert bands[0]["balanced_accuracy"] == pytest.approx(1.0)
    assert bands[1]["balanced_accuracy"] == pytest.approx(0.5)
    assert bands[1]["accuracy"] == pytest.approx(0.5)
    assert bands[1]["errors"] == 1
    assert bands[2]["balanced_accuracy"] == pytest.approx(1.0)
    assert bands[1]["mean_words"] == pytest.approx(50.0)
    assert bands[1]["median_words"] == pytest.approx(50.0)


def test_length_band_metrics_scores_only_classes_present_in_the_band():
    bands = length_band_metrics(
        ["a", "a", "c"], ["a", "b", "c"], [10, 20, 900],
        edges=(0, 100, 800), labels=LABELS3,
    )
    assert bands[0]["classes_present"] == 1
    assert bands[0]["balanced_accuracy"] == pytest.approx(0.5)
    assert bands[1]["n"] == 0
    assert bands[1]["classes_present"] == 0
    assert bands[1]["balanced_accuracy"] is None
    assert bands[1]["accuracy"] is None
    assert bands[1]["mean_words"] is None
    assert bands[1]["errors"] == 0
    assert bands[2]["balanced_accuracy"] == pytest.approx(1.0)


def test_length_band_metrics_splits_provider_errors_per_band():
    bands = length_band_metrics(
        ["gemini-3.6-flash", "gemini-3.7-flash", "gpt-5.5"],
        ["gemini-3.7-flash", "gpt-5.5", "gpt-5.5"],
        [10, 10, 900], edges=(0, 100, 800),
    )
    assert bands[0]["errors"] == 2
    assert bands[0]["within_provider_errors"] == 1
    assert bands[0]["cross_provider_errors"] == 1
    assert bands[2]["errors"] == 0


def test_length_band_metrics_single_band_reproduces_pooled_score():
    y_true = ["a", "b", "b", "c"]
    y_pred = ["a", "b", "c", "c"]
    lengths = [3, 900, 40, 12]
    bands = length_band_metrics(y_true, y_pred, lengths, edges=(0,), labels=LABELS3)
    assert len(bands) == 1
    assert bands[0]["n"] == 4
    assert bands[0]["balanced_accuracy"] == pytest.approx(
        balanced_accuracy(y_true, y_pred, LABELS3)
    )


def test_length_band_metrics_rejects_bad_edges_and_lengths():
    y = ["a", "b"]
    with pytest.raises(ValueError, match="start at 0"):
        length_band_metrics(y, y, [10, 20], edges=(50, 100), labels=LABELS3)
    with pytest.raises(ValueError, match="strictly increasing"):
        length_band_metrics(y, y, [10, 20], edges=(0, 100, 50), labels=LABELS3)
    with pytest.raises(ValueError, match="start at 0"):
        length_band_metrics(y, y, [10, 20], edges=(), labels=LABELS3)
    with pytest.raises(ValueError, match="non-negative"):
        length_band_metrics(y, y, [-1, 20], labels=LABELS3)
    with pytest.raises(ValueError, match="length shape mismatch"):
        length_band_metrics(y, y, [10], labels=LABELS3)
    with pytest.raises(ValueError, match="shape mismatch"):
        length_band_metrics(y, ["a"], [10, 20], labels=LABELS3)
