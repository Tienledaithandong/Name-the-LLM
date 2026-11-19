"""Regression tests for matched scoring and the paired prompt interval."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.run_score_improvement import (
    masked_arms,
    prompt_bootstrap,
    verified_control,
)
from experiments.run_modernbert_probe import pilot_passes
from src.constants import LABELS
from src.ensemble import normalize_scores


def test_mask_only_changes_the_auxiliary_view() -> None:
    rng = np.random.default_rng(4)
    raw_char = normalize_scores(rng.normal(size=(6, 18)))
    style = normalize_scores(rng.normal(size=(6, 18)))
    baseline = 0.5 * raw_char + 0.5 * style
    equivalent_raw_margins = 6 * raw_char + 17
    arms = masked_arms(baseline, raw_char, equivalent_raw_margins)
    np.testing.assert_allclose(arms["masked_blend"], arms["raw_char_reweighted"])


def test_prompt_bootstrap_keeps_correlated_rows_paired() -> None:
    # All labels within a prompt move together. The gain is +1 for one prompt
    # and -1 for the other, so resampling two whole prompts reaches both extremes.
    truth = np.array([0, 1, 0, 1])
    baseline = np.array([1, 0, 0, 1])
    candidate = np.array([0, 1, 1, 0])
    groups = np.array(["a", "a", "b", "b"])
    interval = prompt_bootstrap(truth, baseline, candidate, groups, draws=500)
    assert interval["n_prompts"] == 2
    assert (interval["low"], interval["high"]) == (-1.0, 1.0)
    none = prompt_bootstrap(truth, baseline, baseline, groups, draws=50)
    assert (none["low"], none["high"]) == (0.0, 0.0)


def test_control_rejects_agreeing_artifacts_with_wrong_fold_identity(tmp_path) -> None:
    out = tmp_path / "oof"
    out.mkdir()
    for name in ("char_tfidf_logreg", "style_only"):
        np.savez(
            out / f"{name}.npz", scores=np.zeros((4, 18)), y=np.arange(4),
            labels=np.array(LABELS), validation_fold_ids=[0, 0, 1, 1], ordered_dataset_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="does not match"):
        verified_control(tmp_path, "a" * 64, np.array([0, 1, 0, 1]), np.arange(4))


def test_encoder_pilot_requires_both_declared_folds_to_improve() -> None:
    passing = [
        {"fold": 0, "baseline": 0.35, "encoder": 0.20, "blend": 0.37},
        {"fold": 1, "baseline": 0.38, "encoder": 0.19, "blend": 0.40},
    ]
    assert pilot_passes(passing, [0, 1], 0.01)
    assert not pilot_passes(passing[:1], [0, 1], 0.01)
    assert not pilot_passes([passing[0], {**passing[1], "blend": 0.385}], [0, 1], 0.01)
    assert not pilot_passes([passing[0], {**passing[1], "encoder": 1 / 18}], [0, 1], 0.01)


def test_training_row_span_recovers_sparse_logistic_predictions() -> None:
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression

    from src.provider_specialist import _lift_head, _training_coordinates

    rng = np.random.default_rng(20)
    data = csr_matrix(rng.normal(size=(24, 100)) + rng.normal(size=(1, 100)))
    y = np.tile(np.arange(3), 8)
    expected = LogisticRegression(C=1.0, max_iter=2000, tol=1e-10).fit(data[:21], y[:21])
    train = data[:21]
    coords, dual_basis = _training_coordinates(train)
    reduced = LogisticRegression(C=1.0, max_iter=2000, tol=1e-10).fit(coords, y[:21])
    _lift_head(reduced, train, np.asarray(train.mean(axis=0)).ravel(), dual_basis)
    np.testing.assert_allclose(reduced.predict_proba(data[21:]), expected.predict_proba(data[21:]), atol=1e-5)
