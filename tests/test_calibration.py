"""Unit tests for the k-robust damped Sinkhorn calibration."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import CalibrationError, calibrate_predict, sinkhorn_calibrate


def _skewed_scores(n=200, k=18, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.normal(size=(n, k))
    # Make class 0 dominant so balancing has something to correct.
    S[:, 0] += 3.0
    return S


def test_output_is_row_stochastic_and_shape_preserved():
    S = _skewed_scores()
    P = sinkhorn_calibrate(S, strength=0.6, target_mix=0.0)
    assert P.shape == S.shape
    assert np.allclose(P.sum(axis=1), 1.0)
    assert (P >= 0).all()


def test_strength_zero_is_plain_softmax_argmax():
    S = _skewed_scores()
    pred = calibrate_predict(S, strength=0.0)
    assert np.array_equal(pred, S.argmax(axis=1))


def test_empirical_target_preserves_predictions():
    # target_mix=1 targets the batch's own marginal, so balancing is a no-op net.
    S = _skewed_scores()
    pred = calibrate_predict(S, strength=1.0, target_mix=1.0, n_iter=100)
    base = S.argmax(axis=1)
    assert np.mean(pred == base) > 0.98


def test_uniform_target_balances_column_mass():
    S = _skewed_scores(n=400)
    P = sinkhorn_calibrate(S, strength=1.0, target_mix=0.0, n_iter=200)
    col = P.sum(axis=0)
    # Column sums approach n/18 = 22.2; the dominant class is pulled down hard.
    assert col[0] < 400 / 18 * 1.5
    assert col.min() > 400 / 18 * 0.5


def test_balancing_reduces_dominant_class_share():
    S = _skewed_scores(n=400)
    base = S.argmax(axis=1)
    pred = calibrate_predict(S, strength=0.6, target_mix=0.0)
    base_share = np.mean(base == 0)
    new_share = np.mean(pred == 0)
    assert new_share < base_share


def test_explicit_target_overrides_mix():
    S = _skewed_scores(n=300)
    target = np.zeros(18)
    target[5] = 1.0  # push all mass to class 5
    pred = calibrate_predict(S, strength=1.0, target=target, n_iter=200)
    assert np.mean(pred == 5) > 0.9


def test_non_finite_scores_raise():
    S = _skewed_scores()
    S[0, 0] = np.nan
    with pytest.raises(CalibrationError):
        sinkhorn_calibrate(S)


def test_invalid_parameters_raise():
    S = _skewed_scores()
    with pytest.raises(CalibrationError):
        sinkhorn_calibrate(S, strength=1.5)
    with pytest.raises(CalibrationError):
        sinkhorn_calibrate(S, target_mix=-0.1)
    with pytest.raises(CalibrationError):
        sinkhorn_calibrate(S, target=np.ones(5))


def test_predict_indices_in_range():
    S = _skewed_scores()
    pred = calibrate_predict(S, strength=0.3, target_mix=0.35)
    assert pred.dtype.kind in "iu"
    assert pred.min() >= 0 and pred.max() < 18