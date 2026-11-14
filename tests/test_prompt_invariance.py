from __future__ import annotations

import numpy as np
import pytest

from src.prompt_invariance import (
    NuisanceSuppressor,
    SuppressorError,
    centered_features,
    prompt_centering_diagnostic,
)


def _grouped_matrix(seed=0, n_groups=6, per_group=8, dim=6):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(scale=3.0, size=(n_groups, dim))
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(scale=1.0, size=(groups.size, dim)) + offsets[groups]
    return X, groups


def _between_group_variance(X, groups):
    X = np.asarray(X, dtype=np.float64)
    total = 0.0
    overall = X.mean(axis=0)
    for group in np.unique(groups):
        block = X[groups == group]
        total += block.shape[0] * ((block.mean(axis=0) - overall) ** 2).sum()
    return float(total)


def test_suppressor_reduces_between_prompt_variance():
    X, groups = _grouped_matrix()
    before = _between_group_variance(X, groups)
    suppressor = NuisanceSuppressor(n_components=2, min_group_size=4).fit(X, groups)
    after = _between_group_variance(suppressor.transform(X), groups)
    assert after < 0.5 * before
    assert suppressor.n_groups_ == 6


def test_suppressor_is_deterministic_and_stateless_on_transform():
    X, groups = _grouped_matrix(seed=1)
    first = NuisanceSuppressor(n_components=2).fit(X, groups).transform(X)
    second = NuisanceSuppressor(n_components=2).fit(X, groups).transform(X)
    assert np.allclose(first, second)
    original = X.copy()
    NuisanceSuppressor(n_components=2).fit(X, groups).transform(X)
    assert np.array_equal(X, original)


def test_suppressor_identity_fallback_without_repeated_prompts():
    X = np.random.default_rng(0).normal(size=(5, 3))
    suppressor = NuisanceSuppressor(n_components=2, min_group_size=4).fit(X, np.arange(5))
    assert suppressor.components_.shape[1] == 0
    out = suppressor.transform(X)
    assert out.shape == X.shape and np.isfinite(out).all()


def test_suppressor_requires_fit_and_consistent_shapes():
    with pytest.raises(SuppressorError, match="before fit"):
        NuisanceSuppressor().transform(np.zeros((3, 2)))
    X, groups = _grouped_matrix()
    with pytest.raises(SuppressorError, match="one entry per row"):
        NuisanceSuppressor().fit(X, groups[:-1])
    with pytest.raises(SuppressorError, match="n_components"):
        NuisanceSuppressor(n_components=0)


def test_prompt_centering_diagnostic_and_transform():
    X, groups = _grouped_matrix()
    stats = prompt_centering_diagnostic(X, groups)
    assert 0.0 <= stats["between_prompt_share"] <= 1.0
    assert stats["n_prompts"] == 6

    centered = centered_features(X, groups)
    # Every prompt's centroid is now (numerically) zero.
    for group in np.unique(groups):
        assert np.allclose(centered[groups == group].mean(axis=0), 0.0, atol=1e-5)


def test_centering_share_is_low_for_pure_noise():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    stats = prompt_centering_diagnostic(X, np.repeat(np.arange(12), 10))
    assert stats["between_prompt_share"] < 0.35
