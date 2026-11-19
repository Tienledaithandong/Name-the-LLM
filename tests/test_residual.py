from __future__ import annotations

import numpy as np
import pytest

from src.residual import PromptMeanNeutralizer, ResidualError


def _grouped(seed=0, n_groups=6, per_group=6, dim=5, between=4.0):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(scale=between, size=(n_groups, dim))
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(scale=1.0, size=(groups.size, dim)) + offsets[groups]
    return X, groups


def test_shrinkage_zero_is_exact_identity():
    X, groups = _grouped()
    out = PromptMeanNeutralizer(shrinkage=0.0).fit(X, groups).transform(X)
    assert np.array_equal(out, X.astype(np.float32))


def _learnable(seed=1, n_groups=8, per_group=6, dim=5, between=4.0):
    """Continuous prompt offsets: a single row does reveal its prompt's mean."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(scale=between, size=(n_groups, dim))
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(size=(groups.size, dim)) + offsets[groups]
    return X, groups


def test_removes_most_of_the_prompt_component_when_it_is_learnable():
    X, groups = _learnable()
    neutralizer = PromptMeanNeutralizer(shrinkage=1.0).fit(X, groups)
    out = neutralizer.transform(X)

    def between(matrix):
        means = np.stack([matrix[groups == g].mean(axis=0) for g in np.unique(groups)])
        return float(np.linalg.norm(means, axis=1).mean())

    assert between(out) < 0.5 * between(X)
    assert neutralizer.oof_r2_ > 0.0


def test_diagnostic_is_finite_and_reports_collapse():
    X, groups = _grouped()
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    assert np.isfinite(neutralizer.oof_r2_)
    assert 0.0 <= neutralizer.predicted_norm_ratio_ <= 1.0


def test_unlearnable_prompt_component_is_reported_as_negative_r2():
    # One-hot prompts: the column identifying a prompt never varies inside a
    # training prompt, so an unseen prompt's mean cannot be predicted.  The
    # diagnostic must expose that rather than reporting a comfortable zero.
    groups = np.repeat(np.arange(8), 5)
    X = np.eye(8)[groups] + 0.01 * np.random.default_rng(3).normal(size=(40, 8))
    neutralizer = PromptMeanNeutralizer(alpha=1e-6).fit(X, groups)
    assert neutralizer.oof_r2_ < 0.0


def test_unseen_group_rows_transform_without_nan():
    X, groups = _grouped()
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    out = neutralizer.transform(np.zeros((3, X.shape[1])) + 0.5)
    assert out.shape == (3, X.shape[1]) and np.isfinite(out).all()


def test_single_row_group_falls_back_to_identity_targets():
    X, groups = _grouped(n_groups=4, per_group=1)
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    assert neutralizer.n_groups_ == 0
    assert np.isfinite(neutralizer.transform(X)).all()


def test_suppressed_component_is_removed_only_within_the_fitted_span():
    # A held-out row must be transformed by the fitted function, not by its
    # own (unknown) group mean: two identical rows always map identically.
    X, groups = _grouped()
    neutralizer = PromptMeanNeutralizer(shrinkage=1.0).fit(X, groups)
    row = np.tile(X[0], (2, 1))
    out = neutralizer.transform(row)
    assert np.allclose(out[0], out[1])


def test_transform_before_fit_and_bad_shapes_raise():
    with pytest.raises(ResidualError, match="before fit"):
        PromptMeanNeutralizer().transform(np.zeros((2, 3)))
    X, groups = _grouped()
    with pytest.raises(ResidualError, match="one entry per row"):
        PromptMeanNeutralizer().fit(X, groups[:-1])
    with pytest.raises(ResidualError, match="shrinkage"):
        PromptMeanNeutralizer(shrinkage=1.5)
