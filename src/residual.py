"""Self-neutralized style residual (approach A, spec 2026-11-19, ADR D058).

The hidden test supplies passages from prompts unseen in training, so any
feature component that encodes *which prompt* a passage answers is unavailable
at prediction time.  The Prompt-Neutralized Embedding of "Every Model Leaves a
Trace" removes each prompt's mean representation, but needs every candidate
model's response to the same prompt; the hidden test is believed to be k=2, so
that transform is not deployable.

:class:`PromptMeanNeutralizer` is the deployable substitute: a regressor
``g`` that predicts a row's *own* prompt mean from that row alone, subtracted
with a shrinkage factor,

    r_i = e_i - shrinkage * g(e_i).

``shrinkage=0`` returns the input untouched, so the candidate can never be
forced below the model that uses no neutralizer.  ``g`` is a function of one
passage, which is what makes it usable at any k including k=1.

This is a strict generalisation of ``NuisanceSuppressor`` (D010), which removes
the top-k *global* directions of between-prompt variation: that is
``g(e) = P(e - mean)`` for one fixed projection ``P``, a special case here.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


class ResidualError(ValueError):
    """Raised for an unusable neutralizer configuration or input."""


class PromptMeanNeutralizer:
    """Predict a row's prompt mean from the row and subtract it.

    Parameters
    ----------
    alpha : ridge penalty on standardised features.  ``1.0`` is the default
        because every space we neutralise has at least as many rows as columns
        but is strongly collinear.
    shrinkage : ``lambda`` in ``e - lambda * g(e)``, in ``[0, 1]``.  ``0`` is an
        exact identity and does not fit a regressor at transform time.
    min_group_size : prompts with fewer training rows contribute no target; they
        are still transformed, because ``g`` does not look at the group.
    n_inner_splits : prompt-grouped inner folds for the ``oof_r2_`` diagnostic.
        Grouped, so a row's own prompt never helps predict its prompt mean.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        shrinkage: float = 1.0,
        min_group_size: int = 2,
        n_inner_splits: int = 3,
    ):
        alpha = float(alpha)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ResidualError("alpha must be positive and finite")
        shrinkage = float(shrinkage)
        if not 0.0 <= shrinkage <= 1.0:
            raise ResidualError("shrinkage must be in [0, 1]")
        if int(min_group_size) < 1:
            raise ResidualError("min_group_size must be >= 1")
        if int(n_inner_splits) < 2:
            raise ResidualError("n_inner_splits must be >= 2")
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.min_group_size = int(min_group_size)
        self.n_inner_splits = int(n_inner_splits)

        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.regressor_: Ridge | None = None
        self.n_groups_: int = 0
        self.oof_r2_: float = 0.0
        self.predicted_norm_ratio_: float = 0.0
        self._fitted = False

    # -- fitting -----------------------------------------------------------
    def fit(self, X, groups) -> PromptMeanNeutralizer:
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups).astype(str)
        if X.ndim != 2:
            raise ResidualError("X must be 2-D")
        if len(groups) != X.shape[0]:
            raise ResidualError("groups must have one entry per row")

        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        Z = (X - self.mean_) / self.scale_

        uniques, inverse, counts = np.unique(
            groups, return_inverse=True, return_counts=True
        )
        keep = np.flatnonzero(counts >= self.min_group_size)
        self.n_groups_ = int(keep.size)
        self._fitted = True

        if keep.size == 0:
            # No prompt has enough rows to define a mean target: fall back to
            # the identity (a documented no-op, like D010's fallback).
            self.regressor_ = None
            self.oof_r2_ = 0.0
            self.predicted_norm_ratio_ = 0.0
            return self

        centroids = np.zeros((uniques.size, X.shape[1]), dtype=np.float64)
        for index in keep:
            centroids[index] = Z[inverse == index].mean(axis=0)

        row_mask = np.isin(inverse, keep)
        Z_kept = Z[row_mask]
        target = centroids[inverse[row_mask]]
        kept_groups = inverse[row_mask]

        self.regressor_ = Ridge(alpha=self.alpha).fit(Z_kept, target)
        self.oof_r2_ = self._grouped_r2(Z_kept, target, kept_groups)
        self.predicted_norm_ratio_ = self._norm_ratio(Z_kept)
        return self

    def _grouped_r2(self, Z: np.ndarray, target: np.ndarray, groups: np.ndarray) -> float:
        """Out-of-fold ``R^2`` of predicting a row's prompt mean, grouped.

        One global ``R^2`` over the whole target matrix per fold, referenced to
        the training side's mean target.  Per-output averaging is unusable here:
        a target column that is constant within the held-out prompts has zero
        variance and drives its own ``R^2`` to minus infinity, which says nothing
        about the neutralizer.

        The value is unbounded below and can be negative — with a representation
        whose prompt component never varies inside a training prompt, an unseen
        prompt's mean is not learnable at all, and the diagnostic must say so.
        """
        n_groups = int(np.unique(groups).size)
        n_splits = min(self.n_inner_splits, n_groups)
        if n_splits < 2:
            return 0.0
        scores: list[float] = []
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, valid_idx in splitter.split(Z, target, groups):
            if len(train_idx) == 0 or len(valid_idx) == 0:
                continue
            model = Ridge(alpha=self.alpha).fit(Z[train_idx], target[train_idx])
            pred = model.predict(Z[valid_idx])
            reference = target[train_idx].mean(axis=0)
            residual = float(((target[valid_idx] - pred) ** 2).sum())
            total = float(((target[valid_idx] - reference) ** 2).sum())
            if total > 1e-12:
                scores.append(1.0 - residual / total)
        return float(np.mean(scores)) if scores else 0.0

    def _norm_ratio(self, Z: np.ndarray) -> float:
        """How large the removed component is, relative to the centred features.

        A ratio near zero means the neutralizer predicted almost nothing, i.e.
        the premise (a prompt component recoverable from one passage) failed.
        Both sides are centred and in original units — the global mean is not a
        prediction and must not be counted as one.
        """
        if self.regressor_ is None:
            return 0.0
        # Regressor outputs are in standardised units; undo the scaling only.
        predicted_centred = self.regressor_.predict(Z) * self.scale_
        removed = float(np.linalg.norm(self.shrinkage * predicted_centred, axis=1).mean())
        centred = float(np.linalg.norm(Z * self.scale_, axis=1).mean())
        if centred < 1e-12:
            return 0.0
        return removed / centred

    # -- transformation ----------------------------------------------------
    def transform(self, X) -> np.ndarray:
        if not self._fitted or self.mean_ is None or self.scale_ is None:
            raise ResidualError("PromptMeanNeutralizer.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ResidualError("X must be 2-D")
        if X.shape[1] != self.mean_.size:
            raise ResidualError(
                f"X has {X.shape[1]} columns but the neutralizer was fitted on "
                f"{self.mean_.size}"
            )
        if self.shrinkage == 0.0 or self.regressor_ is None:
            return X.astype(np.float32)
        Z = (X - self.mean_) / self.scale_
        predicted = self.regressor_.predict(Z) * self.scale_ + self.mean_
        return (X - self.shrinkage * predicted).astype(np.float32)

    def fit_transform(self, X, groups) -> np.ndarray:
        return self.fit(X, groups).transform(X)
