"""Prompt-invariance utilities.

Motivation (H2): hidden test passages answer prompts unseen in training, so any
signal that is really "which prompt was this" is a liability.  We therefore
provide

* ``NuisanceSuppressor`` — a *deployable* projection fitted only on training
  prompts that removes the directions in feature space along which prompt
  centroids vary.  It needs no test-time prompt labels.
* ``prompt_centering_diagnostic`` — the textbook ``z_i - mean(z | prompt_i)``
  transform.  It is **diagnostic only**: it needs prompt ids at prediction time,
  which the hidden test set does not provide, so its score is an upper bound,
  not a candidate submission.

The suppressor is applied in the dense (SVD) space for tractability.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD


class SuppressorError(ValueError):
    """Raised when a suppressor cannot be fitted."""


class NuisanceSuppressor:
    """Project out the top directions of between-prompt variation."""

    def __init__(self, n_components: int = 4, min_group_size: int = 4):
        if n_components < 1:
            raise SuppressorError("n_components must be >= 1")
        self.n_components = int(n_components)
        self.min_group_size = int(min_group_size)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.n_groups_: int = 0

    def fit(self, X, groups) -> "NuisanceSuppressor":
        X = np.asarray(X, dtype=np.float64)
        groups = np.asarray(groups)
        if X.ndim != 2:
            raise SuppressorError("X must be 2-D")
        if len(groups) != X.shape[0]:
            raise SuppressorError("groups must have one entry per row")

        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale_ = scale
        Z = (X - self.mean_) / self.scale_

        uniques, inverse, counts = np.unique(
            groups.astype(str), return_inverse=True, return_counts=True
        )
        keep = np.flatnonzero(counts >= self.min_group_size)
        if keep.size < 2:
            # Not enough repeated prompts: suppression would be undefined, so
            # fall back to the identity projection (a documented no-op).
            self.components_ = np.zeros((X.shape[1], 0))
            self.n_groups_ = int(keep.size)
            return self

        centroids = np.zeros((keep.size, X.shape[1]), dtype=np.float64)
        for row, group_index in enumerate(keep):
            centroids[row] = Z[inverse == group_index].mean(axis=0)

        k = int(min(self.n_components, centroids.shape[0] - 1, centroids.shape[1]))
        svd = TruncatedSVD(n_components=max(1, k), random_state=0)
        svd.fit(centroids)
        self.components_ = np.asarray(svd.components_.T, dtype=np.float64)  # (d, k)
        self.n_groups_ = int(keep.size)
        return self

    def transform(self, X) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise SuppressorError("NuisanceSuppressor.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.scale_
        if self.components_.shape[1] == 0:
            return Z.astype(np.float32)
        projection = self.components_ @ self.components_.T
        residual = Z - Z @ projection
        return residual.astype(np.float32)

    def fit_transform(self, X, groups) -> np.ndarray:
        return self.fit(X, groups).transform(X)


def prompt_centering_diagnostic(X, groups) -> dict:
    """Share of feature variance explained by prompt identity.

    A high value means features encode *which prompt* was answered, which is
    unavailable at hidden-test time.  Reported as a diagnostic, never as a
    model input for submission.
    """
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    if X.ndim != 2 or len(groups) != X.shape[0]:
        raise SuppressorError("X must be 2-D with one group per row")
    overall = X - X.mean(axis=0)
    total = float((overall**2).sum())
    if total <= 0:
        return {"between_prompt_share": 0.0, "n_prompts": int(np.unique(groups).size)}

    between = 0.0
    for group in np.unique(groups):
        block = X[groups == group]
        between += float(block.shape[0] * ((block.mean(axis=0) - X.mean(axis=0)) ** 2).sum())
    return {
        "between_prompt_share": float(between / total),
        "n_prompts": int(np.unique(groups).size),
    }


def centered_features(X, groups) -> np.ndarray:
    """Diagnostic-only ``z_i - mean(z | prompt_i)`` transform."""
    X = np.asarray(X, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    out = X.copy()
    for group in np.unique(groups):
        mask = groups == group
        out[mask] -= X[mask].mean(axis=0)
    return out.astype(np.float32)
