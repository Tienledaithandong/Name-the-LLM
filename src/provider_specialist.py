"""Fit provider-specific version classifiers on one official training fold."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from src.constants import LABELS, N_CLASSES, RANDOM_STATE
from src.ensemble import SENTINEL
from src.features_tfidf import TextMatrixBuilder
from src.metrics import provider_of
from src.model import ModelError, ModelSpec, build_classifier


def _provider_columns() -> dict[str, list[int]]:
    columns: dict[str, list[int]] = {}
    for index, label in enumerate(LABELS):
        columns.setdefault(provider_of(label), []).append(index)
    return columns


def _training_coordinates(matrix: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal coordinates in the span of centered training rows.

    With an unpenalized intercept and L2 penalty, coefficients outside this
    span cannot improve training loss. The basis preserves the feature-space
    penalty and reduces the optimizer's largest dimension to the row count.
    """
    kernel = (matrix @ matrix.T).toarray()
    mean = kernel.mean(axis=0)
    kernel -= mean[None, :]
    kernel -= mean[:, None]
    kernel += mean.mean()
    eigenvalues, eigenvectors = eigh(kernel, overwrite_a=True, check_finite=False)
    tolerance = np.finfo(np.float64).eps * len(mean) * max(float(eigenvalues[-1]), 1.0)
    keep = eigenvalues > tolerance
    if not keep.any():
        intercept_only = np.zeros((matrix.shape[0], 1), dtype=np.float64)
        return intercept_only, intercept_only
    roots = np.sqrt(eigenvalues[keep])
    basis = eigenvectors[:, keep]
    return basis * roots, basis / roots


def _lift_head(head: LogisticRegression, matrix: csr_matrix, mean: np.ndarray,
               dual_basis: np.ndarray) -> None:
    """Express a fitted coordinate-space head in the original sparse features."""
    dual = dual_basis @ head.coef_.T
    coefficients = np.asarray(matrix.T @ dual).T - dual.sum(axis=0)[:, None] * mean
    head.intercept_ -= coefficients @ mean
    head.coef_ = coefficients
    head.n_features_in_ = matrix.shape[1]


class ProviderSpecialist:
    """Share one fold-local feature map across provider and flat logistic heads."""

    def __init__(
        self,
        alpha: float = 0.2,
        seed: int = RANDOM_STATE,
        max_features: int = 100_000,
    ) -> None:
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ModelError("provider specialist alpha must be in [0, 1]")
        if max_features < 1:
            raise ModelError("provider specialist max_features must be positive")
        self.alpha = alpha
        self.seed = seed
        self.max_features = max_features
        self.class_order = LABELS
        self.builder: TextMatrixBuilder | None = None
        self.flat_head: LogisticRegression | None = None
        self.provider_heads: dict[str, LogisticRegression] = {}

    def _new_head(self, name: str) -> LogisticRegression:
        return build_classifier(ModelSpec(name=name, classifier="logreg", C=1.0, seed=self.seed))

    def fit(self, texts: Sequence[str], y: ArrayLike) -> ProviderSpecialist:
        """Learn from the supplied training passages and their canonical class indices."""
        targets = np.asarray(y)
        if targets.ndim != 1 or targets.size != len(texts) or targets.dtype.kind not in "iu":
            raise ModelError("expected one integer class index per training passage")
        if not np.array_equal(np.unique(targets), np.arange(N_CLASSES)):
            raise ModelError("provider specialist training requires all 18 canonical class indices")

        builder = TextMatrixBuilder(kinds=("char",), style=True, params={"max_features": self.max_features})
        matrix = builder.fit_transform(texts).astype(np.float64)
        coordinates, dual_basis = _training_coordinates(matrix)
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        flat_head = self._new_head("flat_diagnostic").fit(coordinates, targets)
        _lift_head(flat_head, matrix, mean, dual_basis)
        provider_heads = {}
        for provider, columns in _provider_columns().items():
            if len(columns) < 2:
                continue
            provider_rows = np.isin(targets, columns)
            head = self._new_head(provider).fit(coordinates[provider_rows], targets[provider_rows])
            _lift_head(head, matrix, mean, dual_basis)
            provider_heads[provider] = head

        self.builder = builder
        self.flat_head = flat_head
        self.provider_heads = provider_heads
        return self

    def compare(
        self, texts: Sequence[str], baseline_score_matrix: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the hierarchy and matched flat blend in canonical class order.

        Positive alpha yields relative row weights, without a calibration claim.
        Alpha zero returns the original scores so the control keeps its exact
        ranking, including near ties that softmax could round into ties.
        """
        if self.builder is None or self.flat_head is None:
            raise ModelError("provider specialist must be fitted before scoring")
        if self.class_order != LABELS:
            raise ModelError("provider specialist class order does not match the official labels")
        scores = np.asarray(baseline_score_matrix, dtype=np.float64)
        if scores.shape != (len(texts), N_CLASSES) or len(texts) == 0:
            raise ModelError("expected nonempty baseline scores in passages by 18 labels order")
        if not np.isfinite(scores).all() or (scores < SENTINEL).any():
            raise ModelError("baseline scores must be finite with no absent classes")
        if self.alpha == 0.0:
            return scores.copy(), scores.copy()

        matrix = self.builder.transform(texts)
        baseline = softmax(scores, axis=1)
        hierarchy = baseline.copy()
        for head in self.provider_heads.values():
            columns = head.classes_
            previous = baseline[:, columns]
            # This is the original provider mass times its new conditional
            # weights. The product form also handles zero mass without division.
            hierarchy[:, columns] = (
                (1.0 - self.alpha) * previous
                + self.alpha * previous.sum(axis=1, keepdims=True) * head.predict_proba(matrix)
            )

        flat = (1.0 - self.alpha) * baseline
        flat[:, self.flat_head.classes_] += self.alpha * self.flat_head.predict_proba(matrix)
        return hierarchy, flat

    def save(self, path: str | Path) -> Path:
        """Save a fitted specialist separately from its baseline blend."""
        if self.builder is None or self.flat_head is None:
            raise ModelError("cannot save an unfitted provider specialist")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProviderSpecialist:
        """Load a trusted fitted artifact and require the official class order."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise ModelError("artifact is not a fitted provider specialist")
        if model.class_order != LABELS:
            raise ModelError("provider specialist class order does not match the official labels")
        if model.builder is None or model.flat_head is None:
            raise ModelError("artifact is not a fitted provider specialist")
        return model
