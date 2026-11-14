"""Reference baselines: majority class and stratified random.

They implement the same ``fit`` / ``decision_function`` / ``predict`` surface as
:class:`src.model.ModelBundle` so the CV loop is identical for all models.
Both are sanity checks, not candidates.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.constants import LABELS, N_CLASSES, RANDOM_STATE


class _BaselineBase:
    kind = "baseline"

    def __init__(self, name: str, seed: int = RANDOM_STATE):
        self.name = name
        self.seed = int(seed)
        self.class_order: tuple[str, ...] = LABELS
        self.counts_: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def fit(self, texts, y_idx, groups=None):  # noqa: ARG002 - shared signature
        y_idx = np.asarray(y_idx, dtype=np.int64)
        self.counts_ = np.bincount(y_idx, minlength=N_CLASSES).astype(np.float64)
        self._rng = np.random.default_rng(self.seed)
        return self

    def decision_function(self, texts) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, texts) -> np.ndarray:
        indices = self.decision_function(texts).argmax(axis=1)
        return np.asarray([self.class_order[i] for i in indices], dtype=object)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path):
        """Load a saved baseline, requiring the shared estimator interface."""
        bundle = joblib.load(Path(path))
        if not isinstance(bundle, _BaselineBase):
            raise TypeError(f"{path} is not a saved baseline estimator")
        return bundle


class MajorityClassifier(_BaselineBase):
    def __init__(self, seed: int = RANDOM_STATE):
        super().__init__("majority", seed)

    def decision_function(self, texts) -> np.ndarray:
        if self.counts_ is None:
            raise RuntimeError("fit first")
        out = np.zeros((len(list(texts)), N_CLASSES), dtype=np.float32)
        base = np.log1p(self.counts_)
        return out + base[None, :]


class StratifiedRandomClassifier(_BaselineBase):
    def __init__(self, seed: int = RANDOM_STATE):
        super().__init__("stratified_random", seed)

    def decision_function(self, texts) -> np.ndarray:
        if self.counts_ is None or self._rng is None:
            raise RuntimeError("fit first")
        n = len(list(texts))
        weights = self.counts_ / max(self.counts_.sum(), 1.0)
        out = np.empty((n, N_CLASSES), dtype=np.float32)
        for row in range(n):
            pick = int(self._rng.choice(N_CLASSES, p=weights))
            out[row] = -1.0
            out[row, pick] = 1.0
        return out
