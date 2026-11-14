"""Optional frozen-encoder features.

Two backends:

* ``hashing`` — deterministic char n-gram hashing + random projection.  No
  downloads, no network, CPU-only; always available and used in tests.
* ``sentence-transformers`` — a small public encoder (e.g. MiniLM).  Requires
  the optional dependency and a locally available checkpoint, because Kaggle
  submissions run with internet disabled: the weights must be attached as a
  dataset or baked into the notebook image.

Embeddings are cached on disk (float16) so repeated folds reuse them.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.random_projection import SparseRandomProjection

from src.constants import ARTIFACTS_DIR

BACKENDS = ("hashing", "sentence-transformers")


class EncoderError(RuntimeError):
    """Raised when the requested encoder is unavailable."""


class FrozenEncoder:
    """Lightweight encoder with optional transformers backend."""

    def __init__(
        self,
        backend: str = "hashing",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 256,
        seed: int = 0,
    ):
        if backend not in BACKENDS:
            raise EncoderError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.backend = backend
        self.model_name = model_name
        self.dim = int(dim)
        self.seed = int(seed)
        self._model = None
        self._projection: SparseRandomProjection | None = None

    # -- backends ----------------------------------------------------------
    def _load_transformers(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise EncoderError(
                "sentence-transformers is not installed. Install it and attach a local "
                "checkpoint, or use backend='hashing'. Kaggle inference cannot download "
                "weights (internet disabled)."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - network/weights missing
            raise EncoderError(
                f"could not load encoder {self.model_name!r}: {exc}. Attach the weights "
                "as a Kaggle dataset and pass a local path."
            ) from exc
        return self._model

    def _encode_hashing(self, texts) -> np.ndarray:
        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=2**18,
            alternate_sign=True,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        sparse = vectorizer.transform(list(texts))
        if self._projection is None:
            self._projection = SparseRandomProjection(
                n_components=min(self.dim, sparse.shape[1]),
                random_state=self.seed,
                dense_output=True,
            )
            self._projection.fit(sparse)
        return np.asarray(self._projection.transform(sparse), dtype=np.float32)

    def encode(self, texts) -> np.ndarray:
        texts = [str(t) for t in texts]
        if self.backend == "hashing":
            return self._encode_hashing(texts)
        model = self._load_transformers()
        return np.asarray(
            model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    # -- caching -----------------------------------------------------------
    def _cache_path(self, texts) -> Path:
        digest = hashlib.sha256(
            ("\x00".join(texts) + f"|{self.backend}|{self.model_name}|{self.dim}").encode()
        ).hexdigest()[:16]
        return ARTIFACTS_DIR / "embeddings" / f"{self.backend}_{digest}.pkl"

    def encode_cached(self, texts) -> np.ndarray:
        path = self._cache_path(list(texts))
        if path.is_file():
            with path.open("rb") as fh:
                return np.asarray(pickle.load(fh), dtype=np.float32)
        matrix = self.encode(texts).astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(matrix, fh)
        return np.asarray(matrix, dtype=np.float32)
