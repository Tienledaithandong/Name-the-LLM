"""Sparse text feature construction.

Everything that learns from data (TF-IDF vocabularies, IDF weights, SVD bases,
nuisance projections) is fitted inside :meth:`TextMatrixBuilder.fit`, which is
only ever called on a training fold.  ``transform`` is pure with respect to the
fitted state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.features_style import StyleFeatureExtractor
from src.prompt_invariance import NuisanceSuppressor

TFIDF_KINDS = ("char", "word", "char_wb")

DEFAULT_PARAMS: dict[str, Any] = {
    "max_features": 300_000,
    "min_df": 2,
    "sublinear_tf": True,
    "char_ngram_range": (2, 5),
    "word_ngram_range": (1, 2),
}


class FeatureError(ValueError):
    """Raised for an unusable feature specification."""


def _make_vectorizer(kind: str, params: dict[str, Any]) -> TfidfVectorizer:
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(params.get("char_ngram_range", (2, 5))),
            max_features=params.get("max_features", 300_000),
            min_df=params.get("min_df", 2),
            sublinear_tf=params.get("sublinear_tf", True),
            lowercase=False,
            dtype=np.float32,
        )
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(params.get("word_ngram_range", (1, 2))),
            max_features=params.get("max_features", 300_000),
            min_df=params.get("min_df", 2),
            sublinear_tf=params.get("sublinear_tf", True),
            token_pattern=r"(?u)\b[\w'-]{2,}\b",
            dtype=np.float32,
        )
    if kind == "char_wb":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(params.get("char_ngram_range", (2, 5))),
            max_features=params.get("max_features", 300_000),
            min_df=params.get("min_df", 2),
            sublinear_tf=params.get("sublinear_tf", True),
            lowercase=False,
            dtype=np.float32,
        )
    raise FeatureError(f"unknown tf-idf kind {kind!r}; choose from {TFIDF_KINDS}")


class TextMatrixBuilder:
    """Compose sparse TF-IDF blocks with optional style and dense reductions."""

    def __init__(
        self,
        kinds: Sequence[str] = ("char",),
        style: bool = False,
        params: dict[str, Any] | None = None,
        svd: int | None = None,
        nuisance_k: int = 0,
        nuisance_min_group: int = 4,
    ):
        self.kinds = tuple(kinds)
        self.style = bool(style)
        if not self.kinds and not self.style:
            raise FeatureError("builder needs at least one tf-idf kind or style features")
        for kind in self.kinds:
            if kind not in TFIDF_KINDS:
                raise FeatureError(f"unknown tf-idf kind {kind!r}")
        if nuisance_k and not svd:
            raise FeatureError("nuisance suppression requires an SVD reduction (dense space)")
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.svd = svd
        self.nuisance_k = int(nuisance_k)
        self.nuisance_min_group = int(nuisance_min_group)
        self._vectorizers: list[TfidfVectorizer] = []
        self._style_extractor: StyleFeatureExtractor | None = None
        self._style_scaler: StandardScaler | None = None
        self._svd: TruncatedSVD | None = None
        self._suppressor: NuisanceSuppressor | None = None
        self._fitted = False

    # -- internals ---------------------------------------------------------
    def _style_block(self, texts) -> sparse.csr_matrix:
        """Style features, standardised with statistics from the training fold.

        Length-style columns span several orders of magnitude (0-250 words vs
        0-20 punctuation hits per 100 words), which makes unregularised-ish
        linear classifiers converge poorly.  ``with_mean=False`` keeps the
        block sparse.
        """
        block = self._style_extractor.transform(texts)
        if self._style_scaler is not None:
            block = self._style_scaler.transform(block)
        return sparse.csr_matrix(block, dtype=np.float32)

    def _raw(self, texts) -> sparse.csr_matrix:
        blocks = [vec.transform(texts) for vec in self._vectorizers]
        if self._style_extractor is not None:
            blocks.append(self._style_block(texts))
        return sparse.hstack(blocks, format="csr").astype(np.float32)

    def _reduce(self, matrix: sparse.csr_matrix, groups=None, fit: bool = False):
        if self._svd is not None:
            matrix = (
                self._svd.fit_transform(matrix) if fit else self._svd.transform(matrix)
            )
            matrix = np.asarray(matrix, dtype=np.float32)
        else:
            matrix = matrix
        if self.nuisance_k:
            dense = np.asarray(matrix, dtype=np.float32)
            if fit:
                self._suppressor = NuisanceSuppressor(
                    n_components=self.nuisance_k, min_group_size=self.nuisance_min_group
                ).fit(dense, groups)
            matrix = self._suppressor.transform(dense)
        return matrix

    # -- public API --------------------------------------------------------
    def fit(self, texts, groups=None) -> "TextMatrixBuilder":
        texts = list(texts)
        self._vectorizers = [_make_vectorizer(k, self.params) for k in self.kinds]
        for vec in self._vectorizers:
            vec.fit(texts)
        self._style_extractor = StyleFeatureExtractor() if self.style else None
        self._style_scaler = (
            StandardScaler(with_mean=False).fit(self._style_extractor.transform(texts))
            if self._style_extractor is not None
            else None
        )
        self._svd = (
            TruncatedSVD(
                n_components=int(self.svd), random_state=0, algorithm="randomized"
            )
            if self.svd
            else None
        )
        raw = self._raw(texts)
        if self._svd is not None:
            k = min(self.svd, max(2, raw.shape[1] - 1))
            if k != self.svd:
                self._svd.n_components = k
        self._reduce(raw, groups=groups, fit=True)
        self._fitted = True
        return self

    def transform(self, texts):
        if not self._fitted:
            raise FeatureError("TextMatrixBuilder.transform called before fit")
        return self._reduce(self._raw(list(texts)))

    def fit_transform(self, texts, groups=None):
        return self.fit(texts, groups=groups).transform(texts)

    def get_feature_names_out(self) -> np.ndarray:
        names: list[str] = []
        for kind, vec in zip(self.kinds, self._vectorizers, strict=True):
            names.extend(f"{kind}:{n}" for n in vec.get_feature_names_out())
        if self._style_extractor is not None:
            names.extend(f"style:{n}" for n in self._style_extractor.get_feature_names_out())
        if self._svd is not None:
            return np.asarray([f"svd_{i}" for i in range(self._svd.n_components)], dtype=object)
        return np.asarray(names, dtype=object)

    @property
    def n_output_features_(self) -> int:
        if self._svd is not None:
            return int(self._svd.n_components)
        return int(self.get_feature_names_out().size)


def density(matrix) -> float:
    if sparse.issparse(matrix):
        return float(matrix.nnz) / float(np.prod(matrix.shape)) if matrix.size else 0.0
    arr = np.asarray(matrix)
    return float(np.count_nonzero(arr)) / float(arr.size) if arr.size else 0.0
