"""Frozen-encoder feature path (Section 6 / H3).

These tests use the deterministic ``hashing`` backend, which needs no download
and no network, so the suite stays hermetic.  The pretrained
``sentence-transformers`` backend is exercised only by the optional official
benchmark, never by the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.constants import LABELS, N_CLASSES
from src.data import labels_to_indices
from src.features_embedding import BACKENDS, EncoderError, FrozenEncoder
from src.features_style import FEATURE_NAMES
from src.features_tfidf import FeatureError, TextMatrixBuilder
from src.model import ModelBundle, ModelSpec, estimator_from_config


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.features_embedding.ARTIFACTS_DIR", tmp_path)


def _hashing(dim: int = 32) -> dict:
    return {"backend": "hashing", "dim": dim, "seed": 0}


def test_encoder_rejects_unknown_backend():
    with pytest.raises(EncoderError, match="unknown backend"):
        FrozenEncoder(backend="bogus")
    assert "hashing" in BACKENDS


def test_embedding_only_builder_shape_and_dtype(train_df):
    texts = train_df["text"].head(30).tolist()
    builder = TextMatrixBuilder(kinds=(), style=False, embedding=_hashing(32)).fit(texts)
    matrix = builder.transform(texts[:5])
    assert sparse.issparse(matrix) and matrix.dtype == np.float32
    assert matrix.shape == (5, 32)
    assert builder.n_output_features_ == 32
    assert np.isfinite(matrix.toarray()).all()


def test_embedding_plus_style_builder(train_df):
    texts = train_df["text"].head(30).tolist()
    builder = TextMatrixBuilder(
        kinds=(), style=True, embedding=_hashing(16)
    ).fit(texts)
    matrix = builder.transform(texts[:4])
    assert matrix.shape == (4, 16 + len(FEATURE_NAMES))
    assert builder.n_output_features_ == 16 + len(FEATURE_NAMES)


def test_embedding_only_is_a_valid_feature_source():
    TextMatrixBuilder(kinds=(), style=False, embedding=_hashing(8))
    with pytest.raises(FeatureError):
        TextMatrixBuilder(kinds=(), style=False, embedding=None)


def test_embedding_feature_names(train_df):
    texts = train_df["text"].head(10).tolist()
    builder = TextMatrixBuilder(kinds=(), embedding=_hashing(8)).fit(texts)
    names = builder.get_feature_names_out().tolist()
    assert names == [f"emb:{i}" for i in range(8)]


def test_hashing_projection_is_fitted_on_train(train_df):
    texts = train_df["text"].head(20).tolist()
    builder = TextMatrixBuilder(kinds=(), embedding=_hashing(16)).fit(texts)
    assert builder._encoder._projection is not None
    first = builder.transform(texts[:3]).toarray()
    again = builder.transform(texts[:3]).toarray()
    assert np.allclose(first, again)


def test_embedding_bundle_class_order_and_serialization(train_df, tmp_path):
    subset = train_df.groupby("model", observed=True).head(4).reset_index(drop=True)
    raw = {
        "name": "emb_tiny",
        "features": [],
        "embedding": _hashing(24),
        "classifier": "logreg",
        "C": 1.0,
    }
    bundle = estimator_from_config(raw, seed=0)
    bundle.fit(subset["text"].tolist(), labels_to_indices(subset["model"].tolist()))
    scores = bundle.decision_function(subset["text"].head(7).tolist())
    assert scores.shape == (7, N_CLASSES)
    assert bundle.class_order == LABELS
    assert set(bundle.predict(subset["text"].head(7).tolist())) <= set(LABELS)

    path = bundle.save(tmp_path / "emb.joblib")
    reloaded = ModelBundle.load(path)
    assert np.allclose(scores, reloaded.decision_function(subset["text"].head(7).tolist()))
    assert reloaded.class_order == LABELS


def test_spec_round_trips_embedding():
    spec = ModelSpec.from_dict(
        {"name": "e", "features": [], "embedding": _hashing(12), "classifier": "logreg"}
    )
    assert spec.embedding == _hashing(12)
    assert spec.to_dict()["embedding"] == _hashing(12)