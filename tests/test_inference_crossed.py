"""Hermetic tests for the crossed-design inference path (no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import LABELS, N_CLASSES
from src.features_embedding import FrozenEncoder
from src.inference import (
    blend_score_matrix,
    fit_final,
    load_final,
    predict_dataframe,
    predict_dataframe_crossed,
)


@pytest.fixture()
def bundles(train_df, tiny_char_model, tmp_path):
    fit_final(
        [{"name": "majority", "baseline": "majority"}, tiny_char_model],
        train_df,
        tmp_path / "final",
        dataset_label="synthetic",
    )
    loaded, _ = load_final(tmp_path / "final")
    return loaded


def test_blend_score_matrix_shape_and_finite(bundles, test_df):
    texts = test_df["text"].astype(str).tolist()
    matrix = blend_score_matrix(bundles, texts, weights={"tiny_char": 1.0})
    assert matrix.shape == (len(test_df), N_CLASSES)
    assert np.isfinite(matrix).all()


def test_crossed_prediction_valid_labels(bundles, test_df):
    encoder = FrozenEncoder(backend="hashing", dim=64)
    pred = predict_dataframe_crossed(
        bundles, test_df, weights={"tiny_char": 1.0}, encoder=encoder, n_clusters=4
    )
    assert len(pred) == len(test_df)
    assert set(pred) <= set(LABELS)


def test_crossed_threshold_mode_runs(bundles, test_df):
    encoder = FrozenEncoder(backend="hashing", dim=64)
    pred = predict_dataframe_crossed(
        bundles, test_df.head(20), encoder=encoder, distance_threshold=0.5
    )
    assert len(pred) == 20
    assert set(pred) <= set(LABELS)


def test_crossed_defaults_to_k_estimate(bundles, test_df):
    encoder = FrozenEncoder(backend="hashing", dim=64)
    pred = predict_dataframe_crossed(
        bundles, test_df.head(36), encoder=encoder,
        distance_threshold=None, n_clusters=None, models_per_prompt=18,
    )
    assert len(pred) == 36
    assert set(pred) <= set(LABELS)


def test_crossed_requires_encoder(bundles, test_df):
    with pytest.raises(ValueError):
        predict_dataframe_crossed(bundles, test_df, encoder=None)


def test_crossed_reduces_to_assignment_not_argmax(bundles, test_df):
    # With a single cluster covering all rows, assignment enforces distinct
    # models across the whole batch (when rows <= classes), unlike argmax.
    encoder = FrozenEncoder(backend="hashing", dim=64)
    small = test_df.head(N_CLASSES)
    pred = predict_dataframe_crossed(bundles, small, encoder=encoder, n_clusters=1)
    assert len(np.unique(pred)) == len(small)  # all distinct under one group
    plain = predict_dataframe(bundles, small)
    assert len(pred) == len(plain)