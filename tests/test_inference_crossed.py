"""Hermetic tests for the crossed-design inference path (no downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import N_CLASSES
from src.inference import (
    blend_score_matrix,
    fit_final,
    load_final,
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


@pytest.mark.parametrize("encoder", [None, object()])
def test_crossed_prediction_is_withdrawn_before_using_inputs(encoder):
    with pytest.raises(ValueError, match="crossed inference is withdrawn"):
        predict_dataframe_crossed(None, None, weights={"missing": 1.0}, encoder=encoder)
