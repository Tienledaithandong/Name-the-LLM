from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LABELS
from src.submission import (
    SubmissionError,
    build_submission,
    validate_frame,
    validate_submission,
)


def test_build_and_reload_submission(tmp_path, test_df):
    ids = test_df["id"].tolist()
    predictions = [LABELS[i % len(LABELS)] for i in range(len(ids))]
    path = build_submission(ids, predictions, tmp_path / "submission.parquet", expected_ids=ids)
    frame = validate_submission(path, expected_ids=ids)
    assert list(frame.columns) == ["id", "model"]
    assert len(frame) == len(ids)
    assert frame["model"].isin(LABELS).all()


@pytest.mark.parametrize(
    "ids, predictions, match",
    [
        (["a", "a"], ["gpt-5.5", "gpt-5.5"], "repeats id"),
        (["a", "b"], ["gpt-5.5", "not-a-model"], "invalid label"),
        (["a", "b"], [None, "kimi-k3"], "null"),
        (["a"], ["gpt-5.5", "kimi-k3"], "predictions"),
    ],
)
def test_build_submission_rejects_invalid_input(tmp_path, ids, predictions, match):
    with pytest.raises(SubmissionError, match=match):
        build_submission(ids, predictions, tmp_path / "submission.parquet")


def test_row_count_and_order_are_checked(tmp_path):
    frame = pd.DataFrame({"id": ["b", "a"], "model": ["gpt-5.5", "kimi-k3"]})
    with pytest.raises(SubmissionError, match="do not match"):
        validate_frame(frame, expected_ids=["a", "b"])
    short = pd.DataFrame({"id": ["a"], "model": ["gpt-5.5"]})
    with pytest.raises(SubmissionError, match="rows, expected"):
        validate_frame(short, expected_ids=["a", "b"])


def test_extra_or_missing_columns_are_rejected():
    with pytest.raises(SubmissionError, match="missing column"):
        validate_frame(pd.DataFrame({"id": ["a"]}))
    with pytest.raises(SubmissionError, match="unexpected column"):
        validate_frame(pd.DataFrame({"id": ["a"], "model": ["kimi-k3"], "extra": [1]}))


def test_wrong_extension_is_rejected(tmp_path):
    with pytest.raises(SubmissionError, match="parquet"):
        build_submission(["a"], ["kimi-k3"], tmp_path / "submission.csv")


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(SubmissionError, match="missing"):
        validate_submission(tmp_path / "submission.parquet")


def test_parquet_roundtrip_preserves_dtype(tmp_path, test_df):
    ids = test_df["id"].tolist()
    preds = ["claude-opus-5"] * len(ids)
    path = build_submission(ids, preds, tmp_path / "submission.parquet")
    frame = validate_submission(path)
    assert frame["model"].astype(str).unique().tolist() == ["claude-opus-5"]
    assert np.array_equal(frame["id"].to_numpy(), np.asarray(ids))
