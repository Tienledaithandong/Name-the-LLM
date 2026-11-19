from __future__ import annotations

import pandas as pd
import pytest

from src.constants import LABELS, N_CLASSES, PROVIDER_OF
from src.data import (
    SchemaError,
    dataset_fingerprint,
    indices_to_labels,
    labels_to_indices,
    validate_test_frame,
    validate_train_frame,
)


def test_official_label_set_is_complete_and_unique():
    assert len(LABELS) == 18 == N_CLASSES
    assert len(set(LABELS)) == 18
    assert set(PROVIDER_OF) == set(LABELS)
    assert "gpt-6-astra" in LABELS and "kimi-k3" in LABELS


def test_validate_train_frame_accepts_official_columns(train_df):
    out = validate_train_frame(train_df)
    assert list(out.columns) == ["id", "prompt_id", "text", "model"]
    assert out["model"].isin(LABELS).all()
    assert not out["id"].duplicated().any()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda df: df.drop(columns=["prompt_id"]), "missing column"),
        (lambda df: df.assign(model="not-a-model"), "unknown model label"),
        (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "duplicate ids"),
        (lambda df: df.assign(text=None), "null values"),
        (lambda df: df.assign(text="   "), "empty text"),
    ],
)
def test_validate_train_frame_rejects_bad_frames(train_df, mutation, message):
    with pytest.raises(SchemaError, match=message):
        validate_train_frame(mutation(train_df.copy()))


def test_validate_test_frame_requires_id_and_text(test_df):
    assert list(validate_test_frame(test_df).columns) == ["id", "text"]
    with pytest.raises(SchemaError):
        validate_test_frame(test_df.assign(id=None))
    with pytest.raises(SchemaError):
        validate_test_frame(test_df.drop(columns=["text"]))


def test_label_index_roundtrip(train_df):
    values = train_df["model"].head(50).tolist()
    indices = labels_to_indices(values)
    assert indices.dtype.kind == "i"
    assert list(indices_to_labels(indices)) == values
    assert LABELS[int(indices[0])] == values[0]


def test_fingerprint_is_stable_and_label_sensitive(train_df):
    first = dataset_fingerprint(train_df)
    shuffled = train_df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert dataset_fingerprint(shuffled) != first
    changed = train_df.copy()
    changed.loc[0, "model"] = "kimi-k3" if changed.loc[0, "model"] != "kimi-k3" else "glm-5.3"
    assert dataset_fingerprint(changed) != first


def test_competition_files_are_found_by_expected_layout(tmp_path):
    from src.data import find_competition_file

    nested = tmp_path / "name-that-llm"
    nested.mkdir()
    (nested / "train.csv").write_text("id,prompt_id,text,model\n", encoding="utf-8")
    assert find_competition_file("train.csv", tmp_path) == nested / "train.csv"
    with pytest.raises(FileNotFoundError, match="Download the competition data"):
        find_competition_file("test.csv", tmp_path / "missing")
