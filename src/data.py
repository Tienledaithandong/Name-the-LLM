"""Dataset loading and schema validation.

Column names come from the official Data page (``id``, ``prompt_id``, ``text``,
``model`` for train; ``id``, ``text`` for test) and are validated rather than
assumed: a mismatch raises instead of silently producing garbage features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DATA_DIR, LABEL_TO_INDEX, LABELS, N_CLASSES

TRAIN_COLUMNS: tuple[str, ...] = ("id", "prompt_id", "text", "model")
TEST_COLUMNS: tuple[str, ...] = ("id", "text")


class SchemaError(ValueError):
    """Raised when a dataframe does not match the official schema."""


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{what} is missing column(s) {missing}; found {list(df.columns)}"
        )


def _require_no_nulls(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    nulls = {c: int(df[c].isna().sum()) for c in columns if df[c].isna().any()}
    if nulls:
        raise SchemaError(f"{what} has null values in {nulls}")


def validate_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a training frame; returns a copy."""
    _require_columns(df, TRAIN_COLUMNS, "train frame")
    df = df.loc[:, list(TRAIN_COLUMNS)].copy()
    _require_no_nulls(df, TRAIN_COLUMNS, "train frame")
    df["text"] = df["text"].astype(str)
    df["id"] = df["id"].astype(str)
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["model"] = df["model"].astype(str)

    if (df["text"].str.strip() == "").any():
        raise SchemaError("train frame contains empty text rows")
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].head(5).tolist()
        raise SchemaError(f"train frame has duplicate ids, e.g. {dupes}")

    unknown = sorted(set(df["model"]) - set(LABELS))
    if unknown:
        raise SchemaError(f"unknown model label(s): {unknown}")
    if df["model"].nunique() < 2:
        raise SchemaError("train frame needs at least two distinct model labels")
    return df.reset_index(drop=True)


def validate_test_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a test frame; returns a copy preserving row order."""
    _require_columns(df, TEST_COLUMNS, "test frame")
    df = df.loc[:, list(TEST_COLUMNS)].copy()
    _require_no_nulls(df, TEST_COLUMNS, "test frame")
    df["text"] = df["text"].astype(str)
    df["id"] = df["id"].astype(str)
    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].head(5).tolist()
        raise SchemaError(f"test frame has duplicate ids, e.g. {dupes}")
    return df.reset_index(drop=True)


def find_competition_file(filename: str, data_dir: str | Path | None = None) -> Path:
    """Locate a competition file, checking the usual Kaggle download layouts."""
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    candidates = [
        base / filename,
        base / "name-that-llm" / filename,
        base / "competitions" / "name-that-llm" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"could not find {filename}; looked in:\n  {tried}\n"
        "Download the competition data from "
        "https://www.kaggle.com/competitions/name-that-llm/data and place it in data/, "
        "or run with --synthetic for a pipeline smoke test."
    )


def load_competition_data(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate ``train.csv`` and ``test.csv``."""
    train = pd.read_csv(find_competition_file("train.csv", data_dir))
    test = pd.read_csv(find_competition_file("test.csv", data_dir))
    return validate_train_frame(train), validate_test_frame(test)


def labels_to_indices(labels) -> np.ndarray:
    """Map label strings to indices in ``LABELS`` order."""
    values = np.asarray(list(labels), dtype=object)
    out = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        try:
            out[i] = LABEL_TO_INDEX[str(value)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise SchemaError(f"unknown label {value!r}") from exc
    return out


def indices_to_labels(indices) -> np.ndarray:
    return np.asarray([LABELS[int(i)] for i in indices], dtype=object)


def class_order() -> tuple[str, ...]:
    """The fixed class ordering used by every score matrix and model artifact."""
    assert len(LABELS) == N_CLASSES
    return LABELS


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Order-sensitive sha256 over the training rows, including text."""
    # Include order and text: positional splits and fitted artifacts depend on both.
    columns = ["id", "prompt_id", "model", "text"]
    payload = df.loc[:, columns].astype(str).to_json(orient="values", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def group_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("prompt_id", observed=True)["id"].count()


def class_support(df: pd.DataFrame) -> dict[str, int]:
    counts = df["model"].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}
