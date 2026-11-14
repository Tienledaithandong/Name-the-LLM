"""Submission construction and validation against the official schema.

Rejections listed on the official Evaluation page: missing ``id`` or ``model``
column, null values, repeated ids, or a label outside the 18 candidates.  All
four are checked here, plus row count and row order against ``test.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import LABELS, SUBMISSION_FILENAME


class SubmissionError(ValueError):
    """Raised when a submission would be rejected by Kaggle."""


def validate_frame(frame: pd.DataFrame, expected_ids=None) -> pd.DataFrame:
    missing = [c for c in ("id", "model") if c not in frame.columns]
    if missing:
        raise SubmissionError(f"submission is missing column(s) {missing}")
    extra = [c for c in frame.columns if c not in ("id", "model")]
    if extra:
        raise SubmissionError(f"submission has unexpected column(s) {extra}")

    if frame["id"].isna().any() or frame["model"].isna().any():
        raise SubmissionError("submission contains null values")
    if frame["id"].duplicated().any():
        dupes = frame.loc[frame["id"].duplicated(), "id"].head(5).tolist()
        raise SubmissionError(f"submission repeats id(s), e.g. {dupes}")

    labels = set(frame["model"].astype(str))
    unknown = sorted(labels - set(LABELS))
    if unknown:
        raise SubmissionError(f"submission has invalid label(s): {unknown}")

    if expected_ids is not None:
        expected = np.asarray([str(v) for v in expected_ids], dtype=object)
        actual = np.asarray(frame["id"].astype(str), dtype=object)
        if expected.size != actual.size:
            raise SubmissionError(
                f"submission has {actual.size} rows, expected {expected.size}"
            )
        if not np.array_equal(expected, actual):
            raise SubmissionError(
                "submission ids/order do not match test.csv; Kaggle scores by id, "
                "but a mismatch here usually means an alignment bug"
            )
    return frame


def build_submission(
    ids,
    predictions,
    path: str | Path,
    expected_ids=None,
) -> Path:
    """Write a validated ``submission.parquet`` (and return its path)."""
    ids = np.asarray(list(ids), dtype=object)
    predictions = np.asarray(list(predictions), dtype=object)
    if ids.size != predictions.size:
        raise SubmissionError(
            f"{ids.size} ids but {predictions.size} predictions"
        )
    frame = pd.DataFrame({"id": ids, "model": predictions})
    # Validate before coercion so nulls and odd types are rejected, not stringified.
    validate_frame(frame, expected_ids)
    frame["id"] = frame["id"].astype(str)
    frame["model"] = frame["model"].astype(str)
    path = Path(path)
    if path.suffix != ".parquet":
        raise SubmissionError(f"submission must be parquet named {SUBMISSION_FILENAME}, got {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    validate_submission(path, expected_ids)
    return path


def validate_submission(path: str | Path, expected_ids=None) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise SubmissionError(f"submission file missing: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - corrupt file
        raise SubmissionError(f"could not read {path} as parquet: {exc}") from exc
    return validate_frame(frame, expected_ids)
