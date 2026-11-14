from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import validate_test_frame, validate_train_frame  # noqa: E402
from src.synthetic import make_synthetic_dataset  # noqa: E402
from src.validation import make_splits  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_data():
    train, test, truth = make_synthetic_dataset(n_prompts=40, test_prompts=12, seed=7)
    return (
        validate_train_frame(train),
        validate_test_frame(test),
        truth,
    )


@pytest.fixture(scope="session")
def train_df(synthetic_data):
    return synthetic_data[0].copy()


@pytest.fixture(scope="session")
def test_df(synthetic_data):
    return synthetic_data[1].copy()


@pytest.fixture(scope="session")
def truth_df(synthetic_data):
    return synthetic_data[2].copy()


@pytest.fixture(scope="session")
def folds(train_df):
    splits, _method = make_splits(train_df, n_splits=3, seed=13)
    return splits


@pytest.fixture()
def tiny_char_model():
    return {
        "name": "tiny_char",
        "features": ["char"],
        "classifier": "linearsvc",
        "max_features": 3000,
        "char_ngram_range": [2, 3],
        "min_df": 1,
    }
