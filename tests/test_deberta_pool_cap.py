"""Tests for the self-contained pool-cap helpers in the DeBERTa training kernel.

These are pure (no torch/transformers/GPU) so the class-imbalance fix for the
gemini-3.7-flash blend regression (D023) is unit-tested without a Kaggle run.
``kaggle/`` has no ``__init__.py``, so the script is imported by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "kaggle_deberta_train", ROOT / "kaggle" / "deberta" / "deberta_train.py"
)
train = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train)


def _pool():
    rows = []
    for label, n in (("A", 100), ("B", 10), ("C", 5)):
        for i in range(n):
            rows.append({"id": f"{label}-{i}", "model": label, "text": f"{label} {i}"})
    return pd.DataFrame(rows)


def test_balanced_cap_keeps_rare_whole_and_caps_majority():
    pool = _pool()
    out = train.cap_pool_balanced(pool, per_class_cap=20, seed=42)
    counts = out["model"].value_counts().to_dict()
    assert counts["A"] == 20          # majority downsampled to the floor
    assert counts["B"] == 10          # already below the floor, kept whole
    assert counts["C"] == 5           # rarest kept whole
    assert len(out) == 35


def test_balanced_cap_is_deterministic_per_seed():
    pool = _pool()
    a = train.cap_pool_balanced(pool, per_class_cap=20, seed=7)["id"].tolist()
    b = train.cap_pool_balanced(pool, per_class_cap=20, seed=7)["id"].tolist()
    assert a == b


def test_balanced_cap_nonpositive_returns_full_pool():
    pool = _pool()
    assert len(train.cap_pool_balanced(pool, per_class_cap=0, seed=1)) == len(pool)


def test_cap_pool_dispatches_on_balance():
    pool = _pool()
    balanced = train.cap_pool(pool, {"pool_balance": 20, "pool_cap_per_fold": 12000}, 42)
    assert sorted(balanced["model"].value_counts().values) == [5, 10, 20]
    prop = train.cap_pool(pool, {"pool_balance": 0, "pool_cap_per_fold": 25}, 42)
    # proportional preserves ratios: majority keeps the largest share
    assert prop["model"].value_counts().idxmax() == "A"
    assert len(prop) <= 25
