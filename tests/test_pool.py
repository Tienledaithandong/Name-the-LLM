from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LABELS
from src.pool import (
    PoolError,
    audit_pool,
    decontaminate_fold,
    load_pool,
    per_fold_decontamination,
)

A, B = LABELS[0], LABELS[1]


def _pool(rows):
    return pd.DataFrame(rows, columns=["id", "text", "model", "prompt_canon", "max_cos_train"])


def _train(rows):
    return pd.DataFrame(rows, columns=["text", "model", "prompt_id"])


def test_load_pool_rejects_unknown_label(tmp_path):
    pool = _pool([("p1", "hello world", "not-a-real-model", "pX", 0.1)])
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    with pytest.raises(PoolError, match="outside the official 18"):
        load_pool(path)


def test_load_pool_rejects_missing_column(tmp_path):
    pool = pd.DataFrame({"id": ["p1"], "text": ["hi"], "model": [A]})
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    with pytest.raises(PoolError, match="missing required columns"):
        load_pool(path)


def test_load_pool_strict_drops_after_ruling(tmp_path):
    pool = _pool([("p1", "alpha beta gamma", A, "pX", 0.1),
                  ("p2", "delta epsilon zeta", B, "pY", 0.1)])
    pool["flag_kb_proxy_after_ruling"] = [True, False]
    path = tmp_path / "pool.parquet"
    pool.to_parquet(path)
    kept = load_pool(path, strict=True)
    assert kept["id"].tolist() == ["p2"]
    assert len(load_pool(path, strict=False)) == 2


def test_audit_pool_verifies_flags_and_duplicates():
    pool = _pool([("p1", "shared passage here", A, "pX", 0.2),
                  ("p2", "totally different text", B, "pY", 0.1)])
    pool["flag_gemini37_routing_risk"] = [False, False]
    train = _train([("shared passage here", A, "g1")])
    report = audit_pool(pool, train)
    assert report["n_rows"] == 2
    assert report["exact_text_duplicates_vs_train"] == 1
    assert report["flags"]["flag_gemini37_routing_risk"] == {"False": 2}
    assert report["max_cos_train_max"] == pytest.approx(0.2)


def test_decontaminate_fold_drops_exact_duplicate():
    pool = _pool([("p1", "the quick brown fox jumps", A, "pX", 0.1),
                  ("p2", "unrelated content about stars", B, "pY", 0.1)])
    holdout = _train([("the quick brown fox jumps", A, "g1")])
    keep, stats = decontaminate_fold(pool, holdout)
    assert keep.tolist() == [False, True]
    assert stats["exact_dup"] == 1


def test_decontaminate_fold_drops_near_duplicate():
    base = "machine learning models generate text with distinctive stylistic patterns"
    pool = _pool([("p1", base + " extra", A, "pX", 0.1),
                  ("p2", "completely unrelated cooking recipe instructions", B, "pY", 0.1)])
    holdout = _train([(base, A, "g1")])
    keep, stats = decontaminate_fold(pool, holdout, cos_threshold=0.3)
    assert keep[0] == False  # near-duplicate removed
    assert stats["near_dup"] >= 1
    assert stats["max_cos_holdout_max"] > 0.3


def test_decontaminate_fold_drops_shared_prompt():
    pool = _pool([("p1", "alpha beta gamma delta", A, "g1", 0.1),
                  ("p2", "epsilon zeta eta theta", B, "g2", 0.1)])
    holdout = _train([("some held out passage text", A, "g1")])
    keep, stats = decontaminate_fold(pool, holdout)
    assert keep.tolist() == [False, True]
    assert stats["shared_prompt"] == 1


def test_decontaminate_fold_keeps_clean_rows():
    pool = _pool([("p1", "quarterly financial revenue growth metrics", A, "pX", 0.1),
                  ("p2", "photosynthesis chlorophyll sunlight biology", B, "pY", 0.1)])
    holdout = _train([("medieval castle architecture history", A, "g1")])
    keep, stats = decontaminate_fold(pool, holdout)
    assert keep.all()
    assert stats["removed"] == 0


def test_per_fold_decontamination_unions_removed_ids():
    pool = _pool([("p1", "shared one two three four", A, "g1", 0.1),
                  ("p2", "shared five six seven eight", B, "g2", 0.1),
                  ("p3", "clean unrelated astronomy telescope", A, "pZ", 0.1)])
    train = _train([("shared one two three four", A, "g1"),
                    ("shared five six seven eight", B, "g2"),
                    ("clean unrelated astronomy telescope", A, "g3")])
    folds = [(np.array([1, 2]), np.array([0])), (np.array([0, 2]), np.array([1]))]
    result = per_fold_decontamination(pool, train, folds)
    assert result["n_folds"] == 2
    assert set(result["union_removed_ids"]) == {"p1", "p2"}
    assert result["clean_pool_rows"] == 1