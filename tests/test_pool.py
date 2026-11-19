from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LABELS
from src.pool import (
    PoolError,
    audit_pool,
    decontaminate_fold,
    decontaminate_global,
    load_pool,
    load_prompt_mapping,
    per_fold_decontamination,
)

A, B = LABELS[0], LABELS[1]


def _pool(rows):
    return pd.DataFrame(rows, columns=["id", "text", "model", "prompt_canon", "max_cos_train"])


def _train(rows):
    return pd.DataFrame(rows, columns=["text", "model", "prompt_id"])


def _decon(pool, holdout, *, prompt_mapping=None, **kwargs):
    if prompt_mapping is None:
        prompt_mapping = {str(group): [] for group in holdout["prompt_id"]}
    return decontaminate_fold(
        pool, holdout,
        fit_texts=["the quick brown fox machine learning models generate text with "
                   "distinctive stylistic patterns extra held out passage alpha beta "
                   "gamma delta epsilon zeta eta theta"],
        prompt_mapping=prompt_mapping,
        mapping_provenance="synthetic test mapping",
        **kwargs,
    )


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
    keep, stats = _decon(pool, holdout)
    assert keep.tolist() == [False, True]
    assert stats["exact_dup"] == 1


def test_decontaminate_fold_drops_near_duplicate():
    base = "machine learning models generate text with distinctive stylistic patterns"
    pool = _pool([("p1", base + " extra", A, "pX", 0.1),
                  ("p2", "completely unrelated cooking recipe instructions", B, "pY", 0.1)])
    holdout = _train([(base, A, "g1")])
    keep, stats = _decon(pool, holdout, cos_threshold=0.3)
    assert keep[0] == False  # near-duplicate removed
    assert stats["near_dup"] >= 1
    assert stats["max_cos_holdout_max"] > 0.3


def test_decontaminate_fold_drops_shared_prompt():
    pool = _pool([("p1", "alpha beta gamma delta", A, "canon-A", 0.1),
                  ("p2", "epsilon zeta eta theta", B, "canon-B", 0.1)])
    holdout = _train([("some held out passage text", A, "g1")])
    keep, stats = _decon(pool, holdout, prompt_mapping={"g1": ["canon-A"]})
    assert keep.tolist() == [False, True]
    assert stats["shared_prompt"] == 1


def test_decontaminate_fold_keeps_clean_rows():
    pool = _pool([("p1", "quarterly financial revenue growth metrics", A, "pX", 0.1),
                  ("p2", "photosynthesis chlorophyll sunlight biology", B, "pY", 0.1)])
    holdout = _train([("medieval castle architecture history", A, "g1")])
    keep, stats = _decon(pool, holdout)
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
    result = per_fold_decontamination(
        pool, train, folds, prompt_mapping={"g1": ["g1"], "g2": ["g2"]},
        mapping_provenance="synthetic test mapping",
    )
    assert result["n_folds"] == 2
    assert set(result["union_removed_ids"]) == {"p1", "p2"}
    assert result["clean_pool_rows"] == 1


def test_decontamination_refuses_unverified_or_incomplete_mapping():
    pool = _pool([("p1", "a shared prompt response", A, "canon-A", 0.1)])
    holdout = _train([("different response from same prompt", A, "official-1")])
    with pytest.raises(PoolError, match="verified prompt_mapping"):
        decontaminate_fold(pool, holdout, fit_texts=["training text"])
    with pytest.raises(PoolError, match="lacks 1 held-out prompt"):
        decontaminate_fold(pool, holdout, fit_texts=["training text"],
                           prompt_mapping={}, mapping_provenance="checked")
    with pytest.raises(PoolError, match="sequence of canonical IDs"):
        decontaminate_fold(pool, holdout, fit_texts=["training text"],
                           prompt_mapping={"official-1": "canon-A"},
                           mapping_provenance="checked")


def test_decontamination_requires_training_only_vectorizer_fit():
    pool = _pool([("p1", "validationonlyword", A, "canon-A", 0.1)])
    holdout = _train([("validationonlyword anotherword", A, "official-1")])
    with pytest.raises(PoolError, match="fold-training fit_texts are required"):
        decontaminate_fold(pool, holdout, prompt_mapping={"official-1": []},
                           mapping_provenance="checked")
    with pytest.raises(PoolError, match="overlap held-out text"):
        decontaminate_fold(pool, holdout, fit_texts=holdout["text"].tolist(),
                           prompt_mapping={"official-1": []}, mapping_provenance="checked")
    keep, stats = decontaminate_fold(
        pool, holdout, fit_texts=["training vocabulary alone"],
        prompt_mapping={"official-1": []}, mapping_provenance="checked",
    )
    assert keep.tolist() == [True]  # the held-out-only token was not learned
    assert stats["similarity_fit_scope"] == "fold_training_only"


def test_decontamination_drops_pool_rows_without_canonical_prompt_id():
    pool = _pool([("missing", "unrelated response", A, None, 0.1),
                  ("empty", "another unrelated response", B, " ", 0.1),
                  ("known", "third unrelated response", A, "known", 0.1)])
    holdout = _train([("held out passage", A, "group")])
    keep, stats = _decon(pool, holdout)
    assert keep.tolist() == [False, False, True]
    assert stats["unmapped_pool_prompt"] == 2


def test_per_fold_rejects_same_prompt_on_both_sides():
    pool = _pool([("p1", "pool example", A, "canon", 0.1)])
    train = _train([("training text", A, "group"), ("held out text", B, "group")])
    with pytest.raises(PoolError, match="prompt on both sides"):
        per_fold_decontamination(
            pool, train, [(np.array([0]), np.array([1]))],
            prompt_mapping={"group": ["canon"]}, mapping_provenance="synthetic fixture",
        )


def test_load_prompt_mapping_checks_structure_and_records_digest(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text('{"provenance":"reviewed fixture", "mapping":{"g1":["canon-A"],"g2":[]}}')
    mapping, provenance = load_prompt_mapping(path)
    assert mapping == {"g1": ["canon-A"], "g2": []}
    assert provenance.startswith("reviewed fixture (sha256:")
    assert len(provenance.split("sha256:")[1].rstrip(")")) == 64


@pytest.mark.parametrize("content", [
    '{}',
    '{"provenance":"", "mapping":{"g1":[]}}',
    '{"provenance":"fixture", "mapping":{"g1":"canon"}}',
    '{"provenance":"fixture", "mapping":{"g1":[""]}}',
])
def test_load_prompt_mapping_rejects_unverified_shapes(tmp_path, content):
    path = tmp_path / "mapping.json"
    path.write_text(content)
    with pytest.raises(PoolError):
        load_prompt_mapping(path)


# ---- D028 non-transductive global decontamination -------------------------

_UNIQUE = ("the quick brown fox generates distinctive stylistic patterns across "
           "machine learning language model output with steady rhythm")
_NEAR = ("the quick brown fox generates distinctive stylistic patterns across "
         "machine learning language model output with steady cadence")
_CLEAN = ("entirely unrelated quantum chromodynamics lattice simulation notes "
          "about gluon propagators and renormalization")


def _gtrain():
    # Two official passages; only the first is near the pool rows.
    return _train([(_UNIQUE, A, "g1"), (_CLEAN, B, "g2")])


def _gpool():
    return _pool([
        ("p-exact", _UNIQUE, A, "cX", 0.9),   # exact duplicate of a train passage
        ("p-near", _NEAR, B, "cY", 0.8),      # near duplicate (same words, one differs)
        ("p-clean", _CLEAN + " with extra padding tokens here", B, "cZ", 0.05),
    ])


def test_decontaminate_global_drops_exact_and_near_keeps_clean():
    keep, stats = decontaminate_global(_gpool(), _gtrain(), cos_threshold=0.3)
    assert stats["decontamination_scope"] == "global_full_official_train"
    assert stats["validation_independent"] is True
    assert stats["exact_dup"] == 1 and stats["near_dup"] >= 1
    ids = [i for i, k in zip(_gpool()["id"], keep) if k]
    assert "p-exact" not in ids and "p-near" not in ids


def test_decontaminate_global_is_identical_regardless_of_fold_partition():
    # The mask depends only on (pool, full train), never on which rows are held out,
    # so every fold reuses the exact same clean set. Recompute and compare.
    pool, train = _gpool(), _gtrain()
    k1, s1 = decontaminate_global(pool, train, cos_threshold=0.3)
    k2, s2 = decontaminate_global(pool, train, cos_threshold=0.3)
    assert np.array_equal(k1, k2)
    assert s1["kept"] == s2["kept"]


def test_decontaminate_global_screens_against_full_train_including_heldout_rows():
    # A pool row near a passage that would sit in a *validation* fold is still removed,
    # because screening uses the complete official train set. This is what makes the
    # protocol fold-independent and non-transductive (D028).
    pool = _pool([("p-near-g2", _NEAR + " steady rhythm extra filler words", B, "cY", 0.0)])
    train = _train([("unrelated alpha beta", A, "g1"),
                    (_UNIQUE, B, "g2")])  # this passage will be a validation row
    keep, _ = decontaminate_global(pool, train, cos_threshold=0.3)
    assert not keep.any()


def test_decontaminate_global_rejects_empty_or_bad_params():
    with pytest.raises(PoolError, match="must not be empty"):
        decontaminate_global(_gpool(), _train([]), cos_threshold=0.3)
    with pytest.raises(PoolError, match="cos_threshold"):
        decontaminate_global(_gpool(), _gtrain(), cos_threshold=1.5)
    with pytest.raises(PoolError, match="chunk_size"):
        decontaminate_global(_gpool(), _gtrain(), chunk_size=0)
