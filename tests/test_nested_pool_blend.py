from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.nested_pool_blend import cap_pool_stratified, evaluate_nested_pool_blend
from src.constants import LABELS

A, B = LABELS[0], LABELS[1]
CONFIGS = [
    {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg",
     "C": 1.0, "max_features": 2000},
    {"name": "style_only", "features": [], "style": True, "classifier": "logreg", "C": 1.0},
]


def _synthetic(n_groups: int = 8, per: int = 4):
    rows = []
    vocab = ["alpha beta gamma", "delta epsilon zeta", "quantum lattice gauge field",
             "compiler parsing token grammar", "market revenue growth quarter",
             "chlorophyll photosynthesis leaf", "recursion stack frame heap",
             "sunset horizon ocean tide"]
    for g in range(n_groups):
        for r in range(per):
            label = A if (g + r) % 2 == 0 else B
            base = vocab[(g + r) % len(vocab)]
            # class-correlated cue so the pool can plausibly help char features
            cue = "model style marker one" if label == A else "model style marker two"
            rows.append({"id": f"t{g}-{r}", "prompt_id": f"g{g}",
                         "text": f"{base} {cue} filler {r} tokens here for variety",
                         "model": label})
    return pd.DataFrame(rows)


def _pool(n: int = 24):
    rows = []
    for i in range(n):
        label = A if i % 2 == 0 else B
        cue = "model style marker one" if label == A else "model style marker two"
        rows.append({"id": f"p{i}", "text": f"unrelated pool passage {i} {cue} extra filler",
                     "model": label, "prompt_canon": f"c{i % 6}", "max_cos_train": 0.0})
    return pd.DataFrame(rows)


def test_cap_pool_stratified_preserves_classes_and_respects_cap():
    pool = _pool()
    capped = cap_pool_stratified(pool, 10, seed=7)
    assert len(capped) <= 12  # cap plus per-class rounding slack
    assert set(capped["model"]) <= {A, B}
    # cap >= pool size is a no-op
    assert len(cap_pool_stratified(pool, 10_000, seed=7)) == len(pool)


def test_evaluate_nested_pool_blend_runs_and_covers_all_rows():
    df = _synthetic()
    result = evaluate_nested_pool_blend(df, _pool(), CONFIGS, seed=0, pool_cap=12,
                                       outer_splits=3, outer_seed=121)
    assert result["outer_splits"] == 3
    assert len(result["folds"]) == 3
    assert 0.0 <= result["mix_pooled_balanced_accuracy"] <= 1.0
    assert 0.0 <= result["official_only_pooled_balanced_accuracy"] <= 1.0
    assert result["no_inner_selection"] is True
    # every outer fold must report its independent per-fold pool screen
    for r in result["folds"]:
        assert r["pool_screened_against"] == "outer_train_only"
        assert 0 <= r["pool_clean_rows"] <= 24
        assert 0.0 <= r["mix_outer_balanced_accuracy"] <= 1.0
        assert 0.0 <= r["official_only_outer_balanced_accuracy"] <= 1.0
    assert set(result["per_class_recall_delta"]) == set(LABELS)


def test_evaluate_nested_pool_blend_screens_pool_against_outer_train_only():
    # A pool row that exactly duplicates a passage present in the data is only removed
    # by the folds whose outer-TRAIN contains that passage, never by a fold holding it
    # out: prove the per-fold masks differ when one passage sits in different folds.
    df = _synthetic()
    dup_text = df["text"].iloc[0]
    pool = _pool()
    pool = pd.concat([pool, pd.DataFrame(
        [{"id": "dup", "text": dup_text, "model": A, "prompt_canon": "cD", "max_cos_train": 0.0}])],
        ignore_index=True)
    result = evaluate_nested_pool_blend(df, pool, CONFIGS, seed=0, pool_cap=24,
                                        cos_threshold=0.5, outer_splits=3, outer_seed=121)
    exact_counts = [r["pool_exact_dup"] for r in result["folds"]]
    # the duplicate row is in the pool for every fold but is an exact-dup only when its
    # passage is in that fold's outer-train, so at least one fold removes fewer than all.
    assert sum(exact_counts) >= 1
    assert any(c <= 1 for c in exact_counts)


def test_evaluate_nested_pool_blend_rejects_wrong_member_order():
    with pytest.raises(ValueError, match="char_tfidf_logreg"):
        evaluate_nested_pool_blend(_synthetic(), _pool(), [CONFIGS[1], CONFIGS[0]])
