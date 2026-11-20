"""Guard the proportional vs balanced pool-cap modes used by the pool experiments.

The Open Pool is heavily class-imbalanced while the official train set is uniform
(150/class), so a proportional cap starves thin classes such as gemini-3.7-flash
relative to their better-covered siblings. ``cap_pool_balanced`` (D036) applies a
uniform per-class target instead. Both caps must stay validation-independent: only
class counts and the fixed seed decide which rows enter training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.pool_train import CAP_MODES, cap_pool_balanced, cap_pool_stratified


def _imbalanced_pool(n_classes: int = 4):
    # one thin class + progressively larger siblings
    counts = {f"c{i}": [8, 400, 900, 1200][i] for i in range(n_classes)}
    rows = []
    for label, n in counts.items():
        for j in range(n):
            rows.append({"id": f"{label}-{j}", "text": f"{label} passage {j} token", "model": label})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("cap_fn", [cap_pool_stratified, cap_pool_balanced])
def test_caps_respect_cap_and_are_deterministic(cap_fn):
    pool = _imbalanced_pool()
    a = cap_fn(pool, 1000, seed=7)
    b = cap_fn(pool, 1000, seed=7)
    assert len(a) <= 1000
    assert a["id"].tolist() == b["id"].tolist()


@pytest.mark.parametrize("cap_fn", [cap_pool_stratified, cap_pool_balanced])
def test_caps_never_invent_or_drop_classes_unexpectedly(cap_fn):
    pool = _imbalanced_pool()
    capped = cap_fn(pool, 1000, seed=7)
    # every class present in the pool is still represented after capping
    assert set(capped["model"]) == set(pool["model"])


def test_balanced_gives_uniform_per_class_target():
    pool = _imbalanced_pool()
    capped = cap_pool_balanced(pool, 1000, seed=7)
    counts = capped["model"].value_counts()
    target = round(1000 / pool["model"].nunique())
    # large classes are capped to the uniform target; the thin class keeps all rows
    assert counts["c3"] == target
    assert counts["c2"] == target
    assert counts["c1"] == target
    assert counts["c0"] == 8  # thinner than the target, so no subsampling


def test_proportional_starves_thin_class_relative_to_balanced():
    pool = _imbalanced_pool()
    prop = cap_pool_stratified(pool, 1000, seed=7)["model"].value_counts()
    bal = cap_pool_balanced(pool, 1000, seed=7)["model"].value_counts()
    # the thin class gets strictly more representation under the balanced cap
    assert bal["c0"] > prop["c0"]


def test_cap_modes_registry_matches_functions():
    assert CAP_MODES["proportional"] is cap_pool_stratified
    assert CAP_MODES["balanced"] is cap_pool_balanced


def test_cap_zero_or_small_pool_is_identity():
    pool = _imbalanced_pool()
    assert len(cap_pool_balanced(pool, 0, seed=7)) == len(pool)
    assert len(cap_pool_balanced(pool, 10_000, seed=7)) == len(pool)
    assert len(cap_pool_stratified(pool, 0, seed=7)) == len(pool)
