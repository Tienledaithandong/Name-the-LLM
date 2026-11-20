"""The blend-weight gate must select on inner rows only (D026 leakage class)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import experiments.blend_weights as bw
from src.constants import LABELS
from src.ensemble import normalize_scores
from src.validation import make_splits

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


def test_half_weight_reproduces_the_adopted_recipe() -> None:
    rng = np.random.default_rng(0)
    char, style = rng.normal(size=(20, 18)), rng.normal(size=(20, 18))
    adopted = (0.5 * normalize_scores(char) + 0.5 * normalize_scores(style)).argmax(1)
    assert np.array_equal(bw.blend_predictions(normalize_scores(char), normalize_scores(style), 0.5),
                          adopted)


def test_select_weight_takes_the_best_inner_score() -> None:
    # char is right on every row, style is wrong: the selector must lean on char.
    y = np.array([0, 1, 0, 1, 0, 1])
    char = np.full((6, 18), -5.0)
    char[np.arange(6), y] = 5.0
    style = np.full((6, 18), -5.0)
    style[np.arange(6), 1 - y] = 5.0
    chosen, curve = bw.select_weight(normalize_scores(char), normalize_scores(style), y)
    # Row-z makes the two members equally confident, so 0.5 ties and argmax breaks by
    # index; any weight above 0.5 classifies every row. Ties then break back toward 0.5.
    assert chosen > bw.FIXED_WEIGHT
    assert [r["weight"] for r in curve] == list(bw.WEIGHT_GRID)
    best = max(r["inner_pooled_ba"] for r in curve)
    assert next(r["inner_pooled_ba"] for r in curve if r["weight"] == chosen) == best
    assert next(r["inner_pooled_ba"] for r in curve if r["weight"] == bw.FIXED_WEIGHT) < best


def test_select_weight_breaks_ties_toward_the_prespecified_half() -> None:
    # Identical members make every weight score the same; the recipe must not drift.
    y = np.array([0, 1, 0, 1])
    scores = np.full((4, 18), -5.0)
    scores[np.arange(4), y] = 5.0
    z = normalize_scores(scores)
    chosen, curve = bw.select_weight(z, z, y)
    assert chosen == bw.FIXED_WEIGHT
    assert len({round(r["inner_pooled_ba"], 12) for r in curve}) == 1


def test_curve_reports_zero_paired_diff_at_the_fixed_weight() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=60)
    folds = np.arange(60) % 3
    char, style = rng.normal(size=(60, 18)), rng.normal(size=(60, 18))
    curve = bw.weight_curve(char, style, y, folds, (0.4, 0.5, 0.6))
    fixed = next(r for r in curve if r["char_weight"] == 0.5)
    assert fixed["paired_diff_vs_fixed_0.5"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert len(fixed["fold_ba"]) == 3


def test_nested_selection_never_sees_outer_holdout_rows(monkeypatch) -> None:
    df = _synthetic()
    seen: list[int] = []
    original = bw.select_weight

    def spy(char_z, style_z, y, grid=bw.WEIGHT_GRID):
        seen.append(int(char_z.shape[0]))
        assert char_z.shape[0] == y.size
        return original(char_z, style_z, y, grid)

    monkeypatch.setattr(bw, "select_weight", spy)
    result = bw.evaluate_nested_weight_selection(df, _pool(), CONFIGS, seed=0, pool_cap=12,
                                                 outer_splits=3, outer_seed=121, inner_splits=2,
                                                 inner_seed=71)
    folds, _ = make_splits(df, n_splits=3, seed=121)
    expected = [len(tr) for tr, _ in folds]
    assert seen == expected
    assert sum(seen) < len(df) * len(folds)  # strictly fewer than all rows per fold
    for r, (tr, va) in zip(result["folds"], folds):
        assert r["inner_selection_rows"] == len(tr)
        assert r["outer_holdout_rows"] == len(va)
        assert r["inner_selection_rows"] + r["outer_holdout_rows"] == len(df)


def test_nested_runs_and_reports_both_recipes_on_the_same_folds() -> None:
    df = _synthetic()
    result = bw.evaluate_nested_weight_selection(df, _pool(), CONFIGS, seed=0, pool_cap=12,
                                                 outer_splits=3, outer_seed=121, inner_splits=2,
                                                 inner_seed=71)
    assert len(result["folds"]) == 3
    assert result["fixed_char_weight"] == 0.5
    for r in result["folds"]:
        assert r["selected_char_weight"] in bw.WEIGHT_GRID
        assert r["pool_screened_against"] == "outer_train_only"
        assert 0.0 <= r["selected_outer_balanced_accuracy"] <= 1.0
        assert 0.0 <= r["fixed_outer_balanced_accuracy"] <= 1.0
    assert "paired_outer_fold_diff" in result
    assert result["pooled_ba_delta"] == pytest.approx(
        result["selected_pooled_balanced_accuracy"] - result["fixed_pooled_balanced_accuracy"])


def test_nested_rejects_wrong_member_order() -> None:
    with pytest.raises(ValueError, match="char_tfidf_logreg"):
        bw.evaluate_nested_weight_selection(_synthetic(), _pool(), [CONFIGS[1], CONFIGS[0]])