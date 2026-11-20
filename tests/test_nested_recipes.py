"""The H10 recipe gate must select on inner rows only, and only pool members see pool rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import experiments.blend_members as bm
import experiments.nested_recipes as nr
from src.constants import LABELS
from src.ensemble import normalize_scores
from src.validation import make_splits

A, B = LABELS[0], LABELS[1]
CONFIGS = [
    {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg",
     "C": 1.0, "max_features": 2000},
    {"name": "style_only", "features": [], "style": True, "classifier": "logreg", "C": 1.0},
    {"name": "word_tfidf_linearsvc", "features": ["word"], "classifier": "linearsvc",
     "C": 1.0, "max_features": 2000},
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


def _correct_wrong(n: int = 6):
    """char right on every row, style wrong on every row, word right and confident."""
    y = np.arange(n) % 2
    char = np.full((n, 18), -5.0)
    char[np.arange(n), y] = 5.0
    style = np.full((n, 18), -5.0)
    style[np.arange(n), 1 - y] = 5.0
    word = np.full((n, 18), -5.0)
    word[np.arange(n), y] = 5.0
    zs = {"char_tfidf_logreg": normalize_scores(char), "style_only": normalize_scores(style),
          "word_tfidf_linearsvc": normalize_scores(word)}
    return zs, y


def test_base_recipe_reproduces_the_adopted_blend() -> None:
    rng = np.random.default_rng(0)
    zs = {"char_tfidf_logreg": normalize_scores(rng.normal(size=(20, 18))),
          "style_only": normalize_scores(rng.normal(size=(20, 18))),
          "word_tfidf_linearsvc": normalize_scores(rng.normal(size=(20, 18)))}
    adopted = (0.5 * zs["char_tfidf_logreg"] + 0.5 * zs["style_only"]).argmax(1)
    base = dict(nr.RECIPES[0][1])
    assert np.array_equal(nr.recipe_predictions(zs, base), adopted)


def test_gate_recipes_are_the_screen_recipes() -> None:
    recipes = dict(nr.RECIPES)
    assert recipes[nr.BASE_RECIPE] == {"char_tfidf_logreg": 0.5, "style_only": 0.5}
    keep = recipes["keep_ratio"]
    assert keep["word_tfidf_linearsvc"] == pytest.approx(bm.THIRD_WEIGHT)
    assert keep["char_tfidf_logreg"] == pytest.approx((1.0 - bm.THIRD_WEIGHT) * 0.5)
    assert keep["char_tfidf_logreg"] == pytest.approx(keep["style_only"])
    assert recipes["thirds"]["word_tfidf_linearsvc"] == pytest.approx(1.0 / 3.0)


def test_select_recipe_takes_the_best_inner_score() -> None:
    zs, y = _correct_wrong()
    chosen, curve = nr.select_recipe(zs, y)
    base_ba = next(r["inner_pooled_ba"] for r in curve if r["recipe"] == nr.BASE_RECIPE)
    chosen_ba = next(r["inner_pooled_ba"] for r in curve if r["recipe"] == chosen)
    assert chosen != nr.BASE_RECIPE
    assert chosen_ba == pytest.approx(1.0)
    assert base_ba < chosen_ba
    assert [r["recipe"] for r in curve] == [n for n, _ in nr.RECIPES]


def test_select_recipe_keeps_the_base_on_ties() -> None:
    zs, y = _correct_wrong()
    zs["style_only"] = zs["char_tfidf_logreg"].copy()
    zs["word_tfidf_linearsvc"] = zs["char_tfidf_logreg"].copy()
    chosen, curve = nr.select_recipe(zs, y)
    assert chosen == nr.BASE_RECIPE
    assert len({round(r["inner_pooled_ba"], 12) for r in curve}) == 1


def test_nested_selection_never_sees_outer_holdout_rows(monkeypatch) -> None:
    df = _synthetic()
    seen: list[int] = []
    original = nr.select_recipe

    def spy(zs, y, recipes=nr.RECIPES):
        rows = next(iter(zs.values())).shape[0]
        seen.append(int(rows))
        assert rows == y.size
        return original(zs, y, recipes)

    monkeypatch.setattr(nr, "select_recipe", spy)
    result = nr.evaluate_nested_recipe_selection(df, _pool(), CONFIGS, seed=0, pool_cap=12,
                                                 outer_splits=3, outer_seed=121, inner_splits=2,
                                                 inner_seed=71)
    folds, _ = make_splits(df, n_splits=3, seed=121)
    assert seen == [len(tr) for tr, _ in folds]
    for r, (tr, va) in zip(result["folds"], folds):
        assert r["inner_selection_rows"] == len(tr)
        assert r["outer_holdout_rows"] == len(va)
        assert r["inner_selection_rows"] + r["outer_holdout_rows"] == len(df)


def test_only_pool_members_are_fit_on_pool_rows(monkeypatch) -> None:
    df = _synthetic()
    # Vocabulary disjoint from the official rows, so the global screen keeps every
    # pool row and the fit sizes differ by exactly the pool budget.
    pool = pd.DataFrame([{"id": f"p{i}", "text": f"zzz qqq xxx www vvv uuu ttt sss rrr {i}",
                          "model": A if i % 2 == 0 else B, "prompt_canon": f"c{i % 6}",
                          "max_cos_train": 0.0} for i in range(24)])
    fits: list[tuple[str, int]] = []

    class _Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def fit(self, texts, y, groups=None):
            fits.append((self.name, len(texts)))
            return self

        def decision_function(self, texts):
            return np.zeros((len(texts), len(LABELS)))

    monkeypatch.setattr(nr, "estimator_from_config", lambda cfg, seed=42: _Recorder(str(cfg["name"])))
    nr.evaluate_nested_recipe_selection(df, pool, CONFIGS, seed=0, pool_cap=24,
                                        outer_splits=2, outer_seed=121, inner_splits=2,
                                        inner_seed=71, verbose=False)
    assert fits and len(fits) % len(CONFIGS) == 0
    for i in range(0, len(fits), len(CONFIGS)):
        round_fits = dict(fits[i:i + len(CONFIGS)])
        official = round_fits["style_only"]
        assert official < len(df)  # a strict subset of official rows, never the holdout
        assert round_fits["char_tfidf_logreg"] == official + len(pool)
        assert round_fits["word_tfidf_linearsvc"] == official + len(pool)


def test_nested_runs_and_reports_both_recipes_on_the_same_folds() -> None:
    df = _synthetic()
    result = nr.evaluate_nested_recipe_selection(df, _pool(), CONFIGS, seed=0, pool_cap=12,
                                                 outer_splits=3, outer_seed=121, inner_splits=2,
                                                 inner_seed=71, verbose=False)
    assert len(result["folds"]) == 3
    assert result["base_recipe"] == nr.BASE_RECIPE
    assert result["pool_members"] == sorted(nr.POOL_MEMBERS)
    for r in result["folds"]:
        assert r["selected_recipe"] in [n for n, _ in nr.RECIPES]
        assert r["pool_screened_against"] == "outer_train_only"
        assert 0.0 <= r["selected_outer_balanced_accuracy"] <= 1.0
        assert 0.0 <= r["base_outer_balanced_accuracy"] <= 1.0
    assert result["pooled_ba_delta"] == pytest.approx(
        result["selected_pooled_balanced_accuracy"] - result["base_pooled_balanced_accuracy"])
    assert "paired_outer_fold_diff" in result


def test_nested_rejects_wrong_member_order() -> None:
    with pytest.raises(ValueError, match="char_tfidf_logreg"):
        nr.evaluate_nested_recipe_selection(_synthetic(), _pool(),
                                            [CONFIGS[1], CONFIGS[0], CONFIGS[2]])