"""Per-prompt error concentration must be judged against a binomial null.

Plan §3 L118 asks for performance by prompt group. A prompt holds only ~18 rows,
so a wide accuracy spread is expected even when no prompt is harder than another.
These tests pin ``experiments/prompt_groups.py``: the grouping arithmetic, the
concentration statistic, the null that makes it interpretable, and the fact that
a genuinely concentrated failure is distinguishable from uniform noise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "prompt_groups.py"
_spec = importlib.util.spec_from_file_location("prompt_groups", SCRIPT)
pg = importlib.util.module_from_spec(_spec)
sys.modules["prompt_groups"] = pg
_spec.loader.exec_module(pg)


def _case():
    """Three prompts: 4 rows each, accuracies 1.0, 0.5 and 0.0."""
    prompts = np.array(["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4)
    truth = np.array(["a", "b", "c", "d"] * 3)
    pred = np.array(["a", "b", "c", "d", "a", "b", "x", "x", "x", "x", "x", "x"])
    return truth, pred, prompts


def test_prompt_table_groups_by_prompt_id() -> None:
    truth, pred, prompts = _case()
    table = pg.prompt_table(truth, pred, prompts)
    assert table["prompt_ids"] == ["p1", "p2", "p3"]
    assert table["n"].tolist() == [4, 4, 4]
    assert table["correct"].tolist() == [4, 2, 0]
    assert np.allclose(table["accuracy"], [1.0, 0.5, 0.0])


def test_prompt_table_handles_uneven_group_sizes() -> None:
    truth = np.array(["a", "b", "a"])
    pred = np.array(["a", "x", "a"])
    prompts = np.array(["p1", "p1", "p2"])
    table = pg.prompt_table(truth, pred, prompts)
    assert table["n"].tolist() == [2, 1]
    assert np.allclose(table["accuracy"], [0.5, 1.0])


@pytest.mark.parametrize(
    "truth,pred,prompts",
    [
        (np.array(["a"]), np.array(["a", "b"]), np.array(["p1"])),
        (np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=str)),
        (np.array(["a"]), np.array(["a"]), np.array([""])),
    ],
)
def test_prompt_table_rejects_bad_input(truth, pred, prompts) -> None:
    with pytest.raises(ValueError):
        pg.prompt_table(truth, pred, prompts)


def test_concentration_all_errors_in_one_prompt() -> None:
    table = {"n": np.array([4, 4, 4]), "correct": np.array([4, 4, 0])}
    conc = pg.error_concentration(table, top_fraction=1 / 3)
    assert conc["total_errors"] == 4
    assert conc["top_share_of_errors"] == 1.0
    assert conc["zero_accuracy_prompts"] == 1
    assert conc["perfect_prompts"] == 2


def test_concentration_uniform_errors() -> None:
    table = {"n": np.full(10, 4), "correct": np.full(10, 2)}
    conc = pg.error_concentration(table, top_fraction=0.2)
    assert conc["k_prompts"] == 2
    assert conc["top_share_of_errors"] == pytest.approx(0.2)


def test_concentration_no_errors_is_zero_not_nan() -> None:
    table = {"n": np.array([3, 3]), "correct": np.array([3, 3])}
    conc = pg.error_concentration(table)
    assert conc["top_share_of_errors"] == 0.0
    assert conc["perfect_prompts"] == 2


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_concentration_rejects_bad_fraction(fraction) -> None:
    table = {"n": np.array([4]), "correct": np.array([2])}
    with pytest.raises(ValueError):
        pg.error_concentration(table, top_fraction=fraction)


def test_null_with_no_errors_predicts_all_perfect() -> None:
    null = pg.binomial_null(np.full(20, 18), 0.0, draws=50)
    assert null["top_share_mean"] == 0.0
    assert null["perfect_mean"] == 20.0
    assert null["zero_accuracy_mean"] == 0.0


def test_null_with_all_errors_predicts_all_zero() -> None:
    null = pg.binomial_null(np.full(20, 18), 1.0, top_fraction=0.1, draws=50)
    assert null["zero_accuracy_mean"] == 20.0
    assert null["perfect_mean"] == 0.0
    assert null["top_share_mean"] == pytest.approx(0.1)


def test_null_is_deterministic_for_a_fixed_seed() -> None:
    n = np.full(30, 18)
    a = pg.binomial_null(n, 0.6, draws=200, seed=7)
    b = pg.binomial_null(n, 0.6, draws=200, seed=7)
    assert a == b


def test_null_spread_widens_with_noise() -> None:
    """At a 60% error rate the worst decile carries more than its row share."""
    null = pg.binomial_null(np.full(150, 18), 0.6, draws=400)
    assert null["top_share_mean"] > 0.10
    assert null["top_share_p95"] > null["top_share_mean"] > null["top_share_p05"]


@pytest.mark.parametrize("draws,rate", [(0, 0.5), (-1, 0.5), (10, -0.1), (10, 1.1)])
def test_null_rejects_bad_arguments(draws, rate) -> None:
    with pytest.raises(ValueError):
        pg.binomial_null(np.full(5, 4), rate, draws=draws)


def test_concentrated_failure_exceeds_the_null() -> None:
    """15 prompts absorb every error; uniform noise at the same rate cannot."""
    n = np.full(150, 18)
    correct = n.copy()
    correct[:15] = 0
    table = {"n": n, "correct": correct}
    conc = pg.error_concentration(table, top_fraction=0.1)
    null = pg.binomial_null(n, conc["total_errors"] / (150 * 18), draws=400)
    assert conc["top_share_of_errors"] == pytest.approx(1.0)
    assert conc["top_share_of_errors"] > null["top_share_p95"]
    assert conc["zero_accuracy_prompts"] > null["zero_accuracy_p95"]


def test_uniform_failure_matches_the_null() -> None:
    """Every prompt loses exactly half its rows: no prompt-level concentration.

    A perfectly even split is *less* concentrated than the binomial null, which
    has sampling variance, so the honest assertion is "at or below chance", not
    "inside the null interval".
    """
    n = np.full(150, 18)
    table = {"n": n, "correct": np.full(150, 9)}
    conc = pg.error_concentration(table, top_fraction=0.1)
    null = pg.binomial_null(n, 0.5, draws=400)
    assert conc["top_share_of_errors"] == pytest.approx(0.1)
    assert conc["top_share_of_errors"] < null["top_share_mean"]
    assert conc["top_share_of_errors"] <= null["top_share_p05"]
    assert conc["zero_accuracy_prompts"] == 0


def test_prediction_spread_detects_a_collapsed_prompt() -> None:
    prompts = np.array(["p1"] * 4 + ["p2"] * 4)
    pred = np.array(["z", "z", "z", "z", "a", "b", "c", "d"])
    spread = pg.prediction_spread(pred, prompts)
    assert spread["prompts_collapsed_to_one_label"] == 1
    assert spread["distinct_predictions_min"] == 1
    assert spread["distinct_predictions_max"] == 4
    assert spread["modal_prediction_share_max"] == 1.0
    assert spread["modal_prediction_share_mean"] == pytest.approx(0.625)


def test_wrong_prediction_spread_one_wrong_answer() -> None:
    truth = np.array(["a", "b", "c", "d"])
    pred = np.array(["x", "x", "x", "d"])
    prompts = np.array(["p1"] * 4)
    wrong = pg.wrong_prediction_spread(truth, pred, prompts)
    assert wrong["prompts_with_errors"] == 1
    assert wrong["modal_wrong_share_mean"] == 1.0
    assert wrong["prompts_one_wrong_answer"] == 1


def test_wrong_prediction_spread_with_no_errors() -> None:
    truth = np.array(["a", "b"])
    wrong = pg.wrong_prediction_spread(truth, truth, np.array(["p1", "p1"]))
    assert wrong["prompts_with_errors"] == 0
    assert wrong["modal_wrong_share_mean"] is None


def test_accuracy_summary_orders_quantiles() -> None:
    truth, pred, prompts = _case()
    summary = pg.accuracy_summary(pg.prompt_table(truth, pred, prompts))
    assert summary["min"] <= summary["p10"] <= summary["median"] <= summary["p90"] <= summary["max"]
    assert summary["mean"] == pytest.approx(0.5)


def test_worst_prompts_are_sorted_and_capped() -> None:
    truth, pred, prompts = _case()
    table = pg.prompt_table(truth, pred, prompts)
    rows = pg.worst_prompts(table, None, np.array([10.0, 20.0, 30.0]), limit=2)
    assert [r["prompt_id"] for r in rows] == ["p3", "p2"]
    assert rows[0]["accuracy"] == 0.0
    assert set(rows[0]) == {"prompt_id", "n", "correct", "accuracy", "mean_words"}


def test_worst_prompts_can_pair_a_second_variant() -> None:
    truth, pred, prompts = _case()
    table = pg.prompt_table(truth, pred, prompts)
    other = {"accuracy": np.array([1.0, 1.0, 0.25])}
    rows = pg.worst_prompts(table, other, np.zeros(3), limit=1)
    assert rows[0]["pool_mix_accuracy"] == 0.25


def test_markdown_summary_is_well_formed() -> None:
    variants = {
        "official_blend": {
            "pooled_balanced_accuracy": 0.376,
            "accuracy": {"mean": 0.37, "std": 0.2, "min": 0.0, "median": 0.35, "max": 0.9},
            "concentration": {"top_share_of_errors": 0.2, "zero_accuracy_prompts": 3},
            "binomial_null": {"top_share_mean": 0.18, "top_share_p05": 0.14,
                              "top_share_p95": 0.23, "zero_accuracy_mean": 2.5},
        }
    }
    md = pg.markdown_summary(variants)
    lines = md.splitlines()
    assert len(lines) == 3
    assert all(line.count("|") == 10 for line in lines)
    assert "official_blend" in lines[2]