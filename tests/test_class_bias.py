"""Class-bias tuning must stay out-of-sample, deterministic and shrink to zero.

Plan §7 L196 asks for class-specific intercepts with out-of-sample tuning and
strict separation. These tests pin ``experiments/class_bias.py``: the reference
arm is exactly the adopted row-z blend, the search never invents a bias when the
inner score cannot be improved, a systematic class offset is corrected, and the
adoption verdict needs every seed to win with a paired CI above zero.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.constants import LABELS, N_CLASSES
from src.ensemble import normalize_scores

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "experiments" / "class_bias.py"
_spec = importlib.util.spec_from_file_location("class_bias", SCRIPT)
class_bias = importlib.util.module_from_spec(_spec)
sys.modules["class_bias"] = class_bias
_spec.loader.exec_module(class_bias)


def _scores_with_offset(n_rows: int = 360, offset_class: int = 0, offset: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    y = np.tile(np.arange(N_CLASSES), n_rows // N_CLASSES)
    scores = rng.normal(0.0, 0.3, size=(y.size, N_CLASSES))
    scores[np.arange(y.size), y] += 1.0
    scores[:, offset_class] += offset
    return scores, y


def _frame(n_prompts: int = 12, per_prompt: int = 6) -> pd.DataFrame:
    rows = []
    for p in range(n_prompts):
        for j in range(per_prompt):
            label = LABELS[(p * per_prompt + j) % N_CLASSES]
            rows.append({
                "id": f"r{p}_{j}", "prompt_id": f"p{p}", "model": label,
                "text": f"{label} marker for prompt {p} " + " ".join(["token"] * (6 + j)),
            })
    return pd.DataFrame(rows)


def test_reference_arm_is_the_adopted_row_z_blend() -> None:
    scores, y = _scores_with_offset()
    char = scores
    style = scores * 0.5 + 0.1
    expected = 0.5 * normalize_scores(char) + 0.5 * normalize_scores(style)
    assert class_bias.blend_scores(char, style) == pytest.approx(expected)


def test_prior_offset_is_centred_and_near_zero_when_balanced() -> None:
    y = np.tile(np.arange(N_CLASSES), 20)
    offset = class_bias.prior_log_offset(y)
    assert offset.shape == (N_CLASSES,)
    assert float(offset.mean()) == pytest.approx(0.0, abs=1e-12)
    # official train support is 149/150, i.e. essentially balanced
    assert np.abs(offset).max() < 0.01


def test_prior_offset_penalises_the_majority_class() -> None:
    y = np.concatenate([np.zeros(200, dtype=int), np.tile(np.arange(1, N_CLASSES), 5)])
    offset = class_bias.prior_log_offset(y)
    assert offset[0] < 0.0
    assert offset[1] > 0.0
    assert float(offset.mean()) == pytest.approx(0.0, abs=1e-12)


def test_tune_bias_leaves_a_perfect_classifier_untouched() -> None:
    y = np.tile(np.arange(N_CLASSES), 20)
    scores = np.zeros((y.size, N_CLASSES))
    scores[np.arange(y.size), y] = 100.0
    bias, info = class_bias.tune_bias(scores, y)
    assert np.all(bias == 0.0)
    assert info["shrinkage"] == 0.0
    assert info["inner_balanced_accuracy_no_bias"] == pytest.approx(1.0)


def test_tune_bias_corrects_a_systematic_class_offset() -> None:
    scores, y = _scores_with_offset()
    bias, info = class_bias.tune_bias(scores, y)
    assert bias[0] < 0.0, "the inflated class must be pushed down"
    assert info["inner_balanced_accuracy_shrunk"] > info["inner_balanced_accuracy_no_bias"]
    assert info["shrinkage"] in class_bias.SHRINK_GRID


def test_tune_bias_is_deterministic() -> None:
    scores, y = _scores_with_offset()
    first, first_info = class_bias.tune_bias(scores, y)
    second, second_info = class_bias.tune_bias(scores, y)
    assert np.array_equal(first, second)
    assert first_info == second_info


def test_tune_bias_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        class_bias.tune_bias(np.zeros((10, N_CLASSES - 1)), np.zeros(10, dtype=int))


def test_inner_oof_covers_exactly_the_outer_training_rows() -> None:
    df = _frame()
    configs = load_config(REPO / "configs" / "submission.yaml")["models"]
    train_idx = np.arange(len(df) - 6)  # six rows stand in for the outer holdout
    inner, info = class_bias.inner_oof(
        df, train_idx, configs, model_seed=42, inner_splits=3, inner_seed=212,
    )
    assert inner.shape == (train_idx.size, N_CLASSES)
    assert np.isfinite(inner).all()
    assert info[0]["inner_method"] in {"stratified_group", "group"}
    assert sum(f["n_valid"] for f in info[0]["folds"]) == train_idx.size


def _payload(mean: float, low: float) -> dict:
    return {"arms": {"tuned": {"paired_vs_no_bias": {"mean": mean, "ci_low": low, "ci_high": mean + 0.02}}}}


def test_verdict_requires_every_seed_to_win() -> None:
    decision = class_bias.verdict({42: _payload(0.01, 0.002), 13: _payload(0.008, 0.001)})
    assert decision["adopt_tuned_bias"] is True


@pytest.mark.parametrize("seeds", [
    {42: _payload(0.01, 0.002), 13: _payload(0.008, -0.001)},
    {42: _payload(0.01, 0.002), 13: _payload(-0.004, -0.01)},
    {42: _payload(0.01, -0.002)},
])
def test_verdict_rejects_a_seed_that_does_not_win(seeds) -> None:
    decision = class_bias.verdict(seeds)
    assert decision["adopt_tuned_bias"] is False
    assert "keep the adopted blend unchanged" in decision["reason"]