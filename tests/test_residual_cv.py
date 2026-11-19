from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from experiments.run_residual_cv import run_residual_table, select_lambda

MARKERS = ("alpha", "beta", "gamma")


def _synthetic_case(n_prompts=15, n_models=3, repeats=3):
    texts, groups, labels = [], [], []
    for prompt in range(n_prompts):
        for model in range(n_models):
            for repeat in range(repeats):
                texts.append(
                    f"prompt {prompt} answer {MARKERS[model]} "
                    + ("detail step " * (repeat + 1))
                    + f"topic {prompt % 4}"
                )
                groups.append(f"p{prompt}")
                labels.append(model)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    folds = [
        (np.asarray(tr), np.asarray(va))
        for tr, va in GroupKFold(n_splits=5).split(
            np.zeros((len(texts), 1)), y, groups
        )
    ]
    return texts, y, groups, folds


def test_lambda_zero_learned_equals_control():
    texts, y, groups, folds = _synthetic_case()
    result = run_residual_table(
        texts, y, groups, folds, spaces=("style",), lambdas=(0.0,)
    )
    assert result["cells"]["style|learned"]["pooled"] == pytest.approx(
        result["cells"]["style|none"]["pooled"], abs=1e-9
    )
    assert result["cells"]["style|learned"]["fold_scores"] == pytest.approx(
        result["cells"]["style|none"]["fold_scores"], abs=1e-9
    )


def test_table_has_every_shape_and_reports_diagnostics():
    texts, y, groups, folds = _synthetic_case()
    result = run_residual_table(
        texts, y, groups, folds, spaces=("style",), lambdas=(0.0, 1.0)
    )
    modes = {cell["mode"] for cell in result["cells"].values()}
    assert modes == {"none", "d010", "learned"}
    learned = result["cells"]["style|learned"]
    assert len(learned["fold_scores"]) == len(folds)
    assert len(learned["selected_lambda"]) == len(folds)
    assert all(0.0 <= value <= 1.0 for value in learned["selected_lambda"])
    assert learned["oof_r2"] is not None
    assert result["oracle"]["style|oracle"]["pooled"] >= 0.0


def test_lambda_selection_is_deterministic_on_the_training_side():
    texts, y, groups, _folds = _synthetic_case()
    first, rows_a = select_lambda(texts, y, groups, lambdas=(0.0, 0.5, 1.0))
    second, rows_b = select_lambda(texts, y, groups, lambdas=(0.0, 0.5, 1.0))
    assert first == second
    assert rows_a == rows_b
    assert [row["lambda"] for row in rows_a] == [0.0, 0.5, 1.0]


def test_selection_ties_prefer_the_smaller_shrinkage():
    # All-identical rows make every lambda score the same; the safer default
    # (no neutralisation) must win rather than an arbitrary larger value.
    texts = ["same answer text"] * 30
    y = np.tile(np.arange(3), 10)
    groups = np.repeat(np.arange(10), 3)
    chosen, _rows = select_lambda(texts, y, groups, lambdas=(0.0, 1.0))
    assert chosen == 0.0
