from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from experiments.sweep_d010 import blend_with_third, cell_oof, run_sweep


def _case(n_prompts=12, n_models=3, repeats=2):
    texts, groups, labels = [], [], []
    for prompt in range(n_prompts):
        for model in range(n_models):
            for repeat in range(repeats):
                texts.append(
                    f"prompt {prompt} model {model} "
                    + ("detail step " * (repeat + 1))
                    + f"topic {prompt % 3}"
                )
                groups.append(f"p{prompt}")
                labels.append(model)
    y = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    folds = [
        (np.asarray(tr), np.asarray(va))
        for tr, va in GroupKFold(n_splits=4).split(
            np.zeros((len(texts), 1)), y, groups
        )
    ]
    return texts, y, groups, folds


def test_blend_with_third_is_equal_thirds():
    rng = np.random.default_rng(0)
    adopted = rng.normal(size=(5, 3))
    third = rng.normal(size=(5, 3))
    assert np.allclose(blend_with_third(adopted, third), (2 * adopted + third) / 3)


def test_cell_oof_produces_aligned_full_width_scores():
    texts, y, groups, folds = _case()
    scores = cell_oof(texts, y, groups, folds, svd=8, nuisance_k=2)
    assert scores.shape == (len(texts), 18)
    assert np.isfinite(scores).all()


def test_sweep_reports_every_grid_cell_and_a_blend_delta():
    texts, y, groups, folds = _case()
    adopted = np.zeros((len(texts), 18), dtype=np.float32)
    result = run_sweep(
        texts, y, groups, folds, adopted, svd_widths=(8,), nuisance_ks=(0, 2)
    )
    assert set(result) == {"svd8_k0", "svd8_k2"}
    for cell in result.values():
        assert 0.0 <= cell["blend"] <= 1.0
        assert len(cell["blend_fold_scores"]) == len(folds)
        assert "single_view" in cell and 0.0 <= cell["single_view"] <= 1.0
        ci = cell["bootstrap"]
        assert ci["low"] <= ci["high"] and ci["n_prompts"] == 12


def test_nuisance_k_zero_is_a_no_op_suppressor():
    texts, y, groups, folds = _case()
    with_zero = cell_oof(texts, y, groups, folds, svd=8, nuisance_k=0)
    assert np.isfinite(with_zero).all()
    assert not np.all(with_zero == pytest.approx(-1e9))
