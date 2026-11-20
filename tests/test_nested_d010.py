from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments.nested_d010 import inner_grouped_folds, run_nested, select_cell


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
        for tr, va in GroupKFold(n_splits=4).split(np.zeros((len(texts), 1)), y, groups)
    ]
    return texts, y, groups, folds


def test_select_cell_takes_the_best_inner_score():
    assert select_cell({"svd64_k2": 0.20, "svd128_k8": 0.31, "svd256_k4": 0.25}) == "svd128_k8"


def test_select_cell_breaks_ties_by_smallest_svd_then_k():
    scores = {"svd256_k2": 0.30, "svd128_k16": 0.30, "svd128_k4": 0.30, "svd64_k8": 0.10}
    assert select_cell(scores) == "svd128_k4"


def test_inner_grouped_folds_never_share_a_prompt():
    _, y, groups, _ = _case()
    for train_idx, valid_idx in inner_grouped_folds(y, groups, n_splits=3, seed=71):
        assert len(train_idx) and len(valid_idx)
        assert not set(groups[train_idx]) & set(groups[valid_idx])


def test_run_nested_covers_every_row_and_scores_the_blend_once():
    texts, y, groups, folds = _case()
    adopted = np.zeros((len(texts), 18), dtype=np.float32)
    result = run_nested(
        texts,
        y,
        groups,
        folds,
        adopted,
        svd_widths=(8,),
        nuisance_ks=(0, 2),
        inner_splits=2,
        bootstrap_draws=200,
    )
    assert result["grid"] == ["svd8_k0", "svd8_k2"]
    assert len(result["folds"]) == len(folds)
    assert 0.0 <= result["blend_pooled"] <= 1.0
    assert len(result["blend_fold_scores"]) == len(folds)
    assert result["bootstrap"]["low"] <= result["bootstrap"]["high"]
    for report in result["folds"]:
        assert report["selected_cell"] in {"svd8_k0", "svd8_k2"}
        assert set(report["inner_cells"]) == {"svd8_k0", "svd8_k2"}
        assert 0.0 <= report["candidate_holdout_balanced_accuracy"] <= 1.0
        assert "holdout_delta" in report


def test_a_fold_selection_ignores_its_own_holdout_labels():
    """Rewriting a fold's outer-valid labels must not move that fold's inner scores.

    Only the perturbed fold is compared: its holdout rows are training rows for
    the other folds, so those may legitimately change.
    """
    texts, y, groups, folds = _case()
    adopted = np.zeros((len(texts), 18), dtype=np.float32)
    kwargs = dict(svd_widths=(8,), nuisance_ks=(0, 2), inner_splits=2, bootstrap_draws=100)
    baseline = run_nested(texts, y, groups, folds, adopted, **kwargs)

    for fold, (_, valid_idx) in enumerate(folds):
        perturbed = y.copy()
        perturbed[valid_idx] = (perturbed[valid_idx] + 1) % 3
        shifted = run_nested(texts, perturbed, groups, folds, adopted, **kwargs)
        assert shifted["folds"][fold]["selected_cell"] == baseline["folds"][fold]["selected_cell"]
        assert shifted["folds"][fold]["inner_cells"] == baseline["folds"][fold]["inner_cells"]
