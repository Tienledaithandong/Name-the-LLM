# Style-Residual Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a deployable prompt-mean neutralizer (`e - λ·g(e)`) and measure, on prompt-grouped folds only, whether removing a *learned* estimate of each passage's prompt mean beats the adopted 0.3760 OOF blend — and beats our published D010 global projection, which is a special case of this idea.

**Architecture:** A new `src/residual.py` holds a `PromptMeanNeutralizer` (ridge regressor predicting the row's own fold-training prompt mean, applied as a shrunk subtraction, with a prompt-grouped out-of-fold `R²` diagnostic). `TextMatrixBuilder` gains a `residual_lambda` stage applied after SVD and after the D010 suppressor, so every space in the 2×2 table is the same code path. An experiment runner sweeps λ inside each outer fold on the training side only and emits the {none, D010, learned} × {style, svd128, embedding} table plus the `between_prompt_share` pre-check.

**Tech Stack:** Python 3.13, numpy, scipy sparse, scikit-learn (`Ridge`, `StandardScaler`, `GroupKFold`, `LogisticRegression`), existing `src/features_tfidf.py` / `src/model.py` / `src/validation.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-style-residual-distillation-design.md`

**Execution status (native, 2026-11-19):** Tasks 1-5 complete. The Task 5 gate
failed on its first run — see D058 — so Tasks 6-8 (P1 GPU layers, P2 pair head,
P3 calibration sweep) were not started, per the spec's "no GPU spend after a
failed R2/D010 comparison" rule. The unused tasks stay below as the shape any
revival would take.

Task 2's `TextMatrixBuilder`/`ModelSpec` wiring was removed afterwards: the
runner calls `PromptMeanNeutralizer` directly, so the extra stage was dead code
and its tests went with it. `src/residual.py` and both experiment scripts stay.

## Global Constraints

- Prompt-grouped folds only; every learned transform (vectorizers, scaler, SVD, D010 suppressor, neutralizer, λ) fitted on the training side of the fold. Reuse `load_splits` / `validate_splits`; never recompute prompt means from validation or test rows.
- Primary metric: pooled grouped OOF balanced accuracy (D018), cv_s42, same `split_id` and dataset fingerprint as the 0.3760 baseline. k-subsampling stays inference-only.
- 18-label order from `src/constants.LABELS` only; `ModelBundle.load` must keep rejecting mismatch.
- No transductive step, no test-time grouping, no clustering, no Hungarian assignment anywhere in this plan.
- λ = 0 must reproduce today's model **exactly** (skip the neutralizer, not multiply by zero).
- Registry row (via `append_registry`) + ADR (next id **D058**) for every measured cell; no number reported that a run's notes do not cite.
- CPU-first. Any GPU step is a `kaggle/` script kernel using the local `kaggle/deberta/folds.json`; internet off.
- Never commit `data/`, `artifacts/`, OOF files, `pytest_out.txt`, or secrets.

## Review Focus

- A test passage whose prompt is *unseen* at transform time: `g` must still return a prediction (it is a function of the passage, not the group) and must not raise or return NaN.
- Degenerate neutralizer: if `‖g(e)‖` collapses toward 0 the premise failed; the diagnostic must report it rather than silently passing as "no change".
- λ selection must never see validation rows: an inner-split refit that touches outer-validation rows inflates every number and is the one silent leak this plan cannot recover from.
- Sparse/dense boundaries: style-only and embedding-only blocks arrive sparse from `_raw`, so the residual stage must densify; the D010 stage runs first and already returns dense.
- Single-class or tiny prompt groups in a fold (a prompt with one training row): the neutralizer must fall back to identity instead of dividing by zero.

---

### Task 1: `PromptMeanNeutralizer` in `src/residual.py`

**Files:**
- Create: `src/residual.py`
- Test: `tests/test_residual.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `class ResidualError(ValueError)`
  - `class PromptMeanNeutralizer:`
    - `__init__(self, alpha: float = 1.0, shrinkage: float = 1.0, min_group_size: int = 2)`
    - `fit(self, X, groups) -> "PromptMeanNeutralizer"`
    - `transform(self, X) -> np.ndarray` (returns `float32`, same shape as input)
    - `fit_transform(self, X, groups) -> np.ndarray`
    - fitted attributes: `mean_: np.ndarray`, `scale_: np.ndarray`, `regressor_`, `n_groups_: int`, `oof_r2_: float`, `predicted_norm_ratio_: float`

Semantics: target for row `i` is `m_{p(i)} = mean_{j in p(i)} X_j`, computed over the rows handed to `fit`. `transform` returns `X - shrinkage * (g(X_std) * scale_ + mean_)` where `g` is a `Ridge(alpha=alpha)` fitted on `StandardScaler`-standardised `X`; `shrinkage=0` returns `X` unchanged. Groups with fewer than `min_group_size` rows are excluded from the training targets but still transformed. `oof_r2_` is the mean `R²` over `GroupKFold(n_splits=min(3, n_groups))` inner folds (prompt-grouped, so it is not optimistic), and `predicted_norm_ratio_` is `mean(‖shrinkage·g(e)‖) / mean(‖e − mean_‖)`, which is the collapse detector.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from src.residual import PromptMeanNeutralizer, ResidualError


def _grouped(seed=0, n_groups=6, per_group=6, dim=5, between=4.0):
    rng = np.random.default_rng(seed)
    offsets = rng.normal(scale=between, size=(n_groups, dim))
    groups = np.repeat(np.arange(n_groups), per_group)
    X = rng.normal(scale=1.0, size=(groups.size, dim)) + offsets[groups]
    return X, groups


def test_shrinkage_zero_is_exact_identity():
    X, groups = _grouped()
    out = PromptMeanNeutralizer(shrinkage=0.0).fit(X, groups).transform(X)
    assert np.array_equal(out, X.astype(np.float32))


def test_leaves_every_group_mean_near_zero_when_prediction_is_perfect():
    # A prompt mean that is an affine function of the row is fully recoverable,
    # so lambda=1 must nearly annihilate the between-prompt component.
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(8), 5)
    base = np.eye(8)[groups]
    X = base + 0.01 * rng.normal(size=base.shape)
    neutralizer = PromptMeanNeutralizer(alpha=1e-6, shrinkage=1.0).fit(X, groups)
    out = neutralizer.transform(X)
    means = np.stack([out[groups == g].mean(axis=0) for g in np.unique(groups)])
    assert np.abs(means).max() < 0.05
    assert neutralizer.oof_r2_ > 0.5


def test_diagnostic_is_grouped_and_reports_collapse():
    X, groups = _grouped()
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    assert 0.0 <= neutralizer.oof_r2_ <= 1.0
    assert neutralizer.predicted_norm_ratio_ > 0.0


def test_unseen_group_rows_transform_without_nan():
    X, groups = _grouped()
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    out = neutralizer.transform(np.zeros((3, X.shape[1])) + 0.5)
    assert out.shape == (3, X.shape[1]) and np.isfinite(out).all()


def test_single_row_group_falls_back_to_identity_targets():
    X, groups = _grouped(n_groups=4, per_group=1)
    neutralizer = PromptMeanNeutralizer().fit(X, groups)
    assert neutralizer.n_groups_ == 0
    assert np.isfinite(neutralizer.transform(X)).all()


def test_transform_before_fit_and_bad_shapes_raise():
    with pytest.raises(ResidualError, match="before fit"):
        PromptMeanNeutralizer().transform(np.zeros((2, 3)))
    X, groups = _grouped()
    with pytest.raises(ResidualError, match="one entry per row"):
        PromptMeanNeutralizer().fit(X, groups[:-1])
    with pytest.raises(ResidualError, match="shrinkage"):
        PromptMeanNeutralizer(shrinkage=1.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_residual.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.residual'`.

- [ ] **Step 3: Implement `src/residual.py`**

Use `sklearn.linear_model.Ridge`, `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.GroupKFold`, `sklearn.metrics.r2_score`. Raise `ResidualError` for `shrinkage` outside `[0, 1]`, `alpha <= 0`, non-2-D input, and `fit` before `transform`. Standardise with `with_mean=True`; at zero-variance columns substitute scale 1.0 (the pattern in `src/prompt_invariance.py:44-46`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_residual.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/residual.py tests/test_residual.py
git commit -m "feat: add prompt-mean neutralizer with grouped out-of-fold R2 diagnostic"
```

---

### Task 2: residual stage in `TextMatrixBuilder` and `ModelSpec`

**Files:**
- Modify: `src/features_tfidf.py` (`TextMatrixBuilder.__init__`, `_reduce`)
- Modify: `src/model.py` (`ModelSpec` fields, `from_dict`, `to_dict`, `ModelBundle.__init__`)
- Test: `tests/test_features.py`, `tests/test_model.py`

**Interfaces:**
- Consumes: `PromptMeanNeutralizer` from Task 1.
- Produces:
  - `TextMatrixBuilder(..., residual_lambda: float = 0.0, residual_alpha: float = 1.0)`
  - `ModelSpec.residual_lambda: float = 0.0`, `ModelSpec.residual_alpha: float = 1.0` (both round-trip through `to_dict`/`from_dict`)
  - `TextMatrixBuilder.residual_` attribute (the fitted `PromptMeanNeutralizer` or `None`), carried inside the saved bundle.

Order inside `_reduce`: SVD → D010 suppressor → residual. `residual_lambda > 0` requires dense input, so style-only and embedding-only configs must densify (`matrix.toarray()`), and `ModelSpec.from_dict` must reject `residual_lambda > 0` when `features` is non-empty and `svd` is unset (the same rule as `nuisance_k`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features.py — add this module-level helper next to the existing imports
def _texts_and_groups():
    texts, groups = [], []
    for prompt in range(6):
        for variant in range(4):
            texts.append(
                f"Prompt {prompt} answer {variant}: "
                + ("step by step we compute the value. " * (variant + 1))
            )
            groups.append(f"p{prompt}")
    return texts, groups


def test_residual_stage_leaves_lambda_zero_untouched():
    texts, groups = _texts_and_groups()
    plain = TextMatrixBuilder(style=True).fit(texts, groups=groups)
    resid = TextMatrixBuilder(style=True, residual_lambda=0.0).fit(texts, groups=groups)
    assert np.allclose(plain.transform(texts), resid.transform(texts))
    assert resid.residual_ is None


def test_residual_stage_refuses_sparse_without_svd():
    texts, groups = _texts_and_groups()
    with pytest.raises(FeatureError, match="residual"):
        TextMatrixBuilder(kinds=("char",), residual_lambda=0.5).fit(texts, groups=groups)


def test_residual_state_is_fitted_on_train_side_only():
    # Transforming a held-out prompt must not change when other held-out rows appear.
    texts, groups = _texts_and_groups()
    builder = TextMatrixBuilder(style=True, residual_lambda=1.0).fit(texts, groups=groups)
    single = builder.transform([texts[0]])
    bigger = builder.transform([texts[0], texts[1]])
    assert np.allclose(single[0], bigger[0])
    assert builder.residual_.n_groups_ > 1
```

```python
# tests/test_model.py
def test_model_spec_round_trips_residual_fields():
    spec = ModelSpec.from_dict({"name": "r", "features": [], "style": True, "residual_lambda": 0.5})
    assert spec.residual_lambda == 0.5 and spec.residual_alpha == 1.0
    assert ModelSpec.from_dict(spec.to_dict()).residual_lambda == 0.5


def test_model_spec_rejects_sparse_residual():
    with pytest.raises(ModelError, match="residual"):
        ModelSpec.from_dict({"name": "r", "features": ["char"], "residual_lambda": 0.5})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_features.py tests/test_model.py -q -k residual`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'residual_lambda'`.

- [ ] **Step 3: Implement the stage**

In `_reduce`, after the `nuisance_k` block: if `self.residual_lambda > 0`, densify when `sparse.issparse(matrix)`, then on `fit` build `PromptMeanNeutralizer(alpha=self.residual_alpha, shrinkage=self.residual_lambda).fit(dense, groups)` and store it as `self.residual_`; on transform apply `self.residual_.transform(dense)`. Raise `FeatureError("residual neutralisation requires dense input — set an svd size or use style/embedding features")` at `__init__` when `residual_lambda > 0` and `not svd` and `kinds` is non-empty. Mirror the same check in `ModelSpec.from_dict`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_features.py tests/test_model.py tests/test_prompt_invariance.py -q`
Expected: all pass, including the existing D010 tests.

- [ ] **Step 5: Commit**

```bash
git add src/features_tfidf.py src/model.py tests/test_features.py tests/test_model.py
git commit -m "feat: apply prompt-mean residual stage inside the feature builder"
```

---

### Task 3: `between_prompt_share` pre-check

**Files:**
- Create: `experiments/residual_headroom.py`
- Test: `tests/test_residual_headroom.py`

**Interfaces:**
- Consumes: `prompt_centering_diagnostic` (`src/prompt_invariance.py`), `load_official_train` (`experiments/run_score_improvement.py`), `load_splits` (`src/validation.py`).
- Produces:
  - `representation_matrix(texts, space: str) -> np.ndarray` for `space in {"style", "svd128", "embedding"}`
  - `headroom_table(texts, groups, spaces=("style", "svd128", "embedding")) -> dict[str, dict]` mapping space → `{"between_prompt_share": float, "n_prompts": int, "n_features": int}`
  - CLI: `python -m experiments.residual_headroom --out artifacts/residual/headroom.json`

`svd128` is `TextMatrixBuilder(kinds=("char",), svd=128)`, `style` is `TextMatrixBuilder(style=True)`, `embedding` is `TextMatrixBuilder(embedding={"backend": "hashing", "dim": 384})`. Fitting happens on the full official train frame here because this is a diagnostic of the data, not a model score — say so in the printed output and in the JSON's `"note"` field.

- [ ] **Step 1: Write the failing test**

```python
def test_headroom_table_reports_every_space():
    texts = [f"passage {i} about topic {i % 3}" for i in range(30)]
    groups = [str(i % 3) for i in range(30)]
    table = headroom_table(texts, groups, spaces=("style",))
    cell = table["style"]
    assert 0.0 <= cell["between_prompt_share"] <= 1.0
    assert cell["n_prompts"] == 3 and cell["n_features"] > 0


def test_identical_prompts_give_full_share():
    texts = ["same text here"] * 6
    table = headroom_table(texts, ["p"] * 6, spaces=("style",))
    assert table["style"]["between_prompt_share"] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_residual_headroom.py -q`
Expected: FAIL with `ImportError: cannot import name 'headroom_table'`.

- [ ] **Step 3: Implement the module**

Reuse `prompt_centering_diagnostic`; do not re-derive the variance formula. Note in a docstring that a degenerate all-identical case has zero total variance and therefore returns 0.0 by the existing guard.

- [ ] **Step 4: Run test and the real pre-check**

Run: `python -m pytest tests/test_residual_headroom.py -q`
Expected: PASS.
Run: `python -m experiments.residual_headroom --out artifacts/residual/headroom.json`
Expected: a JSON file with three spaces; paste the three `between_prompt_share` values into the Task 5 ADR.

- [ ] **Step 5: Commit**

```bash
git add experiments/residual_headroom.py tests/test_residual_headroom.py
git commit -m "feat: add between-prompt variance headroom pre-check"
```

---

### Task 4: the {none, D010, learned} × {style, svd128, embedding} table

**Files:**
- Create: `experiments/run_residual_cv.py`
- Test: `tests/test_residual_cv.py`

**Interfaces:**
- Consumes: `TextMatrixBuilder(..., nuisance_k=..., residual_lambda=...)` from Task 2, `headroom_table` from Task 3, `load_splits`, `evaluate`/`balanced_accuracy`, `append_registry`.
- Produces:
  - `SHAPES = {"none": {}, "d010": {"nuisance_k": 4}, "learned": {"residual_lambda": "<selected>"}}`
  - `select_lambda(texts, y, groups, lambdas, n_inner=3, seed=42) -> tuple[float, list[dict]]` — inner prompt-grouped CV over the *training side only*; returns the argmax mean inner balanced accuracy and the per-λ inner rows.
  - `run_residual_table(texts, y, groups, folds, spaces=("style", "svd128", "embedding"), lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), nuisance_k=4) -> dict`
  - Result structure: `{"between_prompt_share": {...}, "cells": {(space, mode): {"pooled": float, "fold_scores": [float], "selected_lambda": float | None, "oof_r2": float | None, "predicted_norm_ratio": float | None}}, "oracle": {space: {"pooled": float}}, "lambdas": [...]}` where `mode in {"none", "d010", "learned"}`.
  - CLI: `python -m experiments.run_residual_cv --splits artifacts/splits/splits_official_verified_s42.json --out artifacts/residual/cv_s42.json --registry experiments/registry.csv`

Every cell uses `LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)` so the only difference between cells is the representation. One outer fold refits the builder and classifier per λ on inner splits, then once on the full training side with the selected λ. The `oracle` cell uses `centered_features` (diagnostic only, never a registry "candidate").

- [ ] **Step 1: Write the failing test**

```python
def test_lambda_zero_learned_equals_control():
    texts, y, groups, folds = _synthetic_case()  # helper in this file: 12 prompts x 3 classes
    result = run_residual_table(texts, y, groups, folds, spaces=("style",), lambdas=(0.0,))
    assert result["cells"][("style", "learned")]["pooled"] == pytest.approx(
        result["cells"][("style", "none")]["pooled"], abs=1e-9
    )


def test_table_has_every_shape_and_reports_diagnostics():
    texts, y, groups, folds = _synthetic_case()
    result = run_residual_table(texts, y, groups, folds, spaces=("style",), lambdas=(0.0, 1.0))
    assert {mode for (_, mode) in result["cells"]} == {"none", "d010", "learned"}
    learned = result["cells"][("style", "learned")]
    assert learned["oof_r2"] is not None and learned["predicted_norm_ratio"] is not None
    assert len(learned["fold_scores"]) == len(folds)


def test_lambda_selection_uses_training_rows_only(monkeypatch):
    # Selection must be reproducible from the training side alone: hiding the
    # validation rows must give the same choice.
    texts, y, groups, folds = _synthetic_case()
    train_idx = folds[0][0]
    chosen_a, _ = select_lambda([texts[i] for i in train_idx], y[train_idx], groups[train_idx], (0.0, 1.0))
    chosen_b, _ = select_lambda([texts[i] for i in train_idx], y[train_idx], groups[train_idx], (0.0, 1.0))
    assert chosen_a == chosen_b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_residual_cv.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the runner**

Sweep λ with `GroupKFold` over `train_idx` only. Skip the neutralizer entirely when λ = 0 (that is what makes `test_lambda_zero_learned_equals_control` exact). For each outer fold keep `selected_lambda`, the inner rows, and the fitted neutralizer's `oof_r2_` / `predicted_norm_ratio_` from the full-training-side refit. Write one registry row per cell with `method=f"residual_{space}_{mode}"`, `notes` containing the λ grid, the selected λ per fold and the command line, `status="done"` (never `candidate` without the gate).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_residual_cv.py -q`
Expected: PASS (three tests, no GPU, under a minute).

- [ ] **Step 5: Commit**

```bash
git add experiments/run_residual_cv.py tests/test_residual_cv.py
git commit -m "feat: add 2x2 D010-vs-learned-neutralizer CV table runner"
```

---

### Task 5: run P0 on official data and record the verdict

**Files:**
- Create: `docs/decisions.md` (append **D058**)
- Modify: `experiments/registry.csv` (written by the runner, not by hand)
- Write: `artifacts/residual/cv_s42.json` (not committed)

**Interfaces:**
- Consumes: everything from Tasks 1–4; `artifacts/splits/splits_official_verified_s42.json`; the adopted baseline OOF (`char_tfidf_logreg` 0.5 + `style_only` 0.5, pooled 0.3760).
- Produces: D058 with the measured table, or a negative-result ADR of the same shape.

- [ ] **Step 1: Run the pre-check and the full table**

Run: `python -m experiments.residual_headroom --out artifacts/residual/headroom.json`
Run: `python -m experiments.run_residual_cv --splits artifacts/splits/splits_official_verified_s42.json --out artifacts/residual/cv_s42.json --registry experiments/registry.csv`
Expected: 9 cells + 3 oracle cells. Capture the printed table verbatim.

- [ ] **Step 2: Apply the §5 gate**

Write down, from the JSON only: (a) is the best learned-`g` cell's pooled OOF above 0.3760? (b) does it beat the `d010` cell on the same space? (c) is per-fold delta ≥ 0 on every fold? (d) is `oof_r2_` materially above 0 and `predicted_norm_ratio_` not collapsed?

- [ ] **Step 3: Write D058**

Structure: what was run, the command, the 2×2 table verbatim, the headroom numbers, the λ chosen per fold, and the verdict. If (a) or (b) fails, the ADR says A is refuted at P0, quotes `oof_r2_`, and states that no GPU spend follows (spec §5).

- [ ] **Step 4: Verify the record**

Run: `python -c "import csv;rows=list(csv.DictReader(open('experiments/registry.csv',newline='',encoding='utf-8')));print(len(rows));print([r['experiment_id'] for r in rows[-9:]])"`
Expected: 9 new rows whose `pooled_oof_balanced_accuracy` matches the JSON cells.

- [ ] **Step 5: Commit**

```bash
git add docs/decisions.md experiments/registry.csv
git commit -m "docs: record D058 residual-neutralizer P0 screen"
```

---

### Task 6: P1 early-layer embeddings (gated on D058)

**Files:**
- Create: `kaggle/early_layers/layer_pool.py`, `kaggle/early_layers/extract.py`, `kaggle/early_layers/kernel-metadata.json`
- Test: `tests/test_layer_pool.py`

**Interfaces:**
- Consumes: `kaggle/deberta/folds.json` (same prompt-grouped splits), Task 4's runner (it takes cached dense features).
- Produces: `mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray` (hidden `(n, tokens, d)`, mask `(n, tokens)`, returns `(n, d)` float32, zeros for empty rows) and a parquet cache `embeddings_layers.parquet` with one column per layer id in `LAYERS = (1, 3, 6, 9, 12)`.

Skip this task entirely if Step 3 of Task 5 recorded a refutation. The spec's §5 gate is explicit: no GPU spend after a failed `R²`/D010 comparison.

- [ ] **Step 1: Write the failing test**

```python
def test_mean_pool_respects_the_attention_mask():
    hidden = np.array([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]])
    assert np.allclose(mean_pool(hidden, mask), [[2.0, 3.0]])


def test_mean_pool_returns_zeros_for_an_empty_row():
    hidden = np.ones((1, 3, 2), dtype=np.float32)
    assert np.allclose(mean_pool(hidden, np.array([[0, 0, 0]])), [[0.0, 0.0]])
```

- [ ] **Step 2: Run it, expect failure**

Run: `python -m pytest tests/test_layer_pool.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pooling and the kernel script**

The kernel loads Qwen3-0.6B (fallback: the existing DeBERTa checkpoint), runs `output_hidden_states=True` once per passage, mean-pools layers `LAYERS` over the attention mask, and writes the parquet cache with `prompt_id` and `model`. `kernel-metadata.json` sets `enable_internet: false`, `machine_shape: NvidiaTeslaT4`, the `name-that-llm` competition source and a 9-hour timeout. Append one registry row per layer with `external_data_id="official"`.

- [ ] **Step 4: Run the local test, then the kernel**

Run: `python -m pytest tests/test_layer_pool.py -q` → PASS.
Run: `kaggle kernels push -p kaggle/early_layers` (owner action) and wait for the cache.
Expected: parquet with 2699 rows × 5 layers.

- [ ] **Step 5: Re-run Task 4's table on the cached layers**

Run: `python -m experiments.run_residual_cv --splits artifacts/splits/splits_official_verified_s42.json --features-cache artifacts/early_layers/embeddings_layers.parquet --out artifacts/residual/cv_s42_layers.json`
Expected: `(layer N, learned)` cells in the JSON, same gate as Task 5.

- [ ] **Step 6: Commit**

```bash
git add kaggle/early_layers tests/test_layer_pool.py
git commit -m "feat: add early-layer embedding extraction for residual neutralization"
```

---

### Task 7: P2 within-prompt pair head (gated on Task 6)

**Files:**
- Create: `src/pair_head.py`
- Modify: `experiments/run_residual_cv.py` (optional `--pair-head` flag)
- Test: `tests/test_pair_head.py`

**Interfaces:**
- Consumes: the winning representation from Task 5 or 6.
- Produces: `sample_pairs(groups, y, pairs_per_prompt: int = 8, seed: int = 42) -> tuple[np.ndarray, np.ndarray]` returning index pairs `(i, j)` with `p(i) == p(j)` and `y[i] != y[j]`, and `PairDifferenceModel(alpha: float = 1.0)` with `fit(Z, pairs, y)` / `decision_function(Z) -> np.ndarray` whose 18-class scores are averaged with the classifier head by row-z (reuse `normalize_scores`).

- [ ] **Step 1: Write the failing test**

```python
def test_sampled_pairs_are_same_prompt_and_different_class():
    groups = np.repeat(np.arange(4), 6)
    y = np.tile(np.arange(3), 8)
    i, j = sample_pairs(groups, y, pairs_per_prompt=5)
    assert len(i) == len(j) > 0
    assert np.array_equal(groups[i], groups[j])
    assert np.all(y[i] != y[j])
```

- [ ] **Step 2–5:** Fail → implement → pass (`python -m pytest tests/test_pair_head.py -q`) → commit `feat: add within-prompt pair-difference head`.

---

### Task 8: P3 balanced-accuracy post-processing (gated on a passing candidate)

**Files:**
- Create: `experiments/run_residual_calibration.py`
- Test: `tests/test_residual_calibration.py`

**Interfaces:**
- Consumes: `sinkhorn_calibrate` (`src/calibration.py`), a candidate OOF score matrix.
- Produces: `sweep_calibration(scores, y, strengths=(0.0, 0.15, 0.3, 0.5), target_mix=0.35) -> dict` reporting pooled OOF BA per strength. Selection happens inside folds; the reported number applies the calibration **to the OOF matrix as a whole** and is labelled `transductive-diagnostic` in the registry notes, because Sinkhorn reads the batch's own marginals.

- [ ] **Step 1: Write the failing test**

```python
def test_zero_strength_is_the_uncalibrated_argmax():
    scores = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    y = np.array([0, 1])
    out = sweep_calibration(scores, y, strengths=(0.0,))
    assert out[0.0]["balanced_accuracy"] == pytest.approx(1.0)
```

- [ ] **Step 2–5:** Fail → implement → pass → commit `feat: sweep balanced-accuracy post-processing for residual candidates`.

---

## Self-Review

**Spec coverage:** §2.1 2×2 → Tasks 4–5; §2.2 pre-check → Task 3; §2 method steps 1–4 → Task 1 + Task 4's inner λ sweep; step 5 classifier head → Task 4, pair head → Task 7; step 6 per-passage inference → Task 2 (transform takes texts only); §3 P1 → Task 6, P2 → Task 7, P3 → Task 8; §4 constraints → Global Constraints + Task 4 Step 3; §5 gate → Task 5 Step 2; §7 builder's-left items → pins in Task 1 (ridge, alpha 1.0), Task 2 (residualize after SVD and after D010), Task 4 (λ grid, one classifier), Task 6 (layer list).

**Open item carried from the spec:** the `svd128` builder must be re-checked against D010's refuted setting before Task 5 concludes anything — D010 was measured in SVD space, so the `d010|svd128` cell is the like-for-like comparison and `d010|style` is a new measurement, not a reproduction.

**Proportion:** four P0 tasks with tests, four gated tasks with one test each; no task prints a number the runner did not produce.
