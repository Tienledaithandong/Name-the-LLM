# Experiment protocol

Frozen before the first scored experiment. Any change here is an ADR in
`docs/decisions.md` and invalidates previously recorded comparisons.

## 1. Splits

- **Primary:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  on `prompt_id` groups, falling back to `GroupKFold(5)` when a class has fewer
  than five prompt groups (`src/validation.make_splits`, method `auto`).
- **Secondary (robustness):** prompt-held-out split with a different seed
  (`make_holdout`). Used only for a final check, never for selection. The seed-7
  holdout has been **examined** repeatedly (D015–D017) and is no longer
  "untouched"; a **fresh** holdout (seed 2027, 25 prompts / 450 rows) is reserved
  for final selection and is not scored until then
  (`experiments/stress_k.py --reserve-fresh-holdout`).
- **Repeatability:** seed checks with 2–3 alternative seeds are reported as a
  stability check, not used to pick the winner. Implemented by
  `src/stability.py` (`python -m src.cli stability`, `make stability`), which
  re-runs grouped CV for the blend members under alternative seeds and scores
  the fixed adopted weights on the secondary holdout (`make_holdout`).
- **Persistence:** `artifacts/splits/splits_<dataset>.json` stores exact indices,
  method, seed, row count and the dataset fingerprint. Loading a split whose
  fingerprint differs from the current data raises; stale indices are never
  silently reused.
- **Invariant:** zero prompt overlap between train and validation in every fold,
  every row validated exactly once, no row in both sides. Enforced by
  `validate_splits` and covered by `tests/test_validation.py`.

Grouped folds are mandatory because hidden test passages come from prompts
absent from training. Row-wise random folds would measure prompt memorisation.

## 2. Fit hygiene

For each fold, and only on the training side, we fit:

1. TF-IDF vocabularies and IDF weights (char, word, `char_wb`),
2. the style-block standardiser,
3. the truncated SVD basis (when used),
4. the nuisance-suppression projection (training prompts only),
5. the classifier and its class weights.

The validation side is only ever transformed. No transductive step is used for
any reported score. `tests/test_features.py` asserts that a token present only
in validation text is absent from the fitted vocabulary.

## 3. Metric and reporting

- **Primary:** pooled out-of-fold balanced accuracy over all validation rows
  (one prediction per row across the five folds). This is the **main comparison
  for independent-passage models** (argmax blend, DeBERTa, GBDT, ...): a row's
  argmax prediction does not depend on how many sibling passages are scored
  beside it, so the full OOF is the most statistically efficient estimate.
- **Stability:** mean ± std of the five per-fold balanced accuracies.
- **Diagnostics:** macro F1 (not the competition metric), per-class recall,
  confusion matrix, within-provider confusion share, fit/inference seconds,
  peak RSS.
- Classes absent from a validation fold are skipped with the absence reported
  (`metrics.evaluate`), never scored as zero.
- Public leaderboard scores are recorded separately in the registry and are
  never used as a selection signal.

### 3.1 k-subsampled stress test (batch-dependent methods only)

The hidden test is believed to have ~1–3 passages per prompt (k≈2), not the train
set's k=18 (D017). Transductive post-processing (recovered-group Hungarian,
damped Sinkhorn) can exploit batch structure the test does not have. We therefore
keep the full grouped OOF as the primary metric and add a **stress test**, not a
second selection score:

- Subsample the pooled OOF to k = 1, 2, 3, 18 passages per prompt, repeated over
  ≥10 subsample seeds (`experiments/stress_k.py`).
- Report **paired differences** (method − argmax) on the *same* subsampled rows:
  mean ± sample std and a 95% Student-t CI across seeds (`metrics.paired_difference`,
  `metrics.mean_ci`). Report the argmax BA ± std too. Never report only the best
  sampled score.
- Settings are **fixed and pre-registered** (e.g. Sinkhorn strength 0.3 / 0.6,
  target_mix 0.35); we do **not** pick a best-of grid on the stress test — that
  would be tuning on it.
- This is **inference-only**. Transductive methods are never part of the primary
  reported CV score (hard rule: no transductive steps in reported scores). A
  batch method is adopted only if its paired-difference CI excludes 0 at test-like
  k on both the CV and the (examined) holdout, and then only on top of a strong
  ensemble (Phase 4).

### 3.2 Holdout reserve

The seed-7 holdout is **examined** (used in D015–D017) and may be optimistic.
Before final selection we either score the **fresh reserved** seed-2027 holdout
once, or use nested grouped evaluation. The fresh holdout is persisted but not
scored until then; examining it early burns the reserve.

## 4. Baselines and ablation order

1. majority class, stratified random (sanity only),
2. char TF-IDF + LinearSVC,
3. word TF-IDF + LinearSVC,
4. `char_wb` TF-IDF + LinearSVC,
5. char/word concatenation,
6. style-only (logistic regression on standardised stylometry),
7. char + style, word + char,
8. SVD reductions, nuisance suppression, optional frozen embeddings,
9. blend of complementary models.

## 5. Acceptance criteria for replacing the current best

All must hold; otherwise the result is logged as a negative result.

- [ ] exact registered split and metric (same `split_id`, same fingerprint),
- [ ] no leakage: split validation passes, no hidden-test tuning, no unregistered
      external data,
- [ ] pooled OOF improves, and the gain is not produced by a single fold
      (inspect per-fold scores as well as the mean),
- [ ] per-class recalls and within-provider confusions inspected,
- [ ] runtime, peak RSS and notebook compatibility measured,
- [ ] config, seed, git commit and artifact paths recorded in
      `experiments/registry.csv`,
- [ ] a written explanation of *why* it might work and *what evidence* supports it.

A blend must beat the best single model on pooled OOF by at least the configured
`min_gain` before it is used for a submission.

## 6. Registry columns

`experiment_id, date, git_commit, method, features, seed, split_id, fold_scores,
pooled_oof_balanced_accuracy, macro_f1, fit_seconds, infer_seconds, peak_ram_gb,
external_data_id, artifact_path, public_lb_score, notes, status`

Rows from the synthetic fixture are marked `external_data_id=synthetic-fixture`
and `notes` mentions the fixture; they are never competition results.

## 7. Known threats to validity

- 2,699 training rows and 150 prompts: per-fold validation is ~20% of rows, so
  per-fold noise is substantial; use pooled OOF plus fold spread.
- Prompt-topic shift between train and hidden test is the dominant failure mode;
  CV can underestimate it.
- Synthetic fixture results only prove the pipeline runs.
