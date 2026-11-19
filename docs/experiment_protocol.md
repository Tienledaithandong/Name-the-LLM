# Experiment protocol

Frozen before the first scored experiment. Any change here is an ADR in
`docs/decisions.md` and invalidates previously recorded comparisons.

## Independent holdout protocol (new; historical scores are exploratory)

Historical secondary holdouts were drawn after CV/selection on the full dataset;
their prompts may have influenced selection. They are not independent. Reserve
prompts **before** any experiments, keeping both CSVs private and ignored:

```bash
python -m src.cli reserve-holdout --data-dir data --out data/reserved --holdout-groups 25 --seed 7
mkdir -p data/selection
cp data/reserved/selection.csv data/selection/train.csv
cp data/test.csv data/selection/test.csv
```

Perform *all* CV, tuning and blend selection using `data/selection` only. Fit
final models on that selection subset and save their provenance-bearing weights.
Then, once choices are frozen, score the reserved rows with:

```bash
python -m src.cli score-reserved --selection data/reserved/selection.csv \
  --holdout data/reserved/holdout.csv --final-dir artifacts/final/official \
  --weights artifacts/cv/official/ensemble/weights.json --config configs/submission.yaml
```

The command checks prompt/ID disjointness and selection artifact provenance;
it cannot prove the user never inspected or used the holdout. Do not retune on
its result. Reserve a new independent dataset for subsequent selection cycles.
The old `stability` command remains exploratory; alternate seeds now evaluate
fixed weights rather than reselecting them.

## 1. Splits

- **Primary:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  on `prompt_id` groups, falling back to `GroupKFold(5)` when a class has fewer
  than five prompt groups (`src/validation.make_splits`, method `auto`).
- **Historical secondary (exploratory):** `make_holdout` partitions the same
  data used for selection; it is not an independent holdout.
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

The validation side is only ever transformed for standard model fitting.
Crossed-design research scores explicitly use transductive grouping of validation
passages, and are reported separately from standard grouped-CV scores. `tests/test_features.py` asserts that a token present only
in validation text is absent from the fitted vocabulary.

## 3. Metric and reporting

- **Primary:** pooled out-of-fold balanced accuracy over all validation rows
  (one prediction per row across the five folds).
- **Stability:** mean ± std of the five per-fold balanced accuracies.
- **Diagnostics:** macro F1 (not the competition metric), per-class recall,
  confusion matrix, within-provider confusion share, fit/inference seconds,
  peak RSS.
- Classes absent from a validation fold are skipped with the absence reported
  (`metrics.evaluate`), never scored as zero.
- Public leaderboard scores are recorded separately in the registry and are
  never used as a selection signal.

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
