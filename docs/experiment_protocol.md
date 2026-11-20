# Experiment protocol

Frozen before the first scored experiment. Any change here is an ADR in
`docs/decisions.md` and invalidates previously recorded comparisons.

## 1. Splits

- **Primary:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  on `prompt_id` groups, falling back to `GroupKFold(5)` when a class has fewer
  than five prompt groups (`src/validation.make_splits`, method `auto`).
- **Secondary (robustness):** prompt-held-out splits such as seed 7 and the
  previously reserved seed 2027 are descriptive checks, not independent final
  evaluation: their prompts participated in earlier full-OOF development.
  Use a genuinely new labelled prompt set or a pre-specified nested grouped
  evaluation, with every selection decision made inside each outer training fold,
  for a stronger estimate (D024).
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

The validation side is only ever transformed. External-pool decontamination
must not fit a vocabulary, IDF or other learned transform on held-out passages.
Shared-prompt screening requires a provenance-backed mapping between namespaces;
zero matches between unrelated identifiers are not evidence of disjoint prompts.

**Pool screening (D026 ban, D028/D037 route).** *Per-fold* screening — selecting
training rows by their similarity or exact match to the held-out texts of that
fold — is transductive and stays **banned**. `experiments/pool_train.py` is
global-only by construction (it exposes no per-fold option) and its `--clean-ids`
input fails closed, recomputing the screen globally instead. The GPU kernel keeps
`--pool-scope fold` only as a disabled diagnostic and aborts when it is selected
(`kaggle/deberta/deberta_train.py`). Audit clean-ID lists that depend on
validation examples are diagnostic only (D026). The adopted route is
`src.pool.decontaminate_global`: screen the pool **once** against the *complete*
official train (exact text + word-TF-IDF cosine > 0.3, vectorizer fit on that same
train), then reuse one fold-independent keep-mask in every fold
(226,191 -> 226,049 clean; 0 exact, 142 near). No held-out passage selects a
training row, so this is not transductive. `cap_pool_stratified` then applies a
validation-independent compute budget (40,000/fold -> 40,001 rows after
per-class rounding). Residual risk: a differently-worded answer to an official
*validation* prompt cannot be excluded without a reviewed prompt crosswalk, which
does not exist (D024); the mitigations are fold-independence, training-only use,
prompt-grouped official folds, and the alternative-seed + nested outer gates.
See `docs/external_data_register.md`.

No transductive step is used for any reported score. `tests/test_features.py`
asserts that a token present only in validation text is absent from the fitted
vocabulary; `tests/test_pool.py` and `tests/test_deberta_pool_guards.py` cover the
global screen and the fail-closed per-fold path.

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

### 3.1 Historical batch-dependent stress tests

Hidden-test prompt multiplicity is not verified. The older k-subsampled
Hungarian and Sinkhorn scores were exploratory transductive diagnostics and do
not satisfy the hard rule against transductive steps in reported scores. The
historical CLI is retired; do not generate or report new such scores under this
protocol (D024). Full grouped OOF remains the primary independent-passage metric.

### 3.2 Holdout reserve

The seed-7 holdout is examined. The seed-2027 subset was not separately scored
when reserved, but its prompts appeared in previous full OOF experiments and
influenced earlier development; it is not an untouched end-to-end test. Report
it as a descriptive check. Before final selection, use genuinely new labelled
prompts or pre-specified nested grouped CV. Model, pool, hyperparameter and blend
selection must occur inside each outer training partition (D024).

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

The boxes above are a per-experiment template, not a progress list. The adopted
cap-40k pool-MIX candidate satisfies all seven: same registered split and
fingerprint (`artifacts/pool_cv_cap40k/official/results.json`,
`validation_independent: true`), global screen instead of per-fold selection
(D028), pooled gain positive in **all five folds** and reproduced on seed 13 and
a nested outer gate (D037/D039, `artifacts/nested_pool_blend_cap40k/`,
`artifacts/mix_baseline_verify_cap40k*/`), per-class recall and within-provider
confusions inspected (including the recorded `gemini-3.7-flash` -0.207 and
`kimi-k3` -0.093 recall harm), cold-start runtime/RSS measured on 4,000 rows
(D043, 13.77 s / 0.370 GB), all identifiers in `experiments/registry.csv`, and the
rationale written up in D028/D037.

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
