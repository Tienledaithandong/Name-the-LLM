# Name That LLM — End-to-End Agent Execution Plan

**Project:** Kaggle Name That LLM (18-class text-to-model attribution)  
**Start:** 2026-11-14  
**Target final submission:** 2027-01-13 (verify the official deadline and timezone)  
**Primary objective:** Maximize **private-leaderboard generalization** while creating a reproducible, research-quality public portfolio.  
**Resources:** CPU-first, low RAM, optional limited Kaggle GPU; heavy use of AI coding agents.  
**Owner's role:** Approve rules-sensitive decisions, review experimental evidence, submit final entry.

> **Agent instruction:** Execute this document sequentially. Do not ask the owner to write routine code. Stop for explicit approval only when required by platform rules, data privacy, spending, or competition submission/account actions. Never fabricate scores, results, citations, or completed experiments. Treat all values below as hypotheses until verified.

> **Next score experiments, 2026-11-18.** Follow the
> [official-data score improvement plan](plans/name-that-llm-score-improvement.md).
> The first unblocked modeling task is a CPU provider specialist. Kaggle
> submission 58502550 reproduced **0.356**. The earlier Open Pool promotion
> path remains outside the current scope.

> **Status reconciled 2026-11-17:** checked items below record individually
> evidenced work, including the historical first submission (notebook v2,
> submission 58344764, public LB **0.356**); they do not imply that a section exit
> gate or a new candidate is ready. Each tick carries its evidence pointer.
> Current validation, pool restrictions, and measured results are in
> `docs/improvement_plan.md`, `docs/experiment_protocol.md`, and ADRs D001-D049.
>
> **Still open, deliberately unticked:** the Discussion-page sweep and the
> unverified facts in `docs/competition_spec.md` §10 (§0); the same-prompt
> crosswalk for the external pool (§6, `docs/external_data_register.md`); the
> optional GPU embedding branch (§6); and the owner-only upload / Kaggle rerun /
> submission of the validated cap-40k pool-MIX candidate (**0.4427** s42 /
> **0.4304** s13 / **0.4426** nested — CV, not leaderboard) plus the D038
> DeBERTa-pool GPU run. Instructions: `docs/submission_checklist.md`.

## 0. Non-negotiable rules and verification gate

Before writing modeling code:

- [x] Open and read the current official competition Overview, Data, Evaluation, Rules, Code Requirements, Timeline, and Discussion pages: https://www.kaggle.com/competitions/name-that-llm
  **Evidence:** Overview, Data, Evaluation, Rules, Code Requirements and
  Timeline read directly 2026-11-14; **Discussion swept 2026-11-18** in the
  in-app browser and recorded with per-claim URLs in
  `docs/competition_spec.md` §11 (threads 758516, 759102, 749447, 757855,
  758998, 759103). Spec §10's k and GPU-quota bullets **stay open**: the
  direct k question (758516) has 0 replies and no host answered any thread
  read. The sweep independently corroborates the D017 Hungarian withdrawal
  (+0.10 CV / +0.002 LB reported by another solo pipeline) and surfaces two
  new testable hypotheses (ModernBERT backbone; hard typographic family
  vetoes) — ADR D052. All neighbourhood scores are self-reported community
  claims, kept out of `experiments/registry.csv`. **Update 2026-11-18: the
  typographic-veto hypothesis (H9) has been tested on CPU and rejected** —
  +0.000741 with a paired 95% CI of [-0.001316, +0.002797] that includes 0
  (ADR D053, `experiments/typo_vetoes.py`). The ModernBERT hypothesis (H8)
  still needs a GPU and is owner-side.
- [x] Record the exact metric, label names, file formats, dataset paths, submission schema, runtime/memory limits, internet setting, GPU allowances, external-data/model rules, pretrained weights rules, collaboration/team rules, AI-assistant/API restrictions, and deadlines in `docs/competition_spec.md` with URLs and retrieval dates.
  **Evidence:** `docs/competition_spec.md` §1-§9 (metric §2, 18 labels §3,
  files/paths §4, schema §5, runtime + internet §6, team/external-data rules
  §7, deadlines §8, compliance rulings §9) with source URLs and the
  2026-11-14 retrieval date. The one unpublished sub-fact (weekly GPU quota)
  is explicitly flagged unverified in §10 rather than assumed.
- [x] Verify whether the metric is balanced accuracy, whether predictions are hard labels, and whether `submission.parquet` is required. **Do not assume earlier descriptions are authoritative.**
- [x] Inspect the actual dataset schema rather than hard-coding column names. The examples in this plan assume `id`, `text`, `model`, and `prompt_id`; adapt only after inspection.
- [x] Verify that model checkpoints and other assets can be included in the notebook or accessed through approved Kaggle datasets with internet disabled.
  **Evidence:** verified end-to-end by a real scored Kaggle run with internet
  off — notebook `tiennamlee/name-that-llm-stylometry-submission` v2,
  submission 58344764, LB 0.356 (registry row
  `20261114T134205-kaggle-submission-58344764`). Locally:
  `tests/test_offline_submission_artifact.py`, `tests/test_notebook_guards.py`,
  `artifacts/kaggle_offline_wheel_smoke`. The adopted package ships only two
  fitted bundles + `src` (no encoder, no pool parquet). Uploading the *new*
  cap-40k package is owner-only and still pending.
- [x] Do not upload competition data, private labels, test data, or private competition materials to third-party LLM APIs unless expressly authorized by the rules and the owner.
  **Evidence:** upheld. `docs/competition_spec.md` §9 ruling "Sending
  competition rows to hosted assistants: **Not allowed**"; AGENTS.md hard
  rule; only schema, configs, aggregates and logs leave the machine.
- [x] AI-assisted programming, external datasets, external model inference, and paid API calls are distinct compliance questions. Document the ruling for each separately. If unclear, ask organizers and pause that branch.
  **Evidence:** four separate rulings in `docs/competition_spec.md` §9
  (AI coding allowed; external corpora allowed under the D028 screen;
  external model inference at test time not used; paid API calls not used),
  each with its rules basis.
- [x] Never use leaked test labels, non-public solutions, unauthorized data, multiple accounts, or rule-violating inference.
  **Evidence:** upheld. Only `train.csv` plus the registered public pool are
  used; one account; no hidden-test tuning (spec §7, §9; AGENTS.md hard
  rules). `test.csv` is read for schema/inference only, never for scoring: the
  post-hoc analyses record `transductive: false`, `fits_performed` and
  `test_rows_read` in their results JSON (`artifacts/near_duplicates`,
  `prompt_groups`, `style_features`, `class_bias`), and protocol §2 plus
  `tests/test_stress_policy.py` ban transductive steps in any reported score.
- [x] Confirm whether public writeups and sharing code/results are allowed and when. Keep sensitive data private.
  **Evidence:** `docs/competition_spec.md` §9 — public writeups/code allowed
  on the competition's own forum/notebooks (Rules 6.b/2.6, OSI-approved
  licence); sensitive rows stay local. Publishing *outside* Kaggle is still
  flagged as needing a host check in spec §10.

**Exit gate:** `docs/competition_spec.md` is complete and the owner has approved any ambiguous compliance issue.

## 1. Repository and engineering setup

Create this structure:

```text
name-that-llm/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── requirements-lock.txt
├── .gitignore
├── configs/
│   ├── baseline.yaml
│   ├── features.yaml
│   ├── embeddings.yaml
│   └── ensemble.yaml
├── data/                       # ignored: raw/private competition files
├── docs/
│   ├── competition_spec.md
│   ├── data_audit.md
│   ├── experiment_protocol.md
│   ├── external_data_register.md
│   └── decisions.md
├── src/
│   ├── data.py
│   ├── validation.py
│   ├── metrics.py
│   ├── features_tfidf.py
│   ├── features_style.py
│   ├── features_embedding.py
│   ├── prompt_invariance.py
│   ├── train.py
│   ├── ensemble.py
│   ├── inference.py
│   └── submission.py
├── tests/
├── notebooks/
├── experiments/
│   ├── registry.csv
│   └── reports/
├── artifacts/                  # ignored: fitted models, OOF, test predictions
├── kaggle/
│   └── submission.ipynb
├── figures/
└── report/
    └── research_report.md
```

- [x] Initialize Git; add `.gitignore` for `data/`, `artifacts/`, secrets, raw rows, caches, and potentially restricted outputs.
- [x] Create a Python 3.11+ environment (verify compatibility). Install NumPy, pandas, SciPy, scikit-learn, pyarrow, matplotlib, seaborn optional, psutil, joblib, PyYAML, pytest, and optional CatBoost/LightGBM/transformers/torch.
  **Evidence:** Python 3.13.15; numpy 2.5.2, pandas 3.0.6, scipy 1.18.1,
  scikit-learn 1.9.1, pyarrow 26.0.1, matplotlib 3.11.1, psutil 7.2.2,
  joblib 1.6.0, PyYAML 6.0.3, pytest 9.1.1. Optional extras installed:
  catboost 1.2.10, sentence-transformers 6.3.0, torch 2.13.0,
  transformers 5.23.0. LightGBM/XGBoost/seaborn are absent and unused by any
  config (noted in `requirements.txt`); the tree family is covered by
  CatBoost (D049) and HistGB (D020).
- [x] Pin dependency versions and seeds. Implement command-line entry points for `audit`, `train`, `evaluate`, `predict`, `package`.
  **Evidence:** `requirements-lock.txt` (23 pinned runtime packages),
  `requirements.txt`, seeds fixed in every `configs/*.yaml`;
  `python -m src.cli` exposes `audit`, `splits`, `train`, `ensemble`,
  `evaluate`, `predict`, `package`, `stability`, `figures`, `demo`
  (`evaluate` added in task #48, `tests/test_cli_evaluate.py`).
- [x] Add automated tests for schema, ID alignment, group separation, label mapping, metrics, model serialization, and submission format.
- [x] Add a one-command reproducibility target (`make baseline` or equivalent).
- [x] Log CPU model, RAM, wall-clock runtime, peak memory, dependency versions, and Git commit for each experiment.
  **Evidence:** every metrics JSON carries `environment` (python, platform,
  processor, cpu_count, ram_gb, rss_gb, numpy/pandas/scipy/sklearn/pyarrow/
  yaml/joblib versions), `fit_seconds`, `infer_seconds`, `peak_ram_gb`,
  `git_commit` and `git_worktree_status`; the same fields are registry
  columns. Example: `artifacts/cv_style_catboost/metrics/style_catboost.json`.

**Exit gate:** clean setup from README succeeds on CPU and tests pass.

## 2. Data audit and evaluation design — highest priority

- [x] Load all official files; report row counts, columns, nulls, duplicates, label frequencies, text lengths, Unicode distribution, prompt counts, and number of labels per prompt.
  **Evidence:** `docs/data_audit.md` + `artifacts/audit/official/audit.json`
  (`python -m src.cli audit`): 2,699 rows, 150 prompts, 18/18 labels,
  columns/nulls, 0 duplicate ids and 0 duplicate texts, rows/labels per
  prompt (17/18/18), class support 149-150, word/char length stats,
  non-ASCII / Markdown / emoji shares, provider support, fingerprint
  `6368e9c31e8d060f`; figures in `figures/`.
- [x] Check whether identical/near-duplicate texts, shared prompt templates, or augmented variants cross folds. Never use hidden test labels.
  **Evidence:** D047 / `experiments/near_duplicates.py` ->
  `artifacts/near_duplicates/results.json`: 0 exact duplicates and **0
  cross-fold near-duplicates at every threshold up to cosine 0.9**;
  `tests/test_near_duplicates.py` (6 tests) pins the detector. Prompt
  templates cannot cross folds because `prompt_id` is the group
  (`tests/test_validation.py::test_splits_never_share_prompts`). No hidden
  test label is ever read.
- [x] Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` **if the data supports it**, with `prompt_id` as group. If not, use `GroupKFold` or a documented custom group-balanced split. Confirm **zero prompt overlap** in each fold.
- [x] Inspect per-fold class support. Balanced accuracy requires attention to classes absent from validation; avoid invalid comparisons and document scoring behavior.
- [x] Create and persist exact train/validation indices (`artifacts/splits/`) with dataset fingerprint. Do not change splits while comparing models.
- [x] For every model, fit vocabulary, scaling, feature selection, PCA, calibrators, and class-bias parameters on training folds only. Never fit transforms on validation or hidden test unless explicitly transductive, permitted, and separately labeled as such.
  **Evidence:** `docs/experiment_protocol.md` §2 lists the five per-fold
  fits; `src/features_tfidf.py:170` fits `StandardScaler(with_mean=False)`
  inside `TextMatrixBuilder.fit`; SVD, nuisance projection and class weights
  are fold-local. `tests/test_features.py` asserts a validation-only token
  is absent from the fitted vocabulary. The pool keep-mask is global and
  fold-independent (D028), never fit on held-out text.
- [x] Report mean/std of per-fold balanced accuracy, **pooled out-of-fold balanced accuracy**, per-class recall, macro F1 (diagnostic), confusion matrix, latency, peak RAM.
  **Evidence:** protocol §3; metrics JSON `fold_aggregate`
  (mean/std/min/max/fold_scores) + `pooled` (balanced_accuracy, macro_f1,
  per_class_recall, support, classes_absent); `python -m src.cli evaluate`
  adds the confusion matrix, within-provider share and length bands;
  fit/infer seconds and peak RSS are logged per run.
- [x] Keep an untouched secondary group-held-out validation scheme or alternative seeds for final robustness checks; avoid iterative tuning on that holdout. Alternative seeds are descriptive robustness checks; previously used prompts are not an untouched final test (D024).
- [x] Establish a leakage checklist and test it automatically.
  **Evidence:** protocol §2 (fit hygiene) + §5 (acceptance gate);
  automated by `tests/test_validation.py` (prompt-overlap detector, full
  coverage, fingerprint guard, holdout disjointness),
  `tests/test_oof_identity.py`, `tests/test_ensemble_identity.py`,
  `tests/test_stress_policy.py`, `tests/test_nested_validation.py`,
  `tests/test_deberta_pool_guards.py`.

**Exit gate:** `docs/experiment_protocol.md` and a reusable validation harness exist before tuning.

## 3. First working baseline and first valid submission

Implement, evaluate, and save these CPU baselines in order:

1. Majority-class and stratified-random reference (sanity only).
2. Character TF-IDF (2–5 grams, capped vocabulary, float32) + LinearSVC (`class_weight='balanced'`).
3. Word TF-IDF (1–2 grams) + LinearSVC.
4. Character `char_wb` TF-IDF + LogisticRegression or LinearSVC.
5. Optional word/character feature concatenation, with sparse matrices and controlled RAM.

For each:

- [x] Run five fixed folds and save OOF predictions and decision scores, class ordering, model config, fit time, inference time, RAM, and metrics.
  **Evidence:** `artifacts/cv*/official/oof/*.npz` store `scores`, `y` and
  `labels` (the 18-label order) for every base model; the matching metrics
  JSON stores config, class order, fit/infer seconds, peak RAM and
  pooled/per-fold metrics, and `artifacts/splits/` persists the exact fold
  indices. The identity-verified re-runs (`artifacts/cv_identity`,
  `artifacts/pool_cv_cap40k`) additionally store `validation_fold_ids` and
  `ordered_dataset_sha256` (D024). 118 registry rows.
- [x] Check performance by class, model family, text length, and prompt group.
  **Evidence:** per-class recall + within-provider confusion
  (`cli evaluate`, report §9); length bands D041
  (`artifacts/length_sensitivity/results.json`: 0.2491 -> 0.4718 official,
  0.3125 -> 0.5858 pool-MIX); **per-prompt-group** error concentration D048
  (`artifacts/prompt_groups/results.json`, binomial null — errors are spread
  at chance level, so the pool gain is broad).
- [x] Refit the selected model on all permitted labeled training data; preserve exact preprocessing and class order.
  **Evidence:** `python -m src.cli predict --config configs/submission.yaml
  --fit` -> `artifacts/final/official`; `ModelBundle` persists the fitted
  `TextMatrixBuilder` and the 18-label order, and `ModelBundle.load` rejects
  a mismatch (`tests/test_model.py`, `tests/test_inference_fit_pool.py`).
- [x] Produce a valid submission using the **official** schema and filenames; verify row count, ID order, unique IDs, valid labels, and no nulls.
- [x] Test the Kaggle notebook with internet disabled and a clean environment; use only approved attached artifacts.
  **Evidence:** the first submission ran on Kaggle with internet disabled
  from attached assets only (notebook v2, submission 58344764, LB 0.356).
  Locally re-verified by `tests/test_notebook_guards.py`,
  `tests/test_offline_submission_artifact.py`, `tests/test_package.py` and
  the offline wheel smoke (`artifacts/kaggle_offline_wheel_smoke`). The
  cap-40k candidate's own Kaggle rerun is owner-only and pending.
- [x] Submit the first valid version if the owner has authorized the submission and platform actions are available. Otherwise provide exact instructions and ready-to-run artifact.
- [x] Record public leaderboard score separately from CV; never optimize solely against public LB.

**Exit gate:** first reproducible CPU-only notebook submission, or a verified submission-ready artifact if account submission is unavailable.

## 4. Stylometric feature engineering

Create `src/features_style.py` with unit-tested, deterministic extraction of:

- Length: words, characters, sentences, paragraphs, average/variance of sentence length.
- Punctuation: comma, semicolon, colon, parentheses, em/en dash, hyphen, quote styles, ellipsis, exclamation/question marks.
- Formatting: whitespace, newline, bullet/numbered lists, Markdown, code fences, digits, Unicode markers.
- Lexical: type-token ratio (with length normalization), function-word frequencies, contraction rates, common discourse markers.
- Syntactic proxies: average token length, capitalization, sentence-start patterns, hedging phrases, transition phrases.

Then:

- [x] Evaluate style-only logistic regression, SVM, and small CatBoost/LightGBM (if permitted).
  **Evidence:** four classifier families measured on the same folds —
  logreg **0.290398** (adopted), LinearSVC 0.284449 (D045), CatBoost
  0.283003 (D049), HistGB 0.263015 (D020). Paired CIs vs logreg span 0 for
  both alternatives; LightGBM is not installed, CatBoost covers the tree
  arm. Configs `configs/style_svm.yaml`, `configs/style_catboost.yaml`.
- [x] Fit imputers/scalers only inside each training fold.
  **Evidence:** `src/features_tfidf.py:163-170` — the style block's
  `StandardScaler(with_mean=False)` is fit inside each training fold and
  only transforms validation. Style extraction is deterministic and emits
  no missing values, so no imputer is required;
  `tests/test_features.py::test_style_features_are_deterministic` and
  `::test_tfidf_builder_is_fitted_on_train_only` pin both properties.
- [x] Compare style features with character TF-IDF through score blending or feature concatenation.
  **Evidence:** both routes measured on the same official folds (seed 42) —
  feature concatenation `char_plus_style` **0.224462** (worse than either
  view alone) vs score blending `0.5 z(char) + 0.5 z(style)` =
  **0.375971**, the submitted LB 0.356 fallback. Blending is the adopted
  route. Report §7-§8; registry ablation rows; `artifacts/cv/official`.
- [x] Make per-feature and per-class analyses without claiming causal attribution.
  **Evidence:** D046 / `experiments/style_features.py` ->
  `artifacts/style_features/results.json` (56 features, per-feature
  coefficients, `refit_reproduces_saved_oof`, max OOF drift); per-class
  recall in every metrics JSON and `figures/per_class_recall*.png`.
  D046 states explicitly these are associations, **not causal attribution**.
- [x] Save ablations: word only, char only, style only, char+style, word+char.
  **Evidence:** all five saved as registry rows and OOF artifacts —
  `word_only`, `char_only`, `style_only`, `char_plus_style`, `word_plus_char`
  (plus `char_wb`, SVD and nuisance variants). Report §8 ablation table and
  `figures/ablation.png`; the candidate's own five-arm ablation is
  `figures/ablation_pool_mix_cap40k.png` (D051).

**Exit gate:** a controlled ablation report, even if no improvement.

## 5. Research track: prompt-invariant LLM fingerprints

**Research question:** Can prompt-robust stylometric features improve exact-model attribution on previously unseen prompts without training a large language model?

Hypotheses to test, not assume:

- H1: Character/typography features generalize better across prompts than semantic word features.
- H2: Prompt-invariance techniques improve grouped validation.
- H3: Frozen encoder representations complement sparse stylometry.
- H4: Distinguishing model versions within a provider is harder than distinguishing providers.

Experiments:

- [x] Separate content-heavy features from style-heavy features; compare both on grouped CV.
  **Evidence:** word TF-IDF (content) vs char TF-IDF / 56 stylometric
  features (style) compared on the same official grouped folds (seed 42):
  `word_only` **0.269652**, `char_only` **0.316711**, `style_only`
  **0.290398** — both typography/style views beat the content view on
  unseen prompts, supporting H1 (report §6, §8; `artifacts/cv/official`).
- [x] Explore training-prompt mean-centering `z_i - mean(z | prompt_i)` as a **diagnostic** only; hidden test prompt IDs/centroids are not available.
  **Evidence:** `src/prompt_invariance.prompt_centering_diagnostic` is
  documented and tested as **diagnostic only** (it needs test-time prompt
  ids, which the hidden test does not provide); `phase_a_*` registry rows
  are labelled oracle/upper-bound, never submission candidates.
- [x] Design deployable alternatives that do not require hidden test prompt labels: style-only projection, learned nuisance suppression on training groups, approximate clustering with fit-on-train-only transforms, or robust feature selection.
  **Evidence:** `NuisanceSuppressor` — a projection fit **only on training
  prompts** in SVD space, needing no test-time prompt labels
  (`char_svd256_nuisance8`, `style_svd64_nuisance8`); style-only projection
  (`style_only`) is the deployed member. `tests/test_prompt_invariance.py`
  covers determinism, shape guards and the identity fallback when prompts
  are not repeated.
- [x] Guard against transductive leakage and unbalanced clusters. Explicitly test the failure of centroid methods when clusters contain unrelated prompts.
  **Evidence:** `tests/test_grouping.py` explicitly tests degraded recovery
  for **merged clusters** and **split prompts**, purity/row-mismatch guards
  and the exactly-one-stop-criterion rule; `tests/test_stress_policy.py`
  bans transductive scores. The crossed-Hungarian candidate was
  **withdrawn** (D017/D018) once its k=18 gain collapsed at k~2.
- [x] Measure performance on held-out prompts, short/long passages, and within-provider confusions.
  **Evidence:** every fold is prompt-held-out by construction; short/long
  passage bands in D041 (`artifacts/length_sensitivity`); within-provider
  vs cross-provider error split in `cli evaluate` and report §9 (H4
  supported: sibling versions are the dominant confusion).
- [x] Require improvements across multiple group splits/seeds, not one lucky fold.
  **Evidence:** the adopted candidate is gated on seed 42 (**0.4427**),
  seed 13 (**0.4304**) and a nested outer gate (**0.4426**) — D037/D039,
  `artifacts/pool_cv_cap40k_s13`, `artifacts/nested_pool_blend_cap40k`,
  independent reproduction `artifacts/mix_baseline_verify_cap40k*/`.
  Single-seed winners were rejected (D017, D035).
- [x] Save negative results and methodological caveats.
  **Evidence:** report negative-results table + README; ADRs D013
  (embeddings not adopted), D017/D018 (crossed withdrawn), D020 (GBDT),
  D027 (official-only DeBERTa adds nothing), D044 (class biases do not
  transfer), D045 (SVM), D049 (CatBoost); protocol §7 threats to validity;
  pool harm recorded (style -0.0174, `gemini-3.7-flash` recall -0.207).

**Exit gate:** at least one clearly specified method with a rigorous ablation and reproducible evaluation, whether positive or negative.

## 6. Public external data and small frozen embeddings

Before using external material:

- [x] Check official rules for allowed external data, checkpoints, and distribution.
  **Evidence:** `docs/competition_spec.md` §9 (external-corpora row,
  pretrained-encoder row) + the rules basis paragraph in
  `docs/external_data_register.md` (Code Requirements; Rules §2.6
  public/free or "reasonably accessible"; nothing downloaded at runtime).
- [x] Record each source URL, license, access conditions, collection date, model label mapping, size, and known distribution differences in `docs/external_data_register.md`.
  **Evidence:** `docs/external_data_register.md` (reconciled 2026-11-17,
  **encoder gap closed 2026-11-18**) records URL, licence, access date, what it
  is, label mapping, size, dedup, where used and the validation result for
  **five** rows: `open-pool-v15`, `minilm-l6-v2`, `deberta-v3-base` (MIT),
  `deberta-v3-large` (MIT) and `modernbert-large` (Apache-2.0, **planned, never
  downloaded**). The two DeBERTa checkpoints had been fine-tuned (D023/D027)
  without a register row; that omission is fixed, with card-verified parameter
  counts and the measured GPU peaks. Local sha256 table covers the pool parquet
  (112,851,698 B) and MiniLM (90,868,376 B) only — **no encoder checkpoint is
  retained locally**, so the encoder sizes are model-card parameter counts, not
  file bytes.
- [x] Use only publicly available and authorized data; no private leaderboard labels or restricted model outputs.
  **Evidence:** Open Pool v15 is a public Kaggle dataset (CC BY-SA 4.0,
  65 sources CC BY/CC BY-SA/CC0); MiniLM is a public HF checkpoint
  (Apache-2.0). No private leaderboard labels, no restricted model outputs,
  no paid API. Share-alike obligations recorded in the register.
- [ ] Deduplicate by exact text and near-duplicate similarity against official training and validation; do not introduce same-prompt contamination across folds.
  **Partly done, deliberately open.** Exact + near-duplicate screening is
  verified and **fold-independent**: `src.pool.decontaminate_global` against
  the full official train (which contains every validation fold's rows)
  removes 0 exact + 142 near at cosine > 0.3, keeping 226,049 (D028);
  official train itself has 0 cross-fold near-duplicates up to 0.9 (D047).
  **Same-prompt contamination is not fully excluded**: no reviewed
  official-prompt -> pool-prompt crosswalk exists (D024), so a
  differently-worded answer to a validation prompt cannot be ruled out.
  The register's admission checklist keeps that one item unticked and
  states the residual risk and its mitigations. **This is upstream-blocked,
  not compute-blocked** (schema verified 2026-11-18): `data/train.csv`
  carries only `id, prompt_id, text, model` and `test.csv` only `id, text`,
  so no official prompt text exists locally to match against — see
  `docs/external_data_register.md` "Prompt-crosswalk requirement (D024)".
- [x] Verify that external model identities map to the 18 official labels; do not silently merge versions.
  **Evidence:** `per_model.csv` matches all **18/18** official labels
  exactly; `src.pool.load_pool` rejects an unknown or missing label and
  versions are never merged (`tests/test_pool.py`,
  `tests/test_pool_class_audit.py`). Uneven support (3,541-23,474) is
  handled by class-proportional capping, not by relabelling.
- [x] Compare official-only versus official+external with the **same** official grouped validation. Track whether external data harms domain transfer.
  **Evidence:** same official grouped folds, same fixed blend rule —
  char 0.3193 -> **0.4104** (paired **+0.0912 [+0.0721, +0.1103]**, all five
  folds positive); MIX **0.4427/0.4304/0.4426** vs control
  0.3760/0.3582/0.3519. **Harm is recorded too:** style+pool 0.2904 ->
  0.2730 (paired -0.0174 [-0.0480, +0.0132]), so style stays official-only;
  `gemini-3.7-flash` recall -0.207 and `kimi-k3` -0.093 under the pool.
  `artifacts/pool_cv_cap40k/official/results.json`, D028/D037.

Frozen embeddings:

- [x] Choose a small publicly available encoder that fits CPU RAM and competition inference constraints; verify license and checkpoint availability.
  **Evidence:** `all-MiniLM-L6-v2` (Apache-2.0, 384-dim, ~90 MB single
  `model.safetensors`) — public, CPU-resident, checksum pinned in
  `docs/external_data_register.md`; attached offline via `package --crossed`
  and loaded with `HF_HUB_OFFLINE=1` (research-only path).
- [x] Batch encode text, cache float16/float32 embeddings, and train regularized linear classifiers on top.
  **Evidence:** `src/features_embedding.py` batches and caches embeddings on
  disk as **float16** (`encode_cached`, `_cache_path`) so folds reuse them;
  `configs/frozen_encoder.yaml` trains a regularized linear classifier on
  top; `tests/test_features_embedding.py` is hermetic (deterministic hashing
  encoder, no download).
- [x] Compare CPU inference time and memory with sparse TF-IDF. Do not adopt embeddings unless they improve grouped OOF or meaningful diversity.
  **Evidence:** D013 — embedding-only pooled OOF **0.1278**, +style 0.3093;
  the greedy char+emb_style blend's +0.0026 CV gain **reversed on the
  holdout** (0.3933 vs 0.3956), so embeddings were **not adopted**. Fit/infer
  seconds and peak RSS are logged for every arm in the registry; the sparse
  candidate's cold start is 13.77 s / 0.370 GB on 4,000 rows (D043).
- [ ] Optional GPU: use Kaggle's permitted GPU for embedding extraction or small encoder adaptation only if measured gains justify it.
  **Optional, not exercised for embeddings.** No measured embedding gain
  justified GPU (D013), so none was used. The one GPU lever that remains is
  the DeBERTa-v3 kernel, re-wired onto the non-transductive global pool path
  at cap 40k (D038, `kaggle/deberta`): official-only DeBERTa added **no**
  diversity gain (D027) and the pool-arm GPU run is **owner-side and has no
  metric yet**. It is adopted only if it beats the MIX baseline with a
  paired CI excluding 0 under all three protocols. Since D054 the kernel's
  backbone is a module-level `BACKBONE` preset (a Kaggle **script** kernel takes
  no CLI args), so the queued **H8 ModernBERT-large** probe — community-reported
  ≈0.55 vs their DeBERTa 0.50 (D052), which makes D027's encoder negative
  backbone-specific rather than general — is runnable by the owner with one
  constant change; `transformers>=4.48` is required and 384/4/1e-5 is held fixed
  for both presets so the probe changes one variable.

**Exit gate:** documented data provenance and measured comparison, no unverified external resources in final notebook.

## 7. Ensembling and metric-aware decisions

- [x] Save fold-aligned OOF decision scores for all base models and class orders.
  **Evidence:** every base model's OOF npz stores `scores` plus `labels`
  (the 18-label order) and `y`, aligned to the persisted fold indices in
  `artifacts/splits/`. The identity-verified runs used for selection
  (`artifacts/cv_identity`, `artifacts/pool_cv_cap40k`) additionally store
  `validation_fold_ids` and `ordered_dataset_sha256`
  `dc60875fd8b536cb8db67bdac8119132c6d0c0cf667fbca262d5ffc47f3e6e5a`;
  `tests/test_oof_identity.py` and `tests/test_ensemble_identity.py` reject
  misaligned or foreign OOF (D024).
- [x] Standardize/convert scores in a controlled way. LinearSVC margins are **not probabilities**; avoid treating them as calibrated probabilities.
  **Evidence:** `src.ensemble.normalize_scores` does row-wise z-scoring with
  a -1e8 sentinel for absent classes; LinearSVC margins are **never** treated
  as probabilities (AGENTS.md hard rule). The CatBoost arm uses
  `log(clip(predict_proba, 1e-12, 1))` because it has no `decision_function`
  (D049); `tests/test_blend_rules.py` pins the conversions.
- [x] Compare simple averaging, rank-based blending, normalized-margin blending, and calibrated probabilities where validation supports it.
  **Evidence:** D042 / `experiments/blend_rules.py` ->
  `artifacts/blend_rules/results.json` compares raw mean, rank, min-max,
  softmax and the adopted row-z on **both** blends, same OOF rows, paired
  fold deltas: row-z stands, min-max indistinguishable. Probability
  calibration was tested separately (damped Sinkhorn, `tests/
  test_calibration.py`) and withdrawn with the crossed candidate (D017).
- [x] Tune ensemble weights on OOF with regularization or nested validation; avoid overfitting 2,699 examples with dozens of parameters.
  **Evidence:** the adopted rule is a **fixed** 0.5/0.5 row-z blend (two
  parameters, not dozens); greedy search is regularized by `min_gain` and
  every selection decision for the candidate was made inside the outer
  training folds of a pre-specified nested grouped evaluation
  (`experiments/nested_pool_blend.py`, `artifacts/nested_pool_blend_cap40k`,
  D035/D039).
- [x] Evaluate class-specific intercepts/biases for balanced accuracy, with out-of-sample tuning and strict separation.
  **Evidence:** D044 / `experiments/class_bias.py` ->
  `artifacts/class_bias/results.json`: inner-OOF coordinate ascent over a
  bias grid with shrinkage, nested inside outer folds, multiple seeds,
  strict separation recorded. The biases **do not transfer** — the blend
  keeps zero intercepts. `tests/test_class_bias.py`.
- [x] Select a small ensemble of genuinely complementary models; prioritize stability, CPU speed, and reproducibility.
  **Evidence:** two members only (char TF-IDF logreg + style logreg).
  Complementarity is measured, not assumed: GBDT (D020), DeBERTa (D027),
  SVM (D045) and CatBoost (D049, row-z corr 0.8678, swap blend -0.0107) all
  failed to add gain. **Against the pool-trained candidate** the question was
  re-opened and re-measured (D055, `experiments/blend_members.py` ->
  `artifacts/blend_members/results.json`): six identity-verified third members
  at pre-specified equal thirds and 0.4/0.4/0.2 **all lose** (best
  `word_tfidf_linearsvc` 0.435240 vs base 0.442675, paired -0.007471
  [-0.022787, +0.007845]); the most diverse arm is the weakest, so diversity
  without strength does not pay. Stability via `cli stability`; CPU cold start
  13.77 s / 0.370 GB (D043); fully reproducible from pinned configs.
- [x] Perform 2–3 split-seed checks and inspect public/private distribution-shift risks.
  **Evidence:** seed 42 + seed 13 + nested outer gate (D037/D039);
  `src/stability.py` re-runs members under alternative seeds and scores the
  fixed weights on the secondary holdout. Public/private shift risk is
  analysed in `docs/strategy_review.md` and report §10 (public ~40% /
  private ~60%, different model mix; hidden-test k unverified, so the
  k-robust argmax path is used and the k=18 crossed gain was withdrawn).

**Exit gate:** chosen ensemble has a reproducible grouped-CV advantage and fits Kaggle runtime/memory constraints.

## 8. Kaggle submission engineering

- [x] Make `kaggle/submission.ipynb` fully self-contained with approved Kaggle input assets and pinned dependencies.
  **Evidence:** `kaggle/submission.ipynb` reads only attached inputs and
  writes `submission.parquet`; `python -m src.cli package` builds
  `artifacts/kaggle_dataset` (two fitted bundles + `weights.json` + `src` +
  manifest/metadata/README). `tests/test_candidate_package.py`,
  `tests/test_package.py`. Dependencies pinned in `requirements-lock.txt`
  (Kaggle's own sklearn 1.6.1 vs local 1.9.1 warning is recorded in the
  submission registry row).
- [x] Ensure internet is disabled; no runtime calls to ChatGPT, Claude, OpenAI APIs, Hugging Face downloads, or other network services.
  **Evidence:** `tests/test_notebook_guards.py` asserts the notebook makes
  no network/HF-download call and rejects an `inference.json` marked
  `crossed: true`; the adopted package contains **no encoder and no pool
  parquet**, so nothing needs downloading. The research-only crossed path
  loads its attached checkpoint with `HF_HUB_OFFLINE=1`.
- [x] Verify test-time preprocessing matches training; preserve exact class mapping.
  **Evidence:** inference reuses the persisted `TextMatrixBuilder` from the
  fitted bundle (same vocabulary, IDF, scaler); `ModelBundle.load` rejects
  any label-order mismatch against `src.constants.LABELS`
  (`tests/test_model.py`, `tests/test_inference_fit_pool.py`,
  `tests/test_offline_submission_artifact.py`).
- [x] Validate schema and file path against official submission requirements, not assumptions.
  **Evidence:** spec §5 (verified from the Evaluation page) —
  `/kaggle/input/competitions/name-that-llm/test.csv` in,
  `submission.parquet` in the notebook working directory, columns `id`,
  `model`; `src/submission.py` enforces it and `tests/test_submission.py`
  pins the fail-fast behaviour. The interactive `test.csv` row count (3) was
  checked against the Data page rather than assumed.
- [x] Benchmark full expected hidden-test size with conservative memory/time headroom; simulate cold start and model load.
  **Evidence:** D043 / `experiments/inference_benchmark.py` ->
  `artifacts/inference_benchmark/results.json`: **4,000 rows** (the expected
  hidden-test size), cold start including imports, bundle load, predict,
  parquet write and read-back validation — `cold_start_wall_seconds`
  **13.7687 s**, peak working set **0.3695 GB** (RSS 0.2198 GB), i.e. ~2,353x
  headroom against the 9 h CPU budget; a **8,000-row** run takes 28.24 s /
  0.5416 GB. Per-stage seconds and RSS are recorded. Rows are synthetic and
  labelled as such (`prediction_counts_note` forbids reading them as a class
  prior); no competition test row is scored locally.
- [x] Add fail-fast checks for missing files, wrong feature dimensions, unexpected labels, NaNs, and ID mismatch.
  **Evidence:** `src/submission.py` + `src/inference.py` reject missing
  files, wrong feature dimensions, labels outside the 18, nulls, duplicate
  or mismatched ids; the notebook re-reads and validates its own output.
  Covered by `tests/test_submission.py`, `tests/test_inference_benchmark.py`,
  `tests/test_offline_submission_artifact.py`.
- [x] Save a notebook version, run Kaggle's actual submission flow, and confirm success. If direct account access is unavailable, provide owner with a concise final submission checklist.
  **Evidence:** done once for the fallback — notebook v2 saved, Kaggle's
  submission flow run, success confirmed by public LB **0.356**
  (submission 58344764, commit 8cd9c69). For the validated cap-40k
  candidate the flow is **owner-only and pending**; the concise checklist
  required by the otherwise-branch is `docs/submission_checklist.md`
  (build/upload, attach, run, submit, record, revert rule).
- [x] Keep a log of submission version, Git commit, CV score, public LB score, timestamp, and changes.
  **Evidence:** registry row `20261114T134205-kaggle-submission-58344764`
  records the submission id, notebook version 2, git commit 8cd9c69 (with
  working-tree changes noted), CV 0.3759706686552324, **public LB 0.356**,
  UTC timestamp, artifact path and the changes/warnings. CV and LB are
  separate columns; `docs/submission_checklist.md` step 11 requires the
  same record for every future submission.
- [x] Avoid chasing public LB fluctuations without supporting grouped-validation evidence.
  **Evidence:** protocol §3 ("public leaderboard scores ... never used as a
  selection signal"); D017/D018 withdrew a candidate on grouped-CV evidence
  rather than LB; the checklist's revert rule treats a CV gain that fails
  to transfer as evidence about prompt shift, not a reason to resubmit.

**Exit gate:** a valid, reproducible, compliant final submission.

## 9. Research report and portfolio

Produce `report/research_report.md` with:

1. Abstract (problem, method, key measured result).
2. Introduction and motivation: model provenance and attribution.
3. Related work: stylometry, authorship attribution, LLM detection, domain generalization, and model fingerprinting; cite actual verified papers.
4. Dataset and competition rules, with privacy restrictions.
5. Experimental protocol and grouped validation.
6. Baselines and proposed prompt-invariant method.
7. Results: per-fold and pooled balanced accuracy, class recalls, confidence intervals or variation, efficiency.
8. Ablations: word, char, style, prompt-invariant, embeddings, ensemble.
9. Confusion analysis: sibling models versus model families; text-length sensitivity.
10. Failure cases, limitations, ethical considerations, and threats to validity.
11. Reproducibility appendix: seeds, environment, code, data access instructions.

Create figures: class balance, length distribution, confusion matrix, per-class recall, ablation chart, runtime/RAM tradeoff, and grouped-CV variation.

**Evidence:** all seven exist for the baseline (`figures/*.png`, `python -m src.cli figures`) **and** for the
cap-40k candidate: confusion/recall + length bands from `experiments/length_sensitivity.py` (D041), and
ablation / runtime-RAM / grouped-CV variation from `experiments/candidate_figures.py` (D051,
`figures/{ablation,runtime_ram,cv_variation}_pool_mix_cap40k.png`). The candidate panels were needed because
`make_report_figures` reads one run's metrics directory while the candidate's two members were fit in separate
runs; the panels are now shared functions. Every arm's pooled and per-fold score is recomputed from the saved
OOF and checked against the recorded value (all matched within 1e-6). The pool arm's peak fit RAM was never
recorded and is labelled "RAM n/r", not estimated (`artifacts/candidate_figures/results.json`).

Create a professional `README.md` with a concise summary, architecture diagram, install/run commands, reproducibility instructions, result table, competition link, and clearly marked final rank **only after confirmed**. Do not call a technical report a peer-reviewed paper unless accepted.

CV bullet template (replace placeholders with verified values only):

> **Kaggle Name That LLM — [verified final rank / percentile]**: Built a CPU-efficient 18-class LLM attribution system using sparse stylometry and [validated methods], achieving [verified balanced accuracy] on [defined evaluation]. Developed reproducible prompt-held-out experiments and published a technical report with ablations and error analysis.

**Exit gate:** reproducible public portfolio without disclosing restricted competition material.

## 10. Timeline and checkpoints

| Period | Focus | Required deliverable |
|---|---|---|
| Nov 14–20 | Rules, data audit, grouped CV, CPU baseline | First valid submission + baseline OOF |
| Nov 21–27 | Stylometry and sparse-model ablations | Feature report + confusion matrix |
| Nov 28–Dec 4 | Prompt-invariant research | Method prototype + controlled ablation |
| Dec 5–11 | External data audit and frozen embeddings | Provenance register + benchmark |
| Dec 12–18 | Ensembling and class-aware decisions | V1 robust ensemble |
| Dec 19–25 | Optional limited-GPU experiments | CPU/GPU efficiency comparison |
| Dec 26–Jan 1 | Stress tests and research writing | V2 candidate + report draft |
| Jan 2–8 | Final model selection, compliance audit | Locked models + clean Kaggle run |
| Jan 9–13 | Final submission and portfolio | Final submission + polished report |

**Important:** Re-check entry and team deadlines now; do not wait for the final week. Do not assume new participants can join after the official entry deadline.

## 11. Experiment acceptance criteria

An experiment may replace the current best only when:

- [x] It uses the exact registered data split and correct metric.
  **Evidence:** enforced — protocol §5; `split_id`
  `stratified_group-s42-n5` and `ordered_dataset_sha256`
  `dc60875f...e5a` are stored in every OOF/metrics file and checked by
  `tests/test_oof_identity.py`; a fingerprint mismatch raises.
- [x] No leakage, hidden-test tuning, or unauthorized external resource is detected.
  **Evidence:** enforced — `validate_splits` + the leakage tests listed in
  §2; per-fold fits (protocol §2); external data only from
  `docs/external_data_register.md`; the per-fold pool screen fails closed
  (D026) and the adopted global screen is fold-independent (D028).
- [x] The improvement is observed in pooled OOF and is not driven by one pathological fold.
  **Evidence:** the adopted candidate's paired gain is positive in **all
  five folds** (+0.0912 [+0.0721, +0.1103] for char), and survives seed 13
  and the nested outer gate (D037/D039). Single-fold winners are logged as
  negative results (D017, D035, D044).
- [x] Per-class recalls and within-provider errors are examined.
  **Evidence:** `pooled.per_class_recall` in every metrics JSON; `cli
  evaluate` confusion matrix + within-provider share; report §9. Harm is
  recorded, not hidden (`gemini-3.7-flash` -0.207, `kimi-k3` -0.093 under
  the pool; D037/D048).
- [x] Runtime, peak RAM, and notebook compatibility are measured.
  **Evidence:** `fit_seconds`, `infer_seconds`, `peak_ram_gb` per run;
  full-size cold start 13.77 s / 0.370 GB on 4,000 rows (D043); offline
  package + notebook guard tests; the scored Kaggle run confirms
  compatibility for the fallback.
- [x] All config/seed/commit/artifact identifiers are logged.
  **Evidence:** `experiments/registry.csv` — 118 rows x the 18 columns of
  protocol §6, including `git_commit`, `seed`, `split_id`, `artifact_path`,
  `external_data_id` and `status`; metrics JSONs add `git_worktree_status`
  and the full `environment` block.
- [x] The agent explains *why* the method might work and *what evidence* supports it.
  **Evidence:** every ADR D001-D049 in `docs/decisions.md` states the
  mechanism, the measured evidence and the consequences; report §6 gives
  the H1-H4 rationale and §10 the threats to validity. Negative results
  carry the same explanation (D013, D017, D020, D027, D044, D045, D049).

No fixed score threshold is guaranteed. Prioritize validated gains and final-rerun reliability over arbitrary target scores.

## 12. Agent execution protocol

At the beginning of every work session:

1. Read `AGENTS.md`, `docs/competition_spec.md`, `docs/experiment_protocol.md`, `docs/decisions.md`, and the experiment registry.
2. Identify the earliest unchecked task whose dependencies are satisfied.
3. Make a minimal implementation plan and execute it; do not ask the owner for routine approvals.
4. Run unit tests and the smallest meaningful experiment before scaling.
5. Update registry, changelog, and reproducibility notes.
6. Compare against the current best on the **same** folds and metric.
7. Report: completed work, exact measured results, files changed, failed experiments, next task, and any blockers.
8. Continue until a rules-sensitive or account-sensitive action requires owner approval, or until the plan is complete.

When using coding assistants, prefer giving them code/config/aggregated logs over private competition rows. Never paste hidden data into a hosted assistant without verified permission.

### Suggested experiment registry columns

`experiment_id,date,git_commit,method,features,seed,split_id,fold_scores,pooled_oof_balanced_accuracy,macro_f1,fit_seconds,infer_seconds,peak_ram_gb,external_data_id,artifact_path,public_lb_score,notes,status`

### Agent completion definition

The task is complete only when the repository contains: a documented rules audit, a validated grouped-CV harness, reproducible CPU baselines, at least one meaningful research ablation, a fully working compliant Kaggle notebook, experiment logs, a technical report, and a polished README. The final Kaggle rank and any medal must be reported only after official confirmation.

---

## Immediate instruction to the agent

**Start now at Section 0.** Verify the official competition specification and available dataset schema. Then create the repository and implement Section 2 and the first character-TF-IDF baseline. Do not skip validation design, do not assume an external API is allowed, and do not fabricate a Kaggle score. Work autonomously through the checklist, requesting approval only for necessary account actions or unresolved compliance questions.
