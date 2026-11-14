# Name That LLM — End-to-End Agent Execution Plan

**Project:** Kaggle Name That LLM (18-class text-to-model attribution)  
**Start:** 2026-11-14  
**Target final submission:** 2027-01-13 (verify the official deadline and timezone)  
**Primary objective:** Maximize **private-leaderboard generalization** while creating a reproducible, research-quality public portfolio.  
**Resources:** CPU-first, low RAM, optional limited Kaggle GPU; heavy use of AI coding agents.  
**Owner's role:** Approve rules-sensitive decisions, review experimental evidence, submit final entry.

> **Agent instruction:** Execute this document sequentially. Do not ask the owner to write routine code. Stop for explicit approval only when required by platform rules, data privacy, spending, or competition submission/account actions. Never fabricate scores, results, citations, or completed experiments. Treat all values below as hypotheses until verified.

## 0. Non-negotiable rules and verification gate

Before writing modeling code:

- [ ] Open and read the current official competition Overview, Data, Evaluation, Rules, Code Requirements, Timeline, and Discussion pages: https://www.kaggle.com/competitions/name-that-llm
- [ ] Record the exact metric, label names, file formats, dataset paths, submission schema, runtime/memory limits, internet setting, GPU allowances, external-data/model rules, pretrained weights rules, collaboration/team rules, AI-assistant/API restrictions, and deadlines in `docs/competition_spec.md` with URLs and retrieval dates.
- [ ] Verify whether the metric is balanced accuracy, whether predictions are hard labels, and whether `submission.parquet` is required. **Do not assume earlier descriptions are authoritative.**
- [ ] Inspect the actual dataset schema rather than hard-coding column names. The examples in this plan assume `id`, `text`, `model`, and `prompt_id`; adapt only after inspection.
- [ ] Verify that model checkpoints and other assets can be included in the notebook or accessed through approved Kaggle datasets with internet disabled.
- [ ] Do not upload competition data, private labels, test data, or private competition materials to third-party LLM APIs unless expressly authorized by the rules and the owner.
- [ ] AI-assisted programming, external datasets, external model inference, and paid API calls are distinct compliance questions. Document the ruling for each separately. If unclear, ask organizers and pause that branch.
- [ ] Never use leaked test labels, non-public solutions, unauthorized data, multiple accounts, or rule-violating inference.
- [ ] Confirm whether public writeups and sharing code/results are allowed and when. Keep sensitive data private.

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

- [ ] Initialize Git; add `.gitignore` for `data/`, `artifacts/`, secrets, raw rows, caches, and potentially restricted outputs.
- [ ] Create a Python 3.11+ environment (verify compatibility). Install NumPy, pandas, SciPy, scikit-learn, pyarrow, matplotlib, seaborn optional, psutil, joblib, PyYAML, pytest, and optional CatBoost/LightGBM/transformers/torch.
- [ ] Pin dependency versions and seeds. Implement command-line entry points for `audit`, `train`, `evaluate`, `predict`, `package`.
- [ ] Add automated tests for schema, ID alignment, group separation, label mapping, metrics, model serialization, and submission format.
- [ ] Add a one-command reproducibility target (`make baseline` or equivalent).
- [ ] Log CPU model, RAM, wall-clock runtime, peak memory, dependency versions, and Git commit for each experiment.

**Exit gate:** clean setup from README succeeds on CPU and tests pass.

## 2. Data audit and evaluation design — highest priority

- [ ] Load all official files; report row counts, columns, nulls, duplicates, label frequencies, text lengths, Unicode distribution, prompt counts, and number of labels per prompt.
- [ ] Check whether identical/near-duplicate texts, shared prompt templates, or augmented variants cross folds. Never use hidden test labels.
- [ ] Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` **if the data supports it**, with `prompt_id` as group. If not, use `GroupKFold` or a documented custom group-balanced split. Confirm **zero prompt overlap** in each fold.
- [ ] Inspect per-fold class support. Balanced accuracy requires attention to classes absent from validation; avoid invalid comparisons and document scoring behavior.
- [ ] Create and persist exact train/validation indices (`artifacts/splits/`) with dataset fingerprint. Do not change splits while comparing models.
- [ ] For every model, fit vocabulary, scaling, feature selection, PCA, calibrators, and class-bias parameters on training folds only. Never fit transforms on validation or hidden test unless explicitly transductive, permitted, and separately labeled as such.
- [ ] Report mean/std of per-fold balanced accuracy, **pooled out-of-fold balanced accuracy**, per-class recall, macro F1 (diagnostic), confusion matrix, latency, peak RAM.
- [ ] Keep an untouched secondary group-held-out validation scheme or alternative seeds for final robustness checks; avoid iterative tuning on that holdout.
- [ ] Establish a leakage checklist and test it automatically.

**Exit gate:** `docs/experiment_protocol.md` and a reusable validation harness exist before tuning.

## 3. First working baseline and first valid submission

Implement, evaluate, and save these CPU baselines in order:

1. Majority-class and stratified-random reference (sanity only).
2. Character TF-IDF (2–5 grams, capped vocabulary, float32) + LinearSVC (`class_weight='balanced'`).
3. Word TF-IDF (1–2 grams) + LinearSVC.
4. Character `char_wb` TF-IDF + LogisticRegression or LinearSVC.
5. Optional word/character feature concatenation, with sparse matrices and controlled RAM.

For each:

- [ ] Run five fixed folds and save OOF predictions and decision scores, class ordering, model config, fit time, inference time, RAM, and metrics.
- [ ] Check performance by class, model family, text length, and prompt group.
- [ ] Refit the selected model on all permitted labeled training data; preserve exact preprocessing and class order.
- [ ] Produce a valid submission using the **official** schema and filenames; verify row count, ID order, unique IDs, valid labels, and no nulls.
- [ ] Test the Kaggle notebook with internet disabled and a clean environment; use only approved attached artifacts.
- [ ] Submit the first valid version if the owner has authorized the submission and platform actions are available. Otherwise provide exact instructions and ready-to-run artifact.
- [ ] Record public leaderboard score separately from CV; never optimize solely against public LB.

**Exit gate:** first reproducible CPU-only notebook submission, or a verified submission-ready artifact if account submission is unavailable.

## 4. Stylometric feature engineering

Create `src/features_style.py` with unit-tested, deterministic extraction of:

- Length: words, characters, sentences, paragraphs, average/variance of sentence length.
- Punctuation: comma, semicolon, colon, parentheses, em/en dash, hyphen, quote styles, ellipsis, exclamation/question marks.
- Formatting: whitespace, newline, bullet/numbered lists, Markdown, code fences, digits, Unicode markers.
- Lexical: type-token ratio (with length normalization), function-word frequencies, contraction rates, common discourse markers.
- Syntactic proxies: average token length, capitalization, sentence-start patterns, hedging phrases, transition phrases.

Then:

- [ ] Evaluate style-only logistic regression, SVM, and small CatBoost/LightGBM (if permitted).
- [ ] Fit imputers/scalers only inside each training fold.
- [ ] Compare style features with character TF-IDF through score blending or feature concatenation.
- [ ] Make per-feature and per-class analyses without claiming causal attribution.
- [ ] Save ablations: word only, char only, style only, char+style, word+char.

**Exit gate:** a controlled ablation report, even if no improvement.

## 5. Research track: prompt-invariant LLM fingerprints

**Research question:** Can prompt-robust stylometric features improve exact-model attribution on previously unseen prompts without training a large language model?

Hypotheses to test, not assume:

- H1: Character/typography features generalize better across prompts than semantic word features.
- H2: Prompt-invariance techniques improve grouped validation.
- H3: Frozen encoder representations complement sparse stylometry.
- H4: Distinguishing model versions within a provider is harder than distinguishing providers.

Experiments:

- [ ] Separate content-heavy features from style-heavy features; compare both on grouped CV.
- [ ] Explore training-prompt mean-centering `z_i - mean(z | prompt_i)` as a **diagnostic** only; hidden test prompt IDs/centroids are not available.
- [ ] Design deployable alternatives that do not require hidden test prompt labels: style-only projection, learned nuisance suppression on training groups, approximate clustering with fit-on-train-only transforms, or robust feature selection.
- [ ] Guard against transductive leakage and unbalanced clusters. Explicitly test the failure of centroid methods when clusters contain unrelated prompts.
- [ ] Measure performance on held-out prompts, short/long passages, and within-provider confusions.
- [ ] Require improvements across multiple group splits/seeds, not one lucky fold.
- [ ] Save negative results and methodological caveats.

**Exit gate:** at least one clearly specified method with a rigorous ablation and reproducible evaluation, whether positive or negative.

## 6. Public external data and small frozen embeddings

Before using external material:

- [ ] Check official rules for allowed external data, checkpoints, and distribution.
- [ ] Record each source URL, license, access conditions, collection date, model label mapping, size, and known distribution differences in `docs/external_data_register.md`.
- [ ] Use only publicly available and authorized data; no private leaderboard labels or restricted model outputs.
- [ ] Deduplicate by exact text and near-duplicate similarity against official training and validation; do not introduce same-prompt contamination across folds.
- [ ] Verify that external model identities map to the 18 official labels; do not silently merge versions.
- [ ] Compare official-only versus official+external with the **same** official grouped validation. Track whether external data harms domain transfer.

Frozen embeddings:

- [ ] Choose a small publicly available encoder that fits CPU RAM and competition inference constraints; verify license and checkpoint availability.
- [ ] Batch encode text, cache float16/float32 embeddings, and train regularized linear classifiers on top.
- [ ] Compare CPU inference time and memory with sparse TF-IDF. Do not adopt embeddings unless they improve grouped OOF or meaningful diversity.
- [ ] Optional GPU: use Kaggle's permitted GPU for embedding extraction or small encoder adaptation only if measured gains justify it.

**Exit gate:** documented data provenance and measured comparison, no unverified external resources in final notebook.

## 7. Ensembling and metric-aware decisions

- [ ] Save fold-aligned OOF decision scores for all base models and class orders.
- [ ] Standardize/convert scores in a controlled way. LinearSVC margins are **not probabilities**; avoid treating them as calibrated probabilities.
- [ ] Compare simple averaging, rank-based blending, normalized-margin blending, and calibrated probabilities where validation supports it.
- [ ] Tune ensemble weights on OOF with regularization or nested validation; avoid overfitting 2,699 examples with dozens of parameters.
- [ ] Evaluate class-specific intercepts/biases for balanced accuracy, with out-of-sample tuning and strict separation.
- [ ] Select a small ensemble of genuinely complementary models; prioritize stability, CPU speed, and reproducibility.
- [ ] Perform 2–3 split-seed checks and inspect public/private distribution-shift risks.

**Exit gate:** chosen ensemble has a reproducible grouped-CV advantage and fits Kaggle runtime/memory constraints.

## 8. Kaggle submission engineering

- [ ] Make `kaggle/submission.ipynb` fully self-contained with approved Kaggle input assets and pinned dependencies.
- [ ] Ensure internet is disabled; no runtime calls to ChatGPT, Claude, OpenAI APIs, Hugging Face downloads, or other network services.
- [ ] Verify test-time preprocessing matches training; preserve exact class mapping.
- [ ] Validate schema and file path against official submission requirements, not assumptions.
- [ ] Benchmark full expected hidden-test size with conservative memory/time headroom; simulate cold start and model load.
- [ ] Add fail-fast checks for missing files, wrong feature dimensions, unexpected labels, NaNs, and ID mismatch.
- [ ] Save a notebook version, run Kaggle's actual submission flow, and confirm success. If direct account access is unavailable, provide owner with a concise final submission checklist.
- [ ] Keep a log of submission version, Git commit, CV score, public LB score, timestamp, and changes.
- [ ] Avoid chasing public LB fluctuations without supporting grouped-validation evidence.

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

- [ ] It uses the exact registered data split and correct metric.
- [ ] No leakage, hidden-test tuning, or unauthorized external resource is detected.
- [ ] The improvement is observed in pooled OOF and is not driven by one pathological fold.
- [ ] Per-class recalls and within-provider errors are examined.
- [ ] Runtime, peak RAM, and notebook compatibility are measured.
- [ ] All config/seed/commit/artifact identifiers are logged.
- [ ] The agent explains *why* the method might work and *what evidence* supports it.

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
