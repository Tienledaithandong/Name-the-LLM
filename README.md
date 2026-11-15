# Name That LLM — CPU-first model attribution

18-class authorship attribution for the Kaggle **Name That LLM** competition:
given a 40–250 word passage, identify which of 18 frontier language models wrote
it. Metric: **balanced accuracy** (mean per-class recall).

Competition: <https://www.kaggle.com/competitions/name-that-llm>

> **Status (2026-11-14, rev 2):** pipeline implemented, tested, and evaluated on
> the **official training data**. The owner submitted the argmax blend
> (`0.5 × char + 0.5 × style`) → **0.356 public LB**; official top is **0.771**
> (5,851 teams). A crossed-design candidate (ADR D016) was built and validated at
> k=18 (holdout 0.5000), but **strategy research + honest k-subsampled CV (ADR
> D017) show the hidden test has ~2 passages per prompt, where the Hungarian gain
> collapses to ~0**. The crossed candidate is therefore **withdrawn as a submission
> default**; the argmax blend remains the safe baseline. The ~0.34 gap to the
> leader is a **base-model capability gap** — the revised roadmap (`docs/improvement_plan.md`,
> evidence in `docs/strategy_review.md`) prioritises a clean external pool +
> fine-tuned DeBERTa over post-processing. Synthetic-fixture numbers are retained
> only as a pipeline check and are labelled as such.

## What is verified vs. assumed

`docs/competition_spec.md` records the official facts read from Kaggle on
2026-11-14: the balanced-accuracy metric, the 18 labels, `train.csv`
(2,699 rows / 150 prompts), `test.csv` (`id`,`text`), `submission.parquet` written
to the working directory, `/kaggle/input/competitions/name-that-llm/test.csv`,
9-hour CPU/GPU limits, internet disabled, external data and pretrained models
allowed if publicly accessible, 5 submissions/day, 2 final submissions, team ≤ 5,
entry deadline 2027-01-06, final deadline 2027-01-13 (all 23:59 UTC).

Nothing in the code depends on an unverified item; open questions are listed at
the end of that document.

## Design in one page

```text
train.csv ──► schema validation ──► prompt-grouped folds (frozen, fingerprinted)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        char / word / char_wb TF-IDF                stylometry (44 features)
        (+ optional SVD, prompt-nuisance            length, punctuation, markdown,
         suppression)                               hedging, function words, TTR
                    └───────────────┬───────────────────────┘
                                    ▼
                      LinearSVC / LogisticRegression
                      (balanced class weights)
                                    │
                  out-of-fold margins (18 columns, frozen order)
                                    │
                     greedy blend (only on validated gain)
                                    │
                    refit on all data ──► joblib artifact dir
                                    │
              kaggle/submission.ipynb ──► submission.parquet
```

Key properties:

- **Prompt-grouped CV only.** Hidden test passages come from prompts absent from
  training; folds hold out whole prompts and the harness raises on any overlap.
- **Per-fold fit hygiene.** Vocabularies, IDF, style standardiser, SVD and
  nuisance projection are fitted on the training side only.
- **Fixed class order.** Every score matrix and artifact uses
  `src/constants.LABELS`; `ModelBundle.load` refuses artifacts with a different
  order.
- **Fail-fast submission.** Schema, ids, order, nulls and label validity are
  checked before and after writing the parquet.
- **CPU-first.** Sparse linear models train in seconds to a couple of minutes on
  CPU; no GPU required.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m pytest -q            # 120 tests
```

Python 3.11+ (developed on 3.13). Versions are pinned in
`requirements-lock.txt`.

## Reproducibility

One command, no competition data, no network:

```bash
python -m src.cli demo
# or: make demo
```

It generates the synthetic fixture, audits it, builds 5 prompt-grouped folds,
cross-validates the baselines, blends, refits, predicts, validates
`submission.parquet`, and reports a held-out score on the fixture.

With the real data in `data/`:

```bash
make audit      # docs/audit + figures
make splits     # frozen folds + fingerprint
make baseline   # 6 reference/TF-IDF models -> OOF, metrics, registry
make features   # stylometry ablations
make embeddings # SVD + prompt-nuisance variants (H2)
make frozen-encoder # H3 pretrained-encoder benchmark (optional sentence-transformers)
make ensemble   # greedy blend, prints whether the gain is real
make figures    # report figures from saved OOF/metrics
make stability  # blend robustness: alternative fold seeds + secondary holdout
make predict    # refit blend members (configs/submission.yaml) -> submission.parquet
make package    # upload-ready Kaggle artifact dataset (models + weights + src)
make predict-crossed   # crossed-design inference (D016): prompt-group assignment
make package-crossed   # dataset + offline MiniLM grouping encoder (needs sentence-transformers)
```

Then attach the packaged dataset to `kaggle/submission.ipynb` and submit. For the
crossed candidate use `make package-crossed` (ships the MiniLM checkpoint offline)
and attach that dataset; the notebook reads `inference.json` and switches to
crossed inference automatically. Owner-only account steps (rules, upload, submit,
final selection) are listed in `docs/submission_checklist.md`.

## Results

### Official training data (2026-11-14)

2,699 passages, 150 prompts, 18/18 labels, 5 prompt-grouped folds (30 held-out
prompts each, zero prompt overlap). Pooled out-of-fold balanced accuracy:

| model | pooled OOF | fold mean ± std | macro F1 | fit s (5 folds) |
|---|---|---|---|---|
| char TF-IDF + LogisticRegression | 0.3193 | 0.3193 ± 0.0119 | 0.3089 | 80.6 |
| char TF-IDF + LinearSVC | 0.3167 | 0.3167 ± 0.0134 | 0.3049 | 75.2 |
| word+char TF-IDF + LinearSVC | 0.3156 | 0.3156 ± 0.0094 | 0.3025 | 73.5 |
| `char_wb` TF-IDF + LinearSVC | 0.2963 | 0.2963 ± 0.0101 | 0.2877 | 49.6 |
| style only + LogisticRegression | 0.2904 | 0.2904 ± 0.0142 | 0.2792 | 29.6 |
| word TF-IDF + LinearSVC | 0.2697 | 0.2697 ± 0.0115 | 0.2539 | 7.5 |
| char+style TF-IDF + LinearSVC (unstable) | 0.2245 | 0.2244 ± 0.0849 | 0.2254 | 1213.8 |
| majority class | 0.0556 | 0.0556 ± 0.0000 | 0.1053 | 0.0 |
| stratified random | 0.0545 | 0.0545 ± 0.0047 | 0.0534 | 0.0 |

**Current best — validated blend:** `0.5 × char TF-IDF LogisticRegression +
0.5 × style-only LogisticRegression` → **0.3760** pooled OOF, **+0.0567** over
the best single model. The gain appears in every fold (+0.035 to +0.074), so it
is not one lucky split:

| fold | blend | best single | gain |
|---|---|---|---|
| 0 | 0.3779 | 0.3149 | +0.0630 |
| 1 | 0.3759 | 0.3407 | +0.0352 |
| 2 | 0.3796 | 0.3056 | +0.0741 |
| 3 | 0.3741 | 0.3222 | +0.0519 |
| 4 | 0.3722 | 0.3130 | +0.0593 |

**Ablation** (`configs/features.yaml`, same folds):

| ablation | pooled OOF |
|---|---|
| char only, LinearSVC | 0.3167 |
| word + char, LinearSVC | 0.3156 |
| style only, LogisticRegression | 0.2904 |
| word only, LinearSVC | 0.2697 |
| char + style, LinearSVC | 0.2245 (2 of 5 folds collapsed; non-convergent) |
| char + style, LogisticRegression | stopped: no convergence, >37 min, ~0.7 GB RSS |
| char → SVD-256, LinearSVC | 0.3034 |
| char → SVD-256 → nuisance-8, LinearSVC | 0.2434 |
| style → SVD-64 → nuisance-8, LogisticRegression | 0.2737 |
| MiniLM embeddings only, LogisticRegression | 0.1278 |
| MiniLM embeddings + style, LogisticRegression | 0.3093 |

Findings: style alone is competitive with a 100k-feature character model using
only 44 features, and the two are complementary enough to blend (+0.0567).
Concatenation is not a free lunch here — `word+char` matches char alone, and the
`char+style` sparse concat failed to converge at this vocabulary size.
Deployable prompt-nuisance suppression (H2) is a **negative result**: projecting
out the top-8 between-prompt directions lowers pooled OOF for char
(0.3034→0.2434) and style (0.2904→0.2737); SVD-256 compression alone also slightly
lowers char (0.3167→0.3034). See report §6.4 and ADR D010.
A frozen semantic encoder (H3) is also a **negative result**: MiniLM embeddings
score 0.1278 alone (topic, not authorship), and the greedy char+encoder blend
gains only +0.0026 on CV (0.3786) which reverses on the holdout (0.3933 vs
0.3956), so it is not adopted. See report §6.6 and ADR D013.

**Error analysis of the blend:** weakest classes `claude-fable-5` (0.121),
`glm-5.3-flash` (0.147), `gpt-5.6-sol` (0.153); strongest
`gemini-3.1-pro-preview` (0.733) and `claude-opus-5` (0.707). 53.3% of errors
stay inside a provider, and the top confusions are siblings:
`gemini-3.8-flash`→`gemini-3.7-flash` (56), `gemini-3.6-flash`→`gemini-3.7-flash`
(40), `glm-5.3`↔`glm-5.3-flash` (70 combined).

Data audit: each class has 149–150 rows, each prompt covers 17–18 models, no
duplicate texts, mean 114.7 words, 57.84% of passages contain non-ASCII
characters, and there is no markdown or emoji in the corpus. Details in
`artifacts/audit/official/data_audit.md` and `docs/data_audit.md`.

### Crossed-design candidate (D016 — WITHDRAWN by D017, k=18 artifact)

> **Withdrawn as a submission default (ADR D017; reframed D018).** The gains below
> were measured at **k=18 passages per prompt** (the train design). The hidden test
> is **k≈2**, where the Hungarian paired gain (method − argmax) shrinks to a
> marginal +0.0068 [+0.0032,+0.0104] on cv_s42 and +0.0029 [−0.0059,+0.0117] on the
> examined holdout (CI spans 0). Do **not** submit this candidate. The k-robust
> replacement (`src/calibration.py`, damped Sinkhorn) is ~neutral on the weak blend
> (paired-diff CIs span 0 at k=2) and is reserved for a future strong ensemble. The
> primary metric for the argmax blend is the **full grouped OOF** (0.3760 cv_s42).
> See `docs/strategy_review.md` and `experiments/stress_k.py`.

The training set is a balanced full factorial (150 prompts × 17–18 of the 18
models). The hidden test does **not** mirror this (k≈2, D017), so the premise
fails and the mechanism below is **research-only**: passages answering the same
prompt can be recovered by semantic clustering and constrained to **one passage
per model**. `src/crossed.py` (Hungarian assignment) + `src/grouping.py` (MiniLM
agglomerative clustering) + `src.inference.predict_dataframe_crossed` implement
this label-free, transductive inference step. The blend members and weights are
unchanged.

k=18 measurements (historical — **not** deployable k≈2 estimates):

| split | argmax blend | recovered-group assignment (cosine thr 0.4) | gain |
|---|---|---|---|
| examined holdout (25 prompts, seed 7) | 0.3956 | **0.5000** | +0.1044 |
| grouped CV seed 7 | 0.3608 | 0.4716 | +0.1108 |
| grouped CV seed 13 | 0.3582 | 0.4657 | +0.1075 |
| grouped CV seed 42 (reference) | 0.3760 | 0.4704 | +0.0944 |

The oracle ceiling (known prompt groups) is 0.4853 on seed 42, so recovered
grouping captures ~88–90% of it. The gain is positive on every split; threshold
0.4 keeps the highest cluster purity at scale (0.960 on all 2,699 passages / 150
prompts). Compliance cleared (spec §9: label-free transductive clustering is
allowed). The packaged crossed pipeline was verified to load the attached MiniLM
checkpoint fully offline and run at 2,699-row scale in ~56 s on CPU, producing a
near-uniform label distribution (max class share 0.056 ≈ 1/18) — the expected
full-factorial signature. See `docs/improvement_plan.md` §5 and ADRs D014–D016.

### Synthetic fixture (superseded pipeline check)

The fixture run is kept only to prove the pipeline works without competition
data; its numbers (char+style logreg 0.4823, style-only 0.4723, char 0.2061) are
artifacts of synthetic style profiles and are **not** a performance estimate. It
lives in `artifacts/cv/synthetic/` and is labelled `synthetic` in the registry.

## Repository layout

```text
configs/            baseline, features (ablations), embeddings, frozen_encoder, ensemble, submission
data/               ignored; official CSVs or the synthetic fixture
docs/               spec, protocol, audit, external-data register, decisions (ADRs)
src/
  constants.py      verified labels, metric, paths
  data.py           schema validation, loading, fingerprint
  validation.py     prompt-grouped folds, leakage guards, split persistence
  metrics.py        balanced accuracy, per-class recall, diagnostics
  features_tfidf.py char/word/char_wb TF-IDF + composition, SVD, scaling
  features_style.py deterministic stylometry (44 features)
  features_embedding.py optional frozen encoder (hashing / sentence-transformers)
  prompt_invariance.py  deployable nuisance suppression + diagnostic centring
  model.py          ModelSpec, ModelBundle (save/load, class-order guard)
  baselines.py      majority, stratified random
  train.py          CV loop, OOF, registry, summary
  ensemble.py       greedy OOF blending on normalised margins
  inference.py      final refit and prediction (argmax + crossed-design)
  crossed.py        within-prompt centering + one-per-model Hungarian assignment
  calibration.py    k-robust damped Sinkhorn-Knopp calibration (no grouping)
  grouping.py       MiniLM embedding clustering + prompt-group purity metrics
  submission.py     parquet writing + validation
  audit.py          data audit, report, class-balance/length figures
  figures.py        report figures from saved OOF/metrics (confusion, recall, ablation)
  cli.py            audit / splits / train / ensemble / figures / predict / package / demo
tests/              120 unit and integration tests
figures/            generated report figures (PNG)
kaggle/             submission.ipynb (offline, attached-artifact)
experiments/        registry.csv, reports
report/             research_report.md
```

## Compliance posture

- No competition rows are sent to any assistant or external service; only code,
  configs, aggregates and schema.
- No runtime network access, no third-party inference, no paid APIs.
- The submitted argmax blend is trained from scratch on the official training
  folds and uses no external data. The **crossed-design candidate** (D016) attaches
  one public encoder (MiniLM, Apache-2.0) **offline** for prompt-group clustering
  only — never as an authorship feature, never downloaded at runtime.
  `docs/external_data_register.md` records its provenance, licence and validation;
  it was benchmarked for H3 authorship and not adopted for that purpose (D013).
- Data is never committed (`data/`, `artifacts/` ignored).
- Ambiguous compliance questions are logged in `docs/competition_spec.md` §10 and
  need owner approval before the final submission.

## Known limitations

- The owner's first submission (argmax blend) scored **0.356 public LB**; the
  0.3760 is local grouped CV. The crossed-design candidate (D016) is validated
  locally (holdout 0.5000) but **not yet submitted**, so there is no leaderboard
  evidence for it yet. The public/private model mixes differ.
- Crossed inference depends on test-time prompt-group recovery: if clustering
  purity collapses on the real ~4,000-row hidden test (larger than the 2,699-row
  local proxy), the gain shrinks. A degenerate label distribution in the notebook
  output is the warning sign; fall back to the argmax blend if so.
- The blend weights were selected on the same out-of-fold predictions they are
  scored on, so the gain is mildly optimistic. It has since been re-checked on
  alternative fold seeds (7, 13) and the secondary prompt-held-out split and is
  positive on all (+0.0452, +0.0441, +0.0222 respectively; report §6.5).
- Prompt-topic shift between train and hidden test can exceed what grouped CV
  simulates, and the public/private model mixes differ.
- `char+style` sparse concatenation did not converge at 100k features; the idea
  is unresolved rather than refuted.
- Deployable prompt-nuisance suppression (k=8) reduced pooled OOF on official
  data and is not adopted; only one k was tested (report §6.4, ADR D010).
- A frozen MiniLM encoder was benchmarked (H3) and not adopted: 0.1278 alone, and
  a +0.0026 greedy blend gain that reverses on the holdout (report §6.6, ADR
  D013). Only one small encoder was tested.
- The synthetic fixture is not a proxy for competition difficulty.
