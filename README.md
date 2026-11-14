# Name That LLM — CPU-first model attribution

18-class authorship attribution for the Kaggle **Name That LLM** competition:
given a 40–250 word passage, identify which of 18 frontier language models wrote
it. Metric: **balanced accuracy** (mean per-class recall).

Competition: <https://www.kaggle.com/competitions/name-that-llm>

> **Status (2026-11-14):** pipeline implemented, tested, and evaluated on the
> **official training data** with prompt-grouped CV (best: **0.3760** pooled OOF
> balanced accuracy, a two-model blend). **No Kaggle submission has been made**, so
> there is no leaderboard score, rank or medal yet. Synthetic-fixture numbers are
> retained only as a pipeline check and are labelled as such.

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
python -m pytest -q            # 71 tests
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
make ensemble   # greedy blend, prints whether the gain is real
make predict    # refit on all rows -> artifacts/final/official/submission.parquet
make package    # upload-ready Kaggle artifact dataset
```

Then attach the packaged dataset to `kaggle/submission.ipynb` and submit.

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

Findings: style alone is competitive with a 100k-feature character model using
only 44 features, and the two are complementary enough to blend (+0.0567).
Concatenation is not a free lunch here — `word+char` matches char alone, and the
`char+style` sparse concat failed to converge at this vocabulary size.

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

### Synthetic fixture (superseded pipeline check)

The fixture run is kept only to prove the pipeline works without competition
data; its numbers (char+style logreg 0.4823, style-only 0.4723, char 0.2061) are
artifacts of synthetic style profiles and are **not** a performance estimate. It
lives in `artifacts/cv/synthetic/` and is labelled `synthetic` in the registry.

## Repository layout

```text
configs/            baseline, features (ablations), embeddings, ensemble
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
  inference.py      final refit and prediction
  submission.py     parquet writing + validation
  audit.py          data audit, report, figures
  cli.py            audit / splits / train / ensemble / predict / package / demo
tests/              71 unit and integration tests
kaggle/             submission.ipynb (offline, attached-artifact)
experiments/        registry.csv, reports
report/             research_report.md
```

## Compliance posture

- No competition rows are sent to any assistant or external service; only code,
  configs, aggregates and schema.
- No runtime network access, no third-party inference, no paid APIs.
- `docs/external_data_register.md` is empty: every reported model is trained from
  scratch on the official training folds.
- Data is never committed (`data/`, `artifacts/` ignored).
- Ambiguous compliance questions are logged in `docs/competition_spec.md` §10 and
  need owner approval before the final submission.

## Known limitations

- No Kaggle submission has been made, so there is no public/private leaderboard
  evidence; the 0.3760 is local grouped CV only.
- The blend weights were selected on the same out-of-fold predictions they are
  scored on, so the gain is mildly optimistic. It is consistent across all five
  folds but has not been re-checked on the secondary prompt-held-out split or
  alternative seeds.
- Prompt-topic shift between train and hidden test can exceed what grouped CV
  simulates, and the public/private model mixes differ.
- `char+style` sparse concatenation did not converge at 100k features; the idea
  is unresolved rather than refuted.
- No frozen-encoder result exists yet; the optional backend is untested against a
  real checkpoint.
- The synthetic fixture is not a proxy for competition difficulty.
