# Name That LLM — CPU-first model attribution

18-class authorship attribution for the Kaggle **Name That LLM** competition:
given a 40–250 word passage, identify which of 18 frontier language models wrote
it. Metric: **balanced accuracy** (mean per-class recall).

Competition: <https://www.kaggle.com/competitions/name-that-llm>

> **Status (2026-11-14):** pipeline implemented, tested and reproducible on a
> synthetic fixture. The official competition files are **not** available on this
> machine (downloading requires the owner's Kaggle account to accept the rules),
> so **no competition score, CV number or leaderboard position exists yet**.
> Every number in this README that is not labelled otherwise is a fixture result.

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

### Synthetic fixture (pipeline validation — NOT competition results)

1671 train rows / 660 held-out rows, 150 prompts, 5 prompt-grouped folds.
Pooled out-of-fold balanced accuracy:

| model | pooled OOF | fold mean ± std | macro F1 |
|---|---|---|---|
| `char_wb` TF-IDF + LinearSVC | 0.2341 | 0.2342 ± 0.0147 | 0.2233 |
| char TF-IDF + LogisticRegression | 0.2091 | 0.2096 ± 0.0090 | 0.2040 |
| char TF-IDF + LinearSVC | 0.2061 | 0.2059 ± 0.0111 | 0.1996 |
| word TF-IDF + LinearSVC | 0.0951 | 0.0945 ± 0.0144 | 0.0946 |
| majority class | 0.0535 | 0.0556 ± 0.0000 | 0.0934 |
| stratified random | 0.0510 | 0.0514 ± 0.0061 | 0.0504 |

Greedy blending selected the single best model (gain 0.0), so no blend was
adopted.

**Feature ablation** (`configs/features.yaml`, same folds):

| ablation | pooled OOF |
|---|---|
| char + style, LogisticRegression | **0.4823** |
| style only, LogisticRegression | 0.4723 |
| char + style, LinearSVC | 0.3650 |
| char only, LinearSVC | 0.2061 |
| word + char, LinearSVC | 0.1674 |
| word only, LinearSVC | 0.0951 |

On the smaller 40-prompt fixture (428 rows, 3 folds) the style-only model
reached 0.3697 pooled OOF vs 0.1477 for a 3k-feature char model. The fixture
encodes style in punctuation/formatting, so this says the grouped protocol works
and the style block carries signal — it says nothing about real model
fingerprints. Exact per-fold scores, timings and RSS are in
`experiments/registry.csv`.

**No official-data result exists.** The table will be replaced with verified
numbers, per-fold spread and per-class recalls once the data is downloaded.

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

- The synthetic fixture is not a proxy for competition difficulty; its scores say
  nothing about the leaderboard.
- 2,699 training rows make per-fold estimates noisy; pooled OOF plus fold spread
  is used, and gains must survive alternative seeds.
- Prompt-topic shift between train and hidden test is the dominant failure mode
  and grouped CV can still underestimate it.
- No frozen-encoder result exists yet; the optional backend is untested against a
  real checkpoint.
