# Data audit

## Verified schema (2026-11-14)

`train.csv` — 2,699 passages, 150 prompts:

| column | type | meaning |
|---|---|---|
| `id` | str | unique anonymised passage id |
| `prompt_id` | str | anonymised prompt group; the CV grouping key |
| `text` | str | generated passage, roughly 40–250 words |
| `model` | str | one of the 18 official labels |

`test.csv` — `id` (str), `text` (str). No `prompt_id`, no `model`. Interactive
sessions receive an example file; the graded rerun has ~4,000 rows.

`sample_submission.parquet` — `id`, `model`.

License: CC BY-SA 4.0. Data is not committed to this repository; `data/` is
ignored.

## How to run the audit

```bash
python -m src.cli audit --data-dir data      # official data, once downloaded
python -m src.cli audit --synthetic          # fixture smoke run
```

Outputs:

- `artifacts/audit/<dataset>/audit.json` — machine-readable numbers,
- `artifacts/audit/<dataset>/data_audit.md` — the tables below, filled in,
- `figures/class_balance.png`, `figures/length_distribution.png` (official runs)
  or the same files under `artifacts/audit/<dataset>/figures/`.

The audit reports: row/column/null counts, duplicate ids, exact-duplicate texts
and their group sizes, label frequencies, rows and distinct labels per prompt,
word/character length quantiles, non-ASCII / markdown / emoji passage shares,
provider support, and the dataset fingerprint.

## Official-data audit (2026-11-14)

Downloaded with `python -m kaggle competitions download -c name-that-llm -p data --unzip`.
Dataset fingerprint `6368e9c31e8d060f`. Full output:
`artifacts/audit/official/data_audit.md`, `artifacts/audit/official/audit.json`,
figures in `figures/`.

| quantity | value |
|---|---|
| rows | 2,699 |
| prompts | 150 |
| labels present | 18 / 18 |
| rows per prompt (min / median / max) | 17 / 18 / 18 |
| labels per prompt (min / median / max) | 17 / 18 / 18 |
| class support (min / max) | 149 / 150 |
| class imbalance ratio | 1.01 |
| duplicate ids | 0 |
| duplicate texts | 0 |
| words per passage (mean / median) | 114.7 / 101 |
| words per passage (p05 / p95) | 44 / 224 |
| passages containing non-ASCII | 57.84% |
| passages containing markdown | 0.00% |
| passages containing emoji | 0.00% |
| test.csv rows in the downloaded copy | 3 |

Provider support: OpenAI 750, Anthropic 749, Google 600, ZAI 300, Qwen 150,
Moonshot AI 150. Only `claude-fable-5` has 149 rows; every other label has 150.

What this means for modelling:

- The training set is effectively a **balanced full factorial** (each prompt
  answers 17–18 of the 18 models), so on training folds balanced accuracy and
  plain accuracy almost coincide — but the hidden test mix is unequal, so
  balanced accuracy remains the right metric.
- **No duplicate texts**, so no exact-duplicate leakage across folds.
- **No near-duplicate crosses a fold either (D047).** Exact equality is a weak
  test, so `experiments/near_duplicates.py` measured all 3,640,951 unordered
  official-train pairs with the same word TF-IDF (1–2 grams, sublinear tf, l2,
  30k features) and 0.3 threshold as the external-pool screen: 2,190 pairs above
  0.3 and 34 above 0.5, **every one same-prompt, zero cross-fold**; nothing above
  0.7, corpus maximum 0.6223. The similarity that exists is the shared question
  vocabulary of the 18 answers to one prompt, which the grouped split keeps on
  one side. Descriptive audit — no predictor fit, no score, no test row; stores
  ids/fold numbers/cosines only. Similarity is a heuristic, not proof that every
  augmented variant was found.
- **No markdown/emoji** anywhere: those stylometric columns are constant on
  official data (harmless after outlier-safe scaling, but they carry no signal,
  unlike on the synthetic fixture).
- 57.84% of passages contain non-ASCII characters (em dashes, curly quotes),
  supporting the typography features.
- 150 prompts × ~18 rows gives 30 held-out prompts per fold with 17–18 models
  each, so every validation fold has support for all 18 classes.
- The downloaded `test.csv` has **3 rows**, matching the Data page (the Overview's
  "single example row" wording is the outlier).

## Checks the audit cannot perform on `test.csv`

- Prompt-level overlap between train and hidden test: `test.csv` has no
  `prompt_id`. The competition states hidden test prompts are unseen, so grouped
  CV is the local proxy.
- Label balance of the hidden test set: labels are hidden, and the public/private
  mixes differ. Balanced accuracy is the metric precisely because of this.

## Remaining checks

- [x] `n_labels_present == 18`; every class has 150 prompt groups, so the
  stratified-group splitter is valid.
- [x] duplicate-text check: zero.
- [x] near-duplicate check across folds: zero at every threshold up to 0.9
  (`artifacts/near_duplicates/results.json`, D047).
- [x] Inspect length-band performance once model results exist — measured by
  `experiments/length_sensitivity.py` (D041): accuracy rises monotonically with
  length, 0.2491 (<50 words) → 0.4718 (≥200) for the official blend and
  0.3125 → 0.5858 for the pool-MIX candidate. Short passages are confirmed as the
  weak spot; all 18 classes occur in every band. Report §8.2.
- [ ] Confirm the hidden test row count and label mix after the first scored
  submission — neither is knowable locally.
