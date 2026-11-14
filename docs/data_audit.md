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

## Current status

The official competition files are **not present on this machine**: downloading
them requires the owner's Kaggle account to accept the competition rules, which
is an owner-only action. No numeric audit of the real data has been performed,
and no competition score exists.

The pipeline was exercised end-to-end on the synthetic fixture
(`src/synthetic.py`); its audit output is at
`artifacts/demo/audit/data_audit.md` and is explicitly labelled synthetic there.

## Checks the audit cannot perform on `test.csv`

- Prompt-level overlap between train and hidden test: `test.csv` has no
  `prompt_id`. The competition states hidden test prompts are unseen, so grouped
  CV is the local proxy.
- Label balance of the hidden test set: labels are hidden, and the public/private
  mixes differ. Balanced accuracy is the metric precisely because of this.

## Next actions when data access exists

1. `python -m src.cli audit --data-dir data` and inspect the duplicate-text and
   class-support tables.
2. Confirm `n_labels_present == 18` and note any class with < 5 prompt groups
   (that would force the `group` fallback splitter).
3. Record the real audit numbers in this file (replace this section), including
   the dataset fingerprint, before running any model.
