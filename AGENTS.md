# AGENTS.md — working rules for this repository

Kaggle **Name That LLM**: 18-class LLM attribution, metric = balanced accuracy,
submission = `submission.parquet`. The authoritative task list is
`name_that_llm_agent_plan.md`; the verified competition facts are
`docs/competition_spec.md`.

## Every session

1. Read `docs/competition_spec.md`, `docs/experiment_protocol.md`,
   `docs/decisions.md`, and `experiments/registry.csv`.
2. Pick the earliest unfinished task whose dependencies are satisfied.
3. Run `python -m pytest -q` before and after changes.
4. Record results in the registry (never invent numbers), and add an ADR to
   `docs/decisions.md` when you change the protocol.

## Hard rules

- **Never fabricate a score, rank, citation or experiment result.** If official
  data is unavailable, say so and label outputs `synthetic`.
- **Never commit competition data, artifacts, secrets or OOF files.** `data/` and
  `artifacts/` are ignored.
- **Never paste competition rows into hosted assistants.** Code, configs,
  aggregated logs and schema only.
- **Never bypass the leakage checks.** Grouped folds, per-fold fit of every
  learned transform, no hidden-test tuning, no transductive steps in reported
  scores.
- Do not treat LinearSVC margins as probabilities.
- Keep the 18-label order from `src/constants.LABELS` everywhere; artifacts store
  it and `ModelBundle.load` rejects a mismatch.
- Account actions (accepting rules, downloading data, submitting) are owner-only;
  prepare artifacts and instructions instead.

## Commands

```bash
python -m pytest -q                                  # tests
python -m src.cli demo                               # synthetic end-to-end smoke run
python -m src.cli audit  --data-dir data             # dataset audit + figures
python -m src.cli splits --data-dir data             # frozen grouped folds
python -m src.cli train  --config configs/baseline.yaml --data-dir data
python -m src.cli ensemble --dataset official
python -m src.cli predict --config configs/baseline.yaml --data-dir data --fit
python -m src.cli package --kaggle-user <user>       # upload-ready artifact folder
make baseline                                        # same as the train step
```

## Conventions

- Python 3.11+, `from __future__ import annotations`, type hints on public
  functions.
- Modules are importable as `from src.<module> import ...` from the repo root.
- New model variants go in a YAML config; new learning behaviour needs a unit
  test and a registry entry.
- Submission code must keep the fail-fast checks (schema, ids, labels, nulls).
