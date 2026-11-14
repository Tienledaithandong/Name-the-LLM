# Decisions

Short ADRs. Newest last. Each entry records the decision, why, and what would
reopen it.

## D001 — Verified spec over the plan's assumptions (2026-11-14)

The plan guessed `id`/`text`/`model`/`prompt_id` and a parquet submission. The
official pages confirm those columns, the 18 labels, balanced accuracy,
`submission.parquet`, and the Kaggle input path. Sample submission is parquet in
both cases. Recorded in `docs/competition_spec.md`; constants live in
`src/constants.py`.

Reopen if: Kaggle changes the page (re-read before final submission).

## D002 — Prompt-grouped CV is mandatory (2026-11-14)

Hidden test passages come from unseen prompts. `prompt_id` groups are held out
whole; row-wise random CV is never reported. Enforced at runtime, not by
convention.

Reopen if: the host publishes that hidden prompts overlap train prompts (would
still keep grouped CV as the conservative choice).

## D003 — Sparse stylometry before dense encoders (2026-11-14)

CPU-first budget: TF-IDF + linear models train in seconds per fold and run
offline. Dense encoders are optional and gated behind a registered, attached
checkpoint. Adoption requires measured grouped-OOF gain.

Reopen if: sparse models plateau and a small encoder shows a reproducible gain
within the 9-hour CPU/GPU budget.

## D004 — No competition rows to hosted assistant APIs (2026-11-14)

The plan asks assistants to work from code/config/aggregated logs. This is
stricter than the rules require and avoids any dependence on provider terms.

Reopen only with explicit owner authorisation and a rules review.

## D005 — Synthetic fixture for tests and smoke runs (2026-11-14)

Official data cannot be committed or downloaded without the owner's account. A
deterministic generator produces schema-valid data so the full pipeline is
testable. Outputs are labelled `synthetic` everywhere, written under
`data/synthetic/` and `artifacts/demo/`, and are never described as competition
results.

Reopen if: a small, licence-cleared public sample becomes available.

## D006 — Style block is standardised per fold (2026-11-14)

Initial style-only logistic regression failed to converge and char TF-IDF
generalised near chance; style columns span orders of magnitude (0–250 length vs
0–20 rates). A `StandardScaler(with_mean=False)` fitted on the training fold's
style block fixes conditioning and keeps the matrix sparse. Diagnostics in
`artifacts/demo` show style-only far above chance on the fixture.

Reopen if: an ablation shows unscaled style features are better on official data
(unlikely, but measure rather than assume).

## D007 — Blend only on validated gain (2026-11-14)

Greedy OOF selection is cheap but can overfit 2,699 rows. `ensemble.yaml` sets
`min_gain`; `python -m src.cli ensemble` prints when there is no validated gain
and the single best model should be used instead.

Reopen if: repeated-seed checks show the blend advantage is unstable.

## D008 — Vocabulary and iteration budgets (2026-11-14)

First ablation run stalled for over ten minutes on `char_plus_style`: liblinear
with 200k features, 18 classes and `max_iter=5000` spends most of its time in the
final iterations without converging (sklearn emitted `ConvergenceWarning`).
Configs now cap char vocabularies at 100k (word 80k) and `LinearSVC` uses
`max_iter=2000`. Rationale: 2,699 training rows cannot support a 200k-feature
vocabulary without heavy regularisation, and the warning showed the extra
iterations were not buying convergence.

Reopen if: official-data ablations show a large-capacity model is better; raise
the caps deliberately and record the runtime.

## D009 — First official CV result and the adopted blend (2026-11-14)

Official data was downloaded with the owner's existing Kaggle credentials
(`python -m kaggle competitions download -c name-that-llm -p data --unzip`).
Audit: 2,699 rows, 150 prompts, 18/18 classes at 149–150 rows each, no duplicate
texts, 114.7 mean words, no markdown or emoji.

Registered grouped CV (5 folds, 30 held-out prompts each): best single model is
char TF-IDF + logistic regression at 0.3193; style-only reaches 0.2904 with 44
features. Greedy blending selects 0.5/0.5 of those two and scores **0.3760**,
+0.0567 over the best single, positive in all five folds. That blend is the
adopted candidate for the first submission.

`char + style` sparse concatenation did not converge: LinearSVC collapsed on two
folds (fold scores 0.094/0.165 vs ~0.31 elsewhere, 1,214 s for five folds) and
the logistic-regression variant was stopped after >37 min and ~0.7 GB RSS. It is
excluded from `configs/features.yaml` with a comment; the concatenation idea is
unresolved, not refuted.

Known caveat: blend weights were selected on the same OOF predictions used to
score them. Before the final submission, confirm on the secondary prompt-held-out
split and one alternative fold seed.

Reopen if: the secondary holdout or an alternative seed does not reproduce the
gain, in which case submit the best single model instead.
