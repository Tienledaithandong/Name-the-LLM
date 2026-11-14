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

## D010 — Deployable prompt-nuisance suppression is a negative result (2026-11-14)

Ran `configs/embeddings.yaml` on the frozen official folds (Section 5 / H2).
SVD-256 compression of char TF-IDF scores 0.3034 pooled OOF (vs 0.3167 raw sparse
char). Adding the deployable nuisance-8 projection drops it to 0.2434 (−0.0600);
style → SVD-64 → nuisance-8 scores 0.2737 vs 0.2904 unsuppressed style (−0.0167).
Every transform was fitted on the training side only; same `split_id`
`stratified_group-s42-n5`. Recorded in `experiments/registry.csv` and report §6.4.

Decision: do not adopt nuisance suppression or the SVD reductions for the
submission; the char+style blend (D009) remains the candidate. The near-factorial
design (17–18 models per prompt) means between-prompt centroid directions are not
pure topic nuisance, so projecting them out deletes model signal. Only k=8 was
tested; a k-sweep or supervised nuisance criterion is future work.

Reopen if: a smaller k or a supervised nuisance criterion shows a reproducible
pooled-OOF gain across seeds on the registered folds.

## D011 — A dedicated submission config reproduces the adopted blend (2026-11-14)

`configs/baseline.yaml` (Section 3 baselines) does not contain `style_only`, but
the adopted blend weights (D009) reference it. `predict --config baseline.yaml`
therefore dropped `style_only` from the weights and would have submitted the
single char model, silently diverging from the validated candidate. Added
`configs/submission.yaml` with exactly the two blend members
(`char_tfidf_logreg`, `style_only`), matching their CV configs; `make predict`
and the documented predict command now use it, and `package` ships both bundles
plus `weights.json`.

Reopen if: the adopted ensemble changes — update `submission.yaml` to the new
members and re-validate on the registered folds before packaging.

## D012 — Blend gain confirmed on alternative seeds and the secondary holdout (2026-11-14)

Ran `src/stability.py` (`make stability`) on official data. Alternative fold
seeds 7 and 13 re-select the identical 0.5/0.5 char+style blend with gains +0.0452
and +0.0441 over the best single; the untouched secondary prompt-held-out split
(25 prompts, 450 rows, seed 7) scores the fixed blend at 0.3956, +0.0222 over the
best single (char 0.3733, style 0.2822). Recorded in `experiments/registry.csv`
and report §6.5.

Decision: the D009 reopen condition (gain must reproduce on the holdout and an
alternative seed) is satisfied, so the char+style blend remains the submission
candidate. The smaller holdout gain is consistent with mild CV selection
optimism, not a failure to generalise.

Reopen if: a future re-run shows a non-positive holdout or alternative-seed gain,
in which case fall back to the best single model per D009.

## D013 — Frozen semantic encoder benchmarked and not adopted (2026-11-14)

Wired an additive `embedding` feature path into `TextMatrixBuilder`/`ModelSpec`
(default off; the 86-test suite stays hermetic via the no-download hashing
backend) and benchmarked a frozen `all-MiniLM-L6-v2` encoder (Apache-2.0,
384-dim) on the registered official folds (`configs/frozen_encoder.yaml`).
Embeddings-only scores 0.1278 pooled OOF; embeddings+style 0.3093 (vs char
0.3193, style 0.2904, blend 0.3760). Greedy blending picks
0.5×char + 0.5×(embeddings+style) at 0.3786 — only +0.0026 over the adopted
blend — but on the untouched secondary holdout that blend scores 0.3933 versus
char+style 0.3956 (−0.0023). The marginal CV gain reverses out of sample, so it
is selection noise. Recorded in `experiments/registry.csv` (external_data_id
`minilm-l6-v2`), `docs/external_data_register.md`, and report §6.6.

Decision: do not adopt the encoder (H3 negative as configured). Semantic
embeddings encode topic, which shifts across unseen prompts, not authorship.
The char+style blend (D009/D012) remains the submission candidate; the submitted
pipeline stays sparse and download-free with no attached checkpoint. The core
`requirements.txt`/`requirements-lock.txt` exclude `sentence-transformers`.

Reopen if: a style-tuned/contrastive or larger encoder shows a reproducible
pooled-OOF gain that also holds on the secondary holdout and an alternative
seed, within the CPU/runtime budget and with the checkpoint attachable offline.

## D014 — Crossed-design lever confirmed; pursue test-time prompt grouping (2026-11-14)

After the owner's submission scored **0.356 public LB** (top score 0.7), Phase A
of `docs/improvement_plan.md` measured the ceiling of exploiting the balanced
full-factorial design (every prompt answered by 17–18 of the 18 models). Using
**known train prompt groups** as an oracle on the saved blend OOF
(`src/crossed.py`, `experiments/phase_a_crossed.py`):

- one-per-model Hungarian assignment within prompt groups: **0.3760 → 0.4853**
  (+0.1093);
- within-prompt score centering: 0.3760 → 0.4152 (+0.0392), and only +0.0007 on
  top of assignment (0.4860 combined);
- retraining `style_only` on within-prompt-centered features (both sides):
  0.2904 → **0.3575** (+0.0671), stable across folds (±0.0099).

These are **oracle** numbers (perfect grouping assumed), not deployable CV
scores; they are labelled as such in the registry. Assignment is the dominant
lever and directly targets the sibling-confusion error pattern (53.3% of errors
within provider).

Decision: **GO for Phase B** — recover test prompt groups by semantic clustering
(the frozen MiniLM encoder that failed authorship is well suited to topic
clustering), then apply within-group centering and/or one-per-model assignment
at inference. The realised gain depends on cluster purity, which B1 must validate
against known `prompt_id` on held-out train prompts before any submission.

Compliance gate: test-time grouping is **transductive but label-free** (uses only
test text). Before submitting, confirm the rules permit it and record the ruling
in `docs/competition_spec.md` §10. Never apply these transforms inside reported
grouped-CV *training* scores. The current submission (D009/D012) is unchanged
until a Phase B pipeline is validated on the holdout and an alternative seed.

Reopen if: cluster purity on held-out prompts is too low to capture a meaningful
share of the +0.11 oracle gain, or the rules forbid transductive test-time
processing — in which case fall back to Phase C (stronger independent-passage
base models) and Phase D (within-prompt contrastive encoder).

## D015 — Recovered prompt grouping captures 88% of the oracle gain (2026-11-14)

Phase B (`src/grouping.py`, `experiments/phase_b_grouping.py`) simulated the
deployable pipeline on grouped CV: embed each fold's validation passages with
frozen MiniLM, cluster them into prompt groups **without** `prompt_id`, then apply
the crossed-design transforms on the recovered groups. With
`n_clusters = n_rows / 18` (the full-factorial prior): purity 0.934, ARI 0.916,
strict per-prompt recovery 0.833, and one-per-model assignment scores **0.4727**
pooled OOF vs 0.3760 argmax — **+0.0967, i.e. 88% of the +0.1093 oracle ceiling**.
A cosine-distance threshold (0.4) that assumes no `k` gives 0.4704, confirming
robustness. Assignment alone is the robust choice; adding centering helps
marginally when groups are clean (0.4756) but hurts when they are not.

Decision: the crossed-design lever is **deployable in principle** — semantic
clustering recovers prompt groups well enough to capture most of the gain. This
reverses the H3 framing: MiniLM is useful for *grouping* (topic), not authorship.
The pipeline now depends on the MiniLM checkpoint, so any submission must attach
the weights offline (`docs/external_data_register.md`).

Not yet adopted. Before it can replace the D009/D012 submission:
1. **Compliance gate** — confirm the rules permit transductive, label-free
   test-time clustering; record the ruling in `docs/competition_spec.md` §10.
2. **Holdout + alternative-seed validation** of the full cluster→assign pipeline
   (D009/D012 discipline), not just seed-42 grouped CV.
3. **k-estimate sensitivity** — verify behaviour when rows/prompt deviates from
   18, and prefer the threshold variant if the test design is uncertain.

Reopen if: the holdout/seed gain is non-positive, the rules forbid transductive
processing, or test-time clustering purity collapses on the real ~4,000-row test
set (larger than the ~540-row per-fold simulation) — fall back to Phase C/D.

## D016 — Recovered-grouping gain survives the holdout and alternative seeds; adopt crossed inference (2026-11-14)

Ran `experiments/phase_b_validate.py` — the full cluster→assign pipeline
(MiniLM embed → agglomerative cluster **without** `prompt_id` → one-per-model
Hungarian assignment) refit per fold on three independent splits. The blend
members (char_tfidf_logreg C=2.0, style_only C=1.0) and their 0.5/0.5 weights are
fixed; only grouping + assignment are added at inference.

| split | argmax | oracle assign | recovered (thr 0.4) | recovered (k=rows/18) | purity (k) |
|---|---|---|---|---|---|
| untouched holdout s7 (25 prompts) | 0.3956 | 0.4933 | **0.5000** | 0.4933 | 1.000 |
| grouped CV seed 7 | 0.3608 | 0.4808 | 0.4716 | 0.4712 | 0.940 |
| grouped CV seed 13 | 0.3582 | 0.4742 | 0.4657 | 0.4631 | 0.940 |

The gain is positive on **every** split: holdout **+0.1044**, seed 7 **+0.1108**,
seed 13 **+0.1075** (thr 0.4). The holdout argmax 0.3956 exactly reproduces
D012, confirming the harness is correct (an earlier holdout bug scored all rows
against a −1 sentinel; fixed to score only evaluated rows, never recorded).
Threshold 0.4 ≥ k-estimate on all three splits, so it is the adopted default —
it assumes no rows/prompt and keeps the highest purity at scale.

Scale diagnostic (full 2,699 passages / 150 prompts, no accuracy): k=150 purity
0.891 / ARI 0.875; thr 0.4 purity 0.960 / ARI 0.918 (216 clusters — over-splits
but stays pure); thr 0.5 purity 0.871. Purity degrades gracefully as the batch
grows; over-splitting pure clusters is harmless to assignment (each sub-cluster
still gets distinct models), while impure merges are the failure mode thr 0.4
avoids. Recorded in `experiments/registry.csv` (`phase_b_holdout_s7_rec_thr`,
`phase_b_cv_s7_rec_thr`, `phase_b_cv_s13_rec_thr`, `phase_b_scale_purity`).

Decision: D015's holdout/seed and k-sensitivity gates are **satisfied**. Adopt
**recovered-group one-per-model assignment (cosine thr 0.4)** as the new
submission candidate, superseding the argmax blend (D009/D012) **pending the
owner's LB submission**. Deployable path: `src.inference.predict_dataframe_crossed`
(tested, hermetic via the hashing encoder). Submission requires the MiniLM
checkpoint attached offline; `package` must ship it and the notebook must load it
from the attached dataset (Kaggle inference has no internet).

Reopen if: the owner's LB submission with crossed inference does not beat 0.356,
or real-test purity collapses (detectable as a degenerate label distribution /
many duplicate-model groups) — fall back to the argmax blend, then Phase C/D.
