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

## D017 — Hidden test is k≈2, not k=18; withdraw crossed Hungarian, adopt honest k-CV (2026-11-14)

Researched the attached `Kaggle LLM Attribution Strategy.md` against primary
sources (`docs/strategy_review.md`). Verified: official leaderboard top **0.771**
(5,851 teams; public 40% / private 60%); the ATLAS paper (arXiv 2610.19583) is real
but attributes to model **family** and is weakest on version siblings; the external
`crtproduction/name-that-llm-open-pool` (211k) is a public Kaggle dataset. The
decisive claim — hidden test has **~1–3 passages per prompt (k≈2)**, not 18 — is
corroborated by three independent community sources and a forensic submission log
(Hard Hungarian 0.639 LB vs Damped Sinkhorn 0.664 LB on the same pipeline). It is
**not** an official Kaggle fact, so we tested it locally.

`experiments/phase_k2_cv.py` subsamples the pooled OOF blend predictions to k
passages per prompt (honest CV) and compares argmax / recovered Hungarian / damped
Sinkhorn (`src/calibration.py`, new, k-robust, 9 tests):

| split | k | argmax | recovered Hungarian | best Sinkhorn |
|---|---|---|---|---|
| cv_s42 | 18 | 0.3760 | 0.4690 (+0.093) | 0.3767 |
| cv_s42 | 2 | 0.3607 | 0.3643 (+0.004) | 0.3611 |
| holdout_s7 | 18 | 0.3956 | 0.5000 (+0.104) | 0.4000 |
| holdout_s7 | 2 | 0.3872 | 0.3833 (−0.004) | 0.3964 |

**The Hungarian gain collapses from +0.10 at k=18 to ~0 (or negative) at k=2.**
D016's +0.10 lift was a k=18 validation artifact, not deployable to a k≈2 test.
Sinkhorn is k-robust but ~neutral on the weak 0.376 blend (it pays off only on a
strong ensemble). Recorded in `experiments/registry.csv` (`phase_k2_*`) and
`artifacts/phase_k2/results.json`.

Decision:
1. **Retract D016's "adopt crossed inference as submission candidate."** Do not
   submit the crossed Hungarian package. The **argmax blend (D009/D012)** remains
   the safe baseline. `predict --crossed` / `package --crossed` stay available as
   research tools but are not the recommended submission path.
2. **Honest k≈2 subsampled CV is the primary deployable estimate** going forward;
   k=18 grouped CV is a training-side fit diagnostic only, never an LB proxy.
3. The ~0.34 gap to the leader is a **base-model capability gap**. Prioritise
   Phase 1 (clean external pool + DeBERTa-v3 sequence head + GBDT stylometry) over
   post-processing. Reserve damped Sinkhorn + typography + sibling residuals for
   Phase 4, on top of a strong ensemble, adopted only on validated honest-CV gain.
4. Compliance: single account only; the strategy source's leaked credentials and
   3-account quota orchestration are **not** used (they violate the one-account
   rule). External pool must be registered + decontaminated + validated.

Reopen if: official Kaggle documentation states the hidden test is k=18 (would
re-enable crossed assignment), or honest k≈2 CV of a future strong ensemble shows
a reproducible Hungarian gain (unlikely given the mechanism).

## D018 — Evaluation reframe: full OOF primary, k-subsampling is a stress test (2026-11-14)

Owner review corrected two framing errors in D017:

1. **D017 over-rotated** by making "honest k≈2 subsampled CV" the primary estimate
   for *everything*. For an **independent-passage model** (argmax blend, DeBERTa,
   GBDT) a row's prediction is identical regardless of how many sibling passages
   are scored beside it, so subsampling k only discards rows and adds noise. The
   **full prompt-grouped out-of-fold balanced accuracy over all rows** is the
   correct, most efficient **primary** metric and the main comparison. k=1,2,3,18
   subsampling is demoted to a **stress test for batch/group-dependent methods**
   (recovered Hungarian, damped Sinkhorn), which *can* change a row's prediction
   based on its batch.
2. **D017 reported a "best Sinkhorn" over a grid** — that is selection on the
   stress test. We now report **fixed, pre-registered settings** and **paired
   differences** (method − argmax) on the same subsampled rows, mean ± sample std
   and a 95% Student-t CI across ≥10 subsample seeds (`metrics.paired_difference`,
   `metrics.mean_ci`, new; 5 tests). Transductive methods stay **out of the
   primary reported CV** (hard rule). The seed-7 holdout is relabelled
   **examined** (not "untouched"); a **fresh seed-2027 holdout** (25 prompts /
   450 rows) is reserved, persisted but unscored, for final selection.

`experiments/phase_k2_cv.py` is replaced by `experiments/stress_k.py`
(`--reserve-fresh-holdout` persists the reserve without scoring). Results
(`artifacts/stress_k/results.json`, registry `stress_*`):

| split | primary full OOF | k=2 Hungarian Δ [95% CI] | k=2 Sinkhorn(0.3) Δ [95% CI] | k=18 Hungarian Δ |
|---|---|---|---|---|
| cv_s42 | **0.3760** (2699 rows, folds ±0.0026) | +0.0068 [+0.0032,+0.0104] | +0.0004 [−0.0025,+0.0032] | +0.0930 |
| holdout_s7 (examined) | **0.3956** (450 rows) | +0.0029 [−0.0059,+0.0117] | −0.0037 [−0.0101,+0.0027] | +0.1044 |

**Reading:** the large Hungarian gain is a k=18 artifact (+0.09/+0.10). At test-like
k=2 it shrinks to a marginal, split-dependent +0.003…+0.007 whose CI spans 0 on the
holdout — not a reliable deployable gain. Sinkhorn is ~neutral at k=2 (CIs span 0).
This **confirms D017's withdrawal** of crossed Hungarian as the submission default,
now with honest uncertainty rather than a best-of-grid point estimate.

Decision:
1. Primary metric for independent-passage models = **full prompt-grouped OOF BA**;
   the argmax blend's headline is **0.3760** (cv_s42), 0.3956 (examined holdout).
2. k-subsampled **stress test** reports paired diffs + CI for batch methods only;
   inference-only; never a selection score; fixed settings, no best-of grid.
3. Adopt a batch method only if its paired-diff CI excludes 0 at k≈2 on **both**
   CV and holdout, and only on a strong ensemble (Phase 4). Currently none qualify.
4. Reserve the fresh seed-2027 holdout for final selection; stop calling seed-7
   "untouched". Reconcile notebook / external-data register / checklist (D017
   withdrawal + this reframe).

Reopen if: a future strong ensemble shows a batch-method paired-diff CI excluding 0
at k≈2 on both splits (would justify Phase 4 post-processing), or official docs
state the hidden test is k=18.

## D019 — Open-pool v15 audited and admitted for training (pending validated gain) (2026-11-14)

Phase 1a external-data audit. Owner downloaded the public Kaggle pool
`crtproduction/name-that-llm-open-pool` **v15** (2026-11-14, CC BY-SA 4.0; 65
sources CC BY 4.0 / CC BY-SA 4.0 / CC0). Verified against the actual parquet, not
the README claims (`src/pool.py`, `experiments/pool_audit.py`, 9 hermetic tests;
artifacts `artifacts/pool_audit/audit.json` + `clean_pool_ids.json`, registry
`pool_audit` row):

- **Schema:** 226,191 rows × 30 cols, 0 null text, `n_words` 36–275. **All 18
  official labels present, exact match** to `constants.LABELS` (`per_model.csv`).
  Uneven: `gemini-3.7-flash` 3,541 … `gemini-3.8-flash` 23,474 → class weights /
  sampling required. Labels are publishers' claims, not re-verified.
- **Routing-risk flags are already cleared.** `flag_gemini37_routing_risk`,
  `flag_near_train_prompt`, `flag_train_prompt_regen` are **all False** for every
  row. The 10,838 unverified gemini-3.7 routing-risk rows were moved out to
  `gemini37_unverified.parquet` upstream; train-prompt answers (284) and mislabeled
  rows (284) already removed. **Do NOT re-filter `flag_gemini37_routing_risk`** —
  it is a no-op. This corrects the plan's original 1a instruction.
- **Dedup vs official train:** dataset `max_cos_train` ≤ 0.30 (mean 0.0994),
  `max_sim_train_prompt` ≤ 0.40, **0 exact text duplicates**. Independent
  **per-fold** decontamination (word TF-IDF (1,2) cosine > 0.30, vectorizer fit on
  each holdout fold only, chunked) over the 5 grouped cv_s42 folds removed
  **4/1/3/6/5** rows (0 exact, 0 shared prompt every fold) → **union 19 removed**,
  **clean pool 226,172**. This is the per-fold leakage guard the protocol requires.

Decision:
1. **Admit the clean 226,172-row pool for training** on official+pool folds, but
   **admission is provisional until a validated gain** — official-only vs
   official+pool must be compared on the **same** grouped folds (full OOF primary,
   D018), reporting BA + per-class / within-provider recall including harm.
2. **Leakage caveats (must honour for honest CV):** `match_weight` and `style_score`
   were fitted on **all** official train prompts → **refit per-fold or drop these
   columns**; never use them as fold-spanning features. `flag_kb_proxy_after_ruling`
   marks 20,227 rows; hosts permit training on public datasets produced via proxy,
   so keep by default (`strict=False`) with an optional strict drop for a
   sensitivity check.
3. **Licence:** CC BY-SA share-alike — derived artefacts (trained weights / pools)
   inherit attribution + share-alike; record before any redistribution. No runtime
   download at inference (internet off); pool ships attached or training-only.
4. Keep the argmax blend (D009/D012) as the submission fallback until a
   pool-trained candidate wins on the registered protocol.

Reopen if: a fold-spanning feature leaks (`match_weight`/`style_score` used without
per-fold refit), the pool licence/terms change, or official+pool shows no validated
gain (then drop the pool and stay official-only).

## D020 — GBDT stylometry/multi-view is a negative standalone result; ensemble footgun (2026-11-14)

*Environment note (2026-11-17): "lightgbm/catboost/xgboost are absent" below was true when
this ADR was written. `catboost 1.2.10` is installed now and the CatBoost style-only arm was
measured in **D049** (pooled 0.283003 — also negative). lightgbm and xgboost remain absent.
The historical text is left as written.*

Phase 1c cheap complementary model. Added a `gbdt` classifier
(sklearn `HistGradientBoostingClassifier`; lightgbm/catboost/xgboost are absent) to
`src/model.py` with a dense-input guard (style-only or SVD-reduced; sklearn 1.9.1
HistGBDT rejects sparse), `class_weight="balanced"`, `decision_function` margins
(not probabilities), +4 hermetic tests; `configs/gbdt.yaml`. Official-only grouped
OOF (cv_s42, registry `gbdt_*`):

| model | pooled OOF BA | fit s | peak RSS GB |
|---|---|---|---|
| gbdt_style (56 style feats) | **0.2630** | 74 | 0.20 |
| gbdt_char_svd200 (char+style→SVD200) | **0.2537** | 334 | 0.39 |
| gbdt_multiview_svd128 (char+word+style→SVD128, 120 iter)* | **0.2426** | 262 | — |

*Reduced budget: the full SVD256 / 300-iter multi-view config exceeded 560 s (heavy
200k-feature SVD + 5400 trees) and was not completed; the reduced run confirms the
same weak range. All GBDT variants are **far below the adopted 0.3760 char+style
blend**. SVD compression discards the
high-dimensional sparse char/word signal that LinearSVC/logreg exploit; stylometry
alone is weak for 18-class attribution. The greedy blend does **not** select any
GBDT member → **no validated blend gain** (D007). Decision: GBDT is **research
only, not adopted**; the cheap-complementary-model hypothesis is rejected for this
task. `experiments/pool_train.py` (official-only vs official+pool, per-fold decon,
stratified pool cap, paired-diff CI, registry rows) is prepared; a GBDT+pool run is
unlikely to reverse this and is deprioritised behind the DeBERTa encoder (D021).

**Ensemble footgun (caution):** `python -m src.cli ensemble` greedily blends **all**
OOF in `artifacts/cv/official/oof`, including the D013-**rejected**
`minilm_embedding_style_logreg`. It re-selects `char_tfidf_logreg +
minilm_embedding_style_logreg` (0.3786 CV) — the blend that **reverses on the
holdout** (0.3933 vs 0.3956, D013). Writing those weights would, after `predict`
filters models absent from `submission.yaml`, leave **char-only** and degrade the
submission. The adopted submission weights are **`char_tfidf_logreg 0.5 +
style_only 0.5` (pooled OOF 0.3760)**, restored to
`artifacts/cv/official/ensemble/weights.json`. Do not treat the blind greedy-over-
all output as the submission blend; keep `submission.yaml` members authoritative.

Reopen if: a richer dense feature set (e.g. cached encoder embeddings + style) makes
GBDT competitive on the grouped folds, or a GBDT member shows a validated blend gain
on both CV and the reserved holdout.

## D021 — Pursue DeBERTa-v3 on Kaggle GPU; official-only vs audited pool (2026-11-14)

Phase 1b stronger neural baseline (the user's priority after the audit). Local torch
is CPU-only, so fine-tuning runs on **Kaggle GPU** (owner authorised "use kaggle
gpu"). Prepared a self-contained script kernel `kaggle/deberta/deberta_train.py` +
`kernel-metadata.json`:

- **Model:** `microsoft/deberta-v3-base` (18-class sequence head) first — **not**
  large/8B–27B; do not assume the strategy's huge stacks fit one Kaggle GPU or give
  their claimed gains. `--model-name` is configurable.
- **Protocol:** same prompt-grouped folds (StratifiedGroupKFold, seed 42); full OOF
  BA primary (D018). **official-only vs official+clean-pool** on the same folds;
  pool decontaminated **per fold** against the held-out prompts (mirror of
  `src.pool.decontaminate_fold`, verified to remove the same 4 rows on fold 0);
  `match_weight`/`style_score` never used (D019). Reports per-class + within-provider
  recall and **paired per-fold diff + 95% CI**.
- **Budget guards:** a benchmark cell measures seq/s + peak GPU memory and
  extrapolates runtime vs the 9h limit; a stratified **pool cap/fold** (default 40k)
  and `--epochs-pool 1`, `--max-len 256`, fp16 keep the pool run in budget.
- **Offline path:** training kernel uses internet ON only to fetch public weights;
  `--save-final` exports weights for a separate **offline** inference notebook
  (internet OFF, weights attached). Submission stays the argmax blend until a
  DeBERTa candidate wins on the registered protocol.

Pushed as a **private** kernel
`tiennamlee/name-that-llm-deberta-official-vs-pool-cv` (version 1, GPU, competition
`name-that-llm` + dataset `crtproduction/name-that-llm-open-pool` attached); status
queued at write time. **No score yet** — results pending the GPU run; record OOF +
metrics in the registry when downloaded (never fabricate). Non-GPU logic (data load,
grouped folds with no prompt overlap, per-fold decon, stratified cap, metrics)
validated locally against the real pool.

Next: download `results.json` + OOF; compare official-only vs +pool; adopt only on a
validated gain over 0.3760 on CV **and** the reserved seed-2027 holdout; then build
the offline inference notebook, reproduce the full offline Kaggle run, and fix the
sklearn 1.9.1 (train) vs 1.6.1 (Kaggle) mismatch before any submission.

Reopen if: DeBERTa-base shows no validated pool gain (try larger context/model within
budget, or cached-embedding readouts Phase 2), or GPU quota/runtime blocks the run.

## D022 — DeBERTa-v3-base + pool helps the encoder but still loses to the sparse blend (2026-11-15)

First Kaggle GPU run (T4, kernel v2; the v1 `Attempting to unscale FP16 gradients`
crash was fixed by upcasting DeBERTa-v3's fp16 checkpoint to fp32 master weights
before AMP — `build_model` now calls `model.float()`). Grouped 5-fold CV,
`deberta-v3-base`, max_len 256, fp16, batch 16, lr 2e-5; official-only 3 epochs,
official+pool 1 epoch with a stratified **40k/fold** cap (registry
`deberta_v3_base_*`, `artifacts/deberta_kernel_output/deberta_cv/`):

| variant | pooled OOF BA | fold BAs | fit s | peak GPU GB |
|---|---|---|---|---|
| official-only | **0.2241** | 0.217/0.183/0.226/0.252/0.243 | ~793 | 6.46 |
| official+pool (40k/fold, 1ep) | **0.3049** | 0.294/0.300/0.307/0.317/0.306 | ~5170 | 6.46 |

- **Pool gain is real:** paired per-fold diff **+0.0808**, 95% CI **[+0.054, +0.108]**
  (excludes 0). External data substantially helps the encoder.
- **But 0.305 < the adopted 0.376 sparse blend.** DeBERTa-base at this budget does
  **not** beat the fallback. The community claims (official-only ~0.39, pool ~0.576)
  are **not reproduced** with base / len256 / 1 pool epoch / capped pool — consistent
  with the owner's warning not to trust the strategy's claimed gains.
- **Benchmark:** 31–40 seq/s, peak **6.46 GB** (T4 16 GB has headroom for a larger
  model), ~2.2 h total ≪ 9 h budget. Training healthy (loss 2.3→1.6).
- **Fold caveat:** the kernel recomputed folds with **Kaggle's sklearn** (version
  differs from local 1.9.1), so its per-fold decon counts ([7,4,8,1,2]) differ from
  the local cv_s42 audit ([4,1,3,6,5]); the OOF is **not yet fold-aligned** with the
  local sparse OOF. The within-kernel official-vs-pool paired diff is still valid
  (same folds). Next run must load the persisted `splits_official.json` for exact
  alignment before any cross-pipeline blend.

Decision:
1. **Keep the 0.376 char+style sparse blend as the submission fallback** (LB 0.356).
   Do not adopt DeBERTa-base.
2. Next GPU run: **stronger config within budget** — `deberta-v3-large`, longer
   context (384), more pool epochs, exact local folds shipped to the kernel — then
   test whether DeBERTa adds **diversity** to the sparse blend (fold-aligned OOF).
3. Fix the sklearn train/Kaggle mismatch (ship folds + pin versions) before any
   submission reproduction.

Reopen if: a stronger DeBERTa config beats 0.376 on CV **and** the reserved seed-2027
holdout, or a fold-aligned DeBERTa+sparse blend shows a validated gain.

## D023 — DeBERTa-v3-large + pool beats the sparse blend; fold-aligned 50/50 blend = 0.444 CV candidate (2026-11-15)

v3 (large, len384, batch8) **failed**: official-only 0.2245 had fold 0 collapse to
0.0537 (~chance) near the T4 ceiling (13.67 GB) and the pool variant CUDA-OOM'd. Two
root causes: (a) **Kaggle script kernels do NOT upload companion files** — the shipped
`folds.json` was dropped so folds were recomputed on Kaggle (not aligned to cv_s42);
(b) large@batch8/len384 exceeds T4 memory in the disentangled attention.

**v4 fixes** (`kaggle/deberta/deberta_train.py`): local cv_s42 folds **embedded in the
script** as a per-row fold-id array (verified byte-identical to `splits_official.json`,
reconstructs the exact folds); **batch 4 + gradient checkpointing** (`use_reentrant=False`)
+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` + per-fold `gc.collect()`/peak-reset;
pool cap 12k/fold. Benchmark: 6.7 seq/s, peak **10.2 GB** (fits), 7.2 h < 9 h.
`split_method = stratified_group-embedded`, so the OOF is **fold-aligned** with the local
sparse OOF — the first DeBERTa result safe to blend across pipelines.

| variant (large, embedded folds) | pooled OOF BA | fold BAs |
|---|---|---|
| official-only (3ep) | **0.2482** | 0.263/0.163/0.294/0.213/0.307 |
| official+pool (12k/fold, 2ep) | **0.3857** | 0.401/0.407/0.396/0.352/0.372 |
| **0.5·(0.5 char+0.5 style) + 0.5·DeBERTa-pool** | **0.4442** | 0.456/0.467/0.431/0.428/0.439 |

- **Encoder pool gain is real:** official-vs-pool paired fold diff **+0.1375**, CI
  **[+0.054, +0.221]** (excludes 0). The pool is the lever, not model size (large
  official-only 0.248 ≈ base 0.224).
- **The blend gain is decisive:** DeBERTa-pool **standalone 0.3857 already beats the
  0.376 sparse blend**; the 50/50 blend reaches **0.4442**, paired vs sparse
  **+0.0682, CI [+0.048, +0.089]** (excludes 0). 50/50 is the pre-registered equal-weight
  default, not a tuned grid. DeBERTa alone vs sparse is *not* significant
  (+0.0097, CI [−0.017, +0.036]) — the improvement is **diversity**, exactly D022's
  hypothesis.
- **Per-class:** improves **15/18** classes (gemini-3.8-flash +0.28, claude-fable-5
  +0.18, claude-opus-4-6/4-8 +0.18, gpt-5.5/luna +0.14); cross-provider errors drop
  787→577. **Regression to watch:** gemini-3.7-flash collapses 0.487→0.100 — likely
  the pool's gemini-3.7 quality (D019 moved the 10,838 unverified 3.7 rows to
  `gemini37_unverified.parquet`, so our pool 3.7 may be thin/noisy and DeBERTa mis-buckets
  3.7 into its 3.6/3.8 flash siblings).

Decision:
1. **Candidate, NOT yet adopted.** Keep the 0.376 sparse blend as the current submission
   fallback (LB 0.356). Do not submit the 0.444 blend until protocol confirmation.
2. **Confirm before adoption** (D018 discipline): re-run DeBERTa-pool on an **alternative
   fold seed (s13)** with embedded folds and re-blend against the s13 sparse OOF; then
   validate on the **reserved seed-2027 holdout** for final selection. The blend must win
   on CV **and** holdout with the CI excluding 0.
3. Investigate the **gemini-3.7-flash** regression: check the clean pool's 3.7 class
   counts and per-fold decon impact; consider down-weighting or excluding pool 3.7, or a
   sibling residual head (Phase 5), before final submission.
4. **Submission engineering (before any submit):** produce final DeBERTa weights
   (`--save-final`) on official+pool, package offline, reproduce the full offline Kaggle
   blend (DeBERTa + char+style), and fix the sklearn 1.9.1 (local) vs 1.6.1 (Kaggle)
   mismatch so the sparse members reload identically.

Reopen/revise if the s13 or seed-2027 confirmation does not reproduce the +0.068 blend
gain, or if the gemini-3.7 regression is found to hurt net BA on the holdout.


## D024 — Re-establish OOF provenance and prompt isolation before adoption (2026-11-16)

Review identified that legacy sparse and DeBERTa OOF files do not store sufficient
row and fold identity, and the external pool's `prompt_canon` is not proven to
share a namespace with official `prompt_id`. The reported 0.4442 blended OOF is
therefore provisional, not an independently verified result. The seed-2027 subset
has not been scored separately, but its prompts were part of earlier full OOF
development, so it is not a fully untouched evaluation. The prior transductive
stress-test entries are historical exploratory diagnostics and do not satisfy the
repository's hard rule against transductive steps in reported scores.

Decision: require exact dataset/row/fold provenance for new OOF comparisons;
require provenance-backed prompt isolation and training-side-only learned
transforms for pool experiments; and use genuinely new prompts or pre-specified
nested grouped evaluation for a stronger assessment. Keep the 0.356-public
submission as fallback. Record all reruns honestly and keep the old results as
historical, provisional measurements. Confirm the full offline inference recipe
and exact label order before considering a new submission.

Reopen if: verified prompt provenance and row/fold identities are established,
and a non-transductive grouped evaluation plus offline inference support adoption.


## D025 — Fail closed on OOF selection and missing weighted models (2026-11-16)

The generic ensemble command previously loaded every OOF file in the directory
and compared only class labels, including models rejected elsewhere. Prediction
then silently discarded weights for absent bundles. This could submit a different
model from the one evaluated. The baseline notebook had the same silent filter
and could activate a withdrawn crossed path.

Decision: require ordered dataset fingerprints and exact fold assignments for
all OOF members, restrict the CLI selector to explicitly configured members,
and fail if weighted model bundles are missing. Official prediction and the
submission notebook require an explicit nonempty finite weights file; custom
model directories do not inherit unrelated global weights. The submission
notebook rejects the withdrawn transductive crossed path. Greedy selection writes `candidate_weights.json` by default instead of replacing
active weights, records whether the configured minimum OOF gain is met, and
remains an exploratory within-OOF measurement rather than independent confirmation
or automatic adoption.

Reopen if: a new member is explicitly configured and evaluated under the same
verified grouped folds, with independent validation before submission.

## D026 — Disable transductive pool selection for reported CV (2026-11-16)

The historical audit combines removals across all held-out folds into one clean
ID list. Using that list in CV lets validation examples influence the pool used
for training. Even the per-fold procedure selects pool rows by comparing their
texts with held-out passages, which is transductive regardless of where the
similarity vocabulary is fitted. A verified prompt crosswalk alone does not fix
either issue.

Decision: disable the local and Kaggle external-pool CV routes, including direct
training entry points, until a non-transductive pool protocol is designed and
validated. The global audit ID list and per-fold overlap reports are diagnostic
only. Continue the official-only baseline; historical pool results remain
provisional. Reopen with independently justified pool eligibility, prompt
isolation, and no validation-text-dependent training selection.


## D027 — Provenance-verified official-only DeBERTa adds no gain; keep the sparse blend (2026-11-16)

The `name-that-llm-official-only-identity-cv` kernel completed (deberta-v3-large,
len384/batch4, 3 epochs, `stratified_group-s42-embedded`, OOF sha `dc60875f…`, folds
539/540/540/540/540). It writes provenance-bearing OOF, so `experiments/blend_deberta.py`
passes the D025 fingerprint/fold alignment guard. The identity-regenerated sparse OOF
(`artifacts/cv_identity/official/`) reproduces the adopted **0.3760** exactly, confirming the
D024 regeneration is faithful and selects no withdrawn model.

Fold-aligned results (paired 95% CI vs the 0.3760 sparse blend):

| candidate | pooled OOF BA | paired vs sparse |
|---|---|---|
| sparse_adopted | 0.3760 | — |
| DeBERTa official-only | 0.1915 | −0.1844 [−0.308, −0.061] |
| sparse+DeBERTa 50/50 | 0.3393 | −0.0367 [−0.117, +0.044] |
| sparse+DeBERTa 70/30 | 0.3667 | −0.0092 [−0.044, +0.026] |
| sparse+DeBERTa 30/70 | 0.2974 | −0.0785 [−0.200, +0.043] |

Decision:
1. **No adoption.** With the pool route disabled (D026), the DeBERTa-v3-large encoder at the
   current official-only budget does **not** beat or improve the 0.3760 sparse blend — every
   blend is at or below sparse with CIs spanning zero, and DeBERTa standalone is significantly
   worse. This confirms D023's finding that the (now-disabled) **pool**, not the encoder, was the
   only source of gain.
2. **Keep the 0.356-public / 0.3760 sparse blend as the active submission fallback.** Do not
   prepare or submit an official-only DeBERTa blend; it is not a validated candidate.
3. The still-running `official-vs-pool` kernel is **diagnostic only** under D026; its numbers stay
   provisional and are not an adoption basis until a non-transductive pool protocol exists.
4. Reopen the encoder route only if (a) a **non-transductive pool eligibility** protocol is
   designed and validated (D026), or (b) an official-only encoder becomes competitive
   (larger token budget / sibling-aware head / curriculum) on grouped CV **and** the reserved
   seed-2027 holdout.

Registry: `20261116T053042-official-deberta-officialonly-identity-blend`.


## D028 — Non-transductive global pool decontamination protocol (2026-11-16)

D026 disabled the pool CV route because `per_fold_decontamination` screened the pool
against each fold's **held-out** passages, so validation text changed which pool rows
entered training. The fix is to make the clean pool **fold-independent**:

- **`src.pool.decontaminate_global(pool, train_df)`** screens the pool **once** against the
  complete official training set — exact text matches plus word TF-IDF cosine similarity
  (vectorizer fit on that same full train set) above `cos_threshold` (default 0.3). The
  reference is known training data, never a fold's held-out subset, so the returned keep-mask
  is identical for every fold and cannot encode validation information.
- `experiments/pool_train.py` now computes the clean pool once, then reuses the **same** rows
  in every fold; per-fold `cap_pool_stratified` only randomly subsamples the *identical* clean
  set for a compute budget (independent of validation text and labels).
- Because the pool's `prompt_canon` namespace is not proven to match official `prompt_id`
  (D019/D024), this protocol uses **text-level** decontamination only and does **not** require
  the unverified prompt crosswalk; a separately verified crosswalk may later tighten it.
- Legacy `decontaminate_fold` / `per_fold_decontamination` are retained but deprecated; their
  union-removed lists stay **diagnostic only**. `--clean-ids` fails closed (IDs are recomputed).

This reopens the encoder/pool lever honestly: a pool gain that survives `decontaminate_global`
is non-transductive and admissible. Validation is cheap-first: the **sparse** model on CPU
(`configs/baseline.yaml --models char_tfidf_logreg`) confirms the gain before any GPU rerun, then
the DeBERTa-pool kernel must adopt `decontaminate_global` and win grouped CV + the reserved
seed-2027 holdout (D018/D024) to displace the 0.3760 sparse fallback.

**Sparse CPU validation (2026-11-16, registry `official_pool_global-char_s42`): the gain
SURVIVES the non-transductive protocol.** `decontaminate_global` screened 226,191 → **226,049**
clean (exact 0 + near 142 removed), one fold-independent keep-mask reused by all folds; cap
12,000/fold class-proportional (validation-independent compute budget). `char_tfidf_logreg`
official-only **0.3193 → +pool 0.3667** (delta **+0.0474**); paired fold diff **+0.0474 ± 0.0257
[+0.0217, +0.0731]**, **all five folds positive** (per-fold +0.0185 … +0.0667). The per-class
signature **matches the earlier transductive run** (gemini-3.8-flash +0.227, gemini-3.6-flash
+0.147, qwen3.8-max +0.127 up; gemini-3.7-flash −0.173 and kimi-k3 −0.093 down), which is the
decisive evidence that the pool effect was **genuine coverage, not the transductive leak** D026
feared. **But `style_only`+pool is NOT helped**: official-only 0.2904 → +pool 0.2730 (paired
−0.0174 ± 0.0306 [−0.0480,+0.0132], CI spans 0). The pool is a **char-tfidf-specific** gain.

**Blend result (2026-11-16, registry `official_pool_global-blend_mix_s42`): the pool reopens a
real gain over the fallback.** Blends were rebuilt with the fallback's own rule — row-wise
`normalize_scores` (z) then fixed 0.5/0.5 weights — which reproduces the 0.3760 sparse blend
**exactly** (so the comparison is apples-to-apples and scale-invariant). Candidates vs 0.3760:

| blend (fixed weights, no in-sample re-selection) | pooled OOF BA | paired vs 0.3760 [95% CI] |
|---|---|---|
| char+pool, style official-only (MIX) | **0.3990** | **+0.0230 [+0.0079,+0.0381]** |
| 4-member (off char, off style, pool char, pool style) | 0.4012 | +0.0252 [+0.0058,+0.0446] |
| both members +pool | 0.3812 | +0.0052 [−0.0183,+0.0287] (CI spans 0) |

The clean adoption-shaped candidate is **MIX = 0.5·z(char_trained_with_pool) + 0.5·z(style
official-only) = 0.3990**, a **+0.023 CI-positive** improvement, at cap 12,000/fold (the full
clean pool is 226,049 — headroom remains). It is non-transductive (D028 global screen), uses
fixed weights, and never touches `match_weight`/`style_score` (D019).

**Alternative-seed confirmation (s13, registry `official_pool_global-blend_mix_s13`):** the same
rule on `splits_official_verified_s13.json` with the identical fold-independent mask (226,191 →
226,049) gives char official-only **0.3141 → +pool 0.3686** (paired **+0.0545 [+0.0276,+0.0813]**,
all folds positive) and **MIX = 0.3989 vs the s13 base blend 0.3582 = paired +0.0408 ± 0.0207
[+0.0200,+0.0615], all five folds positive** (+0.017…+0.063). MIX pooled **0.3989 (s13) ≈ 0.3990
(s42)** — **stable across both grouped-CV seeds, both CIs exclude 0**, which satisfies the D018
two-seed requirement. **Not yet adopted:** the only remaining gate before it may displace the
0.3760/0.356 fallback is the **reserved seed-2027 holdout** (D018/D024). Until that clears, the
0.356 LB / 0.3760 sparse blend stands as the submission and nothing is submitted.

Reopen/revisit if a verified prompt crosswalk is established, or if the fold-independent global
screen is shown to leak (it must not depend on held-out text or labels).

## D035 — Pool-MIX candidate passes the nested final gate; wire a pool-augmented submission (2026-11-16)

D028 reopened the pool non-transductively and the fixed-weight MIX blend beat the 0.3760 fallback on
both 5-fold grouped seeds (s42 0.3990, s13 0.3989, CIs exclude 0). The plan's remaining gate was a
**nested grouped evaluation**, and it corrected the belief that the seed-2027 subset is a clean holdout:
it **participated in earlier full-OOF development**, so it is **examined** and must not be presented as
an untouched final pass (improvement_plan §4/§7).

`experiments/nested_pool_blend.py` (+`tests/test_nested_pool_blend.py`) evaluates the exact submission
recipe on grouped **outer folds** (n=3, seed 121): each fold refits char on outer-train + the pool
screened **only against outer-train text** (the outer holdout never changes pool eligibility, so the
estimate is fold-independent and non-transductive), refits style on outer-train, and applies the
**fixed** 0.5/0.5 row-wise-z blend identical to the fallback rule — there is no inner selection, hence
no CV-selection optimism to remove. Result: official-only pooled **0.3519 → MIX 0.3930** (delta
**+0.0411**), paired outer-fold diff **+0.0411 ± 0.0339 [+0.0072,+0.0751]**, **all three outer folds
positive** (+0.036/+0.031/+0.057). Per-fold near-dup removals (46/63/69) confirm the fold-internal
screen. MIX 0.393 also exceeds the 0.3760 5-fold fallback despite training on only 2/3 of the rows.

Decision: the pool-MIX recipe is **validated under three independent grouped protocols** (s42, s13,
nested outer). It becomes the **recommended submission candidate** above the 0.3760/0.356 fallback.
Because the notebook's `blend_score_matrix` already z-blends the packaged bundles with fixed weights,
**no inference-code change is needed** — only the final fit must train the char bundle on
(full official train + the global clean pool at the validated 12k cap); `src.inference.fit_final` gains
optional pool params and `cli predict --fit --pool ...` wires them. `match_weight`/`style_score` remain
unused (D019); the pool is screened vs the full train at final fit (no holdout exists there).

**Compliance:** submitting is **owner-only** (AGENTS.md); this repo prepares `artifacts/kaggle_dataset`
+ commands, never runs `kaggle competitions submit`. The real hidden test is not present locally (the
`data/test.csv` stub is 3 rows); `submission.parquet` is produced by the Kaggle notebook. Do not treat
the examined seed-2027 subset as the adoption gate.

Reopen/revisit if a genuinely new labelled prompt set or the owner's submission disagrees with the
nested estimate, or if the 3.7-flash sibling regression (−0.17 per class, task #34) proves material on
the hidden test.

---

## D036 — gemini-3.7-flash regression root-caused; class-balanced pool cap tested and rejected (2026-11-16)

Context: D035 left one open thread — the pooled `char_tfidf_logreg` member loses
gemini-3.7-flash recall (−0.173) while its flash siblings gain (3.6 +0.147, 3.8 +0.227),
and the DeBERTa fold-aligned blend showed a sharper 0.487→0.100 collapse. Task #34 asks
whether to guard that sibling before the submission widens the pool budget or retrains DeBERTa.

Root cause (quantified): the official train set is class-**balanced** (150 rows/class) but
the Open Pool is heavily skewed, and a **proportional** cap inherits that skew. At cap 12k
the pooled char member sees gemini-3.7-flash at only **188 rows** against siblings
**974 (3.6) / 1246 (3.8)** — a ~5× under-representation — so the linear model routes held-out
3.7 into the better-represented flash siblings. It is coverage starvation, not a leakage artifact.

Experiment: added `--cap-mode {proportional,balanced}` to `experiments/pool_train.py`.
`cap_pool_balanced` applies a **uniform per-class target** (`cap // n_classes`, all rows kept
when a class is thinner), matching the official uniform prior; selection stays
validation-independent (class counts + fixed seed only). Tests: `tests/test_pool_cap_modes.py`.

Result:
- char standalone: gemini-3.7-flash flips **−0.173 → +0.013**; family still up (3.8 +0.133,
  3.6 +0.060, 3.1 +0.047, qwen +0.193); overall char BA **0.3678** vs proportional **0.3667**
  (paired +0.0486 [+0.0292,+0.0680], a wash).
- MIX blend (`0.5·z(char-pool) + 0.5·z(style official)`): balanced **0.3949** vs the validated
  proportional candidate **0.3990**; paired fold diff **−0.0041 ± 0.0075 [−0.0116,+0.0034]** —
  CI spans 0, point estimate slightly **negative**. The balanced cap recovers gemini-3.7 (+0.133)
  but forfeits more on gemini-3.8 (−0.120), gemini-3.6 (−0.033) and claude-opus-5 (−0.047).

Decision: **keep the validated proportional-cap MIX candidate** (0.3990 / 0.3989 / 0.3930). The
class-balanced cap is aggregate-neutral-to-negative; the gemini-3.7 weakness is real but is more
than paid for by the net blend gain, and a per-class override hand-tuned on CV would be
selection-optimistic (violating the no-CV-tuning spirit of D024/D028). No submission change.

Reopen only if the owner's hidden-test LB shows a gemini-3.7-flash collapse materially worse than
CV, or if a principled (non-CV-selected) class-balancing prior is introduced for DeBERTa retraining.

---

## D037 — pool cap widened 12k → 40k/fold; new validated submission candidate (2026-11-16)

Context: D028/D035 capped the global clean pool at 12,000 rows/fold purely as a CPU
budget guard, but 226,049 clean rows leave large headroom. Task #39 tests whether the
cap — not the recipe — was the binding constraint.

Experiment (`experiments/pool_train.py --cap-mode proportional --pool-cap 40000`):
- char standalone s42: 0.3193 → **0.4104** (+0.0911 [+0.0721,+0.1103], all folds +).
- char standalone s13: 0.3141 → **0.4115** (+0.0974 [+0.0584,+0.1364]).

Rebuilt the fixed 0.5/0.5 row-wise-z MIX blend (`0.5·z(char trained with pool) + 0.5·z(style official)`):

| protocol | cap 12k | cap 40k | paired 40k vs 12k |
|---|---|---|---|
| s42 5-fold | 0.3990 | **0.4427** | +0.0437 [+0.0238,+0.0636] |
| s13 5-fold | 0.3989 | **0.4304** | +0.0315 [+0.0089,+0.0540] |
| nested outer gate (D035) | 0.3930 | **0.4426** | +0.0908 [+0.0460,+0.1355] |

The nested gate (`experiments/nested_pool_blend.py --pool-cap 40000`) refits char on each
outer-train + pool screened vs **outer-train only** (non-transductive D028), all 3 outer
folds positive (0.450/0.433/0.445). vs the 0.3760 official-only base the 40k MIX gains
+0.0667 [0.0432,0.0903] (s42) / +0.0722 [0.0461,0.0983] (s13).

Per-class: the D036 gemini-3.7-flash sibling starvation **grows** with a larger
proportional cap (nested 3.7 −0.127) but is overwhelmed by sibling gains (3.6 +0.160,
3.8 +0.273, kimi +0.067) → large net positive. Balanced-at-40k (fix 3.7 with more data)
is an optional future probe, not required for adoption.

Decision: the cap was the binding constraint. **Promote cap-40k proportional MIX as the
submission candidate** (0.4427 s42 / 0.4304 s13 / 0.4426 nested), replacing the 12k
candidate. Rebuild `artifacts/final/official` + `artifacts/kaggle_dataset` at
`--pool-cap 40000` (recipe/weights/blend-rule unchanged; only the char bundle's pool
budget grows). Compliance unchanged: submitting is **owner-only**; grouped-CV scores are
not an LB claim until the owner submits; 0.356/0.3760 remains the confirmed fallback.

Revisit if the owner's hidden-test LB contradicts the +0.06 CV gain (bigger pool may over-fit
the community distribution), or if balanced-at-40k beats 40k-proportional under the nested gate.

**Balanced-at-40k probe (resolves D036's optional item):** re-tested the class-balanced cap at the
wider budget, since gemini-3.7 starvation grows at 40k and 40k-per-class is still ample for big
classes. char standalone **does** improve (bal40k 0.4234 > prop40k 0.4104; uniform ~2222/class feeds
gemini-3.7 the data proportional starves it). But the **MIX blend regresses**: bal40k 0.4378 vs
prop40k 0.4427, paired −0.0048 [−0.0217,+0.0120] (CI spans 0, point negative) — balanced forfeits
gemini-3.8 (−0.140)/3.6 (−0.040) more than it gains on 3.7, and official-only `style` already covers
3.7 so the blend doesn't need char to. **Rejected; the cap-40k proportional candidate stands.** D036's
decision generalizes to the wider cap.

## D038 — DeBERTa kernel re-opened on the non-transductive global pool path (2026-11-16)

Context: D026 disabled *all* external-pool CV because the per-fold screen was
transductive (a fold's held-out passages selected which pool rows got trained on).
The DeBERTa kernel `kaggle/deberta/deberta_train.py` enforced that with a hard abort
(`run_cv(augment=True)` and `main()` both raised), and its pool-loading code was
stripped so `pool` stayed `None` unconditionally — the encoder-pool lever was
unrunnable even after D028 validated a non-transductive replacement.

Decision: re-open the pool route for the kernel under D028 only. Concretely:
- add a kernel-local `decontaminate_global(pool, train_df)` mirroring `src/pool.py` —
  screen the pool **once** against the full official training set (exact + word
  TF-IDF cos>0.3, vectorizer fit on that same full train); the keep-mask is
  fold-independent and text-level (no prompt-crosswalk namespace assumption, D019/D024);
- restore pool loading (skip gracefully if no `ntl_open_pool.parquet` attached);
- `run_cv` consumes the pre-screened pool and applies only `cap_pool` per fold;
  `--pool-scope fold` (transductive) still raises, `global` is the default;
- default `--pool-cap-per-fold` raised 12k→**40000** to match D037's validated budget;
- the `--save-final` fit uses the globally-clean pool instead of the old
  `prompt_canon`-known filter.

Compliance / scope: this is enabling infrastructure, **not** a scored experiment — no
DeBERTa-pool metric is recorded (none exists). The GPU run and any fold-aligned blend
vs the 0.4427 sparse pool-MIX candidate are **owner/external** (local torch is CPU-only)
and must clear the same three grouped protocols (s42 / s13 / nested) before adoption.
If it does not beat 0.4427 under all three, keep the sparse pool-MIX candidate.

## D039 — Blend evaluator rebased on the adopted candidate; D037 numbers independently reproduced (2026-11-16)

Context: `experiments/blend_deberta.py` still compared new members against the
**official-only 0.376 sparse blend**, which is no longer the bar — the adopted
candidate is the cap-40k pool-MIX at 0.4427 (D037). Any DeBERTa-pool arm evaluated
after the D038 kernel run would have been measured against the wrong baseline. Also,
all three D037 headline numbers came from `pool_train.py`/`nested_pool_blend.py`, i.e.
one toolchain, so the candidate's score had no independent reproduction.

Decision: rebase the evaluator and use it as the reproduction check.
- `--char-oof` / `--style-oof` let each MIX member be addressed separately, so the
  baseline is `0.5 z(char trained on GLOBAL pool cap-40k) + 0.5 z(style official-only)`
  instead of two official-only members; the report key is now `mix_baseline`
  (`paired_vs_mix`), and `--no-deberta` scores the baseline alone.
- Provenance stays mandatory: each member is loaded through `load_oof` (needs
  `ordered_dataset_sha256` + `validation_fold_ids`) and `verify_oof_alignment`
  rejects any member whose folds differ from the requested splits.

Measured (stored artifacts, no GPU): **s42 0.442675**, **s13 0.430435**, and the
official-only control **0.375971** — matching the recorded 0.4427 / 0.4304 / 0.3760 to
four decimals. The candidate's headline is therefore reproducible by a second code path
under enforced fold+hash identity, not just by the tool that produced it.
`tests/test_blend_deberta_baseline.py` pins the rule, the per-member overrides and the
misalignment rejection.

Also set `FOLD_SEED = "s42"` in the D038 kernel so the owner's next pool run is
comparable to the candidate's primary protocol. Compliance unchanged: no DeBERTa-pool
number exists yet; grouped-CV reproduction is not an LB claim; submission stays owner-only.

## D040 — Packaged artifacts must not carry a CV score measured for a different fit (2026-11-16)

Context: auditing the upload-ready package for the owner's submission step (#38) found
two provenance defects. `artifacts/kaggle_dataset/manifest.json` correctly recorded
`char_tfidf_logreg` as `fit_mode: train+global_pool(capped=40001)`, but `cmd_package`
always copied `artifacts/cv/<dataset>/ensemble/weights.json` — the **official-only** CV
run's document — over it. That file asserted `pooled_oof_balanced_accuracy: 0.37597` and
a note describing the pre-pool sparse blend, i.e. it attributed an official-only score to
a pool-trained bundle. Separately, `kaggle/submission.ipynb` still headed its candidate
section "official-only sparse blend (D026)" and stated "External-pool candidates are
disabled under D026", while printing the `--pool-cap 40000` build command directly below.

The operative weights were identical (0.5/0.5), so **predictions were never affected**;
the defect was in what the artifact claims about itself. Under the repository rule against
fabricated or misattributed scores, a shipped document must not describe a fit it does not
contain.

Decision:
- `cmd_package` now calls `_align_packaged_weights`: when any shipped member's `fit_mode`
  mentions the pool, the official-only descriptive fields (`pooled_oof_balanced_accuracy`,
  `best_single`, `gain_over_best_single`, `singles`) are removed and replaced with the
  manifest's `fit_modes` plus `cv_score_of_record: experiments/registry.csv`. The score of
  record stays in the registry, where it is tied to a run id, seed and artifact path —
  never copied into a package that cannot verify it. For an all-`train_only` package
  (the 0.356 fallback) the fields are kept unchanged, because there they are correct.
- The notebook's candidate section now names the cap-40k pool-MIX blend (D028/D037),
  states the global fold-independent screen and the 226,191 → 226,049 clean-row count,
  labels 0.4427 / 0.4304 / 0.4426 explicitly as **local CV, not a leaderboard score**, and
  keeps the official-only 0.356 submission as the confirmed fallback with rebuild instructions.

Verified: `tests/test_package.py` gained two cases (pool-trained member drops the
official-only fields and keeps the weights; all-`train_only` keeps them). Full suite green.
The package was rebuilt from the intact `artifacts/final/official` and re-run offline in a
notebook-equivalent simulation: bundles load, class order matches `LABELS`, weights 0.5/0.5,
and a schema-valid `submission.parquet` is written for the 3 example test rows (smoke check
only — not a submission). No new experiment was run, so no registry score row is added.
Submission remains owner-only; the LB gate (#38) is still open.

## D041 — Length-band error analysis protocol; the pool gain is length-dependent (2026-11-17)

Context: plan §9 item 9 requires text-length sensitivity in the report's error analysis.
Until now §8 carried only the confusion analysis and admitted the length dimension had not
been run; the report also mis-described it as "wired into the audit" (the audit produces a
length *distribution* figure, not per-band accuracy).

Decision — the diagnostic is a **post-hoc analysis of saved OOF**, never a new fit:
- `src.metrics.length_band_metrics(y_true, y_pred, lengths, edges)` scores bands
  `[edges[i], edges[i+1])` with an open-ended top band. A band's balanced accuracy averages
  only the classes present in it, and `classes_present` is reported so a reader cannot
  mistake a narrow band for a pooled score. Empty bands return `None`, not 0.
- `experiments/length_sensitivity.py` loads the official train frame, requires its
  `ordered_dataset_fingerprint` to equal the split fingerprint, and requires every member's
  OOF fingerprint, labels and `validation_fold_ids` to match the frozen splits before
  scoring. It rebuilds both blends with the adopted rule (0.5 row-z char + 0.5 row-z style)
  and fits nothing, reads no pool row and scores no test row — so it cannot introduce a
  transductive step (D026) or a hidden-test tuning step.
- Bands are word counts under the same rule as `src.audit`: **(0, 50, 100, 150, 200)**. The
  corpus spans 36–274 words, so the top band is left open-ended to keep every band above 300
  rows instead of scoring a 25-row tail. All 18 classes occur in all five bands here, which
  makes these particular band scores mutually comparable.
- The per-band MIX − official difference is reported as a **paired point difference on the
  same rows**, not as a confidence interval: band-level resampling would need a new
  bootstrap protocol, and the fold-level paired CIs of D037/D039 remain the gate.

Measured (registry rows `*-length_bands`, `artifacts/length_sensitivity/results.json`):
- The run reproduces both headline scores exactly through a third code path:
  official-only **0.375971** and pool-MIX cap-40k **0.442675**, plus the singles
  char 0.319306 / style 0.290398 / char+pool 0.410445.
- Accuracy rises monotonically with length for every variant: official blend
  0.2491 → 0.3393 → 0.3841 → 0.4457 → 0.4718; candidate
  0.3125 → 0.3824 → 0.4534 → 0.5356 → 0.5858.
- The pool gain is **length-dependent**: +0.0634 / +0.0431 / +0.0693 / +0.0899 / +0.1140.
  It is largest on the longest passages and smallest in the 50–100 band, so external data
  amplifies existing evidence rather than repairing the short-passage failure mode.
- Within-provider share of errors rises with length (official 40.9% → 66.7%, candidate
  40.0% → 64.1%): short passages fail at provider level, long ones at version level (H4).
- The candidate removes 180 errors, 114 of them within-provider (63%); 14 of 18 classes
  improve, `gemini-3.8-flash` +0.253 and `gemini-3.7-flash` −0.180, i.e. the D036/D037
  starvation trade is visible per class, not only in the aggregate.

Consequences: report §8 is split into §8.1 (confusion) and §8.2 (length sensitivity) with the
measured table; §11 lists the command; §12 now shows ten figures, including
`length_band_accuracy.png` and the candidate's `confusion_matrix_pool_mix_cap40k.png` /
`per_class_recall_pool_mix_cap40k.png`. `src.figures` gained `make_prediction_figures`
(extracted from `make_report_figures`, same output) and `make_length_band_figure`.
No protocol gate changed: the candidate, its weights and the fallback are untouched, and
**no leaderboard number is claimed**. The local `test.csv` is the 3-row interactive example,
so the report states explicitly that the hidden test's length mix is unknown and that any
length explanation of the −0.020 CV/LB gap is an untested hypothesis.

## D042 — Blend-rule comparison: row-z stands, min-max is indistinguishable (2026-11-17)

Context: plan §7 asks for simple averaging, rank-based and normalized-margin blending to be
compared against the adopted rule. The adopted rule (D020) is `0.5·z(char) + 0.5·z(style)`
with row-wise z-scores, but that choice had never been measured against alternatives, and the
report once carried an untraceable "raw average 0.3241" figure that had to be deleted.

Decision — compare five scale-only rules on the **same saved OOF**, changing nothing else:
`raw_mean`, `row_z_mean` (adopted), `rank_mean` (average ranks mapped to [0,1]),
`minmax_mean` (per-row min-max), `softmax_mean`. `experiments/blend_rules.py` reuses the D041
identity guards (dataset fingerprint, per-member OOF fingerprint/labels/fold ids), fits
nothing, reads no pool row and scores no test row. Weights stay 0.5/0.5 and both member pairs
are scored, so a rule is only interesting if it wins on the candidate **and** does not lose on
the fallback.

Compliance rulings carried into the implementation:
- **Margins are not probabilities.** `softmax_mean` is admitted only as a monotone-transform
  diagnostic; it is never described as a calibrated probability and never used for calibration.
- Absent-class sentinel cells (`src.ensemble.SENTINEL`) are masked to the rule's neutral value
  (0.5 for rank/min-max, 1/k for softmax, the row mean for raw, 0 for row-z) so a rule cannot
  be scored on a fabricated extreme. Both pairs contain **zero** sentinel cells here.
- A rule that wins this OOF is still only mildly optimistic evidence: adoption additionally
  requires beating row-z on the alternative seed (s13) and the nested outer gate (D035/D039).

Measured (registry rows `*-blend_rules`, `artifacts/blend_rules/results.json`):

| rule | official-only BA | Δ vs row-z (95% CI) | pool-MIX cap40k BA | Δ vs row-z (95% CI) |
| --- | --- | --- | --- | --- |
| raw_mean | 0.324114 | −0.0518 [−0.0747, −0.0289] | 0.420808 | −0.0219 [−0.0269, −0.0169] |
| **row_z_mean (adopted)** | **0.375971** | — | **0.442675** | — |
| rank_mean | 0.361921 | −0.0140 [−0.0296, +0.0015] | 0.406028 | −0.0367 [−0.0603, −0.0130] |
| minmax_mean | 0.375973 | −0.0000 [−0.0096, +0.0096] | 0.440823 | −0.0019 [−0.0055, +0.0018] |
| softmax_mean | 0.302988 | −0.0730 [−0.0984, −0.0475] | 0.401158 | −0.0415 [−0.0553, −0.0277] |

- **Row-z is best on the candidate** (`best_rule = row_z_mean`) and every alternative loses
  there, three of them with a paired CI excluding zero. The adopted rule is confirmed, not
  replaced.
- On the fallback the script's `best_rule` is `minmax_mean` by **2e-6** — rounding noise, not a
  win: the paired CI spans zero and min-max is worse on the candidate. It is recorded as
  *indistinguishable*, and no adoption follows from it.
- Rank averaging loses on both pairs; raw averaging and softmax lose badly on both. Raw-margin
  averaging is dominated by whichever member has the larger margin scale, which is exactly the
  scale mismatch row-z removes. The previously untraceable 0.3241 is now a measured, citable
  number.
- The ordering is stable across the two member pairs, so the conclusion does not depend on the
  pool cap or on which char member is used.

Consequences: `tests/test_blend_rules.py` (9 cases) pins the transforms — hand-computed ranks
and min-max, softmax positivity/ordering/sum, neutral handling of a constant row and of the
sentinel, finiteness of an all-absent row, exact agreement of the `row_z_mean` arm with
`src.ensemble.normalize_scores`, rejection of an unknown rule, and blend = weighted sum.
Report §5 carries the table. **No config, weight, artifact or packaged dataset changes**: the
candidate and the fallback are byte-identical, nothing is re-packaged, and no leaderboard number
is claimed. The LB gate (#38) stays open.

## D043 — Full-size cold-start inference is measured, not estimated (2026-11-17)

Context: plan §8 (L208) requires benchmarking the full expected hidden-test size with
conservative memory/time headroom, simulating cold start and model load. Report §9 claimed
"seconds for ~4,000 test passages" — an **estimate** with no measurement behind it, so the
acceptance criterion "runtime, peak RAM and notebook compatibility are measured" was only
partly met for the inference path.

Decision — measure the **packaged** candidate through the notebook's exact code path in a
**fresh process**, on synthetic rows:
- `experiments/inference_benchmark.py` spawns a worker with `cwd` = a scratch working dir,
  `PYTHONPATH` removed, `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE=1`, and the package dir at
  `sys.path[0]`. The worker asserts `src.__file__` resolves **inside the package**, so the
  benchmark cannot accidentally time the repo's `src` instead of the shipped one.
- The worker replays notebook cells 1–4 verbatim: `load_final` + class-order check,
  `load_weights` + missing-bundle check, the `inference.json` `crossed: true` rejection,
  `predict_dataframe`, `build_submission`, then parquet read-back with the
  schema/id-uniqueness/null/label assertions. Every stage is timed separately and RSS is
  sampled at each boundary; Windows `peak_wset` gives the true peak working set.
- The test frame is **synthetic** (`src.synthetic`, k≈2 passages/prompt per
  `docs/strategy_review.md`), generated to an exact row count. **No competition row is
  scored and nothing is fit.** Only `train.csv` word counts are read, as a length reference.
- Sizes: 4,000 (expected hidden-test size) and 8,000 (2× stress for headroom).

Measured (registry `20261117T030454-inference_benchmark-cold_start_4000_8000`,
`artifacts/inference_benchmark/results.json`):
- **4,000 rows: 13.77 s cold start** (imports 1.20, bundle load 0.37, predict 11.14,
  write+validate 0.07), **peak working set 0.370 GB** (sampled RSS 0.220 GB).
- **8,000 rows: 28.24 s**, predict 25.21 s, peak 0.542 GB. Predict time scales linearly with
  rows; headroom is ≈2,350× / ≈1,150× the 9-hour CPU budget.
- A repeat 4,000-row run took 15.58 s, so the honest statement is **~14–16 s** with machine
  noise; the table of record is the saved run.
- Fixture passages are **longer** than official ones (mean 190.1 vs 114.7 words = 1.66×, max
  593 vs 274), which makes the timing **conservative** for official-length text.
- All schema/id/label/null assertions passed at both sizes; the packaged `src` was confirmed
  as the imported one; weights 0.5/0.5, char `fit_mode` `train+global_pool(capped=40001)`,
  style `train_only`.

Limits stated with the number: this is a **local Windows/CPU** measurement, **not** a
Kaggle-runtime benchmark — it bounds the work, not the platform, and the graded rerun on the
full hidden test remains owner-side (#38). `peak_wset` is a Windows counter; on Linux only
sampled RSS is available. The synthetic prediction distribution (heavily one class) is a
runtime artefact of fixture template styles and is **not** an accuracy, calibration or
class-prior result; no leaderboard number is claimed.

Consequences: report §9 replaces the estimate with the measured table and caveats; §11 lists
the command; README carries the numbers. Five tests (`tests/test_inference_benchmark.py`)
cover exact-size generation, the official test schema, determinism, an odd small size,
rejection of a non-positive size, and propagation of a worker failure (a broken package can
never be reported as a successful timing). **No model, weight, config or package change** —
this measures the already-packaged candidate, so nothing is re-packaged.

## D044 — Class-specific intercepts do not transfer; the blend keeps zero biases (2026-11-17)

Plan §7 L196 asks for class-specific intercepts/biases for balanced accuracy "with
out-of-sample tuning and strict separation". This was the last unmeasured ensembling lever.
`experiments/class_bias.py` measures it on the official-only fallback under both frozen seeds.

Protocol (no transductive step, no test row read): for each outer fold of the frozen grouped
splits, build 3 inner grouped folds on the **outer-train rows only**, refit both members on
inner-train, score inner-valid, and blend with the adopted `0.5 z(char) + 0.5 z(style)`. Tune
18 intercepts on that inner OOF by coordinate ascent (grid −1.5…1.5, step 0.25, 3 passes,
ties prefer the smaller |bias|), then pick a shrinkage factor from (0, 0.25, 0.5, 1.0) on the
same inner OOF. Apply the frozen bias once to the saved outer-fold OOF scores. 30 refits per
seed. A second arm adds the analytic `-log(class prior)` from outer-train label counts, which
requires no tuning at all.

| seed | arm | pooled BA | paired Δ vs no_bias (95% CI) |
|---|---|---|---|
| s42 | no_bias (adopted) | **0.375971** | — |
| s42 | prior_log_offset | 0.375600 | −0.000370 [−0.001399, +0.000658] |
| s42 | tuned | 0.366344 | **−0.009655 [−0.024830, +0.005520]** |
| s13 | no_bias (adopted) | **0.358190** | — |
| s13 | prior_log_offset | 0.358190 | +0.000000 [0, 0] (identical on every fold) |
| s13 | tuned | 0.363398 | **+0.005172 [−0.014428, +0.024773]** |

Ruling: **do not adopt.** The pre-registered rule is that the tuned arm must beat `no_bias` on
*every* seed with a paired CI excluding 0. s42 is negative on the point estimate and neither
CI excludes 0. The mechanism is visible in the optimism gap: inner-OOF gain +0.012152 (s42) /
+0.011686 (s13) versus outer gain −0.009655 / +0.005172, i.e. **+0.021807 / +0.006514** of the
inner improvement evaporates out of sample. Shrinkage did not rescue it — the grid chose 1.0
on nine of ten folds. The prior arm is a near no-op by construction because official train is
balanced (support 149/150, imbalance ratio 1.01), which also explains why prior correction was
never a promising lever here.

Limits stated with the number: the bias is tuned on the *inner* models' score scale and
transferred to the *outer* model's saved OOF, so the two scales are close but not identical;
the hidden test's class distribution is unknown, so a train-prior-derived bias might not
transfer even if it had won; and this measures the **official-only fallback** — the pool-MIX
candidate's char member is fit with external rows and was not re-measured, so nothing here
licenses a bias on the candidate. Prior development used all official prompts (D024).

Consequences: no config, weight, artifact or package change; the submission blend keeps zero
intercepts. Registry rows `…-official-class_bias_nested-s42` / `-s13`, artifact
`artifacts/class_bias/results.json`, report §7. Twelve tests (`tests/test_class_bias.py`) pin
the reference arm to the adopted row-z rule, the centred prior offset, zero bias when the inner
score cannot be improved, correction of a synthetic class offset, determinism, shape rejection,
inner-fold coverage, and the all-seeds-must-win verdict.

## D045 — The style-only SVM arm loses to logistic regression (2026-11-17)

Plan §4 L139 asks for style-only logistic regression, SVM and a small GBDT. `style_only`
(logreg, 0.290398) and `gbdt_style` (0.263015) were registered; the SVM arm was missing.
`configs/style_svm.yaml` adds it — the same 56 per-fold-standardised style features,
`LinearSVC(C=1.0, class_weight='balanced')` — in its own config so the ~20-minute
`char_plus_style` ablation is not re-run to add one variant.

Same frozen `stratified_group-s42-n5` folds, same rows, identity-verified OOF:

| member | pooled BA | fold scores | macro F1 | fit s | infer s | peak GB |
|---|---|---|---|---|---|---|
| style_only (logreg) | **0.290398** | 0.292784 0.307407 0.298148 0.264815 0.288889 | 0.2791 | 41.4 | 2.80 | 0.203 |
| style_svm (LinearSVC) | 0.284449 | 0.281545 0.296296 0.290741 0.259259 0.294444 | 0.257414 | 26.6 | 2.07 | 0.176 |

Paired SVM − logreg: **−0.005951, 95% CI [−0.014493, +0.002590]**. Within-provider errors 876
vs 873, cross-provider 1055 vs 1042 (1931 vs 1915 total). Ruling: **keep logreg.** The SVM is
cheaper and lighter but worse on the point estimate with a CI that includes 0 — there is no
evidence of a gain, and the adopted blend's style member is unchanged. LinearSVC margins are
not probabilities and were only row-z normalised, never treated as calibrated.

Consequences: `configs/style_svm.yaml` is a research ablation, not a submission member; no
weight, artifact or package change. Registry `20261117T034251-official-style_svm-s42`,
artifact `artifacts/cv_style_svm/`. This closes §4 L139 with all three classifier families
measured on the same folds.

*Follow-up (2026-11-17): a fourth family became measurable once `catboost 1.2.10` was
installed — see **D049** (CatBoost 0.283003, also negative). "All three families" above was
the state of the environment when this ADR was written.*

## D046 — Style-feature associations are measured, and are explicitly not causal (2026-11-17)

Plan §4 L142 asks for per-feature *and* per-class analyses "without claiming causal
attribution". Per-class recall, confusion pairs and length bands already existed (D041); the
per-feature half was missing. `experiments/style_features.py` adds it.

Identity first: the script refits `style_only` on each of the five frozen training folds and
**proves the refit reproduces the saved OOF** — max absolute drift **0.0** on all five folds —
before reporting any coefficient. The weights therefore provably belong to the models that
produced the recorded 0.290398, not to a lookalike refit. It aborts on drift > 1e-4 (the saved
OOF is float32) or on any feature-name mismatch across folds.

56 style features, `StandardScaler(with_mean=False)` fit inside each training fold, so
magnitudes are comparable across features. Aggregated per class and feature: mean coefficient,
mean |coefficient|, fold std, sign agreement. The overall ranking uses the class-averaged mean
|coefficient|; the signed class-average is omitted from the overall table because multinomial
coefficients are centred across the 18 classes and cancel.

Strongest overall associations: `rate_dash` **1.3816** (sign agreement 1.00), `rate_nonascii`
0.6182, `length_sentences` 0.5819, `rate_colon` 0.4973, `rate_squote` 0.4954, `rate_comma`
0.4852, `mean_word_len` 0.4566. `rate_dash` is positive for every `claude-*` and `glm-*` label
(e.g. claude-opus-5 +1.961, glm-5.3-flash +1.761) and strongly negative for the Gemini flash
siblings (gemini-3.6-flash −2.010, gemini-3.7-flash −1.804, gemini-3.8-flash −2.017) — the
typography fingerprint the char member also exploits. Weakest style-only recalls:
`claude-fable-5` **0.074** and `glm-5.3` **0.087**, both dash-heavy labels whose style profile
overlaps their siblings.

Ruling on interpretation: **associational only.** Correlated style columns share credit inside
one linear model, so a large weight is not evidence that the feature causes a model's style,
nor that removing it would change the score. No feature was removed, reweighted or selected
from this analysis, and no causal claim is made anywhere in the report.

Consequences: artifacts `artifacts/style_features/results.json` + `feature_table.md` (overall
table and an 18-row per-class table with recalls); report §8 references it; no config, weight
or package change. Seven tests (`tests/test_style_features.py`) pin the fold/class aggregation,
the class-averaged profile, row ordering, well-formed markdown and the unsupported-class marker.

## D047 — No near-duplicate official text crosses a fold (2026-11-17)

Plan §2 L94 asks whether identical/near-duplicate texts, shared prompt templates or augmented
variants cross folds. The audit already reported exact duplicates (0) and the grouped split
guarantees zero prompt overlap, but near-duplicate similarity **inside** the official corpus
had never been measured. `experiments/near_duplicates.py` measures it with the same word
TF-IDF settings (1–2 grams, sublinear tf, l2 norm, max_features 30000) and the same 0.3
threshold as the external-pool screen (`src.pool.decontaminate_global`), so the internal and
external audits are comparable.

2,699 rows, 150 prompts, **3,640,951 unordered pairs** examined (self-pairs excluded, each pair
counted once):

| threshold | pairs | cross-fold | same-prompt | cross-fold & different-prompt |
|---|---|---|---|---|
| cos > 0.3 | 2,190 | **0** | 2,190 | **0** |
| cos > 0.5 | 34 | **0** | 34 | **0** |
| cos > 0.7 | 0 | 0 | 0 | 0 |
| cos > 0.9 | 0 | 0 | 0 | 0 |

Exact duplicate texts **0**; prompts appearing in more than one fold **0**. Max off-diagonal
cosine over the whole corpus **0.6223** (median 0.2985, p90 0.4149, p99 0.5278).

Ruling: the similarity that exists is the shared question vocabulary of the 18 answers to one
prompt, and the prompt-grouped split keeps **every** such pair on one side of every fold. There
is no cross-fold near-duplicate leakage at any threshold tested, which is the empirical backing
for the grouped-split requirement rather than an assumption. Similarity remains a heuristic,
not proof that every augmented variant was found.

Compliance: this is a descriptive audit. The vectorizer is fit on the whole corpus, but no
predictor is fit, no score is reported, and no validation row influences any training decision —
so it is not a transductive step and does not touch the per-fold-fit rule. Only ids, fold
numbers and cosine values are stored; **no passage text** is written to the artifact.

Consequences: artifact `artifacts/near_duplicates/results.json`; report §4 and
`docs/data_audit.md` reference it; no model, config or package change. Six tests
(`tests/test_near_duplicates.py`) pin self-pair exclusion, unordered counting, the
same-prompt vs cross-fold breakdown, exact duplicates reaching cosine 1, threshold monotonicity
and sorted unique top pairs.

## D048 — Errors are spread across prompts at chance level; the pool gain is broad (2026-11-17)

Status: accepted (descriptive analysis; no model, config, weight or package change).

Context: plan §3 L118 asks for performance by class, model family, text length **and prompt
group**. The first three existed (D041 length bands, D046 per-feature/per-class, §8.1 provider
split); per-prompt performance had never been measured — `src/audit.py` only counts rows and
labels per prompt. The prompt is the unit the CV splits group on, so it is also the right unit
for asking whether a few hard questions drive the error, and whether the external-pool gain is
broad or concentrated.

Decision: measure it on saved identity-verified OOF with an explicit noise null, and report the
negative concentration result rather than only the favourable paired gain.

Protocol (`experiments/prompt_groups.py`): verify the local train fingerprint against the frozen
splits, verify every member's `ordered_dataset_sha256` and `validation_fold_ids`, then group the
2,699 rows into their 150 prompts (17–18 rows each). Five variants: the official-only blend, the
cap-40k pool-MIX candidate, and the three members. **No fit, no pool row read, no test row.**
Because a prompt holds only ~18 rows, every concentration statistic is compared against a
**binomial null** that keeps each prompt's real row count and draws its error count at the
observed overall error rate (2,000 draws, seed 20261117).

Measured:

| variant | prompt acc mean ± std | min | max | errors | worst-15 share | null mean [p05, p95] |
|---|---|---|---|---|---|---|
| official blend | 0.3761 ± 0.1209 | 0.1111 | 0.6667 | 1684 | **0.1336** | 0.1305 [0.1269, 0.1344] |
| pool-MIX cap40k | 0.4427 ± 0.1234 | 0.1667 | 0.7222 | 1504 | **0.1396** | 0.1357 [0.1315, 0.1404] |
| char official | 0.3194 ± 0.1107 | 0.0556 | 0.6667 | — | 0.1274 | 0.1266 [0.1235, 0.1300] |
| char pool 40k | 0.4105 ± 0.1064 | 0.1667 | 0.6667 | — | 0.1301 | 0.1331 [0.1289, 0.1375] |
| style only | 0.2905 ± 0.0981 | 0.0556 | 0.5556 | — | 0.1238 | 0.1244 [0.1213, 0.1277] |

- **No prompt-level concentration beyond chance.** Every variant's worst-decile error share sits
  *inside* its binomial null interval. **Zero** prompts have 0 accuracy and **zero** are perfect,
  for every variant. The failure is spread across questions, not carried by a hard subset.
- **No prompt collapses to one answer.** Distinct predicted labels per prompt average **10.21**
  (official, range 4–15) and **11.18** (pool, range 6–15) out of 18; modal share 0.226 / 0.205;
  `prompts_collapsed_to_one_label = 0`. Among wrong rows the modal wrong answer carries only
  **0.287 / 0.288** of a prompt's errors, and **no** prompt sends all its errors to one label.
  The closest thing to a prompt-level attractor is one official prompt whose modal share is
  0.778 (14/18 rows); the pool candidate's maximum is 0.444.
- **The pool gain is broad, on a third independent unit.** Paired per-prompt delta
  **+0.066645**, std 0.105199, 95% CI **[+0.049672, +0.083618]** over n=150 prompts:
  **96 improved, 28 worsened, 26 unchanged** (max gain +0.3333, max loss −0.2222). Rows and folds
  already showed the gain (D037/D039); prompts confirm it is not a few questions flipping.
- **Prompt-level length effect is real but modest.** Spearman ρ(prompt accuracy, prompt mean
  words) = **+0.169** (p=0.038) official and **+0.239** (p=0.0032) pool. So §8.2's row-level
  length gradient is only partly a prompt-level phenomenon.
- Per-prompt accuracy mean reproduces pooled BA (0.3761 vs 0.3760; 0.4427 vs 0.4427) because each
  prompt contains ~one row per class — a useful internal consistency check, not a new score.

Limits: the binomial null assumes rows within a prompt are independent; they share a question, so
the true null could be wider and the "inside the interval" conclusion is conservative in the
direction of *not* claiming concentration. Prompt ids, counts and scores are stored — **no passage
text**. This is descriptive: nothing is fit and no validation row influences a training decision.

Consequences: artifact `artifacts/prompt_groups/{results.json,prompt_groups.md}`; report §8.4,
README subsection, plan §3 L118 ticked. **No model, config, weight or package change.** The
negative concentration result argues against prompt-difficulty-targeted remedies (consistent with
the H2 nuisance-suppression negative, D010) and in favour of per-passage representation work.
28 tests (`tests/test_prompt_groups.py`) pin the grouping arithmetic, uneven group sizes, input
rejection, the concentration statistic, null determinism and edge rates, that a genuinely
concentrated failure exceeds the null's p95 while a perfectly even split falls below it,
prediction/wrong-answer spread, quantile ordering, capped sorted worst-prompt output and table
shape.

## D049 — CatBoost adds nothing to the style view; four classifier families now measured (2026-11-17)

Status: accepted (negative result; keep `logreg` for the style member).

Context: plan §4 L139 asks for a style-only comparison across classifier families. D045 added
the missing LinearSVC arm (0.284449) and D020 measured sklearn's HistGradientBoosting
(0.263015), but D020 could not test a real gradient-boosting library because **catboost,
lightgbm and xgboost were absent from the environment at that time**. That has changed:
`catboost 1.2.10` is now installed locally (lightgbm and seaborn are still absent). CatBoost
is the standard strong tabular baseline for a 56-feature stylometry view, so the missing
family was worth measuring rather than assumed.

Decision: add an **optional, lazily imported** `catboost` classifier to `src/model.py`, measure
it on the same registered folds as the other style arms, and adopt it only if it beats logreg
with a paired CI excluding 0 *or* adds blend diversity. It did neither.

Implementation: `CLASSIFIERS` gains `catboost`; a new `DENSE_CLASSIFIERS = ("gbdt", "catboost")`
tuple drives both the dense-input guard (style-only or SVD-reduced, never raw tf-idf) and the
`.toarray()` conversion in `fit`/`decision_function`. `build_classifier` imports catboost inside
the branch and raises `ModelError` if it is missing, so the default install is unaffected.
Parameters: `loss_function="MultiClass"`, iterations 300, depth 6, learning_rate 0.1,
l2_leaf_reg 3.0, `auto_class_weights="Balanced"` when `class_weight: balanced`,
`random_seed=spec.seed`, `verbose=False`, `allow_writing_files=False` (no `catboost_info`
directory is created). Config: `configs/style_catboost.yaml`, kept separate from
`configs/gbdt.yaml` so the SVD variants are not re-run.

**Score path.** CatBoost exposes no `decision_function`. `ModelBundle.decision_function` now
branches: estimators with `decision_function` keep their margins; otherwise
`log(clip(predict_proba, 1e-12, 1.0))` mapped into the 18-column output with the usual −1e9
sentinel for absent classes. Log-probability keeps the score on an unbounded additive scale,
which is what row-z blending (D042) expects. This does **not** violate the AGENTS.md rule
against treating LinearSVC margins as probabilities — CatBoost's `MultiClass` output *is* a
probability model, and no probability claim is made for margins anywhere.

Measured (official grouped CV, seed 42, five registered folds, fingerprint
`dc60875fd8b536cb8db67bdac8119132c6d0c0cf667fbca262d5ffc47f3e6e5a`; identity, fold assignments
and label vectors verified identical to the identity-verified `style_only` OOF):

| style-only classifier | pooled OOF BA | macro F1 | fit s | peak RSS GB |
|---|---|---|---|---|
| logreg (adopted member) | **0.290398** | 0.279150 | 29.6 | 0.20 |
| linearsvc (D045) | 0.284449 | 0.257414 | 26.6 | 0.18 |
| **catboost (this ADR)** | **0.283003** | 0.275370 | 115.0 | 0.21 |
| gbdt / HistGB (D020) | 0.263015 | 0.256768 | 74.2 | 0.20 |

Logreg timings are from the original `artifacts/cv` run, as in report §7; the later
identity re-run of the same model took 41.4 s / 0.203 GB (D045's table). Same score either way.

CatBoost folds: 0.294764 / 0.303704 / 0.285185 / 0.266667 / 0.264815 (mean 0.283027 ± 0.015292);
infer 1.308 s.

- **Paired against logreg on the same folds: −0.007382, 95% CI [−0.021211, +0.006448].**
  Against linearsvc: −0.001430, CI [−0.022784, +0.019923]. Both intervals include 0 and both
  point estimates are negative: no evidence of a gain, weak evidence of a loss.
- **It is not complementary either.** Within-provider errors 888 vs 873 for logreg, cross-provider
  1047 vs 1042 (1935 vs 1915 total) — the same failure mode, slightly worse. The two style
  members' row-z scores correlate **0.8678**, i.e. largely redundant.
- **Blend tests, both negative.** `0.5 z(char) + 0.5 z(style_catboost)` pooled **0.365240**
  (paired vs the adopted blend −0.010690, CI [−0.022630, +0.001251]); equal thirds
  char + style_logreg + style_catboost **0.368566** (−0.007356, CI [−0.022003, +0.007290]).
  Adding a third, highly correlated member dilutes rather than diversifies.
- Cost: ~3.9× logreg's fit time for no measured benefit. Macro F1 tells the same story as BA
  (catboost 0.2754 vs logreg 0.2792); the SVM arm's much lower macro F1 (0.2574) at a similar
  BA is a per-class precision effect, already recorded in D045's macro-F1 column.

Limits: one seed and the registered five folds; CatBoost was not hyperparameter-searched
(depth/iterations at their conventional defaults), so this rules out the *default* CatBoost arm,
not every possible tuning. Given that the style view is only half of the adopted blend and that
all four families land in 0.263–0.290, further search on this view is not where the remaining
gap to the leaderboard lives (D048: errors are spread, not concentrated).

Consequences: registry row `20261117T041323-official-style_catboost-s42`; artifact
`artifacts/cv_style_catboost`; report §7 and README carry the row; plan §4 L139 closed.
**No blend member, weight, submission config or package change.** catboost stays an **optional**
dependency — it is *not* added to `requirements.txt`, and `tests/test_model_catboost.py` uses
`pytest.importorskip` so the suite is green without it. D020's statement that catboost was absent
was true when written and is annotated, not rewritten. 12 tests pin the dense-classifier
registration, raw-tf-idf rejection, SVD acceptance, factory parameters (including the
no-balanced-weights case), 18-column class order, log-probability monotonicity and argmax
equality, the −1e9 absent-class sentinel, determinism, save/load round-trip and the absence of
any `catboost_info` side-effect directory.
## D050 — The upload-ready package must carry the pool's share-alike licence and attribution (2026-11-17)

**Context.** `cli package` hardcoded `licenses: [{"name": "CC0-1.0"}]` in
`dataset-metadata.json`, and its generated README said only "no competition data
included". The shipped `char_tfidf_logreg` bundle is fitted on official train
**plus** Open Pool v15, and **both** sources are CC BY-SA 4.0
(`docs/competition_spec.md` §4; `docs/external_data_register.md`). The register
already ruled that "attribution + CC BY-SA must travel with any redistributed
pool-derived weights", so the package contradicted the project's own compliance
posture: CC0 is a public-domain dedication and drops both the share-alike
obligation and the credit. Uploading it unchanged would have misdeclared the
licence of a derived artifact on a public Kaggle dataset.

**Decision.** `src.cli._package_licence(manifest_path)` derives the declaration
from the manifest actually being shipped:

- official (any non-synthetic) dataset -> **CC-BY-SA-4.0**, plus a README
  "Licence and attribution" block that links the licence, credits the competition
  for the official passages, and credits **Open Pool v15**
  (`crtproduction/name-that-llm-open-pool`) for exactly those members whose
  `fit_mode` contains `pool`, naming the `external_data_id`;
- `dataset: synthetic` -> **CC0-1.0** with no competition attribution, because
  fixture text is generated locally by `src.synthetic` and is not competition data.

`CC-BY-SA-4.0` is the Kaggle API's own documented licence name (the
`dataset_download_files` docstring in `kaggle/api/kaggle_api_extended.py`), not a
guessed identifier. The block also states that no raw competition rows, test text
or labels are redistributed — only the fitted weights.

**Package refresh (task #56).** Re-running
`python -m src.cli package --dataset official --kaggle-user tiennamlee` also fixed
a stale `src/`: the uploaded copy predated four source files. Verified delta
against the previous package, byte for byte (sha256):

| changed | unchanged |
|---|---|
| `src/cli.py`, `src/figures.py`, `src/metrics.py`, `src/model.py` (were stale); `README.md` and `dataset-metadata.json` (licence + attribution) | both `models/*.joblib` (**byte-identical** — no refit), `manifest.json`, `weights.json` (the D040 provenance fix is intact: `fit_modes`, `cv_score_of_record`, official-only CV fields absent) |

No file was added or removed. The package still ships no encoder, no
`inference.json` and no pool parquet.

**Re-validation.** Full suite **334 passed, 2 warnings** — three new tests in
`tests/test_package.py` (share-alike + pool credit; official-only credit without
the pool; synthetic stays CC0). The cold-start benchmark was re-run on the
refreshed package loading its **own shipped `src`**
(`artifacts/inference_benchmark_refresh/results.json`): 4,000 synthetic rows
**9.4 s** wall / **0.3702 GB**
peak working set (sampled RSS 0.2188 GB); 8,000 rows
16.8795 s / 0.542 GB.
`fits_performed: 0`, `competition_rows_scored: 0`, `validation_passed: true`,
returncode 0 for both. The D043 measurement (13.7687 s, and a 15.58 s repeat) is
preserved unchanged in `artifacts/inference_benchmark/`; this run is faster, but
the difference is **not attributed** to the source refresh — machine noise
dominates and no causal claim is made. Both are ~3 orders of magnitude inside the
9 h CPU budget. Timings bound the work, not Kaggle's platform.

**Consequences.** The owner uploads the dataset with the CC BY-SA 4.0 declaration
intact (`kaggle datasets create -p artifacts/kaggle_dataset`); it must not be
relabelled CC0. Attribution travels inside the package README, so a public
redistribution of the weights satisfies the pool's share-alike. **No score,
candidate, bundle or blend rule changed** — this is a compliance and freshness fix
only. The graded Kaggle rerun and the submission remain owner-only (#38).

## D051 — Candidate figure panels are rebuilt from the saved OOF, and unrecorded peak RAM stays unrecorded (2026-11-18)

**Context.** `docs/improvement_plan.md` and report §12 both carried the same open
gap: the candidate's confusion, recall and length-band figures existed (D041), but
its **ablation, runtime/RAM and grouped-CV variation** panels were still the
official-only baseline's. The cause is structural, not an oversight:
`src.figures.make_report_figures` reads `metrics/*.json` from **one** training run,
while the adopted cap-40k candidate's two members were fitted in **separate** runs
— `char_tfidf_logreg` with the globally screened pool under
`artifacts/pool_cv_cap40k/official`, `style_only` official-only under
`artifacts/cv_identity/official`. No single metrics directory describes the
candidate, so the existing entry point cannot draw it.

**Decision.** Split the three panels out of `make_report_figures` into
`make_ablation_figure`, `make_runtime_ram_figure` and `make_cv_variation_figure`,
each taking the same metrics-shaped mapping. `make_report_figures` now composes
them, so the baseline and the candidate render through **identical** plotting code.
New `experiments/candidate_figures.py` builds the five-arm mapping from the
identity-verified OOF (fingerprint, fold-id and label equality enforced against
`artifacts/splits/splits_official.json`) and writes
`artifacts/candidate_figures/results.json`.

Two honesty rules are encoded rather than left to the author:

1. **Every score is cross-checked, not asserted.** Each arm's pooled *and* per-fold
   balanced accuracy is recomputed from the OOF and compared to the value already
   recorded in the artifacts (`cv_identity` metrics, `pool_cv_cap40k/results.json`,
   `mix_baseline_verify_*/blend_official_pool.json`). A mismatch beyond `1e-6`
   aborts with "the saved OOF no longer reproduces the published candidate".
2. **Unrecorded peak RAM is never estimated.** The pool arm's originating run
   recorded `fit_seconds` but no `peak_ram_gb`. It is `null` in `results.json`,
   empty in the registry, and drawn as a **hollow marker labelled "RAM n/r"** with
   a title note — not a zero-area bubble, which would read as "no memory used".
   A blend has no fit of its own: `aggregate_costs` **sums** member fit seconds and
   takes the **larger** member peak (the fits are sequential, never co-resident),
   and one `null` member peak makes the blend peak `null` too.

**Measured (post-hoc; `fits_performed: 0`, `test_rows_read: 0`, no pool row read).**

| arm | pooled OOF BA | fold mean ± std | fit s (all folds) | peak fit RAM GB |
|---|---|---|---|---|
| `style_only` | 0.290398 | 0.2904 ± 0.0142 | 41.42 | 0.203 |
| `char_official` | 0.319306 | 0.3193 ± 0.0120 | 127.35 | 0.205 |
| `official_blend` (fallback) | 0.375971 | 0.3760 ± 0.0026 | 168.77 | 0.205 |
| `char_pool_cap40k` | 0.410445 | 0.4105 ± 0.0187 | 1599.62 | not recorded |
| `pool_mix_cap40k` (candidate) | 0.442675 | 0.4427 ± 0.0166 | 1641.03 | not recorded |

All ten pooled/per-fold comparisons matched within `1e-6` — a **fourth independent
reproduction** of 0.375971 / 0.442675 after D039 (`blend_deberta`), D041
(`length_sensitivity`) and D042 (`blend_rules`).

**New observation — the pool buys mean accuracy at the cost of fold stability.**
The candidate's fold spread is **6.31× wider** than the fallback's
(std 0.016640 vs 0.002637; range 0.412963–0.457407 vs 0.372222–0.379630). The
per-fold gain is nonetheless positive in **all five** folds
(+0.0708 / +0.0815 / +0.0574 / +0.0389 / +0.0852), so D037's all-folds-positive
acceptance gate still holds and the candidate is **not** demoted. But the wider
spread is a real property of the pool arm and is now visible in
`figures/cv_variation_pool_mix_cap40k.png` instead of hidden behind the baseline's
unusually tight folds. It reinforces protocol §7's warning that per-fold noise is
substantial at 2,699 rows, and it is a reason to keep the alternative-seed and
nested gates rather than trust s42 alone.

Fit cost is **9.72×** the fallback (1641.03 s vs 168.77 s). This is **offline
preparation only**: the graded notebook fits nothing, and its measured cold-start
inference is 9.40 s / 0.3702 GB (D050). The runtime panel plots *fit* cost and must
not be read as notebook runtime.

**Consequences.** Three new figures exist —
`figures/ablation_pool_mix_cap40k.png`, `figures/runtime_ram_pool_mix_cap40k.png`,
`figures/cv_variation_pool_mix_cap40k.png` — and report §12's "the candidate's
panels are the baseline's" caveat is retired for these three. **No score, weight,
config, bundle or packaged file changed**; nothing was refit and nothing was
re-packaged. Suite **342 passed, 2 warnings** (3 new panel tests in
`tests/test_figures.py`, 5 in `tests/test_candidate_figures.py`). Registry
`20261118T065212-pool_mix_cap40k-candidate_panels`. The remaining gates are
unchanged and owner-only: #38 upload/run/submit/report-LB, and the D038
DeBERTa-pool GPU run.

## D052 — Discussion sweep: the k question is unanswered, the Hungarian withdrawal is corroborated, and the encoder negative is backbone-specific (2026-11-18)

**Context.** Plan item L31 was the last unticked item that an agent could actually
do: the Discussion pages had never been swept, and `docs/competition_spec.md` §10
still listed hidden-test prompt multiplicity and GPU quota as unverified. Swept
2026-11-18 in the in-app browser; six threads read and recorded with per-claim
URLs in spec §11. **No organizer replied in any thread read.** Every number below
is a participant's self-report, not a host statement and not our measurement.

**Decision 1 — the unverified items stay unverified.** The question was asked
directly: *"Test set structure: passages per prompt?"*
([758516](https://www.kaggle.com/competitions/name-that-llm/discussion/758516))
has **0 comments**. No host answer on GPU quota either. So §10's k and quota
bullets remain open, the k-robust argmax path remains the safe default, and no
code may assume a test-side grouping. **Nothing was upgraded from "community
consensus" to "verified".**

**Decision 2 — the D017/D018 withdrawal now has external support.** A separate
solo pipeline (MIT code, self-reported LB 0.389) reports **"Hungarian assignment
within prompt groups gave +0.10 CV but +0.002 LB"**
([757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855)).
That is the same collapse our k-subsampled stress test measured. The crossed
lever stays **withdrawn**; this is corroboration, not a new experiment, so **no
registry row was created**. One of their claims contradicts our stated rationale
(random CV *pessimistic* 0.104 vs grouped 0.315, where protocol §1 argues random
folds would be *optimistic*); it is recorded as **unreconciled** and we adopt
neither number. Grouped folds remain mandatory.

**Decision 3 — the encoder negative is backbone-specific, so reopen it as a named hypothesis.**
D022/D023/D027 concluded "the encoder adds no diversity gain" from
**DeBERTa-v3** runs (official-only 0.2482; every fold-aligned blend at or below
sparse). The sweep reports **ModernBERT-large ≈ 0.55 OOF and 0.55 LB from two
independent participants**, above their own DeBERTa-v3-large 0.50
([759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103)).
Our conclusion was therefore about **a backbone**, not about encoders. Recorded as
**hypothesis H8: a ModernBERT-large sequence head, trained on the same globally
screened pool at cap 40k, beats our DeBERTa-v3 result and adds diversity to the
MIX candidate.** It is *not* adopted and *not* scored. It must clear the same bar
as D038: paired CI excluding 0 against `mix_baseline` under all three grouped
protocols (s42, s13, nested), GPU-side, owner-run. The existing
`kaggle/deberta/deberta_train.py` already has the non-transductive
`decontaminate_global` path and embedded folds, so the change is the backbone and
its tokenizer/max-length, not the protocol.

**Decision 4 — hard typographic family vetoes are a new cheap hypothesis.** Two
threads independently describe near-deterministic punctuation rules
([759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103):
family accuracy 87–99% vs within-family version 67–76%;
[757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855):
per-rule chance level < 1e-4). **Verified in our own code:** the adopted char
member already keeps raw characters — `src/features_tfidf.py:47` sets
`lowercase=False` (`:67` for `char_wb`) and neither the feature builder nor
`src/data.py` normalises unicode or strips punctuation. So "don't normalise
punctuation" is **already satisfied; no defect found.** The untried part is using
such rules as **hard vetoes** instead of soft features. Recorded as
**hypothesis H9**, CPU-only, with two conditions: rules must be mined **per fold
on that fold's training rows only**, and a rule may only veto a family if its
training-side support and chance level are recorded — a rule fitted on all train
and applied to validation is the same leak class D026 banned.

**Decision 5 — an external CV→LB calibration, explicitly not a prediction.** The
same thread gives a prompt-grouped ladder: TF-IDF+logreg 0.31 OOF / 0.32 LB;
+external data 0.46 / 0.40; +XGBoost style stack 0.49 / 0.43; encoder blend
0.625 / 0.616. Our only measured pair is CV 0.3760 → LB 0.356 (−0.020). Their
external-data row implies −0.06 at a similar level. **A −0.02 to −0.06 offset is
plausible, so the candidate's 0.4427 CV is consistent with an LB near 0.38–0.42,
not 0.44.** This is an expectation to hold loosely, it does **not** change the
candidate or the revert rule (LB ≤ 0.356 → rebuild the official-only fallback),
and only #38 can settle it. Non-LLM methods are reported to flatten at 0.60–0.62,
and two participants (one 8th-placed) report external data eventually *degrading*
results — consistent with our cap sweep finding a binding budget rather than
unlimited headroom.

**Consequences.** Spec §11 is the evidence record; plan L31 is ticked with the
k/quota items still open. **No score, model, weight, config, bundle or package
changed, and no registry row was added** — a literature sweep is not an
experiment, and community numbers must never enter our registry as results.
Public code worth reading later is listed in spec §11.5 (an Apache-2.0 notebook
self-reported at 0.687 built on a public v11 recipe, a GLM-likelihood notebook
self-reported at 0.689, and the MIT solo pipeline); none has been read or adopted.
H8 and H9 are queued in `docs/improvement_plan.md`. The remaining gates are
unchanged and owner-only: #38 upload/run/submit/report-LB, and the GPU runs.

## D053 — H9 hard typographic family vetoes: measured, not adopted

**Date.** 2026-11-18. **Status.** accepted (negative result; candidate unchanged).
**Supersedes nothing.** Implements the H9 hypothesis queued by D052.

**Context.** The Discussion sweep (D052, spec §11.4) reported participants using
punctuation *invariants* — "this provider never emits a curly quote" — as hard
filters. Our style block already carries the matching soft rates
(`src/features_style.py:92-95`), so the open question was whether a hard mask adds
anything on top of the adopted row-z blend. This was the last lever an agent could
test locally on CPU; H8 needs a GPU.

**Decision.** Build the mechanism, measure it under the registered protocol, and
**do not adopt it**: the paired confidence interval includes zero.

`src/typography.py` mines rules of the form *family f never emits marker m* and
masks f's classes to `SENTINEL`. Eight markers are tested on the raw string with
no case folding and no unicode normalisation, because the difference between
`'` and `\u2019` **is** the signal. Three safeguards:

1. rules are mined **per fold, on that fold's official training rows only**;
2. every rule carries its support, the family's row and prompt counts, and the
   chance probability of seeing zero hits by luck;
3. a row whose every class would be vetoed is left untouched and counted.

`experiments/typo_vetoes.py` applies the mask to the identity-verified OOF of both
blends. It fits nothing, reads no test row and no pool row, and aborts unless the
rebuilt candidate reproduces 0.442675.

**Measured (s42, 2,699 rows, 5 grouped folds).**

| arm | base | vetoed | Δ | paired 95% CI | all folds ≥ 0 |
| --- | --- | --- | --- | --- | --- |
| `pool_mix_cap40k` | 0.442675 | 0.443415 | +0.000741 | [−0.001316, +0.002797] | yes |
| `official_blend` | 0.375971 | 0.376341 | +0.000370 | [−0.000658, +0.001399] | yes |

Only fold 3 moves (0.412963 → 0.416667); the other four folds are bit-identical.
Nine predictions flip out of 2,699 rows, **all from a wrong prediction and none
from a correct one**, in every fold of both arms. The mechanism is harmless and
almost useless: a veto only changes the answer when the blend's argmax was already
inside the vetoed family, which the soft style rates have usually pushed down
already. The sensitivity grid (support 10/20/40 × chance 1e-2/1e-3/1e-6) returns
the **identical** score in all five settings, so this is not a threshold artefact.

**Two findings worth more than the score.**

*The chance level must count prompts, not rows.* The first implementation used
`(1 - p) ** n_rows`. The 2,699 rows are 150 prompt groups — roughly one row per
model per prompt — so rows inside a prompt are near-replicates, not independent
trials, and row-level confidence was overstated by orders of magnitude. Counting
distinct `prompt_id` within the family gives every family 120 trials instead of
240–600. That correction rejects the two `curly_dquote → ZAI` rules
(1.04e-04 → 1.48e-02) and 10 of 22 rules overall, **with the score unchanged**.
Group-aware chance is now the default in the experiment and is covered by a test.

*A tiny chance probability is not evidence of an invariant.* `em_dash → Google`
is mined in only **1 of 5 folds**, survives even the corrected chance at
2.74e-26, and is still wrong: validation row **1554** (prompt `p149`, true label
`gemini-3.7-flash`) contains an em dash. It is the only counterexample in the run;
the four `curly_apostrophe` rules, mined in 5/5 folds, have none. The reason is
structural — the pooled prevalence `p` overstates Google's own em-dash rate, and
the family-specific rate that would fix the null is exactly what a zero-hit
family cannot supply. Absence rules therefore cannot be validated by an i.i.d.
chance calculation when family base rates differ.

**The tempting fix is banned.** Keeping only rules that recur across folds would
delete the bad rule and keep the four good ones. It is not usable: fold k's
validation rows sit inside fold j's training rows for j ≠ k, so selecting rules by
cross-fold recurrence feeds validation information back into the rule set — the
transductive class D026 bans. Recurrence is reported as **descriptive only** and
no score uses it. A legitimate version would need inner folds inside each fold's
training rows; given the effect size that is not worth the machinery.

**Oracle control.** Handing all 754 veto-fired rows to their true label yields
BA 0.597166, which looks like a large headroom for H9. Handing the **same rows** to
truth with **no veto at all** yields 0.597537. The veto's own share of that ceiling
is **−0.000370**: the headroom is simply where the blend is wrong, not evidence for
this lever. Reporting the first number without the control would have overstated
H9 by two orders of magnitude, so the control is part of the script.

**Consequences.** The candidate stays at **0.442675** (s42) / 0.430435 (s13) /
0.442647 (nested); the packaged dataset, weights, configs and notebook are
untouched. `src/typography.py`, `experiments/typo_vetoes.py` and nine tests are
kept as the measured negative result and as reusable, leakage-safe machinery.
Registry row `20261118T071941-pool_mix_cap40k-typo_vetoes`; evidence in
`artifacts/typo_vetoes/results.json` and `veto_report.md`. Suite **351 passed**.
H9 is closed. The remaining levers are owner-side: **#38** upload/run/submit/
report-LB, the D038 DeBERTa-pool GPU run, and **H8** (ModernBERT-large backbone).

## D054 — H8 enablement: the encoder backbone is now switchable from a script kernel

**Date.** 2026-11-18. **Status.** accepted (enabling infrastructure; **no metric**).
**Follows** D038 (pool route re-wired) and D052 (H8 queued).

**Context.** H8 — swap DeBERTa-v3-large for ModernBERT-large — was queued first in
Phase 2 on the strength of two participants self-reporting ≈0.55 against their own
DeBERTa ≈0.50 (spec §11.3). Checking whether the owner could actually run it found
**two blockers**, both invisible from the local side:

1. `kaggle/deberta/kernel-metadata.json` declares `kernel_type: script`. Kaggle runs
   a script kernel as `python deberta_train.py` with **no arguments**, so the
   existing `--model-name` flag was unreachable on a pushed run. This is the same
   constraint that forced the folds to be embedded in D023.
2. `requirements_gpu.txt` floored `transformers>=4.45`. ModernBERT was added in
   **4.48**, so an environment satisfying the old floor could fail to recognise the
   architecture at all.

**Decision.** Make the backbone a module-level switch, in the same style as
`FOLD_SEED`, and keep the default byte-comparable with the D038 run.

- `BACKBONES = {"deberta-v3-large": "microsoft/deberta-v3-large",
  "modernbert-large": "answerdotai/ModernBERT-large"}` and `BACKBONE =
  "deberta-v3-large"`. Editing that one line is what changes a pushed run.
- `build_arg_parser()` and a pure `resolve_model_name(backbone, model_name=None)`
  were extracted from `main()` so the no-argument path is **testable**, not just
  asserted in a comment. An explicit `--model-name` still wins; an unknown backbone
  raises rather than silently falling back.
- The chosen backbone is recorded in `results.json`'s `config` and in
  `inference_manifest.json`, so a run can never be misattributed to the wrong
  encoder. `schema_version` stays `deberta-inference/v1` and `backbone` is additive,
  so existing manifests still validate.
- `transformers>=4.48`, with a comment noting DeBERTa alone works from 4.45.

**Deliberately not preset per backbone:** `max_len`, `batch_size` and `lr` stay at
the measured D038 budget (384 / 4 / 1e-5) for **both** presets. The first H8 probe
must change one variable. ModernBERT's long-context capability is a real potential
advantage, but raising `max_len` would change the memory profile against a T4
ceiling we cannot measure locally (torch here is CPU-only) and would confound the
backbone comparison. It is a separate later probe. A test pins the shared budget so
this cannot drift silently.

`kaggle/deberta/predict.py` needs **no change**: it loads
`AutoModelForSequenceClassification` / `AutoTokenizer` from the attached folder, so
it is already backbone-agnostic.

**Consequences.** Seven tests in `tests/test_deberta_backbone.py` (six for the
switch, one asserting every preset checkpoint already has a register row with a
recorded licence); suite **358 passed, 2 warnings**. **No registry row** — this is enabling infrastructure
with no measured score, the same treatment as D038 and D052, and community numbers
must never enter our registry as results. Nothing was adopted; the candidate stays
**0.442675** (s42) / 0.430435 (s13) / 0.442647 (nested) and the packaged dataset,
weights, configs and submission notebook are untouched.

**Owner steps for H8 (GPU quota, owner account only).**

1. Set `BACKBONE = "modernbert-large"` in `kaggle/deberta/deberta_train.py`.
2. `kaggle kernels push -p kaggle/deberta`, then poll
   `kaggle kernels status tiennamlee/name-that-llm-deberta-official-vs-pool-cv`.
3. Score it against the adopted candidate, not the retired 0.376 blend:
   `python experiments/blend_deberta.py --deberta-dir <out>/deberta_cv
   --variant official_pool --char-oof
   artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz --style-oof
   artifacts/cv_identity/official/oof/style_only.npz`
4. Adopt only if `mix+deberta_*` beats `mix_baseline` with a paired CI excluding 0
   under **all three** grouped protocols (s42, s13, nested). Otherwise keep the MIX.
5. Restore `BACKBONE = "deberta-v3-large"` before any D038 DeBERTa-pool run so the
   two arms stay distinguishable.

**Verified checkpoint facts (model cards read 2026-11-18).** The H8 premise came from
community self-reports, so the underlying claims were checked against primary sources
before being written into the kernel requirements:

| claim | verified wording / source |
|---|---|
| ModernBERT-large: **28 layers, 395M params** | card: "ModernBERT-large - 28 layers, 395 million parameters" |
| ModernBERT licence **Apache-2.0** | card: "We release the ModernBERT model architectures, model weights, training codebase under the Apache 2.0 license." |
| needs **transformers >= 4.48** | card: "You can use these models directly with the `transformers` library starting from v4.48.0" |
| 8,192 native context, **no `token_type_ids`** | card (long-context section; usage notes) |
| DeBERTa-v3-large: **MIT**, 24 layers, hidden 1024, **304M backbone + 131M embedding** (128K vocab) | card |
| DeBERTa-v3-base: **MIT**, 12 layers, hidden 768, **86M backbone + 98M embedding** | card |

**Counter-evidence that tempers H8.** The same ModernBERT card's Table 1 puts
ModernBERT-large at **90.4** on GLUE against DeBERTaV3-large's **91.4**, and its prose
says "ModernBERT-large is second only to Deberta-v3-large". The published evidence
therefore does **not** support "ModernBERT is the stronger encoder". H8 is worth one
probe as a **diversity** bet only: a different architecture, tokenizer and pretraining
mix may err differently from the sparse members even at equal or lower standalone
accuracy. D027 already showed a weaker encoder can add nothing, so the acceptance bar
stays the paired-CI blend gate, never the standalone score.

**Do not enable Flash Attention 2 on a T4.** The card recommends FA2, but upstream
flash-attn lists "Ampere, Ada, or Hopper GPUs (e.g., A100, RTX 3090, RTX 4090, H100)"
and redirects Turing (T4, RTX 2080) to a separate community fork. `build_model` passes
no `attn_implementation` and `requirements_gpu.txt` does not install `flash-attn`; keep
it that way for the probe. Separately, 395M total params is *fewer* than
DeBERTa-v3-large's 304M + 131M, which is mildly encouraging for the 16 GB T4 that peaked
at **10.175 GB** — but a parameter count is not a memory measurement, and ModernBERT has
no measured peak here.

**Register gap closed (provenance repair, no metric).** Verifying these cards exposed a
real omission: `docs/external_data_register.md` listed only `minilm-l6-v2` and
`open-pool-v15`, so **both DeBERTa checkpoints had been fine-tuned (D023/D027) without a
register row** while plan §6 claimed complete coverage. Three rows are now added —
`deberta-v3-base`, `deberta-v3-large`, `modernbert-large` (**planned, never downloaded,
no metric**) — with card-verified licences and parameter counts, the measured GPU peaks
(6.463 GB base / 10.175 GB large), and an explicit note that **no encoder checkpoint is
retained locally**, so their sizes are card parameters rather than hashed file bytes.
The plan's evidence annotation is corrected from "both rows" to five, and spec §9's
pretrained-encoder ruling now covers fine-tuned encoders and the training-time
(internet-on) checkpoint fetch versus offline inference. No score, weight, config or
package changed.

The remaining gates are unchanged and owner-only: **#38** upload/run/submit/
report-LB, the **D038** DeBERTa-pool run, and this **H8** probe.

## D055 — Larger ensembles: no cheap third member helps the pool-MIX blend (2026-11-18)

**Date.** 2026-11-18. **Status.** accepted (negative result; candidate unchanged).
**Supersedes nothing.** Extends D049, whose equal-thirds measurement used the
**official-only** char member, not the pool-trained one.

**Context.** Priority 5 of the standing mandate is "larger ensembles", and it had
never been tested against the adopted candidate. D049 measured
char+style_logreg+style_catboost at equal thirds against the *official-only* blend
(0.368566, paired −0.007356). The candidate is a different object: the pool lifted
char from 0.3193 to **0.4104** while style stayed official-only at 0.2904, giving
**0.442675**. Every cheap official-only member (solo 0.13–0.31) is now far weaker
than the blend it would join, so the open question was whether a third view
diversifies or merely dilutes.

**Decision.** Measure it with **pre-specified** weights, record the negative, and
adopt nothing.

`experiments/blend_members.py` screens six identity-verified arms under two fixed
recipes: **equal thirds** and **0.4/0.4/0.2**, the latter preserving the adopted
char:style ratio. Each arm reports solo pooled BA, the row-z correlation with the
base blend (redundancy), and a paired five-fold CI against the base. It fits
nothing, reads no test row and no pool row.

**Provenance repair first.** The 2026-11-14 `artifacts/cv/official/oof/*.npz` files
predate D024 and carry no fingerprint or fold ids, so they cannot be paired against
the base and are refused, not guessed. Two runs on the frozen s42 folds produced
verifiable OOF: the new `configs/blend_members.yaml`
(`word_tfidf_linearsvc` **0.269652**, `char_wb_tfidf_linearsvc` **0.296336**) and the
existing `configs/frozen_encoder.yaml` run offline (`HF_HUB_OFFLINE=1`;
`minilm_embedding_style_logreg` **0.309294** and `minilm_embedding_logreg`
**0.127793** reproduce D013 exactly). `style_svm`/`style_catboost` already carried
provenance (D045/D049).

**Measured (s42, 2,699 rows, 5 grouped folds; base 0.442675).**

| third member | solo | row-z corr | equal thirds | Δ paired 95% CI | 0.4/0.4/0.2 | Δ paired 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| `word_tfidf_linearsvc` | 0.269652 | 0.546 | 0.407823 | −0.034853 [−0.054544, −0.015162] | 0.435240 | −0.007471 [−0.022787, +0.007845] |
| `char_wb_tfidf_linearsvc` | 0.296336 | 0.600 | 0.413786 | −0.028927 [−0.048060, −0.009794] | 0.432680 | −0.010038 [−0.025935, +0.005859] |
| `style_svm` | 0.284449 | 0.808 | 0.406353 | −0.036322 [−0.045172, −0.027472] | 0.430067 | −0.012618 [−0.022303, −0.002933] |
| `style_catboost` | 0.283003 | 0.819 | 0.401921 | −0.040754 [−0.053151, −0.028356] | 0.428230 | −0.014444 [−0.023260, −0.005628] |
| `minilm_embedding_style_logreg` | 0.309294 | 0.900 | 0.418963 | −0.023729 [−0.028484, −0.018974] | 0.430075 | −0.012618 [−0.022075, −0.003161] |
| `minilm_embedding_logreg` | 0.127793 | 0.343 | 0.361904 | −0.080792 [−0.100763, −0.060821] | 0.411539 | −0.031149 [−0.054359, −0.007940] |

**All six arms lose, in both recipes.** Every point estimate is negative; all six
equal-thirds CIs and four of six keep-ratio CIs exclude zero, and no arm is
non-negative on all five folds. The pattern is a strength/diversity trade-off with
no winner: the *most* diverse arm (embedding-only, corr 0.343) is the weakest
(0.1278) and loses most; the *least* diverse (embedding+style, corr 0.900) is
redundant with the style member it duplicates. Even the best keep-ratio arm
(`word_tfidf_linearsvc`, 0.435240) costs −0.0075. Diversity without strength does
not pay once one member has been lifted by external data.

**Not screened — an honest gap.** `gbdt_style` (0.263015), `gbdt_multiview_svd128`
(0.242608), `char_svd256_linearsvc` (0.303378), `style_svd64_nuisance8` (0.273729),
`word_only`, `word_plus_char` and `char_tfidf_linearsvc` keep only pre-D024 OOF and
were not re-run. Their solo scores sit inside the same 0.24–0.31 band as the six
measured arms, so the dilution mechanism should apply, but they are **unmeasured as
third members** and no claim is made about them.

**Why this is descriptive, not a gate.** Picking the best arm off these curves and
shipping it would be selection on the reporting OOF (D026). Because no arm has a
positive point estimate there is nothing to promote to an outer-fold gate, so the
negative is recorded and the recipe stays the two-member 0.5/0.5 blend. A third
member is only worth a gate if it is *strong* — a pool-trained or GPU encoder
member (D038/H8), not another official-only view.

**Consequences.** `experiments/blend_members.py`, `configs/blend_members.yaml`,
six tests in `tests/test_blend_members.py` (per-fold scoring, the invariant that
adding the blend itself changes no argmax, dilution by a weak distractor, the
preserved char:style ratio, and fail-closed identity/provenance screening). Five
registry rows: the four re-run members plus `blend_third_member_screen`
(**126 rows**). No blend weight, member, config, submission or package change.

**Update (same day, see D056).** The screen was extended before it was closed: a
third pre-specified recipe (**swap**: 0.5 char + 0.5 candidate, style dropped) and
a seventh arm (the **pool-trained** word member) were added, so the script's
default table is now seven arms × three recipes and the test file has **eight**
tests. The six-arm table above is the measurement as first recorded and is
unchanged; the swap column and the seventh arm are in D056. Extending the screen
also exposed a real defect: two runs export the same member stem
(`word_tfidf_linearsvc.npz`, official-only and pool-trained), and
`members[name] = scores` silently overwrote one, so the first seven-arm run
printed six rows and mislabelled its best arm. `load_members` now qualifies
colliding stems by run directory (`pool_cv_word40k/word_tfidf_linearsvc`), a
regression test covers it, and the registry row written by the buggy run
(`20261118T083510-official-blend_members-s42`) is marked **superseded** by the
corrected re-run (`20261118T084438-official-blend_members-s42`) — its numbers were
the pool arm's all along and re-measure identically.

## D056 — H10: a pool-trained third member is the first positive ensemble signal, and it is still not adopted (2026-11-18)

**Date.** 2026-11-18. **Status.** accepted (descriptive positive on two seeds;
honest gate running; **candidate unchanged**). Follows D055, whose stated
mechanism was "diversity without strength does not pay".

**Context.** D055 closed the cheap-third-member question negative for six
official-only arms and named the follow-up itself: a third member is only worth a
gate if it is *strong*. H10 tests exactly that — give the word view the **same**
non-transductive external-pool advantage the char member already has (D028 global
screen, cap 40k/fold), then re-measure the pre-specified recipes. Nothing about
the pool protocol changes: `decontaminate_global` screens once against the full
official train, the same keep-mask serves every fold, and the pool is never
scored.

**Decision.** Measure the pool-trained word member on both frozen seeds, record
the descriptive result, and gate it on grouped outer folds before any adoption.

**1. The pool lifts the word view, and the lift replicates.**

| seed | official-only | +global pool cap40k | paired fold Δ [95% CI] | fit (5 folds) |
| --- | --- | --- | --- | --- |
| s42 | 0.269652 | **0.354877** | +0.0852 [+0.0567, +0.1137] | 246.4 s |
| s13 | 0.255938 | **0.346339** | +0.0904 [+0.0566, +0.1242] | 282.8 s |

Both runs screen 226,191 → 226,049 (0 exact + 142 near at cosine > 0.3) and cap to
40,001 rows/fold. The word view is still far below the pool-trained char member
(0.4104 s42 / 0.4115 s13), but it is no longer in the 0.13–0.31 dilution band that
made every D055 arm lose.

**2. Seven arms × three pre-specified recipes (s42, base 0.442675).** The screen
gained a third recipe, **swap** (0.5 char + 0.5 candidate, style dropped), which
asks a different question: is the candidate a better *partner* than style?

| third member | solo | corr | thirds | keep 0.4/0.4/0.2 | swap |
| --- | --- | --- | --- | --- | --- |
| `cv_blend_members/word_tfidf_linearsvc` | 0.2697 | 0.546 | 0.407823 | 0.435240 | 0.371514 |
| `char_wb_tfidf_linearsvc` | 0.2963 | 0.599 | 0.413786 | 0.432680 | 0.372297 |
| `style_svm` | 0.2844 | 0.808 | 0.406353 | 0.430067 | 0.427472 |
| `style_catboost` | 0.2830 | 0.819 | 0.401921 | 0.428230 | 0.429334 |
| `minilm_embedding_style_logreg` | 0.3093 | 0.900 | 0.418963 | 0.430075 | **0.447129** |
| `minilm_embedding_logreg` | 0.1278 | 0.343 | 0.361904 | 0.411539 | 0.306729 |
| `pool_cv_word40k/word_tfidf_linearsvc` | **0.3549** | 0.670 | 0.442665 | **0.450082** | 0.406366 |

**3. The positive addition replicates on the alternative seed (s13, base 0.430435).**

| arm | solo | thirds | keep 0.4/0.4/0.2 | swap |
| --- | --- | --- | --- | --- |
| `cv_word_s13/word_tfidf_linearsvc` | 0.2559 | 0.388556 (−0.0419) | 0.421148 (−0.0093) | 0.363005 |
| `char_wb_tfidf_linearsvc` | 0.2930 | 0.413398 (−0.0170) | 0.425983 (−0.0044) | 0.381173 |
| `pool_cv_word40k_s13/word_tfidf_linearsvc` | **0.3463** | 0.432647 (**+0.0022** [−0.0169, +0.0213]) | **0.440057** (**+0.0096** [−0.0099, +0.0292]) | 0.404504 (−0.0259) |

So the keep-ratio recipe gains **+0.007407 [−0.012572, +0.027387]** at s42 and
**+0.009642 [−0.009867, +0.029152]** at s13: same sign, similar size, on
independent grouped folds. It is up on **4 of 5 folds on both seeds**, and the
losing fold is a *different* one each time (s42 fold 4 −0.0185; s13 fold 0
−0.0129), so the gain is not one lucky group. Equal thirds is indistinguishable
from the base at s42 (−0.000026) and slightly positive at s13.

**4. Why nothing is adopted.** Both paired CIs **include zero**: the five-fold
paired std is ~0.016, twice the ~0.008 effect. Reading the better of three
recipes × seven arms off the reporting OOF is precisely the D026 selection class,
and a replicated *descriptive* sign is not a gate. The swap result is negative for
the pool word member on both seeds (−0.0363 / −0.0259): style remains the better
partner, and the third member earns its place only as a small addition.

**5. The honest gate (running).** `experiments/nested_recipes.py` +
`configs/nested_recipes.yaml`: grouped **outer** folds (3, seed 121), recipe
chosen by **inner** grouped CV (3, seed 71) inside outer-train only, ties keep the
base, and the untouched outer holdout scored with both the selected recipe and
the pre-specified base. The recipe set is *identical* to the screen's
{base 0.5/0.5, thirds, 0.4/0.4/0.2} — a unit test asserts that equality so the
gate cannot drift onto a different object. The pool is screened against
outer-train only and capped at 40k; only `char_tfidf_logreg` and
`word_tfidf_linearsvc` ever see pool rows (a test asserts the exact fit sizes).
Because the outer folds match the running blend-weight gate (#62, same outer
seed 121 / 3 folds, same cap and pool seed), the base recipe must reproduce that
gate's fixed-0.5 outer scores — a free cross-check of two independent
implementations. Adoption requires the paired outer CI to exclude 0.

**Not replicated — an honest gap.** The s42 **swap** of style for
`minilm_embedding_style_logreg` is the second positive point estimate
(0.447129, +0.004444 [−0.007570, +0.016459]), but no s13 embedding OOF exists, so
it was **not** replicated and is not gated. It is also the most redundant arm
(corr 0.900) and its CI includes 0; H10 is the stronger lead.

**Consequences.** `experiments/nested_recipes.py`, `configs/nested_recipes.yaml`,
eight tests in `tests/test_nested_recipes.py`; `experiments/blend_members.py`
gains the swap recipe, the pool-word default arm, run-dir-qualified member names
and a `--seed` label. Registry **135 rows** (word s13 official-only 0.255938,
char_wb s13 0.293003, word+pool s13 0.346339, the corrected s42 screen, the s13
screen, and the superseded marker on the buggy s42 row). Suite **381 passed,
2 warnings**. **No blend weight, member, config, submission or package change**:
the packaged candidate is still the two-member 0.5/0.5 pool-MIX blend
(0.442675 s42 / 0.430435 s13), and no leaderboard number exists for it (#38).

## D057 — Run the official-data score improvement screens (2026-11-18)

Use the fixed recipes in [the score improvement plan](../plans/name-that-llm-score-improvement.md)
and compare against the identity-verified sparse control on the same rows.
The Open Pool route remains outside the scope of these experiments. All learned
features and heads fit on the training side only. The first sparse screen uses
five prompt-grouped folds and requires +0.010 pooled balanced accuracy and
gains in four folds. The paired uncertainty intervals resample whole prompts.

The provider specialist used the fixed 0.2 weight and shared character/style
features. Its first run was stopped before scoring, while the full-dimensional
flat head was still fitting. The completed run fits the same L2 penalized
logistic objective in the centered training row span and maps coefficients
back to the original features. A numerical test checks prediction equivalence
with the original sparse formulation. This change was made for runtime before
reading any provider validation scores.

- Provider specialist **0.373388**, control **0.375971**, difference **−0.002583**
  (paired prompt 95% interval **[−0.013704, +0.007839]**, one positive fold).
  Within-provider errors rose from 897 to 918, while cross-provider errors
  fell from 787 to 773. Matched flat diagnostic **0.361148**.
- Masked blend **0.364111**, difference **−0.011859** (paired prompt 95%
  interval **[−0.022243, −0.000780]**, one positive fold). The same-weight
  raw-character diagnostic scored **0.371526**. On passages under 50 words,
  the masked blend scored **0.236176**, below the control's **0.249083**.

Neither CPU arm advances to seed 13 or fixed nested confirmation. The fallback
uses a public, pinned **frozen** ModernBERT-large checkpoint. Run the same heads
and 25% blend on frozen folds 0 and 1 before the five-fold screen. Checkpoint
provenance is in the [source register](external_data_register.md). These stages
are exploratory, since both pilot folds are reused in the full screen.

The pilot gains were **+0.039017** and **+0.027778**. The five-fold encoder
blend scored **0.405620**, a **+0.029650** gain over the sparse control
(paired prompt interval **[+0.016667, +0.042608]**, all five folds positive).
The encoder head alone scored **0.310425**. Both error categories improved
in the blend (within-provider 897 → 872, across-provider 787 → 732).
The local RTX 4060 encoded the 2,699 rows in 45.57 s after model load,
using 0.792 GiB peak torch-allocated VRAM. The logistic readout hit the fixed
500-iteration cap on every seed-42 fold. The results use that exact recipe,
so any optimizer change needs a new comparison. No Kaggle score was measured.

The identical encoder blend scored **0.391534** on seed 13 versus the matched
**0.358190** control, a **+0.033343** gain (paired prompt interval
**[+0.020375, +0.045941]**, all five folds positive). The head hit the iteration
cap on four of the five seed-13 folds. Both seeds showed gains in all populated
length bands. GPT-5.6 Sol gained about 0.087 and 0.113 recall on the two seeds.
The largest decline was about 0.027 recall on seed 13 for both GPT-5.5 and
Gemini 3.7 Flash. The secondary split reuses previously explored prompts.

If the full encoder screen clears the primary gate, check seed 13 as a fixed
rejection-only stability check. A nested selection replay then uses the three
registered outer folds (seed 121) and five grouped inner folds per outer
training partition (seed 212 plus outer index). The five inner folds retain
the primary four-of-five rule. In each outer training partition, screen
the CPU arms first, choosing the provider arm before the masked arm if both
qualify. Only when neither qualifies, replay the encoder pilot on inner folds
0 and 1, then the full inner screen if the pilot passes. Keep the baseline
when no arm qualifies. Score the selected recipe and matched baseline once
on the outer holdout. The replay evaluates the whole selection procedure,
and all prompts have been explored before this project phase. It does not
by itself establish an untouched evaluation of a fixed encoder-only choice.

## D058 — Self-neutralized style residual is refuted at P0 (2026-11-19)

Approach A of the [design doc](../superpowers/specs/2026-11-19-style-residual-distillation-design.md)
replaces the non-deployable Prompt-Neutralized Embedding with a learner: a
ridge regressor `g` predicts a passage's own prompt mean from that passage, and
the representation becomes `e - lambda * g(e)`. It is deployable at any k because
it never sees a sibling. The spec required this to be falsified on CPU before any
GPU spend, against **D010**, whose global projection of the top between-prompt
directions is a special case of the same idea and was already published as a
negative result.

Pre-check (`artifacts/residual/headroom.json`, `python -m experiments.residual_headroom`,
2699 rows / 150 prompts): the share of feature variance explained by prompt
identity is **0.2587** for the 56 style features, **0.8449** for `char` TF-IDF
reduced to 128 SVD components, and **0.3494** for the 384-dimensional hashing
embedding. This is a property of the labelled data, not a CV score.

Screen (`python -m experiments.run_residual_cv --splits
artifacts/splits/splits_official_verified_s42.json`, split
`stratified_group-s42-n5`, 2699 rows, 629.92 s, `30bc1ef`). One fixed
`LogisticRegression(C=1.0, class_weight=balanced)` in every cell, so only the
representation transform differs. Lambda chosen per fold by inner prompt-grouped
CV on the training side, then refit on the whole training side.

| space | none | D010 (k=4) | learned g | oracle centring |
|---|---|---|---|---|
| style | 0.2897 | 0.2856 | 0.2863 | 0.3575 |
| svd128 | 0.2293 | **0.2563** | 0.2215 | 0.2982 |
| embedding | 0.1967 | 0.1841 | **0.2004** | 0.2393 |

Learned-`g` diagnostics: out-of-fold `R^2` of predicting a row's prompt mean was
**0.316 / 0.259 / 0.255 / 0.278** in style space, **-0.243 / -0.243 / -0.236 /
-0.228** in svd128, and **-0.012 / -0.025 / -0.011 / -0.025** in embedding space
(one value per fold that selected lambda > 0). The removed-component ratio
`mean ||lambda*g(e)|| / mean ||e - mean||` was 0.08-0.34 for style, up to 0.85
for svd128 and up to 0.38 for embedding.

Verdict: **A is refuted at P0, and no GPU work follows** (spec section 5).

- The learned neutralizer never beats D010 by a meaningful margin on the space
  where D010 helps. In `svd128`, D010 gains **+0.0270** over no transform while
  learned `g` loses **-0.0078**; a large predicted component (ratio up to 0.85)
  with a negative `R^2` means the neutralizer subtracts a confidently wrong mean.
- Style space is the only one where the prompt mean is genuinely recoverable
  (`R^2` 0.26-0.32), yet the removal ratio stays low and the classifier does not
  benefit: 0.2863 versus 0.2897 untransformed, with per-fold deltas
  **-0.002 / 0.000 / +0.002 / -0.004 / -0.013**.
- The two positive-looking cells (embedding 0.2004 vs 0.1967, style learned vs
  D010 +0.0007) are inside fold noise and do not survive the four-of-five-fold
  rule, let alone the seed-13 gate.
- These are single-view logistic cells. They are not directly comparable with the
  adopted 0.3760 two-member blend, and the gate comparison that matters here is
  within-space: every learned cell is at or below its own untransformed control,
  and the spec's condition that learned `g` beat the D010 projection fails.

The oracle ceiling is retained as the measured headroom: exact within-prompt
centring still reaches 0.3575 on style features, so the between-prompt component
is worth removing *when the groups are known*. The failure is specifically that a
single passage cannot predict its own prompt's mean well enough to do it without
them. `src/residual.py` and the two experiment scripts stay in the tree: they are what
produced these numbers, and the module is covered by tests. The
`residual_lambda` stage that had been added to `TextMatrixBuilder` and
`ModelSpec` was removed afterwards, since the experiment called the neutralizer
directly and nothing else used the wiring.

## D059 — D010 rank and SVD-width sweep as a third blend member (2026-11-19)

D058 left one cell where removing prompt directions helped: `char` TF-IDF ->
128 SVD -> project out the top 4 between-prompt directions, then logistic
regression, at **0.2563** pooled OOF. The adopted blend scores **0.375971**.
This screen asks whether that cell, swept over SVD width and removed rank, lifts
the blend.

Recipe fixed before scoring: equal weight on the two adopted members and the
candidate, `(z_char + z_style + z_candidate) / 3`, with no weight search on the
OOF. `python -m experiments.sweep_d010 --splits
artifacts/splits/splits_official_verified_s42.json`, split
`stratified_group-s42-n5`, 1376.08 s for the 12-cell grid. The adopted control was
rebuilt from the registered members and matched 0.375971 before any delta was
reported.

| SVD width | single-view range over k = 2, 4, 8, 16 | blend range | delta vs adopted |
|---|---|---|---|
| 64 | 0.1978-0.2082 | 0.3697-0.3719 | -0.0063 to -0.0041 |
| 128 | 0.2560-0.2593 | **0.3830-0.3856** | **+0.0070 to +0.0096** |
| 256 | 0.2679-0.2786 | 0.3619-0.3656 | -0.0141 to -0.0104 |

Best cell **svd128, k=8**: 0.3856 (+0.0096). Per-fold deltas against the adopted
blend (folds 0.3779 / 0.3759 / 0.3796 / 0.3741 / 0.3722): **+0.0056 / -0.0019 /
+0.0056 / +0.0037 / +0.0352**. Paired prompt bootstrap, 4000 draws, second run
(`artifacts/residual/d010_svd128_ci.json`, 514.37 s):

| cell | blend | delta | paired 95% interval |
|---|---|---|---|
| svd128, k=2 | 0.3841 | +0.0081 | [-0.0019, +0.0182] |
| svd128, k=4 | 0.3830 | +0.0070 | [-0.0030, +0.0170] |
| svd128, k=8 | 0.3856 | +0.0096 | [-0.0004, +0.0196] |
| svd128, k=16 | 0.3849 | +0.0089 | [-0.0015, +0.0193] |

Verdict: **not promoted**. The selection gate requires a gain that is not carried
by one fold and a paired interval excluding zero; neither holds. Fold 4 supplies
+0.0352 of the +0.0096 pooled gain while folds 0-3 are flat to slightly negative,
and every interval includes zero. The effect also depends on SVD width alone: all
four ranks inside the 128-wide view land within 0.003 of each other, while the
64-wide view loses the signal and the 256-wide view turns negative. A plausible
reading is that a 128-dim char view is complementary to the adopted char/style
members, while 64 dims is too lossy and 256 dims is dominated by prompt variance
that the projection does not remove — a hypothesis, not a measured cause.

Before this could be adopted it needs the protocol's nested route: choose
(svd, k) by inner prompt-grouped CV inside each outer training fold, score the
fixed equal-thirds blend once on the outer holdout, then repeat on seed 13. None
of that has been run, no submission uses this cell, and the adopted blend and
submission are unchanged.

The nested replay scored **0.373743** versus **0.351892** for the matched
baseline, a gain of **+0.021852** (paired prompt interval
**[+0.010720, +0.033704]**). It chose the frozen blend in two outer folds and
the sparse fallback in the other when the inner encoder pilot failed. This
clears the plan's local gate for the selection procedure and justifies staging
the exact 25% frozen blend that the rule chose on the full training data after
the CPU arms failed. Kaggle runtime verification and a new public score remain
separate gates. See `artifacts/score_improvement/nested_replay/results.json`.

The local offline refit and inference check produced a valid 4,000-row synthetic
Parquet in 83.94 s of prediction time, using 0.792 GiB of peak torch-allocated
VRAM. A separate local run validated all three competition example IDs. Both
used the exact 25% ModernBERT recipe and hit the fixed 500-iteration head cap.
These checks do not measure Kaggle accuracy or Kaggle GPU compatibility.

---

## ADR — SASCR sibling-aware source-conditioned readout (D056)

**Date:** 2026-11-19
**Status:** ACCEPTED — design only; measurement blocked on optional `gpu` deps
           (`torch`, `transformers`, `sentence-transformers`) and a local Qwen3-8B
           checkpoint. Backend is `hashing` (CPU-only deterministic) in tests.

### 1. Motivation
The 18-class task is dominated by within-lab version siblings (~50% of errors):
gemini-flash variants, glm-5.3↔flash, gpt-5.6 trio, anthropic 4.6/4.8/5/sonnet-5,
gemini 3.7/3.8. The 18-class sparse/styometric trunk (pool-MIX cap40k CV
0.442675, D037/D039) is strong but plateaus because encoders trained on the
sparse views generalise poorly to unseen prompts. Two complementary levers
address the sibling ceiling directly:
1. A **frozen multi-layer causal readout** (Member B): extract hidden states at
   several transformer depths, LayerNorm-standardise each, content-token-pool,
   concatenate. Early/middle layers retain generative style far better than the
   final layer (the fleet that lifted public scores from ~0.69 to ~0.76).
2. A **sibling residual classifier** (Member C): a residual head, fit only on
   the hard intra-lab clusters, producing 18-column decision margins used as a
   correction in the row-z blend — not a stand-alone 18-class model.

### 2. Scope and risks
- Member B listens to optional `backend="transformers"`/`"sentence-transformers"`
  if installed; defaults to `backend="hashing"` otherwise, so the codebase
  stays runnable offline. Real backbone download (~2-6 GB) is separate from
  this repo and cached on disk. TAKE CARE NOT TO SUBMIT non-transductive
  measurements as competition scores.
- Member C is deliberately constrained: it sees only the sibling labels, its
  `decision_function` preserves the 18-label order (seeded -1e8 padding), and
  it is blended in log-margin space, so it cannot mask a member's confident
  errors. It is a residual, not a replacement.
- The 18-class classifier path (the trunk) is unchanged and remains the
  primary comparison.

### 3. Measurement protocol (D018/D024)
- 5-fold StratifiedGroupKFold on `prompt_id`, scored ONLY on official train.csv
  rows (no pool rows in the denominator), repeated over k=2 prompt
  subsamples (the hidden-test prompt multiplicity, k≈2 is multiply
  corroborated by the community logs). See `docs/experiment_protocol.md`.
- Each member refit per fold; pool rows screen non-transductively (D028) and
  are excluded from the validation denominator.
- Report pooled OOF BA per fold + paired prompt CI vs the 0.442675 candidate;
  the seed-2027 holdout (D024) is gated before any submission.

### 4. Config
`configs/sascr.yaml` (design block) fixes the member list, weights (adopted
0.5 char + 0.5 style, D037/D039), backbone depths and sibling clusters.
`status: design` members are NOT loadable by the CV runner until their
feature builder and classifier are implemented — this blocks unfinished-member
end-to-end runs by construction.

### 5. Reviewer checks
- [ ] `python -m pytest tests/test_sascr.py -q` (10 passed + 1 skipped)
- [ ] `python -m pytest tests/test_model.py -q` (SiblingResidualClassifier
      decision_function contract, 18-column shape)
- [ ] k=2 grouped CV run vs the 0.442675 candidate (after optional deps + backbone)

### 6. For the record
This ADR is consistent with the strategy doc (ATLAS is strong cross-family but
weak on version siblings — the sibling residual targets exactly that),
arXiv:2610.28434 (multi-resolution fingerprints survive distillation),
arXiv:2610.19583 (ATLAS source-conditioned likelihood), and the
Multi-Layer LoRA Readouts / WSDM-LMSYS recipes. It is the design-only
implementation stage; scores are not claimed.

## D061 — Provenance and independent evaluation (review follow-up)

Renumbered from D017: this branch already used D017–D059 (and D060 lives on
`sascr-methodology`), so the ADR that arrived with `origin/master`'s hardening
commit (`d3547e3`) is recorded here instead. Content is unchanged apart from the
number.

Saved folds and OOF files now require order-sensitive data and fold provenance;
legacy artifacts must be regenerated. Blend selection enforces its minimum gain,
and alternate-seed checks use fixed weights. The former secondary holdout was
not independent of selection; its numbers remain historical exploratory results.
Future independent evaluations reserve prompt groups before selection as
specified in `docs/experiment_protocol.md`. No new official score is claimed.

Reopen if: a caller needs the older lenient behaviour for legacy artifacts, in
which case regenerate the artifacts rather than relaxing the checks.
