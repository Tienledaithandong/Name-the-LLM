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
