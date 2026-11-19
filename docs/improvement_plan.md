# Improvement plan — 0.356 → 0.7 (revised after strategy research)

> **Current execution plan, 2026-11-18.** Kaggle submission 58502550 matched
> the previous **0.356** public score. Start with a CPU experiment to distinguish
> versions from the same provider. The [score improvement plan](../plans/name-that-llm-score-improvement.md)
> sets the official-only experiments, controls and promotion gates. Its priorities
> supersede the pool submission recommendations in the historical notes below.

> **2026-11-16 validation hold (D024):** the DeBERTa-large + sparse blend
> records 0.4442 grouped OOF, but is provisional: legacy OOF files do not prove
> row/fold identity and the external pool prompt isolation is unverified. The
> seed-2027 subset is not untouched end-to-end. Do not adopt, submit, or treat
> a rerun as comparable until identity, leakage and offline inference checks pass.
> See `docs/experiment_protocol.md` and the latest ADR. Historical statements
> below about k≈2, transductive calibration being safe, or a fresh holdout are
> superseded: hidden k is unverified and transductive scores violate hard rules.

> **Official-only continuation (2026-11-16, D026):** the user confirmed there is
> no trustworthy external-prompt crosswalk and requested the official-data
> baseline. Per-fold pool filtering uses held-out text and is transductive; a
> global mask built from all official text also uses validation text. Therefore
> the D028/D037 pool claims and packaging recommendation below are not eligible
> for reported CV or adoption under this repository's rules. Do not run or submit
> those candidates. The identity-verified DeBERTa official-only run scored 0.1915;
> its fixed 50/50 blend with sparse fell from 0.3760 to 0.3393 on the same OOF.
> Keep the 0.3760 sparse result as the research baseline; no new candidate is
> submission-ready.


> **Status (2026-11-14, revision 2):** active. Public LB **0.356** (argmax blend).
> Official leaderboard top **0.771** (5,851 teams; gold ≈ top 21, silver ≈ top 50;
> public 40% / private 60%). Evidence base: `docs/strategy_review.md`.
>
> **Major correction (ADR D017/D024):** hidden-test prompt multiplicity is
> unverified; community reports suggest ~1–3 passages per prompt. Our crossed Hungarian candidate (D016) was
> validated only at k=18. **Honest k-subsampled CV (Phase 0 below) shows the
> Hungarian gain collapses from +0.10 at k=18 to ~0 (or negative) at k=2.** The
> crossed candidate is therefore **withdrawn as a submission default**. The ~0.34
> gap to the leader remains large; the leading private methods are unknown.

## Current work — 2026-11-16

- **Identity-checked sparse baseline:** fresh seed-42 grouped OOF reproduces
  **0.375971**. Its ordered dataset fingerprint and fold assignments match an
  independently regenerated split. The previous sparse and DeBERTa OOF files
  without these metadata are rejected by the new comparison guards.
- **Official-only nested check:** three outer / three inner grouped folds scored
  **0.351892** fixed-weight pooled BA. Every inner selection chose the existing
  0.5 char / 0.5 style blend from the pre-declared {0, 0.5, 1} grid. This has
  smaller outer training folds, uses only the two sparse members, and prior
  development involved all official prompts; it does not validate DeBERTa or
  constitute a completely untouched final test. A diagnostic rerun recorded
  macro F1 **0.342986**; 940 of 1,749 errors stayed within a provider.
- **External pool — REOPENED non-transductively (D028):** the D026 block was on
  *per-fold* decontamination (held-out text selected which pool rows trained), so
  `src.pool.decontaminate_global` now screens the pool **once** against the **full** official
  train set (exact text + word TF-IDF cos>0.3) into one **fold-independent** clean set
  (226,191 → 226,049; exact 0 + near 142 removed). Text-level screening is stricter than the
  unverified `prompt_canon`↔`prompt_id` crosswalk, so the 1,533 unmapped-ID rows are kept
  (only true near-dupes drop). Measured on CPU (char_tfidf, cap 12k/fold): char official-only
  **0.3193 → +pool 0.3667** (paired **+0.0474 [+0.0217,+0.0731]**, all 5 folds positive,
  class signature matches the transductive run → the gain was coverage, not the leak);
  `style_only` is **not** helped (−0.0174, CI spans 0). Rebuilding the blend with the
  fallback's own rule (row-wise z + fixed 0.5/0.5, reproduces **0.3760 exactly**) gives the
  candidate **MIX = 0.5·z(char+pool) + 0.5·z(style official) = 0.3990**, paired **+0.0230
  [+0.0079,+0.0381]** (4-member 0.4012, +0.0252; both-pool 0.3812 spans 0). **Two-seed confirmed:**
  the identical fold-independent mask on the **s13 alt-seed** gives MIX **0.3989** vs s13 base 0.3582
  = paired **+0.0408 [+0.0200,+0.0615]**, all folds positive (char alone +0.0545 [+0.0276,+0.0813]) —
  MIX pooled ≈0.399 on both seeds. **Not yet adopted** — only the reserved seed-2027 holdout
  (D018/D024) remains. Until then the 0.3760 / 0.356 fallback stands. Historic transductive pool
  metrics stay provisional.
- **Gemini diagnostic:** Open Pool v15 contains 3,541 Gemini 3.7 rows vs 23,474
  Gemini 3.8 rows; its legacy DeBERTa OOF predicts 3.7 correctly 0/150 times.
  Class imbalance is a hypothesis, not an established cause. Aggregate-only
  diagnostics: `artifacts/diagnostics/pool_class_audit.json`.
- **Independent official-only DeBERTa check:** a separate private, seed-42 GPU
  notebook was started; its result is pending and is not a leaderboard score.
  It does not modify the earlier running pool notebook.
- **Offline package:** the candidate builder requires real weights, exact label
  metadata and matching sparse model configuration. A CPython 3.13 Linux
  scikit-learn 1.9.1 wheel is prepared under `artifacts/kaggle_wheels/` (no
  dependencies included). A private internet-disabled Kaggle CPU kernel loaded
  both sparse models and inferred 4,000 synthetic passages in 2.685 seconds;
  this is a synthetic pipeline smoke, not a leaderboard score. Real DeBERTa
  candidate weights and the combined GPU-scale offline benchmark are not yet
  available.

## 1. Diagnosis (corrected)

- **0.356 LB ≈ 0.36–0.39 honest k≈2 CV.** The argmax blend generalises about as
  well as honest CV predicts. The gap to 0.771 is **capability**: we score each
  passage independently with char n-grams + 56 style features; top solutions use
  fine-tuned encoders, external labelled pools, and multi-backbone readouts.
- **The crossed-design lever was overstated.** Train is a full factorial (k=18),
  but the hidden-test multiplicity is unknown. One-per-model assignment depends
  on recovered groups; the historical hypothetical-k≈2 stress test suggests that
  its gain is fragile and can overwrite correct predictions.
- **Batch post-processing is not validated.** Sinkhorn is transductive, can hurt
  under unequal class priors, and is prohibited by the current reporting rule.
  Typography and sibling features require controlled training-only evaluation.
- **The dominant remaining error is intra-lab siblings** (community: 77.78% of
  errors; ours: 53.3% within provider). gemini-flash variants, glm-5.3↔flash,
  gpt-5.6 siblings, claude-opus versions. This is a fine-grained style problem.

## 2. The real ladder (community tiers, bracketed by official 0.771 and our 0.356)

| tier | method | reported public BA |
|---|---|---|
| 1 | TF-IDF / classical (us) | 0.33–0.42 |
| 2 | GBDT on stylometry + SVD | 0.39–0.42 |
| 3 | DeBERTa-v3 sequence head (official only) | 0.39–0.42 |
| 4 | DeBERTa-v3 + clean external pool | ~0.576 unseen-prompt CV → ~0.66 LB |
| 5 | Qwen3-8B/14B LoRA multi-depth readouts | 0.55–0.585 |
| 6 | multi-backbone LoRA stack + pool + calibration | 0.66–0.69 |
| 7 | + source-conditioned likelihood / GLM token features | 0.69–0.77 |

Tiers are **self-reported** by community sources (not audited), but the official
top (0.771) and our 0.356 bracket them credibly. We climb one rung at a time,
validating each on the registered grouped non-transductive protocol before submitting.

## 3. Phased plan (ordered by impact ÷ cost, within our constraints)

Constraints: **single account**, CPU-first (GPU notebook ≤9h allowed), **internet
OFF at inference**, external data/pretrained models allowed if public + attached
offline, owner-only account actions (rules, download, upload, submit). No leaked
credentials, no multi-accounting (see `docs/strategy_review.md` §3).

### Phase 0 — honest validation + k-robust post-processing — DONE 2026-11-14

- Implemented `src/calibration.py` (damped Sinkhorn-Knopp, k-robust, no grouping)
  + 9 unit tests. Implemented `experiments/stress_k.py` (primary full grouped OOF
  + k-subsampled **paired-difference stress test** with 95% CIs; replaces
  `phase_k2_cv.py` per D018) + `metrics.paired_difference`/`mean_ci` (5 tests).
- **Result (decisive):** the recovered-Hungarian gain is a k=18 artifact.

| split | k | argmax | recovered Hungarian | best Sinkhorn |
|---|---|---|---|---|
| cv_s42 | 18 | 0.3760 | **0.4690 (+0.093)** | 0.3767 |
| cv_s42 | 2 | 0.3607 | 0.3643 (+0.004) | 0.3611 |
| cv_s42 | 3 | 0.3830 | 0.3835 (+0.001) | 0.3850 |
| holdout_s7 | 18 | 0.3956 | **0.5000 (+0.104)** | 0.4000 |
| holdout_s7 | 2 | 0.3872 | 0.3833 (**−0.004**) | 0.3964 |
| holdout_s7 | 3 | 0.4278 | 0.4238 (−0.004) | 0.4304 |

- **Decision (D017 + D018):** withdraw crossed Hungarian as the submission default;
  keep the **argmax blend** as the safe baseline. Sinkhorn is implemented and
  k-robust but ~neutral on the weak blend → reserve it for Phase 4 (on a strong
  ensemble). **Primary metric for independent-passage models = full prompt-grouped
  OOF BA** (argmax blend: **0.3760** cv_s42 / 0.3956 examined holdout). The k=1,2,3,18
  subsampling is an **inference-only stress test** for batch-dependent methods,
  reported as **paired differences (method − argmax) with 95% CIs** over 10 seeds —
  not a selection score, and transductive methods stay out of the primary CV. At
  k=2 the Hungarian paired gain is only +0.0068 [+0.0032,+0.0104] (cv_s42) /
  +0.0029 [−0.0059,+0.0117] (holdout, CI spans 0); Sinkhorn CIs span 0. The seed-7
  holdout is **examined**; a fresh seed-2027 holdout is reserved for final selection.

### Phase 1 — clean external pool + stronger base models (highest ROI)

The single biggest lever is **more labelled data + a fine-tuned encoder**, not
post-processing. Compare on full grouped OOF and a pre-specified grouped evaluation.

- **1a — external pool audit + per-fold decontamination (DONE, D019):**
  `crtproduction/name-that-llm-open-pool` **v15** (2026-11-14, CC BY-SA 4.0, 65
  sources) registered in `docs/external_data_register.md`. Verified against the
  actual parquet (`src/pool.py`, `experiments/pool_audit.py`, 9 tests):
  **226,191 rows**, all 18 labels exact match, 0 null text, uneven
  (3,541…23,474 → class weights). **Routing-risk flags all False** — the 10,838
  unverified gemini-3.7 rows are already in `gemini37_unverified.parquet`, so
  re-filtering `flag_gemini37_routing_risk` is a no-op (corrects the original
  instruction). Dataset `max_cos_train` ≤ 0.30, 0 exact dups; independent per-fold
  word-TFIDF cos>0.30 removed **19** rows union → **clean 226,172**
  (`artifacts/pool_audit/audit.json` + `clean_pool_ids.json`). **Caveats:**
  `match_weight`/`style_score` fitted on all train → refit per-fold or drop;
  `flag_kb_proxy_after_ruling` 20,227 (keep by default, optional strict drop);
  CC BY-SA share-alike. Admission provisional pending the 1b/1c validated gain.
- **1b — DeBERTa-v3 sequence head (Kaggle GPU ≤9h, offline inference).** The
  training kernel `kaggle/deberta/deberta_train.py` (self-contained script kernel)
  fine-tunes `deberta-v3-base` (18-class) on the **same grouped folds**, official-only
  vs official+clean-pool, with per-fold decontamination (D019), a throughput/memory
  **benchmark cell** vs the 9h budget, and a stratified **pool cap/fold** (default
  40k). `--save-final` exports weights for a separate OFFLINE inference notebook.
  **Prepared** (this session): the offline inference pipeline
  `kaggle/deberta/predict.py` (loads weights from an attached folder, never
  downloads; writes valid `submission.parquet` with fail-fast checks), and
  `kaggle/deberta/requirements_gpu.txt` (pinned GPU deps). `kaggle/deberta/`
  kernel-metadata.json pushes as a private GPU kernel. All offline-logic tests
  pass (3/3, `tests/test_offline_submission_artifact.py`). **GPU runs** (private
  kernel `tiennamlee/name-that-llm-deberta-official-vs-pool-cv`,
  `kaggle kernels push -p kaggle/deberta`):
  - **v2 base** (len256, 3ep official / 1ep pool, cap40k): official-only **0.2241**,
    +pool **0.3049**; pool paired gain **+0.081** [+0.054,+0.108] (real) but **still
    below the 0.376 sparse blend**. Community ~0.39/~0.576 **not reproduced** at this
    budget (D022). Fixed the DeBERTa-v3 fp16-gradient crash (`model.float()`).
  - **v3 large** (len384, batch8, 3ep/2ep, cap20k, lr1e-5): **FAILED** — official-only
    pooled 0.2245 but **fold 0 collapsed to 0.0537 (~chance)** while folds 1–4 reached
    0.23–0.31 (training instability near the T4 memory ceiling, peak 13.67 GB); the pool
    variant then **CUDA-OOM'd** (14.5 GB). Two root causes found: (1) **Kaggle script
    kernels do NOT upload companion files** — `folds.json` was dropped, so folds were
    recomputed on Kaggle (not aligned to local cv_s42); (2) large@len384/batch8 exceeds
    T4 memory in the disentangled attention. Diagnostic only, not a clean comparison.
  - **v4 large** (len384, **batch4 + gradient checkpointing**, `expandable_segments`,
    cap12k/fold, 3ep/2ep, lr1e-5, **folds embedded in the script** as a per-row fold-id
    array verified byte-identical to local cv_s42): **complete** — memory-safe
    (peak 10.2 GB, 7.2 h < 9 h, 6.7 seq/s) and fold-aligned (`split_method =
    stratified_group-embedded`), the first DeBERTa OOF safe to blend across pipelines.
    Pooled grouped-CV BA (D023): official-only **0.2482**, official+pool **0.3857**
    (pool paired **+0.1375**, CI [+0.054,+0.221]), and the fold-aligned 50/50 blend
    `0.5·(0.5 char+0.5 style)+0.5·DeBERTa-pool` = **0.4442** (paired vs sparse
    **+0.0682**, CI [+0.048,+0.089]; DeBERTa-alone vs sparse *not* significant, +0.0097
    CI [−0.017,+0.036] — the gain is diversity). **Provisional, not adopted** (D024):
    legacy sparse/DeBERTa OOF lacked row+fold provenance, so 0.4442 is a within-OOF
    measurement pending provenance-backed confirmation on the alternative seed (s13) and
    the reserved seed-2027 holdout before it can displace the 0.376 sparse fallback.

  Offline-inference flow: attach `final_model/` → `HF_HUB_OFFLINE=1
  python kaggle/deberta/predict.py --model-dir /kaggle/working/deberta_cv/final_model`
  → `submission.parquet`.

  **Package (this session):** the offline artifact dataset lives at
  `artifacts/kaggle_deberta_offline/` — a self-contained Kaggle dataset folder
  (no repo checkout at runtime). It ships `predict.py`, `requirements_gpu.txt`,
  a minimal `src/` (`constants.py` + `submission.py`), `manifest.json`, and
  `dataset-metadata.json`; `weights/dummy_tiny/` is a reproducible stand-in that
  fills the offline smoke. The real weights replace `weights/dummy_tiny` via
  `weights/<final_model>`. Upload-ready bundle content: see
  `artifacts/kaggle_deberta_offline/README.md` and `dataset-metadata.json`.
- **1c — GBDT on richer stylometry + multi-view (CPU, DONE — negative standalone):**
  `gbdt` classifier (sklearn `HistGradientBoostingClassifier`; lightgbm/catboost/
  xgboost were absent *at that time* — catboost was installed later and measured in
  **#55/D049**, also negative) added to `src/model.py` (+4 tests, dense-input guard) and
  `configs/gbdt.yaml`. Official-only grouped OOF: **style-only 0.263**,
  **char+SVD200 0.254**, **multi-view SVD128 0.243** — all **below the 0.376 argmax
  blend**; SVD compression discards the sparse signal LinearSVC exploits.
  `experiments/pool_train.py` runs the official-only vs official+pool comparison
  (per-fold decon, stratified pool cap, paired-diff CI, registry rows). GBDT is a
  weak standalone here; adopt only if it adds a **validated** blend gain (D007).
- **1d — multi-view TF-IDF + calibrated LinearSVC:** char_wb 3–5, subword-boundary
  and punctuation-cadence views; blend with 1b/1c. CPU-feasible.

### Phase 2 — multi-backbone cached-embedding readouts (heavier GPU)

- **H8 (D052, queued first — cheapest encoder test).** Swap the sequence-head backbone from
  DeBERTa-v3-large to **ModernBERT-large**, keeping the D038 kernel's non-transductive
  `decontaminate_global` pool path, embedded folds and cap 40k. Rationale: two independent
  participants self-report ModernBERT-large ≈ **0.55 OOF / 0.55 LB** versus their own
  DeBERTa-v3-large 0.50 (spec §11.3), while our DeBERTa-v3 measured 0.2482 official-only
  (D023/D027). **D027's "encoders add no diversity" verdict was about one backbone, not about
  encoders.** Community numbers are unverified and are NOT our results. Owner/GPU-side; adopt
  only if it beats `mix_baseline` with a paired CI excluding 0 under all three grouped protocols.
  **ENABLED (D054) — it was not runnable before.** `kernel-metadata.json` is a **script** kernel,
  which Kaggle runs with **no CLI args**, so `--model-name` was unreachable on a pushed run; and
  `requirements_gpu.txt` floored `transformers>=4.45` while ModernBERT needs **4.48**. Now a
  module-level `BACKBONE` constant (same pattern as `FOLD_SEED`) selects the preset, the choice is
  recorded in `results.json` config + the inference manifest, and the floor is 4.48.
  `max_len/batch_size/lr` stay at the measured D038 budget (384/4/1e-5) for **both** presets so the
  first probe changes only the backbone; ModernBERT's long context is a separate later probe.
  Owner steps are in ADR D054. Still **no metric** — the GPU run is owner-side.
- Cache multi-layer hidden states from Qwen3-8B/14B (depths [2,4,6,8,12,16,20,24]),
  LayerNorm-standardised, content-token pooled; fit ridge/logreg heads on official
  + clean pool. Community ~0.55–0.585. Single-account GPU, offline weights.
- Centred-twin / shared-prompt readouts only where prompt groups are reliable
  (train-side); do **not** assume test-time groups at k≈2.

### Phase 3 — source-conditioned likelihood (optional, cross-family signal)

- ATLAS-style `P(text|model)` with per-source LoRA + clipped token likelihoods,
  applied only to contested rows (top-2 margin < ~0.32) at low weight (~0.20).
  **Caveat:** ATLAS is strongest cross-family; our errors are intra-lab siblings,
  so expect modest gains. Verify on honest CV before adopting. Heavy; defer.

### Phase 4 — independent-passage decision features

- Damped Sinkhorn (`src/calibration.py`) is transductive and excluded from
  reported scores and submission under the current hard rule.
- Typography guardrails: verify provider-invariant tokens on **train** first
  (e.g. curly quotes / em-dash patterns); apply as soft penalties, not blind masks.
  **H9 (D052): test hard family-level vetoes as well, CPU-only.** Two Discussion threads
  independently report near-deterministic punctuation rules — family accuracy 87–99% versus
  within-family version 67–76%, per-rule chance level < 1e-4 (spec §11.4). That is evidence
  against the blanket "not blind masks" guidance above, so the two are now an open comparison
  rather than a settled rule. **Conditions:** mine rules on each fold's training rows only, and
  record each rule's training support and chance level; a rule fitted on all train and applied
  to validation is the leak class D026 banned. **Already verified:** our adopted char member
  keeps raw characters (`lowercase=False`, no unicode/punctuation normalisation), so the
  community's "don't normalise punctuation" advice needs no code change.
  **H9 CLOSED — measured, not adopted (D053).** `src/typography.py` +
  `experiments/typo_vetoes.py`: candidate 0.442675 → 0.443415 (+0.000741), paired 95% CI
  [−0.001316, +0.002797] **includes 0**, so the gate fails. 9 flips / 2,699 rows, all from
  wrong predictions, none from a correct one. The soft style rates already do this work.
  Two durable findings: chance must count **distinct prompts, not rows** (2,699 rows = 150
  groups; the correction drops 10 of 22 rules with the score unchanged), and a tiny chance
  is **not** proof of an invariant — `em_dash → Google` survives at 2.74e-26 yet is broken by
  row 1554 (`gemini-3.7-flash`, prompt p149). Filtering rules by cross-fold recurrence would
  fix it but is transductive (D026) and is recorded as descriptive only. The oracle ceiling
  attributable to the veto is **−0.000370** once the no-veto control on the same rows is
  subtracted. The original "soft penalties, not blind masks" guidance above stands.
- Gentle sibling residual heads (max_alpha ≤ 0.08) targeting the dominant sibling
  confusions; preserve cross-family probability mass.

### Phase 5 — sibling discrimination (throughout)

- Hierarchical provider→version features; provider-conditional models for
  gemini-flash, glm-5.3↔flash, gpt-5.6 siblings, claude-opus versions.
- Length-band analysis (short passages are a likely weak spot).

## 4. Validation discipline (updated)

- **Primary:** full prompt-grouped OOF balanced accuracy, with exact row, dataset,
  and fold identity and training-side-only learned transforms.
- Earlier transductive stress-test numbers are historical exploratory results;
  do not create or report new transductive scores under the current hard rule.
- The seed-2027 subset participated in earlier full-OOF development and is not an
  untouched end-to-end evaluation. Use genuinely new labelled prompts or a
  pre-specified nested grouped protocol with selection inside each outer fold.
- Record each experiment and protocol change; never tune against hidden test or
  public leaderboard feedback.

## 5. Compliance posture

- Single account; no leaked credentials; no multi-account quota evasion.
- No competition rows to hosted assistants; no runtime downloads; offline weights.
- External data registered + decontaminated + validated on grouped folds before use.
- Owner accepts rules, downloads data, uploads artifacts, and submits.

## 6. Historical results (superseded framing)

Phase A oracle (known train groups, k=18) and Phase B recovered-grouping results
(ADR D014–D016) remain valid **as k=18 oracle/train-side measurements** but are
**not deployable estimates for a k≈2 test**. The D016 "adopt crossed inference"
decision is **retracted by D017**; D018 reframes the k-subsampled evaluation as an
inference-only stress test (primary = full OOF). See `experiments/registry.csv`
(`phase_a_*`, `phase_b_*`, `phase_k2_*` historical; `stress_*` current) and
`artifacts/stress_k/results.json` (current), `artifacts/phase_k2/results.json`
(historical).

## 7. Immediate next action

**Cap-40k Pool-MIX candidate is validated (D037) and being packaged; awaiting an owner submission.**
D028/D035 reopened the external pool non-transductively; task #39 then showed the 12k/fold cap was
the binding constraint, not the recipe. The fixed 0.5/0.5 blend `0.5·z(char trained on
train+global-clean-pool at cap 40k) + 0.5·z(style official)` now beats the 0.3760 fallback under three
independent grouped protocols — **s42 0.4427, s13 0.4304, nested outer-fold 0.4426** (all paired CIs
excluding 0, and +0.0667/+0.0722/+0.0908 over the 12k candidate). `src.inference.fit_final --pool` +
`cli predict --fit --pool --pool-models --pool-cap 40000` build the bundles, `cli package` writes
`artifacts/kaggle_dataset`, and `kaggle/submission.ipynb` (unchanged inference: row-normalise + 0.5/0.5
blend) is ready. **Submitting is owner-only** — upload the dataset, run the notebook on the real full
test, `kaggle competitions submit`, and report the public LB so we record the true number. Until a
leaderboard result confirms it, **0.356 / 0.3760 remains the last confirmed submission fallback** and
the pool-MIX score is grouped-CV only, not an LB claim.

Task #34 is **closed (D036):** the gemini-3.7-flash regression is root-caused as proportional-cap
sibling-starvation; a class-balanced cap flips 3.7 −0.173→+0.013 but loses on 3.8/3.6/opus-5 (blend
0.3949 < 0.3990, CI spans 0). **Kept the proportional cap; do not hand-tune a per-class prior on CV.**
At the wider 40k cap the 3.7 starvation grows (nested −0.127) but is overwhelmed by sibling gains
(3.6 +0.160, 3.8 +0.273). Balanced-at-40k was probed and **rejected** (D037: char standalone rises to
0.4234 but the blend drops to 0.4378 < 0.4427, paired CI spans 0). Remaining lever: retrain the
DeBERTa-pool kernel under `decontaminate_global` at 40k, only if it beats 0.4427 under the same three
protocols. **Task #40 (2026-11-16): the kernel is now wired for that route** — `kaggle/deberta/deberta_train.py`
no longer hard-aborts pool CV; it screens the pool once against the full official train (D028, fold-independent),
defaults `--pool-cap-per-fold 40000`, and keeps the transductive `--pool-scope fold` path disabled (D026). The
GPU run and any blend validation are **owner/external** (local torch is CPU-only); no DeBERTa-pool number may be
recorded until a real kernel `results.json` exists. The kernel's `FOLD_SEED` is set to **s42** so that run is
comparable to the candidate's primary protocol.

**Task #41 / D039 (2026-11-16): the blend bar is the candidate itself, and the D037 headline now reproduces
independently.** `experiments/blend_deberta.py` previously scored new members against the old official-only 0.376
blend; it takes `--char-oof` / `--style-oof` separately so the baseline is the adopted cap-40k MIX, and `--no-deberta`
scores that baseline alone from stored artifacts. Rebuilt from identity-verified OOF it reproduces **s42 0.442675** and
**s13 0.430435**, with the official-only control at **0.375971** — the recorded 0.4427/0.4304/0.3760, from a **second
code path** under enforced fold+fingerprint alignment (registry `official_pool_global-blend_reproduce_*`;
`tests/test_blend_deberta_baseline.py`). Once the owner's D038 kernel output lands, score it with:

```
python experiments/blend_deberta.py --deberta-dir <kernel_out>/deberta_cv --variant official_pool \
  --char-oof artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz \
  --style-oof artifacts/cv_identity/official/oof/style_only.npz
```

Adopt a DeBERTa arm only if `mix+deberta_*` beats `mix_baseline` with a paired CI excluding 0 under all three
protocols. Do not submit the crossed Hungarian candidate (D017). Keep strict OOF identity and the
non-transductive global screen.

**Tasks #42/#43/#44 / D040 (2026-11-16): portfolio and submission artifacts now match the evidence.** An audit
against the plan's §9 exit gate found the three owner-facing documents were ~20 ADRs stale, and the packaged
candidate misdescribed itself:

- `report/research_report.md` claimed **"No leaderboard submission has been made"** (contradicted by registry
  `20261114T134205-kaggle-submission-58344764`, public LB 0.356) and still headed §6.6 crossed inference
  **"— ADOPTED"**. Rewritten: current-state banner (D017→D040), abstract carries the pool result, §4 adds the
  paired-CI / OOF-identity / global-screen / nested-gate protocol, §5 adds the pool training-row path and the
  fixed-rule blend, §6.3 adds H5–H7, **new §6.8** (pool audit, D026 leak, D028 global screen, cap table, paired
  CIs, nested gate, D039 reproduction, starvation + balanced-cap rejection, deployment) and **new §6.9** (GBDT,
  DeBERTa base/large, the D027 identity table, the D038 open lever), §9 efficiency uses measured pool fit times
  (1,599.6 s / 2,142.4 s five-fold; notebook does no fitting), §10 limitations and §11 appendix refreshed, §12
  states the figures are the baseline's. Every "untouched holdout" is now "examined". All numbers traced to
  `experiments/registry.csv`, the ADRs, or `artifacts/pool_audit/audit.json`; one unverified figure carried over
  from a summary (raw-probability averaging 0.3241) was **removed** rather than published.
- `README.md` status block no longer advertises the superseded 0.4442 provisional DeBERTa candidate; it names the
  cap-40k pool-MIX candidate as **packaged but not submitted**, with new result tables (§External-pool candidate,
  §Neural and tree-based models), the pool build commands, an updated architecture diagram, compliance posture
  covering the CC BY-SA 4.0 corpus, and rewritten limitations. Test count corrected to 234.
- `docs/submission_checklist.md` D024 "validation hold" banner replaced with the candidate-ready state; adds the
  candidate's evidence, the rejected variants, the exact rebuild/upload/submit commands and the **revert rule**
  (public LB ≤ 0.356 → rebuild official-only and record the disagreement).
- **D040 (code fix).** `cmd_package` copied the *official-only* ensemble `weights.json` over a package whose char
  bundle is `train+global_pool(capped=40001)`, so the shipped file asserted `pooled_oof_balanced_accuracy: 0.37597`
  for a pool-trained fit. Operative weights were identical (0.5/0.5) — **predictions were never affected** — but the
  artifact misattributed a score. `_align_packaged_weights` now strips the official-only descriptive fields when any
  shipped member's `fit_mode` mentions the pool and records `fit_modes` + `cv_score_of_record: experiments/registry.csv`;
  an all-`train_only` package keeps them. `kaggle/submission.ipynb` cell 0 no longer says "External-pool candidates are
  disabled under D026" while printing the `--pool-cap 40000` command. Two new tests in `tests/test_package.py`;
  full suite **234 passed**. Package rebuilt from the intact `artifacts/final/official` and re-verified offline in a
  notebook-equivalent simulation (bundles load, class order matches `LABELS`, weights 0.5/0.5, schema-valid
  `submission.parquet` for the 3 example rows — smoke check, not a submission).

**Task #45 / D041 (2026-11-17): the last uncovered plan §9 requirement — text-length sensitivity — is now measured.**
Plan §9 item 9 asks for "sibling models versus model families; text-length sensitivity"; report §8 had only the
confusion half and admitted the length half had not been run (and wrongly called it "wired into the audit" — the
audit draws a length *distribution*, not per-band accuracy).

- **New metric + tests.** `src.metrics.length_band_metrics` scores `[edges[i], edges[i+1])` bands with an
  open-ended top band, reports `classes_present` (a band's BA averages only the classes it contains) and returns
  `None` for empty bands. Five unit tests cover edge assignment, absent classes, per-band provider errors, the
  single-band == pooled invariant, and rejected edges/lengths. One test caught a real bug (`zip(..., strict=True)`
  on a shifted sequence) before it reached the report.
- **New analysis script.** `experiments/length_sensitivity.py` is post-hoc on saved OOF: it verifies the train
  frame's ordered fingerprint against the frozen splits and every member's OOF fingerprint/labels/fold ids before
  scoring, rebuilds both blends with the adopted 0.5 row-z char + 0.5 row-z style rule, and **fits nothing, reads
  no pool row and scores no test row**. Bands (0, 50, 100, 150, 200) words; the corpus spans 36–274 (median 101),
  so the open-ended top band keeps every band ≥300 rows and all 18 classes appear in all five bands.
- **Third independent reproduction.** The run returns official-only **0.375971** and pool-MIX cap-40k **0.442675**,
  plus char 0.319306 / style 0.290398 / char+pool 0.410445 — all equal to the recorded values.
- **Findings.** Accuracy rises monotonically with length for every variant (official 0.2491→0.4718, candidate
  0.3125→0.5858). The pool gain is length-dependent: **+0.0634 / +0.0431 / +0.0693 / +0.0899 / +0.1140**, largest on
  the longest passages, so external data amplifies existing evidence instead of fixing short passages. The
  within-provider share of errors rises with length (40.9%→66.7% official, 40.0%→64.1% candidate): short passages
  fail at provider level, long ones at version level. The candidate removes 180 errors, 114 of them sibling
  confusions (63%); 14/18 classes improve, `gemini-3.8-flash` +0.253 vs `gemini-3.7-flash` −0.180 — the D036/D037
  starvation trade visible per class.
- **Figures.** `src.figures` gained `make_prediction_figures` (extracted from `make_report_figures`, identical
  output) and `make_length_band_figure`; three new portfolio PNGs: `length_band_accuracy.png`,
  `confusion_matrix_pool_mix_cap40k.png`, `per_class_recall_pool_mix_cap40k.png`. Report §12 now shows ten figures
  and states which belong to the baseline and which to the candidate.
- **Recorded.** Two registry rows (`*-length_bands`, status `done`, no fit seconds claimed), ADR **D041**, report
  §8 split into §8.1/§8.2 with the measured table, §11 command added. The report states explicitly that the local
  `test.csv` is the 3-row interactive example, so the hidden test's length mix is **unknown** and any length
  explanation of the −0.020 CV/LB gap is an untested hypothesis, not evidence.

**Task #46 / D042 (2026-11-17): plan §7's blend-rule comparison is now measured — the adopted row-z rule stands.**
Plan §7 asks for "simple averaging, rank-based and normalized-margin blending" to be compared. Until now the row-z
choice (D020) had never been tested against alternatives, and the report had carried an untraceable "raw average
0.3241" figure that had to be deleted for lack of evidence.

- **New script.** `experiments/blend_rules.py` scores five scale-only rules (`raw_mean`, `row_z_mean`, `rank_mean`,
  `minmax_mean`, `softmax_mean`) on the **same saved OOF rows** with the same 0.5/0.5 weights, for both member pairs
  (official-only fallback and pool-MIX cap-40k candidate). It reuses the D041 identity guards, fits nothing, reads no
  pool row and scores no test row. Absent-class sentinels are masked to each rule's neutral value; both pairs contain
  zero sentinel cells.
- **Result.** Candidate: row-z **0.442675** is best; raw 0.420808 (−0.0219, CI [−0.0269, −0.0169]), rank 0.406028
  (−0.0367, CI [−0.0603, −0.0130]), min-max 0.440823 (−0.0019, CI [−0.0055, +0.0018]), softmax 0.401158 (−0.0415,
  CI [−0.0553, −0.0277]). Fallback: row-z 0.375971, raw 0.324114 (−0.0518), rank 0.361921 (−0.0140), min-max 0.375973
  (−0.0000, CI [−0.0096, +0.0096]), softmax 0.302988 (−0.0730).
- **Min-max is indistinguishable, not a win.** The script's `best_rule` picks it on the fallback by 2e-6 (rounding
  noise) with a CI spanning zero, and it is worse on the candidate. Recorded as a negative result; no adoption.
- **Compliance.** Softmax is reported only as a monotone-transform diagnostic — LinearSVC margins are not
  probabilities. Selecting a rule on this OOF is mildly optimistic, so the pre-registered bar is unchanged: beat
  row-z on s13 **and** the nested outer gate. Nothing clears it.
- **Recorded.** Nine tests in `tests/test_blend_rules.py` (hand-computed rank/min-max, softmax ordering and sum,
  neutral constant/sentinel rows, all-absent finiteness, exact agreement of the row-z arm with
  `src.ensemble.normalize_scores`, unknown-rule rejection, blend = weighted sum); full suite **251 passed**. Two
  registry rows (`*-blend_rules`), ADR **D042**, report §5.1 table + §7 cross-reference + banner + §11 command,
  README subsection. **No config, weight, artifact or packaged dataset changed** — the candidate and fallback are
  untouched and nothing was re-packaged.

**Task #47 / D043 (2026-11-17): plan §8 L208 — full-size cold-start inference is now MEASURED, not estimated.**
Report §9 previously claimed "seconds for ~4,000 test passages" with no measurement behind it, so the §11 acceptance
criterion "runtime, peak RAM and notebook compatibility are measured" was only partly met for the inference path.

- **New script.** `experiments/inference_benchmark.py` spawns a **fresh process** with the packaged `src` at
  `sys.path[0]`, `PYTHONPATH` cleared, `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE=1`, cwd = a scratch working dir, and
  asserts `src.__file__` resolves inside the package (so it cannot time the repo's `src` by accident). The worker
  replays notebook cells 1–4 verbatim, including the `crossed: true` rejection and every schema/id/label/null assert,
  timing each stage and sampling RSS at each boundary (Windows `peak_wset` = true peak).
- **Synthetic rows only.** `src.synthetic` at k≈2 passages/prompt (the shape `docs/strategy_review.md` infers),
  generated to an exact row count. **No fit, no competition row scored, no accuracy or LB claim.** Only `train.csv`
  word counts are read, as a length reference.
- **Measured.** 4,000 rows: **13.77 s** cold start (imports 1.20, bundle load 0.37, predict 11.14, write+validate
  0.07), peak working set **0.370 GB** (sampled RSS 0.220). 8,000 rows (2× stress): **28.24 s**, predict 25.21 s,
  peak **0.542 GB**. Predict scales linearly; headroom ≈2,350× / ≈1,150× the 9 h CPU budget. A repeat 4,000-row run
  took 15.58 s → honest range **~14–16 s**. All assertions passed at both sizes.
- **Conservative + bounded claims.** Fixture passages are longer than official (mean 190.1 vs 114.7 words = 1.66×,
  max 593 vs 274), so the timing over-estimates official-length work. It is a **local Windows/CPU** measurement,
  **not** a Kaggle-runtime benchmark; `peak_wset` is Windows-only. The synthetic prediction distribution (heavily
  one class) is a runtime artefact of fixture styles, explicitly **not** an accuracy/class-prior result.
- **Recorded.** Registry `20261117T030454-inference_benchmark-cold_start_4000_8000` (infer_seconds 13.7687,
  peak_ram_gb 0.369, no BA claimed), ADR **D043**, report §9 measured table + banner + §11 command, README
  subsection. Five tests in `tests/test_inference_benchmark.py` (exact size, official test schema, determinism, odd
  small size, non-positive rejection, worker-failure propagation so a broken package can never be reported as a
  successful timing). **No model, weight, config or package change** — nothing re-packaged.

**Task #48 (2026-11-17): plan §1 L84's missing `evaluate` CLI entry point now exists — the last uncovered harness requirement.**
- **Gap.** §1 lists `audit / train / evaluate / predict / package` as the required entry points; `evaluate` was the only
  one absent. Saved OOF could be scored only through `experiments/*` scripts, so the documented command surface did not
  match the code.
- **Implemented.** `src/cli.py` gains `cmd_evaluate` + `_variant_metrics` + `LENGTH_BAND_EDGES` (the D041 edges). It loads
  explicitly named members with `load_oof`, re-derives the expected fold assignment from the frozen splits and rejects any
  fingerprint or fold mismatch, then reports pooled BA, per-fold mean ± std with a 95% CI, macro F1, confusion matrix,
  per-class recall, within-provider errors and — with `--weights` — the row-z blend plus paired blend-minus-member CIs.
  `--length-bands` re-checks the local train frame against the split fingerprint before adding bands.
- **Verified on real OOF.** `make evaluate` reproduces the official-only fallback through a fourth independent code path:
  char **0.3193** (±0.0120), style **0.2904** (±0.0142), blend **0.3760** (±0.0026, CI [0.3723, 0.3796]),
  blend − char **+0.0567 [+0.0387, +0.0746]**, blend − style +0.0855 [+0.0672, +0.1039]; 2,699 rows, fingerprint
  `dc60875fd8b536cb`, `stratified_group-s42-n5`. Output: `artifacts/evaluate/official/metrics.json`.
- **No new numbers, no protocol change.** The command fits nothing (`fits_performed: 0`, `transductive: false`), reads no
  test row, and deliberately does not re-measure fit/infer seconds or peak RAM — those belong to the training run and stay
  in the registry. Hence no ADR and no registry row: this is harness plumbing, not learning behaviour.
- **Tests + docs.** Ten tests in `tests/test_cli_evaluate.py` (metric recomputation, blend equals an independent row-z
  argmax, fingerprint/fold rejection, missing member, four invalid-weight cases, length-band train-frame guard); suite
  **266** green. `Makefile` help + target, AGENTS.md command block, README (command list, reproducibility paragraph, layout
  line), report §11 command + note. Known limit: `--artifacts` takes one directory, so the cross-directory pool-MIX
  candidate is still scored by `experiments/blend_deberta.py` / `blend_rules.py`. Suite green (**266** then, **291**
  after #49–#53).

**Task #49 / D044 (2026-11-17): plan §7 L196 — class-specific intercepts measured under nested separation and REJECTED.**
- **Question.** Balanced accuracy is mean per-class recall, so a per-class intercept on the blend scores could in
  principle trade a little overall accuracy for recall on the weak classes. §7 asked for the measurement; it did not exist.
- **Protocol (strict, non-transductive).** `experiments/class_bias.py`: for each of the five frozen outer folds, cut 3
  inner grouped folds from **outer-train rows only** (`make_splits` + `validate_splits`), refit both members on inner-train
  (**30 refits per seed**), search an 18-vector by coordinate ascent on inner OOF (±1.5 grid, 13 points, 3 passes, ties
  prefer smaller |value|), then pick a shrinkage λ ∈ {0, 0.25, 0.5, 1.0} on the same inner OOF and score outer-valid **once**
  with the frozen bias. Three arms: no bias, analytic `−log(prior)` (centred), nested-tuned. Split seed/fingerprint, OOF
  fingerprint and `fold_assignments` are all verified before scoring; no test row read.
- **Measured.** s42: no_bias **0.375971**, prior 0.375600 (Δ −0.000370 [−0.001399, +0.000658]), tuned **0.366344**
  (Δ **−0.009655** [−0.024830, +0.005520]). s13: no_bias **0.358190**, prior identical (Δ exactly 0), tuned **0.363398**
  (Δ **+0.005172** [−0.014428, +0.024773]). Inner gains **+0.012152 / +0.011686** vs outer **−0.009655 / +0.005172** →
  optimism gaps **+0.021807 / +0.006514**. Shrinkage chose λ=1.0 on **9 of 10** folds (0.5 on s13 fold 4). Wall time
  295.77 s / 320.34 s.
- **Why it fails.** The intercepts fit the *inner* models' idiosyncratic class offsets; the outer models, fit on more data,
  do not share them. The analytic arm is a near no-op because official train is balanced (support 149/150, ratio 1.01), so
  there is no prior imbalance to correct. Shrinkage never saw enough inner evidence to prefer a smaller correction.
- **Ruling.** Adoption requires **every** seed to improve with a paired CI above zero; s42 is negative and neither CI
  excludes zero → **not adopted**. No config, weight, package or submission change; the blend keeps zero intercepts.
  Scope limit recorded: the subject is the official-only fallback, not the pool-MIX char member.
- **Recorded.** Registry rows `…-official-class_bias_nested-s42` / `-s13`, ADR **D044**, report **§7.1** + banner + §11
  command, README subsection. **12 tests** in `tests/test_class_bias.py` (reference arm equals the adopted row-z blend,
  balanced-prior offset ≈ 0, majority penalised, zero bias on a perfect classifier, correction of a synthetic systematic
  offset, determinism, shape rejection, inner coverage == train_idx, all-seeds-must-win verdict + 3 rejection cases).

**Task #53 / D045 (2026-11-17): plan §4 L139 — the missing style-only SVM arm is measured; logreg stays.**
- **Gap.** §4 names three classifier families for the style view. logreg (0.290398) and gbdt (0.263015) were measured;
  LinearSVC on style alone was not, so the classifier comparison was incomplete.
- **Run.** New `configs/style_svm.yaml` (single `style_svm` model: `features: []`, `style: true`, `classifier: linearsvc`,
  C=1.0) — deliberately separate from `configs/features.yaml` so the ~20-min `char_plus_style` ablation is not re-run.
  Same five registered folds, same fingerprint.
- **Measured.** style_svm pooled **0.284449** (folds 0.2815 0.2963 0.2907 0.2593 0.2944; mean 0.2845 ± 0.0152), macro F1
  **0.257414**, fit **26.573 s**, infer 2.066 s, peak **0.176 GB** — vs style_only logreg **0.290398**, F1 0.2792, fit 29.6 s,
  peak 0.20 GB. Paired SVM − logreg **−0.005951, 95% CI [−0.014493, +0.002590]** (includes zero). Within-provider errors
  **876 vs 873**, cross-provider **1055 vs 1042** (1931 vs 1915 total).
- **Ruling.** No evidence of a gain, point estimate negative, error profile no better → **keep logreg**. The SVM is cheaper,
  which is why the negative result is recorded rather than left as an empty cell. All three §4 families are now covered.
- **Recorded.** Registry `20261117T034251-official-style_svm-s42`, ADR **D045**, report §7 table row + interpretation bullet,
  README result/ablation rows. No blend member, weight or package change.

**Task #50 / D046 (2026-11-17): plan §4 L142 — per-feature style analysis added, with an identity proof.**
- **Gap.** §4 asks for per-feature analysis alongside per-class recall/confusion/length bands; the latter three existed
  (D041) but nothing said *which* of the style features carries the signal.
- **Identity first.** `experiments/style_features.py` refits `style_only` on each frozen training fold and **aborts** unless
  the refit reproduces the saved identity-verified OOF: measured max absolute drift **0.0** on all five folds, so the
  reported coefficients provably belong to the models that produced the recorded 0.290398. Feature names and coefficient
  shape `(18, 56)` are asserted identical across folds.
- **Measured.** **56** features (the docs' old "44" was stale and is corrected everywhere), `StandardScaler(with_mean=False)`
  fit inside each fold. Aggregates mean / mean-abs / fold-std / sign-agreement per class and feature. Strongest overall
  (class-averaged magnitude): `rate_dash` **1.3816** (sign agreement **1.00**), `rate_nonascii` 0.6182, `length_sentences`
  0.5819, `rate_colon` 0.4973, `rate_squote` 0.4954, `rate_comma` 0.4852, `mean_word_len` 0.4566. `rate_dash` is positive for
  every claude-*/glm-* label (claude-opus-5 **+1.961**, glm-5.3-flash +1.761) and strongly negative for gemini-3.6/3.7/3.8-flash
  (**−2.010 / −1.804 / −2.017**). Weakest style-only recalls: claude-fable-5 **0.074**, glm-5.3 **0.087** — both lean on the
  same two features as their better-recalled siblings, which is the within-provider error concentration of report §8.1.
- **Method note.** The overall ranking uses class-averaged **magnitude**: multinomial coefficients are centred across the 18
  classes, so the signed class-average cancels to ~0 and would rank every feature as unimportant. Signed per-class weights
  are kept in the second table.
- **Ruling.** **Associational only.** Correlated style columns share credit arbitrarily; no causal attribution is claimed and
  **no feature was added, removed or reweighted**. No new score, no config/weight/package change, no test row read.
- **Recorded.** Registry `…-official-style_feature_associations-s42`, ADR **D046**, report **§8.3** + banner + §11 command,
  README subsection; artifacts `artifacts/style_features/results.json` + `feature_table.md` (overall ranking + 18-row
  per-class table). **7 tests** in `tests/test_style_features.py`.

**Task #52 / D047 (2026-11-17): plan §2 L94 — cross-fold near-duplicates measured: ZERO.**
- **Gap.** The protocol asserted prompt-grouped separation but only ever checked *exact* text equality and prompt-id
  overlap. Near-duplicate text spanning a fold would inflate every CV number in the repository, and nobody had measured it.
- **Run.** `experiments/near_duplicates.py` examines **all 3,640,951 unordered pairs** of the 2,699 official rows (self-pairs
  excluded, each pair counted once via an upper-triangle mask) with the **same** vectorizer settings and 0.3 threshold as
  `src.pool.decontaminate_global`, so the internal and external audits are comparable.
- **Measured.** Exact duplicate texts **0**; prompts appearing in more than one fold **0**. cos > 0.3: **2,190** pairs,
  cross-fold **0**, same-prompt 2,190, cross-fold-different-prompt **0**. cos > 0.5: **34**, all same-prompt, cross-fold 0.
  cos > 0.7 and > 0.9: **0**. Max off-diagonal cosine **0.6223** (median 0.2985, p90 0.4149, p99 0.5278).
- **Interpretation.** The similarity that exists is the shared question vocabulary of the 18 answers to one prompt, and the
  grouped split keeps every such pair on one side of every fold. Group separation is now **measured, not assumed**.
- **Compliance.** Descriptive audit: the vectorizer is fit on the whole corpus but **no predictor is fit**, no score is
  reported and no validation row influences a training decision (`fits_performed: 0`, `predictors_fit: 0`,
  `test_rows_read: 0`). Stores ids/fold numbers/cosines only — **never passage text**. Similarity is a heuristic, not proof
  that every augmented variant was found.
- **Recorded.** Registry `…-official-near_duplicate_audit-s42`, ADR **D047**, report §4 bullet + banner, `docs/data_audit.md`
  (finding + ticked check), README subsection. **6 tests** in `tests/test_near_duplicates.py` (self-pairs excluded,
  same-prompt similarity is not cross-fold, the same similarity across folds *is* flagged, exact duplicates reach cosine 1
  as one unordered pair, counts monotone in threshold, top pairs sorted/unique/ordered).
- **Bug found and fixed during the run.** The first version counted *ordered* pairs, doubling every count (4,380 instead of
  2,190). Fixed with the upper-triangle mask, a `pair_counting` payload field, and the audit was **re-run** so the recorded
  numbers are the corrected ones.

**Task #54 / D048 (2026-11-17): plan §3 L118 — per-prompt-group performance measured; errors are NOT concentrated.**
- **Gap.** L118 asks for performance by class, model family, text length **and prompt group**. The first three existed
  (D041/D046/§8.1); per-prompt performance had never been measured — `src/audit.py` only counts rows and labels per prompt.
- **Run.** `experiments/prompt_groups.py` verifies the train fingerprint and every member's OOF fingerprint + fold
  assignment, then groups the 2,699 rows into their **150 prompts** (17–18 rows each). Five variants (both blends + three
  members). Because ~18 rows per prompt is noisy, each concentration statistic is compared against a **binomial null** that
  keeps the real row counts and draws errors at the observed overall rate (2,000 draws, seed 20261117). No fit, no pool row
  read, no test row.
- **Measured — negative concentration.** Worst-decile share of all errors: official blend **0.1336** (1,684 errors) vs null
  **0.1305 [0.1269, 0.1344]**; pool-MIX **0.1396** (1,504 errors) vs null **0.1357 [0.1315, 0.1404]**; char official 0.1274 vs
  0.1266; char pool 0.1301 vs 0.1331; style only 0.1238 vs 0.1244. **Every variant is inside its null interval.** Zero prompts
  at 0 accuracy, zero perfect prompts. Prompt accuracy mean 0.3761 ± 0.1209 (official, min 0.1111, max 0.6667) and
  0.4427 ± 0.1234 (pool, min 0.1667, max 0.7222).
- **Measured — no prompt-level attractor.** Distinct predicted labels per prompt **10.21** (range 4–15) official vs **11.18**
  (6–15) pool; modal share 0.226 / 0.205; max modal share 0.778 official vs 0.444 pool; **0** prompts collapsed to one label.
  Among wrong rows the modal wrong answer carries **0.287 / 0.288**, and **no** prompt sends all its errors to one label.
- **Measured — the pool gain is broad on a third independent unit.** Paired per-prompt delta **+0.066645**, std 0.105199,
  95% CI **[+0.049672, +0.083618]**, n=150: **96 improved / 28 worsened / 26 unchanged** (max gain +0.3333, max loss −0.2222).
  Rows (D037) and folds (D039) already agreed; prompts rule out "a few questions flipping".
- **Also.** Spearman ρ(prompt accuracy, prompt mean words) **+0.169** (p=0.0382) official, **+0.239** (p=0.0032) pool — the
  §8.2 length gradient is only partly prompt-level. Per-prompt accuracy mean reproduces pooled BA (0.3761 vs 0.3760;
  0.4427 vs 0.4427) since each prompt holds ~one row per class: an internal consistency check, not a new score.
- **Ruling.** Descriptive; **no model, config, weight or package change**. The negative concentration result argues against
  prompt-difficulty-targeted remedies (consistent with the H2 negative, D010) and in favour of per-passage representation work.
  Caveat recorded: the null assumes within-prompt independence while rows share a question, so the true null may be wider.
- **Recorded.** Registry `…-official-prompt_group_errors-s42`, ADR **D048**, report **§8.4** + banner + §11 command, README
  subsection, artifacts `artifacts/prompt_groups/{results.json,prompt_groups.md}` (ids/counts/scores only, no passage text).
  **28 tests** in `tests/test_prompt_groups.py`, including that a genuinely concentrated failure exceeds the null's p95 while
  a perfectly even split falls below it.

**Task #55 / D049 (2026-11-17): plan §4 L139 — the CatBoost style arm is measured and is NEGATIVE.**
- **Gap.** D045 closed L139 with three families (logreg / LinearSVC / sklearn HistGB), but D020 could not test a real
  gradient-boosting library: catboost, lightgbm and xgboost were **absent** then. `catboost 1.2.10` is installed now, so the
  strongest conventional tabular baseline for a 56-feature view was still unmeasured. lightgbm/xgboost remain absent.
- **Implementation.** `src/model.py` gains an **optional, lazily imported** `catboost` classifier and a
  `DENSE_CLASSIFIERS = ("gbdt", "catboost")` tuple that drives both the dense-input guard (style-only or SVD-reduced, never
  raw tf-idf) and `.toarray()`. `ModelError` if the package is missing, so the default install is unaffected.
  `configs/style_catboost.yaml` is separate from `configs/gbdt.yaml` so the SVD variants are not re-run.
  `allow_writing_files=False` → no `catboost_info` side-effect directory (tested).
- **Score path.** CatBoost has no `decision_function`, so `ModelBundle.decision_function` branches to
  `log(clip(predict_proba, 1e-12, 1.0))` in the 18-column output with the usual −1e9 absent-class sentinel. Log-probability
  keeps an unbounded additive scale for row-z blending (D042). This does **not** breach the "LinearSVC margins are not
  probabilities" rule: CatBoost MultiClass *is* a probability model, and no margin is ever called calibrated.
- **Measured.** Same frozen `stratified_group-s42-n5` folds; fingerprint, fold assignments and label vector verified identical
  to the identity-verified `style_only` OOF. Pooled **0.283003**, macro F1 **0.275370**, folds
  0.294764 / 0.303704 / 0.285185 / 0.266667 / 0.264815 (mean 0.283027 ± 0.015292), fit **114.967 s**, infer 1.308 s,
  peak **0.213 GB**. Four families now: logreg **0.290398**, linearsvc 0.284449, catboost 0.283003, HistGB 0.263015.
- **Paired.** catboost − logreg **−0.007382, 95% CI [−0.021211, +0.006448]**; catboost − svm −0.001430
  [−0.022784, +0.019923]. Both include 0, both point estimates negative. Within-provider errors **888** vs 873 logreg,
  cross **1047** vs 1042 (1935 vs 1915 total) — same failure mode, slightly worse. ~3.9× logreg's fit time for nothing.
- **No blend diversity either.** Row-z correlation between the two style members **0.8678**. Swap
  `0.5 z(char) + 0.5 z(style_catboost)` → **0.365240** (paired vs adopted **−0.010690 [−0.022630, +0.001251]**); equal thirds
  char + style_logreg + style_catboost → **0.368566** (**−0.007356 [−0.022003, +0.007290]**). A third correlated member
  dilutes the blend.
- **Ruling.** **Keep logreg.** No blend member, weight, submission config or package change. catboost stays **optional** —
  not added to `requirements.txt` (documented in a comment block instead), `pytest.importorskip` in the tests. Limit recorded:
  one seed, conventional defaults, no hyperparameter search — this rules out the default CatBoost arm, not every tuning.
- **Recorded.** Registry `20261117T041323-official-style_catboost-s42`, ADR **D049**, report **§7** (table row + two bullets)
  + banner + §11 command, README results/ablation/negative-results rows + a new family-comparison subsection + layout,
  artifact `artifacts/cv_style_catboost/`. **12 tests** in `tests/test_model_catboost.py` (dense registration, raw-tf-idf
  rejection, SVD acceptance, factory params incl. the no-balanced-weights case, 18-column order, log-proba monotonicity and
  argmax equality, −1e9 sentinel, determinism, save/load round-trip, no `catboost_info`). D020's "catboost is absent" claim is
  **annotated, not rewritten**; D045's "all three families" likewise carries a follow-up note.

### Task #56 / D050 — refresh the upload-ready package and fix its licence (2026-11-17)

- **Gap.** Two defects in `artifacts/kaggle_dataset`, found while reconciling the plan checklist (#51).
  (1) Its `src/` predated the live source in **four** files (`cli.py`, `figures.py`, `metrics.py`, `model.py`).
  (2) `cli package` hardcoded **CC0-1.0** in `dataset-metadata.json` even though the shipped char bundle is a
  derived adaptation of **CC BY-SA 4.0** text (official train + Open Pool v15) — contradicting the register's own
  ruling that "attribution + CC BY-SA must travel with any redistributed pool-derived weights". CC0 drops both
  the share-alike obligation and the credit, so uploading it would have misdeclared a public derived artifact.
- **Implementation.** `src.cli._package_licence(manifest_path)` derives the declaration from the manifest being
  shipped: non-synthetic → **CC-BY-SA-4.0** + a README "Licence and attribution" block (licence link, competition
  credit, and Open Pool v15 credit for exactly the members whose `fit_mode` contains `pool`, naming the
  `external_data_id`); `dataset: synthetic` → **CC0-1.0** with no competition attribution, since fixture text is
  locally generated. `CC-BY-SA-4.0` is the Kaggle API's own documented licence name, not a guess.
- **Refresh.** `python -m src.cli package --dataset official --out artifacts/kaggle_dataset --kaggle-user tiennamlee`.
  Verified byte-for-byte (sha256) against the previous package: **only** those 4 `src/` files + `README.md` +
  `dataset-metadata.json` changed. Both `models/*.joblib` are **byte-identical** (no refit), `manifest.json`
  unchanged, `weights.json` unchanged with the D040 provenance fix intact (`fit_modes`, `cv_score_of_record`,
  official-only CV fields absent). No file added or removed; still no encoder, no `inference.json`, no pool parquet.
- **Re-validation.** Cold-start benchmark re-run from the package's **own shipped `src`**
  (`artifacts/inference_benchmark_refresh/results.json`): 4,000 synthetic rows **9.40 s** / **0.3702 GB** peak
  working set (RSS 0.2188), 8,000 rows 16.88 s / 0.542 GB; `validation_passed: true`, returncode 0,
  `fits_performed: 0`, `competition_rows_scored: 0`. D043's 13.7687 s record is **preserved unchanged**; the new
  run is faster but the difference is **not attributed** to the refresh — machine noise dominates.
- **Ruling.** Compliance + freshness only. **No score, candidate, bundle, weight or blend rule changed.** The
  owner must upload with the CC BY-SA 4.0 declaration intact and must not relabel it CC0.
- **Recorded.** Registry `20261117T050105-official-package_refresh_cold_start_4000_8000`, ADR **D050**,
  report banner + §11 bullet, README package-licence paragraph, register admission item, submission checklist.
  **3 new tests** in `tests/test_package.py` (share-alike + pool credit; official-only credit without the pool;
  synthetic stays CC0).

### Task #57 / D051 — candidate ablation, runtime/RAM and CV-variation panels (2026-11-18)

- **Gap.** The last open item in this roadmap's own closing note: the candidate's confusion, recall and
  length-band figures existed (D041), but its **ablation, runtime/RAM and grouped-CV variation** panels were
  still the baseline's. Structural cause — `src.figures.make_report_figures` reads `metrics/*.json` from **one**
  training run, while the candidate's members were fit in **separate** runs (`char_tfidf_logreg` with the pool in
  `artifacts/pool_cv_cap40k/official`, `style_only` in `artifacts/cv_identity/official`). No single metrics
  directory describes the candidate.
- **Implementation.** The three panels became shared functions (`make_ablation_figure`,
  `make_runtime_ram_figure`, `make_cv_variation_figure`) that `make_report_figures` now composes, so baseline and
  candidate plot through identical code. New `experiments/candidate_figures.py` rebuilds five arms from the
  identity-verified OOF (fingerprint + fold-id + label equality against the frozen splits) and writes
  `artifacts/candidate_figures/results.json` + `candidate_panels.md`.
- **Guards, not assertions.** Every arm's pooled **and** per-fold BA is recomputed and compared to the value
  already recorded in the artifacts; a mismatch beyond `1e-6` aborts. All ten comparisons matched — a **fourth
  independent reproduction** of 0.375971 / 0.442675 after D039, D041 and D042. `fits_performed: 0`,
  `test_rows_read: 0`, no pool row read.
- **Measured.** `style_only` 0.290398 (41.42 s, 0.203 GB); `char_official` 0.319306 (127.35 s, 0.205 GB);
  `official_blend` 0.375971 (168.77 s, 0.205 GB); `char_pool_cap40k` 0.410445 (1599.62 s, RAM not recorded);
  `pool_mix_cap40k` 0.442675 (1641.03 s, RAM not recorded). A blend's fit seconds are the **sum** of its members'
  and its peak the **larger** member peak, because the fits are sequential, never co-resident.
- **New finding — the pool trades fold stability for mean accuracy.** Candidate fold std **0.016640** vs the
  fallback's **0.002637** (**6.31x**), range 0.412963–0.457407 vs 0.372222–0.379630. The per-fold gain is still
  positive in **all five** folds (+0.0708/+0.0815/+0.0574/+0.0389/+0.0852), so D037's all-folds-positive gate
  holds and the candidate is **not** demoted — but the wider spread is now visible instead of hidden behind the
  baseline's unusually tight folds, and it is a further reason to keep the s13 + nested gates.
- **Honesty rule.** The pool arm's originating run never recorded peak fit RSS. It is `null` in `results.json`,
  empty in the registry, and a **hollow "RAM n/r" marker** in the figure — never a zero-area bubble and never an
  estimate. Fit cost (9.72x the fallback) is **offline prep only**; the graded notebook fits nothing and its
  measured cold start is 9.40 s / 0.3702 GB (D050).
- **Ruling.** Reporting only. **No score, weight, config, bundle or packaged file changed**; nothing refit,
  nothing re-packaged.
- **Recorded.** Registry `20261118T065212-pool_mix_cap40k-candidate_panels`, ADR **D051**, report banner + §11
  bullet + §12 (caveat retired, three figures embedded), README subsection. **8 new tests** (3 in
  `tests/test_figures.py`, 5 in `tests/test_candidate_figures.py`). Figures: `figures/ablation_pool_mix_cap40k.png`,
  `figures/runtime_ram_pool_mix_cap40k.png`, `figures/cv_variation_pool_mix_cap40k.png`.

### Task #58 / D052 — Discussion sweep, the last agent-doable plan item (2026-11-18)

- **Gap.** Plan L31 was the only unticked item an agent could act on: the Discussion pages had never been
  swept, and `docs/competition_spec.md` §10 still listed hidden-test prompt multiplicity and GPU quota as
  unverified.
- **Done.** Six threads read in the in-app browser and recorded with per-claim URLs in **spec §11**
  (758516, 759102, 749447, 757855, 758998, 759103). **No organizer replied in any of them.**
- **The unknowns stay unknown.** The direct question *"Test set structure: passages per prompt?"* (758516)
  has **0 comments**. §10's k and GPU-quota bullets remain open; nothing was promoted from community
  consensus to verified, and no code may assume test-side grouping.
- **External corroboration of D017.** Another solo pipeline (MIT code, self-reported LB 0.389) reports
  **"Hungarian assignment within prompt groups gave +0.10 CV but +0.002 LB"** — the same collapse our
  k-subsampled stress test measured. The crossed lever stays withdrawn. One of their claims contradicts our
  stated rationale (random CV *pessimistic* 0.104 vs grouped 0.315, where protocol §1 argues random folds would
  be *optimistic*); recorded as **unreconciled**, neither number adopted.
- **Two new hypotheses queued, not adopted.** **H8** — ModernBERT-large backbone instead of DeBERTa-v3-large
  (two participants self-report ≈0.55 OOF/LB vs their DeBERTa 0.50; our DeBERTa measured 0.2482), so D027's
  encoder negative is **backbone-specific**. **H9** — hard typographic family vetoes, CPU-only, rules mined
  per fold with recorded support and chance level. Both are in Phases 2 and 4 above.
- **A calibration, not a prediction.** Their prompt-grouped ladder (TF-IDF+logreg 0.31 OOF/0.32 LB;
  +external data 0.46/0.40; encoder blend 0.625/0.616) plus our only measured pair (CV 0.3760 → LB 0.356)
  suggests a **−0.02 to −0.06 CV→LB offset**, so the candidate's 0.4427 CV is consistent with an LB near
  **0.38–0.42, not 0.44**. The revert rule is unchanged; only #38 can settle it.
- **Verified in our own code.** The adopted char member already preserves raw characters
  (`src/features_tfidf.py:47` `lowercase=False`; no unicode/punctuation normalisation anywhere in the feature
  builder or `src/data.py`), so "don't normalise punctuation" is satisfied — **no defect found**.
- **Ruling.** Evidence-gathering only. **No registry row** — a literature sweep is not an experiment, and
  community numbers must never enter our registry as results. No score, model, weight, config, bundle or
  package changed.
- **Recorded.** ADR **D052**, spec **§11** (+ §10 k bullet annotated), plan **L31 ticked** with k/quota still
  open, H8/H9 queued in Phases 2/4.

### Task #59 / D053 — H9 hard typographic family vetoes (2026-11-18)

- **Question.** D052's sweep found two threads reporting near-deterministic punctuation rules.
  Does a hard family mask beat the soft rates already in `src/features_style.py`?
- **Built.** `src/typography.py` (8 raw-character markers, per-fold rule mining with support +
  prompt-aware chance, never-mask-every-class guard), `experiments/typo_vetoes.py` (identity-checked
  OOF, paired per-fold CI, sensitivity grid, counterexample audit, oracle **with a no-veto control**),
  `tests/test_typography.py` (9 tests).
- **Answer: no.** +0.000741 on the candidate, CI [−0.001316, +0.002797]; +0.000370 on the fallback,
  CI [−0.000658, +0.001399]. Both include 0. Only fold 3 moves. Identical score across all five
  grid settings, so it is not a threshold artefact.
- **Why it cannot pay.** A veto only flips a row whose argmax was already inside the vetoed family.
  The oracle control makes this concrete: 0.597166 with vetoes vs 0.597537 without on the same rows.
- **Method correction kept.** Group-aware chance is now the default; the i.i.d. null cannot validate
  an absence rule when family base rates differ, because the family-specific rate is exactly what a
  zero-hit family cannot supply.
- **Nothing adopted.** No score, weight, config, bundle or package changed. Registry row
  `20261118T071941-pool_mix_cap40k-typo_vetoes`; ADR **D053**.

### Task #60 / D054 — H8 made runnable (2026-11-18)

- **Found.** H8 was queued but unrunnable: a Kaggle **script** kernel gets no CLI args, so
  `--model-name` never reached a pushed run, and the `transformers>=4.45` floor predates
  ModernBERT (4.48).
- **Fixed.** Module-level `BACKBONE` + `BACKBONES` presets; `build_arg_parser()` and a pure
  `resolve_model_name()` extracted so the no-arg path is tested, not just documented; backbone
  recorded in the run config and inference manifest; floor raised. `predict.py` needed no change.
- **Held constant.** 384 / 4 / 1e-5 for both presets — one variable per probe, and no unmeasured
  memory change against the T4 ceiling. A test pins it.
- **No metric, no registry row** (enabling infra, as with D038/D052). Six tests; nothing adopted.

### Task #61 / D054 — external-data register gap closed (2026-11-18)

- **Found while verifying the H8 cards.** `docs/external_data_register.md` had only **two** rows
  (`minilm-l6-v2`, `open-pool-v15`), yet **two DeBERTa checkpoints had actually been fine-tuned**
  (D023/D027) — and plan §6 was ticked claiming the register covered every source. A ticked box
  backed by an incomplete register is the failure mode this repo's evidence rules exist to prevent.
- **Verified, not guessed.** Model cards read 2026-11-18: DeBERTa-v3-**base** MIT, 12 layers,
  86M backbone + 98M embedding; DeBERTa-v3-**large** MIT, 24 layers, 304M + 131M; ModernBERT-large
  Apache-2.0, 28 layers, 395M, `transformers>=4.48`. Measured GPU peaks taken from the returned
  `benchmark.json` files (base **6.463 GB**, large **10.175 GB**), not from memory.
- **Added three rows.** base + large marked **used, not adopted**, with the identity-aligned
  official-only **0.191501** and an explicit warning that the **0.4442** blend used the D026-disabled
  transductive screen; `modernbert-large` marked **planned, never downloaded, no metric**.
- **Honest about what we do not have.** No encoder checkpoint is retained locally (only OOF came
  back from Kaggle), so no sha256 exists for them and their `size` column is card **parameter
  counts**, not file bytes. The checksum section now says so.
- **H8 counter-evidence recorded.** The ModernBERT card's own GLUE table puts it at **90.4** vs
  DeBERTaV3-large **91.4** ("second only to Deberta-v3-large"), so H8 is a **diversity** bet, not a
  stronger-encoder claim. Also: upstream Flash Attention 2 supports Ampere/Ada/Hopper only — the
  Kaggle **T4 is Turing**, so do not `pip install flash-attn` there.
- **Guard added.** `test_every_preset_checkpoint_is_registered_with_a_licence` fails if a future
  backbone is added to `BACKBONES` without a register row, so this gap cannot silently recur.
- **Also corrected.** Plan §6 evidence ("both rows" → five) and spec §9's pretrained-encoder ruling
  (now covers fine-tuned encoders and training-time internet-on fetch vs offline inference).
- **Second defect, same audit.** Every registry `artifact_path` was checked for existence: the two
  base-DeBERTa rows pointed at `deberta_v3_base_official.npz` / `deberta_v3_base_pool.npz`, but the
  kernel writes generic `official_only.npz` / `official_pool.npz`. The scores were re-verified against
  that run's `results.json` (0.2241088… / 0.3049043…, matching the recorded 0.224109 / 0.304904)
  **before** correcting the two path cells, so no number changed. All **121** rows now resolve, and
  every `artifacts/…` / `src/…` pointer cited in the plan, protocol, spec, report and README resolves.
- **No metric, no registry row** — provenance repair. No score, weight, config, bundle or package changed.

### Task #63 / D055 — larger ensembles: no cheap third member helps the pool candidate (2026-11-18)

- **The untested priority.** Mandate priority 5 is "larger ensembles". D049 had tried equal thirds, but
  against the **official-only** char member (0.368566, −0.007356). Nobody had tried a third member against
  the **pool-trained** candidate, whose char member is much stronger (0.4104 vs 0.3193) — a different
  trade-off, so the old negative did not transfer.
- **Provenance first, not convenience.** The 2026-11-14 `artifacts/cv/official/oof/*.npz` files predate D024
  (no fingerprint, no fold ids) and are **refused** by the screen. Re-ran on the frozen s42 folds: new
  `configs/blend_members.yaml` → `word_tfidf_linearsvc` **0.269652**, `char_wb_tfidf_linearsvc` **0.296336**;
  `configs/frozen_encoder.yaml` offline (`HF_HUB_OFFLINE=1`, no download) → `minilm_embedding_style_logreg`
  **0.309294** and `minilm_embedding_logreg` **0.127793**, both reproducing D013 exactly.
- **Measured, pre-specified weights only.** `experiments/blend_members.py`: equal thirds and 0.4/0.4/0.2
  (keeps the adopted char:style ratio), paired five-fold CIs vs base **0.442675**, plus row-z correlation.
  **All six arms lose in both recipes** — best keep-ratio arm `word_tfidf_linearsvc` 0.435240 (−0.007471
  [−0.022787, +0.007845]); worst `minilm_embedding_logreg` 0.361904 at thirds (−0.080792). All six thirds CIs
  and four of six keep-ratio CIs exclude 0; no arm is non-negative on all five folds.
- **The mechanism, stated.** Diversity without strength does not pay: the most diverse arm (embedding-only,
  corr **0.343**) is the weakest (0.1278) and loses most; the least diverse (embedding+style, corr **0.900**)
  is redundant. A third member is only worth a gate if it is *strong* — pool-trained or a GPU encoder (D038/H8).
- **Descriptive, never a gate.** Choosing an arm off these curves is selection on the reporting OOF (D026).
  No arm has a positive point estimate, so nothing is promoted to an outer-fold gate.
- **Honest gap.** `gbdt_style`, `gbdt_multiview_svd128`, `char_svd256_linearsvc`, `style_svd64_nuisance8`,
  `word_only`, `word_plus_char`, `char_tfidf_linearsvc` keep only pre-D024 OOF and were **not** re-run; their
  solo scores (0.24–0.31) sit in the same band, but they are unmeasured as third members.
- **Recorded.** Five registry rows (four members + `blend_third_member_screen`), six tests in
  `tests/test_blend_members.py`. No blend weight, member, config, submission or package change.
- **Extended before closing (same day).** A third pre-specified recipe — **swap** (0.5 char + 0.5 candidate,
  style dropped), which asks whether the candidate is a better *partner* than style — and a seventh arm, the
  **pool-trained** word member (#64/H10), were added; the test file grew to **eight**. The extension exposed a
  real defect: the official-only and pool-trained runs export the same stem `word_tfidf_linearsvc.npz`, and
  `members[name] = scores` silently overwrote one, so the first seven-arm run printed six rows and mislabelled
  its best arm. `load_members` now qualifies colliding stems by run directory
  (`pool_cv_word40k/word_tfidf_linearsvc`), a regression test covers it, and the buggy run's registry row
  (`20261118T083510`) is marked **superseded** by the corrected re-run (`20261118T084438`) — the numbers were
  the pool arm's all along and re-measure identically. The six-arm negative above is unchanged.

### Task #64 / D056 — H10: give the third member the same pool advantage (2026-11-18)

- **The hypothesis D055 named.** "A third member is only worth a gate if it is *strong*." So the word view was
  trained on the **same** non-transductive global pool (D028 screen, cap 40k/fold) as the char member — no new
  data, no new protocol, just the same advantage extended to a second view.
- **The lift replicates.** s42 0.269652 → **0.354877** (paired +0.0852 [+0.0567, +0.1137], 246.4 s);
  s13 0.255938 → **0.346339** (paired +0.0904 [+0.0566, +0.1242], 282.8 s). Both screen 226,191 → 226,049
  (0 exact + 142 near) and cap to 40,001/fold. The s13 official-only word/char_wb arms were re-run first so the
  comparison is paired on the same folds (`artifacts/cv_word_s13`).
- **First positive addition signal, on both seeds.** Keep-ratio 0.4/0.4/0.2 with the pool word member:
  s42 **0.450082** (+0.007407 [−0.012572, +0.027387]) vs base 0.442675; s13 **0.440057**
  (+0.009642 [−0.009867, +0.029152]) vs base 0.430435. Up on **4 of 5 folds on both seeds**, and the losing
  fold differs (s42 fold 4, s13 fold 0), so it is not one lucky group. Equal thirds is flat at s42 (−0.000026)
  and +0.0022 at s13. Every other arm still loses.
- **Swap is negative for this member** (−0.0363 s42 / −0.0259 s13): style stays the better partner; the word
  view earns its place only as a small addition. The s42 emb+style **swap** (+0.004444 [−0.007570, +0.016459])
  is the other positive point estimate but has **no s13 replication** (no s13 embedding OOF) and is not gated.
- **Not adopted.** Both CIs include 0 (five-fold paired std ≈0.016 vs an effect ≈0.008), and choosing the best
  of 7 arms × 3 recipes off the reporting OOF is D026 selection. A replicated descriptive sign is not a gate.
- **The gate is built and running.** `experiments/nested_recipes.py` + `configs/nested_recipes.yaml`:
  outer 3/s121, inner 3/s71, recipe set **identical** to the screen's {base, thirds, keep_ratio} (a test asserts
  the equality), ties keep the base, pool screened against outer-train only, and only the two pool members ever
  see pool rows (a test asserts exact fit sizes). Its base recipe shares #62's outer folds, so it must reproduce
  #62's fixed-0.5 outer scores — a free cross-check of two implementations. Adoption needs the paired outer CI
  to exclude 0; otherwise the two-member 0.5/0.5 candidate stands.
- **Recorded.** Eight new tests (`tests/test_nested_recipes.py`), registry rows for both s13 arms, the s13
  pool-word run and both screens; the buggy s42 screen row is marked **superseded**. No weight, member, config,
  submission or package change.

**Suite after #49–#64: 381 passed, 2 warnings** (the two warnings are the pre-existing benign `normalize_scores`
empty-slice notices on an all-sentinel row). Registry **135 rows** (the Discussion sweep, the H8 enablement and the
register repair add none; H9 adds one; #63 adds five; #64 adds the s13 word/char_wb arms, the s13 pool-word run,
both screens and a superseded marker, and the earlier fallback re-verification job adds two); ADRs **D001–D056**.
Nothing committed.

**Correction to the previous closing claim.** The #61 block ended with "no agent-doable lever remains". That was
**wrong**, and two levers found immediately afterwards prove it: the blend **weight** had never been swept (#62 —
0.5/0.5 was inherited from D009's official-only blend and carried into the pool candidate unchanged), and
**larger ensembles** had never been tested against the *pool-trained* candidate (#63 — D049's thirds trial used the
official-only char member). The lesson is procedural: a "nothing left" claim is only valid after re-auditing every
mandate priority against the **current candidate**, not against the baseline the priority was first tested on.
#63 is closed negative (D055). #64 (H10) then tested the follow-up D055 itself named — give the third member the
*same* pool advantage so it is strong enough to diversify — and found the first positive addition signal, replicated
on both seeds (D056); it is **not adopted** and is under an honest nested recipe gate. #62's weight gate runs on the
same outer folds, so the two gates cross-check each other's base recipe.

**Remaining owner-only gates are unchanged:** #38 upload + run on the full hidden test + submit + report the
public LB, and the GPU encoder runs (D038 DeBERTa-pool, plus the queued **H8 ModernBERT-large** probe).
**H9 is closed** (D053, measured and not adopted) and **H8 is runnable** (D054). No agent-side submission is
permitted (AGENTS.md), and no leaderboard number exists for the candidate. The last unticked plan item
(§6 same-prompt contamination) is **upstream-blocked, not compute-blocked**: `train.csv` has no prompt text
(verified 2026-11-18), so no local crosswalk is possible at any threshold. **All candidate figure panels now
exist** — confusion/recall and length bands (D041) plus ablation, runtime/RAM and CV variation (D051,
`experiments/candidate_figures.py`), so report §12's "the candidate's panels are the baseline's" caveat is retired.
