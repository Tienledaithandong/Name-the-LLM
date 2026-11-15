# Improvement plan — 0.356 → 0.7 (revised after strategy research)

> **Status (2026-11-14, revision 2):** active. Public LB **0.356** (argmax blend).
> Official leaderboard top **0.771** (5,851 teams; gold ≈ top 21, silver ≈ top 50;
> public 40% / private 60%). Evidence base: `docs/strategy_review.md`.
>
> **Major correction (ADR D017):** the hidden test has **~1–3 passages per prompt
> (k≈2)**, not the train set's k=18. Our crossed Hungarian candidate (D016) was
> validated only at k=18. **Honest k-subsampled CV (Phase 0 below) shows the
> Hungarian gain collapses from +0.10 at k=18 to ~0 (or negative) at k=2.** The
> crossed candidate is therefore **withdrawn as a submission default**. The ~0.34
> gap to the leader is a **base-model capability gap**, not a post-processing gap.

## 1. Diagnosis (corrected)

- **0.356 LB ≈ 0.36–0.39 honest k≈2 CV.** The argmax blend generalises about as
  well as honest CV predicts. The gap to 0.771 is **capability**: we score each
  passage independently with char n-grams + 44 style features; top solutions use
  fine-tuned encoders, external labelled pools, and multi-backbone readouts.
- **The crossed-design lever was overstated.** Train is a full factorial (k=18),
  but the hidden test is k≈2 (community-corroborated; locally confirmed by the
  Hungarian-gain collapse). One-per-model assignment exploits a structure the
  test does not have. At k≈2 it adds nothing and can overwrite correct predictions.
- **Post-processing is a small lever (~+0.01–0.03), not the main one.** Damped
  Sinkhorn calibration is k-robust and safe, but on our weak 0.376 blend it is
  ~neutral; it pays off only on top of a strong ensemble (the community applies it
  at 0.66+). Typography guardrails and gentle sibling residuals are similarly
  incremental.
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
validating each on honest k≈2 CV before submitting.

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
post-processing. All steps validate on honest k≈2 CV + holdout + alternative seed.

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
    array verified identical to local cv_s42): **running**. Memory-safe and fold-aligned
    by construction. Adopt only if it beats 0.376 on CV + the reserved holdout, or adds a
    validated fold-aligned blend gain (`experiments/blend_deberta.py`, guard accepts
    `*-embedded`/`*-shipped`).

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
  xgboost are absent) added to `src/model.py` (+4 tests, dense-input guard) and
  `configs/gbdt.yaml`. Official-only grouped OOF: **style-only 0.263**,
  **char+SVD200 0.254**, **multi-view SVD128 0.243** — all **below the 0.376 argmax
  blend**; SVD compression discards the sparse signal LinearSVC exploits.
  `experiments/pool_train.py` runs the official-only vs official+pool comparison
  (per-fold decon, stratified pool cap, paired-diff CI, registry rows). GBDT is a
  weak standalone here; adopt only if it adds a **validated** blend gain (D007).
- **1d — multi-view TF-IDF + calibrated LinearSVC:** char_wb 3–5, subword-boundary
  and punctuation-cadence views; blend with 1b/1c. CPU-feasible.

### Phase 2 — multi-backbone cached-embedding readouts (heavier GPU)

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

### Phase 4 — k-robust post-processing ON TOP of the strong ensemble

- Damped Sinkhorn (`src/calibration.py`): sweep strength 0.3–0.6, target_mix 0.35
  on honest k≈2 CV of the strong ensemble; adopt only on validated gain.
- Typography guardrails: verify provider-invariant tokens on **train** first
  (e.g. curly quotes / em-dash patterns); apply as soft penalties, not blind masks.
- Gentle sibling residual heads (max_alpha ≤ 0.08) targeting the dominant sibling
  confusions; preserve cross-family probability mass.

### Phase 5 — sibling discrimination (throughout)

- Hierarchical provider→version features; provider-conditional models for
  gemini-flash, glm-5.3↔flash, gpt-5.6 siblings, claude-opus versions.
- Length-band analysis (short passages are a likely weak spot).

## 4. Validation discipline (updated)

- **Primary metric (independent-passage models) = full prompt-grouped OOF BA**
  (pooled over all rows, plus per-fold mean ± std). This is the main comparison for
  the argmax blend, DeBERTa, GBDT, etc. **k=1,2,3,18 subsampling is a stress test**
  for batch/group-dependent methods only: report paired differences (method −
  argmax) with mean ± std and a 95% CI over ≥10 subsample seeds, on fixed
  pre-registered settings (never a best-of grid). Confirm any candidate on the
  examined holdout + ≥1 alternative fold seed; reserve the fresh seed-2027 holdout
  for final selection. k=18 grouped CV is a training-side fit diagnostic, not an LB
  proxy (community r≈0.13 vs r≈0.91 for k≈2; locally the Hungarian-gain collapse
  confirms the k sensitivity).
- Every change gets a registry row + ADR. Never tune on the LB. LB is confirmation
  only (5 submits/day, 2 final). Owner-only account actions.
- Transductive methods stay label-free and are never applied inside reported
  grouped-CV *training* scores. Post-processing is evaluated only on the
  k-subsampled **stress test** (paired diffs + CI), never as a primary score.

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

**Phase 1a/1b:** prepare the external-pool decontamination + DeBERTa fine-tuning
pipeline (code + offline-attach packaging), validate on honest k≈2 CV, and hand the
owner a single-account GPU notebook. The argmax blend remains the safe fallback
submission until a stronger base model is validated. Do **not** submit the crossed
Hungarian candidate.