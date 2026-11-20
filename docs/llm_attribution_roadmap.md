# Brainstorm → Plan: 0.356 → 0.7 on Kaggle Name That LLM

**Status:** brainstorm → concrete plan (rev 1). **Evidence:** `Kaggle LLM Attribution
Strategy.md`, `docs/improvement_plan.md` (rev 2), `docs/strategy_review.md`,
`docs/competition_spec.md`, `experiments/registry.csv` (official rows only), `src/*`.
**Anchor numbers (all from this repo, official training data):** pooled-OOF
balanced accuracy argmax blend = **0.3760** (cv_s42, 2699 rows, folds 0.3760±0.0026);
examined holdout seed-7 = **0.3956** (450 rows); public LB **0.356**. Honest k≈2
stress (inference-only, paired diffs) shows Hungarian gain collapses from +0.093
at k=18 to ~0 at k=2.

---

## 0. What this plan does and does not claim

- **Claims only what this repo measured** (official-train grouped CV, honest
  k-subsampled stress, holdout). Everything else is labelled *hypothesis, community
  self-reported, needs local validation*.
- The strategy doc's headline claim (hidden test k≈2) is **high-confidence but
  unverified**; we act on it because the evidence is multiply corroborated and
  because k-robust post-processing is safe either way.
- This is a **base-model capability plan**. The repo's own stress test (D018)
  proves post-processing on the 0.376 blend buys ~0; +0.01–0.03 of calibration only
  pays off on top of a strong ensemble.

---

## 1. Diagnosis (grounded in this repo, no guesswork)

| Finding | Evidence in this repo |
|---|---|
| Safe baseline is weak | pooled OOF 0.3760 (cv_s42); k=2 honest CV 0.3607; holdout 0.3956. Public LB 0.356. |
| Crossed/hungarian design is a k=18 artifact | stress_k: at k=2, recovered-Hungarian paired gain +0.0068 [CI +0.0032,+0.0104] (cv_s42) / +0.0029 [−0.0059,+0.0117] (holdout, CI spans 0). Withdrawn by D017. |
| Post-processing on a weak base is ~neutral | Sinkhorn best at k=2 ≈ argmax (0.3611 vs 0.3607 cv_s42). |
| Within-lab siblings are the real enemy | repo within-provider error share 53.3% (vs community ~78%). |
| Classical sparse features plateau | char 0.3167, style 0.2904, char+style 0.3760; GBDT-style 0.263, char+SVD 0.254 — all < argmax blend. |
| External pool exists and is clean | open-pool-v15: 226,172 clean rows, all 18 labels exact, 0 exact dups, `max_cos_train` ≤ 0.30. |

**Bottom line:** +0.34 to the leader is a *capability* gap (fine-tuned encoders +
clean external data + multi-backbone readouts), not a post-processing gap.

---

## 2. Realistic ladder (community tiers, bracketed by our 0.356 and the 0.771)

| Tier | Method | Reported public BA | Status |
|---|---|---|---|
| 1 | TF-IDF / classical (us) | 0.33–0.42 | **0.356 done** |
| 2 | GBDT on stylometry + SVD | 0.39–0.42 | weak standalone here (0.263/0.254) — skip unless validated blend gain |
| 3 | DeBERTa-v3 sequence head (official-only) | 0.39–0.42 | **Phase 1b target** |
| 4 | DeBERTa-v3 + clean external pool | ~0.576 unseen-prompt CV → ~0.66 LB | **Phase 1b core lever** |
| 5 | Qwen3-8B/14B LoRA multi-depth readouts | 0.55–0.585 | **Phase 2** |
| 6 | multi-backbone LoRA stack + pool + calibration | 0.66–0.69 | **Phase 2 combined** |
| 7 | source-conditioned likelihood / GLM token features | 0.69–0.77 | **Phase 3 optional** |

Community figures in rows 4/5/6/7 are **self-reported, not independently audited**;
we validate locally via honest grouped CV + held-out prompts before any adoption.

---

## 3. Phased plan to 0.7 (ordered by impact ÷ cost, single-account CPU-first + optional limited GPU)

### Phase 0 — honest validation + k-robust post-processing  [DONE 2026-11-14]

- Implemented `src/calibration.py` (damped Sinkhorn-Knopp, k-robust) + 9 tests;
  `experiments/stress_k.py` + `metrics.paired_difference`/`mean_ci` + 5 tests.
- **Established:** primary metric = full prompt-grouped OOF BA (**0.3760**); k=1/2/3/18
  subsampling is an inference-only *stress test* (paired diffs + 95% CI over ≥10 seeds),
  never a selection score; transductive methods stay out of primary CV.
- Safe fallback submission remains the argmax blend until a stronger base is validated.
- *If you only do one thing: do not re-measure the crossed/candidate; it is withdrawn.*

### Phase 1 — clean external pool + a fine-tuned encoder  (highest ROI, CPU-feasible)

**1a. External-pool decontamination — DONE (D019).**
- `crtproduction/name-that-llm-open-pool` v15: 226,191 rows → clean **226,172**
  (per-fold word-TFIDF cos>0.30 removal of 19 rows union). Full audit in
  `artifacts/pool_audit/audit.json` + `clean_pool_ids.json`.
- Caveats: `match_weight`/`style_score` currently fitted on all train (refit per-fold
  or drop before CV); `flag_kb_proxy_after_ruling=20,227` (keep default, optional strict);
  CC BY-SA share-alike.

**1b. DeBERTa-v3 sequence head — the core Phase-1 lever.**
- Self-contained script `kaggle/deberta/deberta_train.py`; fine-tune `deberta-v3-base`
  (18-class) on the **same frozen grouped folds**, official-only vs official+clean-pool,
  per-fold decontamination, throughput benchmark against the 9h budget, stratified pool
  cap (default 40k).
- **Start with `base`**, not `large` — the strategy's 8B–27B stacks do not fit one
  Kaggle GPU under our single-account posture; community figures are support, not
  verified evidence here.
- Record: official-only vs official+pool on honest k=2 CV + per-class/within-provider
  recall + paired-diff CI + alternative fold seed. This decides Phase-1 adoption.
- Community claim to verify locally: ~0.39 official-only → ~0.576 with clean pool
  (deberta_v3_large). Our own run decides whether to push toward 0.66.

**1c. Multi-view TF-IDF + calibrated LinearSVC (CPU, DONE-ready).**
- char_wb 3–5, subword-boundary, punctuation-cadence views; blend with 1b/1c.
- CPU-feasible; complements the 0.376 argmax blend without GPU.

**1d. Pool-training batch.**
- `experiments/pool_train.py` (official-only vs official+pool, per-fold decon,
  paired-diff CI, registry rows) — referenced for every model in 1b/1d.

### Phase 2 — multi-backbone cached-embedding readouts (heavier single-GPU)

- Cache multi-layer hidden states from **Qwen3-8B/14B-base** (depths [2,4,6,8,12,16,20,24]),
  LayerNorm-standardised, content-token pooled; fit ridge/logreg heads on official +
  clean pool. Expected standalones ~0.55–0.585 (self-reported).
- **Centred-twin / shared-prompt readouts only on train-side groups**; never assume
  test-time groups at k≈2.
- Combine Qwen3-Embedding / Llama-3.1-8B-base / WeMM-Embedding experts for family
  diversity → multi-backbone stack ~0.66–0.69 (self-reported).

### Phase 3 — ATLAS-style source-conditioned likelihood  (optional, cross-family)

- `P(text|model)` with per-source LoRA + clipped token likelihoods. Apply only to
  contested rows (top-2 margin < ~0.32) at low weight (~0.20).
- **Caveat:** ATLAS is strongest cross-family; our errors are intra-lab siblings,
  so expect modest gains. Verify on honest CV before adopting. Defer if budget-limited.

### Phase 4 — k-robust post-processing *on top of the strong ensemble*

- Sweep Sinkhorn strength 0.3–0.6, target_mix 0.35 on honest k≈2 CV of the strong
  ensemble; adopt only on validated gain (no best-of grid).
- Typography guardrails as soft penalties (verify provider-invariant tokens on train).
- Gentle sibling residual heads (max_alpha ≤ 0.08) for the dominant sibling confusions
  (gemini-flash, glm-5.3↔flash, gpt-5.6 trio, claude-opus versions) while preserving
  cross-family probability mass.

### Phase 5 — sibling discrimination (throughout)

- Provider-conditional / version-aware modelling for gemini-flash, glm-5.3↔flash,
  gpt-5.6 siblings, claude-opus versions.
- Length-band analysis (short passages likely a weak spot; check confusion by length).

---

## 4. Validation discipline (unchanged, frozen)

- **Primary comparison** = full prompt-grouped pooled OOF BA (per-fold mean ± std).
- **Stress test** = inference-only paired differences (method − argmax) with mean ± std
  and 95% CI over ≥10 subsample seeds at fixed pre-registered k. Never a best-of grid.
- **Any new base model must validate on** the examined holdout + ≥1 alternative fold
  seed, **plus** the reserved fresh seed-2027 holdout at final selection.
- **Registry + ADR** for every change. LB is confirmation only (5 submits/day, 2 final).
- No new external data without registration + decontamination + grouped-fold validation.

---

## 5. What the plan gets us — expected progression (anchored to this repo)

| Stage | Measure | Expected k≈2 CV (hypothesis) | Notes |
|---|---|---|---|
| Now | argmax blend (0.3760 k=18 OOF) | 0.3607 | safe baseline |
| +1a | clean pool assembled, per-fold rules fixed | 0.361–0.365 | hygiene only, no score |
| +1b | DeBERTa-v3-base, honest k=2 CV, official+pool | +0.08–0.15 → ~0.44–0.51 | **biggest single step** |
| +1bcc | + multi-view TF-IDF/SVC | +0.01–0.03 | CPU complement |
| +2 | multi-backbone cached embeddings stack | +0.08–0.14 → ~0.55–0.65 | community self-reported; validate |
| +4cc | Sinkhorn + typography + gentle siblings | +0.01–0.03 | pays off on strong ensemble |
| **≈0.70** | 0.66–0.69 base stack + post-processing | **0.68–0.74 private** | private can add +0.03–0.06; top 1% zone |

- **0.70 is achievable but not guaranteed.** The documented route reaches 0.68–0.69
  public; private is where +0.03–0.06 typically appears, and where sibling
  discrimination (Phase 5) is the final differentiator. The official top is 0.771.

---

## 6. Verified-vs-hypothesized quick reference

**Verified (this repo / official):**
- Metric = balanced accuracy, 18 labels, official order.
- train.csv = 2,699 rows / 150 prompts; test.csv = id+text only.
- Grouped K-fold is mandatory; k=18 OOF is the primary comparison.
- Argmax blend = 0.3760 pooled OOF; public LB 0.356.
- Hungarian crosses collapse from +0.093 (k=18) to ~0 (k=2) — crossed candidate withdrawn.
- External pool v15 = 226,172 clean rows.

**Hypothesis (community self-reported, need local validation):**
- DeBERTa + clean pool → ~0.576 unseen-prompt CV / ~0.66 LB.
- Qwen3-8B/14B multi-depth readouts → 0.55–0.585.
- Multi-backbone stack → 0.66–0.69.
- Sinkhorn + typography + gentle siblings add +0.01–0.03 on a strong base.
- 0.70 reachable after Phase 1–2 + Phase 4/5.

---

## 7. Immediate next action (owner decision)

**Phase 1b GPU notebook**: prepare `kaggle/deberta/deberta_train.py` (offline-attached,
pinned, 9h budget), validate official-only vs official+clean-pool on honest grouped
CV, and hand the owner a single-account GPU notebook. The argmax blend stays the
safe fallback submission until 1b beats 0.3760 on the same folds.

---

*Built from the repo's own measured numbers; community figures are hypotheses until
validated here. No submission or account action is implied.*
