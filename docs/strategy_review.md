# Strategy review — verifying "Kaggle LLM Attribution Strategy.md"

**Date:** 2026-11-14
**Scope:** research the attached strategy document carefully, separate verified
fact from community claim from fabrication/risk, and reconcile it with our
current pipeline. This document is the evidence base for the revised
`docs/improvement_plan.md` and ADR D017.

> **Headline finding:** the strategy's most consequential claim — that the hidden
> test has **~1–3 passages per prompt (k≈2), not 18** — is corroborated by several
> independent community sources and by a forensic log of real submissions. Our
> just-built **crossed Hungarian candidate (D016) assumes k=18** and is therefore
> **at risk of underperforming on the real test**. We must not submit it as-is.
> The robust fix (damped Sinkhorn calibration) is safe at *any* k, so the pivot is
> justified even without official confirmation of k.

---

## 1. What is VERIFIED (read directly from primary sources)

### 1.1 Official leaderboard (read from kaggle.com, 2026-11-14)

| rank | team | public BA | entries |
|---|---|---|---|
| 1 | NguyenThanhNhan | **0.771** | 15 |
| 2 | James Liebel | 0.765 | 38 |
| 3 | Meni David Gottesman | 0.765 | 108 |
| 4 | You're Absolutely Right! | 0.756 | 79 |
| 5 | Team Aillis | 0.756 | 139 |
| ~21 | (gold cutoff) | 0.737 | — |
| ~50 | (silver cutoff) | 0.717 | — |

- **5,851 teams.** Public LB ≈ **40%** of test; private ≈ **60%** (stated on the
  leaderboard page). "2 months to go" matches the verified final deadline
  2027-01-13.
- Confirms the owner's "top 1 is 0.7". The strategy doc's "0.769 top" is slightly
  stale; the live top is **0.771**. Our **0.356** is far below the medal zone.
- Source: <https://www.kaggle.com/competitions/name-that-llm/leaderboard>

### 1.2 ATLAS paper is real (arXiv, read in full)

- **"Who's Behind the Text? Robust Model Authorship Attribution"**, Ko, Lee,
  Song, Kim, Shin, Kim (KAIST), arXiv **2610.19583**, 14 Oct 2026.
  <https://arxiv.org/abs/2610.19583> · code <https://github.com/komseok0109/ATLAS>
- Method exactly as the strategy describes: source-conditioned likelihood
  `P(text | model)` instead of `P(model | text)`; frozen backbone + per-source
  prefix `e_k` + per-source LoRA up-projection `B_k` + **shared** down-projection
  `A`; **clipped token-level relative likelihoods** so no single token dominates.
- Results (MAABench, accuracy): ID 99.72, Domain-OOD 97.97, Version-OOD 84.30,
  **Joint-OOD 74.27 (+9.00pp over best baseline)**. Backbone Qwen2.5-7B / Qwen3-8B.
- **Critical nuance the strategy glosses over:** ATLAS attributes to model
  **family** (K=5: GPT, Claude, Gemini, Mistral, Llama), and the paper shows
  **version shift is far harder than domain shift** (−15.42pp vs −1.75pp from ID).
  Our hardest errors are **intra-lab version siblings** (gemini-flash variants,
  glm-5.3↔flash, gpt-5.6 siblings) — precisely ATLAS's weak spot. ATLAS is a
  strong **cross-family** signal (we already score ~0.96 family-level), **not** a
  silver bullet for the sibling problem that dominates our remaining errors.

### 1.3 The external "open pool" is a real, public Kaggle dataset

- **crtproduction/name-that-llm-open-pool** — "211k Clean Passages", discussion
  titled "Prompt-paired Claude, verified Gemini 3.7, recovered prompts,
  typography weights." Public on Kaggle.
  <https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool>
- The Hugging Face mirror `crtproduction/name-that-llm-open-pool` returned **401**
  (gated/private); use the **Kaggle** original, which is public.
- This is exactly the "external text corpora labelled by model" our spec §9
  permits **if registered, decontaminated against train, and validated on the
  official grouped folds**. Not used yet.

---

## 2. What is COMMUNITY-CLAIMED (corroborated, but NOT official Kaggle fact)

### 2.1 Hidden-test prompt multiplicity k≈2 (the decisive claim)

Three independent community sources converge:

1. **yauwes/name-that-llm-sota-artifacts** (HF dataset README, read in full):
   "~4,000 passages across ~2,000 unseen prompts, **k≈2**"; offline CV on full
   k=18 groups has **r=0.13** correlation with public LB, while **k=2 subsampled**
   CV has **r=0.91**.
   <https://huggingface.co/datasets/yauwes/name-that-llm-sota-artifacts>
2. **Jun Matsui, dev.to** (read in full): "the hidden test set contains **only 1
   to 3 passages per prompt**"; Hungarian matching across full 18-passage groups
   "inflated CV by +8.5%, but tanked the public Leaderboard by −0.015"; k=2 CV →
   r=0.91.
   <https://dev.to/jun-matsui/breaking-0669-to-reach-top-5-in-kaggles-name-that-llm-the-decoy-trap-dual-svc-sequence-368n>
3. **note.com strategy** + **patimakornsrinamen** CPU notebook (search snippets):
   "test set contains only 1 to 3 rows per prompt"; a k=1/2/3 passages-per-prompt
   evaluation ladder.

**Forensic submission log (yauwes, real submission IDs):** the same pipeline with
**Hard Hungarian** scored **0.639 LB** (sub 57889656) vs **Damped Sinkhorn**
**0.664 LB** (sub 57876141). The E10 ablation reports Hungarian on k=2 "fake
clusters" drops BA **−0.0509** and high-confidence accuracy **1.000 → 0.745**:
clustering merges 6–9 unrelated k≈2 prompts into fake 18-row blocks and forces 18
distinct labels, overwriting correct high-confidence predictions.

**Status:** the hidden-test structure is **not officially published** (it is
hidden by design). k≈2 is a **strong, multiply-corroborated community inference**,
not a verified Kaggle fact. We treat it as **high-confidence but unverified**, and
— crucially — **test it locally** with k-subsampled CV (Phase 0) rather than
trusting it blindly. Our spec §10 already lists the test structure as unverified.

### 2.2 Leaderboard capability tiers (community-reported, plausible)

TF-IDF ~0.33 · classical/GBDT ~0.35–0.42 (our 0.356 fits) · DeBERTa-v3 ~0.39–0.42 ·
Qwen3-8B LoRA multi-depth ~0.55–0.585 · multi-backbone LoRA stack ~0.66–0.69 ·
GLM-likelihood / top ensembles ~0.69–0.77. The yauwes personal best is 0.665; the
dev.to author claims 0.669 top-5. These are **self-reported**, not independently
audited, but the official top (0.771) and our own 0.356 bracket them credibly.

### 2.3 "Decoy camouflage" claim (treat skeptically)

The dev.to article asserts public notebooks are **deliberately de-tuned** with
"true optimum" hyperparameters hidden. This is **unverifiable** and partly
self-serving. We do **not** adopt its specific magic numbers on faith; any
hyperparameter we use must be validated on our own honest CV.

---

## 3. What is FALSE, RISKY, or NON-COMPLIANT (do not replicate)

1. **Leaked credentials.** The yauwes README publishes **live API tokens**
   (Kaggle `KGAT_…`, an HF write token, a GitHub PAT) for three accounts. These
   are someone else's secrets. **We will not use them**, and we flag their public
   exposure as a security concern. (They are not reproduced in this repo.)
2. **Multi-account quota evasion.** The same source orchestrates **3 Kaggle
   accounts** (one submitter + two GPU workers) to dodge submission/GPU quota.
   This **violates the one-account-per-participant rule** (spec §7, verified).
   We use **one account only**; all owner account actions remain owner-only.
3. **k=18 hidden-test assumption (our own D014–D016).** Our crossed Hungarian
   pipeline assumed the test mirrors train's full factorial. The evidence above
   contradicts this. **Retracted as a submission default** (ADR D017). The
   *oracle* Phase A numbers remain valid **as oracle ceilings on train**, but they
   are **not** deployable estimates for a k≈2 test.
4. **Heavy multi-account GPU stacks.** Llama-3.1-8B + Qwen3-14B + Qwen3-27B over
   8-hour multi-account runs are out of scope for our **single-account, ≤9h,
   offline, CPU-first** posture. We adopt only the parts that fit one notebook.

---

## 4. Reconciliation with our pipeline

| our assumption (D014–D016) | corrected understanding |
|---|---|
| hidden test ≈ 222 prompts × 18 models | hidden test ≈ 2,000 prompts × ~2 models (community, unverified) |
| recover prompt groups → Hungarian one-per-model | grouping is unreliable at k≈2; **Hungarian can hurt** |
| crossed candidate ready to submit (holdout 0.5000) | holdout was **k=18**; **not** evidence for a k≈2 test |
| MiniLM clustering is the key lever | clustering is fragile at small k; **global calibration is k-robust** |
| post-processing is the main gain | post-processing is ~+0.01–0.03; **base-model capability is the ~0.34 gap** |

**Dominant-strategy argument (why we can act without official k):** damped
Sinkhorn calibration balances predicted class marginals **globally** and assumes
**no prompt groups and no k**. It is safe whether k=2 or k=18. Hungarian is safe
**only** at k=18. Therefore preferring Sinkhorn (or plain argmax) over Hungarian
is the robust choice under uncertainty — and we confirm it locally with
k-subsampled CV before adopting.

**Where the real gain is:** the ~0.34 gap to the leader is a **capability gap**
(independent-passage style signal), not a post-processing gap. The credible
ladder is: clean external pool → fine-tuned DeBERTa-v3 sequence head →
multi-backbone cached-embedding readouts → (optionally) source-conditioned
likelihood features. Post-processing (Sinkhorn + typography guardrails + gentle
sibling residuals) sits on top and is worth ~+0.01–0.03, not +0.34.

---

## 5. Sources

- Official leaderboard: <https://www.kaggle.com/competitions/name-that-llm/leaderboard>
- ATLAS paper: <https://arxiv.org/abs/2610.19583> · <https://github.com/komseok0109/ATLAS>
- Open pool (public Kaggle): <https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool>
- yauwes checkpoint (community; contains leaked creds — do not use): <https://huggingface.co/datasets/yauwes/name-that-llm-sota-artifacts>
- Jun Matsui writeup (community): <https://dev.to/jun-matsui/breaking-0669-to-reach-top-5-in-kaggles-name-that-llm-the-decoy-trap-dual-svc-sequence-368n>

**Compliance posture unchanged:** no competition rows to hosted assistants; no
runtime downloads; no leaked credentials; one account; owner only accepts rules,
downloads data, uploads, and submits. External data must be registered
(`docs/external_data_register.md`), decontaminated, and validated on grouped folds.