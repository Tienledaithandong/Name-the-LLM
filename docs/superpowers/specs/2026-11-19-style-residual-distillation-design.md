# Design — Self-neutralized style residual (approach A)

**Date:** 2026-11-19
**Status:** approved 2026-11-19; implementation plan at
[../plans/2026-11-19-style-residual-distillation.md](../plans/2026-11-19-style-residual-distillation.md).
**Outcome:** refuted at P0 by the section 5 gate — see D058 in
[../../decisions.md](../../decisions.md). P1/P2/P3 were not run and no GPU spend
followed, as the spec requires.
**Approach chosen:** A (style-residual distillation). Approaches B (early-layer Qwen
LoRA) and C (multi-signal combined head) were considered and deferred.

## 1. Intent and why

Current position: adopted CPU blend, pooled grouped OOF **0.3760** (cv_s42), public LB
**0.356**. Live leaderboard top is **0.776** (verified 2026-11-19; #2 0.771, #3 0.770).
Community best single model is a Qwen LoRA at ~0.644–0.657; ensembles reach 0.776.

So ~0.8 is **above the current leader**. It is not reachable by another rung on the
"bigger encoder, more pool data" ladder — everyone is already on that ladder. It needs a
signal the field is not exploiting.

**The signal we identified.** "Every Model Leaves a Trace: Multi-Resolution LLM
Fingerprinting" (arXiv 2610.28434) evaluates a **Prompt-Neutralized Embedding (PN
Embedding)**: subtract the mean embedding of all candidate models' responses to the
same prompt. It wins on every view and every setting:

| setting | best standard signal | PN Embedding |
|---|---|---|
| cross-family, raw text | 81.41 (combined) | **83.75** |
| cross-family, punctuation/case stripped (T3) | 67.04 | **69.35** |
| full paraphrase (P2) | 71.86 | **75.23** |
| cross-lingual EN→ZH | 39.00 | **59.10** |
| rewrite task | 35.94 | **48.75** |

The paper states the catch itself: PN Embedding "requires access to peer responses from
all candidate models for the same prompt". That is exactly why the competition field
cannot use it — the hidden test is believed to be k≈2.

**Our edge.** Training data is a full factorial: 150 prompts × 18 models (k=18). We can
compute prompt means on train. We cannot compute them on test. Approach A's bet is that
the prompt component of a representation is *predictable from the passage itself* — which
is precisely what the community layer-scan found independently ("deep layers encode what
the text is about, early layers capture how it is written"; frozen Qwen3-4B, prompt-grouped
CV: layer 3 = 0.391, six layers concatenated = 0.414, last layer = 0.311). If a passage
reveals its own topic, a learned function can estimate the prompt mean and subtract it,
recovering most of the PN gain **without peers**.

We already have two pieces of local evidence that the prompt component is worth removing:
oracle within-prompt centering gave **+0.067** on style features (0.290 → 0.357) and
**+0.039** on the blend OOF (0.376 → 0.415). Both need known groups and are therefore
not deployable. Approach A is the deployable version of that measurement.

## 2. Method

Notation: passage `i` has prompt `p(i)`, representation `e_i` (any fixed feature map).

1. **Prompt mean.** `m_p = mean_{i in p} e_i`, computed **only from training rows of the
   current fold** (never from validation rows, never from test).
2. **Oracle residual (ceiling, not deployable).** `r_i = e_i − m_{p(i)}`. Uses the true
   group, so it is a diagnostic only, consistent with D018's oracle-ceiling framing.
3. **Learned neutralizer (the candidate).** Fit a regressor `g: e → m` on training rows,
   target = that row's own prompt mean. Features = the same per-passage representation.
   Then `r̂_i = e_i − g(e_i)`, which is a function of one passage and is deployable at
   any k, including k=1.
4. **Shrinkage.** `r̂_i = e_i − λ·g(e_i)`, with `λ ∈ {0, 0.25, 0.5, 0.75, 1.0}` swept
   inside each fold on the training side only. `λ=0` is exactly today's model, so the
   candidate can never be forced to beat nothing by construction.
5. **Task.** Two heads on the residualized representation:
   - classifier head: the existing 18-class `logreg` / `linearsvc` on `r̂`;
   - optionally a **prompt-centred auxiliary head**: predict the class of `e_i − e_j` for
     same-prompt pairs `(i, j)` with different models (a within-prompt discrimination
     objective). This is cheap (pair sampling, no new data) and directly optimises the
     residual to carry label signal rather than topic.
6. **Inference.** `r̂` for each test passage independently, then the existing blend +
   argmax. No clustering, no grouping, no peers, no transductive step.

The neutralizer is fitted **per fold** on training rows only; prompt means for validation
rows are never computed. That is the whole leakage story: a validation row's `g(e)` must
not have seen its own prompt.

### 2.1 Relationship to D010 (must not repeat a measured negative)

D010 is our own published negative result: `NuisanceSuppressor` (`src/prompt_invariance.py`)
projects out the **top-k global directions of between-prompt centroid variation**, and it
hurt. Stated plainly: **A contains D010 as a special case.** D010 corresponds to
`g(e) = P(e − μ)/σ` for a single fixed projection `P` of rank k.

What is genuinely different, and what the experiment must therefore isolate:

1. **Per-passage, nonlinear prediction.** `g` predicts *this passage's* prompt mean from
   its own content, not one global subspace. A fixed `P` cannot express "this passage is
   about physics, so subtract the physics centroid".
2. **Shrinkage `λ`** swept inside the fold, with `λ=0` recovering today's model exactly.
   D010 had no such guard and could only help or hurt.
3. **Applied to frozen/early-layer embeddings**, where the source paper measures the PN
   gain, rather than to the SVD space where D010 was refuted.
4. **Auxiliary within-prompt pair head**, which shapes the residual to keep label signal
   while dropping topic.

P0 therefore reports a 2×2 table before anything else: {global projection (D010),
learned `g`} × {SVD space, style space, embedding space}. If learned `g` never beats
D010's projection, the idea is dead and is logged as such.

### 2.2 Pre-check that costs minutes

`prompt_centering_diagnostic` already returns `between_prompt_share`, the share of
feature variance explained by prompt identity. Run it first on every candidate
representation. A representation with a low share has no prompt component to remove and
cannot benefit from A; a high share says the headroom exists but says nothing yet about
whether `g` can find it. This number is reported with the result either way.

## 3. Scope

In scope (in this order, each gated on the previous):

- **P0 — neutralizer on existing CPU representations.** SVD/TF-IDF blocks, the 44 style
  features, and the local frozen MiniLM embedding already registered. Answers one
  question cheaply: *how much of the prompt mean is predictable from a passage?* Ships
  the §2.2 `between_prompt_share` pre-check and the §2.1 D010-vs-learned-`g` table.
- **P1 — early-layer embeddings.** Qwen3-0.6B/4B (or the existing DeBERTa) early-layer
  mean-pooled embeddings, layer chosen by grouped CV, cached once. Locally this needs the
  Kaggle GPU kernel we already have; the CPU path stays the fallback.
- **P2 — auxiliary within-prompt pair head** on top of the best P0/P1 representation.
- **P3 — balanced-accuracy post-processing** on the candidate ensemble: logit adjustment
  toward the balanced objective / damped Sinkhorn (`src/calibration.py`), swept inside
  folds. Cheap, and the metric is balanced accuracy with an unequal, shifted test mix.

Out of scope:

- Test-time grouping, clustering, Hungarian assignment, any transductive operation
  (D017/D024/D026 withdrawn). The neutralizer is a fixed function of one passage.
- New external data. The pool (open-pool-v15) is already audited/admitted; if used, the
  neutralizer must be refit within the pool-augmented training side only.
- Public-LB tuning, multi-account GPU stacks, paid generation.
- Replacing the adopted submission until a candidate wins the gate in §5.

## 4. Constraints

- Prompt-grouped folds only; every learned transform (vectorizers, scalers, SVD,
  neutralizer, λ) fitted on the training side of the fold. `src/validation.py` already
  enforces prompt isolation; the new code must reuse `load_splits`/`validate_splits`.
- Primary metric stays **pooled grouped OOF balanced accuracy** (D018); k-subsampling
  stays an inference-only stress test with paired differences and 95% CIs.
- No LinearSVC margins treated as probabilities; blending keeps row-z normalisation.
- Artifacts: 18-label order from `src/constants.LABELS`; `ModelBundle.load` still rejects
  a mismatch. New feature state must travel inside the saved bundle.
- Registry row + ADR for every candidate; no number reported that was not produced by a
  run whose command is in the row's notes.
- CPU-first. Any GPU step is a `kaggle/` script kernel with the local `folds.json`.

## 5. Acceptance gate

A candidate replaces the adopted blend only if **all** hold:

1. Pooled grouped OOF BA beats **0.3760** (cv_s42), same `split_id`, same fingerprint.
2. The gain is not carried by one fold (per-fold deltas all ≥ 0, or the mean is reported
   with the fold spread).
3. The gain survives **seed 7** (fresh folds, same protocol).
4. `mean-normalized` paired difference vs the current blend has a 95% CI that excludes 0.
5. **Neutralizer diagnostic passes first:** out-of-fold `R²` of `g(e)` predicting its own
   prompt mean must be materially above 0 (report the value; a collapsed neutralizer —
   `‖g(e)‖ ≈ 0` — means the premise failed and the idea is logged negative, not tuned),
   **and** the learned `g` must beat the D010 global projection on the same space, or A
   adds nothing over a result we already published as negative.
6. Oracle residual is reported alongside, so we know how much headroom remains unused.

If (5) fails, A is recorded as a negative result with the measured `R²` and the line stops
before any GPU spend.

## 6. Approaches considered

| | approach | what it trades | why not now |
|---|---|---|---|
| **A** | Self-neutralized style residual (this doc) | needs a predictable prompt component; unproven, but cheap to falsify on CPU and unique to us | chosen |
| B | Early-layer multi-depth Qwen LoRA on the 226k pool | proven (0.647–0.657 LB single), but crowded, GPU-heavy, and lands ~0.65 not 0.8 | deferred; becomes the base model if A shows signal |
| C | Multi-signal combined head (behavioral + POS TF-IDF + word/char TF-IDF + embedding + metric head) | cheap floor-lifter, +1.3pp evidence at best | deferred; can be folded into P0 representations later |

A and C are not exclusive; C is a feature-set question inside P0 and will be picked up if
a single-view residual shows nothing.

## 7. What is left to the builder

- Which representation to neutralize first (SVD vs style vs MiniLM) and its dimension.
- Neutralizer family (ridge vs `HistGradientBoostingRegressor` vs a small MLP) and its
  regularisation; ridge on standardized features is the sane default.
- Whether to residualize before or after the SVD reduction.
- Pair-sampling regime for the auxiliary head (pairs per prompt, loss temperature).
- Runtime budget for P1 (one Kaggle session, ≤9h) and the layer list to scan.

Owner decisions already made: approach A; 0.8 is a stretch target, not a gate — the
gate is the validated-gain test in §5, and every number reported comes from a real run.
