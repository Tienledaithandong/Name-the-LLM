# Improvement plan — 0.356 → 0.7

> **Status (2026-11-14):** active. Public LB **0.356** (owner submission of the
> argmax blend), top score **0.7**. Argmax blend = `0.5 × char_tfidf_logreg +
> 0.5 × style_only`, pooled grouped-CV **0.3760**. The **crossed-design candidate**
> (D016) adds test-time prompt-group assignment, is validated (holdout **0.5000**,
> seeds 0.4716/0.4657) and **packaged ready, pending the owner's submission**.
> This plan is the working roadmap; results are filled in below as experiments
> land. Numbers here are local grouped-CV unless explicitly labelled `public LB`.

## 1. Diagnosis

- **0.356 LB ≈ 0.376 CV.** The model generalises about as well as grouped CV
  predicts, so the ~0.34 gap to the leader is a **capability gap**, not
  overfitting or prompt-shift. We are missing a signal the top solutions use.
- Our models score each passage **independently**. Char n-grams + 44 style
  features conflate *which prompt was answered* (topic) with *who wrote it*
  (style). H2 tried an SVD nuisance projection and failed (ADR D010) — a blunt
  instrument whose ceiling we never measured.
- **The untapped structure:** the audit (`docs/data_audit.md`) confirms the
  training set is a **balanced full factorial** — 150 prompts, each answered by
  17–18 of the 18 models. The hidden test (~4,000 rows) almost certainly mirrors
  this (≈222 prompts × 18 models). Within a prompt, passages differ mainly by
  **model style**; across prompts, by **topic**. Top solutions almost certainly
  exploit this crossed design. We do not.

## 2. The headline lever — exploit the crossed design

`test.csv` ships no `prompt_id`, but passages answering the same prompt share
topic/vocabulary, so prompt groups are **recoverable by semantic clustering**.
Three label-free ways to isolate style from topic once groups are known:

1. **Within-group centering.** Subtract each prompt group's mean feature/score
   vector so a passage is judged relative to its prompt-mates, removing topic
   and prompt-difficulty offsets.
2. **One-passage-per-model assignment.** The design has ~one passage per model
   per prompt. Enforce mutual exclusion within a group (Hungarian assignment on
   the score matrix). This directly attacks the dominant sibling confusions.
3. **Within-prompt contrastive encoder.** Fine-tune a small encoder to pull
   same-model passages together *across* prompts and push different-model
   passages apart *within* a prompt — learning prompt-invariant style directly.

A useful twist: the frozen MiniLM encoder that **failed authorship** (H3, 0.1278
alone — it captures topic) is exactly the right tool for **prompt clustering**.

## 3. Phased plan (ordered by impact ÷ cost)

### Phase A — measure the ceiling first (cheap, decisive) — DONE (GO for B)

Oracle experiments on grouped CV using **known train prompt groups** (upper bound
assuming perfect test grouping). Implemented in `src/crossed.py` +
`experiments/phase_a_crossed.py`.

- **A1-score:** center the blend's OOF scores within prompt groups → argmax.
- **A2:** one-per-model assignment within prompt groups on the blend's scores.
- **A1-feature:** retrain `style_only` on within-prompt-centered features
  (centered on both train and validation sides) → grouped-CV score.

**Go/no-go:** if the oracle lift is large (≥ +0.05–0.15 over 0.376), the lever is
real → invest in Phase B (and D). If negligible, pivot to Phase C.

### Phase B — deployable test-time grouping (if A confirms)

- **B1:** prompt-group recovery — MiniLM / word-TF-IDF embeddings + agglomerative
  clustering; validate cluster purity against known `prompt_id` on held-out train
  prompts; tune the distance threshold.
- **B2:** inference = cluster test → center within clusters → classify (model
  trained on centered features) → optional assignment constraint.
- **B3:** validate on a simulated crossed holdout. **Compliance check:** this is
  transductive but label-free (uses only test text). Confirm the rules permit it;
  record the ruling in `docs/competition_spec.md` §10 before submitting.
- **B4:** submit, compare LB (one of 5/day).

### Phase C — stronger base models (parallel, independent of B)

- **C1:** char n-gram sweep (2–6), vocab/C, sublinear; word+char with scaling.
- **C2:** richer stylometry — POS n-grams via attached spaCy, syntactic patterns,
  function-word profiles.
- **C3:** LightGBM/CatBoost on style + SVD.
- **C4:** larger diverse ensemble + class-specific intercepts tuned for balanced
  accuracy on OOF.

### Phase D — within-prompt contrastive encoder (highest ceiling, heavy)

- **D1:** fine-tune a small encoder with a (prompt, model)-keyed contrastive /
  triplet loss; attach weights for offline CPU inference.
- **D2:** grouped-CV validate, blend.

### Phase E — sibling discrimination

- **E1:** hierarchical provider→version classification; provider-conditional
  features targeting gemini-flash variants, glm-5.3↔flash, gpt-5.6 siblings.
- **E2:** length-band analysis (short passages are the likely weak spot).

## 4. Validation discipline (unchanged)

- Every change measured on the **frozen grouped folds + secondary holdout**,
  never tuned on the LB. Each experiment gets a registry row + ADR.
- LB is confirmation only (5 submits/day, 2 final). Owner-only account actions.
- Transductive methods are reported explicitly and kept **label-free**; they are
  never applied inside reported grouped-CV *training* scores. The Phase A oracle
  uses known train prompt groups only to measure the ceiling, and is labelled
  `oracle` (not a deployable CV score).

## 5. Results

_Filled in as experiments land. All numbers local grouped-CV unless labelled._

### Phase A (oracle ceiling) — DONE 2026-11-14

Oracle = known train prompt groups (perfect grouping assumed). **Not deployable
CV scores**; the deployable version needs test-time group recovery (Phase B).
Source: `artifacts/phase_a/results.json`, registry rows `phase_a_*`, ADR D014.

| experiment | pooled bal. acc | Δ vs 0.3760 | note |
|---|---|---|---|
| baseline blend (argmax) | 0.3760 | — | reference |
| A1-score: within-prompt score centering | 0.4152 | +0.0392 | oracle |
| A2: one-per-model assignment | 0.4853 | **+0.1093** | oracle, dominant lever |
| A1-score + A2 combined | 0.4860 | +0.1100 | oracle ceiling |
| A1-feature: style retrained on centered features | 0.3575 | +0.0671 vs style 0.2904 | deployable-consistent, ±0.0099 |

Per-single oracle (assignment): char 0.3193 → 0.4138; style 0.2904 → 0.3401.

**Decision: GO for Phase B.** The crossed-design lever is real and large
(+0.11 oracle on the blend). One-per-model assignment dominates and directly
attacks the sibling-confusion errors; score centering is secondary; centered
features also make the style model itself stronger. The realised gain hinges on
test-time prompt-group recovery purity, which B1 must validate before submitting.
Compliance gate (transductive but label-free) recorded in D014 / spec §10.

### Phase B (recovered grouping) — B1/B2 simulation DONE 2026-11-14

Deployable-consistent grouped CV: embed each fold's validation passages with
frozen MiniLM, cluster **without** `prompt_id`, apply transforms on recovered
groups. Source: `artifacts/phase_b/results.json`, registry `phase_b_*`, ADR D015.

| grouping config | purity | ARI | strict recovery | argmax | oracle assign | recovered assign | rec center+assign |
|---|---|---|---|---|---|---|---|
| k = rows/18 | 0.934 | 0.916 | 0.833 | 0.3760 | 0.4853 | **0.4727** | 0.4756 |
| cosine thr 0.4 | 0.989 | 0.944 | 0.753 | 0.3760 | 0.4853 | 0.4704 | 0.4649 |
| cosine thr 0.5 | 0.940 | 0.921 | 0.833 | 0.3760 | 0.4853 | 0.4690 | 0.4741 |

**Result: recovered-group assignment captures +0.0967 of the +0.1093 oracle gain
(88%).** Assignment alone is robust to imperfect clustering; centering adds little
and hurts when clusters are wrong → use **assignment-only**. The `k = rows/18`
prior works best and needs no threshold tuning; the threshold variant confirms
robustness when the design is uncertain.

**Decision: the lever is deployable in principle.** Remaining before adoption
(D015): (1) compliance gate — **CLEARED 2026-11-14**: the Rules forbid only
hand-labelling/human prediction of test records (Foundational §4.b), not
label-free feature-space clustering; §6 allows public pretrained models/AMLT; no
transductive prohibition (spec §9); (2) full cluster→assign validation on the
secondary holdout + an alternative seed; (3) k-estimate sensitivity /
larger-batch check (real test ~4,000 rows vs ~540 per fold here). The pipeline
now depends on MiniLM → weights must be attached offline
(`docs/external_data_register.md` updated).

### Phase B validation (holdout + alternative seeds) — DONE 2026-11-14

Full cluster→assign pipeline refit per fold on three independent splits (blend
members + 0.5/0.5 weights fixed; only grouping + assignment added at inference).
Source: `artifacts/phase_b/validation.json`, registry `phase_b_*_rec_thr`, ADR
D016.

| split | argmax | oracle assign | recovered thr 0.4 | recovered k=rows/18 | purity (k) |
|---|---|---|---|---|---|
| untouched holdout s7 (25 prompts) | 0.3956 | 0.4933 | **0.5000** | 0.4933 | 1.000 |
| grouped CV seed 7 | 0.3608 | 0.4808 | 0.4716 | 0.4712 | 0.940 |
| grouped CV seed 13 | 0.3582 | 0.4742 | 0.4657 | 0.4631 | 0.940 |

**Result: the gain holds out of sample on every split** (+0.1044 holdout, +0.1108
seed 7, +0.1075 seed 13). Holdout argmax 0.3956 exactly reproduces D012 →
harness correct. Threshold 0.4 ≥ k-estimate on all three splits and keeps the
highest purity at scale → **adopt thr 0.4** (no rows/prompt assumption).

Scale check (2,699 passages / 150 prompts, no accuracy): k=150 purity 0.891;
thr 0.4 purity 0.960 (216 clusters, over-splits but pure); thr 0.5 purity 0.871.
Purity degrades gracefully; over-splitting pure clusters is harmless to
assignment, impure merges are the failure mode thr 0.4 avoids.

**Decision: D015 gates satisfied → adopt crossed inference as the submission
candidate (D016), superseding the argmax blend pending the owner's LB
submission.** Deployable path `src.inference.predict_dataframe_crossed` is
implemented and tested. Remaining: productionize (CLI `--crossed`, notebook
loads attached MiniLM offline, `package` ships the checkpoint), then owner
submits and we read the LB.

### Phase B productionization — DONE 2026-11-14

The crossed candidate is packaged and verified end-to-end offline:

- `src.cli predict --crossed` (flags `--distance-threshold` 0.4, `--n-clusters`,
  `--encoder-backend`, `--encoder-path`) calls `predict_dataframe_crossed`.
- `src.cli package --crossed` ships the MiniLM checkpoint (`encoder/minilm`,
  88 MB) + `inference.json` (`crossed`, `encoder_path`, `distance_threshold` 0.4)
  alongside the blend bundles, `weights.json` and `src/`. Auto-resolves the HF
  cache offline (`snapshot_download(local_files_only=True)`) or takes
  `--encoder-path`.
- `kaggle/submission.ipynb` sets `HF_HUB_OFFLINE=1`, reads `inference.json`, loads
  the encoder from the attached folder and runs crossed inference; falls back to
  argmax when `inference.json` is absent.
- `make predict-crossed` / `make package-crossed` (reuse `configs/submission.yaml`;
  no duplicate config, avoiding member drift per D011).

Verification: the packaged `src` unpickles the bundles, MiniLM loads with
`HF_HUB_OFFLINE=1` (no network), and the notebook cells execute end-to-end. At
2,699-row scale (all train passages as an unlabelled proxy batch) crossed
inference runs in **~56 s CPU** and yields a near-uniform label distribution
(max class share **0.056 ≈ 1/18**) — the full-factorial signature, confirming
assignment works at scale. 111 unit tests pass (3 new: CLI crossed ×2, package
crossed ×1).

Caveats: the public `test.csv` is a **3-row format sample** (spec §10); the real
~4,000-row hidden test is only available on Kaggle, so full-scale accuracy cannot
be measured locally — the 2,699-row train proxy + holdout/seed gains are the
available evidence. The bundles were refit with current code (the earlier
`final/official` pickle predated the `_encoder` attribute and failed to load);
config/seed/data are unchanged so the blend is equivalent. The crossed pipeline
needs `sentence-transformers` (+ `transformers`, `torch`) importable in the Kaggle
image; the weights are attached, never downloaded.

**Next: owner uploads `artifacts/kaggle_dataset_crossed` (set `--kaggle-user`),
attaches it to the notebook, submits, and reads the LB.** If it beats 0.356,
proceed to Phase C/D to push toward 0.7; if clustering collapses on the real test
(degenerate label distribution), fall back to the argmax blend.