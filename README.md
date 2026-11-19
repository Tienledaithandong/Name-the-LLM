# Name That LLM — CPU-first model attribution

18-class authorship attribution for the Kaggle **Name That LLM** competition:
given a 40–250 word passage, identify which of 18 frontier language models wrote
it. Metric: **balanced accuracy** (mean per-class recall).

Competition: <https://www.kaggle.com/competitions/name-that-llm>

> **Status (2026-11-16):** the submitted official-only sparse blend scored **0.356
> public balanced accuracy** and remains the confirmed fallback. The current
> candidate is the **cap-40k pool-MIX blend** — `0.5 × char TF-IDF LogisticRegression
> trained on official train + a globally decontaminated external pool` + `0.5 ×
> style-only LogisticRegression (official-only)` — at **0.4427 / 0.4304 / 0.4426**
> local grouped OOF under three independent protocols (seed 42, seed 13, nested
> outer folds), independently reproduced from identity-verified OOF (D039). It is
> **packaged but not submitted**: uploading and submitting are owner-only, so no
> leaderboard number exists for it yet. The earlier 0.4442 DeBERTa+sparse candidate
> is **superseded** — its pool arm used a transductive per-fold screen (D026) and
> official-only DeBERTa adds no diversity gain (D027). Synthetic results are pipeline
> checks only.

## What is verified vs. assumed

`docs/competition_spec.md` records the official facts read from Kaggle on
2026-11-14: the balanced-accuracy metric, the 18 labels, `train.csv`
(2,699 rows / 150 prompts), `test.csv` (`id`,`text`), `submission.parquet` written
to the working directory, `/kaggle/input/competitions/name-that-llm/test.csv`,
9-hour CPU/GPU limits, internet disabled, external data and pretrained models
allowed if publicly accessible, 5 submissions/day, 2 final submissions, team ≤ 5,
entry deadline 2027-01-06, final deadline 2027-01-13 (all 23:59 UTC).

Nothing in the code depends on an unverified item; open questions are listed at
the end of that document.

## Design in one page

```text
train.csv ──► schema validation ──► prompt-grouped folds (frozen, fingerprinted)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        char / word / char_wb TF-IDF                stylometry (56 features)
        (+ optional SVD, prompt-nuisance            length, punctuation, markdown,
         suppression)                               hedging, function words, TTR
                    │                                       │
        external pool (optional, char only)                 │
        screened ONCE vs full train ──► capped              │
                    └───────────────┬───────────────────────┘
                                    ▼
                      LinearSVC / LogisticRegression
                      (balanced class weights)
                                    │
                  out-of-fold margins (18 columns, frozen order,
                   dataset hash + per-row fold ids recorded)
                                    │
              blend on row-wise z-scores (adopted: fixed 0.5/0.5)
                                    │
              refit on all data (+ clean pool) ──► joblib artifact dir
                                    │
              kaggle/submission.ipynb ──► submission.parquet
```

Key properties:

- **Prompt-grouped CV only.** Hidden test passages come from prompts absent from
  training; folds hold out whole prompts and the harness raises on any overlap.
- **Per-fold fit hygiene.** Vocabularies, IDF, style standardiser, SVD and
  nuisance projection are fitted on the training side only.
- **Non-transductive external data.** The pool keep-mask is computed once against the
  full official train and reused unchanged by every fold; per-fold screening is rejected
  by both the CPU experiment and the GPU kernel (D026/D028).
- **OOF identity.** Every score matrix carries the ordered-dataset SHA-256 and per-row
  fold ids; loading and blending refuse misaligned or legacy files (D024).
- **Fixed class order.** Every score matrix and artifact uses
  `src/constants.LABELS`; `ModelBundle.load` refuses artifacts with a different
  order.
- **Fail-fast submission.** Schema, ids, order, nulls and label validity are
  checked before and after writing the parquet.
- **CPU-first.** Sparse linear models train in seconds to a couple of minutes on
  CPU; no GPU required. The DeBERTa research kernel is the only GPU path.

## Install and test

```bash
python -m pip install -r requirements.txt
python -m pytest -q            # 381 tests
```

Python 3.11+ (developed on 3.13). Versions are pinned in
`requirements-lock.txt`.

## Reproducibility

One command, no competition data, no network:

```bash
python -m src.cli demo
# or: make demo
```

It generates the synthetic fixture, audits it, builds 5 prompt-grouped folds,
cross-validates the baselines, blends, refits, predicts, validates
`submission.parquet`, and reports a held-out score on the fixture.

With the real data in `data/`:

```bash
make audit      # docs/audit + figures
make splits     # frozen folds + fingerprint
make baseline   # 6 reference/TF-IDF models -> OOF, metrics, registry
make features   # stylometry ablations
make embeddings # SVD + prompt-nuisance variants (H2)
make frozen-encoder # H3 pretrained-encoder benchmark (optional sentence-transformers)
make ensemble   # identity-checked, explicitly listed OOF candidates; writes exploratory weights
make evaluate   # score saved identity-verified OOF: pooled/fold BA, CI, paired blend deltas
make figures    # report figures from saved OOF/metrics
make stability  # blend robustness: alternative fold seeds + secondary holdout
make predict    # refit blend members (configs/submission.yaml) -> submission.parquet
make package    # upload-ready Kaggle artifact dataset (models + weights + src)
make predict-crossed   # crossed-design inference (D016): prompt-group assignment
make package-crossed   # dataset + offline MiniLM grouping encoder (needs sentence-transformers)
```

The **current candidate** adds the globally decontaminated external pool to the char
member only (D028/D037); the blend rule and weights are unchanged:

```bash
python -m src.cli predict --config configs/submission.yaml --data-dir data --fit \
  --pool artifacts/pool_audit/ntl_open_pool.parquet \
  --pool-models char_tfidf_logreg --pool-cap 40000
python -m src.cli package --dataset official --out artifacts/kaggle_dataset --kaggle-user <you>
```

Omit `--pool`/`--pool-models` to rebuild the confirmed 0.356 official-only fallback.
`experiments/pool_train.py` reproduces the CV numbers (`--cap-mode proportional
--pool-cap 40000`), `experiments/nested_pool_blend.py` the nested gate,
`experiments/blend_deberta.py --no-deberta` the independent reproduction, and
`experiments/length_sensitivity.py --data-dir data` the §8 error analysis (length bands,
confusion pairs, candidate figures) and `experiments/blend_rules.py --data-dir data` the §5.1
blend-rule comparison, both from the same saved OOF.

`make evaluate` (or `python -m src.cli evaluate --dataset official --artifacts
artifacts/cv_identity/official --members char_tfidf_logreg style_only --weights 0.5 0.5`)
re-scores any saved, identity-verified OOF without fitting anything or reading a test row:
pooled and per-fold balanced accuracy with a 95% CI, macro F1, the confusion matrix,
per-class recall, within-provider errors, and paired blend-minus-member deltas. It refuses
OOF whose fingerprint or validation-fold assignment disagrees with the frozen splits, and
`--length-bands` adds the D041 word-count bands after re-checking the local train frame
against that fingerprint. Fit/infer seconds and peak RAM belong to the training run and are
recorded by `cli train`, not re-measured here. The cross-directory pool-MIX candidate is
scored by `experiments/blend_deberta.py` / `blend_rules.py`, which take one path per member.

Official-only ablations and audits (D044–D049) — none changes a config, weight or package:

```bash
python -m src.cli train --config configs/style_svm.yaml --data-dir data --out artifacts/cv_style_svm
python -m src.cli train --config configs/style_catboost.yaml --data-dir data --out artifacts/cv_style_catboost  # optional catboost
python experiments/class_bias.py --data-dir data --seeds 42 13   # ~5 min per seed
python experiments/style_features.py --data-dir data
python experiments/near_duplicates.py --data-dir data
python experiments/prompt_groups.py --data-dir data
```

Package licence and freshness (D050): `cli package` derives the Kaggle dataset licence from
the shipped manifest — **CC-BY-SA-4.0** plus a README attribution block crediting the
competition and, for any pool-fitted member, Open Pool v15, because the weights are a derived
adaptation of CC BY-SA 4.0 text. A synthetic-fixture package stays CC0-1.0. **Do not relabel
the uploaded dataset CC0.** Re-packaging also refreshes `src/` without refitting: the bundles
are copied byte-for-byte from `artifacts/final/official`, and `weights.json` keeps the D040
provenance fix.

Then attach the packaged dataset to `kaggle/submission.ipynb` and submit. The notebook
requires `weights.json` and **rejects** an `inference.json` marked `crossed: true`, so the
withdrawn D016 crossed pipeline cannot be submitted by accident; `make package-crossed`
remains only as a historical research artifact. Owner-only account steps (rules, upload,
submit, final selection) are listed in `docs/submission_checklist.md`.

## Results

### Official training data (2026-11-14)

2,699 passages, 150 prompts, 18/18 labels, 5 prompt-grouped folds (30 held-out
prompts each, zero prompt overlap). Pooled out-of-fold balanced accuracy:

| model | pooled OOF | fold mean ± std | macro F1 | fit s (5 folds) |
|---|---|---|---|---|
| char TF-IDF + LogisticRegression | 0.3193 | 0.3193 ± 0.0119 | 0.3089 | 80.6 |
| char TF-IDF + LinearSVC | 0.3167 | 0.3167 ± 0.0134 | 0.3049 | 75.2 |
| word+char TF-IDF + LinearSVC | 0.3156 | 0.3156 ± 0.0094 | 0.3025 | 73.5 |
| `char_wb` TF-IDF + LinearSVC | 0.2963 | 0.2963 ± 0.0101 | 0.2877 | 49.6 |
| style only + LogisticRegression | 0.2904 | 0.2904 ± 0.0142 | 0.2792 | 29.6 |
| style only + LinearSVC (D045) | 0.2844 | 0.2845 ± 0.0152 | 0.2574 | 26.6 |
| style only + CatBoost (D049) | 0.2830 | 0.2830 ± 0.0153 | 0.2754 | 115.0 |
| word TF-IDF + LinearSVC | 0.2697 | 0.2697 ± 0.0115 | 0.2539 | 7.5 |
| char+style TF-IDF + LinearSVC (unstable) | 0.2245 | 0.2244 ± 0.0849 | 0.2254 | 1213.8 |
| majority class | 0.0556 | 0.0556 ± 0.0000 | 0.1053 | 0.0 |
| stratified random | 0.0545 | 0.0545 ± 0.0047 | 0.0534 | 0.0 |

**Current best — validated blend:** `0.5 × char TF-IDF LogisticRegression +
0.5 × style-only LogisticRegression` → **0.3760** pooled OOF, **+0.0567** over
the best single model. The gain appears in every fold (+0.035 to +0.074), so it
is not one lucky split:

| fold | blend | best single | gain |
|---|---|---|---|
| 0 | 0.3779 | 0.3149 | +0.0630 |
| 1 | 0.3759 | 0.3407 | +0.0352 |
| 2 | 0.3796 | 0.3056 | +0.0741 |
| 3 | 0.3741 | 0.3222 | +0.0519 |
| 4 | 0.3722 | 0.3130 | +0.0593 |

**Ablation** (`configs/features.yaml`, same folds):

| ablation | pooled OOF |
|---|---|
| char only, LinearSVC | 0.3167 |
| word + char, LinearSVC | 0.3156 |
| style only, LogisticRegression | 0.2904 |
| style only, LinearSVC (D045) | 0.2844 |
| style only, CatBoost (D049) | 0.2830 |
| word only, LinearSVC | 0.2697 |
| char + style, LinearSVC | 0.2245 (2 of 5 folds collapsed; non-convergent) |
| char + style, LogisticRegression | stopped: no convergence, >37 min, ~0.7 GB RSS |
| char → SVD-256, LinearSVC | 0.3034 |
| char → SVD-256 → nuisance-8, LinearSVC | 0.2434 |
| style → SVD-64 → nuisance-8, LogisticRegression | 0.2737 |
| MiniLM embeddings only, LogisticRegression | 0.1278 |
| MiniLM embeddings + style, LogisticRegression | 0.3093 |

Findings: style alone is competitive with a 100k-feature character model using
only 56 features, and the two are complementary enough to blend (+0.0567).
Concatenation is not a free lunch here — `word+char` matches char alone, and the
`char+style` sparse concat failed to converge at this vocabulary size.
Deployable prompt-nuisance suppression (H2) is a **negative result**: projecting
out the top-8 between-prompt directions lowers pooled OOF for char
(0.3034→0.2434) and style (0.2904→0.2737); SVD-256 compression alone also slightly
lowers char (0.3167→0.3034). See report §6.4 and ADR D010.
A frozen semantic encoder (H3) is also a **negative result**: MiniLM embeddings
score 0.1278 alone (topic, not authorship), and the greedy char+encoder blend
gains only +0.0026 on CV (0.3786) which reverses on the holdout (0.3933 vs
0.3956), so it is not adopted. See report §6.6 and ADR D013.

**Error analysis of the blend:** weakest classes `claude-fable-5` (0.121),
`glm-5.3-flash` (0.147), `gpt-5.6-sol` (0.153); strongest
`gemini-3.1-pro-preview` (0.733) and `claude-opus-5` (0.707). 53.3% of errors
stay inside a provider, and the top confusions are siblings:
`gemini-3.8-flash`→`gemini-3.7-flash` (56), `gemini-3.6-flash`→`gemini-3.7-flash`
(40), `glm-5.3`↔`glm-5.3-flash` (70 combined).

Data audit: each class has 149–150 rows, each prompt covers 17–18 models, no
duplicate texts, mean 114.7 words, 57.84% of passages contain non-ASCII
characters, and there is no markdown or emoji in the corpus. Details in
`artifacts/audit/official/data_audit.md` and `docs/data_audit.md`.

### External-pool candidate (D028/D037 — current, packaged, not submitted)

External data: **Open Pool v15** (`crtproduction/name-that-llm-open-pool`, CC BY-SA 4.0),
226,191 rows covering all 18 labels. The audit found the routing-risk flags already
`False` (risky rows had been moved to a separate file), so re-filtering them is a no-op;
`match_weight`/`style_score` are fitted on all of train and are **never** used as features.

Decontamination is **global and fold-independent**: the pool is screened *once* against
the full official train (exact text match + word TF-IDF cosine > 0.3, vectorizer fit on
that same train) → **226,049** clean rows (0 exact, 142 near-duplicates). One keep-mask is
reused by every fold, so held-out passages never select which pool rows get trained on.
The earlier *per-fold* screen was transductive and is disabled (D026); `--pool-scope fold`
is rejected by both the CPU experiment and the GPU kernel.

| model / blend | s42 | s13 | nested outer gate |
|---|---|---|---|
| char TF-IDF, official only | 0.3193 | 0.3141 | — |
| char TF-IDF + pool, cap 12k/fold | 0.3667 | — | — |
| char TF-IDF + pool, cap 40k/fold | **0.4104** | **0.4115** | — |
| MIX cap 12k | 0.3990 | 0.3989 | 0.3930 |
| **MIX cap 40k (candidate)** | **0.4427** | **0.4304** | **0.4426** |
| official-only sparse blend (control) | 0.3760 | 0.3582 | 0.3519 |

MIX = `0.5·z(char trained with pool) + 0.5·z(style official-only)`, row-wise z-scored —
the *same* fixed rule as the fallback, which reproduces 0.3760 exactly. Style does **not**
benefit from the pool (−0.0174, CI spans 0), so the pool is char-specific.

The cap, not the recipe, was the binding constraint (D037). Paired 40k vs 12k:
**+0.0437 [+0.0238,+0.0636]** s42, **+0.0315 [+0.0089,+0.0540]** s13,
**+0.0908 [+0.0460,+0.1355]** nested; vs the 0.3760 base, +0.0667 [0.0432,0.0903] s42 and
+0.0722 [0.0461,0.0983] s13. All folds positive under all three protocols. The nested gate
(3 outer prompt-grouped folds, pre-specified weights, pool screened against outer-train
only) replaces seed-2027, which turned out to be **examined** rather than untouched (D035).
`experiments/blend_deberta.py` independently reproduces the headline from stored
identity-verified OOF: **0.442675** s42 / **0.430435** s13 / control **0.375971** (D039).

**Sibling starvation and the balanced-cap rejection (D036/D037).** Official train is
balanced (~150 rows/class) but the pool is skewed, so a class-proportional cap starves
`gemini-3.7-flash` (188 rows at cap 12k vs 974/1246 for its 3.6/3.8 siblings) and char
routes it into them (recall −0.173). A class-balanced cap fixes 3.7 (−0.173 → +0.013) and
even improves char alone at 40k (0.4234 > 0.4104), but the **blend regresses** — 0.3949 vs
0.3990 at 12k, 0.4378 vs 0.4427 at 40k (paired −0.0048, CI spans 0) — because it forfeits
more on 3.8/3.6 than it gains on 3.7, and official-only style already covers 3.7.
**Rejected at both budgets; proportional-40k stands.** No per-class prior was hand-tuned on CV.

### Error analysis: text-length sensitivity (D041)

`python experiments/length_sensitivity.py --data-dir data` recomputes both blends from the
saved identity-verified OOF (it fits nothing and scores no test row) and reproduces their
pooled scores exactly — 0.375971 official-only, 0.442675 candidate. Passages span 36–274
words; every band below contains all 18 classes, so the band scores are comparable.

| passage words | n | official blend (LB 0.356) | pool-MIX cap 40k (candidate) | paired gain |
|---|---|---|---|---|
| [0, 50) | 301 | 0.2491 | 0.3125 | +0.0634 |
| [50, 100) | 1,021 | 0.3393 | 0.3824 | +0.0431 |
| [100, 150) | 619 | 0.3841 | 0.4534 | +0.0693 |
| [150, 200) | 444 | 0.4457 | 0.5356 | +0.0899 |
| [200, ∞) | 314 | 0.4718 | 0.5858 | +0.1140 |

- **Length is the strongest error predictor measured here:** both blends roughly double
  from the shortest to the longest band. Short passages carry too little stylistic evidence.
- **The pool helps most on long passages** (+0.1140 at ≥200 words vs +0.0431 at 50–100), so
  external data amplifies existing evidence rather than repairing short-passage failures.
- **Errors change type with length:** the within-provider share rises 40.9% → 66.7%
  (official) and 40.0% → 64.1% (candidate) — short passages fail at provider level, long
  ones at version level. The candidate removes 180 errors, 114 of them sibling confusions;
  14 of 18 classes improve (`gemini-3.8-flash` +0.253, `gemini-3.7-flash` −0.180).
- The hidden test's length mix is **unknown** (the local `test.csv` is the 3-row interactive
  example), so this is a train-OOF diagnostic, not a transfer measurement.

Figures: `figures/length_band_accuracy.png`, `figures/confusion_matrix_pool_mix_cap40k.png`,
`figures/per_class_recall_pool_mix_cap40k.png`. Full tables: report §8.1–§8.2.

The candidate's ablation, runtime/RAM and grouped-CV variation panels come from
`python experiments/candidate_figures.py --data-dir data` (D051): the two members were fit in
separate runs, so `cli figures` can only draw the baseline. It recomputes all five arms from
the saved OOF and aborts if any pooled or per-fold score disagrees with the recorded value.
The candidate's fold spread is wider than the fallback's (std 0.0166 vs 0.0026) while the gain
stays positive in every fold; the pool arm's peak fit RAM was never recorded and is labelled
as such rather than estimated. Figures: `figures/ablation_pool_mix_cap40k.png`,
`figures/runtime_ram_pool_mix_cap40k.png`, `figures/cv_variation_pool_mix_cap40k.png`.

### Hard typographic vetoes (D053) — measured, not adopted

`python experiments/typo_vetoes.py --data-dir data` tests the Discussion-reported idea that a
provider's punctuation habits are near-invariants worth a **hard** mask. Rules are mined per
fold on that fold's training rows only, with support and a prompt-aware chance level recorded,
and a row is never left with every class vetoed.

| arm | base | vetoed | Δ | paired 95% CI |
| --- | --- | --- | --- | --- |
| pool-MIX cap40k | 0.442675 | 0.443415 | +0.000741 | [−0.001316, +0.002797] |
| official-only | 0.375971 | 0.376341 | +0.000370 | [−0.000658, +0.001399] |

Both intervals include 0, so the candidate is unchanged. Nine predictions flip in 2,699 rows,
all from wrong ones. The chance level counts **distinct prompts, not rows** (2,699 rows are 150
prompt groups); even so `em_dash → Google` passes at 2.74e-26 and is broken by a real
`gemini-3.7-flash` text, so a small chance is not proof of an invariant. Evidence:
`artifacts/typo_vetoes/veto_report.md`.

### Blend-rule comparison (D042)

`python experiments/blend_rules.py --data-dir data` scores five scale-only blending rules on
the same saved OOF rows, same 0.5/0.5 weights, both member pairs. No fit, no test row, no pool
row read; paired per-fold differences remove shared fold noise.

| rule | official-only | Δ vs row-z (95% CI) | pool-MIX cap 40k | Δ vs row-z (95% CI) |
|---|---|---|---|---|
| raw margin mean | 0.3241 | −0.0518 [−0.0747, −0.0289] | 0.4208 | −0.0219 [−0.0269, −0.0169] |
| **row-z mean (adopted)** | **0.3760** | — | **0.4427** | — |
| average-rank mean | 0.3619 | −0.0140 [−0.0296, +0.0015] | 0.4060 | −0.0367 [−0.0603, −0.0130] |
| min-max mean | 0.3760 | −0.0000 [−0.0096, +0.0096] | 0.4408 | −0.0019 [−0.0055, +0.0018] |
| softmax mean | 0.3030 | −0.0730 [−0.0984, −0.0475] | 0.4012 | −0.0415 [−0.0553, −0.0277] |

- **Row-z stands.** It is best on the candidate and every alternative loses there. Raw
  averaging is dominated by whichever member has the larger margin scale.
- **Min-max is indistinguishable, not better:** 2e-6 on the fallback with a CI spanning zero,
  and worse on the candidate. No weight, config or packaged artifact changed.
- Softmax is a monotone-transform **diagnostic only** — LinearSVC margins are not probabilities.
- Adoption bar is unchanged (alternative seed s13 + nested outer gate); nothing here clears it.
  Nine tests in `tests/test_blend_rules.py` pin the transforms. Details: report §5.1.

### Inference cost at full test size (D043)

`python experiments/inference_benchmark.py --sizes 4000 8000 --data-dir data` times the
**packaged** candidate in a fresh process running the notebook's exact path (packaged `src` on
`sys.path`, `PYTHONPATH` cleared, offline env, `load_final` → `load_weights` → crossed guard →
`predict_dataframe` → `build_submission` → parquet read-back + schema/id/label/null asserts).
Rows are **synthetic** (k≈2 passages/prompt); nothing is fit and no competition row is scored.

| synthetic rows | cold start | bundle load | predict | write+validate | peak working set | headroom vs 9 h |
|---|---|---|---|---|---|---|
| 4,000 | **13.77 s** | 0.37 s | 11.14 s | 0.07 s | **0.370 GB** | ≈2,350× |
| 8,000 (2× stress) | 28.24 s | 0.34 s | 25.21 s | 0.11 s | 0.542 GB | ≈1,150× |

- Predict time is linear in rows and memory stays under 0.6 GB at twice the expected size, so
  neither the 9-hour CPU budget nor the RAM ceiling is close to binding. A repeat 4,000-row run
  took 15.58 s → **~14–16 s** with machine noise.
- Fixture passages are **longer** than official ones (mean 190.1 vs 114.7 words, 1.66×), so the
  timing is conservative for official-length text.
- **Local Windows/CPU measurement, not a Kaggle-runtime benchmark.** It bounds the work, not the
  platform; the graded rerun on the full hidden test is still owner-only. `peak_wset` is a
  Windows counter (Linux: sampled RSS 0.220 / 0.234 GB).
- The synthetic prediction distribution is a runtime artefact of fixture styles — **not** an
  accuracy or class-prior result. Five tests in `tests/test_inference_benchmark.py`. Report §9.

### Class-intercept tuning (D044 — negative)

`python experiments/class_bias.py --data-dir data --seeds 42 13` searches an 18-vector of
per-class intercepts on the adopted row-z blend under **nested separation**: per outer fold,
3 inner grouped folds on outer-train rows only, both members refit on inner-train (30 refits
per seed), bias tuned on inner OOF, outer-valid scored once with the frozen bias.

| seed | no bias | analytic −log(prior) | nested-tuned | paired Δ (95% CI) | inner gain | optimism gap |
|---|---|---|---|---|---|---|
| 42 | **0.3760** | 0.3756 | 0.3663 | **−0.0097** [−0.0248, +0.0055] | +0.0122 | +0.0218 |
| 13 | **0.3582** | 0.3582 | 0.3634 | +0.0052 [−0.0144, +0.0248] | +0.0117 | +0.0065 |

- **Not adopted.** The rule is that every seed must improve with a paired CI above zero; s42 is
  negative and neither CI excludes zero. The searched intercepts fit the *inner* models' class
  offsets, which the outer models do not share — the inner gain evaporates or reverses.
- The analytic prior offset is a near no-op because official train is balanced (support 149/150,
  ratio 1.01). Shrinkage picked λ=1.0 on 9 of 10 folds, so it did not rescue the search either.
- Subject is the official-only fallback; the pool-MIX char member was not re-measured here.
  No config, weight or package change. 12 tests in `tests/test_class_bias.py`. Report §7.1.

### Style-feature associations (D046)

`python experiments/style_features.py --data-dir data` refits `style_only` on each frozen fold
and first proves the refit **is** the recorded model: max absolute drift from the saved
identity-verified OOF is **0.0** on all five folds. 56 features, standardised inside each fold.

| feature | class-averaged \|coef\| | sign agreement |
|---|---|---|
| `rate_dash` | **1.3816** | 1.00 |
| `rate_nonascii` | 0.6182 | 0.98 |
| `length_sentences` | 0.5819 | 0.91 |
| `rate_colon` | 0.4973 | 0.99 |
| `rate_squote` | 0.4954 | 0.93 |

- `rate_dash` is positive for every `claude-*`/`glm-*` label (claude-opus-5 +1.961) and strongly
  negative for gemini-3.6/3.7/3.8-flash (−2.010 / −1.804 / −2.017): one punctuation habit splits
  providers, stably across folds.
- The weakest style-only recalls — claude-fable-5 **0.074**, glm-5.3 **0.087** — lean on the same
  two features as their better-recalled siblings, which is why style alone cannot separate them.
- **Associational, not causal:** correlated style columns share credit arbitrarily. No feature was
  added, removed or reweighted. 7 tests in `tests/test_style_features.py`. Report §8.3.

### Near-duplicate audit of the official corpus (D047)

`python experiments/near_duplicates.py --data-dir data` measures whether group separation is
real. Same word TF-IDF settings and 0.3 threshold as the pool screen, so internal and external
audits are comparable. Fits no predictor, reports no score, reads no test row.

| cosine threshold | unordered pairs | cross-fold | same-prompt |
|---|---|---|---|
| > 0.3 | 2,190 | **0** | 2,190 |
| > 0.5 | 34 | **0** | 34 |
| > 0.7 | 0 | 0 | 0 |
| > 0.9 | 0 | 0 | 0 |

3,640,951 pairs examined over 2,699 rows / 150 prompts; 0 exact duplicate texts, 0 prompts
spanning more than one fold, max off-diagonal cosine **0.6223**. All the similarity that exists
is the shared question vocabulary of the 18 answers to one prompt, and the grouped split keeps
every such pair on one side. Stores ids/fold numbers/cosines only, never passage text. Similarity
is a heuristic, not proof every augmented variant was found. 6 tests in
`tests/test_near_duplicates.py`. Report §4.

### Per-prompt-group error concentration (D048)

`python experiments/prompt_groups.py --data-dir data` groups the same saved OOF rows into their
150 prompts (17–18 rows each). Because ~18 rows is noisy, every concentration statistic is
compared against a **binomial null** (same row counts, errors drawn at the observed overall rate,
2,000 draws, fixed seed). No fit, no test row.

| variant | prompt acc mean ± std | worst-15 share of errors | null mean [p05, p95] |
|---|---|---|---|
| official blend | 0.3761 ± 0.1209 | **0.1336** | 0.1305 [0.1269, 0.1344] |
| pool-MIX cap40k | 0.4427 ± 0.1234 | **0.1396** | 0.1357 [0.1315, 0.1404] |

- **No hard-question core.** Every variant sits inside its null interval; zero prompts score 0 and
  zero score perfectly. No prompt collapses to one label (10.21 / 11.18 distinct predictions per
  prompt out of 18), and no prompt sends all its errors to a single wrong answer.
- **The pool gain is broad on a third unit:** paired per-prompt delta **+0.0666
  [+0.0497, +0.0836]**, **96 improved / 28 worsened / 26 unchanged** (rows and folds already agreed).
- Prompt-level length correlation is positive but modest (Spearman ρ +0.169 official, +0.239 pool),
  so §8.2's row-level length gradient is only partly a prompt effect.
- Errors are a **per-passage** problem, which argues against prompt-difficulty-targeted remedies.
  28 tests in `tests/test_prompt_groups.py`. Report §8.4.

### Style-only classifier families (D045/D049 — logreg kept)

Plan §4 asks for style-only logistic regression, SVM and a small GBDT. All are now measured on
the same 56 per-fold-standardised features and the same five registered folds, plus CatBoost
once it became installed (it was absent when D020 was written):

| classifier | pooled OOF | macro F1 | fit s | paired Δ vs logreg (95% CI) |
|---|---|---|---|---|
| **LogisticRegression (adopted)** | **0.2904** | 0.2792 | 29.6 | — |
| LinearSVC (`configs/style_svm.yaml`) | 0.2844 | 0.2574 | 26.6 | −0.0060 [−0.0145, +0.0026] |
| CatBoost (`configs/style_catboost.yaml`) | 0.2830 | 0.2754 | 115.0 | −0.0074 [−0.0212, +0.0064] |
| HistGradientBoosting (D020) | 0.2630 | 0.2568 | 74.2 | — |

- **No family beats logreg**, and every CI includes zero: "no evidence of a gain", with negative
  point estimates. Error profiles are no better either (within-provider confusions 876 SVM /
  888 CatBoost vs 873 logreg). The representation matters more than the classifier.
- **CatBoost adds no blend diversity.** Its row-z scores correlate **0.8678** with the logreg
  style member; swapping it in gives 0.3652 (−0.0107 [−0.0226, +0.0013] vs the adopted blend)
  and an equal-thirds blend gives 0.3686 (−0.0074 [−0.0220, +0.0073]). A third correlated
  member dilutes rather than diversifies.
- CatBoost is **optional**: lazily imported in `src/model.py`, not in `requirements.txt`, and
  `tests/test_model_catboost.py` (12 tests) skips without it. It exposes no `decision_function`,
  so its scores are `log(predict_proba)` — a real probability model, unlike a LinearSVC margin.
  Details: report §7, ADRs D045/D049.

### Neural and tree-based models (negative results)

| model | official only | + external pool | verdict |
|---|---|---|---|
| GBDT, style only | 0.2630 | — | ≪ 0.376 blend, no blend gain (D020) |
| CatBoost, style only | 0.2830 | — | below logreg 0.2904; row-z corr 0.868, no blend gain (D049) |
| GBDT, char → SVD-200 | 0.2537 | — | same |
| GBDT, multi-view SVD-128 | 0.2426 | — | same |
| DeBERTa-v3-base, len 256, 1 ep | 0.2241 | 0.3049 | pool gain real (+0.081) but below blend (D022) |
| DeBERTa-v3-large, len 384 | 0.2482 | 0.3857 | transductive pool arm — diagnostic only (D023/D026) |
| DeBERTa-v3-large, identity-verified | 0.1915 | — | every sparse+DeBERTa blend ≤ sparse, CIs span 0 (D027) |

The 0.4442 `0.5·sparse + 0.5·DeBERTa-pool` blend is **superseded**: its pool arm used the
disabled per-fold screen, and the identity-checked official-only rerun showed the encoder
alone adds no diversity gain — only the pool did. The GPU kernel is now re-wired onto the
non-transductive global screen at cap 40,000 with `FOLD_SEED = s42` (D038), but **no
DeBERTa-pool metric exists yet**; that run is owner-side and must beat 0.4427 under all
three protocols before adoption.

### Crossed-design candidate (D016 — WITHDRAWN by D017, k=18 artifact)

> **Withdrawn as a submission default (ADR D017; reframed D018).** The gains below
> were measured at **k=18 passages per prompt** (the train design). Hidden-test
> multiplicity is unverified; under a hypothetical **k≈2**, the paired gain shrinks to a
> marginal +0.0068 [+0.0032,+0.0104] on cv_s42 and +0.0029 [−0.0059,+0.0117] on the
> examined holdout (CI spans 0). Do **not** submit this candidate. Damped
> Sinkhorn is also transductive and is not permitted in reported scores. The
> primary metric for the argmax blend is the **full grouped OOF** (0.3760 cv_s42).
> See `docs/strategy_review.md` and `experiments/stress_k.py`.

The training set is a balanced full factorial (150 prompts × 17–18 of the 18
models). Whether the hidden test mirrors this is unverified, so the mechanism
below is **research-only**: passages answering the same
prompt can be recovered by semantic clustering and constrained to **one passage
per model**. `src/crossed.py` (Hungarian assignment) + `src/grouping.py` (MiniLM
agglomerative clustering) + `src.inference.predict_dataframe_crossed` implement
this label-free, transductive inference step. The blend members and weights are
unchanged.

k=18 measurements (historical — **not** deployable k≈2 estimates):

| split | argmax blend | recovered-group assignment (cosine thr 0.4) | gain |
|---|---|---|---|
| examined holdout (25 prompts, seed 7) | 0.3956 | **0.5000** | +0.1044 |
| grouped CV seed 7 | 0.3608 | 0.4716 | +0.1108 |
| grouped CV seed 13 | 0.3582 | 0.4657 | +0.1075 |
| grouped CV seed 42 (reference) | 0.3760 | 0.4704 | +0.0944 |

The oracle ceiling (known prompt groups) is 0.4853 on seed 42, so recovered
grouping captures ~88–90% of it. The gain is positive on every split; threshold
0.4 keeps the highest cluster purity at scale (0.960 on all 2,699 passages / 150
prompts). Historical transductive results do not satisfy the repository hard
rule against transductive steps in reported scores (D024). The crossed pipeline was verified to load the attached MiniLM
checkpoint fully offline and run at 2,699-row scale in ~56 s on CPU, producing a
near-uniform label distribution (max class share 0.056 ≈ 1/18) — the expected
full-factorial signature. See `docs/improvement_plan.md` §5 and ADRs D014–D016.

### Synthetic fixture (superseded pipeline check)

The fixture run is kept only to prove the pipeline works without competition
data; its numbers (char+style logreg 0.4823, style-only 0.4723, char 0.2061) are
artifacts of synthetic style profiles and are **not** a performance estimate. It
lives in `artifacts/cv/synthetic/` and is labelled `synthetic` in the registry.

## Repository layout

```text
configs/            baseline, features (ablations), style_svm, style_catboost, embeddings, frozen_encoder, ensemble, submission
data/               ignored; official CSVs or the synthetic fixture
docs/               spec, protocol, audit, external-data register, decisions (ADRs)
src/
  constants.py      verified labels, metric, paths
  data.py           schema validation, loading, fingerprint
  validation.py     prompt-grouped folds, leakage guards, split persistence
  metrics.py        balanced accuracy, per-class recall, length bands, diagnostics
  features_tfidf.py char/word/char_wb TF-IDF + composition, SVD, scaling
  features_style.py deterministic stylometry (56 features)
  features_embedding.py optional frozen encoder (hashing / sentence-transformers)
  prompt_invariance.py  deployable nuisance suppression + diagnostic centring
  pool.py           external-pool loading, global decontamination, stratified/balanced caps
  oof_identity.py   OOF provenance: dataset hash + fold ids, alignment guards
  model.py          ModelSpec, ModelBundle (save/load, class-order guard)
  baselines.py      majority, stratified random
  train.py          CV loop, OOF, registry, summary
  ensemble.py       greedy OOF blending on normalised margins
  nested_validation.py  outer prompt-grouped folds for the final gate
  stability.py      alternative-seed + secondary-holdout robustness checks
  inference.py      final refit (optionally pool-augmented) and prediction
  crossed.py        within-prompt centering + one-per-model Hungarian assignment
  calibration.py    k-robust damped Sinkhorn-Knopp calibration (no grouping)
  grouping.py       MiniLM embedding clustering + prompt-group purity metrics
  submission.py     parquet writing + validation
  audit.py          data audit, report, class-balance/length figures
  figures.py        report figures from saved OOF/metrics (confusion, recall, ablation, length bands)
  config.py         YAML config loading; constants.py also holds paths/labels
  cli.py            audit / splits / train / ensemble / evaluate / figures / stability / predict / package / demo
tests/              unit and integration tests
figures/            generated report figures (PNG)
kaggle/             submission.ipynb (offline, attached-artifact); deberta/ GPU kernel
experiments/        registry.csv + pool_train, nested_pool_blend, blend_deberta, length_sensitivity, blend_rules, inference_benchmark, class_bias, style_features, near_duplicates, prompt_groups, stress_k, audits
report/             research_report.md
```

## Compliance posture

- No competition rows are sent to any assistant or external service; only code,
  configs, aggregates and schema.
- No runtime network access, no third-party inference, no paid APIs.
- The **submitted** argmax blend (public LB 0.356) is trained from scratch on the
  official training folds and uses no external data.
- The **current candidate** additionally trains its char member on one public external
  corpus, **Open Pool v15** (`crtproduction/name-that-llm-open-pool`, CC BY-SA 4.0),
  which the competition permits. It is screened once against the full official train and
  the keep-mask is fold-independent (D028), so no held-out passage influences which pool
  rows are trained on. Pool rows are used **only at local fit time**: the uploaded package
  ships fitted bundles, `weights.json`, `manifest.json` and `src/` — no pool rows, no
  competition data, no runtime download. Provenance, licence and validation are recorded
  in `docs/external_data_register.md`.
- The withdrawn **crossed-design candidate** (D016) attached one public encoder (MiniLM,
  Apache-2.0) **offline** for prompt-group clustering only — never as an authorship
  feature, never downloaded at runtime. It was benchmarked for H3 authorship and not
  adopted for that purpose (D013).
- Data is never committed (`data/`, `artifacts/` ignored).
- Ambiguous compliance questions are logged in `docs/competition_spec.md` §10 and
  need owner approval before the final submission.

## Known limitations

- The only **confirmed leaderboard** number is the first submission's **0.356 public LB**.
  Every other score in this README is local grouped CV, including the candidate's
  0.4427 / 0.4304 / 0.4426. The candidate is packaged but **not submitted**; uploading and
  submitting are owner-only. If its public LB lands at or below 0.356, revert to the
  official-only fallback and record the disagreement.
- The hidden test's group structure, prompt multiplicity and class distribution remain
  unknown. The crossed-design candidate is withdrawn; its historical transductive scores
  are not protocol-compliant evidence, and neither are the per-fold-screen pool arms
  (DeBERTa 0.3857, the 0.4442 blend).
- The fallback's blend weights were selected on the same out-of-fold predictions they are
  scored on, so that gain is mildly optimistic. It was re-checked on alternative fold seeds
  (7, 13) and the secondary prompt-held-out split and is positive on all (+0.0452, +0.0441,
  +0.0222; report §6.5). The **candidate's** 0.5/0.5 rule is pre-specified, not re-selected,
  and its nested gate refits inside outer folds, so it carries no CV-selection optimism.
- The reserved seed-2027 holdout turned out to be **examined** (it participated in earlier
  full-OOF development), so it is not a clean final gate; the nested outer-fold protocol
  replaces it (D035). No fully untouched holdout remains.
- The external pool is community-generated and its distribution may not match the hidden
  test. A larger cap helps on CV but could over-fit that distribution; the pool's
  `prompt_canon` namespace is still not proven equivalent to the official `prompt_id`, so
  decontamination is text-level only (stricter, but no prompt crosswalk is claimed).
- Class-proportional capping starves `gemini-3.7-flash`; the balanced-cap fix improves char
  alone but regresses the blend, so the candidate keeps a known per-class weakness
  (D036/D037). No per-class prior was tuned on CV.
- Prompt-topic shift between train and hidden test can exceed what grouped CV simulates,
  and the public/private model mixes differ.
- `char+style` sparse concatenation did not converge at 100k features; the idea is
  unresolved rather than refuted.
- Deployable prompt-nuisance suppression (k=8) reduced pooled OOF on official data and is
  not adopted; only one k was tested (report §6.4, ADR D010).
- A frozen MiniLM encoder was benchmarked (H3) and not adopted: 0.1278 alone, and a +0.0026
  greedy blend gain that reverses on the holdout (report §6.6, ADR D013). Only one small
  encoder was tested. DeBERTa-v3 was tested at base and large; the identity-verified
  official-only large run adds no blend gain (D027), and the non-transductive pool rerun
  has not been scored yet (D038).
- The synthetic fixture is not a proxy for competition difficulty.
