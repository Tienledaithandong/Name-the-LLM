# Final submission checklist (owner actions)

> **2026-11-18 official-data run:** Kaggle submission **58502550** scored
> **0.356** public balanced accuracy, matching submission 58344764. The private
> score is pending. This run used version 1 of
> [Name That LLM Official Sparse Submission](https://www.kaggle.com/code/tiennamlee/name-that-llm-official-sparse-submission).
> It refits the same 50/50 blend on Kaggle from the attached official training
> data using the verified offline scikit-learn 1.9.1 wheel. The Open Pool
> package described below is outside this run's scope. Its global mask was
> selected using the full official corpus, including rows held out for CV,
> so the reported pool gain does not establish eligibility under the project's
> no-transductive-scoring rule.

> **2026-11-16 — candidate ready, submission still owner-only (D037/D040).** The
> upload-ready package in `artifacts/kaggle_dataset` is the **cap-40k pool-MIX blend**:
> `0.5 × char_tfidf_logreg (official train + globally decontaminated external pool)` +
> `0.5 × style_only (official-only)`, row-wise z-normalised. Local grouped OOF **0.4427**
> (s42) / **0.4304** (s13) / **0.4426** (nested outer gate) — **these are CV numbers, not
> leaderboard scores.** The earlier 0.4442 DeBERTa+sparse candidate is **superseded and
> must not be submitted**: its pool arm used the transductive per-fold screen (D026) and
> official-only DeBERTa adds no diversity gain (D027). The confirmed fallback is still the
> official-only argmax blend at public LB **0.356**. Hidden-test prompt multiplicity remains
> unverified; the seed-2027 subset is **examined**, not a clean gate (D035).


Account-sensitive steps below are **owner-only**; the agent prepares artifacts
and instructions but does not accept rules, upload, or submit. Everything else
is already reproducible from this repository.

## Verified evidence to date (local CV, not leaderboard)

> **Withdrawn (ADR D017; reframed D018):** the crossed Hungarian candidate below
> was validated at **k=18** passages/prompt. The k-subsampled **stress test**
> (`experiments/stress_k.py`) tests a hypothetical **k≈2**, where the Hungarian
> paired gain (method − argmax) is marginal: +0.0068 [+0.0032,+0.0104] on cv_s42 and
> +0.0029 [−0.0059,+0.0117] on the examined holdout (CI spans 0). **Do not submit
> the crossed candidate.** The safe baseline is the **argmax blend** (`make predict`
> / `make package`, no `--crossed`); its primary metric is the full grouped OOF
> **0.3760**. See `docs/strategy_review.md`.

- Safe baseline: **argmax blend** `0.5 × char_tfidf_logreg + 0.5 × style_only`
  (`configs/submission.yaml`), **primary full grouped OOF 0.3760** (cv_s42) /
  0.3956 (examined holdout); k≈2 stress-test argmax ~0.36–0.39; submitted public LB
  **0.356**. Official top **0.771**.
- **Current candidate (D028/D037): cap-40k pool-MIX.** Same fixed 0.5/0.5 row-wise-z
  rule, but `char_tfidf_logreg` is trained on official train **+ the globally
  decontaminated external pool** (Open Pool v15, CC BY-SA 4.0; screened once against the
  full official train, 226,191 → **226,049** clean, capped at **40,000**/fold).
  `style_only` stays official-only (the pool hurts it: −0.0174, CI spans 0).
  Grouped OOF **0.4427 s42 / 0.4304 s13 / 0.4426 nested outer gate**; char alone
  0.3193 → **0.4104** (s42). Paired vs the 12k predecessor: +0.0437 [+0.0238,+0.0636] s42,
  +0.0315 [+0.0089,+0.0540] s13, +0.0908 [+0.0460,+0.1355] nested — all folds positive.
  Independently reproduced from identity-verified OOF by `experiments/blend_deberta.py`
  (D039): 0.442675 / 0.430435, official-only control 0.375971.
  The screen is **fold-independent**, so it is not the transductive route D026 disabled.
- **Rejected variants (do not submit):** class-balanced pool cap at 12k (blend 0.3949) and
  at 40k (blend 0.4378 < 0.4427) — both fix gemini-3.7-flash starvation but lose more on
  its siblings (D036/D037). DeBERTa-v3 official-only identity run **0.1915**, and every
  fold-aligned sparse+DeBERTa blend is at or below sparse with CIs spanning 0 (D027).
  The 0.4442 pool-DeBERTa blend used the disabled per-fold screen (D026) — diagnostic only.
- **Open lever, no metric yet (D038):** `kaggle/deberta/deberta_train.py` is re-wired onto
  the global screen at cap 40,000 with `FOLD_SEED = s42`. The GPU run is owner-side; score
  its output with `experiments/blend_deberta.py --deberta-dir <out>/deberta_cv --variant
  official_pool --char-oof artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz
  --style-oof artifacts/cv_identity/official/oof/style_only.npz` and adopt only if
  `mix+deberta_*` beats `mix_baseline` with paired CI excluding 0 under all three protocols.
- **Second open lever, no metric yet (H8 / D054):** the same kernel can now train
  **ModernBERT-large** — set `BACKBONE = "modernbert-large"` at the top of
  `kaggle/deberta/deberta_train.py` and push. That constant, not `--model-name`, is what a
  pushed run uses, because Kaggle **script** kernels receive no CLI arguments.
  `transformers>=4.48` is required (already updated in `requirements_gpu.txt`). The budget
  stays 384/4/1e-5 so only the backbone changes. Score it with the same
  `experiments/blend_deberta.py` command and the same three-protocol bar; restore
  `BACKBONE = "deberta-v3-large"` before any D038 run so the arms stay distinguishable.
- **Tested and rejected (H9 / D053):** hard typographic family vetoes. +0.000741 on the
  candidate with a paired CI of [−0.001316, +0.002797] that includes 0. Nothing to submit;
  `src/typography.py` is deliberately **not** shipped in the package.
- Crossed inference (D016, **withdrawn**): recovered-group Hungarian assignment on
  blend OOF. k=18 holdout 0.5000 / seeds 0.4716 / 0.4657 — **a k=18 artifact**;
  at k=2 the paired gain is marginal (cv_s42 +0.0068 CI[+0.0032,+0.0104]; holdout
  +0.0029 CI[−0.0059,+0.0117], spans 0). Stress test is inference-only, not a
  selection score.
- Damped Sinkhorn calibration (`src/calibration.py`) is transductive and is
  excluded from reported scores and submission under the current hard rule.
- Blend-only (argmax) pooled OOF balanced accuracy **0.3760**, +0.0567 over the
  best single model, positive in all five folds (k=18 fit diagnostic).
- Alternative seed 7 reproduces the blend gain (+0.0452); full table in
  `artifacts/stability/official/stability.md` (regenerate with `make stability`).
- Prompt-nuisance suppression (H2) is a negative result and is **not** in the
  submission (report §6.4, ADR D010).
- Kaggle submission 58344764, notebook version 2, completed on 2026-11-14 with
  public balanced accuracy 0.356. The private score and final rank are not yet
  available. The notebook emitted scikit-learn version mismatch warnings (trained
  with 1.9.1; Kaggle runtime 1.6.1).

## Before submitting

1. Accept the competition rules and confirm entry/team deadlines on the official
   pages (entry 2027-01-06, final 2027-01-13, 23:59 UTC; team ≤ 5; 5
   submissions/day; 2 final submissions). Re-read `docs/competition_spec.md`
   against the live pages; reopen ADR D001 if anything changed.
2. Confirm the blend gain is still positive on the secondary holdout and the
   alternative seeds (`make stability`). If it is not, submit the best single
   model instead (ADR D009) by editing `configs/submission.yaml` to that member
   and re-running `make predict`.
3. Resolve any open compliance question in `docs/competition_spec.md` §10.

## Build and upload the artifact

**Current candidate (cap-40k pool-MIX, D037):** already built in
`artifacts/kaggle_dataset` (10 MB: two bundles, `weights.json` 0.5/0.5, `src/`,
`manifest.json` with pool scope `global_full_official_train` and
`validation_independent: true`). To rebuild from scratch:

```bash
python -m src.cli predict --config configs/submission.yaml --data-dir data --fit \
  --pool artifacts/pool_audit/ntl_open_pool.parquet \
  --pool-models char_tfidf_logreg --pool-cap 40000
python -m src.cli package --dataset official --out artifacts/kaggle_dataset --kaggle-user <you>
kaggle datasets create -p artifacts/kaggle_dataset        # owner-only
```

`weights.json` deliberately carries **no** pooled-OOF number for this package: the
official-only CV fields are stripped because the char bundle was refit with the pool
(D040). The score of record is `experiments/registry.csv`.

**Licence (D050) — do not change it.** `dataset-metadata.json` declares
**CC-BY-SA-4.0** and the package README carries the attribution block (licence link,
competition credit, Open Pool v15 credit for the pool-fitted char member). The weights
are a derived adaptation of CC BY-SA 4.0 text, so share-alike applies; the earlier
CC0-1.0 declaration was wrong and is fixed. Upload it as-is — do not relabel it CC0.
Re-running `cli package` refreshes `src/` **without refitting**: the bundles are copied
byte-for-byte from `artifacts/final/official`. The refreshed package was re-validated
offline from its own shipped `src` (4,000 synthetic rows, 9.40 s / 0.370 GB peak,
schema/id/label/null checks passed; `artifacts/inference_benchmark_refresh/results.json`).

**Confirmed fallback (official-only, public LB 0.356):** omit `--pool`/`--pool-models`
from the predict step, then `make package`. No encoder dependency either way.

**Crossed candidate (D016): withdrawn; do not submit.** Earlier results used
transductive processing and do not satisfy the repository reporting rule. The
hidden-test prompt multiplicity and class distribution are not known, so a
near-uniform prediction histogram is not a valid acceptance criterion.

**DeBERTa blends: do not submit.** Official-only DeBERTa adds no gain (D027) and the
0.4442 pool blend used the disabled transductive screen (D026). A new pool run must first
clear the D038/D039 evaluation above.

## Run the Kaggle notebook

4. Attach the competition data and the currently approved baseline artifact.
5. Disable Internet, run all cells, and verify schema, IDs, labels, nulls, and
   model dependency versions. The interactive test file contains example rows;
   Kaggle supplies hidden rows during the graded rerun.
6. For any future DeBERTa+sparse blend, first run the D038 global-scope kernel and clear
   the D039 evaluation (`mix+deberta_*` must beat `mix_baseline` with paired CI excluding 0
   under s42, s13 and the nested gate), then use the separately prepared offline candidate
   package with verified real weights and measured runtime. Never upload dummy weights or
   fixture submissions.

## Submit and record

11. Submit the notebook output (owner-only):

    ```bash
    kaggle competitions submit -c name-that-llm -f submission.parquet \
      -m "pool-MIX cap40k, non-transductive global screen (D028/D037)"
    ```

    Record in `experiments/registry.csv`: submission version, git commit, CV score,
    **public LB score**, timestamp and changes. Keep public LB separate from CV; never
    select on public LB alone. **If the public LB is ≤ 0.356, revert to the official-only
    fallback** (rebuild without `--pool`) and record the CV/LB disagreement as an ADR —
    a grouped-CV gain that does not transfer is evidence about prompt shift, not a reason
    to keep submitting the same candidate.
12. Use the two final submissions deliberately (e.g. the blend and a fallback
    single model), not to chase public-LB noise.
13. Only after the private leaderboard is published, record the final rank /
    percentile in `README.md` and the report CV bullet. Until then leave those
    placeholders unconfirmed.

## Hard rules (do not break)

- Never commit `data/`, `artifacts/`, OOF files or secrets.
- Never paste competition rows into hosted assistants or third-party APIs.
- No runtime downloads or external inference in the notebook.
- Do not tune on the hidden test or the secondary holdout.
