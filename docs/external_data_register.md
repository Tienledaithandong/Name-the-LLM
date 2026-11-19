# External data / model register

> **2026-11-16 audit correction (D024):** the historic pool counts and OOF
> results below are provisional, not proof of prompt isolation. The official and
> pool prompt identifiers have no verified crosswalk, and 1,533 pool rows lack a
> usable canonical prompt ID. Pool CV is disabled (D026): filtering pool rows by
> held-out text is transductive. A complete reviewed crosswalk remains necessary
> but is not sufficient to enable a new non-transductive protocol. **Partly superseded by D028 (2026-11-17): the adopted text-level, fold-independent screen does not require a crosswalk — see the banner below.**

> **2026-11-17 status update (D028/D035/D037/D039): the pool route is RE-OPENED
> non-transductively.** `src.pool.decontaminate_global` screens the pool **once**
> against the complete official train (exact text + word-TF-IDF cosine > 0.3, the
> vectorizer fit on that same train): 226,191 -> **226,049** clean (0 exact, 142
> near). One fold-independent keep-mask is reused by every fold, so no held-out
> passage selects training rows. The *per-fold* screen in the historic row below
> is **deprecated and diagnostic only** (D026). The cap-40k MIX candidate
> `0.5 z(char trained with pool) + 0.5 z(style official-only)` is validated at
> **0.4427** (s42) / **0.4304** (s13) / **0.4426** (nested outer gate) and is
> **packaged, not submitted**; the confirmed public LB **0.356** fallback remains
> official-only. Evidence: `artifacts/pool_cv_cap40k/official/results.json`
> (`decontamination_scope: global_full_official_train`,
> `validation_independent: true`, `kept: 226049`, `capped: 40001`) and the `pool`
> block of `artifacts/kaggle_dataset/manifest.json`. The prompt crosswalk is still
> unverified and is **not** required by this text-level protocol (see below).


Rules basis: Code Requirements allow "freely & publicly available external data,
including pre-trained models"; Rules §2.6 require external data to be publicly
available and equally accessible at no cost, or "reasonably accessible" at
minimal cost. Nothing may be downloaded at submission time (internet disabled).

## Register

**The submitted argmax blend (D009/D012, public LB 0.356) uses no external data or
pretrained model** — it is sparse TF-IDF + stylometry, fitted from scratch on the
official training folds. The **cap-40k pool-MIX candidate** (D028/D037) adds one
external *text corpus* (Open Pool v15) to the char member and still ships **no
encoder**. Four public checkpoints are registered below for provenance: one frozen
encoder benchmarked for H3 authorship (MiniLM, **not adopted**), two encoders
**fine-tuned on Kaggle GPU** for the neural-baseline hypothesis (DeBERTa-v3 base and
large, **not adopted** — D023/D027), and one **frozen local pilot**
(ModernBERT-large, D057). **No submission path depends on any of them.**

**Withdrawn for grouping (2026-11-14, ADR D017; reframed D018):** the same MiniLM
encoder was used to *recover prompt groups* by semantic clustering for the
crossed-design assignment pipeline (`src/grouping.py`,
`src.inference.predict_dataframe_crossed`). That candidate is **WITHDRAWN as a
submission default**. Its apparent gain was a **k=18 validation artifact**: the
hidden test is believed to have ~1–3 passages/prompt (k≈2), where the
recovered-Hungarian paired gain collapses to a marginal, split-dependent
+0.003…+0.007 (95% CI spans 0 on the examined holdout). The earlier "0.5000 vs
0.3956 on the untouched holdout" was a **k=18** measurement on what is now an
**examined** (not untouched) holdout — not a deployable k≈2 estimate
(`experiments/stress_k.py`, `artifacts/stress_k/results.json`, registry `stress_*`).
`predict --crossed` / `package --crossed` (which attach `encoder/minilm` +
`inference.json`, loaded offline with `HF_HUB_OFFLINE=1`) remain available as
**research tools only** and are **not** the recommended submission path. The
submitted argmax blend depends on **no external data or checkpoint**; MiniLM is
registered below for provenance only.

| id | source URL | license | accessed | what it is | labels / model mapping | size | dedup vs train | used in | validation result |
|---|---|---|---|---|---|---|---|---|---|
| minilm-l6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | 2026-11-14 | frozen sentence encoder (384-dim); grouping only, not a feature extractor for authorship | n/a — no labelled text; maps passages to vectors for clustering, not to the 18 labels | ~90 MB checkpoint (88 MB snapshot, single `model.safetensors`) | n/a — no external text corpus added | H3: `minilm_embedding_logreg`, `minilm_embedding_style_logreg` (registry, seed 42); **withdrawn crossed research: prompt-group clustering** (`phase_b_*`/`stress_*` registry rows; `package --crossed` → `encoder/minilm`, research-only) | **Authorship: not adopted** (H3). Embedding-only 0.1278 pooled OOF; +style 0.3093. Greedy char+emb_style 0.3786 CV (+0.0026) reverses on the holdout (0.3933 vs 0.3956); report §6.6, ADR D013. **Grouping: WITHDRAWN** (D017/D018). The k=18 holdout 0.5000 (+0.1044) was an artifact; at k≈2 the paired gain is +0.003…+0.007 (95% CI spans 0 on the examined holdout). Research-only; the submitted argmax blend has no checkpoint dependency. |
| open-pool-v15 | https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool | CC BY-SA 4.0 (65 sources: CC BY 4.0 / CC BY-SA 4.0 / CC0) | 2026-11-14 | 226,191 clean community passages (v15, 2026-11-14), 30 cols; labels are publishers' claims, not re-verified | **all 18 official labels, exact match** (`per_model.csv`); uneven 3,541 (`gemini-3.7-flash`) … 23,474 (`gemini-3.8-flash`) — use class weights/sampling | 112 MB `ntl_open_pool.parquet` (226,191 rows); + `prompts.parquet`, `gemini37_unverified.parquet` (10,838 excluded) | dataset `max_cos_train` ≤ 0.30, 0 exact text dups vs train, train-prompt answers (284) + mislabeled (284) already removed by the publisher. **Adopted screen (D028): global, fold-independent** — exact text + word-TF-IDF cos > 0.3 against the *full* official train → 0 exact + **142** near removed, **226,049** kept; then `cap_pool_stratified` 40,000/fold class-proportional (**40,001** rows after per-class rounding), a validation-independent compute budget. *Historic per-fold screen (D026, deprecated): 19 rows union → 226,172 — diagnostic only, never used in a reported score.* | **Cap-40k MIX candidate** (char member only): `experiments/pool_train.py` → `artifacts/pool_cv_cap40k/official/` (+ `_s13`), nested gate `artifacts/nested_pool_blend_cap40k/`, independent reproduction `artifacts/mix_baseline_verify_cap40k*/`; shipped by `cli predict --fit --pool … --pool-models char_tfidf_logreg --pool-cap 40000` → `artifacts/final/official` → `cli package` → `artifacts/kaggle_dataset`. Phase 1b/1c official-only vs official+pool (DeBERTa #31, GBDT #32); audit `artifacts/pool_audit/audit.json` + `clean_pool_ids.json`; `src/pool.py`, `experiments/pool_audit.py`. **Training-only: the pool parquet is NOT in the submission package** — only the two fitted bundles. | **Validated, packaged, NOT submitted** (D028/D035/D037/D039; supersedes the D019 "admission pending" status). Same official grouped folds, same fixed blend rule: char official-only **0.3193 → +pool 0.4104** (paired **+0.0912 [+0.0721, +0.1103]**, all five folds positive); MIX **0.4427** (s42) / **0.4304** (s13) / **0.4426** (nested) vs the official-only control 0.3760 / 0.3582 / 0.3519. **Harm is recorded too:** style+pool 0.2904 → **0.2730** (paired −0.0174 [−0.0480, +0.0132], CI spans 0) so style stays official-only; under the pool `gemini-3.7-flash` recall falls **−0.207** and `kimi-k3` **−0.093**. No leaderboard number exists for this candidate. Verified: `flag_gemini37_routing_risk`/`flag_near_train_prompt`/`flag_train_prompt_regen` **all False** (routing-risk rows already in `gemini37_unverified.parquet` — do NOT re-filter that flag); labels complete; 0 null text. Caveats: `match_weight`/`style_score` fitted on **all** train prompts → refit per-fold or avoid for honest CV; `flag_kb_proxy_after_ruling` 20,227 rows (hosts allow training on public datasets made via proxy; optional strict drop); CC BY-SA share-alike on derived artifacts. |
| deberta-v3-base | https://huggingface.co/microsoft/deberta-v3-base | MIT | 2026-11-15 (fetched by the Kaggle GPU kernel during that run; card re-verified 2026-11-18) | pretrained English encoder, **fine-tuned** for 18-way attribution on a Kaggle T4 — 12 layers, hidden 768, **86M backbone params + 98M embedding params** (128K vocab), 160 GB pretraining corpus (model card). Downloaded from HF at **training** time (`kernel-metadata.json` `enable_internet: true`); inference is offline (`kaggle/deberta/predict.py` sets `HF_HUB_OFFLINE=1`) | n/a — the checkpoint carries no authorship labels; the 18-way head is fitted on official train in `src.constants.LABELS` order | 86M + 98M params (card). **No local checkpoint retained** — only `oof/*.npz`, `results.json`, `benchmark.json` came back, so no sha256 can be pinned. Measured: peak GPU **6.463 GB**, **31.0 seq/s**; len 256 / batch 16 / lr 2e-5 | n/a — a checkpoint adds no external text corpus. Its pool arm used the **D026-era per-fold screen** (deprecated, transductive) | registry `20261115T030130-kaggle-deberta_v3_base_official-s42` and `..._pool-s42`; `kaggle/deberta/deberta_train.py` (the kernel now defaults to *large*); ADR D023 | **Not adopted.** Official-only **0.224109**; +pool (cap 40k/fold, per-fold screen) **0.304904** (paired +0.0808 [+0.0540, +0.1076]) — both far below the 0.3760 sparse blend, and the pool arm's screen is the route D026 disabled, so it is **diagnostic only**. The community 0.576 claim was not reproduced at base / len 256 / 1 epoch. |
| deberta-v3-large | https://huggingface.co/microsoft/deberta-v3-large | MIT | 2026-11-16 (fetched by the Kaggle GPU kernel during those runs; card re-verified 2026-11-18) | pretrained English encoder, **fine-tuned** on a Kaggle T4 — 24 layers, hidden 1024, **304M backbone params + 131M embedding params** (128K vocab, ~435M total), 160 GB pretraining corpus (model card). Training-time HF download; offline inference | n/a — same 18-way head; the kernel writes `id2label`/`label2id` from `LABELS` | 304M + 131M params (card). **No local checkpoint retained** (OOF + results only). Measured: peak GPU **10.175 GB**, **7.16 seq/s**, est. total **2.094 h** inside the 9 h budget; len 384 / batch 4 / lr 1e-5 / gradient checkpointing | n/a for the checkpoint. Pool arms: the 12k runs used the **D026-disabled per-fold screen**; the re-wired D038 kernel uses the **D028 global screen** at cap 40,000 | registry `20261116T033901-kaggle-deberta_v3_large_official-s42` / `..._pool-s42`, `20261116T033901-official-deberta_sparse_blend50-s42`, `20261116T053042-official-deberta-officialonly-identity-blend`, `20261116T103200-kaggle-official-only-deberta-s42`; `kaggle/deberta/deberta_train.py` `BACKBONE = "deberta-v3-large"` (default); ADRs D023/D027/D038 | **Not adopted.** Identity-aligned official-only **0.191501**; every fold-aligned sparse+DeBERTa blend is at or below sparse with paired CIs spanning 0 (D027). The earlier non-aligned numbers — official-only 0.248200, +pool 12k 0.385700, blend **0.444200**, official-only blend 0.3393 — are **diagnostic**: the 0.4442 blend used the transductive per-fold screen and **must not be submitted**. D038 (global screen, cap 40k) is wired but **has no metric**; the GPU run is owner-side. |
| modernbert-large | https://huggingface.co/answerdotai/ModernBERT-large/commit/45bb4654a4d5aaff24dd11d4781fa46d39bf8c13 | Apache-2.0 | 2026-11-18, exact pinned revision | frozen English encoder, 395M parameters, 1,024-dimensional masked mean, 384-token cap, fp16 SDPA | logistic head trained on official labels only in canonical 18-class order per fold | 1,583,544,840-byte safetensors, SHA-256 `44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f` | no external labeled corpus | D057 local RTX 4060 pilot and full seed-42 screen; exact tokenizer files verified; supervised H8/D054 is a separate experiment | Pilot blend gains +0.039017 and +0.027778. Full seed-42 blend 0.405620 vs 0.375971, standalone 0.310425. All 5 folds gained. Exploratory; no Kaggle score. |

## Asset checksums (computed locally 2026-11-17)

| asset | bytes | sha256 |
|---|---|---|
| `artifacts/pool_audit/ntl_open_pool.parquet` (open-pool-v15) | 112,851,698 | `5a3dc82288328abedf49b7dd5e866e5c69eb922f9d6008e6fd78cf5153c12332` |
| `artifacts/kaggle_dataset_crossed/encoder/minilm/model.safetensors` (research-only, withdrawn path) | 90,868,376 | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |

Both files are git-ignored. The checksums pin the exact local copies used for the
recorded runs; they are not a claim about the upstream dataset's current version.

**No encoder checkpoint is retained locally for the DeBERTa rows.** The
GPU runs happened on Kaggle and only `oof/*.npz`, `results.json` and `benchmark.json`
were brought back, so there is nothing to hash. Their `size` column therefore reports
**model-card parameter counts**, not local file bytes, and their measured GPU peaks come
from `artifacts/deberta_kernel_output/deberta_cv/benchmark.json` (base, 6.463 GB) and
`artifacts/deberta_official_identity_output/deberta_cv/benchmark.json` (large,
10.175 GB). ModernBERT is retained in the ignored checkpoint folder and used
0.792 GiB peak torch-allocated VRAM in the local D057 pilot.

## Prompt-crosswalk requirement (D024)

**Current state (D028):** pool *training* is enabled again under a
**text-level, fold-independent** screen, which does **not** require a prompt
crosswalk. Pool-based *CV selection* remains bound by the D026 rule: filtering
pool rows by held-out text is transductive and is rejected:
`experiments/pool_train.py` is global-only by construction and its `--clean-ids`
input fails closed, while the GPU kernel aborts on `--pool-scope fold`.

A complete official-prompt-to-pool-prompt crosswalk, supported by independently
reviewed provenance, would *tighten* the screen: it could drop pool rows that
answer an official training prompt in different wording. The competition
training file does not provide the original prompt text and no complete crosswalk
has been established. Do not fabricate empty mappings or infer disjointness from
unrelated IDs, an upstream flag, or a response-only similarity threshold.

**Why this is not an agent-doable task (schema verified 2026-11-18).**
`data/train.csv` has exactly four columns — `id`, `prompt_id`, `text`, `model` —
and `data/test.csv` has `id`, `text`. There is **no official prompt text anywhere
in the provided data**, so no lexical or semantic crosswalk can be computed
locally at any threshold; `prompt_id` is an opaque identifier. Closing this item
requires upstream provenance from the pool publisher or the host, not more
compute. The upstream `max_sim_train_prompt` column (max 0.40 over all 226,191
rows) is a **publisher claim**, not an independent measurement against official
prompts, and is deliberately not used as evidence of disjointness.

**Residual risk, stated plainly:** text-level decontamination cannot rule out a
pool row that answers an official *validation* prompt with different wording. The
mitigations are that the keep-mask is fold-independent, the pool is training-only
(never scored), the official grouped folds keep every official prompt on one side,
and the candidate is gated on an alternative seed plus a nested outer gate. This
risk is why the same-prompt item in the admission checklist below stays unticked.

Once provenance is available, store the reviewed crosswalk only under ignored
`artifacts/` or `data/` (never commit it). Its JSON shape is:

```json
{"provenance": "description of reviewed evidence", "mapping": {
  "<official-prompt-id>": ["<pool-canonical-prompt-id>"]
}}
```

Every official prompt needs an entry; an empty list is an evidence-backed claim
that no matching pool prompt exists. `experiments/pool_audit.py` requires
`--prompt-mapping <path>` for diagnostic screening. Rows without canonical pool
IDs are excluded from that diagnostic, and approximate similarity is only an
additional heuristic. A mapping is **not** a precondition for the D028 training
route; `--no-pool` still gives the standalone official-only path (the submitted
0.356 fallback).

## Admission checklist (before adding a row)

Status for the **five** registered rows (open-pool-v15, minilm-l6-v2,
deberta-v3-base, deberta-v3-large, modernbert-large), reviewed 2026-11-18. The items
below are written for the **data-bearing** rows: a checkpoint adds no text corpus, so
dedup and label mapping reach the encoder rows only through the pool they were
fine-tuned on. **ModernBERT-large was used in the local frozen probe** (D057).

- [x] Rules check recorded in `docs/competition_spec.md` §9 (external-corpora row,
      pretrained-encoder row; Code Requirements + Rules §2.6 basis).
- [x] Source is public and freely accessible to every participant (public Kaggle
      dataset / public HF checkpoint; URLs in the register).
- [x] License permits competition use and redistribution of derived artifacts —
      CC BY-SA 4.0 (pool), Apache-2.0 (MiniLM, and ModernBERT-large if H8 runs),
      **MIT** (DeBERTa-v3 base and large, verified on both cards 2026-11-18). The
      permissive licences require attribution but impose no share-alike;
      **share-alike on derived pool artifacts comes from the pool itself**: attribution
      + CC BY-SA must travel with any redistributed pool-derived weights, including
      weights fine-tuned from a permissively licensed checkpoint.
- [x] Access date, exact file/version and checksum recorded (v15, accessed
      2026-11-14; sha256 table above, computed 2026-11-17).
- [x] Model-name mapping to the 18 official labels is explicit; versions are
      never merged silently — `per_model.csv` matches all 18 exactly and
      `src.pool.load_pool` rejects an unknown or missing label.
- [x] Exact-duplicate and near-duplicate dedup against `train.csv` text; threshold
      and dropped-row count documented (cos > 0.3, 0 exact + 142 near → 226,049).
- [ ] Same-prompt contamination check: **partially satisfied, deliberately left
      open.** Text-level screening is verified and fold-independent (D028) and the
      publisher already removed 284 train-prompt answers, but no reviewed prompt
      crosswalk exists (D024), so a differently-worded answer to an official
      validation prompt cannot be excluded. See the residual-risk note above.
- [x] Compared official-only vs official+external on the **same** official grouped
      folds; delta **and harm** reported (char +0.0912 [+0.0721, +0.1103]; style
      −0.0174 [−0.0480, +0.0132]; per-class recall deltas in
      `artifacts/pool_cv_cap40k/official/results.json`).
- [x] If weights are used: **packaged for attachment** as a Kaggle dataset, no
      runtime download, licence recorded, and CPU runtime measured —
      `artifacts/kaggle_dataset` ships the two fitted bundles + `src`, no encoder
      and no pool parquet. **D050:** `dataset-metadata.json` now declares
      **CC-BY-SA-4.0** (was wrongly CC0-1.0) and the package README carries the
      attribution block — licence link, competition credit, and Open Pool v15
      credit for the pool-fitted char member — so share-alike travels with the
      weights. Cold-start inference on 4,000 synthetic rows **13.77 s / 0.370 GB**
      (D043, `artifacts/inference_benchmark/results.json`), re-measured after the
      D050 `src/` refresh at **9.40 s / 0.370 GB**
      (`artifacts/inference_benchmark_refresh/results.json`; bundles byte-identical,
      no refit). **The upload itself, the graded Kaggle rerun and the submission
      are owner-only and have not happened for this candidate.**

## Fine-tuned encoder checkpoints (GPU, research-only)

The DeBERTa arms differ from MiniLM in kind: they were **fine-tuned** on the official
train (and, in the pool arms, on external text), not used frozen. Three consequences are
recorded so the register is not misread:

1. **Download timing.** The training kernel runs with internet **on**
   (`kaggle/deberta/kernel-metadata.json`) and pulls the checkpoint from Hugging Face.
   That is a *training-time* fetch, which the rules allow; the graded submission run has
   internet off and `kaggle/deberta/predict.py` sets `HF_HUB_OFFLINE=1` and loads from a
   local `--model-dir`. No encoder is attached to any current package.
2. **Nothing was adopted.** Official-only DeBERTa-v3-large scores **0.191501** on the
   identity-aligned folds and adds no diversity gain to the sparse blend (D027). The
   attractive **0.4442** blend came from the per-fold pool screen that D026 disabled, so
   it is diagnostic and must not be submitted. The adopted candidate remains the sparse
   cap-40k pool-MIX blend.
3. **Two arms are wired but unmeasured.** D038 re-pointed the kernel at the D028 global
   screen (cap 40,000, `FOLD_SEED = s42`); D054 made the backbone a module-level
   `BACKBONE` switch because a Kaggle **script** kernel receives no CLI arguments.
   Neither has a score. Both are owner-side GPU runs, gated by
   `experiments/blend_deberta.py` under s42, s13 and the nested outer gate.

## Notes on frozen embeddings

`src/features_embedding.py` provides a deterministic hashing encoder (no
download, used by the hermetic unit tests) and an optional `sentence-transformers`
path. The transformer path gave no validated *authorship* gain on the official
grouped folds (`configs/frozen_encoder.yaml`) and is **not adopted as a feature**
(ADR D013). It was used for prompt-group clustering in the crossed-design
candidate, which is **withdrawn** (ADR D017/D018): `predict_dataframe_crossed`
embeds passages with the attached MiniLM checkpoint and clusters them, but the
one-per-model gain is a k=18 artifact and does not deploy to a k≈2 test. That path
is **research-only**. It requires the optional `sentence-transformers` dependency
(+ `transformers`, `torch`) and the checkpoint attached offline (`package
--crossed` ships `encoder/minilm`, loaded with `HF_HUB_OFFLINE=1`). The core
`requirements.txt` / `requirements-lock.txt` exclude `sentence-transformers`
because the *training* pipeline stays sparse and download-free, and the
**submitted argmax blend needs no encoder at all**. The stress-test experiment
(`experiments/stress_k.py`) uses MiniLM only to cluster pooled-OOF passages
locally for the inference-only stress test, never in a reported primary score.
