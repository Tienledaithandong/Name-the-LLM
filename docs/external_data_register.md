# External data / model register

Rules basis: Code Requirements allow "freely & publicly available external data,
including pre-trained models"; Rules §2.6 require external data to be publicly
available and equally accessible at no cost, or "reasonably accessible" at
minimal cost. Nothing may be downloaded at submission time (internet disabled).

## Register

**The submitted argmax blend (D009/D012, public LB 0.356) uses no external data or
pretrained model** — it is sparse TF-IDF + stylometry, fitted from scratch on the
official training folds. One public pretrained encoder was *benchmarked* for the
H3 authorship hypothesis (Section 6) and **not adopted for authorship**; it is
registered below for provenance.

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
| open-pool-v15 | https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool | CC BY-SA 4.0 (65 sources: CC BY 4.0 / CC BY-SA 4.0 / CC0) | 2026-11-14 | 226,191 clean community passages (v15, 2026-11-14), 30 cols; labels are publishers' claims, not re-verified | **all 18 official labels, exact match** (`per_model.csv`); uneven 3,541 (`gemini-3.7-flash`) … 23,474 (`gemini-3.8-flash`) — use class weights/sampling | 112 MB `ntl_open_pool.parquet` (226,191 rows); + `prompts.parquet`, `gemini37_unverified.parquet` (10,838 excluded) | dataset `max_cos_train` ≤ 0.30, 0 exact text dups vs train, train-prompt answers (284) + mislabeled (284) already removed; **independent per-fold** word-TFIDF cos>0.3 (per-fold fit) removed **19** rows union (0 exact, 0 shared prompt) → clean **226,172** | Phase 1b/1c official-only vs official+pool training (DeBERTa #31, GBDT #32); audit `artifacts/pool_audit/audit.json` + `clean_pool_ids.json`; `src/pool.py`, `experiments/pool_audit.py` | **Audited, admission pending training gain** (D019). Verified: `flag_gemini37_routing_risk`/`flag_near_train_prompt`/`flag_train_prompt_regen` **all False** (routing-risk rows already in `gemini37_unverified.parquet` — do NOT re-filter that flag); labels complete; 0 null text. Caveats: `match_weight`/`style_score` fitted on **all** train prompts → refit per-fold or avoid for honest CV; `flag_kb_proxy_after_ruling` 20,227 rows (hosts allow training on public datasets made via proxy; optional strict drop); CC BY-SA share-alike on derived artifacts. |

## Admission checklist (before adding a row)

- [ ] Rules check recorded in `docs/competition_spec.md` §9.
- [ ] Source is public and freely accessible to every participant.
- [ ] License permits competition use and redistribution of derived artifacts.
- [ ] Access date, exact file/version and checksum recorded.
- [ ] Model-name mapping to the 18 official labels is explicit; versions are
      never merged silently (e.g. `gpt-5.6-luna` ≠ `gpt-5.6-sol`).
- [ ] Exact-duplicate and near-duplicate dedup against `train.csv` text; document
      the threshold and how many rows were dropped.
- [ ] Same-prompt contamination check: external rows must not import prompts that
      appear in official train folds in a way that leaks across the split.
- [ ] Compared official-only vs official+external on the **same** official
      grouped folds; report the delta, including harm.
- [ ] If weights are used: attached as a Kaggle dataset, no runtime download,
      licence recorded, and CPU/GPU runtime measured.

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
