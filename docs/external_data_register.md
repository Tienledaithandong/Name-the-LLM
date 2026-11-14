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

**Adopted for grouping (2026-11-14, ADR D016):** the same MiniLM encoder is used to
*recover prompt groups* by semantic clustering (topic, not authorship) for the
crossed-design assignment pipeline (`src/grouping.py`,
`src.inference.predict_dataframe_crossed`). Recovered-group assignment is
validated on the untouched secondary holdout (**0.5000** vs 0.3956 argmax,
+0.1044) and alternative fold seeds 7/13 (0.4716 / 0.4657), capturing ~90% of the
oracle ceiling. The crossed submission candidate therefore **depends on the
MiniLM checkpoint**, attached offline via `python -m src.cli package --crossed`
(`encoder/minilm` + `inference.json`); the notebook loads it from disk with
`HF_HUB_OFFLINE=1`. It is used only for clustering, never as an authorship
feature, and is absent from `configs/submission.yaml` (the blend members are
unchanged). Pending the owner's LB submission to confirm the gain on the hidden
test.

| id | source URL | license | accessed | what it is | labels / model mapping | size | dedup vs train | used in | validation result |
|---|---|---|---|---|---|---|---|---|---|
| minilm-l6-v2 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | 2026-11-14 | frozen sentence encoder (384-dim); grouping only, not a feature extractor for authorship | n/a — no labelled text; maps passages to vectors for clustering, not to the 18 labels | ~90 MB checkpoint (88 MB snapshot, single `model.safetensors`) | n/a — no external text corpus added | H3: `minilm_embedding_logreg`, `minilm_embedding_style_logreg` (registry, seed 42); **crossed submission: prompt-group clustering** (`phase_b_*` registry rows; `package --crossed` → `encoder/minilm`) | **Authorship: not adopted** (H3). Embedding-only 0.1278 pooled OOF; +style 0.3093. Greedy char+emb_style 0.3786 CV (+0.0026) reverses on the holdout (0.3933 vs 0.3956); report §6.6, ADR D013. **Grouping: adopted** (D016). Recovered-group assignment holdout 0.5000 (+0.1044), seeds 7/13 0.4716/0.4657; compliance cleared (spec §9); attached offline, loads with `HF_HUB_OFFLINE=1`. |

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
(ADR D013). It **is adopted for prompt-group clustering** in the crossed-design
submission (ADR D016): `predict_dataframe_crossed` embeds test passages with the
attached MiniLM checkpoint and clusters them to recover prompt groups. This
requires the optional `sentence-transformers` dependency (+ `transformers`,
`torch`) importable in the Kaggle image and the checkpoint attached offline
(`package --crossed` ships `encoder/minilm`); Kaggle inference has internet
disabled, so the notebook sets `HF_HUB_OFFLINE=1` and loads from disk. The core
`requirements.txt` / `requirements-lock.txt` still exclude `sentence-transformers`
because the *training* pipeline stays sparse and download-free; it is an
inference-only dependency of the crossed candidate, verified to load offline.
