# External data / model register

Rules basis: Code Requirements allow "freely & publicly available external data,
including pre-trained models"; Rules §2.6 require external data to be publicly
available and equally accessible at no cost, or "reasonably accessible" at
minimal cost. Nothing may be downloaded at submission time (internet disabled).

## Register

**No external data or pretrained model is currently used by any reported model.**
The pipeline is sparse TF-IDF + stylometry, fitted from scratch on the official
training folds.

| id | source URL | license | accessed | what it is | labels / model mapping | size | dedup vs train | used in | validation result |
|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — |

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
download) and an optional `sentence-transformers` path. The transformer path is
**not** used in any reported result: it requires an attached local checkpoint and
has not been benchmarked here. Adopting it requires a register row plus a
measured gain on the grouped folds, as with any other external resource.
