# Final submission checklist (owner actions)

Account-sensitive steps below are **owner-only**; the agent prepares artifacts
and instructions but does not accept rules, upload, or submit. Everything else
is already reproducible from this repository.

## Verified evidence to date (local grouped CV, not leaderboard)

- Adopted candidate: **crossed inference** (D016): recovered-group one-per-model
  Hungarian assignment on blend OOF, with test-time prompt groups recovered by
  clustering frozen-miniLM topic embeddings (cosine distance threshold 0.4).
  Blend members are `0.5 × char_tfidf_logreg + 0.5 × style_only`
  (`configs/submission.yaml`, `artifacts/cv/official/ensemble/weights.json`).
- Blend-only (argmax) pooled OOF balanced accuracy **0.3760**, +0.0567 over the
  best single model, positive in all five folds.
- Crossed inference (grouping + assignment) on three splits: untouched holdout
  **0.5000** (+0.1044), grouped CV seed 7 **0.4716** (+0.1108), seed 13
  **0.4657** (+0.1075). Purity held-out 1.000, CV s7 0.980, CV s13 0.990.
  ~88% of the oracle ceiling (+0.1100).
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

## Build and upload the artifact (crossed candidate, D016)

4. `make predict-crossed` — refits the two blend members on all labelled rows and
   writes `artifacts/submissions/official/submission.parquet` using crossed
   inference (schema-checked). Needs `sentence-transformers` installed locally and
   a MiniLM checkpoint (auto-resolved from the HF cache, or set
   `ENCODER_PATH=/path/to/checkpoint`). The public `test.csv` is only a 3-row
   sample, so this local parquet is a mechanics check, not the graded output.
5. `make package-crossed` then set your username:
   `python -m src.cli package --dataset official --crossed --kaggle-user <your-username>`.
   This produces `artifacts/kaggle_dataset/` containing the blend bundles,
   `weights.json`, `src/`, `manifest.json`, **`inference.json`** and the offline
   MiniLM grouping encoder (**`encoder/minilm`**, ~88 MB). It contains **no
   competition data**. (Shortcut: `make package-crossed`.)
6. Upload: `kaggle datasets create -p artifacts/kaggle_dataset` (or update the
   existing `name-that-llm-artifacts` dataset to a new version). The crossed
   package supersedes the argmax one; the notebook auto-detects `inference.json`.

## Run the Kaggle notebook

7. Open `kaggle/submission.ipynb`; attach **both** the competition dataset
   (`name-that-llm`) and the crossed artifacts dataset from step 6.
8. Set **Internet = Off** (required). The notebook sets `HF_HUB_OFFLINE=1` and
   loads the attached MiniLM encoder from disk; it makes no network calls.
   `sentence-transformers` (+ `transformers`, `torch`) must be importable in the
   Kaggle image — if not, attach wheels or use an image that includes them.
9. Run all cells. It fails fast if the manifest, class order, test schema, ids,
   nulls or labels are wrong, and writes `submission.parquet` (`id, model`).
   Expect a `scikit-learn` version-mismatch warning (trained 1.9.1 vs Kaggle
   1.6.1); the prior submission loaded fine despite it.
10. Confirm the printed row count equals the hidden-test row count and the label
    histogram is **near-uniform (~1/18 per class)** — the crossed-design
    signature. A single class dominating signals clustering collapse on the real
    test; if so, fall back to the argmax blend (`make package`, no `--crossed`).

## Submit and record

11. Submit the notebook output. Record in `experiments/registry.csv`:
    submission version, git commit, CV score, **public LB score**, timestamp and
    changes. Keep public LB separate from CV; never select on public LB alone.
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