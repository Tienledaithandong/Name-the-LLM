# SASCR on Kaggle / Google Colab

The SASCR multi-layer causal readout (**Member B**, frozen Qwen3-8B-base) and
sibling residual head (**Member C**) need a GPU + a causal backbone at runtime.
Run them in the cloud (Kaggle Notebooks or Google Colab) instead of on your
local box.

This folder contains a single orchestrator you can paste into a Colab cell (or
Kaggle Notebook cell) to run the whole pipeline without touching this repo's
local environment. It reuses the repo's own `src.cli` + `src.features_multilayer`
+ `src.model` modules, so the code that computes is the code that was tested.

---

## Before you open it

1. **Sign in to Kaggle/Colab** and attach/give access to a dataset that holds
   `train.csv` (competition `train.csv`) — e.g. the competition dataset or a
   Kaggle dataset you own. The script downloads it itself.
2. **A causal backbone.** The first run downloads `~6 GB` of Qwen3-8B (CPU or GPU)
   into the runtime cache the first time; subsequent runs reuse it. If you want
   to skip the download for your first run, attach a Kaggle dataset containing
   the `pytorch_model.bin` / `config.json` / `tokenizer.json` instead.
3. Nothing here touches the competition leaderboard directly — scores are
   exploratory until they pass the seed-2027 holdout.

---

## Run on Google Colab

1. Open [colab.new](https://colab.research.google.com) → **File → Save a copy in
   Drive** of `cloud/setup_sascr_on_colab.py` (or just **Edit → Paste cell source
   → Shift+Enter**).
2. Run the cell. Pick the backbone when it asks:
   - `hashing` — fast, CPU, no download, only a sanity check for the protocol.
   - `transformers` (default) — the real frozen Qwen3-8B-base readout. Downloads
     the weights on first run.
3. Watch the printed OOF balanced accuracy + per-member stats.
4. The cell writes `submission.parquet` to `/content`. **Download it** (right-click
   the file in the Files panel → Download) and submit via the Kaggle API.

## Run on Kaggle Notebooks

1. Create a Kaggle Notebook, set **Accelerator → GPU**, and set **Internet ON**.
2. Add your `train.csv` containing dataset as an input (or paste your API token).
3. Upload this `cloud/` folder as a Kaggle dataset input, then run the cell in
   `setup_sascr_on_colab.py`.
4. Download `submission.parquet` and submit.

## How the heavy parts are isolated

| Member | What runs where | Requires |
|---|---|---|
| A (char + global pool trunk) | `src.train.run_cv` + `src.inference.fit_final` | CPU only |
| B (multi-layer causal readout) | `MultiLayerReadoutBuilder(backend=transformers)` over Qwen3-8B-Base | GPU/Colab GPU, ~6 GB first run |
| C (sibling residual head) | `SiblingResidualClassifier` fit on B's folds | CPU after B is fit |
| Fusion | `row-z` + damped Sinkhorn-Knopp | CPU |

`backend=hashing` (Member B disabled) lets you re-run the sparse+SASCR fusion on
CPU and compare against the 0.442675 pool-MIX candidate without any download.

## Note on scores

Scores produced here are **exploratory only**: they come from a holdout computed
on the platform's CPU/GPU. They are not yet the seed-2027 holdout gate. Do not
claim them in a leaderboard submission unless you have run the reserved
seed-2027 holdout (D024).
