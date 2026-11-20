# Prompt-aware GPU attribution experiment

## Goal and controls

Compare three variants from the **same pretrained encoder**, folds, batches,
initialization seeds and cross-entropy training budget:

1. `classification`: 18-class cross-entropy only.
2. `contrastive`: cross-entropy plus supervised contrastive loss, where positives
   are same-model passages from **different** prompts and all negative weights
   are 1.
3. `hard_negative`: the same loss but upweights different-model pairs answering
   the **same** prompt in the contrastive denominator.

Two-prompt batches contain matched authors, so each anchor has a cross-prompt
positive and same-prompt negatives. The sampler only sees fold-training rows.
The evaluation is ordinary independent passage classification; it does **not**
use label information from validation groups or crossed assignment. Embeddings
use masked mean pooling. Positive anchors lacking a valid match fail loudly.
The comparisons test a hypothesis; no gain or leaderboard score is assumed.

The historical CV/holdout experiments used the full training data. Although
this script sets aside prompt groups before this new experiment, **those groups
are not an independent final holdout for choices informed by prior work**.
They are not evaluated by this script. A truly untouched future evaluation
requires new prompts or a genuinely unused dataset; use the public LB carefully.

## Run on Kaggle

1. Attach the competition data. Enable GPU in notebook settings.
2. Upload this repository's `src/`, `experiments/`, and `configs/` as a Kaggle
   input dataset (or otherwise make this source available offline). Do not bundle
   competition data or secrets with it. The repository has not been uploaded by
   this script; that is an owner action.
3. Attach a **public, competition-eligible** Hugging Face checkpoint directory,
   including tokenizer and weights. DeBERTa-v3-base is a candidate, not a proven
   best choice; verify availability, license, and attribution in the external-data
   register before use. `transformers` must support the chosen architecture.
4. Open `kaggle/contrastive_gpu.ipynb` and set `REPO_DIR` and `CHECKPOINT` to
   the exact mounted paths. Keep Kaggle internet disabled. Its command reads
   `train.csv` only. The notebook image must already contain PyTorch,
   transformers, tokenizers, sentencepiece (for DeBERTa), pandas, scipy, scikit-learn and PyYAML; no runtime
   installation or model downloads are performed.
5. The default config is 5 folds × 3 variants × 3 epochs × 80 batches. This is
   a proposed budget, **not a measured runtime**. Adjust a copy of the config for
   a pilot, watch GPU memory and Kaggle's time limit; repeat the full fixed
   protocol before drawing conclusions. New output directories are mandatory,
   preventing accidental overwrite.

`metrics.json` contains aggregate results, and `paired_bootstrap.json` contains
paired prompt-level bootstrap intervals versus classification. `manifest.json` records config,
partition fingerprints, checkpoint content hash, versions and class order. OOF `.npz` files contain
row IDs, labels, scores and fold assignments, allowing alignment checks, but
are sensitive competition-derived artifacts. Never commit or publicly upload
these files without checking the rules. Store them privately. The script does
not save fitted models and cannot generate a Kaggle submission; training a final
model, blending and packaging require a separate validated implementation.

## Decision and evaluation

Compare the three variants on the same folds using pooled balanced accuracy,
per-class recall and the paired **prompt-level** bootstrap on OOF predictions.
Recompute the existing char/style baseline on the *same selection partition
and folds*; its historical OOF files are not aligned with this run. Preserve
row order and do not tune on the reserved prompts or hidden test. Once a fixed
candidate demonstrates a robust improvement and runtime is feasible, design a
separate final-fit/packaging workflow and evaluate on genuinely new prompt
holdout data if available. Scores from a tiny synthetic smoke test are solely
engineering checks and say nothing about leaderboard performance.

## Local tests

`python -m pytest -q` checks batch construction. With the optional `torch` and
`transformers` dependencies installed, it also tests the loss and runs a tiny
randomly initialized BERT offline on synthetic data using CPU. It downloads no
checkpoints. Kaggle GPU execution and official-data performance remain untested
until the owner runs the notebook.

## Research references

- Beigi et al., *Model Attribution in Machine-Generated Disinformation*:
  supervised contrastive learning for prompt/domain generalization, tested on a
  different attribution task and model set. <https://arxiv.org/abs/2407.21264>
- McGovern et al., *Your Large Language Models are Leaving Fingerprints*:
  lexical and morphosyntactic fingerprint features, with family similarities.
  <https://aclanthology.org/2025.genaidetect-1.6/>
- La Cava et al., *Authorship Attribution in Multilingual Machine-Generated
  Texts*: fine-tuned and contrastive attribution under multilingual shift.
  Its benchmark is not directly comparable to this 18-class competition.
  <https://arxiv.org/abs/2508.01656>
