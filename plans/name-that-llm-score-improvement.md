# Name That LLM score improvement plan

Prepared 2026-11-18 after Kaggle scored submission **58502550**. The goal is to beat the official-data sparse baseline on new prompts and earn a measurable improvement on Kaggle.

**Implementation update:** The seed-42 [execution record in D057](../docs/decisions.md)
closes the provider and masking screens at 0.373388 and 0.364111. The ModernBERT
blend passed the pilot and seed-42 gate at **0.405620** versus **0.375971**.
See the staged checks below before considering a submission.

| Control | Balanced accuracy |
| --- | ---: |
| Kaggle public score, submissions 58344764 and 58502550 | 0.356 |
| Official-only seed 42, five frozen prompt-grouped folds | 0.375971 |
| Official-only seed 13, five frozen prompt-grouped folds | 0.358190 |
| Official-only nested sparse benchmark, three outer and three inner folds | 0.351892 |

The control is the submitted equal-weight blend of row-normalized character TF-IDF logistic regression and stylometry logistic regression. Compare candidates to the control rerun on the **same rows and folds**. The nested benchmark above is context, since the training set size and the evaluation procedure differ from the five-fold result. Evidence is in the [registry](../experiments/registry.csv), [submission checklist](../docs/submission_checklist.md) and [Kaggle submissions](https://www.kaggle.com/competitions/name-that-llm/submissions).

## 1. Freeze the comparisons

**Inputs:** [experiment protocol](../docs/experiment_protocol.md), D024–D027 in the [decision log](../docs/decisions.md), the official OOF artifacts in `artifacts/cv_identity/official` and `artifacts/cv_identity/official_s13`, and their matching split files under `artifacts/splits`.

- Reproduce both registered controls with `python -m src.cli evaluate --dataset official --artifacts artifacts/cv_identity/official --splits artifacts/splits/splits_official_verified_s42.json --members char_tfidf_logreg style_only --weights 0.5 0.5 --out artifacts/score_improvement/baseline_s42.json`. For seed 13 use the `official_s13` artifacts, `splits_official_verified_s13.json`, `--seed 13` and a separate `--out artifacts/score_improvement/baseline_s13.json`.
- Before fitting, lock each trial's model recipe, feature transformations, blend weight, seeds and resource budget. Write the candidate to a separate YAML config. Verify the ordered dataset fingerprint, 18-label order, exact fold IDs and prompt isolation. A failed check stops the comparison.
- Use seed 42 for the primary screen and seed 13 to check the direction of the effect. Both reuse previously explored prompts. For confirmation, use a predeclared nested grouped procedure with separate inner training and validation folds. Refit the full control and candidate on matched outer training sets. Any model or weight selection in that procedure uses inner OOF only. Use genuinely new labeled prompts as an additional check if an eligible set becomes available.

**Proposed promotion gate to lock before the first run:** gain at least **0.010 absolute pooled balanced accuracy** on both the seed-42 and nested comparisons, improve at least four of five seed-42 folds, show a positive difference on seed 13, and have a positive lower bound on an approximate paired 95% prompt-bootstrap interval for the nested gain. Resample entire prompts with replacement and recompute the pooled candidate-minus-control metric on the same rows in each draw. Inspect per-class recall, error counts within and across providers, and the existing length bands. These are proposed decision criteria, not a predicted gain or a replacement for the repository protocol. The nested check controls tuning within the new procedure and still reuses previously explored prompts.

## 2. First experiment: distinguish model versions within a provider

**Why:** 897 of 1,684 seed-42 baseline mistakes stayed within the correct provider. The existing hard punctuation veto and class-bias experiments did not establish a gain (D044 and D053).

- Fit one shared character and style feature map on the official training side of each fold. Train a regularized logistic classifier for each provider with multiple classes, using only training rows from that provider. Use `src.metrics.provider_of` and `src.features_tfidf.TextMatrixBuilder`. Fix `C=1.0` for the specialist and diagnostic flat heads.
- For each passage, apply softmax to the current blended scores to obtain relative score weights, without claiming calibration. Within each provider, mix the baseline's conditional weights and the specialist's logistic output with fixed weights **0.8/0.2**, then multiply by the original provider weight. Keep single-label providers unchanged. Choose the label with the highest combined weight in canonical label order. This lets a specialist change the preferred version while preserving total weight for each provider. No prompt ID or other test passage is required for prediction.
- Include the original blend as the primary control. Include a flat 18-class logistic head using exactly the specialist's features and analogous fixed blend weight as a diagnostic, to check whether any gain comes from the hierarchy or the additional features. Confirm that specialist weight zero reproduces baseline predictions exactly. Test label alignment, score conservation, singleton behavior, group isolation and identical results with single-passage and batched prediction.

**Budget and exit:** CPU only, one seed-42 screen in up to two hours. Advance to seed 13 and nested confirmation if the primary screen meets its pooled and fold checks. Otherwise log the result and keep the existing model.

## 3. Separate CPU experiment: reduce dependence on topic words

**Why:** The baseline scores **0.2491** on passages under 50 words and **0.4718** on passages of at least 200 words (D041). The current style vector counts function words. This trial tests whether their sequence and the surrounding punctuation supply additional signal. Prior authorship research motivates a masking ablation and does not establish a gain on this competition. [What represents “style” in authorship attribution?](https://aclanthology.org/C18-1238/)

- Preserve the existing function words, punctuation, whitespace and line breaks. Match function words without regard to case and keep their original text. Replace other word tokens with fixed length-bucket markers for lengths 1–3, 4–6 and 7+. Freeze the transformation before scoring. Train one character TF-IDF logistic model using the same configuration as the registered raw character arm, with vocabulary and IDF fitted only on fold training.
- Compare the untouched baseline to a fixed blend with 25% weight on the masked model and 75% on the existing blend, using the same per-member normalization. Also run 25% raw character and 75% existing blend to distinguish masking from giving the character model more weight. Report the existing length bands without choosing length cutoffs or adjusting weights from validation results.

**Budget and exit:** CPU only, one seed-42 screen in up to two hours. Advance only if overall pooled score and fold stability clear the primary screen. Improvement confined to short passages motivates later research and leaves the submission unchanged.

## 4. GPU fallback: one official-data encoder probe

**Why:** Identity-verified DeBERTa-large scored **0.1915** alone and **0.3393** in a fixed blend on seed 42, below the sparse control. Try a different representation with a small budget and a clear stop. D054 makes the backbone selectable in the supervised trainer. The frozen feature extractor requires separate implementation. [ModernBERT paper](https://arxiv.org/abs/2412.13663) · [ModernBERT-large model card](https://huggingface.co/answerdotai/ModernBERT-large)

- After checking permitted use and recording the Apache-2.0 checkpoint and exact revision, implement **one** frozen ModernBERT-large feature extractor. Use mean pooling over non-padding tokens with a 384-token cap and a regularized logistic classifier. Cache passage embeddings by passage hash, checkpoint and tokenizer revision, token cap and pooling method. Fit any learned normalization and the classifier only within training folds. Use no Open Pool rows.
- Benchmark throughput, peak RSS and GPU memory on a training-only slice first. Screen frozen seed-42 folds 0 and 1, including the fold where DeBERTa fell to chance, with a fixed 25% encoder contribution to the sparse blend. Run the full seed-42 screen only if standalone balanced accuracy exceeds the 18-class chance rate and the blend gains at least 0.010 on both pilot folds. The full screen reuses those folds, so neither result is independent confirmation.
- Set a **45-minute GPU pilot budget**. Set a separate two-hour budget for a full seed-42 screen if the pilot passes. Fine-tuning remains a separate experiment with its own approval and budget after an encoder signal is established.

## 5. Confirm and submit only an improvement

- Steps 2 and 3 can run independently after step 1, with distinct configs and artifact paths. Predeclare the fixed specialist in step 2 as the primary comparison. Step 3 and the GPU fallback in step 4 are exploratory. To promote an alternative discovered by these screens, test it on genuinely new labeled prompts or replay the full candidate selection rule inside each outer training partition, including any pilot and fallback choices. Picking a finalist using the full-data screens before running nested CV would still expose the outer holdouts to selection. Log negative and inconclusive runs.
- Require the promotion gate and explain any substantial class or length-band regression. Lock the recipe and refit it on all eligible labeled training rows. Package it in a separate private notebook. Verify internet-disabled startup with pinned dependencies, a 4,000-row synthetic load check for runtime and memory, and an interactive competition example run. Verify exact IDs, label order, null checks and `submission.parquet`. The example and synthetic runs check the pipeline.
- Keep the scored [version 1 official-data notebook](https://www.kaggle.com/code/tiennamlee/name-that-llm-official-sparse-submission) as the fallback. Use a single new submission slot for a candidate that passes. Record the version, ID, pooled local OOF and live public score separately. Follow the account authorization in effect at submission time.

## Open Pool eligibility

Keep the higher Open Pool numbers in the research record. The full-corpus filter learned from passages later used for validation, and the nested recipe's outer-training mask can still see inner validation passages. Reopen this branch only with authorization, a verified prompt crosswalk or independently proven prompt separation, and eligibility established without using the evaluated passages. A crosswalk alone does not validate old masks. Review provenance and licensing, write an ADR for the new protocol, and rerun against the same eligible control before considering promotion.

Plan updates should name any changed recipe, dependency or gate and when it changed. Lock changes before further evaluation and label any comparisons they affect. Follow the repository's test and registry requirements when implementing each step.
