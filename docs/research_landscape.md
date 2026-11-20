# Research review: model attribution and competition implementations

Review date: 2026-11-19. These are research leads, not independently reproduced
leaderboard scores. In particular, external repositories' self-reported numbers
can use different data, splits, models, runtime budgets, and leaderboard samples.
No third-party source code, weights, or competition records were copied.

## Findings and critical assessment

1. **satriadaffa/name-that-llm**
   (<https://github.com/satriadaffa/name-that-llm>): reports a prompt-grouped
   multi-view character/word/shape/function-word SVM, external-pool gains, and
   negative results for self-training and hard prompt assignment. This supports
   prioritizing complementary sparse representations and well-screened extra
   data. But the results are self-reported, its pool and exact folds differ from
   ours, and the public score cannot establish private-set performance. This
   repository already has pool and multi-view research (`src/pool.py`,
   `configs/masked_character.yaml`, `docs/decisions.md`), so do not assume that
   simply adding another SVM will beat existing work. Check the external data's
   license, labels, and contamination risk before use.

2. **omkarpattanaik/Name-That-LLM**
   (<https://github.com/omkarpattanaik/Name-That-LLM>): describes a large
   multi-backbone ensemble, external data, and global Sinkhorn balancing, and
   reports that hard one-per-model matching failed on its public LB. The exact
   run and code have not been independently verified here. Global Sinkhorn can
   be just as risky if it forces an assumed uniform test prior: balanced accuracy
   weights *recall* equally, but does not imply equal hidden class frequencies.
   Nor does near-uniform output prove grouping quality. Do not adopt either
   method without a prompt-grouped, incomplete-group stress test and verified
   constraints. The large ensemble's GPU, offline and 9-hour feasibility must
   also be established for its exact checkpoints.

3. **OpenTuringBench / OTBDetector**, La Cava & Tagarelli, EMNLP 2025
   (<https://aclanthology.org/2025.emnlp-main.1354/>): compares attribution
   methods under domain and generation shifts and studies a contrastive model.
   It supports our prompt-aware contrastive study, not a guarantee that its
   headline F1 transfers to 18 short passages from specific frontier versions.
   Different generators and substantially larger training corpora confound a
   direct performance comparison. Our existing experiment matches positives
   across prompts and negatives within prompts, then tests the hypothesis with
   prompt-disjoint CV rather than borrowing the published number.

4. **Your Large Language Models are Leaving Fingerprints**, McGovern et al.
   (<https://aclanthology.org/2025.genaidetect-1.6/>): finds lexical and
   morphosyntactic signals that can persist across domains, with similarities
   within model families. That both motivates multiview features and warns that
   identifying close siblings is harder than distinguishing broad families.
   The source focuses on machine-text detection and its reported family behavior
   is not proof that individual versions are separable in our dataset.

5. **Authorship Attribution in Multilingual Machine-Generated Texts**
   (<https://arxiv.org/abs/2508.01656>): reports strong fine-tuned and
   contrastive results in its own multilingual evaluation but significant
   cross-language/domain limitations. It tests different authors and conditions;
   neither its 0.9 F1 nor another team's ~0.6 public score validates a target of
   0.8 balanced accuracy here. Test any transfer on identical prompt-held-out
   folds and report per-class recall, runtime, and prompt-resampled uncertainty.

## Implemented follow-up

The GPU encoder experiment previously **always truncated to the first 256
subword tokens**, so a passage's ending could be lost. A candidate suggested by
style-attribution research is to expose more of each passage rather than adding
unverified external model complexity. `configs/contrastive_multicrop_gpu.yaml`
adds a controlled opt-in recipe: same-prompt hard negatives with seeded random
training windows and head/tail logits averaged at evaluation. The script reports
selection-set truncation and length slices and compares with a matched head-only
hard-negative control using prompt bootstrap. This is an untested hypothesis, not
an established improvement; two inference passes can increase runtime. Since
both cropping and evaluation pooling change, an eventual gain would need further
ablation before attributing the cause. See `docs/contrastive_gpu.md`.

No official training data or Kaggle GPU was available in this checkout. The
current experiment does not write a submission or save trained final models.
The older, already explored prompt holdouts are not independent final evidence;
use nested grouped evaluation or genuinely new labeled prompts before promotion.
