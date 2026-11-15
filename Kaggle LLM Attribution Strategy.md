# **Roadmap from 0.356 to Top-1 Contention in Kaggle Name That LLM**

Your GitHub repository (Tienledaithandong/Name-the-LLM) was inaccessible during this review, so a line-by-line audit of your scripts is not possible1. Even without inspecting the raw code, your public leaderboard score of 0.356 places your current pipeline squarely in the classical lexical, stylometric, or un-pooled small-encoder tier—nearly identical to TF-IDF Ridge (0.3638), 4-view LinearSVC (0.3575), and public CPU stylometry baselines (0.348–0.353)2. Reaching the 0.70+ Top 1% zone (0.711) or contending for the Top-10 prize cutoff (0.749) and Top-1 score (0.769, as tracked on Mark Barney's competitor research terminal on November 13, 2026\) before the January 13, 2027 deadline requires replacing surface n-gram modeling on train.csv alone3.  
Closing that 0.413-point gap in roughly eight weeks depends on four structural upgrades documented across top public notebooks (0.681–0.693), community SOTA benchmarks, and recent attribution research2. First, align cross-validation with the hidden test set's prompt-multiplicity structure (k ≈ 2 passages per prompt rather than k \= 18\)2. Second, scale training using route-verified, decontaminated external pools instead of noisy self-generated scrapes2. Third, combine truncated multi-layer causal LLM and embedding backbones with real open-weight GLM-5.3 / GLM-5.3-Flash token likelihoods and source-conditioned likelihood scoring (ATLAS)3. Fourth, attack the intra-lab sibling confusions that account for roughly half to three-quarters of remaining errors using family residual heads and Damped Sinkhorn-Knopp calibration2.

## **Score Diagnosis and the Competitive Landscape**

Submissions in Name That LLM are graded 100% on Balanced Accuracy (macro-averaged recall across all 18 candidate models from OpenAI, Google, Anthropic, Qwen, Z.ai, and Moonshot AI)2. Because class counts in the \~4,000-passage hidden test set are unequal and the model mix differs between the \~40% public leaderboard and \~60% private leaderboard, every one of the 18 models receives an equal 1/18 weight4. On the 150-prompt train.csv (2,699 passages, mean 116.4 words, range 40–250 words), each model averages roughly 150 examples—too sparse for TF-IDF or standard sequence classification heads to separate within-lab siblings such as gpt-5.6-luna, gpt-5.6-sol, and gpt-5.6-terra, or claude-opus-4-6, claude-opus-4-8, claude-opus-5, and claude-sonnet-52.

| Leaderboard Tier / Method | Representative Score | Source & Vintage | Core Capability and Bottleneck |
| :---- | :---- | :---- | :---- |
| Starter TF-IDF \+ Logistic Regression | 0.322 | Kaggle Public Notebook (Oct 2026\)5 | Captures crude formatting/lexical priors; fails on unseen prompts2. |
| Classical CPU / Stylometry / TF-IDF LinearSVC | 0.348–0.364 (Your Score: 0.356) | Competitor Logs & Community Benchmark (k=2 Test)2 | Moderate 6-lab separation (0.474–0.579 family BA); fails on within-lab siblings2. |
| Train-Only DeBERTa-v3-Large / 4-Stack GBDT | 0.389–0.418 | Community SOTA Benchmark (k=2 Test)2 | Overfits the 150 training prompts unless enriched with clean external pools2. |
| Standalone Custom Qwen3-8B / ModernBERT | 0.550–0.585 | Competitor LB Runs (Oct–Nov 2026\)3 | Early/middle causal layers retain generative style far better than encoders3. |
| Dual-DeBERTa OpenPool Calibrated | 0.665 | Public LB Submission (ajikajadeh, Nov 2026\)2 | Gains \+5.33 pp unseen-prompt BA from clean pool training, but single-family cap2. |
| Multi-Backbone LoRA \+ WeMM \+ Qwen3.8-27B GSQ | 0.681–0.687 | Top 10% Public Recipes (0.684 cutoff, Nov 2026\)3 | Combines Qwen/Llama/WeMM embeddings, Dual-SVC residuals, and GSQ experts3. |
| Real GLM-5.3 / Flash Likelihood \+ Style \+ LLMs | 0.692–0.693 | Top Public SOTA Notebooks (Nov 13–14, 2026\)5 | Adds quantized/streamed GLM-5.3 & Flash likelihoods on 2xT4 to break GLM confusion3. |
| Top 1% / Top 10 / Top 1 Contenders | 0.711 / 0.749 / 0.769 | Competitor LB Tracker (NguyenThanhNhan \#1, Nov 13, 2026\)3 | Private methods unreported; likely driven by clean custom routes, generative likelihoods, and sibling tuning2. |

Before scaling architectures, audit your existing code against six submission and validation checks so silent notebook bugs do not erase future gains:

> 1. Confirm your notebook writes /kaggle/working/submission.parquet (Parquet, not CSV) with columns id,model, zero nulls, no duplicate ids, and exact string matches to the 18 valid model labels4.  
> 2. Verify that inference loads /kaggle/input/competitions/name-that-llm/test.csv dynamically rather than assuming the 1–3 fixture rows visible during interactive commits4.  
> 3. Ensure no late post-processing cell is skipped, OOM-killed, or bypassed during the \~4,000-row hidden rerun—a failure mode reported on the competition forums7.  
> 4. Audit your cross-validation splits to ensure every prompt\_id stays strictly inside a single fold (GroupKFold or StratifiedGroupKFold)9.  
> 5. Confirm all model weights and external pools are attached as offline Kaggle datasets (Internet off) and execute safely inside the 9-hour CPU/GPU runtime limit4.  
> 6. Optimize class predictions for macro-averaged recall (via calibrated probabilities or column balancing) rather than raw micro accuracy, because argmax on uncalibrated probabilities systematically under-predicts hard sibling classes2.

## **Fixing Validation First: The Hidden-Test Structural Law**

Every prompt in train.csv was answered once by each of the 18 candidate models (149 prompts with 18 rows and one prompt with 17, giving 2,699 passages and prompt multiplicity k \= 18\)2. Standard random K-Fold validation creates severe "reverse leakage": seeing 17 responses for a prompt in training teaches the model which label is absent from that prompt rather than stylistic authorship, collapsing random K-Fold scores to roughly 0.11 compared to 0.35 under prompt-grouped folds (0.355 LB)3.  
Grouping strictly by prompt\_id solves prompt leakage, yet full k \= 18 group evaluation still misleads whenever models or post-processors exploit shared-prompt context. According to the practitioner benchmark archive yauwes/name-that-llm-sota-artifacts, the \~4,000 hidden test passages span roughly 2,000 unseen prompts, yielding an average prompt multiplicity of k ≈ 2 passages per prompt rather than 182. In that benchmark, offline CV evaluated on full k \= 18 groups correlated with the Public Leaderboard at only r \= 0.13, whereas subsampling validation prompts down to k \= 2 passages raised correlation to r \= 0.912. Competitor ablations reinforce this divergence: group-aware heads gained \+6 points on complete prompt groups but lost \-3 points on split pairs3.  
Equally critical is isolating external data from your validation denominator. Mark Barney's experiment logs warn that grouped CV inflates 0.03–0.04 above the Public LB once external candidate pools enter training3. Even more starkly, the yauwes Dual-DeBERTa pipeline logged 0.929–0.937 on its internal 70/15/15 split while its actual Public LB score was 0.6652. That gap demonstrates that internal pool-contaminated validation splits measure pool consistency rather than hidden-test generalization. Train on train\_fold \+ verified\_external\_pool, but score OOF Balanced Accuracy exclusively on the official train.csv passages in each held-out prompt\_id fold, evaluated under repeated k \= 2 prompt subsamples2.

## **High-Leverage Modeling Upgrades to Reach the 0.69+ Public Frontier**

Climbing from 0.356 into the 0.68–0.693 public frontier requires simultaneous upgrades across training-pool hygiene, multi-depth causal/embedding backbones, and direct open-weight likelihood extraction2.

### **Clean, Exact-Route External Data Hygiene**

Freely and publicly available external datasets and pre-trained models are permitted under competition rules4. Scaling beyond the 150 official training prompts is essential for neural models: in yauwes experiments, adding 36,000 decontaminated Open Pool rows lifted deberta\_v3\_large unseen-prompt validation Balanced Accuracy from 0.52297 to 0.57628 (+5.33 percentage points)2. However, unvetted synthetic scrapes degrade generalization because API routing and formatting instructions alter surface fingerprints2.  
For example, prvsiyans-extended-name-that-llm-dataset (15,226 passages across 810 prompts) documents that its new\_kimi\_nothink split—served by Alibaba via OpenRouter with reasoning disabled—diverged from the competition's true kimi-k3 typography, while pilot\_range injected artificial length-range suffixes16. Earlier versions of crtproduction/name-that-llm-open-pool also contained 7,919 misrouted rows (flag\_gemini37\_routing\_risk) and 284 mislabelled rows removed in Open Pool v142. Similarly, Mark Barney logged negative results from paid self-generated API data: \$7.72 of OpenRouter Qwen passages failed to improve word-pattern models (0.481 vs 0.484), \$7.10 of bought synthetic passages underperformed control (0.580 vs 0.586), and pre-training style networks on 245,081 general writing passages boosted official-train accuracy (0.324 vs 0.278) while hurting creative-writing evaluation sets3.  
Follow four hygiene steps to scale training data cleanly without buying unverified API outputs:

> 1. Ingest crtproduction/name-that-llm-open-pool (v14 or newer, 211k passages) and drop every row flagged flag\_gemini37\_routing\_risk \== True before training2.  
> 2. Supplement high-confusion classes with vetted community sets: the new split of prvsiyans-extended-name-that-llm-dataset (excluding new\_kimi\_nothink and pilot\_range), seyitkaangunes/ntl-extra-api-passages (notably ntl\_extra\_passages\_claude.csv), and zeminglii/ntl-own-exact-route-generations (verified Kaggle proxy and first-party OpenRouter routes)2.  
> 3. Filter external passages to match train.csv's 40–250 word distribution and cover diverse genres—including code explanations and creative fiction—rather than uniform essay styles2.  
> 4. Run an adversarial domain check (training a binary classifier to separate official train.csv from each candidate external batch) and discard any provider route or prompt template that separates trivially from official completions3.

### **Multi-Depth Causal LLM and Embedding Backbones**

Encoder baselines such as RoBERTa-Large (0.3398 in yauwes k=2 tests) and DistilBERT (0.2515 OOF) struggle because pooling only the final layer captures semantic topic rather than syntactic and lexical habits2. Competitor benchmarks established that early and middle layers of causal LLMs encode authorship far better than final layers (0.39 vs 0.31 on Qwen3-8B), explaining why top teams truncate Qwen3-8B and Qwen3-14B to 24 layers and aggregate hidden states across multiple depths2. Across the Kaggle competition's Models leaderboard, top 0.679–0.684 solutions converge on diverse causal and embedding backbones including Qwen3-4B/8B/14B-base, Qwen3-Embedding-0.6B/8B, Qwen2.5-7B/14B, Llama-3.1-8B-base, Tencent/WeMM-Embedding-4B/9B, Gemma-4-12B, and Mistral/OLMo LoRAs3.

| Model / Architecture Family | Realistic k=2 Test BA | 6-Lab Family BA | Architectural Takeaway |
| :---- | :---- | :---- | :---- |
| 4-View Calibrated LinearSVC (Word \+ Char \+ CharWB \+ Pool) | 0.35752 | 0.53552 | Weak standalone, yet valuable later as a residual refinement layer (0.672–0.684 notebooks)3. |
| CatBoost / XGBoost / 4-Stack GBDT (129 Stylometric \+ SVD) | 0.4149–0.41782 | 0.5938–0.61012 | Outperforms RoBERTa-Large (0.3398) and official-only DeBERTa (0.3892)2. |
| DeBERTa-v3-Large (Official Only vs \+36k Clean Pool) | 0.3892 (+5.33 pp Val lift w/ Pool)2 | 0.5769+2 | Strong complementary family to causal LLMs (0.665 LB when blended as Dual-DeBERTa)2. |
| Qwen3-8B / Qwen3.5-9B / Qwen3-14B LoRA24 (Centred Twin) | 0.4621–0.47042 | 0.5983–0.61502 | Centred Twin and multi-depth pooling beat plain 8-layer readouts (0.4401–0.4509) by \+1.2–3.0 pp2. |
| Qwen3-14B LoRA24 (Shared-Prompt Gaussian Joint Readout) | 0.52092 | 0.70382 | \+5.82 pp jump over Centred Twin (0.4627), though shared-prompt heads weaken when k drops (0.5338 on k=18)2. |
| Multi-Backbone LoRA Stack (Core V31, k=2 Calibrated) | 0.56622 | 0.60822 | Combining diverse causal LoRA backbones lifts Top-15% high-confidence accuracy to 0.95422. |

A high-ROI, compute-efficient tactic noted in yauwes SOTA documentation is frozen multi-layer embedding readout fitting: pass all \~190,588 decontaminated Open Pool passages plus train.csv once offline through Qwen3-14B, Qwen3-8B, and Llama-3.1-8B to cache multi-layer hidden representations, then fit fast ridge or logistic linear heads (HEADS) over those embeddings (+0.004 to \+0.008 estimated Public LB gain at zero extra Kaggle inference cost)2. Pairing those causal multi-layer experts with DeBERTa-v3-Large, ModernBERT (0.55–0.56 standalone3), and WeMM-Embedding LoRA experts establishes true architectural diversity3.

### **Direct Open-Weight Likelihood Scoring**

The most significant recent leap in public scores—lifting top public notebooks from 0.681–0.684 to 0.692–0.693 as of November 13–14, 2026—came from scoring candidate passages directly under the open-weight GLM-5.3 and GLM-5.3-Flash models3. Unlike proprietary OpenAI, Anthropic, and Google endpoints, Z.ai released open weights for GLM-5.3 (744B total / 40B active MoE)21, and MindStudio's technical guide notes that GSQ \+ RCO quantization compresses the 320B GLM-5.3-Flash to \~137 GB (\~3.5 bits per parameter) within roughly two benchmark points of full precision22. Running such massive models inside Kaggle's 9-hour 2xT4 GPU budget requires careful memory engineering—public 0.693 notebooks and forum threads point to Real GLM-5.3 & GLM-5.3-Flash Likelihood on 2xT4, public GGUF segment loading, and layer streaming3. Alongside GLM likelihoods, incorporating Qwen3.8-27B GSQ (a 64-layer expert used at 3%–12% weight in 0.682 and 0.692 public notebooks and Trained27 pipelines) adds strong high-capacity discrimination2.

## **Bridging 0.693 Public SOTA to 0.749–0.769 Top-1 Contention**

Because the codebases behind \#1 NguyenThanhNhan (0.769) and \#10 kulshekhar (0.749) remain private, the exact recipe separating 0.693 from 0.769 is not publicly confirmed3. Synthesizing the competition's confusion matrices with October 2026 attribution literature highlights three mechanisms capable of pushing a strong 0.69 ensemble toward the top 10\.

### **ATLAS Source-Conditioned Clipped Likelihood**

Discriminative classifiers (P(model \| text)) frequently overfit prompt domains6. The ATLAS framework (arXiv:2610.19583, Ko et al., October 2026)—explicitly referenced in name-that-llm community benchmarks and notebooks—inverts the objective to generative modeling (P(text \| model))2. On a frozen pretrained LLM backbone, ATLAS learns a prefix embedding and a source-specific LoRA up-projection (B\_k) paired with a shared down-projection (A) for each candidate source, optimizing mean negative log-likelihood on that source's clean texts6. During inference, it evaluates token-level log-likelihoods under every source condition and clips each token's relative log-likelihood evidence so a few extreme topic tokens cannot dominate the prediction6.  
On MAABench (evaluating GPT, Claude, Gemini, Mistral, and Llama across domain and version shifts), ATLAS outperformed the strongest encoder-tuning baseline by \+2.66 percentage points on Domain shift, \+12.32 pp on Version shift, and \+9.00 pp on Joint shift, while reaching 95.71% OOD accuracy (+33.52 pp) on OpenTuringBench and surpassing multi-task baselines with just 1,000 texts per family6. One important caveat applies: MAABench evaluated five cross-lab families rather than 18 intra-lab version siblings simultaneously6, and running 18 source conditions across \~4,000 test passages on Kaggle requires sharing the frozen backbone computation and batching only the lightweight B\_k up-projections to stay well under the 9-hour runtime cap4.

### **Attacking the Intra-Lab Sibling Error Ceiling**

Across 2,015 OOF test evaluations in the yauwes benchmark, 6-lab family Balanced Accuracy already reaches 0.9683, meaning 77.78% (112 of 144\) of all errors occur between sibling models inside the same lab—consistent with Mark Barney's independent log that roughly 50% of errors are within-family2. Adding generic 18-class encoders yields diminishing returns on these sibling clusters; breaking 0.70+ demands targeted sibling residual heads and soft typographic priors.

| Confused Sibling Cluster | Documented Error Pattern | Targeted Remediation Strategy |
| :---- | :---- | :---- |
| gemini-3.8-flash vs gemini-3.7-flash | 42 of 144 errors (29.2%) in yauwes audit; traced to 7,919 misrouted Open Pool rows2. | Prune flag\_gemini37\_routing\_risk \== True before fitting Dual-SVC (--svc\_w 12\) and deb\_pool2. |
| glm-5.3 vs glm-5.3-flash (and Claude confusion) | 35 of 144 errors (24.3%) in yauwes; both labs show near-0% curly quotes and high em-dash density2. | Direct GLM-5.3 & GLM-5.3-Flash token likelihoods \+ within-family GLM residual head (0.03)2. |
| gpt-5.6-terra vs gpt-5.6-sol vs gpt-5.6-luna | Version and post-training overlap within the OpenAI 5.6 release tier2. | Qwen3.8-27B GSQ (Trained27 64-layer expert) \+ Gentle Sibling-Style residual (max\_alpha=0.08)2. |
| claude-opus-4-6 / 4-8 / 5 / sonnet-5 | Shared Anthropic register (0% curly quotes, 0.98–1.37 em-dashes per 100 words)2. | Dedicated Anthropic sibling head (Qwen3-8B LoRA Claude head) trained on prompt-paired Claude texts2. |

Typographic habits provide high-signal soft priors when modeled probabilistically rather than as brittle rules. In prvsiyan's clean 300-prompt audit, gpt-5.5 uses curly apostrophes on 98.9% of passages and curly quotes on 22.0%, and the four Google Gemini models use curly apostrophes on 56.7%–82.3% with rare curly quotes (0.3%–1.7%)16. By contrast, the five Anthropic models and two GLM models reported in that table exhibit 0%–1.1% curly apostrophes and 0% curly quotes while producing 2.7x–8.6x higher em-dash density (0.98–1.51 vs 0.18–0.37 per 100 words across those Gemini and gpt-5.5 baselines)16. Because Kaggle applied light formatting cleanup and API routes can alter punctuation7, avoid hard if/else thresholds; instead apply a Hierarchical Beta-Bernoulli log-odds prior (scale=1.2) and feed typographic ratios into a Dual-SVC or CatBoost residual stack2.

### **Ensemble Optimization and Balanced-Accuracy Post-Processing**

How predictions from diverse model families are blended and calibrated shifts Balanced Accuracy dramatically, especially when test class priors differ from training priors4.

| Ensemble / Post-Processing Method (yauwes 22-Model Ablation) | Val BA (15%) | Realistic Test BA (k=2) | Top-15% High-Conf Acc | Operational Verdict |
| :---- | :---- | :---- | :---- | :---- |
| E01 Equal Soft Voting (Arithmetic Mean) | 0.6913 | 0.6851 | 0.9930 | Suboptimal; weak classifiers dilute strong experts2. |
| E02 Rank Averaging (Quantile Blend) | 0.5937 | 0.5964 | 0.9735 | Avoid; destroys log-probability margin calibration (-0.0887 vs E01)2. |
| E06 Nelder-Mead Simplex Log-Prob Optimization | 0.9296 | 0.9340 | 1.0000 | Strong geometric blending in log-probability space2. |
| E07 E06 \+ Damped Sinkhorn-Knopp (strength=0.60, rho=0.35) | 0.9317 | 0.9371 | 1.0000 | Highest k=2 Test BA in ablation; gently balances column marginals for macro recall2. |
| E09 E07 \+ Typography (1.2) \+ Sibling (0.08), No Hungarian | 0.9322 | 0.9367 | 1.0000 | Production-safe soft post-processing (0.9683 6-lab BA)2. |
| E10 Ablation: Hard Hungarian Override (linear\_sum\_assignment) | 0.9322 | 0.8858 (-0.0509) | 0.7447 (-0.2553) | Harmful failure; forces 18 distinct labels onto fake clusters when k=22. |

Absolute 0.93 validation figures in this table reflect the optimistic split calibration of the yauwes artifact archive (whose Dual-DeBERTa component scored 0.665 on the Public LB)2, so use these results strictly for relative method selection. The decisive takeaway is that Damped Sinkhorn-Knopp scaling boosts Balanced Accuracy across uneven test sets without relying on prompt grouping, whereas Hard Hungarian bipartite matching assumes k \= 18 distinct authors per prompt cluster and drops k \= 2 accuracy by \-0.0509 (-25.53 percentage points on high-confidence predictions)2.

## **Phased 8-Week Execution Plan to January 13, 2027**

With roughly eight weeks remaining until the final submission deadline—and a January 6, 2027 cutoff for team mergers (merging up to five participants with genuinely orthogonal OOF models such as Mistral/OLMo LoRAs or exact-route embeddings is a high-leverage option)4—execute the upgrade path in five sequential stages:

> 1. **Week 1 — Fix CV & Classical Residual Baseline (Target: 0.356 \-\> \~0.42–0.45):** Lock a 5-fold StratifiedGroupKFold split on prompt\_id evaluated solely on official train.csv with k \= 2 subsampling checks2. Upgrade your 0.356 pipeline into a calibrated 4-view LinearSVC (word, char, char-WB, and stylometric features) blended with CatBoost over 129+ stylometric/typographic/SVD features (0.415–0.418 in k=2 benchmarks; public beginner ladders span 0.34–0.45)2, persisting OOF log-probabilities for final residual stacking2.  
> 2. **Weeks 2–3 — Decontaminate External Pools & Train Dual Encoders (Target: \~0.45 \-\> \~0.65):** Assemble Open Pool v14+, prvsiyans (new split), ntl-extra-api-passages, and zeminglii exact-route passages, pruning flag\_gemini37\_routing\_risk \== True and new\_kimi\_nothink2. Fine-tune DeBERTa-v3-Large (both official-only and clean-pool variants) and ModernBERT inside training folds only, verifying real OOF gains on held-out official prompt\_ids2.  
> 3. **Weeks 4–5 — Truncated Multi-Layer Causal LoRAs & Cached Readouts (Target: \~0.65 \-\> \~0.684):** Train 24-layer truncated Qwen3-8B-base and Qwen3-14B-base LoRAs with multi-depth pooling and Centred Twin readouts2, alongside Qwen3-Embedding-8B, Llama-3.1-8B-base, and WeMM-Embedding-4B/9B experts3. Cache frozen multi-layer embeddings across the clean \~190k Open Pool to fit fast regularized ridge/logistic heads (HEADS) with negligible inference overhead2.  
> 4. **Weeks 6–7 — Real GLM-5.3/Flash Likelihoods & ATLAS Source-Conditioned Scoring (Target: 0.693 \-\> 0.72–0.75+):** Integrate streamed/quantized GLM-5.3 and GLM-5.3-Flash likelihoods plus Qwen3.8-27B GSQ (3%–12% weight) budgeted tightly for Kaggle's 2xT4 9-hour limit3. Train an 18-source ATLAS likelihood scorer (prefix \+ source-specific B\_k up-projection with shared A on a frozen Qwen3-8B or Llama-3.1-8B backbone and clipped token log-likelihoods) to gain generative robustness against unseen test prompts6.  
> 5. **Week 8 (By Jan 6 Merger / Jan 13 Final) — Sibling Heads, Sinkhorn Calibration & Runtime Hardening:** Train four dedicated intra-lab sibling residual heads (Gemini 3.7/3.8, GLM-5.3/Flash, GPT-5.6 trio, Claude 4.6/4.8/5/Sonnet-5) blended gently (max\_alpha \<= 0.08)2. Optimize ensemble log-probability weights via Nelder-Mead on k \= 2 OOF CV, apply Damped Sinkhorn-Knopp (strength=0.60, rho=0.35) plus Beta-Bernoulli typography priors (scale=1.2), verify end-to-end submission.parquet generation on a synthetic 4,000-row benchmark test under 6.5–7 hours runtime, and finalize your top two diverse private-LB candidates2.

#### **Works cited**

> 1. [https\://github.com/Tienledaithandong/Name-the-LLM](https://github.com/Tienledaithandong/Name-the-LLM)  
> 2. yauwes/name-that-llm-sota-artifacts · Datasets at Hugging Face, [https\://huggingface.co/datasets/yauwes/name-that-llm-sota-artifacts](https://huggingface.co/datasets/yauwes/name-that-llm-sota-artifacts)  
> 3. Name That LLM · Research Terminal \- Mark Barney, [https\://markbarney.net/research/name-that-llm/](https://markbarney.net/research/name-that-llm/)  
> 4. Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm](https://www.kaggle.com/competitions/name-that-llm)  
> 5. Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm/code](https://www.kaggle.com/competitions/name-that-llm/code)  
> 6. Who's Behind the Text? Robust Model Authorship Attribution, [https\://komseok0109.github.io/ATLAS/](https://komseok0109.github.io/ATLAS/)  
> 7. Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm/discussion?sort=undefined](https://www.kaggle.com/competitions/name-that-llm/discussion?sort=undefined)  
> 8. Who's Behind the Text? Robust Model Authorship Attribution \- arXiv, [https\://arxiv.org/html/2610.19583v1](https://arxiv.org/html/2610.19583v1)  
> 9. Name That LLM Kaggle Competition 2026 \- ReconScribe, [https\://reconscribe.com/name-that-llm-kaggle-competition-2026/](https://reconscribe.com/name-that-llm-kaggle-competition-2026/)  
> 10. Leaderboard \- Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm/leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)  
> 11. Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm/data](https://www.kaggle.com/competitions/name-that-llm/data)  
> 12. Models \- Name That LLM \| Kaggle, [https\://www\.kaggle.com/competitions/name-that-llm/models](https://www.kaggle.com/competitions/name-that-llm/models)  
> 13. 0.693 · Real GLM-5.3 & GLM-5.3-Flash Likelihood \| Kaggle, [https\://www\.kaggle.com/code/sanpier/0-693-real-glm-5-3-glm-5-3-flash-likelihood](https://www.kaggle.com/code/sanpier/0-693-real-glm-5-3-glm-5-3-flash-likelihood)  
> 14. Name That LLM 第四周作业 \- Kaggle, [https\://www\.kaggle.com/code/xiewenwei29/name-that-llm/notebook](https://www.kaggle.com/code/xiewenwei29/name-that-llm/notebook)  
> 15. Name That LLM: the Prompt Hides the Author (CPU) \- Kaggle, [https\://www\.kaggle.com/code/patimakornsrinamen/name-that-llm-the-prompt-hides-the-author-cpu?scriptVersionId=360010467](https://www.kaggle.com/code/patimakornsrinamen/name-that-llm-the-prompt-hides-the-author-cpu?scriptVersionId=360010467)  
> 16. prvsiyan's Extended Name That LLM Dataset \- Kaggle, [https\://www\.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset)  
> 17. Name That LLM Open Pool: 211k Clean Passages \- Kaggle, [https\://www\.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/discussion](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool/discussion)  
> 18. Name That LLM: Extra API Passages \- Kaggle, [https\://www\.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)  
> 19. NTL own exact-route generations \- Kaggle, [https\://www\.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations](https://www.kaggle.com/datasets/zeminglii/ntl-own-exact-route-generations)  
> 20. Dual-SVC Residual Refinement Apex \- Kaggle, [https\://www\.kaggle.com/code/ozertuu/0-672-dual-svc-residual-refinement-apex](https://www.kaggle.com/code/ozertuu/0-672-dual-svc-residual-refinement-apex)  
> 21. 25 Best GLM-5.3 Prompts for Coding & Web Development, [https\://promptsrush.com/blog/best-glm-5-3-coding-prompts](https://promptsrush.com/blog/best-glm-5-3-coding-prompts)  
> 22. Run GLM 5.3 Flash Locally: GSQ and RCO Quantization Explained, [https\://www\.mindstudio.ai/blog/glm-5-3-flash-gguf-gsq-rco-local](https://www.mindstudio.ai/blog/glm-5-3-flash-gguf-gsq-rco-local)  
> 23. Name That LLM 0.669: All Public \+ What Failed \- Kaggle, [https\://www\.kaggle.com/code/alexchilton/name-that-llm-0-667-all-public-what-failed](https://www.kaggle.com/code/alexchilton/name-that-llm-0-667-all-public-what-failed)  
> 24. Name That LLM: Beginner's Ladder from 0.34 to 0.45 \- Kaggle, [https\://www\.kaggle.com/code/sometimessubodh/name-that-llm-beginner-s-ladder-from-0-34-to-0-45](https://www.kaggle.com/code/sometimessubodh/name-that-llm-beginner-s-ladder-from-0-34-to-0-45)