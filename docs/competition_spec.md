# Competition specification — Name That LLM

**Verified:** 2026-11-14 (all pages read directly from kaggle.com)
**Sources**

| Page | URL |
|---|---|
| Overview / Description / Timeline / Evaluation / Prizes / Code Requirements | https://www.kaggle.com/competitions/name-that-llm |
| Data | https://www.kaggle.com/competitions/name-that-llm/data |
| Rules | https://www.kaggle.com/competitions/name-that-llm/rules |

Anything not listed as verified below is **unverified** and must not be treated as
fact by code or by the report.

## 1. Task

Identify which of 18 frontier language models wrote each unlabeled passage.
Passages are authentic, self-contained responses to prompts on philosophy,
physics, mathematics, biology, computer science and creative prose; lengths vary
between 40 and 250 words. The hidden test set is ~4,000 passages written for
prompts that **do not appear in `train.csv`**.

Providers: OpenAI, Google, Anthropic, Qwen, ZAI, Moonshot AI (6 labs).

## 2. Metric (verified: Evaluation)

**Balanced accuracy**, i.e. the mean over all candidate models of that model's
recall (per-class recall). K = 18.

- Test passages per model are **not equal**, and the model mix differs between
  public and private leaderboards. Balanced accuracy weights every model equally.
- Consequence for local evaluation: pooled out-of-fold balanced accuracy is the
  primary comparison metric; plain accuracy would be dominated by the largest
  class and is not a valid proxy.

## 3. Labels (verified: Evaluation → Valid Labels)

`gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`,
`gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`,
`gemini-3.8-flash`, `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`,
`claude-opus-5`, `claude-sonnet-5`, `qwen3.8-max`, `glm-5.3`,
`glm-5.3-flash`, `kimi-k3`.

These are hard-coded once in `src/constants.py`; every artifact stores the same
order, and `submission.py` rejects anything outside the set.

## 4. Data (verified: Data page)

| file | contents |
|---|---|
| `train.csv` | 2,699 passages covering 150 prompts. Columns: `id` (str), `prompt_id` (str), `text` (str), `model` (str, ground truth). |
| `test.csv` | Columns: `id` (str), `text` (str) only. Interactive sessions get a tiny example file; the graded rerun replaces it with ~4,000 rows. |
| `sample_submission.parquet` | Correct format: one row per `id`, columns `id`, `model`. |

- Dataset size listed: 2.14 MB; license **CC BY-SA 4.0**.
- Only light formatting cleanup was applied (e.g. removing a whole-passage
  `**bold**` wrapper).
- **Discrepancy noted:** the Overview says the interactive `test.csv` has "a
  single example row"; the Data page says "three example rows". The code never
  assumes a row count, so either is fine.

## 5. Submission (verified: Evaluation → Submission File)

- Code competition: the notebook reads
  `/kaggle/input/competitions/name-that-llm/test.csv` and writes
  `submission.parquet` **to the notebook working directory**.
- Columns `id`, `model`; header required; every `id` exactly once.
- A submission is **rejected** if it is missing `id`/`model`, contains nulls,
  repeats an `id`, or predicts a label outside the 18.

## 6. Runtime and environment (verified: Code Requirements)

- Notebooks only; "Submit" requires the code-competition conditions.
- CPU notebook ≤ 9 hours; GPU notebook ≤ 9 hours.
- **Internet access disabled.**
- **Freely & publicly available external data is allowed, including
  pre-trained models.**
- Submission file must be named `submission.parquet`.

## 7. Competition rules (verified: Rules page)

- Team size ≤ **5**; one team per participant; team merger deadline same as entry
  deadline. Submissions ≤ **5 per day**; up to **2 Final Submissions**.
- External Data is allowed if publicly available and equally accessible at no
  cost to other participants, or "reasonably accessible" / "minimal cost" under
  the host's Reasonableness Standard. Paid access that excludes other
  participants is not acceptable.
- Private sharing of competition code or data outside a team is prohibited;
  public sharing is allowed on the competition's own forum/notebooks and is then
  licensed OSI-approved open source.
- No hand-labelling or human prediction of validation/test records; no
  information from leaked test labels; one account only.

## 8. Timeline (verified: Overview → Timeline)

| Milestone | Date (11:59 PM UTC) |
|---|---|
| Start | 2026-10-14 |
| Entry deadline | 2027-01-06 |
| Team merger deadline | 2027-01-06 |
| Final submission deadline | 2027-01-13 |

The plan's 2027-01-13 target matches the verified final submission deadline.

## 9. Compliance rulings for this project

**D024 project constraint:** the competition-rules interpretation below does not
override this repository's stricter hard rule: no transductive steps in reported
scores. Historical transductive results are exploratory and cannot support
selection. Hidden-test prompt multiplicity remains unverified.

These are the project's own conservative policies. They are stricter than, or
explicitly bounded by, the verified rules.

| Question | Ruling | Basis |
|---|---|---|
| AI coding assistants writing code | **Allowed.** Assistants write/refactor code, tests and docs. | Rules restrict sharing, not tooling; host allows external tools/AMLT with appropriate licences. |
| Sending competition rows to hosted assistants | **Not allowed.** Only schema, aggregates, configs and logs leave the machine. | Owner's data-privacy instruction in the plan; keeps the ruling independent of any future change to provider terms. |
| External model inference to label test text | **Not used.** No API calls at inference time; internet is disabled during the graded run anyway. | Code Requirements; also avoids Reasonableness ambiguity. |
| Paid API calls | **Not used.** Would require owner approval plus a written Reasonableness assessment. | Rules §2.6. |
| Public writeups / code | **Allowed on Kaggle** for this competition; publish the report there, not via private channels. | Rules 6.b/2.6. |
| Pretrained encoders | **Allowed if freely and publicly available and attached** (no download at runtime). The argmax blend (D009/D012, submitted 0.356) and the current cap-40k pool-MIX candidate (D028/D037) are sparse/stylometric and use **no encoder**. MiniLM was benchmarked for H3 authorship and **not adopted** (ADR D013). It was later adopted for prompt-group clustering in the crossed-design candidate (ADR D016), but that candidate is **WITHDRAWN** (D017/D018: its gain was a k=18 artifact and the path is transductive, which this repository forbids in reported scores). `package --crossed` survives only as a historical research artifact and `kaggle/submission.ipynb` **rejects** an `inference.json` marked `crossed: true`. **Fine-tuned encoders:** DeBERTa-v3 **base** and **large** (both **MIT**) were fine-tuned on Kaggle GPU (D023/D027) and **not adopted** — official-only large scores 0.191501 on the identity-aligned folds and adds no blend diversity; the 0.4442 pool blend used the D026-disabled transductive screen. The training kernel fetches checkpoints with internet **on**, which is a training-time download; inference is offline (`predict.py` sets `HF_HUB_OFFLINE=1`) and **no encoder ships in any package**. ModernBERT-large (Apache-2.0) is registered as **planned only** for H8/D054 and has never been downloaded. See `docs/external_data_register.md`. | Code Requirements. |
| Transductive test-time clustering of test passages | **Allowed.** Automatically clustering test text (no labels, no human input) to recover prompt groups, then within-group assignment, is permitted. The Rules forbid only *hand-labelling / human prediction* of test records (Foundational §4.b), not label-free feature-space processing; §6 allows public pretrained models and AMLT. No rule prohibits transductive learning. Verified by reading the Rules page 2026-11-14. Final acceptance remains the owner's submission decision. | Foundational Rules §4.b; §6 (External Data / AMLT); absence of any transductive prohibition. |
| External text corpora labelled by model | **In use for the candidate, under a non-transductive screen.** Open Pool **v15** (`crtproduction/name-that-llm-open-pool`, 226,191 rows, CC BY-SA 4.0) is registered in `docs/external_data_register.md`, screened **once** against the full official train by `src.pool.decontaminate_global` (exact text + word-TF-IDF cosine > 0.3 → 226,049 clean; 0 exact, 142 near) and capped at 40,000 rows for the char member only (D028/D037). The keep-mask is fold-independent, so this is **not** the per-fold transductive route D026 disabled. Validated on the official grouped folds under three protocols (s42 0.4427 / s13 0.4304 / nested 0.4426); **packaged, not submitted** — the confirmed LB 0.356 fallback remains official-only. | Rules §2.6 + plan §6; ADRs D019/D026/D028/D037. |

## 10. Still unverified (blocking final submission only)

- [x] Interactive test-file row count: the downloaded `test.csv` has **3 rows**,
      matching the Data page; the Overview's "single example row" wording is the
      outlier. The code remains agnostic to the count.
- [x] Exact public/private split ratio: the leaderboard page states **public ≈ 40%
      of test, private ≈ 60%** (read directly 2026-11-14). Verified top score
      **0.771**, 5,851 teams.
- [ ] Hidden-test prompt multiplicity **k**: NOT officially published (a direct
      Discussion question, 758516, had **0 replies** as of the 2026-11-18 sweep
      — see §11.1). Community
      sources converge on **k≈1–3** (~2,000 prompts × ~2 passages), and local
      honest k-subsampled CV confirms the crossed-Hungarian gain collapses from
      +0.10 at k=18 to ~0 at k=2 (ADR D017, `docs/strategy_review.md`). Treated as
      high-confidence but unverified; the k-robust argmax/Sinkhorn path is safe at
      any k.
- [ ] Whether any GPU quota applies beyond the 9-hour runtime limit (the page
      states only the runtime limit; no weekly quota is published there).
- [ ] Official confirmation that a standalone technical report may be published
      outside Kaggle (rules permit public code sharing on Kaggle; publishing
      elsewhere should be treated as needing a host check).
- [ ] The owner's Kaggle account must accept the competition rules before the
      data can be downloaded; that action is owner-only.

**Exit gate status:** spec recorded; no modelling code relies on an unverified
item. Owner approval is required before downloading data or submitting.

## 11. Discussion sweep (2026-11-18, plan item L31)

Read directly in the signed-in in-app browser, logged out of any account action.
**Everything below is a community claim, not an official host statement.** No
organizer replied in any thread read. URLs are recorded so each claim can be
re-checked; none of it is used as a selection signal for our own models.

### 11.1 The unverified items in §10 stay unverified

- **Hidden-test passages per prompt (k).** Asked directly as
  *"Test set structure: passages per prompt?"*
  ([758516](https://www.kaggle.com/competitions/name-that-llm/discussion/758516),
  6 days before the sweep): **0 comments, no host answer.** The question also asks
  whether a test-side prompt identifier is released and how the per-model mix
  differs between public and private. All three remain unanswered, so §10's k
  bullet stays open and the k-robust argmax path stays the safe default.
- **GPU quota.** No host statement found. One participant asserts "up to 30 hrs of
  G4 card weekly per person + 9 hours submits per team"
  ([749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447)).
  **Unverified**; the only published limit remains the 9-hour runtime.

### 11.2 Independent corroboration of the D017 withdrawal

[757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855)
("Notes from a solo pipeline", MIT code, notebook v3 self-reported LB 0.389)
reports **"Hungarian assignment within prompt groups gave +0.10 CV but +0.002
LB"** and that the gain mostly vanishes under unequal per-model counts. That is
the same collapse our own k-subsampled stress test measured (+0.09/+0.10 at k=18
to ~0 at k=2, ADR D017/D018) and it is the strongest external support for keeping
the crossed/Hungarian lever **withdrawn**.

One claim there runs *opposite* to our stated rationale and is **not reconciled**:
the author reports random CV as *pessimistic* (0.104 vs 0.315 prompt-grouped),
whereas `docs/experiment_protocol.md` §1 argues row-wise random folds would be
*optimistic* (prompt memorisation). Both can be true for different models; we have
not measured it, so neither number is adopted. Grouped folds remain mandatory.

### 11.3 A community OOF/LB ladder under our own protocol

[759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103)
contains a self-reported table from a participant using **prompt-grouped CV**
(their words: "prompts split across folds"), balanced accuracy throughout:

| approach (their run) | OOF | public LB |
|---|---|---|
| TF-IDF + logistic regression | 0.31 | 0.32 |
| + external data + class weights | 0.46 | 0.40 |
| + stylometric features stacked with XGBoost | 0.49 | 0.43 |
| DeBERTa-v3-large | 0.50 | – |
| ModernBERT-large | 0.55 | 0.55 |
| UltraBERT-380M on 241k external rows | 0.60 | – |
| equal blend of encoders + TF-IDF stack | 0.625 | 0.616 |

**These are not our measurements and must never be entered in our registry as
results.** Their value is calibration and hypothesis generation:

- Their TF-IDF+logreg row (0.31 OOF / 0.32 LB) is close to our own `char_official`
  0.3193 CV, and their external-data row (0.46 OOF / 0.40 LB) sits just above our
  pool candidate's 0.4427 CV. Our one measured pair is CV 0.3760 -> LB 0.356
  (-0.020); theirs suggests a gap of -0.06 at that level. **A CV-to-LB offset of
  roughly -0.02 to -0.06 is plausible, so the candidate's 0.4427 CV is consistent
  with an LB near 0.38-0.42, not 0.44.** This is an expectation, not a prediction,
  and only #38 can settle it.
- **ModernBERT-large is reported at ~0.55 by two independent participants**
  ([759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103)),
  above their DeBERTa-v3-large 0.50. Our own DeBERTa-v3-large measured far lower
  (0.2482 official-only, D023/D027). **Backbone choice is therefore a live,
  testable hypothesis, not a settled negative** — see D052.
- Non-LLM methods are reported to **flatten around 0.60-0.62**, and two
  participants say more external data eventually *degrades* results
  ([759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103),
  [749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447),
  where an 8th-placed participant reports models that stopped improving as data
  grew). Consistent with our cap sweep finding a binding budget, not unlimited
  headroom.
- A 17th-placed participant states they reached the top 10 **without paid data
  generation** ([749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447)).

### 11.4 Typography: the strongest family signal, and a check on our own code

Two threads independently describe near-deterministic typographic rules:

- [759103](https://www.kaggle.com/competitions/name-that-llm/discussion/759103):
  "Gemini almost never uses em-dashes, Claude and GLM use em-dashes but never
  curly quotes, GPT uses curly quotes"; family accuracy **87-99%**, within-family
  version accuracy **67-76%**; **"keep the raw characters, and don't normalise
  punctuation."**
- [757855](https://www.kaggle.com/competitions/name-that-llm/discussion/757855):
  negative invariants mined from train with per-rule chance level < 1e-4 —
  unspaced em dash absent from gemini-3.1/3.6/3.8; straight apostrophe absent from
  gpt-5.6-sol; straight double quotes absent from gpt-5.6-luna/sol/terra,
  gpt-6-astra, qwen3.8-max and gemini-3.7.

**Verified against our own code (not a community claim):** the adopted
`char_tfidf_logreg` member already preserves raw characters —
`src/features_tfidf.py:47` sets `lowercase=False` on the `char` analyzer (and
`:67` on `char_wb`), and neither `src/features_tfidf.py` nor `src/data.py` applies
unicode normalisation or punctuation stripping. So this advice is **already
satisfied**; no defect was found. The family/version split also matches our own
measured within-provider error share rising 40% -> 67% with passage length (D041).

The untried part is using such rules as **hard family-level vetoes** rather than
soft features. That is a new hypothesis (D052), not an adopted method: any rule
mined on train must be fitted per fold and validated under the registered
protocol before it can touch a submission.

### 11.5 Public code worth reading (not yet read, not adopted)

- `pengyipeng1/0-687-it-actually-works` — Apache-2.0, self-reported scored **0.687**
  (submission 58260754), built on **Areeb Ali's public v11 recipe** plus a
  "row-local, family-mass-preserving official-style residual"
  ([759102](https://www.kaggle.com/competitions/name-that-llm/discussion/759102)).
  The author explicitly states their local residual experiment **failed its own
  confidence/robustness admission checks** and that one public score does not
  establish a causal gain over the 0.684 base — the same discipline as our ADRs.
- A **GLM token-likelihood** notebook by Jose Padilla, reported scored **0.689**
  (version 4) in the same thread. Related:
  [757199](https://www.kaggle.com/competitions/name-that-llm/discussion/757199)
  on running GLM-5.3 / GLM-5.3-Flash GGUF segments inside the 9-hour limit.
- `github.com/pedrohbenetton/kaggle-name-that-llm` (MIT) — the solo pipeline above.

**Score claims in this section are self-reported by participants and were not
verified against the leaderboard by us.** Our own verified numbers remain
LB 0.356 (submitted) and CV 0.4427 (candidate, not submitted).
