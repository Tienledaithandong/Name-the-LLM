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

These are the project's own conservative policies. They are stricter than, or
explicitly bounded by, the verified rules.

| Question | Ruling | Basis |
|---|---|---|
| AI coding assistants writing code | **Allowed.** Assistants write/refactor code, tests and docs. | Rules restrict sharing, not tooling; host allows external tools/AMLT with appropriate licences. |
| Sending competition rows to hosted assistants | **Not allowed.** Only schema, aggregates, configs and logs leave the machine. | Owner's data-privacy instruction in the plan; keeps the ruling independent of any future change to provider terms. |
| External model inference to label test text | **Not used.** No API calls at inference time; internet is disabled during the graded run anyway. | Code Requirements; also avoids Reasonableness ambiguity. |
| Paid API calls | **Not used.** Would require owner approval plus a written Reasonableness assessment. | Rules §2.6. |
| Public writeups / code | **Allowed on Kaggle** for this competition; publish the report there, not via private channels. | Rules 6.b/2.6. |
| Pretrained encoders | **Allowed if freely and publicly available and attached** (no download at runtime). The argmax blend (D009/D012, submitted 0.356) is sparse/stylometric and uses no encoder. MiniLM was benchmarked for H3 authorship and **not adopted** for authorship (ADR D013), but is **adopted for prompt-group clustering** in the crossed-design submission candidate (ADR D016): the `package --crossed` dataset attaches the checkpoint offline and the notebook loads it from disk. See `docs/external_data_register.md`. | Code Requirements. |
| Transductive test-time clustering of test passages | **Allowed.** Automatically clustering test text (no labels, no human input) to recover prompt groups, then within-group assignment, is permitted. The Rules forbid only *hand-labelling / human prediction* of test records (Foundational §4.b), not label-free feature-space processing; §6 allows public pretrained models and AMLT. No rule prohibits transductive learning. Verified by reading the Rules page 2026-11-14. Final acceptance remains the owner's submission decision. | Foundational Rules §4.b; §6 (External Data / AMLT); absence of any transductive prohibition. |
| External text corpora labelled by model | **Not used yet.** Any use must be registered in `docs/external_data_register.md`, deduplicated against train, and validated on the official grouped folds. | Rules §2.6 + plan §6. |

## 10. Still unverified (blocking final submission only)

- [x] Interactive test-file row count: the downloaded `test.csv` has **3 rows**,
      matching the Data page; the Overview's "single example row" wording is the
      outlier. The code remains agnostic to the count.
- [x] Exact public/private split ratio: the leaderboard page states **public ≈ 40%
      of test, private ≈ 60%** (read directly 2026-11-14). Verified top score
      **0.771**, 5,851 teams.
- [ ] Hidden-test prompt multiplicity **k**: NOT officially published. Community
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
