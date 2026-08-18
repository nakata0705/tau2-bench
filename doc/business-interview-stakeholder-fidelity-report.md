# business_interview — Stakeholder Fidelity Report

**Date:** 2026-08-18
**Branch:** `business-interview`
**Scope:** Stakeholder simulator **prompt / response policy** only. Evaluator,
node matching, Observation capture, necessity, and scenario Ground Truth were
**not** changed.

---

## 1. Observed failures before the fix (real DeepSeek runs, seeds 1000–1004)

Two classes of Ground Truth deviation:

1. **Unsupported facts (common-sense completion).** The stakeholder invented a
   business fact not in the Ground Truth — a customer-information-missing /
   error branch:
   - seed 1002: *"If the customer information isn't found in the CRM or there's
     no pricing on file, I'd have to look into it — probably check with someone
     or ask the customer..."*
   - seed 1004: *"If the customer information isn't found in the CRM, we'd need
     to look into it and get the correct details before we can proceed..."*
   When the agent DAG-ified this, it became a **fabricated node** (`resolve_*`).

2. **False denial of a known fact.** The stakeholder denied the approval branch
   that *is* in its Known info:
   - seed 1004: *"No, there aren't any other steps or branches between creating
     and sending."* (the approval branch exists).

## 2. New Stakeholder fidelity rules

Added explicit rules to the stakeholder **task prompt** (the per-task
`user_scenario.instructions.task_instructions` in `tasks.json`). The rules are
deliberately short and DeepSeek-friendly (no state machine), built around a few
guiding maxims:

- **Ground Truth only / no speculation**
  - Your Known info is the ONLY source of business facts.
  - *Plausible does not mean known* — do NOT add any process step / branch /
    condition / actor / system / data / reason / behavior not in your Known info.
  - *Do not use general business common sense to fill gaps* (explicitly:
    "for example, do not invent what happens if customer information is missing").
  - If asked about something not in your Known info, do not guess — say you do
    not know.
- **No false denial + negative-answer check**
  - Never deny a fact that IS in your Known info.
  - Before a negative answer ("No", "nothing else", "no special cases"),
    re-check whether any Known fact is relevant to the question, and mention it.
  - "Not having been asked yet" is different from "it does not exist": keep
    unasked facts to yourself, but never deny them.
- **Progressive disclosure**
  - Answer only what the question's scope calls for; do not dump everything.
  - normal flow → normal flow only; exceptions/conditions/thresholds/approvals →
    approval branch; periodic/recurring/monthly/month-end → month-end summary;
    reason for month-end → unknown.
- **Unknown handling**
  - Keep known facts and unknown attributes distinct: month-end summary **exists**
    (KNOWN), why it is required (**UNKNOWN**). For UNKNOWN, never invent a reason.

The existing `###STOP###` / don't-end-the-conversation behavior was preserved.

## 3. Prompt change summary

- `data/tau2/domains/business_interview/tasks.json` → `task_instructions` updated
  for **quotation_workflow_1** (EN), **quotation_workflow_1_ja** (JA), and
  **lab_sample_flow** with the fidelity rules above. The `known_info` /
  `unknown_info` (Ground Truth facts) are unchanged.
- No change to the global user-simulator guidelines (avoids affecting other
  domains) and no change to scenario Ground Truth / `scenario.py`.
- Helper: `scripts/update_stakeholder_fidelity_prompts.py` (one-shot, committed).

## 4. Deterministic prompt/config tests (no LLM output hard-coded)

Added to `tests/test_domains/test_business_interview/test_dag_business_interview.py`:

- `test_stakeholder_ground_truth_only_no_speculation_rule` — "answer only from
  your Known info", "plausible does not mean known", "do not use general business
  common sense to fill gaps", "do not add any process step".
- `test_stakeholder_no_false_denial_of_known_fact` — "never deny a fact that IS in
  your Known info".
- `test_stakeholder_negative_answer_relevant_fact_check` — "before giving a
  negative answer", "re-check whether any known fact is relevant".
- `test_stakeholder_progressive_disclosure_no_volunteering` — "do not dump
  everything at once", "keep unasked facts to yourself", flow/approval/month-end
  bullets.
- `test_stakeholder_unknown_no_speculation` — "if asked about something not in
  your Known info, do not guess", "never invent, guess, or speculate".
- `test_stakeholder_quotation_instructions_consistent_with_truth` — approval →
  credit risk; month-end exists but reason unknown ("you do not know the reason",
  "vague impression that Accounting needs it").
- `test_stakeholder_instructions_ja_present` / `_lab_present` — JA and lab task
  prompts carry the fidelity rules.

These check the **prompt/config text**, not the LLM's natural-language output.

## 5. Verification

- `pytest tests/test_domains/test_business_interview/` → **59 passed** (51 + 8 new
  fidelity prompt tests), including quotation reference trajectory (`reward ==
  1.0`), lab scenario full pass, EN/JA equivalence, node-matching and
  observation-capture regressions.
- `make check-all` (ruff lint + format) → clean.

## 6. Real DeepSeek smoke (5 runs, seeds 3000–3004)

Interview Agent + Stakeholder both `deepseek/deepseek-chat` (→ `deepseek-v4-flash`).

### Per-run fidelity classification (`artifacts/business_interview_real_llm/stakeholder_fidelity_classification.json`)

| run | approval | month-end | unknown rationale | failures |
|-----|----------|-----------|-------------------|----------|
| run_00_seed3000 | ✅ disclosed | ✅ | ✅ unknown kept | `none` |
| run_01_seed3001 | ✅ disclosed | ✅ | ✅ unknown kept | `improper_negative_answer` (self-corrected) |
| run_02_seed3002 | ✅ disclosed | ✅ | ✅ unknown kept | `none` |
| run_03_seed3003 | ✅ disclosed | ✅ | ✅ unknown kept | `premature_disclosure` (mild) |
| run_04_seed3004 | ✅ disclosed | withheld (not asked) | n/a | `none` |

### Result summary (0–5 per class)

| class | count | note |
|-------|-------|------|
| `unsupported_fact` | **0** | was 2/5 pre-fix (customer-info-missing branch) |
| `contradicted_known_fact` | **0** | was 1/5 pre-fix (approval denied) |
| `improper_negative_answer` | 1 | run_01: "no other steps/branches" then self-corrected in the next turn |
| `premature_disclosure` | 1 | run_03: month-end volunteered in a send-step question (mild) |
| `fabricated_unknown` | **0** | no invented month-end rationale anywhere |
| `none` | 3 | run_00, run_02, run_04 |

### Key checks from the objective

1. **Unsupported customer-info error branch** — **gone** (0/5; 3 mentions in 2
   pre-fix runs).
2. **Approval disclosed when asked about exceptions/conditions** — **5/5**.
3. **Month-end disclosed when asked about month-end/periodic** — yes where asked
   (runs 00, 01, 02, 03); **withheld** when never asked (run_04) — correct
   progressive disclosure, no denial.
4. **Month-end rationale kept unknown** — preserved in every run that discussed it
   ("I don't know why... no documentation... vague impression").
5. **No premature full dump on normal-flow questions** — approval sometimes
   surfaced while exploring a step (mild, natural), but month-end was never dumped
   unless relevant; no run dumped the whole Ground Truth unprompted.
6. **No contradictory negative answer** — approval was never denied; the one
   transient "no other steps/branches" (run_01) was immediately self-corrected.
7. **Natural variation** — wording/order/emphasis differed run to run (e.g. how
   approval and month-end were phrased), so variability is preserved.

### Remaining failures
- **run_01** `improper_negative_answer` (self-corrected): said "I don't have any
  other steps or branches" in one turn, then disclosed approval + month-end the
  next. Minor; no persistent false denial.
- **run_03** `premature_disclosure` (mild): volunteered the month-end summary
  while answering about the send step. Correct fact, slightly early.

Neither is a clear or persistent Ground Truth deviation (no unsupported facts, no
fabricated unknowns, no final false denial).

### Is a response fidelity self-check needed next?
Not strictly required — no clear GT deviation persisted across the 5 runs and the
two residual issues were self-corrected / mild. However, because 2/5 runs showed a
minor residual (a transient negative before self-correction, and a mild premature
disclosure), a **response fidelity self-check (second-pass)** is a reasonable next
candidate for further robustness. **Per scope, it is recorded here as a
recommendation and NOT implemented in this task.**

## 7. Scope / non-goals respected

- No DisclosureController / DisclosureTrace, deterministic fact router, embeddings,
  LLM judge in the evaluator, or second-pass self-check was added.
- Evaluator, node matching, Observation capture, necessity, and scenario Ground
  Truth were not changed.
- This measures how far prompt/policy changes alone stabilize stakeholder fidelity.

## 8. Deliverables

- Prompt: `data/tau2/domains/business_interview/tasks.json`
- Helper: `scripts/update_stakeholder_fidelity_prompts.py`
- Fidelity classifier: `scripts/stakeholder_fidelity_classify.py`
- Tests: `tests/test_domains/test_business_interview/test_dag_business_interview.py`
- Smoke artifacts: `artifacts/business_interview_real_llm/run_*_seed300*.json`
- Fidelity summary: `artifacts/business_interview_real_llm/stakeholder_fidelity_classification.json`
- This report: `doc/business-interview-stakeholder-fidelity-report.md`
