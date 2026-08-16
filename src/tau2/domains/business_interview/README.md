# business_interview domain (v2 — Workflow-first)

A benchmark for evaluating agents that must **reconstruct a hidden business
workflow from an interview** with a stakeholder, capture each step's necessity,
and **question the requirement** before proposing improvements.

This is a breaking redesign (v2). The central concept is a **Workflow graph** —
not a bag of facts. The previous Fact / Exception / Rationale-centric model,
the old `record_fact` API, legacy `rationale_status`/`rationale_correct`, the old
`Topic` enum and topic fallback, and the old quotation_* scenarios are removed.

## What the benchmark evaluates

1. Can the agent reconstruct a hidden workflow (steps, order, branches) from the
   interview?
2. For each step, can it gather actor, system, data read/written, condition, and
   what comes next?
3. Can it capture the rationale / necessity of each step (a confirmed rationale,
   an UNKNOWN rationale)?
4. Can it question a weakly-evidenced (legacy) step's necessity — asking why it
   exists, who requires it, and what evidence supports it — before suggesting
   automation?
5. Does it follow the improvement order Question → Delete → Simplify → Accelerate
   → Automate (not jumping straight to RPA/AI)?

## Scenario

One scenario is bundled: `quotation_workflow_1` (EN) and its Japanese variant
`quotation_workflow_1_ja` (pre-localized). It is a small workflow:

- 6 steps: receive request → check customer in CRM → create quote in quoting
  system → (branch) approve high-value quote → send quote → (conditional) send
  month-end Excel summary to Accounting.
- 1 confirmed rationale: the high-value approval is for credit risk management
  (the stakeholder knows this as certain).
- 1 UNKNOWN rationale: the month-end Excel step — the stakeholder does not know
  why it exists (only a vague belief that Accounting needs it, no evidence).
- 1 questionable legacy step: the month-end Excel step → the challenge target.

The agent-visible policy and tools are **general** (workflow discovery, necessity
questioning, improvement order). They contain **no** scenario-specific answers
(no step list, no branch answer, no "credit risk", no canonical ids). The
evaluator-only ground truth lives in domain code (`ground_truth.py`).

## Files

```
src/tau2/domains/business_interview/
├── data_model.py    # Workflow / WorkflowStep / Transition / Branch / Necessity / WorkflowDB
├── environment.py   # get_environment(), get_tasks(), get_tasks_split()
├── ground_truth.py  # evaluator-only canonical workflow (never agent-visible)
├── semantic.py      # workflow-reconstruction evaluator + diagnostics
├── tools.py         # agent tools + assertion helpers + diagnostics hook
└── utils.py         # data paths
data/tau2/domains/business_interview/
├── policy.md        # general BA + workflow guidance (no leakage)
├── tasks.json       # the workflow scenario (EN + JA)
└── split_tasks.json # base / base_en / base_ja
```

## Agent tools

The agent reconstructs the workflow with a natural tool set (no hidden canonical
ids):

| Tool | Purpose |
|------|---------|
| `create_workflow(name, trigger?, purpose?, outcome?)` | Create the workflow |
| `add_step(step_id, action, actor?, system?, reads?, writes?, condition?)` | Record a step |
| `connect_steps(from_step, to_step, condition?)` | Record a transition / conditional path |
| `add_branch(from_step, condition, paths)` | Record an explicit branch |
| `set_step_rationale(step_id, content, epistemic_status, source?)` | Record the "why" result (FACT / BELIEF / UNKNOWN) |
| `set_step_unknown(step_id, note?)` | Record an explicit UNKNOWN "why" result |
| `challenge_step(step_id, dimension, question)` | **Ask** one necessity question (why / owner / evidence / removal / deletion). Asking alone records nothing — the answer must be recorded separately. |
| `record_necessity_detail(step_id, owner?, owner_state?, evidence?, evidence_state?, removal_impact?, removal_state?, requirement_type?)` | **Record** the owner / evidence / removal observation results (KNOWN / UNKNOWN / NONE_FOUND) |
| `propose_improvement(step_id?, kind, note?)` | Record an improvement (order-aware) |
| `finish_interview(summary?)` | Close the interview |

The agent uses its own step ids (`s1`, `s2`, ...) and its own action text; the
evaluator resolves actions to language-independent *concepts* and maps the
agent's step ids to the ground-truth ids, so arbitrary ids and equivalent EN/JA
free text both score identically. This is **not** a game of guessing hidden ids.

## Evaluation

The scalar reward is the product of a few env assertions:

1. `assert_finish_interview` — protocol.
2. `assert_workflow_reconstructed` — the reconstruction structurally matches the
   ground truth (steps + transitions + branches + actor/system/data).
3. `assert_rationale_handled` — confirmed rationale captured and UNKNOWN
   preserved (no fabricated rationale).
4. `assert_necessity_challenged` — the questionable step was challenged with the
   correct improvement order.

`interview_quality_pass` (in diagnostics) = the last three; `protocol_pass` is
independent. All granular metrics are surfaced via `get_eval_diagnostics`:

- `step_recall`, `unexpected_step_count`
- `trigger_accuracy`, `purpose_accuracy`, `outcome_accuracy`
- `actor_accuracy`, `system_accuracy`
- `data_read_recall` / `data_read_precision`, `data_write_recall` / `data_write_precision`
- `transition_accuracy` (from/to **and** condition)
- `branch_recall`, `branch_condition_accuracy`
- `rationale_coverage`, `confirmed_rationale_ok`, `uncertainty_handling`, `fabricated_rationale`
- `challenge_target_identified`, `why_asked` / `why_recorded`, `owner_asked` / `owner_recorded` / `owner_result`, `evidence_asked` / `evidence_recorded` / `evidence_result`, `removal_asked` / `removal_recorded` / `removal_result`, `deletion_considered`, `challenge_done`, `improvement_order_ok`

## Necessity / challenge model

Each step carries a `Necessity` that **separates ASKED from RESULT RECORDED** on
every necessity dimension. The benchmark evaluates whether a *result was
recorded*, not merely whether a question was posed.

- `why_asked` / `owner_asked` / `evidence_asked` / `removal_asked` — the question
  was posed (set by `challenge_step`). Asking alone records nothing.
- `rationale_result` (FACT / BELIEF / UNKNOWN / NOT_RECORDED) — the recorded
  "why" outcome (`set_step_rationale` / `set_step_unknown`).
- `owner_result` / `evidence_result` / `removal_result` (KNOWN / UNKNOWN /
  NONE_FOUND / NOT_RECORDED) — the recorded observation outcomes
  (`record_necessity_detail`).
- `deletion_considered` / `deletion_candidate` — the **analyst's own assessment**, kept separate from the stakeholder observations.

`UNKNOWN` means the stakeholder was asked and confirmed they do not know, and
that was recorded. `NONE_FOUND` means it was investigated and nothing exists.
An asked-but-unrecorded dimension is `NOT_RECORDED` — it is **never** treated as
`UNKNOWN` / `NONE_FOUND`. `KNOWN` **requires a value** — a `KNOWN` result with no
value is invalid (the tool rejects it; the evaluator also treats it as not
recorded). `FACT` means the source asserted it as certain — it is **not**
objective truth; recording an UNKNOWN as a FACT is an epistemic fail.

## Observation correctness (ASKED / RECORDED / CORRECT)

The challenge is judged in three tiers, not two:

1. **asked** — the question was posed (`*_asked`).
2. **recorded** — a valid result was recorded (`*_result`).
3. **correct** — the recorded result matches the ground-truth expectation
   (`owner_correct` / `evidence_correct` / `removal_correct`).

For the questionable legacy step the ground truth declares the expected
observations (`GTNecessity.expected_owner_result` etc.). For the bundled
scenario: owner `UNKNOWN`, evidence `NONE_FOUND`, removal `UNKNOWN`. A recorded
`KNOWN` that contradicts the expectation (e.g. an invented owner) is recorded but
**incorrect**. `challenge_pass` requires every required observation to be
recorded **and correct**, plus deletion considered.

Diagnostics surface all three tiers per dimension, e.g.:

```
owner:   { asked: true, recorded: true, result: KNOWN,  correct: false }
evidence: { asked: true, recorded: true, result: NONE_FOUND, correct: true }
```

## Stakeholder Truth (structured, step-unit)

Stakeholder Truth is structured per step (by concept) and axis
(`StakeholderTruth` / `StakeholderStepTruth` in `ground_truth.py`), kept separate
from the Evaluator Ground Truth and the Agent Reconstruction. Completeness
(`stakeholder_truth_completeness`) is a **structural** per-step comparison
between the evaluator-required axes and each step's truth — a word that happens
to appear in another step can never satisfy a missing axis. A separate
consistency check (`missing_from_instructions`) verifies the user simulator can
answer every structured truth value, so we never create a state where the
evaluator has truth the simulator cannot answer. The structured truth is never
shown to the agent.

## Same-concept multi-step matching

More than one ground-truth step may share an action concept. Step matching
resolves a reconstructed step to a concept, then disambiguates same-concept
candidates deterministically by priority: **1. action concept, 2. actor/system,
3. data**. No LLM judge, no hidden step ids. A `same_concept_workflow` scenario
(not a task) exercises this in the falsification tests.

## Running

```bash
tau2 run --domain business_interview --task-split base_en \
  --agent-llm <model> --user-llm <model> --num-trials 3 --max-concurrency 1
```

EN / JA and comparison runs use the `base_en` / `base_ja` / `base` splits.

## Verification

```bash
uv run pytest tests/test_domains/test_business_interview/
```

The tests cover the workflow tools, leakage-free agent-visible context,
structured step-unit stakeholder truth and structural completeness,
ASKED / RECORDED / CORRECT observation correctness, KNOWN-value validation, the
UNKNOWN / NOT_RECORDED / NONE_FOUND distinction, the same-concept matcher None
fix, arbitrary step-id canonicalisation, EN/JA semantic equivalence,
precision-aware data, condition-aware transitions, and the workflow
falsification suite A-V.

## Design notes

- **Workflow structure is the object of evaluation.** Steps, transitions,
  branches, actor, system, data and conditions are compared structurally against
  the ground truth.
- **Epistemic layer is secondary.** Source / Claim / FACT / BELIEF / UNKNOWN
  describe who knows what about a step's necessity; they do not override the
  workflow structure.
- **No leakage.** The agent sees only general policy + tools; the stakeholder
  sees only their own knowledge; the full ground truth is evaluator-only.
- **No compatibility shims.** Old scenarios, APIs, metrics and fields are
  removed, not retained for backward compatibility.
- **Generic core unchanged.** This is entirely domain / data / tests / docs.
