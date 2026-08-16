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
| `set_step_rationale(step_id, content, epistemic_status, source?)` | Record why a step is needed (FACT/BELIEF) |
| `set_step_unknown(step_id, note?)` | Record an UNKNOWN necessity (marks investigated) |
| `challenge_step(step_id, dimension, question)` | Investigate one necessity dimension (why / owner / evidence / removal / deletion) |
| `record_necessity_detail(step_id, owner?, evidence?, removal_impact?, requirement_type?)` | Record who requires / evidence / removal impact |
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
- `challenge_target_identified`, `why_investigated`, `owner_investigated`,
  `evidence_investigated`, `removal_investigated`, `deletion_considered`,
  `challenge_done`, `improvement_order_ok`

## Necessity / challenge model

Each step carries a `Necessity` with an explicit **investigation vs outcome**
split so that "the agent did not ask" is never confused with "the agent asked
and got UNKNOWN / NONE":

- `investigated` — the "why is this needed?" question was asked and its result
  recorded (including a recorded UNKNOWN).
- `owner_investigated` / `evidence_investigated` / `removal_investigated` /
  `deletion_considered` — the corresponding necessity question was asked
  (regardless of whether a concrete answer exists).
- `rationale` / `epistemic_status` / `source` / `owner` / `evidence` — the found
  values. An investigated "no owner" / "no evidence" / "reason unknown" is a
  valid finding; recording an UNKNOWN as a FACT is an epistemic fail.

`FACT` means the source asserted it as certain — it is **not** objective truth;
the objective rationale state and the expected rationale content / source are in
the ground truth.

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
stakeholder truth completeness, arbitrary step-id canonicalisation, NONE vs
UNKNOWN separation, confirmed-rationale content/source/status evaluation, EN/JA
semantic equivalence, precision-aware data, condition-aware transitions, and the
workflow falsification suite A-R.

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
