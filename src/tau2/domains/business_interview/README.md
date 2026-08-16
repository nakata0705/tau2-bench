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
| `set_step_unknown(step_id, note?)` | Record an UNKNOWN necessity |
| `challenge_step(step_id, question)` | Question a step's necessity |
| `record_necessity_detail(step_id, owner?, evidence?, requirement_type?)` | Record who requires / evidence |
| `propose_improvement(step_id?, kind, note?)` | Record an improvement (order-aware) |
| `finish_interview(summary?)` | Close the interview |

The agent uses its own step ids (`s1`, `s2`, ...) and its own action text; the
evaluator matches reconstructed steps to the ground truth by content, so this is
**not** a game of guessing hidden ids.

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
- `actor_accuracy`, `system_accuracy`
- `data_read_accuracy`, `data_write_accuracy`
- `transition_accuracy`
- `branch_recall`, `branch_condition_accuracy`
- `rationale_coverage`, `uncertainty_handling`, `fabricated_rationale`
- `challenge_done`, `improvement_order_ok`

## Necessity / challenge model

Each step carries a `Necessity`: whether a rationale is known, who stated it /
owns the requirement, whether evidence was identified, the epistemic status
(FACT / BELIEF / UNKNOWN), whether it was challenged, and the challenge
questions. `FACT` means the source asserted it as certain — it is **not**
objective truth; the objective rationale state is in the ground truth.

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

The tests cover the workflow tools, leakage-free agent-visible context, the
workflow falsification suite A-L, and an end-to-end EnvironmentEvaluator check.

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
