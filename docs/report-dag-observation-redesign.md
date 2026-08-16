# Report: Business Interviews as Evidence-Backed DAG Inference (v3)

This report documents the breaking redesign of `business_interview` from a
"workflow Step restoration" benchmark to an **"observation-based business DAG
inference"** benchmark. The domain model itself was replaced: `Step` /
`Transition` / `Branch` are gone; the business process is a `BusinessDAG` and
each stakeholder statement is an immutable `Observation` that the agent
integrates into an inferred DAG.

Scope: entirely `src/tau2/domains/business_interview/`,
`data/tau2/domains/business_interview/`,
`tests/test_domains/test_business_interview/`, and `docs/`. **The generic tau2
core is unchanged** (verified via `git diff`: only business_interview files).

---

## 1. Problems with the old model

- The old model treated the process as a flat set of **Steps / Transitions /
  Branches**, which forced branch-specific evaluation and a `concept -> single
  step` matching bias.
- **Truth and Agent result used different types** (`GTStep`/`GTTransition` vs
  `WorkflowStep`/`Transition`), so the objective workflow and the reconstruction
  were not directly comparable as the same object.
- **Observations were not first-class evidence.** Confidence / provenance were
  bolted on as "asked" flags rather than being inherent to an inferred value.
- **Conditional flow lived on nodes / a branch concept**, not uniformly on edges.
- The benchmark evaluated a "workflow reconstruction", not the incremental
  integration of observations into a DAG.

## 2. New domain semantics

The real business process is a **Truth DAG** (1 start, 1+ ends, acyclic, all
nodes reachable from the start, end nodes out-degree 0). A **stakeholder is a
filter** over that single Truth DAG — they observe part of it in natural
language. An interview yields **Observations** (independent, immutable evidence).
The agent integrates observations into an **Inferred DAG**, attaching multiple
observations to a node and updating its attributes / confidence, creating a node
only when no existing node corresponds.

## 3. Truth DAG / Node / Edge

`BusinessDAG` (`dag.py`) has `nodes`, `edges`, `start_node_id`, `end_node_ids`,
plus structural validation (cycle, reachability, dangling edges, end out-degree).
`Node`: `id`, `action`, `actor`, `system`, `reads`, `writes`, `necessity`,
`observation_ids`. `Edge`: `id`, `from_node`, `to_node`, optional `predicate`,
`observation_ids`. Unconditional flow is `predicate=None`; conditional branches
are **multiple outgoing edges with different predicates** (no Branch class, no
node-level condition).

## 4. Truth and Inferred are the same class

`scenario.truth` is a `BusinessDAG`; the agent's result is a `BusinessDAG`. There
are **no GT-only node/edge classes** (`GTStep`/`GTTransition`/`GTBranch`/
`GTNecessity`/`StakeholderStepTruth` are deleted). On the Truth side
confidence/provenance are simply unset.

## 5. Observation

`Observation` is independent evidence: `id`, `source_id`, `text`, `order`,
`locale`. Multiple observations attach to one node (`Node.observation_ids`, and
per-attribute `InferredValue.observation_ids`). The tools implement
record-observation → attach-to-node / update-node → create-node-only-when-needed.

## 6. Incremental node update

`add_node` creates a node; `update_node` / `set_node_necessity` update an
existing node's attributes / necessity with a new observation, **merging**
observation provenance (multiple observations can support one attribute) rather
than replacing it.

## 7. Necessity

`Necessity` is a node property (0 or 1 per node) with `rationale`, `owner`,
`evidence`, `removal_impact`, each an `InferredValue` — an integrated estimate
across observations with its own confidence and provenance. Unknown reasons stay
unknown (a fabricated rationale fails); the old `why_asked` / `challenge_done`
state is gone from the final DAG (question history is an interview trace, not the
business DAG).

## 8. Confidence / provenance

`InferredValue { value, confidence [0,1], observation_ids }`. Confidence is per
attribute (action / actor / system / predicate / necessity property differ),
validated to [0,1] (out-of-range is rejected), and surfaced in diagnostics. No
calibration scoring (non-goal).

## 9. Stakeholder filter

`StakeholderFilter` (`stakeholder.py`) expresses a stakeholder as visibility over
the single Truth DAG: visible nodes / edges / attributes / necessity. `apply`
returns a filtered `BusinessDAG` so hidden information never leaks; the full Truth
DAG is never serialized into a simulator prompt. Multiple filters can apply to the
same Truth DAG (sales vs finance shown in tests).

## 10. Concept is evaluator-only

Concepts (`concepts.py`) are **not domain objects** and are not stored on nodes.
`EvaluationSpec.truth_node_concepts` (evaluator-only) maps Truth node id →
bilingual concept id, so "approve high-value quotation" and
"100万円超の見積を営業部長が承認" are judged equivalent, and arbitrary agent node ids
never leak into the score.

## 11. Evaluator

`evaluation.py` maps agent nodes → Truth nodes by action concept, then compares
edges. Metrics (replacing `step_recall` / `transition_accuracy` / `branch_recall`):
`node_recall` / `node_precision`, `edge_recall` / `edge_precision`,
`start_correct`, `end_recall` / `end_precision`, `predicate_correctness`,
`actor_correctness`, `system_correctness`, `read_correctness`,
`write_correctness`, `necessity_correctness`, `fabricated_node_count`,
`fabricated_edge_count`, `fabricated_necessity`. No branch-specific evaluator.
`quality_pass` = `structural_pass` AND `necessity_pass`; `protocol_pass` is
independent.

## 12. Interview Result

`InterviewResult` = `{ dag: BusinessDAG, observations: list[Observation] }` — the
current business understanding plus the evidence it is based on.

## 13. Quotation migration

The bundled `quotation_workflow_1` scenario (EN) and `quotation_workflow_1_ja`
were migrated to the DAG model:

- Nodes: receive request (`r`), check customer (`cc`), create quotation (`cq`),
  approve high-value quotation (`ap`), send quotation (`sq`), month-end accounting
  summary (`me`).
- Edges: `r→cc`, `cc→cq`, `cq→ap [amount > 1,000,000]`,
  `cq→sq [amount ≤ 1,000,000]`, `ap→sq`, `cq→me [month-end]`. 1 start (`r`), 2 ends
  (`sq`, `me`). The month-end summary is expressed as a conditional tail edge with
  a predicate (a natural reorganization for a 1-start / 2-end DAG).
- Necessity: approval rationale is credit-risk (confirmed); the month-end step's
  rationale / owner / evidence / removal are unknown (must not be fabricated).

## 14. Falsification A-AE

The test file (`test_dag_business_interview.py`, 42 tests) covers:

| ID | Case | Result |
|----|------|--------|
| A | Truth and Agent result use the same BusinessDAG class | PASS |
| B | No Truth-only GT classes | PASS |
| C | 1 start / multiple ends valid | PASS |
| D | cycle rejected | PASS |
| E | unreachable node rejected | PASS |
| F | dangling edge rejected | PASS |
| G | condition only on Edge.predicate (no Node.condition) | PASS |
| H | no Branch class | PASS |
| I | no Transition class | PASS |
| J | no Step / WorkflowStep class | PASS |
| K | multiple observations attach to one node | PASS |
| L | observation updates existing node (no duplicate) | PASS |
| M | observation alone does not create a duplicate node | PASS |
| N | observation keeps source_id and text | PASS |
| O | necessity is a node property | PASS |
| P | necessity has multiple-observation provenance | PASS |
| Q | per-property confidence differs | PASS |
| R | out-of-range confidence rejected | PASS |
| S | arbitrary agent node ids → full pass | PASS |
| T | missing truth node → node recall drops | PASS |
| U | fabricated node → precision drops | PASS |
| V | wrong edge → edge metrics drop | PASS |
| W | wrong predicate → predicate correctness drops | PASS |
| X | branch evaluated as multiple outgoing edges | PASS |
| Y | EN / JA equivalent DAG evaluation | PASS |
| Z | hidden truth not leaked (policy / tools / agent prompt / scenario) | PASS |
| AA | two stakeholder filters on one Truth DAG | PASS |
| AB | out-of-filter information not leaked to simulator | PASS |
| AC | correct necessity → PASS | PASS |
| AD | fabricated necessity → FAIL | PASS |
| AE | unknown promoted to fact → FAIL | PASS |

## 15. Tests / core tests

- `uv run pytest tests/test_domains/test_business_interview/` → **42 passed**.
- `make check-all` (ruff lint + format) → **All checks passed**.
- `make test` (core) → **271 passed, 1 xfailed, 1 failed**. The single failure is
  `tests/test_run.py::test_run_tasks_nl_assertions` — the mock domain with live
  `gpt-3.5-turbo`, unrelated to this change (only business_interview files
  changed) and flaky/network-dependent: it **passes in isolation**. Not a
  regression.

## 16. EN / JA real run (DeepSeek)

DeepSeek (`deepseek/deepseek-chat`, `DEEPSEEK_API_KEY` set) ran the `base` split:
6 simulations (3× EN, 3× JA) through the new evaluator. The evaluator ran
normally and produced all required metrics on real agent output, e.g. one run
achieved `node_recall 1.0, edge_recall 0.67, necessity_correctness 1.0,
fabricated_necessity False`; others showed `dag_valid False`, fabricated nodes /
edges, lower necessity correctness — all surfaced in diagnostics. The reward was
0 because the bundled deepseek-chat agent rarely follows the full tool protocol
(some runs made zero tool calls and just chatted), so observation generation /
DAG formation could not be observed from the live LLM in those runs.

The full observation → attach → DAG formation → conditional edges → necessity →
evaluator comparison flow is verified end-to-end by the reference trajectory
(`test_evaluator_rewards_full_reconstruction`), which replays the reference
actions (record_observation, add_node, add_edge with predicates,
set_node_necessity with confidence, set_dag_endpoints, finish) and scores reward
1.0. Observation generation / attach and incremental node update are also covered
by the unit tests (K, L, M, N, P, Q).

## 17. Generic core diff

Zero. `git diff` touches only business_interview files. Deleted:
`data_model.py`, `ground_truth.py`, `semantic.py`, and the old workflow test file.
Added: `dag.py`, `evaluation.py`, `scenario.py`, `stakeholder.py`,
`test_dag_business_interview.py`. Updated: `concepts.py`, `aliases.py`,
`environment.py`, `tools.py`, `README.md`, `policy.md`, `tasks.json`.

## 18. Remaining limitations

- **Node merge / split is out of scope.** Same-concept multiple nodes are
  distinguished only by actor/system/data tie-break; two nodes identical on all
  of those but differing by graph position would be ambiguous.
- **No sophisticated confidence calibration.** Confidence is stored, validated,
  and shown in diagnostics, but not used for scoring.
- **The bundled deepseek-chat agent is weak** at the tool protocol, so real-run
  reward is low; observation generation is verified via the reference trajectory
  and unit tests, not by the live LLM.
- **The month-end summary edge was reorganized** from the old "send → month-end"
  to "create → month-end [month-end]" to obtain a natural 1-start / 2-end DAG.
- **Stakeholder simulator instructions are hand-authored** (not generated from the
  filter), so a future scenario must keep them consistent with its filter; the
  leakage tests guard this.

## 19. Readiness for a second scenario

The model is scenario-generic: adding a second workflow means authoring its Truth
`BusinessDAG` + `EvaluationSpec` + a `StakeholderFilter` (and a matching
`user_scenario`), with no evaluator changes expected. The infrastructure
(shared DAG class, observations, per-attribute confidence, filter, evaluator
metrics) is reusable as-is.
