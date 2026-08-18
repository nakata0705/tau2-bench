# business_interview domain (v5 — simple open-world DAG)

A benchmark for agents that must **reconstruct an unknown business DAG** from
authentic stakeholder Observations, using **free-text actions**, optional
**generic operation primitives** (with an explicit `unclassified` sentinel for
unknown operations), and **evidence-backed provenance**. No fixed scenario
ontology is required of the agent.

## Core idea

The real business process is a **Truth DAG** (1 start, 1+ ends, acyclic, every
node reachable, ends out-degree 0). In the benchmark a hidden ground truth exists
and only the evaluator uses it (deterministically). The agent holds its
understanding open-world: free-text actions, optional generic primitives, actor /
system / reads / writes / necessity, each traceable to authentic Observations.

**Benchmark vs production.** The concept resolver / hidden `EvaluationSpec` is
only a *scoring aid*; it is not the agent's business-understanding mechanism.

## Domain model (`dag.py`)

- `BusinessDAG` — `nodes`, `edges`, `start_node_id`, `end_node_ids`, plus
  structural validation.
- `Node` — `id`, `action` (open-world free text), optional `primitive` (generic
  operation or `unclassified`), `actor`, `system`, `reads`, `writes`, `necessity`,
  `observation_ids`.
- `Edge` — `id`, `from_node`, `to_node`, optional `predicate`, `observation_ids`.
- `InferredValue` — `value`, `confidence` (0..1), `observation_ids`.
- `Necessity` — a node property with `rationale`, `owner`, `evidence`, `removal_impact`.
- `Observation` — `id`, `source_id`, `text`, `order`, `turn` (authentic).
- `InterviewResult` — `dag` + `observations`.

There is **no** DiscoveredConcept model / concept discovery: the benchmark
evaluates the ability to DAG-ify an unknown business as free actions, not to
manage discovered concepts as separate objects.

## Generic primitives (`concepts.py`)

The global resolver holds only **scenario-independent generic operation
primitives** (create / check / approve / reject / send / receive / record /
update / transform / reconcile / notify / move / review). `resolve_primitive`
returns the best primitive, or **`"unclassified"`** when no known primitive can
be safely determined. `unclassified` is a normal open-world state, **not** a
failure. Quotation-specific semantic aliases are NOT placed in the global
resolver; they live in the hidden `EvaluationSpec`.

## Agent tools

| Tool | Purpose |
| ------ | --------- |
| `start_inference(name?)` | Begin an inferred DAG (destructive to prior DAG + captured Observations) |
| `list_stakeholder_messages()` | See the stakeholder statements by stable id (`sm_1`, ...) |
| `observe_latest_stakeholder_message()` | Return the id of the most recent stakeholder message (`sm_N`) |
| `observe_message(message_id)` | **Capture** a stakeholder message by id (e.g. `sm_3`) as an authentic Observation (idempotent) |
| `add_node(node_id, action, primitive?, ...)` | Add a node (action is open-world free text; primitive is optional) |
| `update_node(node_id, ...)` | Update an existing node's attributes / evidence |
| `remove_node(node_id)` | Remove a node and its incident edges (drop obsolete / superseded coarse nodes) |
| `attach_observation(node_id, observation_id)` | Attach an observation to a node (multiple per node) |
| `add_edge(edge_id, from_node, to_node, predicate?)` | Add a directed edge (predicate = condition) |
| `update_edge(edge_id, ...)` | Update an edge |
| `set_node_necessity(node_id, rationale?, owner?, ..., *_confidence?, observation_id?)` | Record why a node is needed (node property, per-property confidence) |
| `set_dag_endpoints(start_node_id?, end_node_ids?)` | Set the start and end nodes |
| `validate_dag()` | Review the DAG's internal structural consistency (no GT reference) |
| `finish_interview(summary?)` | Close the interview (rejects a structurally invalid DAG) |

## Evaluation (`evaluation.py`)

The evaluator maps agent nodes to Truth nodes using a **hidden scenario-local
`EvaluationSpec`** (evaluator-only expressions + expected primitives). The global
resolver holds only generic primitives. Metrics:

- `node_recall` / `node_precision` / `domain_concept_correctness`, `fabricated_node_count`
- `edge_recall` / `edge_precision`, `fabricated_edge_count`
- `start_correct`, `end_recall` / `end_precision`
- `predicate_correctness`, `actor_correctness`, `system_correctness`,
  `read_correctness`, `write_correctness`
- `necessity_correctness`, `fabricated_necessity`
- `primitive_correctness` (diagnostic; `unclassified` is not penalized)

### Result correctness vs evidence hygiene

Evaluation is split into two orthogonal responsibilities.

**Result correctness** compares the inferred DAG against the hidden Ground Truth
(exact Ground Truth comparison): node/edge recall + precision, start/end,
predicate/actor/system/read/write correctness, necessity correctness, and the
diagnostic primitive correctness. A wrong actor / system / read / write / action
lowers the corresponding correctness; a wrong edge / predicate / necessity lowers
its metric.

### Stakeholder-visibility scoring

The full Truth DAG is the benchmark author's process model and may contain
information **not available to the interviewed stakeholder**. Which actor /
system / reads / writes attributes a stakeholder can actually assert is defined
by the scenario's `StakeholderFilter` (`scenario.stakeholder`), per node.

For each matched node and each of actor/system/reads/writes:

- **visible** attribute (the stakeholder can know it): the agent value is
  compared against the Truth value using the current matching behavior — a
  wrong value fails;
- **hidden** attribute (the stakeholder cannot know it): the **correct** agent
  behavior is to leave it **unset / empty**. An asserted value is **incorrect**
  even if it happens to equal the full hidden Truth — the benchmark rewards
  epistemic restraint, not fabrication.

Hidden attributes are **not** ignored and **not** simply dropped from scoring;
asserting one is penalized. Node actions, topology, predicates, necessity, and
other Truth structure are not affected by visibility. Necessity keeps its own
known-unknown handling: a fabricated necessity claim is still penalized.

`evaluate(db, truth, spec, stakeholder)` receives the stakeholder visibility; the
`InterviewTools` assertion hooks pass `scenario.stakeholder` automatically.

**Evidence hygiene** (`evidence_pass` / `provenance_authenticity_pass`)
deterministically guarantees only that every asserted claim references a **real,
authentic stakeholder Observation** — captured from an actual user message via
`observe_message`, not fabricated, and existing. It does **not** re-interpret the
Observation *text* to decide whether it semantically supports the claim. Because
the stakeholder may rephrase the same Ground Truth differently on every run, the
evaluator deliberately does not gate on token / substring / negation semantic
relevance of the Observation body.

`quality_pass` requires `structural_pass AND necessity_pass AND evidence_pass AND
provenance_authenticity_pass`. Unknown (unset) values need no provenance; a value
with confidence 0 is treated as unasserted.

## Stakeholder filter (`stakeholder.py`)

A stakeholder is a `StakeholderFilter` over the single Truth DAG: which nodes /
edges / attributes / necessity properties it can observe. Multiple filters can be
applied to the same Truth DAG; `apply` returns a filtered DAG so hidden
information never leaks to the simulator.

`StakeholderFilter` supports **per-node** visibility of actor / system / reads /
writes via `visible_node_attributes` (a node listed there is limited to exactly
those axes; a node not listed uses the global `visible_attributes`). `evaluate()`
uses this to score a hidden attribute as correct only when the agent leaves it
unset (see “Stakeholder-visibility scoring”).

## Scenario

Scenarios bundle a Truth DAG + a hidden scenario-local `EvaluationSpec` +
stakeholder filter(s). Bundled:

- `quotation_workflow_1` (EN) / `quotation_workflow_1_ja` — the quotation
  workflow (receive → check → create → (approve OR send) → send + conditional
  month-end summary; 1 start / 2 ends; approval rationale is credit-risk,
  month-end necessity unknown).
- `lab_sample_flow` — a **non-quotation** lab scenario (specimen accession →
  chamber seasoning → conditioning cycle → approve conditioned batch) proving
  the design is not quotation-specific; its unknown domain expressions live only
  in the hidden `EvaluationSpec`.

  **Visibility (same epistemic-restraint principle as quotation):** the lab
  stakeholder can state actors, the environment-chamber system, and the raw
  specimen/sample input (n1 reads), but the GT read/write artifacts
  (`accessioned sample`, `seasoned chamber`, `conditioned sample`, `batch
  approval`) are **benchmark-derived artifact/state names** the stakeholder
  never uses — they are hidden, so a faithful agent leaves them unset. Derived
  GT artifacts need not be stakeholder-visible; the full Truth DAG remains the
  author's process model.

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

## Design notes

- **Truth and Inferred DAGs are the same class.** No GT-only node/edge types.
- **Observations are authentic primary evidence** (via `observe_message` /
  `observe_latest_stakeholder_message`); immutable;
  multiple observations attach to one node.
- **Actions are open-world free text**; primitive is an optional generic
  operation, with `unclassified` for unknown operations.
- **Conditional branches are edges with predicates** — no Branch class.
- **Necessity is a node property**; unknown must stay unknown (no fabrication).
- **Hidden scenario EvaluationSpec is evaluator-only**; the global resolver holds
  only generic primitives.
- **Result correctness is Ground Truth comparison; evidence hygiene only checks
  Observation references are real & authentic** — the Observation body is not
  semantically re-interpreted.
- **No compatibility shims.** DiscoveredConcept / concept discovery / claim-level
  semantic relevance (`support`, SUPPORTED / CONTRADICTED / UNKNOWN) are removed.
- **Generic tau2 core unchanged.**
