# business_interview domain (v5 — simple open-world DAG)

A benchmark for agents that must **reconstruct an unknown business DAG** from
authentic stakeholder Observations, using **free-text actions**, optional
**generic operation primitives** (with an explicit `unclassified` sentinel for
unknown operations), and **claim-level evidence provenance**. No fixed scenario
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
|------|---------|
| `start_inference(name?)` | Begin an inferred DAG (destructive to prior DAG + captured Observations) |
| `list_stakeholder_messages()` | See the stakeholder statements and their turn indices |
| `observe_turn(turn_idx)` | **Capture** a stakeholder (user) message at a turn as an authentic Observation (idempotent) |
| `add_node(node_id, action, primitive?, ...)` | Add a node (action is open-world free text; primitive is optional) |
| `update_node(node_id, ...)` | Update an existing node's attributes / evidence |
| `attach_observation(node_id, observation_id)` | Attach an observation to a node (multiple per node) |
| `add_edge(edge_id, from_node, to_node, predicate?)` | Add a directed edge (predicate = condition) |
| `update_edge(edge_id, ...)` | Update an edge |
| `set_node_necessity(node_id, rationale?, owner?, ..., *_confidence?, observation_id?)` | Record why a node is needed (node property, per-property confidence) |
| `set_dag_endpoints(start_node_id?, end_node_ids?)` | Set the start and end nodes |
| `finish_interview(summary?)` | Close the interview |

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

### Evidence & claim-level relevance

Every asserted claim must be traceable to an authentic recorded Observation
(`evidence_pass` / `provenance_authenticity_pass`), and each claim is evaluated
**independently** against its own value (`support(observation, claim_kind,
claim_value) -> SUPPORTED / CONTRADICTED / UNKNOWN`): action, primitive, actor,
system, each read, each write, edge relation, predicate, and each necessity
property. An observation that only supports one claim cannot substitute for
another; an edge needs a relation word + both endpoint actions; a predicate needs
its own direction/value; an explicit negation is CONTRADICTED (not SUPPORTED).

`quality_pass` requires `structural AND necessity AND evidence AND
provenance_authenticity AND relevance`. Unknown (unset) values need no
provenance; a value with confidence 0 is treated as unasserted.

## Stakeholder filter (`stakeholder.py`)

A stakeholder is a `StakeholderFilter` over the single Truth DAG: which nodes /
edges / attributes / necessity properties it can observe. Multiple filters can be
applied to the same Truth DAG; `apply` returns a filtered DAG so hidden
information never leaks to the simulator.

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
- **Observations are authentic primary evidence** (via `observe_turn`); immutable;
  multiple observations attach to one node.
- **Actions are open-world free text**; primitive is an optional generic
  operation, with `unclassified` for unknown operations.
- **Conditional branches are edges with predicates** — no Branch class.
- **Necessity is a node property**; unknown must stay unknown (no fabrication).
- **Hidden scenario EvaluationSpec is evaluator-only**; the global resolver holds
  only generic primitives.
- **Claim-level relevance** evaluates each claim against its own value.
- **No compatibility shims.** DiscoveredConcept / concept discovery are removed;
  the old Step / Transition / Branch model is removed.
- **Generic tau2 core unchanged.**
