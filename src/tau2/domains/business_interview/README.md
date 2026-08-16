# business_interview domain (v4 — generic primitives + open-world concept discovery)

A benchmark for agents that must **discover an unknown business DAG** from
observations during an interview, using **generic operation primitives** and
**open-world discovered concepts** rather than any fixed scenario ontology.

## Core idea

The real business process is a **Truth DAG** (1 start, 1+ ends, acyclic, every
node reachable, ends out-degree 0). In the benchmark a hidden ground truth
exists and only the evaluator uses it (deterministically). The agent holds its
understanding open-world: free-text actions, optional generic primitives, and
agent-discovered ``DiscoveredConcept``s — independent of any quotation-specific
or domain-specific ontology. Each stakeholder statement is an **Observation**
(authentic, immutable evidence).

**Benchmark vs production.** The concept resolver / hidden ``EvaluationSpec`` is
only a *scoring aid*; it is not the agent's business-understanding mechanism.

## Domain model (`dag.py`)

- `BusinessDAG` — `nodes`, `edges`, `concepts` (discovered), `start_node_id`,
  `end_node_ids`, plus structural validation.
- `Node` — `id`, `action` (open-world text), optional `primitive` (generic
  operation), optional `concept_id` (→ a DiscoveredConcept), `actor`, `system`,
  `reads`, `writes`, `necessity`, `observation_ids`.
- `Edge` — `id`, `from_node`, `to_node`, optional `predicate`, `observation_ids`.
- `InferredValue` — `value`, `confidence` (0..1), `observation_ids`.
- `DiscoveredConcept` — `id`, `label`, `description?`, `aliases[]`, `observation_ids`, `confidence`.
- `Necessity` — a node property with `rationale`, `owner`, `evidence`, `removal_impact`.
- `Observation` — `id`, `source_id`, `text`, `order`, `turn` (authentic).
- `InterviewResult` — `dag` + `observations`.

## Agent tools

| Tool | Purpose |
|------|---------|
| `start_inference(name?)` | Begin an inferred DAG (destructive to prior DAG + captured Observations) |
| `list_stakeholder_messages()` | See the stakeholder statements and their turn indices |
| `observe_turn(turn_idx)` | **Capture** a stakeholder (user) message at a turn as an authentic Observation (idempotent) |
| `discover_concept(concept_id, label, description?, aliases?, confidence?, observation_id?)` | **Discover** (or merge into) an open-world domain concept |
| `add_node(node_id, action, primitive?, concept_id?, ...)` | Add a node (action is open-world; primitive is the optional generic operation) |
| `update_node(node_id, ...)` | Update an existing node's attributes / evidence |
| `attach_observation(node_id, observation_id)` | Attach an observation to a node (multiple per node) |
| `add_edge(edge_id, from_node, to_node, predicate?)` | Add a directed edge (predicate = condition) |
| `update_edge(edge_id, ...)` | Update an edge |
| `set_node_necessity(node_id, rationale?, owner?, ..., *_confidence?, observation_id?)` | Record why a node is needed (node property, per-property confidence) |
| `set_dag_endpoints(start_node_id?, end_node_ids?)` | Set the start and end nodes |
| `finish_interview(summary?)` | Close the interview |

## Evaluation (`evaluation.py`)

The evaluator maps agent nodes to Truth nodes using a **hidden scenario-local
``EvaluationSpec``** (evaluator-only expressions + expected primitives), NOT a
global ontology. The global resolver (`concepts.py`) holds only reusable generic
primitives. Metrics:

- `node_recall` / `node_precision` / `domain_concept_correctness`, `fabricated_node_count`
- `edge_recall` / `edge_precision`, `fabricated_edge_count`
- `start_correct`, `end_recall` / `end_precision`
- `predicate_correctness`, `actor_correctness`, `system_correctness`,
  `read_correctness`, `write_correctness`
- `necessity_correctness`, `fabricated_necessity`
- `primitive_correctness` (separate from domain-concept correctness)
- concept discovery: `discovered_concept_recall`, `discovered_concept_precision`,
  `fabricated_concept_count`, `duplicate_concept_count`, `concept_discovery_pass`

### Evidence & claim-level relevance

Every asserted claim must be traceable to an authentic recorded Observation
(`evidence_pass` / `provenance_authenticity_pass`), and each claim's provenance
is checked for relevance (SUPPORTED / CONTRADICTED / UNKNOWN) so that clearly-
unrelated or partially-poisoned evidence fails (`relevance_pass`).

`quality_pass` requires `structural AND necessity AND evidence AND
provenance_authenticity AND relevance AND concept_discovery`. Unknown (unset)
values need no provenance; a value with confidence 0 is treated as unasserted.

## Stakeholder filter (`stakeholder.py`)

A stakeholder is a `StakeholderFilter` over the single Truth DAG: which nodes /
edges / attributes / necessity properties it can observe. Multiple filters can be
applied to the same Truth DAG; `apply` returns a filtered DAG so hidden
information never leaks to the simulator. The full Truth DAG is never serialized
into a simulator prompt.

## Scenario

Scenarios bundle a Truth DAG + a hidden scenario-local ``EvaluationSpec`` +
stakeholder filter(s). Bundled:

- `quotation_workflow_1` (EN) / `quotation_workflow_1_ja` — the quotation
  workflow (receive → check → create → (approve OR send) → send + conditional
  month-end summary; 1 start / 2 ends; approval rationale is credit-risk,
  month-end necessity unknown).
- `lab_sample_flow` — a **non-quotation** lab scenario (specimen accession →
  chamber seasoning → conditioning cycle → approve conditioned batch) proving
  the design is not quotation-specific: its unknown domain concepts live only in
  the hidden scenario-local ``EvaluationSpec``.

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

The tests cover the DAG model and validation, Observation-as-evidence and
incremental node update, per-attribute confidence / provenance, necessity as a
node property, the Stakeholder filter, the evaluator metrics, EN/JA equivalence,
leakage, and the falsification suite A-AE.

## Design notes

- **Truth and Inferred DAGs are the same class.** No GT-only node/edge types.
- **Observations are authentic primary evidence.** They are captured only from
  actual stakeholder (user) messages via `observe_turn` — the agent cannot write
  arbitrary Observation text, source, or turn. Observations are immutable, and
  multiple observations attach to one node.
- **Conditional branches are edges with predicates** — no Branch class, no
  node-level condition.
- **Necessity is a node property** with per-property confidence / provenance;
  unknown must stay unknown (no fabrication).
- **Concepts are evaluator-only** — never stored on nodes, never agent-visible.
- **No compatibility shims.** The old Step / Transition / Branch / GTStep /
  GTTransition / GTBranch / StakeholderStepTruth / GTNecessity model is removed.
- **Generic tau2 core unchanged.**
