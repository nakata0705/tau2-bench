# business_interview domain (v3 — evidence-backed DAG inference)

A benchmark for agents that must **infer a business DAG from observations** during
an interview with a stakeholder. This is a breaking redesign from the old
"Step / Transition / Branch workflow restoration" model.

## Core idea

The real business process is a **Truth DAG**: 1 start node, 1+ end nodes,
acyclic, every node reachable from the start, and end nodes with out-degree 0.
The stakeholder does not know the whole DAG — they are a **filter** over it,
observing part of it in natural language. Each thing they say is an
**Observation** (independent, immutable evidence). The agent incrementally
integrates observations into an **Inferred DAG**, attaching multiple observations
to a node and updating its attributes / confidence, creating a node only when no
existing node corresponds.

**The Truth DAG and the agent's Inferred DAG use the same `BusinessDAG` class.**
There are no Truth-only node/edge classes.

## Domain model (`dag.py`)

- `BusinessDAG` — `nodes` (by id), `edges` (by id), `start_node_id`,
  `end_node_ids`, plus structural validation (acyclic, reachable, no dangling
  edges, ends have out-degree 0).
- `Node` — `id`, `action`, `actor`, `system`, `reads`, `writes`, `necessity`,
  `observation_ids`.
- `Edge` — `id`, `from_node`, `to_node`, optional `predicate` (control-flow
  condition), `observation_ids`. Unconditional flow is `predicate=None`;
  conditional branches are **multiple outgoing edges** with different predicates.
- `InferredValue` — `value`, `confidence` (0..1), `observation_ids` (provenance).
  Every inferred attribute carries its own confidence / provenance; confidence is
  not a single number per node.
- `Necessity` — a node property (0 or 1 per node) with `rationale`, `owner`,
  `evidence`, `removal_impact`, each an `InferredValue`.
- `Observation` — `id`, `source_id`, `text`, `order`, `locale`.
- `InterviewResult` — `dag` + `observations`.

## Agent tools

| Tool | Purpose |
|------|---------|
| `start_inference(name?)` | Begin an inferred DAG |
| `record_observation(text, source_id?, locale?)` | Record an observation (immutable evidence) |
| `add_node(node_id, action, ...)` | Add a node (only when no existing node corresponds) |
| `update_node(node_id, ...)` | Update an existing node's attributes / evidence |
| `attach_observation(node_id, observation_id)` | Attach an observation to a node (multiple per node) |
| `add_edge(edge_id, from_node, to_node, predicate?)` | Add a directed edge (predicate = condition) |
| `update_edge(edge_id, ...)` | Update an edge |
| `set_node_necessity(node_id, rationale?, owner?, ..., *_confidence?, observation_id?)` | Record why a node is needed (node property, per-property confidence) |
| `set_dag_endpoints(start_node_id?, end_node_ids?)` | Set the start and end nodes |
| `finish_interview(summary?)` | Close the interview |

## Evaluation (`evaluation.py`)

The evaluator maps agent nodes to Truth nodes by **action concept**
(`EvaluationSpec.truth_node_concepts`), which are evaluator-only annotations that
make matching language-independent and hide agent node ids. Metrics:

- `node_recall` / `node_precision`, `fabricated_node_count`
- `edge_recall` / `edge_precision`, `fabricated_edge_count`
- `start_correct`, `end_recall` / `end_precision`
- `predicate_correctness`, `actor_correctness`, `system_correctness`,
  `read_correctness`, `write_correctness`
- `necessity_correctness`, `fabricated_necessity`

There is no Step / Transition / Branch-specific evaluator. Confidence /
provenance are stored, validated, and surfaced in diagnostics (no calibration
scoring).

## Stakeholder filter (`stakeholder.py`)

A stakeholder is a `StakeholderFilter` over the single Truth DAG: which nodes /
edges / attributes / necessity properties it can observe. Multiple filters can be
applied to the same Truth DAG; `apply` returns a filtered DAG so hidden
information never leaks to the simulator. The full Truth DAG is never serialized
into a simulator prompt.

## Scenario

One scenario is bundled: `quotation_workflow_1` (EN) and its Japanese variant
`quotation_workflow_1_ja`. Truth DAG: receive request → check customer → create
quotation → (approve high-value OR send) → send, plus a conditional month-end
accounting-summary tail. 1 start (receive), 2 ends (send, month-end). Edge
predicates: `amount over 1,000,000` (approve), `amount at or below 1,000,000`
(send), `month-end` (summary). Necessity: approval rationale is credit-risk
(confirmed); the month-end step's rationale / owner / evidence / removal are
unknown (must not be fabricated).

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
- **Observations are independent evidence**, not nodes; multiple observations
  attach to one node.
- **Conditional branches are edges with predicates** — no Branch class, no
  node-level condition.
- **Necessity is a node property** with per-property confidence / provenance;
  unknown must stay unknown (no fabrication).
- **Concepts are evaluator-only** — never stored on nodes, never agent-visible.
- **No compatibility shims.** The old Step / Transition / Branch / GTStep /
  GTTransition / GTBranch / StakeholderStepTruth / GTNecessity model is removed.
- **Generic tau2 core unchanged.**
