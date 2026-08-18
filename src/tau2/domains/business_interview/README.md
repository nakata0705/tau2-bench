# business_interview domain (v6 — agent-local data concepts + hidden provenance)

A benchmark for agents that must **reconstruct an unknown business DAG** from
authentic stakeholder Observations, using **free-text actions**, optional
**generic operation primitives** (with an explicit `unclassified` sentinel for
unknown operations), **agent-local data concepts** for reads/writes, and
**evidence-backed provenance**. No fixed scenario ontology is required of the
agent.

## Core idea

The real business process is a **Truth DAG** (1 start, 1+ ends, acyclic, every
node reachable, ends out-degree 0). In the benchmark a hidden ground truth exists
and only the evaluator uses it (deterministically). The agent holds its
understanding open-world: free-text actions, optional generic primitives,
actor / system / reads / writes / necessity, each traceable to authentic
Observations. **Reads/writes are agent-local data concepts** — the Agent LLM,
not the evaluator, interprets stakeholder wording variation.

**Benchmark vs production.** The concept resolver / hidden `EvaluationSpec` /
claim catalog are only *scoring aids*; they are not the agent's
business-understanding mechanism.

## The pipeline

```text
stakeholder natural language
  -> Agent LLM local concept identity
  -> consistent ConceptRefs
  -> Observation ids
  -> hidden Truth-claim provenance
  -> deterministic evaluator binding
```

## Domain model (`dag.py`)

- `BusinessDAG` — `nodes`, `edges`, `data_concepts`, `start_node_id`,
  `end_node_ids`, plus structural validation (including: every `ConceptRef`
  must reference an existing concept).
- `Node` — `id`, `action` (open-world free text), optional `primitive` (generic
  operation or `unclassified`), `actor`, `system`, `reads` / `writes`
  (`list[ConceptRef]`), `necessity`, `observation_ids`.
- `DataConcept` — `id` (agent-local, arbitrary), `preferred_label`, `terms`
  (`list[ConceptTerm]`).
- `ConceptTerm` — `text` (an observed stakeholder wording), `observation_ids`.
- `ConceptRef` — `concept_id`, `confidence`, `observation_ids`.
- `Edge` — `id`, `from_node`, `to_node`, optional `predicate`, `observation_ids`.
- `InferredValue` — `value`, `confidence` (0..1), `observation_ids` (used for
  action/actor/system/predicate/necessity).
- `Necessity` — a node property with `rationale`, `owner`, `evidence`, `removal_impact`.
- `Observation` — `id`, `source_id`, `text`, `order`, `turn` (authentic).
- `InterviewResult` — `dag` + `observations`.

There is **no** DiscoveredConcept model: the benchmark evaluates the ability to
DAG-ify an unknown business as free actions + local concepts, not to manage
discovered concepts as separate objects.

## Agent-local data concepts (reads/writes)

The Agent LLM decides whether different stakeholder expressions refer to one
business object:

- `create_concept(concept_id, label, observation_id?)` — create a local concept
  when an object is first discovered; prefer the stakeholder's wording as the
  label.
- `add_concept_term(concept_id, term, observation_id?)` — record a later,
  different wording the Agent judges to be the same object.
- `merge_concepts(target, sources)` — repair an initial mistaken split: every
  node reference is re-pointed to the target and terms are folded in.
- `list_concepts()` — see the agent's own working vocabulary.

`add_node` / `update_node` take `reads` / `writes` as **concept ids** (validated
against the DAG's concepts). Reuse one concept id consistently across every
node+axis where the same object flows; the evaluator requires this (see
“Evaluator binding”).

## Generic primitives (`concepts.py`)

The global resolver holds only **scenario-independent generic operation
primitives** (create / check / approve / reject / send / receive / record /
update / transform / reconcile / notify / move / review). `resolve_primitive`
returns the best primitive, or **`"unclassified"`** when no known primitive can
be safely determined. `unclassified` is a normal open-world state, **not** a
failure. There are NO read/write aliases anywhere in the global resolver.

## Agent tools

| Tool | Purpose |
| ------ | --------- |
| `start_inference(name?)` | Begin an inferred DAG (destructive to prior DAG + captured Observations) |
| `list_stakeholder_messages()` | See the stakeholder statements by stable id (`sm_1`, ...) |
| `observe_latest_stakeholder_message()` | Return the id of the most recent stakeholder message (`sm_N`) |
| `observe_message(message_id)` | **Capture** a stakeholder message by id (e.g. `sm_3`) as an authentic Observation (idempotent) |
| `create_concept(concept_id, label, observation_id?)` | Create an agent-local data concept (business object) |
| `add_concept_term(concept_id, term, observation_id?)` | Add an observed wording to a concept |
| `merge_concepts(target, sources)` | Merge mistakenly split concepts (re-points all refs) |
| `list_concepts()` | List the agent's own concepts / terms |
| `add_node(node_id, action, primitive?, ...)` | Add a node (action is open-world free text; reads/writes are concept ids) |
| `update_node(node_id, ...)` | Update an existing node's attributes / evidence |
| `remove_node(node_id)` | Remove a node and its incident edges (drop obsolete / superseded coarse nodes) |
| `attach_observation(node_id, observation_id)` | Attach an observation to a node (multiple per node) |
| `add_edge(edge_id, from_node, to_node, predicate?)` | Add a directed edge (predicate = condition) |
| `update_edge(edge_id, ...)` | Update an edge |
| `set_node_necessity(node_id, rationale?, ..., observation_id?)` | Record why a node is needed (per-property confidence) |
| `set_dag_endpoints(start_node_id?, end_node_ids?)` | Set the start and end nodes |
| `validate_dag()` | Review the DAG's internal structural consistency (no GT reference) |
| `finish_interview(summary?)` | Close the interview (rejects a structurally invalid DAG) |

## Hidden Truth claims + private StakeholderFacts (`claims.py`, `facts.py`)

The benchmark's private business knowledge is structured, not textual:

    StakeholderFact
       ├─ public natural language ─▶ Agent ─▶ Agent-local DataConcept
       └─ private used_fact_ids ─▶ TruthClaims
                                      ▲
    ConceptRef ─▶ Observation ─▶ turn ┘

- **TruthClaim** — one evaluator-only Truth claim: a stakeholder-visible
  read/write fact (Truth node + axis + Truth data concept), e.g.
  `cq.writes.tc_quote`, `cc.reads.tc_customer`, `cq.reads.tc_pricing`. Only
  claims allowed by the scenario's `StakeholderFilter` enter the private
  catalog — hidden axes (`sq.reads`, `sq.writes`, `me.reads`, ...) produce
  **no** claims and can never be referenced.
- **StakeholderFact** — a hidden structured business fact of the stakeholder
  (``id``, natural ``text``, ``supported_claim_ids``). The stakeholder
  simulator answers **only from these facts** and returns a **private sidecar**
  (``used_fact_ids``) alongside its natural-language response; only the message
  enters the conversation. Fact/claim ids are evaluator/simulator-private.
- **Sidecar ledger** (`StakeholderFactLedger`) — stores ``used_fact_ids``
  against that exact stakeholder message's turn. Provenance is **never
  derived or reconstructed from message text**.
- **Validation** — at response ingestion (and again in the evaluator),
  ``used_fact_ids`` are validated deterministically: every id exists, belongs
  to this stakeholder, every supported TruthClaim exists, and every claim is
  stakeholder-visible. Invalid metadata is rejected.

The Agent never sees fact ids, claim ids, the catalogs, or the ledger: they
live only in `claims.py` / `facts.py` and the evaluator's inputs; the ledger is
not part of `InterviewDB`, tool outputs, or serialized state. The evaluator
**never infers semantic support by reading Observation text** — it consumes
only the private provenance.

## Stakeholder simulator (`user_simulator.py`)

The ``business_interview_user`` user implementation is a small adapter over
tau2's ``UserSimulator`` that preserves the public-message/private-sidecar
separation:

- the stakeholder LLM's system prompt carries the hidden StakeholderFacts
  (id + text) and the JSON sidecar contract
  ``{"message": ..., "used_fact_ids": [...]}``;
- only ``message`` becomes the ``UserMessage``; ``used_fact_ids`` travel on a
  private, never-serialized field and the environment stores them in the
  private ledger against the exact turn;
- the stakeholder speaks naturally (it need not copy fact text), never exposes
  fact ids in its natural-language output, and keeps the persona/conversation
  instructions (disclosure, unknown behavior) separate from the facts.

The stakeholder simulator naturalizes facts. The Agent LLM interprets
language. Private instrumentation records Truth provenance. The evaluator
binds provenance; it does no NLP semantic matching.

## Evaluation (`evaluation.py`)

The evaluator maps agent nodes to Truth nodes using a **hidden scenario-local
`EvaluationSpec`** (evaluator-only action expressions + expected primitives;
**no read/write equivalence**). Metrics:

- `node_recall` / `node_precision` / `domain_concept_correctness`, `fabricated_node_count`
- `edge_recall` / `edge_precision`, `fabricated_edge_count`
- `start_correct`, `end_recall` / `end_precision`
- `predicate_correctness`, `actor_correctness`, `system_correctness`,
  `read_correctness`, `write_correctness`
- `concept_correctness` — agent-local concept binding integrity
- `necessity_correctness`, `fabricated_necessity`
- `primitive_correctness` (diagnostic; `unclassified` is not penalized)

### Result correctness vs evidence hygiene

**Result correctness** compares the inferred DAG against the hidden Ground
Truth: node/edge recall + precision, start/end, predicate/actor/system/read/
write correctness, concept correctness, necessity correctness, and the
diagnostic primitive correctness. **Evidence hygiene** (`evidence_pass` /
`provenance_authenticity_pass`) deterministically guarantees only that every
asserted claim references a **real, authentic stakeholder Observation** —
captured from an actual user message via `observe_message`, not fabricated. It
does **not** re-interpret Observation *text* to decide whether it semantically
supports a claim.

### Evaluator binding (reads/writes)

For each matched node+axis, the evaluator grounds each visible `ConceptRef`
through private provenance only:

    ConceptRef ─▶ cited Observation ids ─▶ Observation.turn
        ─▶ private used_fact_ids ─▶ supported TruthClaims ─▶ Truth data concept

1. maps the Agent node to its Truth node with the existing node matching;
2. determines the expected hidden Truth claims for that node+axis;
3. follows the ref/concept's cited Observation ids into the private
   used-fact ledger (never reading Observation text);
4. keeps the TruthClaims at this node+axis that the cited facts support;
5. binds the agent-local concept id to the Truth data concept(s) its
   citations support, requiring a **unique perfect matching** between used
   Agent concepts and visible Truth data concepts.

This deterministically enforces the concept invariants:

- one Agent concept may bind to only **one** Truth data concept (a concept
  whose citations pin it to several Truth concepts is unassignable);
- one visible Truth data concept must use **one** Agent concept identity
  (duplicate/split concepts for one Truth concept fail until merged;
  merging repairs them);
- merging distinct Truth concepts into one Agent concept fails.

Thus one `customer_information` concept may be reused across `cc.reads` and
`cq.reads` regardless of its label. Failures:

- no cited Observation supports an expected claim → unsupported ref;
- no unique perfect concept binding exists → `concept_correctness` 0;
- a hidden read/write axis is asserted → epistemic failure (private
  provenance never rescues a hidden assertion).

Labels are never compared: the evaluator does not know or care whether the
agent wrote “quote”, “quotation”, “the price doc”, or anything else. Wording
alone cannot create a match; an authentic-but-unrelated Observation cannot
support a claim; identical text with different private provenance grounds
differently.

### Stakeholder-visibility scoring

The full Truth DAG is the benchmark author's process model and may contain
information **not available to the interviewed stakeholder**. Which actor /
system / reads / writes attributes a stakeholder can actually assert is defined
by the scenario's `StakeholderFilter` (`scenario.stakeholder`), per node.

For each matched node and each of actor/system/reads/writes:

- **visible** attribute (the stakeholder can know it): compared against Truth —
  actor/system via role normalization, reads/writes via hidden-claim binding;
- **hidden** attribute (the stakeholder cannot know it): the **correct** agent
  behavior is to leave it **unset / empty**. An asserted value is **incorrect**
  even if it happens to equal the full hidden Truth — the benchmark rewards
  epistemic restraint, not fabrication. Hidden axes are a **prior gate**: no
  provenance or terminology mechanism can rescue a hidden assertion.

Node actions, topology, predicates, necessity, and other Truth structure are
not affected by visibility. Necessity keeps its own known-unknown handling.

`evaluate(db, truth, spec, stakeholder, claims=..., provenance=...)` receives
the stakeholder visibility plus the hidden claim catalog and provenance ledger
(the `InterviewTools` assertion hooks pass `scenario.claims` and the derived
ledger automatically).

## Terminology alignment (interview behavior)

Terminology is a **conversation-level** concern. The intended separation:

1. **Ground Truth** defines the benchmark's concepts/facts (evaluator-only).
2. **Stakeholder** speaks natural domain language ("I create the quotation").
3. **Interviewer (Agent)** establishes a **shared working vocabulary** with the
   stakeholder and records it as **agent-local data concepts**: identify
   important roles/systems/objects, use the stakeholder's own terms by default,
   and briefly confirm stable labels for recurring concepts.
4. **The Agent LLM decides concept identity.** When later wording may mean the
   same object, the Agent decides whether to reuse the concept; if materially
   ambiguous, it asks the stakeholder; observed/confirmed terms are added to
   the same local concept; one concept id is reused consistently throughout
   reads/writes.
5. **The evaluator never decides synonymy.** There is no synonym table, no
   label comparison — equivalence comes only from hidden claim provenance.

Implementation is prompt/policy behavior only (`policy.md` + task
`task_instructions`). There is **no** glossary model, ontology, or terminology
API. A design note for an optional future interview-confirmed terminology
record lives in `doc/business-interview-terminology-design.md`.

`quality_pass` requires `structural_pass AND necessity_pass AND evidence_pass AND
provenance_authenticity_pass`. Unknown (unset) values need no provenance; a value
with confidence 0 is treated as unasserted.

## Stakeholder filter (`stakeholder.py`)

A stakeholder is a `StakeholderFilter` over the single Truth DAG: which nodes /
edges / attributes / necessity properties it can observe. Multiple filters can be
applied to the same Truth DAG; `apply` returns a filtered DAG so hidden
information never leaks to the simulator.

`StakeholderFilter` supports **per-node** visibility of actor / system /
reads / writes via `visible_node_attributes` (a node listed there is limited to
exactly those axes; a node not listed uses the global `visible_attributes`). `evaluate()`
uses this to score a hidden attribute as correct only when the agent leaves it
unset (see “Stakeholder-visibility scoring”). The claim catalog is derived from
the same visibility (`build_claims`), so hidden facts never enter the private
catalog.

## Scenario

Scenarios bundle a Truth DAG (with evaluator-only Truth data concepts) + a
hidden scenario-local `EvaluationSpec` + stakeholder filter(s) + the hidden
claim catalog. Bundled:

- `quotation_workflow_1` (EN) / `quotation_workflow_1_ja` — the quotation
  workflow (receive → check → create → (approve OR send) → send + conditional
  month-end summary; 1 start / 2 ends; approval rationale is credit-risk,
  month-end necessity unknown). Truth concepts: `tc_request`, `tc_customer`,
  `tc_pricing`, `tc_quote`, `tc_approval`, `tc_sent_quote`,
  `tc_excel_summary`. Visible claims: `r.writes.tc_request`,
  `cc.reads.tc_customer`, `cq.reads.tc_customer`, `cq.reads.tc_pricing`,
  `cq.writes.tc_quote`, `me.writes.tc_excel_summary`. EN and JA use
  locale-appropriate surface terms for the derivation.
- `lab_sample_flow` — a **non-quotation** lab scenario (specimen accession →
  chamber seasoning → conditioning cycle → approve conditioned batch) proving
  the design is not quotation-specific. Its derived read/write artifacts are
  hidden; the only visible data claim is `n1.reads.tc_sample`.

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
- **Reads/writes are agent-local concepts** with hidden-claim binding; the
  evaluator contains no synonym inference and never reads Observation text.
- **Observations are authentic primary evidence** (via `observe_message` /
  `observe_latest_stakeholder_message`); immutable;
  multiple observations attach to one node.
- **Actions are open-world free text**; primitive is an optional generic
  operation, with `unclassified` for unknown operations.
- **Conditional branches are edges with predicates** — no Branch class.
- **Necessity is a node property**; unknown must stay unknown (no fabrication).
- **Hidden scenario EvaluationSpec + claim catalog are evaluator-only**; the
  global resolver holds only generic primitives.
- **Result correctness is Ground Truth comparison; evidence hygiene only checks
  Observation references are real & authentic** — the Observation body is not
  semantically re-interpreted.
- **No compatibility shims.** DiscoveredConcept / concept discovery /
  claim-level semantic relevance (`support`, SUPPORTED / CONTRADICTED /
  UNKNOWN) / string read-write matching / `data_expressions` are removed.
- **Generic tau2 core unchanged.**
