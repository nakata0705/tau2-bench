# business_interview (v11 — graph is the semantic source of truth)

The agent interviews a stakeholder to discover an unknown team's business
process. The **graph is the semantic model**:

- Truth = `BusinessProcessGraph` + `TruthConcept[]` (no generated claims);
- the stakeholder's world model = `StakeholderKnowledge` (a masked
  `StakeholderKnowledgeGraph` with three-valued property slots
  `ConceptRef | None | DONT_KNOW` + `StakeholderKnowledgeConcept[]`);
- the agent builds an `AgentGraph` + `AgentConcept[]`.

Correctness is grounded **only** through private provenance — no evaluator
semantic NLP, no aliases, no embeddings, no label matching:

    Agent EvidenceRef -> Observation span -> private annotation
        (stakeholder semantic ID) -> StakeholderKnowledgeGraph element /
        StakeholderKnowledgeConcept

## Semantic IDs

Every addressable graph element has a **stable semantic ID** (never a list
index; IDs survive reordering):

    node:<node_id>                     the node itself (existence, NOT activity)
    node:<node_id>:activity            scalar property slot
    node:<node_id>:actor | :system | :rationale
    node:<node_id>:reads               whole-property slot
    node:<node_id>:reads:<concept_id>  one reads element
    node:<node_id>:writes:<concept_id> one writes element
    edge:<edge_id>                     the edge itself
    edge:<edge_id>:condition           the edge-condition slot
    <concept_id>                       the knowledge concept itself

**Stakeholder semantic IDs are opaque and stakeholder-local** (`skn_001` /
`ske_001` / `skc_001` style): assigned deterministically from sorted Truth
ids and invariant to collection reordering; never derived from Truth ids,
labels, terms, node ids or edge ids. The private Truth mappings
(`node_truth_ids` / `edge_truth_ids` / per-concept `truth_concept_id`) live
only in the evaluator-side `StakeholderKnowledge`. **One canonical
resolver** — `StakeholderKnowledgeGraph.resolve` — interprets every semantic
ID (node existence, exact property slot, exact reads/writes element, edge
existence, exact condition slot, knowledge concept); annotation validation,
provenance, concept binding, DONT_KNOW handling, knowledge coverage and
diagnostics all go through it.

Property slots are three-valued: `ConceptRef` (value known), `None` (value
known absent), `DONT_KNOW` (element known, value unknown). The AgentGraph
uses the same three-valued slots; `DONT_KNOW` markers carry the Observation
evidence that the stakeholder really said "I don't know" for that exact
slot.

## StakeholderKnowledge (the stakeholder's world model)

`project_knowledge(truth, stakeholder_filter)` builds the world model:

- nodes/edges the stakeholder does not know exist are **removed** — never
  shortcut edges (A->B->C with unknown B yields no A->C);
- known properties keep their values (as `ConceptRef`s into the
  stakeholder's own concepts);
- known-absent properties are `None`;
- unknown properties of known elements are `DONT_KNOW`.

`StakeholderKnowledgeConcept` is the stakeholder's local understanding of one
Truth concept: `{id, truth_concept_id (private), kind, description, terms}`
where description/terms are each `str | list[str] | DONT_KNOW` and vary
independently (term known/details unknown; details known/local wrong term;
details known/term unknown; both known). Hidden Truth ids/canonical terms
never enter the knowledge or the stakeholder prompt.

## Provenance (graph-native)

Private Observation annotations point **directly** at stakeholder semantic
IDs:

    {"semantic_id": "node:skn_002:reads:skc_013",
     "quote": "customer information", "occurrence": 0}

There is no `StakeholderSemanticAssertion`: no subject/property/value
duplication — the semantic meaning is resolved from the
StakeholderKnowledgeGraph. DONT_KNOW speech anchors to the relevant property
slot's semantic id. A span covering several distinct semantic IDs is
**globally ambiguous and grounds nothing** (no cross-credit); the
single-assertion exception applies when the span is one annotation.

## Dialogue events

Concept identity / terminology keep minimal private dialogue events (a bare
"Yes." does not encode the act), addressing `StakeholderKnowledgeConcept`
ids:

    ConceptAlignmentAssertion {semantic_id, quote, occurrence, act}
                              act: confirm | partial | unknown | dispute
    TerminologyConfirmation   {semantic_id, proposed_term, quote, occurrence}

Ordinary workflow mentions never create these events.

## Agent-side redesign

- `AgentConcept.mentions` = Observation spans the Agent interprets as
  referring to the concept (identity may use them; property scoring never
  does).
- **Every AgentGraph property reference carries its own EvidenceRef**
  (activity/actor/system/reads/writes/rationale/condition) — pass a property
  as `{"concept_id": ..., "evidence": [...]}` in `add_node`/`update_node`/
  `add_edge`.
- Property scoring uses **property evidence ONLY**; concept identity may use
  mentions; validation uses explicit validation/dialogue evidence ONLY.
  There is no evaluator helper that mixes ref evidence + mentions +
  validation evidence.

## Concept status

    hypothesized  ->  grounded  ->  confirmed

`grounded` = authentic provenance binds the Agent concept to the stakeholder
knowledge (`ground_concept` is BINDING-AWARE: its evidence must resolve —
global span rule + the canonical resolver — to exactly one kind-compatible
knowledge concept; ambiguous, unrelated or kind-incompatible evidence is
rejected, private stakeholder ids never appear in tool output, and the
Agent-visible `grounded` status always agrees with the evaluator binding);
no confirmation dialogue needed;
`confirmed` = explicit identity confirmation (`confirm_concept` with a
private alignment event). `finish_interview` normally requires referenced
concepts >= grounded. `unknown`/`disputed`/`partially_confirmed` are backed
by the corresponding private events.

## Evaluation

Primary achievable target: **AgentGraph vs StakeholderKnowledgeGraph**
(Truth mapping stays private). `evaluate(db, knowledge, spec, ...)`:

- node/edge correspondence falls out of the semantic IDs (deterministic
  assignment maximizing property-level matches);
- property scoring per slot is ASYMMETRIC: stakeholder `ConceptRef` needs a
  correct grounded ref; stakeholder `None` (known absent) is satisfied by an
  unasserted slot (`DONT_KNOW` is NOT equivalent); stakeholder `DONT_KNOW`
  is satisfied ONLY by an explicit evidenced `DONT_KNOW` marker (unasserted
  is NOT `DONT_KNOW`, and a hidden-Truth guess is wrong);
- concept identity per kind (mentions + graph provenance incl. grounding
  evidence may participate; bijection over the knowledge concepts the graph
  references; conflicting grounding evidence leaves the concept unresolved);
- glossary validation: grounded/confirmed/unknown/disputed/terminology backed
  by the appropriate private evidence;
- `knowledge_coverage` (Truth vs StakeholderKnowledge) is reported
  separately and never mixed into Agent performance: known values AND known
  absence count as known; DONT_KNOW slots and removed nodes/edges count as
  unknown; node existence is never confused with the activity slot.

`start_inference` resets the AgentGraph/glossary/completion state but
preserves Observations and the conversation ledger.

## Files

| Module | Purpose |
| -------- | --------- |
| `graph.py` | shared primitives, semantic-ID scheme, TruthConcept, AgentConcept, BusinessProcessGraph (Truth), AgentGraph, Observation/InterviewDB |
| `knowledge.py` | StakeholderKnowledge / StakeholderKnowledgeGraph / StakeholderKnowledgeConcept + `project_knowledge` |
| `stakeholder.py` | StakeholderFilter (element/property/concept knowledge knobs) |
| `facts.py` | SemanticAnnotation + private dialogue events + SemanticLedger + catalog |
| `grounding.py` | shared global-span provenance (evidence refs -> semantic ids) |
| `scenario.py` | Truth graphs + filters + knowledge (quotation / lab / JA) |
| `evaluation.py` | provenance-only evaluator (AgentGraph vs StakeholderKnowledgeGraph) |
| `tools.py` | glossary + graph tools (per-property evidence, binding-aware ground/confirm/unknown/disputed/terminology, `record_dont_know` / `record_edge_condition_dont_know`) |
| `user_simulator.py` | semantic stakeholder realization (graph-native sidecar) |
| `environment.py` | conversation ledger + private sidecar binding + `episode_complete` |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/`.
