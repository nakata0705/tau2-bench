# business_interview (v12 — graph is the semantic source of truth)

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

Property slots are three-valued on the StakeholderKnowledge side:
`ConceptRef` (value known), `None` (value known absent), `DONT_KNOW`
(element known, value unknown).

Truth (`BusinessProcessGraph` with `TruthNode` / `TruthEdge`) is complete
canonical data with its OWN two-valued slots: `ConceptRef | None` (`None` =
canonical absence). Truth never uses the Agent's four-state markers.

The AgentGraph is **inference-in-progress** and uses FOUR explicit
epistemic states (never `None` for both UNSET and ABSENT):

    UNSET      = not yet investigated / no conclusion (the default)
    ConceptRef = known value
    ABSENT     = explicitly established absent (evidenced)
    DONT_KNOW  = explicitly established unknowable (evidenced)

`UnsetType` / `AbsentType(evidence)` / `DontKnowType(evidence)` are the
explicit marker types; new Agent properties default to UNSET and
reset/unset means UNSET, never ABSENT. ABSENT/DONT_KNOW are valid ONLY when
their evidence resolves to the EXACT mapped stakeholder slot
(`node:<mapped>:<prop>` / `edge:<mapped>:condition`) whose value is None /
DONT_KNOW — evidence about another node's or edge's slot never supports a
marker.

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
IDs and declare their semantic mode (what the message asserts about the
element, validated deterministically against the knowledge):

    {"semantic_id": "node:skn_002:reads:skc_013",
     "mode": "value",
     "quote": "customer information", "occurrence": 0}

Modes: `value` (the slot holds a known value), `absent` (the slot is known
absent), `dont_know` (the slot is DONT_KNOW), `exists` (the element
itself), `mention` (a knowledge concept). A mode that contradicts the
stakeholder's own world model (e.g. `dont_know` on a known-value slot) is
REJECTED at ingestion — the message cannot be accepted.

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
- property scoring per slot: stakeholder `ConceptRef` needs a matching
  evidenced `ConceptRef`; stakeholder `None` (known absent) needs an
  evidenced `ABSENT` marker resolving to the exact mapped slot (`UNSET` /
  `DONT_KNOW` / a concept are incorrect — "not asserted" never scores as
  known absence); stakeholder `DONT_KNOW` needs an evidenced `DONT_KNOW`
  marker resolving to the exact mapped slot (`UNSET` / `ABSENT` / a concept
  are incorrect; a hidden-Truth guess remains wrong);
- concept identity per kind (mentions + graph provenance incl. grounding
  evidence may participate; bijection over the knowledge concepts the graph
  references; conflicting grounding evidence leaves the concept unresolved);
  concept completeness is measured as `concept_recall` / `concept_precision`
  against the EXPECTED StakeholderKnowledgeConcept set — an empty AgentGraph
  gets recall 0 (no vacuous success), and `glossary_complete` separates
  reconstruction completeness from `glossary_pass` (validation correctness
  of the referenced concepts);
- glossary validation: grounded/confirmed/unknown/disputed/terminology backed
  by the appropriate private evidence;
- `knowledge_coverage` (Truth vs StakeholderKnowledge) is reported
  separately and never mixed into Agent performance: known values AND known
  absence count as known; DONT_KNOW slots and removed nodes/edges count as
  unknown; node existence is never confused with the activity slot.

`start_inference` resets the AgentGraph/glossary/completion state but
preserves Observations and the conversation ledger.

## Environment-owned Observations

Every ACCEPTED stakeholder utterance automatically becomes one immutable
Observation BEFORE the Agent sees it (``environment.py``): the environment
validates the private sidecar, creates the Observation (id ``obs_<turn>``,
raw public text, source ``stakeholder``), and delivers the Observation id
inline at the front of the Agent-visible message (``[Observation obs_N]
<text>``). There is NO observation-capture tool — ``observe_latest_stakeholder_message``
and ``observe_message`` are removed, so the Agent never creates or mutates
Observations and never makes an observation round trip. A rejected
(failed/retried) Stakeholder generation creates no Observation and consumes
no id. Guarantee: one accepted Stakeholder utterance <-> one Observation
<-> one Agent-visible Observation id.

With the Observation id delivered alongside the response, independent
evidence operations batch in one Agent turn (e.g. several ``create_concept``
calls carrying ``evidence=obs_N``). Genuine dependencies are preserved: every
tool in a batch must have all required arguments known before the batch
starts — a future Observation id/concept id that does not exist yet is
rejected, never guessed or fabricated.

## Semantic Response Plan (WHAT before HOW)

``user_simulator.py`` separates WHAT the stakeholder semantically answers
from HOW it is worded:

1. **Plan** — given the question + its own knowledge, the stakeholder builds a
   private Semantic Response Plan: intended semantic addresses + modes
   (``semantic_id = node:skn_002:rationale, mode = value``), based only on
   StakeholderKnowledge (never Truth-only information).
2. **Validate** — every planned item is checked through the canonical
   resolver (`ConceptRef -> value`, `None -> absent`, `DONT_KNOW ->
   dont_know`, node/edge -> exists, StakeholderKnowledgeConcept -> mention);
   a plan that contradicts the knowledge (known value planned as dont_know,
   DONT_KNOW planned as value, ...) is rejected before any text is produced.
3. **Realize** — the validated plan is expressed in natural language; the
   private sidecar must contain, for EVERY planned item, an exact
   public-text span anchored to the SAME semantic_id + mode, and nothing
   outside the plan. Unplanned assertions, missing assertions, contradictions
   and quotes not in the public text are rejected (bounded retry). Ordinary
   terminology references stay `mention`; terminology agreement is never
   inferred.

The Agent never sees the plan, the semantic ids, or the sidecar — only the
Observation id + public text.

## Files

| Module | Purpose |
| -------- | --------- |
| `graph.py` | shared primitives, semantic-ID scheme, TruthConcept, AgentConcept, BusinessProcessGraph (Truth), AgentGraph, Observation/InterviewDB |
| `knowledge.py` | StakeholderKnowledge / StakeholderKnowledgeGraph / StakeholderKnowledgeConcept + `project_knowledge` |
| `stakeholder.py` | StakeholderFilter (element/property/concept knowledge knobs) |
| `facts.py` | SemanticAnnotation + PlanResponseItem + private dialogue events + SemanticLedger + catalog (annotation/plan validation) |
| `grounding.py` | shared global-span provenance (evidence refs -> semantic ids) |
| `scenario.py` | Truth graphs + filters + knowledge (quotation / lab / JA) |
| `evaluation.py` | provenance-only evaluator (AgentGraph vs StakeholderKnowledgeGraph) |
| `tools.py` | glossary + graph tools (per-property evidence, binding-aware ground/confirm/unknown/disputed/terminology, `record_dont_know` / `record_edge_condition_dont_know` / `record_absent` / `record_edge_condition_absent`); NO observation tools |
| `user_simulator.py` | semantic stakeholder: chooses the Semantic Response Plan, validates it, realizes it (graph-native sidecar) |
| `environment.py` | conversation ledger + private sidecar validation/binding + environment-owned Observation creation + `episode_complete` |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/` and
`tests/test_business_interview_roundtrips.py`.
