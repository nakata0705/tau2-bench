# business_interview (v13 — Truth reconstruction is the primary score)

The agent interviews a stakeholder to discover an unknown team's business
process. The **graph is the semantic model**:

- Truth = `BusinessProcessGraph` + `TruthConcept[]` (the ground-truth target);
- the stakeholder's world model = `StakeholderKnowledge` (a masked
  `StakeholderKnowledgeGraph` with three-valued property slots
  `ConceptRef | None | DONT_KNOW` + `StakeholderKnowledgeConcept[]`) — a
  **simulator constraint**, not the scored target;
- the agent builds an `AgentGraph` + `AgentConcept[]`.

**Scoring is Truth-reconstruction.** The final AgentConcepts / AgentGraph
are compared to the TruthConcepts / TruthGraph by **content** (deterministic
signatures over the glossary labels/terms and slot values). It does NOT
require conversational provenance: an Agent that infers a fact that was never
explicitly exposed — but happens to match the Truth — is counted correct.
Unsupported-but-correct reconstruction is not penalised. Fabricated / wrong
graph elements still are.

Private provenance (the sidecar annotations / dialogue events) is retained
as **diagnostic** metadata and for simulator-integrity checks (the
stakeholder must not reveal facts outside its StakeholderKnowledge), but it
is never a hard gate for Agent scoring.

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
  referring to the concept (diagnostic only).
- Graph/property/edge references may carry optional `EvidenceRef`s, but
  evidence is **diagnostic only**: a tool call never fails merely because a
  quoted span is missing, ambiguous, or does not resolve to a private
  stakeholder slot.

## No concept validation lifecycle

There is no hypothesized/grounded/confirmed/unknown/disputed lifecycle: the
obsolete `ground_concept` / `confirm_concept` / `mark_concept_unknown` /
`mark_concept_disputed` tools and the ``AgentConcept`` validation fields are
removed. Concept identity is judged by content against Truth, and nothing
gates the interview on a concept status. `record_terminology_agreement`
independently records an agreed term.`

## Evaluation

Primary target: **AgentConcepts / AgentGraph vs TruthConcepts / TruthGraph**.
`evaluate(db, knowledge, spec, *, truth=..., ...)`:

- concept identity is content-based (deterministic signatures over the agent
  glossary labels/descriptions vs the Truth concept canonical terms/
  descriptions, scoped per kind, followed by a content bijection);
  `concept_recall` / `concept_precision` / `concept_correctness`;
- node/edge correspondence follows content matching (nodes by their
  referenced-concept content signature; edges by endpoint pair on the Truth
  graph); `node_recall` / `node_precision` / `edge_recall` / `edge_precision`
  / fabricated counts; `start_correct` / `end_recall` / `end_precision`;
- property scoring per slot (epistemic, Truth-based): a Truth `ConceptRef`
  slot needs a matching agent ConceptRef; a Truth-absent (`None`) slot needs
  an explicit ABSENT marker (UNSET / DONT_KNOW / a concept are NOT correct —
  no answer is not a lucky guess); the same rule applies to edge conditions;
  `reads` / `writes` score recall x precision over the element set and
  require an explicit ABSENT when Truth has none;
- `knowledge_coverage` (Truth vs StakeholderKnowledge) is reported
  separately as informational and is never mixed into Agent performance.

`quality_pass` / `structural_pass` require full reconstruction correctness:
all structural/property/concept metrics == 1.0, valid endpoints, valid graph.
Provenance (hypothesis / evidence hygiene) is reported as diagnostic only and
never gates `quality_pass`.

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
| `evaluation.py` | content/Truth-reconstruction evaluator (AgentGraph + AgentConcepts vs TruthGraph + TruthConcepts); provenance reported as diagnostics |
| `tools.py` | glossary + graph tools (optional diagnostic evidence; ground/confirm/unknown/disputed/terminology as Agent belief records; `record_dont_know` / `record_edge_condition_dont_know` / `record_absent` / `record_edge_condition_absent` as belief markers); NO observation tools |
| `user_simulator.py` | semantic stakeholder: chooses the Semantic Response Plan, validates it, realizes it (graph-native sidecar) |
| `environment.py` | conversation ledger + private sidecar validation/binding + environment-owned Observation creation + `episode_complete` |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/` and
`tests/test_business_interview_roundtrips.py`.
