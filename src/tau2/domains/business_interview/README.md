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

## Canonical TruthGraph contract

`BusinessProcessGraph` (also exported as `TruthGraph`) always contains the
explicit nodes `STRUCTURAL_SOURCE_ID` and `STRUCTURAL_SINK_ID`. Their
`structural_role` is respectively `source` and `sink`, and both are
`protected`. Every boundary edge is typed `edge_kind="structural_boundary"`,
`structural_only=True`, and `protected=True`.

The canonical invariant requires exactly one topology source (SOURCE) and one
topology sink (SINK), SOURCE indegree zero, SINK outdegree zero, no dangling
or isolated node, and every business node on a SOURCE-to-SINK path. Multiple
business entries fan out from SOURCE and multiple business exits fan in to
SINK. A business exit is therefore never itself a topology sink.

Use `canonical_structure_errors()` for a non-throwing audit and
`validate_canonical_graph()` for fail-fast validation. Boundary nodes/edges
are structural metadata, not business concepts or business relations.

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
reset/unset means UNSET, never ABSENT. ABSENT/DONT_KNOW are Agent beliefs,
not provenance-gated claims. Their EvidenceRef lists are optional diagnostic
metadata; when supplied, an Observation id must exist, but quote spans and
exact stakeholder-slot binding never determine whether a marker is recorded
or whether reconstruction is correct.

Policy reminder: UNSET means no conclusion yet, while DONT_KNOW means the
Stakeholder cannot provide the value. Do not oscillate between them on one
property; ask the Stakeholder for more information, then record the resulting
state and continue. The runtime tool-operation guard is a safeguard, not a
replacement for this reasoning.

## StakeholderKnowledge (the stakeholder's world model)

`project_knowledge(truth, stakeholder_filter)` accepts only a canonical Truth
(graph construction uses `canonicalize_truth_graph`) and always validates the
resulting canonical graph. It never repairs a non-canonical Truth or an invalid
forgetting sample:

- structural SOURCE/SINK and boundary edges are always retained and protected;
- semantic forgetting keeps a node/edge in the topology while replacing
  semantic slots with `DONT_KNOW`;
- a forgotten business node may be removed only by safe serial-path
  contraction (`indegree == outdegree == 1`), with no self-loop, no ambiguous
  parallel edge, and condition-free Truth incident edges. If the stakeholder
  does not know the resulting shortcut's condition slot, it remains
  `DONT_KNOW`; no condition is composed heuristically;
- conditioned paths and branch/merge nodes are rejected, never repaired or
  composed heuristically;
- invalid samples are discarded and forgetting is re-drawn up to
  `StakeholderForgettingConfig.max_retries`. Exhaustion raises
  `KnowledgeProjectionError` with the configuration and validation reasons.

A derived edge has `is_shortcut=True`, `contracted_nodes`,
`derived_from_edges`, and evaluator-private `shortcut_provenance`. It is a
path contraction, not a newly asserted business fact.

Known properties keep their values (as `ConceptRef`s into the stakeholder's
own concepts); known-absent properties are `None`; unknown properties of
known business elements are `DONT_KNOW`. The semantic forgetting probability
applies to node slots and edge-condition slots; it never removes topology.

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

## Graph hypothesis editing

The AgentGraph is a revisable working hypothesis. `add_edge` and `update_edge`
record or refine relations; `remove_edge(edge_id)` deletes exactly one
existing edge while preserving both endpoint nodes and every unrelated edge.
Use it when a speculative shortcut becomes obsolete after discovering
intermediate steps. Use `remove_node` only when the node itself is obsolete;
it also removes that node's incident edges. The runtime never deletes edges
using hidden Truth.

## Model refusal diagnostics

`generate()` instrumentation records every Agent/Stakeholder generation
attempt, including private Stakeholder plan/realization calls and retries.
`model_refusal_count` / `model_refusals` in real-run artifacts use only these
call-level records. Explicit refusal text or a provider `message.refusal` field
counts; `I don't know`, generic uncertainty/apologies, malformed JSON,
sidecar/tool validation errors, and provider exceptions do not. The accepted
public trajectory count is retained only as a separate compatibility diagnostic
and is never added to the call-level count.

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
- node/edge correspondence follows content matching on the **business
  projection** (structural SOURCE/SINK and boundary edges are excluded from
  all ordinary denominators); `node_recall` / `node_precision` /
  `edge_recall` / `edge_precision` / fabricated counts; `start_correct` /
  `end_recall` / `end_precision`;
- property scoring per slot (epistemic, Truth-based): a Truth `ConceptRef`
  slot needs a matching agent ConceptRef; a Truth-absent (`None`) slot needs
  an explicit ABSENT marker (UNSET / DONT_KNOW / a concept are NOT correct —
  no answer is not a lucky guess); the same rule applies to edge conditions;
  `reads` / `writes` score recall x precision over the element set and
  require an explicit ABSENT when Truth has none;
- `knowledge_coverage` (business Truth vs StakeholderKnowledge) is reported
  separately as informational and is never mixed into Agent performance.
  `EvaluationDiagnostics.canonical_contract` reports source/sink validity,
  structural/business counts, shortcut provenance, and explicitly records that
  topology-derived END inference was not used.

`quality_pass` / `structural_pass` require full reconstruction correctness:
all structural/property/concept metrics == 1.0, valid endpoints, valid graph.
Provenance (evidence hygiene, sidecar annotations and dialogue events) is
reported as diagnostic only and never gates `quality_pass`.

### Stakeholder Truth reference scores (reference only)

The benchmark's primary score is always **AgentGraph ↔ TruthGraph** Truth
reconstruction. `StakeholderKnowledge` is not evaluator ground truth and never
replaces Truth, relaxes the primary denominator, excuses Agent errors, changes
`quality_pass`, or changes leaderboard ranking. A stakeholder may forget part
of the process; an Agent that copies that incomplete view is still incomplete
against Truth.

`EvaluationResult.stakeholder_truth_reference` stores one named
`StakeholderKnowledge ↔ TruthGraph` evaluation per configured stakeholder. Each
entry includes the same Truth-reconstruction components where applicable:
node/concept/process-edge recall and precision, slot correctness for activity,
actor, system, reads, writes, rationale and conditions, endpoint checks,
graph validity, structural/quality component scores, and an aggregate
reference value. `stakeholder_truth_reference_aggregate` contains descriptive
min/max/mean values only. Stable `stakeholder_id` plus name/role identify each
entry; results do not depend on positional ordering or opaque local IDs.
Reference diagnostics also retain the forgetting configuration when the
scenario provides it, plus contracted-node and shortcut-edge counts. A high
reference score with a poor Agent score points to elicitation/recording or
matching loss; a strong Agent score above several low individual references
can indicate successful integration of distributed partial views.

Reference comparison reuses the primary evaluator's Truth business projection,
concept-reference sets, node/edge alignment contracts, scalar/list slot
scorers, endpoint semantics, and structural denominator. Stakeholder `None` is
known absence; `DONT_KNOW` is incomplete and never matches a Truth value or
absence. Structural SOURCE/SINK nodes and boundary edges are excluded from
business reference denominators, so protected boundaries cannot inflate
completeness.

Safe serial shortcut contraction is reported with `is_shortcut`,
`contracted_nodes`, `derived_from_edges`, and `shortcut_provenance`, but the
derived edge receives no automatic exact Truth-business-edge credit. The
Truth graph is never rewritten. Union/combined recoverability across multiple
stakeholder views is intentionally not implemented yet because conflicts,
local concept identity, `DONT_KNOW`, and shortcut evidence need a separate
well-defined merge semantics.

The matcher is deterministic lexical matching, not semantic understanding: it
uses normalized token/Dice overlap, a small low-information-token set, and
scenario-provided locale terms. It can miss genuine paraphrases that share no
canonical/local tokens; it does not claim language-independent semantic
 equivalence and never calls an LLM, embedding model or web service.

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
| `scenario.py` | Truth graphs + filters + named stakeholder reference views (quotation / lab / JA) |
| `evaluation.py` | content/Truth-reconstruction evaluator (AgentGraph + AgentConcepts vs TruthGraph + TruthConcepts), plus per-stakeholder Truth reference diagnostics; provenance reported as diagnostics |
| `tools.py` | glossary + graph tools (optional diagnostic evidence; mentions and terminology bookkeeping plus belief markers; no concept-grounding lifecycle and NO observation tools) |
| `user_simulator.py` | semantic stakeholder: chooses the Semantic Response Plan, validates it, realizes it (graph-native sidecar) |
| `environment.py` | conversation ledger + private sidecar validation/binding + environment-owned Observation creation + `episode_complete` |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/` and
`tests/test_business_interview_roundtrips.py`.
