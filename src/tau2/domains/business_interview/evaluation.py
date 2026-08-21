"""Evaluator for the graph-native business_interview benchmark (v11).

The agent's inferred ``AgentGraph`` is compared to the stakeholder's world
model (``StakeholderKnowledgeGraph``) — the primary achievable target.
Correctness is grounded **only** through private provenance:

    Agent EvidenceRef
      -> Observation span
      -> private annotation (stakeholder semantic ID)
      -> StakeholderKnowledgeGraph element / StakeholderKnowledgeConcept

A span covering several distinct semantic IDs is **globally ambiguous and
grounds nothing** (no cross-credit); a span matching exactly one annotation
grounds that element. The evaluator performs no semantic NLP: no label
matching, no aliases, no embeddings. Semantic-id interpretation goes through
the single canonical resolver ``StakeholderKnowledgeGraph.resolve`` (never
``node:X`` == activity, never ``reads:<item>`` == whole list).

Scoring rules:

- **Node/edge correspondence** falls out of the semantic IDs: an agent
  property ref whose evidence grounds a ``node:<m>:*`` id binds its node to
  stakeholder node ``m`` (deterministic assignment maximizing matches); an
  agent edge whose evidence grounds ``edge:<e>`` maps to stakeholder edge
  ``e`` when the mapped endpoints match.
- **Property scoring uses property evidence ONLY** (never mentions, never
  validation evidence). The three-valued epistemic states score
  asymmetrically:
  - stakeholder ``ConceptRef`` -> the agent must assert the correct grounded
    ConceptRef (recall x precision over the slot's concepts);
  - stakeholder ``None`` (known absent) -> an unasserted agent slot is
    correct; ``DONT_KNOW`` is NOT equivalent to ``None``;
  - stakeholder ``DONT_KNOW`` -> only an explicit, evidenced agent
    ``DONT_KNOW`` marker is correct; an unasserted slot does NOT count as
    DONT_KNOW, and a hidden-Truth guess is wrong.
  Asserting an element the stakeholder does not know exists is a precision
  error. ``DONT_KNOW`` markers must carry evidence that resolves (via the
  canonical resolver) to the corresponding stakeholder DONT_KNOW slots.
- **Concept identity may use mentions + graph provenance** (mapped property
  refs AND grounded concepts' validation evidence): all authentic bindings
  of one Agent concept must agree on one knowledge concept of the same
  kind; per kind the mapping is a bijection over the knowledge concepts
  referenced by the knowledge graph. Conflicting grounding evidence leaves
  the concept unresolved.
- **Validation uses explicit validation/dialogue evidence ONLY**:
  confirmed/partially_confirmed/unknown/disputed statuses and terminology
  agreements must be backed by the corresponding private dialogue events
  addressing the concept's bound knowledge concept. ``grounded`` requires
  the grounding evidence to resolve to exactly one kind-compatible
  knowledge concept (identical rule to the ``ground_concept`` tool, so the
  Agent-visible status and the evaluator binding never disagree).

Truth vs StakeholderKnowledge coverage is reported separately
(``knowledge_coverage``) and never mixed into Agent performance. Known
values AND known absence count as known; ``DONT_KNOW`` and removed
nodes/edges count as unknown.
"""

from typing import Optional

from pydantic import BaseModel

from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    SemanticAnnotation,
    TerminologyConfirmation,
    message_contains_span,
)
from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentGraph,
    ConceptRef,
    DontKnowType,
    EvidenceRef,
    InterviewDB,
    is_dont_know,
    spans_correspond,
)
from tau2.domains.business_interview.grounding import (
    grounded_ids as _grounded_ids,
)
from tau2.domains.business_interview.grounding import (
    grounded_refs,
)
from tau2.domains.business_interview.grounding import (
    obs_by_id as _obs_by_id,
)
from tau2.domains.business_interview.grounding import (
    resolve_span_text as _resolve_span_text,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    slot_concepts,
)

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")

# kind of a knowledge/truth concept = kind of the slots that reference it
_PROPERTY_KIND: dict[str, str] = {
    "activity": "activity",
    "actor": "actor",
    "system": "system",
    "reads": "data",
    "writes": "data",
    "condition": "condition",
    "rationale": "rationale",
}


class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    v11: the spec carries NO semantic matchers. Everything semantic lives in
    the Truth graph / StakeholderKnowledge. Kept as a model for API
    stability; scenarios may leave it empty.
    """


class EvaluationResult(BaseModel):
    protocol_completed: bool
    graph_created: bool
    graph_valid: bool

    node_recall: float
    node_precision: float
    edge_recall: float
    edge_precision: float
    start_correct: bool
    end_recall: float
    end_precision: float
    activity_correctness: float
    actor_correctness: float
    system_correctness: float
    read_correctness: float
    write_correctness: float
    rationale_correctness: float
    condition_correctness: float
    concept_correctness: float
    concept_recall: float
    concept_precision: float
    unsupported_ref_count: int
    fabricated_node_count: int
    fabricated_edge_count: int

    # glossary completion + genuine validation
    glossary_pass: bool
    glossary_complete: bool
    referenced_hypothesized_concepts: list[str]
    glossary_validation_errors: list[str]

    # evidence hygiene
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
    marker_evidence_errors: int
    invalid_observation_reference_count: int
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool
    evidence_pass: bool

    structural_pass: bool
    protocol_pass: bool
    quality_pass: bool

    # scenario knowledge coverage (Truth vs StakeholderKnowledge; NOT part
    # of Agent performance)
    knowledge_coverage: float


# ---------------------------------------------------------------------------
# Private metadata validation (deterministic)
# ---------------------------------------------------------------------------


def _validate_annotations(
    annotations: dict[int, list[SemanticAnnotation]],
    knowledge: StakeholderKnowledge,
    messages: list[dict],
) -> None:
    """Deterministically reject invalid private annotation metadata.

    Every annotation's semantic id must be an element of the stakeholder
    knowledge graph (including DONT_KNOW slots), and the quote/occurrence
    must exactly match the stakeholder message at that turn. Raises
    ``ValueError``.
    """
    resolver = knowledge.graph.resolve
    for turn, turn_annotations in (annotations or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for annotation in turn_annotations:
            if resolver(annotation.semantic_id) is None:
                raise ValueError(
                    f"annotation semantic_id {annotation.semantic_id!r} "
                    f"(turn {turn}) is not an element of the stakeholder "
                    f"knowledge graph"
                )
            if not message_contains_span(
                message, annotation.quote, annotation.occurrence
            ):
                raise ValueError(
                    f"annotation {annotation.semantic_id!r} (turn {turn}): "
                    f"quote {annotation.quote!r} occurrence "
                    f"{annotation.occurrence} does not exactly match the "
                    f"stakeholder message"
                )


def _validate_events(
    alignments: Optional[dict[int, list[ConceptAlignmentAssertion]]],
    terminology: Optional[dict[int, list[TerminologyConfirmation]]],
    knowledge: StakeholderKnowledge,
    messages: list[dict],
) -> None:
    """Deterministically reject invalid private dialogue-event metadata.

    Every event must address a StakeholderKnowledgeConcept id, and
    quote/occurrence must exactly match the message at that turn. Raises
    ``ValueError``.
    """
    resolver = knowledge.graph.resolve
    for turn, turn_events in (alignments or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for event in turn_events:
            resolved = resolver(event.semantic_id)
            if resolved is None or resolved.kind != "concept":
                raise ValueError(
                    f"alignment event (turn {turn}) for concept "
                    f"{event.semantic_id!r} is not a knowledge concept"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"alignment event {event.act!r} for {event.semantic_id!r} "
                    f"(turn {turn}): quote {event.quote!r} occurrence "
                    f"{event.occurrence} does not exactly match the "
                    f"stakeholder message"
                )
    for turn, turn_events in (terminology or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for event in turn_events:
            resolved = resolver(event.semantic_id)
            if resolved is None or resolved.kind != "concept":
                raise ValueError(
                    f"terminology event (turn {turn}) for concept "
                    f"{event.semantic_id!r} is not a knowledge concept"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"terminology event for {event.semantic_id!r} (turn {turn}): "
                    f"quote {event.quote!r} occurrence {event.occurrence} does "
                    f"not exactly match the stakeholder message"
                )


# ---------------------------------------------------------------------------
# Span-based grounding (provenance only; shared machinery in grounding.py)
# ---------------------------------------------------------------------------


def grounded_semantic_ids(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    evidence: list[EvidenceRef],
) -> tuple[set[str], int, int]:
    """Semantic IDs grounded by a list of EvidenceRefs via the GLOBAL span
    rule (shared machinery in ``grounding.grounded_ids``). Returns (grounded
    ids, invalid count, ambiguous count)."""
    return _grounded_ids(db, annotations, evidence)


def resolve_grounding_refs(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    evidence: list[EvidenceRef],
) -> tuple[list[tuple[EvidenceRef, str]], int, int]:
    """Per-ref single-id grounding resolution via the GLOBAL span rule
    (shared machinery in ``grounding.grounded_refs``): [(ref, semantic_id)]
    for refs resolving to exactly one id, plus (invalid, ambiguous) counts."""
    return grounded_refs(db, annotations, evidence)


def _ref_evidence(ref: ConceptRef) -> list[EvidenceRef]:
    """The property evidence of a ref — property evidence ONLY (mentions and
    validation evidence never enter property scoring)."""
    return list(ref.evidence)


def _candidate_node_ids(grounded: set[str], knowledge) -> set[str]:
    """Stakeholder node ids (opaque local ids) appearing in grounded
    ``node:*`` semantic ids (node existence, slots and elements — resolved
    through the canonical resolver)."""
    out: set[str] = set()
    resolver = knowledge.graph.resolve
    for sid in grounded:
        resolved = resolver(sid)
        if resolved is not None and resolved.kind in (
            "node",
            "node_slot",
            "node_element",
        ):
            out.add(resolved.node_id)
    return out


def _candidate_edge_ids(grounded: set[str], knowledge) -> set[str]:
    """Stakeholder edge ids (opaque local ids) appearing in grounded
    ``edge:*`` semantic ids (edge existence or the condition slot)."""
    out: set[str] = set()
    resolver = knowledge.graph.resolve
    for sid in grounded:
        resolved = resolver(sid)
        if resolved is not None and resolved.kind in ("edge", "edge_slot"):
            out.add(resolved.edge_id)
    return out


def _knowledge_value_concepts(knowledge: StakeholderKnowledge, sid: str) -> set[str]:
    """The knowledge concept(s) a grounded semantic id stands for (canonical
    resolver): element ids pin exactly one concept; slot ids resolve to the
    slot's concepts; concept ids resolve to themselves; node/edge existence
    resolves to NO concept."""
    resolved = knowledge.graph.resolve(sid)
    if resolved is None:
        return set()
    if resolved.kind == "node_element":
        assert resolved.ref is not None
        return {resolved.ref.concept_id}
    if resolved.kind in ("node_slot", "edge_slot"):
        return slot_concepts(resolved.value)
    if resolved.kind == "concept":
        assert resolved.concept is not None
        return {resolved.concept.id}
    return set()  # node / edge existence: no concept


def _match_nodes_and_edges(
    agent: AgentGraph,
    knowledge: StakeholderKnowledge,
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Deterministic node/edge correspondence.

    Node candidates come from the property-level provenance (any agent ref
    whose evidence grounds a ``node:<m>:*`` id); the assignment maximizes
    the number of refs whose grounded ids belong to the assigned node's
    element-id set, ties broken lexicographically. Edges map when their
    evidence grounds exactly one ``edge:<e>`` id whose endpoints match the
    mapped nodes.

    Returns (node mapping, agent edge id -> stakeholder edge id).
    """
    tn: dict[str, set[str]] = {}
    for nid, node in agent.nodes.items():
        cand: set[str] = set()
        for prop in _NODE_PROPS:
            for ref in node.asserted_refs(prop):
                grounded, _i, _a = _grounded_ids(db, annotations, _ref_evidence(ref))
                cand.update(_candidate_node_ids(grounded, knowledge))
        # ABSENT/DONT_KNOW marker evidence is property provenance too and
        # contributes node candidates
        for prop in _NODE_PROPS:
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                grounded, _i, _a = _grounded_ids(db, annotations, slot.evidence)
                cand.update(_candidate_node_ids(grounded, knowledge))
        if cand:
            tn[nid] = cand
    order = sorted(tn)

    def node_match_count(nid: str, mnid: str) -> int:
        """How many refs of agent node ``nid`` ground an element of
        stakeholder node ``mnid``."""
        count = 0
        node = agent.nodes[nid]
        prefix = f"node:{mnid}:"
        for prop in _NODE_PROPS:
            for ref in node.asserted_refs(prop):
                grounded, _i, _a = _grounded_ids(db, annotations, _ref_evidence(ref))
                if any(s.startswith(prefix) for s in grounded):
                    count += 1
        return count

    best_mapping: dict[str, str] = {}
    best_score = -1

    def enumerate_assignments(i: int, mapping: dict[str, str]) -> None:
        nonlocal best_mapping, best_score
        if i == len(order):
            score = sum(node_match_count(nid, mnid) for nid, mnid in mapping.items())
            if score > best_score:
                best_score = score
                best_mapping = dict(mapping)
            return
        nid = order[i]
        for mnid in sorted(tn[nid]):
            if mnid in mapping.values():
                continue
            mapping[nid] = mnid
            enumerate_assignments(i + 1, mapping)
            del mapping[nid]

    if order:
        enumerate_assignments(0, {})

    mapping = best_mapping
    # edges: grounded edge ids whose endpoints match the mapped nodes
    edge_map: dict[str, str] = {}
    for eid, edge in agent.edges.items():
        grounded, _i, _a = _grounded_ids(db, annotations, edge.evidence)
        candidates = sorted(_candidate_edge_ids(grounded, knowledge))
        a = mapping.get(edge.from_node)
        b = mapping.get(edge.to_node)
        if a is None or b is None:
            continue
        for teid in candidates:
            te = knowledge.graph.edges.get(teid)
            if te is not None and te.from_node == a and te.to_node == b:
                edge_map[eid] = teid
                break
    return mapping, edge_map


# ---------------------------------------------------------------------------
# Property scoring (property evidence ONLY)
# ---------------------------------------------------------------------------


def _slot_value(knowledge: StakeholderKnowledge, mnid: str, prop: str):
    """The three-valued knowledge slot value for (mnid, prop)."""
    node = knowledge.graph.nodes.get(mnid)
    if node is None:
        return None
    if prop in ("reads", "writes"):
        return getattr(node, prop)
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


def _expected_ids(knowledge: StakeholderKnowledge, mnid: str, prop: str) -> set[str]:
    """The element ids an agent ref must ground to fill (mnid, prop).

    Scalars resolve to the property slot id; reads/writes resolve to their
    element ids (``node:m:reads:<k>``)."""
    value = _slot_value(knowledge, mnid, prop)
    if prop in ("reads", "writes") and isinstance(value, list):
        return {f"node:{mnid}:{prop}:{ref.concept_id}" for ref in value}
    if prop in ("reads", "writes"):
        return set()
    return {f"node:{mnid}:{prop}"}


def _slot_value_concepts(
    knowledge: StakeholderKnowledge, mnid: str, prop: str
) -> set[str]:
    """The knowledge concept ids a correct ref must bind to for this slot."""
    value = _slot_value(knowledge, mnid, prop)
    if isinstance(value, list):
        return {r.concept_id for r in value}
    if isinstance(value, ConceptRef):
        return {value.concept_id}
    return set()


def _marker_slot_evidence(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    marker,
) -> set[str]:
    """Semantic ids grounded by an ABSENT/DONT_KNOW marker's evidence
    (property evidence ONLY — never mentions or validation evidence)."""
    grounded, _i, _a = _grounded_ids(db, annotations, list(marker.evidence))
    return grounded


def _property_score(
    agent: AgentGraph,
    knowledge: StakeholderKnowledge,
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    mnid: str,
    anid: str,
    prop: str,
    agent_to_knowledge: dict[str, str],
) -> tuple[float, int]:
    """Score one node property (activity/actor/system/reads/writes/rationale)
    under the four-state Agent model.

    Uses property evidence ONLY. The four Agent epistemic states score
    against the three stakeholder states as follows:
    - stakeholder ``ConceptRef``: a matching evidenced Agent ConceptRef is
      correct; UNSET / ABSENT / DONT_KNOW / wrong value are incorrect;
    - stakeholder ``None`` (known absent): an Agent ABSENT marker whose
      evidence resolves to the EXACT mapped stakeholder slot
      (``node:<mnid>:<prop>``) is correct; UNSET / DONT_KNOW / ConceptRef
      are incorrect — "not asserted" never scores as known absence;
    - stakeholder ``DONT_KNOW``: an Agent DONT_KNOW marker whose evidence
      resolves to the EXACT mapped stakeholder DONT_KNOW slot is correct;
      UNSET / ABSENT / ConceptRef are incorrect, and a hidden-Truth guess
      remains wrong.

    Evidence about ANOTHER node's slot never supports this node's marker.
    For reads/writes, v1 keeps whole-property ABSENT/DONT_KNOW only.

    Returns (score, unsupported_ref_count).
    """
    value = _slot_value(knowledge, mnid, prop)
    agent_slot = agent.nodes[anid].slot_value(prop)
    if is_dont_know(value):
        if not isinstance(agent_slot, DontKnowType):
            return 0.0, 0
        # exact mapped-slot provenance: the marker's evidence must ground
        # node:<mnid>:<prop> itself
        slot_sid = f"node:{mnid}:{prop}"
        grounded = _marker_slot_evidence(db, annotations, agent_slot)
        return (1.0 if slot_sid in grounded else 0.0), 0
    if value is None:
        if not isinstance(agent_slot, AbsentType):
            return 0.0, 0
        # exact mapped-slot provenance: the marker's evidence must ground
        # node:<mnid>:<prop> itself
        slot_sid = f"node:{mnid}:{prop}"
        grounded = _marker_slot_evidence(db, annotations, agent_slot)
        return (1.0 if slot_sid in grounded else 0.0), 0
    refs = agent.nodes[anid].asserted_refs(prop)
    if not refs:
        return 0.0, 0
    expected = _expected_ids(knowledge, mnid, prop)
    concepts = _slot_value_concepts(knowledge, mnid, prop)
    covered: set[str] = set()
    supported = 0
    unsupported = 0
    for ref in refs:
        grounded, _i, _a = _grounded_ids(db, annotations, _ref_evidence(ref))
        hits = grounded & expected
        bound = agent_to_knowledge.get(ref.concept_id)
        if not hits or bound is None or bound not in concepts:
            unsupported += 1
            continue
        supported += 1
        # element ids pin the exact concept; slot ids cover all concepts
        for sid in hits:
            if sid.startswith("node:") and len(sid.split(":")) == 4:
                covered.add(sid.split(":")[3])
            else:
                covered.update(concepts)
    recall = len(covered & concepts) / len(concepts) if concepts else 1.0
    precision = supported / len(refs)
    return recall * precision, unsupported


# ---------------------------------------------------------------------------
# Concept identity (per kind; mentions may participate)
# ---------------------------------------------------------------------------


def _concept_bindings(
    agent: AgentGraph,
    knowledge: StakeholderKnowledge,
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    mapping: dict[str, str],
    edge_map: dict[str, str],
) -> tuple[float, float, dict[str, str]]:
    """Concept-level binding integrity against the EXPECTED
    StakeholderKnowledgeConcept set (requirement 5: no vacuous success).

    Binding sources (all must agree on ONE knowledge concept):
    property refs, edge conditions, mentions, and grounded concepts'
    validation evidence. A claim is an ATTEMPTED Agent concept (one the
    Agent references in its graph) with exactly one candidate knowledge
    concept of a compatible kind.

    Returns (concept_recall, concept_precision, agent concept id ->
    knowledge concept id) where:
    - expected = the knowledge concepts the stakeholder graph actually
      references with values;
    - attempted = the Agent concepts the Agent actually references in its
      graph (mapped and unmapped nodes/edges) — the precision denominator;
    - correct = expected knowledge concepts claimed by exactly ONE Agent
      concept (one-to-one identity);
    - concept_recall = |correct| / |expected|  (missing expected concepts
      reduce it; an EMPTY AgentGraph has recall 0.0 — never 1.0);
    - concept_precision = |correct Agent concepts| / |attempted| — an
      attempted concept is correct only when it binds one-to-one to a
      knowledge concept; EXTRA (no binding), unbound (no evidence),
      ambiguous, conflicting, kind-incompatible and duplicated/split
      attempts all reduce precision (an EMPTY AgentGraph has no attempted
      concepts: precision uses the documented neutral value 1.0);
    - agent_to_knowledge carries ONLY the correct, unambiguous bindings
      (used by property scoring).
    """
    candidates: dict[str, set[str]] = {}
    referenced: set[str] = set()

    def add_binding(agent_cid: str, knowledge_ids: set[str]) -> None:
        if not knowledge_ids:
            return
        prev = candidates.setdefault(agent_cid, set(knowledge_ids))
        candidates[agent_cid] = prev & knowledge_ids

    for anid, mnid in mapping.items():
        for prop in _NODE_PROPS:
            for ref in agent.nodes[anid].asserted_refs(prop):
                referenced.add(ref.concept_id)
                grounded, _i, _a = _grounded_ids(db, annotations, _ref_evidence(ref))
                expected = _expected_ids(knowledge, mnid, prop)
                for sid in grounded & expected:
                    add_binding(
                        ref.concept_id, _knowledge_value_concepts(knowledge, sid)
                    )
    for eid, meid in edge_map.items():
        edge = agent.edges[eid]
        if isinstance(edge.condition, ConceptRef) and edge.condition.asserted:
            referenced.add(edge.condition.concept_id)
            grounded, _i, _a = _grounded_ids(
                db, annotations, _ref_evidence(edge.condition)
            )
            sid = f"edge:{meid}:condition"
            if sid in grounded:
                add_binding(
                    edge.condition.concept_id,
                    _knowledge_value_concepts(knowledge, sid),
                )
    # unmapped node/edge refs are STILL attempted references: their concepts
    # must satisfy the same one-to-one identity rules (fabricated or
    # unbound claims reduce precision, never silently vanish from the
    # denominator).
    for anid in agent.nodes:
        if anid in mapping:
            continue
        for prop in _NODE_PROPS:
            for ref in agent.nodes[anid].asserted_refs(prop):
                referenced.add(ref.concept_id)
                grounded, _i, _a = _grounded_ids(db, annotations, _ref_evidence(ref))
                for sid in grounded:
                    add_binding(
                        ref.concept_id, _knowledge_value_concepts(knowledge, sid)
                    )
    for eid in agent.edges:
        if eid in edge_map:
            continue
        edge = agent.edges[eid]
        if isinstance(edge.condition, ConceptRef) and edge.condition.asserted:
            referenced.add(edge.condition.concept_id)
            grounded, _i, _a = _grounded_ids(
                db, annotations, _ref_evidence(edge.condition)
            )
            for sid in grounded:
                add_binding(
                    edge.condition.concept_id,
                    _knowledge_value_concepts(knowledge, sid),
                )
    # mentions may participate in concept identity (never in property scoring)
    for cid, concept in agent.concepts.items():
        for mention in concept.mentions:
            grounded, _i, _a = _grounded_ids(db, annotations, [mention])
            for sid in grounded:
                add_binding(cid, _knowledge_value_concepts(knowledge, sid))
    # grounded concepts' validation evidence is graph provenance and
    # participates in identity exactly like mentions (the tool guarantees the
    # same unique kind-compatible resolution, so the Agent-visible grounded
    # status and the evaluator binding never disagree). Conflicting
    # evidence (property refs / mentions disagreeing with the grounding)
    # leaves the concept unresolved (empty candidate set).
    for cid, concept in agent.concepts.items():
        if concept.validation_status != "grounded":
            continue
        for ev in concept.validation_evidence:
            grounded, _i, _a = _grounded_ids(db, annotations, [ev])
            for sid in grounded:
                add_binding(cid, _knowledge_value_concepts(knowledge, sid))

    # knowledge concepts the graph actually references (with values)
    knowledge_kinds: dict[str, str] = {}
    for mnid, node in knowledge.graph.nodes.items():
        for prop in _NODE_PROPS:
            value = _slot_value(knowledge, mnid, prop)
            if isinstance(value, list):
                for ref in value:
                    knowledge_kinds.setdefault(
                        ref.concept_id, _PROPERTY_KIND.get(prop, "data")
                    )
            elif isinstance(value, ConceptRef):
                knowledge_kinds.setdefault(
                    value.concept_id, _PROPERTY_KIND.get(prop, "data")
                )
    for meid, edge in knowledge.graph.edges.items():
        if isinstance(edge.condition, ConceptRef):
            knowledge_kinds.setdefault(edge.condition.concept_id, "condition")

    # claims: referenced agent concepts with a UNIQUE kind-compatible
    # candidate (conflicting/ambiguous evidence leaves the concept
    # unresolved — no claim)
    claims: dict[str, str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts.get(cid)
        if concept is None:
            continue
        cand = candidates.get(cid, set())
        if len(cand) != 1:
            continue
        kid = next(iter(cand))
        if knowledge_kinds.get(kid) != concept.kind:
            continue
        claims[cid] = kid

    claimed_by_kid: dict[str, list[str]] = {}
    for cid, kid in claims.items():
        claimed_by_kid.setdefault(kid, []).append(cid)
    correct = {kid for kid, cids in claimed_by_kid.items() if len(cids) == 1}

    expected = set(knowledge_kinds)
    concept_recall = len(correct) / len(expected) if expected else 1.0
    # precision is diagnostic over the Agent's ATTEMPTED references: every
    # attempted concept that is not a correct one-to-one binding (extra /
    # unbound / ambiguous / conflicting / kind-incompatible / duplicated)
    # reduces it. An empty AgentGraph has no attempted concepts and uses the
    # documented neutral precision 1.0 (concept_correctness is still 0).
    attempted = referenced
    correct_agent = {cid for cid, kid in claims.items() if kid in correct}
    concept_precision = len(correct_agent) / len(attempted) if attempted else 1.0
    agent_to_knowledge = {cid: kid for cid, kid in claims.items() if kid in correct}
    return concept_recall, concept_precision, agent_to_knowledge


# ---------------------------------------------------------------------------
# Glossary validation (grounded / confirmed via explicit evidence)
# ---------------------------------------------------------------------------


def _events_corresponding_at_span(
    events: list[ConceptAlignmentAssertion],
    text: str,
    ev_span: tuple[int, int],
    acts: set[str],
    semantic_id: Optional[str],
) -> bool:
    """True when an event with one of ``acts`` (and, when given, that exact
    knowledge concept id) corresponds to the evidence span."""
    for event in events:
        if event.act not in acts:
            continue
        if semantic_id is not None and event.semantic_id != semantic_id:
            continue
        event_span = _resolve_span_text(text, event.quote, event.occurrence)
        if event_span is not None and spans_correspond(ev_span, event_span):
            return True
    return False


def _grounding_error(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    knowledge: StakeholderKnowledge,
    concept,
) -> Optional[str]:
    """Why a ``grounded`` concept's grounding evidence is invalid, or None.

    The rule is IDENTICAL to the ``ground_concept`` tool's: the evidence
    must resolve (global span rule + canonical resolver) to exactly one
    knowledge concept of a kind compatible with the Agent concept. Because
    the tool and the evaluator run the same deterministic check, the
    Agent-visible ``grounded`` status and the evaluator binding never
    disagree; conflicting grounding evidence is rejected at the tool and
    leaves the concept unresolved at evaluation.
    """
    evs = concept.validation_evidence
    if not evs:
        return "grounded requires evidence"
    results, invalid, ambiguous = grounded_refs(db, annotations, evs)
    if invalid or ambiguous or len(results) != len(evs):
        return "grounding evidence must resolve to exactly one semantic id per span"
    concepts: set[str] = set()
    for _ref, sid in results:
        concepts.update(_knowledge_value_concepts(knowledge, sid))
    if len(concepts) != 1:
        return (
            "grounding evidence must represent exactly one knowledge concept "
            "(ambiguous or unrelated evidence)"
        )
    kid = next(iter(concepts))
    if knowledge.graph.concepts[kid].kind != concept.kind:
        return "grounding evidence represents a kind-incompatible knowledge concept"
    return None


def _glossary_validation(
    agent: AgentGraph,
    db: InterviewDB,
    alignments: dict[int, list[ConceptAlignmentAssertion]],
    terminology: dict[int, list[TerminologyConfirmation]],
    agent_to_knowledge: dict[str, str],
    knowledge: StakeholderKnowledge,
    annotations: dict[int, list[SemanticAnnotation]],
) -> tuple[bool, list[str], list[str]]:
    """Completion + genuine-validation rules.

    - every referenced concept must be resolved (not hypothesized);
    - ``grounded``: the grounding evidence must resolve (via the canonical
      resolver) to exactly one kind-compatible knowledge concept — the same
      deterministic rule as the ``ground_concept`` tool, so the
      Agent-visible status and the evaluator binding never disagree;
    - ``confirmed`` / ``partially_confirmed`` / ``unknown`` / ``disputed``:
      validation evidence must correspond to the matching private dialogue
      event (act confirm/partial/unknown/dispute) addressing the concept's
      bound knowledge concept (disputed: >= 2 distinct Observations);
    - terminology agreements need a terminology-confirmation event with the
      same bound knowledge concept, the same proposed term, and a
      corresponding cited span;
    - **no bulk self-validation**: one span may back at most one concept's
      validation evidence.

    Returns (pass, referenced_hypothesized, validation_errors).
    """
    referenced = agent.referenced_concepts()
    hypothesized = sorted(
        cid
        for cid in referenced
        if agent.concepts.get(cid) is not None
        and agent.concepts[cid].validation_status == "hypothesized"
    )
    if hypothesized:
        return False, hypothesized, []

    errors: list[str] = []
    span_usage: dict[tuple[str, str, int], str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts[cid]
        bound = agent_to_knowledge.get(cid)
        if concept.validation_status == "grounded":
            error = _grounding_error(db, annotations, knowledge, concept)
            if error is not None:
                errors.append(f"concept {cid}: {error}")
            continue
        if concept.validation_status in (
            "confirmed",
            "partially_confirmed",
            "unknown",
            "disputed",
        ):
            for ev in concept.validation_evidence:
                key = (ev.observation_id, ev.quote, ev.occurrence)
                prev = span_usage.get(key)
                if prev is not None and prev != cid:
                    errors.append(
                        f"evidence span {key} backs both {prev} and {cid} "
                        f"(bulk self-validation)"
                    )
                span_usage[key] = cid
                obs = _obs_by_id(db, ev.observation_id)
                if obs is None:
                    errors.append(
                        f"concept {cid}: validation evidence references "
                        f"unknown observation"
                    )
                    continue
                ev_span = ev.resolve_span(obs.text)
                if ev_span is None:
                    errors.append(
                        f"concept {cid}: validation evidence span is not exact"
                    )
                    continue
                events_at_turn = alignments.get(obs.turn, [])
                if concept.validation_status == "confirmed":
                    acts = {"confirm"}
                elif concept.validation_status == "partially_confirmed":
                    acts = {"partial"}
                elif concept.validation_status == "unknown":
                    acts = {"unknown"}
                else:  # disputed handled below
                    acts = set()
                if (
                    acts
                    and bound is not None
                    and not _events_corresponding_at_span(
                        events_at_turn, obs.text, ev_span, acts, bound
                    )
                ):
                    errors.append(
                        f"concept {cid}: {concept.validation_status} evidence "
                        f"does not correspond to a private dialogue event "
                        f"(act={sorted(acts)[0]}) for knowledge concept "
                        f"{bound!r}"
                    )
                elif acts and bound is None:
                    errors.append(
                        f"concept {cid}: {concept.validation_status} evidence "
                        f"does not correspond to a private dialogue event for "
                        f"its bound knowledge concept (concept is not bound)"
                    )
    # disputed needs >= 2 distinct observations with matching dispute events
    for cid in sorted(referenced):
        concept = agent.concepts[cid]
        if concept.validation_status != "disputed":
            continue
        bound = agent_to_knowledge.get(cid)
        obs_with_support = set()
        for ev in concept.validation_evidence:
            obs = _obs_by_id(db, ev.observation_id)
            if obs is None:
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                continue
            if bound is not None and _events_corresponding_at_span(
                alignments.get(obs.turn, []), obs.text, ev_span, {"dispute"}, bound
            ):
                obs_with_support.add(ev.observation_id)
        if bound is None or len(obs_with_support) < 2:
            bound_desc = repr(bound) if bound is not None else "(unbound)"
            errors.append(
                f"concept {cid}: disputed needs private dialogue events "
                f"(act=dispute) for knowledge concept {bound_desc} from at "
                f"least two distinct Observations"
            )
    # terminology agreements
    for agreement in agent.terminology_agreements:
        if agreement.concept_id not in agent.concepts:
            errors.append(
                f"terminology agreement for unknown concept {agreement.concept_id!r}"
            )
            continue
        bound = agent_to_knowledge.get(agreement.concept_id)
        if bound is None:
            continue
        matched = False
        for ev in agreement.evidence:
            obs = _obs_by_id(db, ev.observation_id)
            if obs is None:
                errors.append(
                    f"terminology agreement for {agreement.concept_id}: "
                    f"evidence references unknown observation"
                )
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                errors.append(
                    f"terminology agreement for {agreement.concept_id}: "
                    f"evidence span is not exact"
                )
                continue
            for event in terminology.get(obs.turn, []):
                if event.semantic_id != bound:
                    continue
                if event.proposed_term != agreement.term:
                    continue
                event_span = _resolve_span_text(obs.text, event.quote, event.occurrence)
                if event_span is not None and spans_correspond(ev_span, event_span):
                    matched = True
                    break
        if not matched:
            errors.append(
                f"terminology agreement {agreement.term!r} for "
                f"{agreement.concept_id}: no private terminology-confirmation "
                f"event for knowledge concept {bound!r} with the same "
                f"proposed term at a corresponding cited span"
            )
    return (not errors), hypothesized, errors


# ---------------------------------------------------------------------------
# Evidence hygiene
# ---------------------------------------------------------------------------


def _evidence_metrics(
    db: InterviewDB,
    agent: AgentGraph,
) -> tuple[int, int, float, float, float]:
    """(invalid, invalid_obs_refs, node_cov, ref_cov, edge_cov)."""
    invalid = 0
    invalid_obs_refs = 0
    ref_total = ref_hit = 0
    node_total = node_hit = 0
    edge_total = edge_hit = 0
    obs_text = {o.id: o.text for o in db.observations}

    def span_ok(ev: EvidenceRef) -> bool:
        nonlocal invalid, invalid_obs_refs
        text = obs_text.get(ev.observation_id)
        if text is None:
            invalid += 1
            if ev.observation_id not in obs_text:
                invalid_obs_refs += 1
            return False
        if ev.resolve_span(text) is None:
            invalid += 1
            return False
        return True

    def check_evidence(evs: list[EvidenceRef]) -> bool:
        ok = True
        for ev in evs:
            if not span_ok(ev):
                ok = False
        return ok

    for node in agent.nodes.values():
        node_refs: list[ConceptRef] = []
        marker_evs: list[EvidenceRef] = []
        for prop in _NODE_PROPS:
            node_refs.extend(node.refs(prop))
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                marker_evs.extend(slot.evidence)
        node_total += 1
        ref_ok = any(
            r.asserted and any(span_ok(ev) for ev in _ref_evidence(r))
            for r in node_refs
        )
        slot_ok = False
        for ev in marker_evs:
            ref_total += 1
            if span_ok(ev):
                ref_hit += 1
                slot_ok = True
        if ref_ok or slot_ok:
            node_hit += 1
        for r in node_refs:
            if not r.asserted:
                continue
            ref_total += 1
            evs = _ref_evidence(r)
            if evs and check_evidence(evs):
                ref_hit += 1
    for edge in agent.edges.values():
        edge_total += 1
        if edge.evidence and check_evidence(edge.evidence):
            edge_hit += 1
        cond = edge.condition
        if isinstance(cond, ConceptRef):
            check_evidence(_ref_evidence(cond))
        elif isinstance(cond, DontKnowType):
            for ev in cond.evidence:
                ref_total += 1
                if span_ok(ev):
                    ref_hit += 1
    node_cov = node_hit / node_total if node_total else 1.0
    ref_cov = ref_hit / ref_total if ref_total else 1.0
    edge_cov = edge_hit / edge_total if edge_total else 1.0
    return invalid, invalid_obs_refs, node_cov, ref_cov, edge_cov


def _all_referenced_observation_ids(agent: AgentGraph) -> set[str]:
    ids: set[str] = set()
    for node in agent.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                for ev in _ref_evidence(ref):
                    ids.add(ev.observation_id)
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                for ev in slot.evidence:
                    ids.add(ev.observation_id)
    for edge in agent.edges.values():
        for ev in edge.evidence:
            ids.add(ev.observation_id)
        cond = edge.condition
        if isinstance(cond, ConceptRef):
            for ev in _ref_evidence(cond):
                ids.add(ev.observation_id)
        elif isinstance(cond, (AbsentType, DontKnowType)):
            for ev in cond.evidence:
                ids.add(ev.observation_id)
    return ids


def _marker_evidence_errors(
    agent: AgentGraph,
    knowledge: StakeholderKnowledge,
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    mapping: dict[str, str],
    edge_map: dict[str, str],
) -> int:
    """ABSENT/DONT_KNOW markers whose evidence does not resolve to the EXACT
    mapped stakeholder slot.

    ABSENT is valid only when its evidence resolves (global span rule +
    canonical resolver) to the mapped stakeholder slot
    ``node:<mnid>:<prop>`` / ``edge:<meid>:condition`` whose value is None;
    DONT_KNOW only when it resolves to the mapped stakeholder DONT_KNOW
    slot. Evidence about ANOTHER node's or edge's slot never supports this
    marker (``grounded == {expected_sid}``). Markers on unmapped agent
    elements are unsupported."""
    errors = 0
    resolver = knowledge.graph.resolve

    def check(evs: list[EvidenceRef], expected_sid: str, expect_absent: bool) -> None:
        nonlocal errors
        if not evs:
            errors += 1
            return
        results, invalid, ambiguous = grounded_refs(db, annotations, evs)
        if invalid or ambiguous or len(results) != len(evs):
            errors += 1
            return
        grounded = {sid for _r, sid in results}
        if grounded != {expected_sid}:
            errors += 1
            return
        resolved = resolver(expected_sid)
        if resolved is None:
            errors += 1
            return
        if expect_absent:
            if resolved.value is not None:
                errors += 1
        elif not is_dont_know(resolved.value):
            errors += 1

    for anid, mnid in mapping.items():
        for prop in _NODE_PROPS:
            slot = agent.nodes[anid].slot_value(prop)
            if isinstance(slot, AbsentType):
                check(slot.evidence, f"node:{mnid}:{prop}", expect_absent=True)
            elif isinstance(slot, DontKnowType):
                check(slot.evidence, f"node:{mnid}:{prop}", expect_absent=False)
    for eid, meid in edge_map.items():
        cond = agent.edges[eid].condition
        if isinstance(cond, AbsentType):
            check(cond.evidence, f"edge:{meid}:condition", expect_absent=True)
        elif isinstance(cond, DontKnowType):
            check(cond.evidence, f"edge:{meid}:condition", expect_absent=False)
    # markers on unmapped agent elements are unsupported
    for anid in agent.nodes:
        if anid in mapping:
            continue
        for prop in _NODE_PROPS:
            slot = agent.nodes[anid].slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                errors += 1
    for eid in agent.edges:
        if eid in edge_map:
            continue
        if isinstance(agent.edges[eid].condition, (AbsentType, DontKnowType)):
            errors += 1
    return errors


def _knowledge_node_slot(node, prop: str):
    """The three-valued knowledge slot value of a stakeholder node."""
    if prop in ("reads", "writes"):
        return getattr(node, prop)
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


def _knowledge_coverage(truth, knowledge: StakeholderKnowledge) -> float:
    """Truth vs StakeholderKnowledge coverage, via the private Truth
    mappings and the canonical resolver (informational; never part of Agent
    performance). Known values AND known absence count as known; DONT_KNOW
    slots and removed nodes/edges count as unknown; node existence is never
    confused with the activity slot."""
    kg = knowledge.graph
    node_t2l = {t: k for k, t in kg.node_truth_ids.items()}
    edge_t2l = {t: k for k, t in kg.edge_truth_ids.items()}
    concept_t2l = {c.truth_concept_id: k for k, c in kg.concepts.items()}
    total = 0
    known = 0

    for nid, node in truth.nodes.items():
        local = node_t2l.get(nid)
        kn = kg.nodes.get(local) if local is not None else None
        total += 1
        if kn is not None:
            known += 1  # node existence is known (removed nodes are not)
        for prop in _NODE_PROPS:
            total += 1
            if kn is None:
                if prop in ("reads", "writes"):
                    for _ref in getattr(node, prop) or []:
                        total += 1  # removed node -> elements unknown
                continue
            value = _knowledge_node_slot(kn, prop)
            if not is_dont_know(value):
                known += 1  # known value OR known absence
            if prop in ("reads", "writes"):
                # each Truth element is addressable: known when the whole
                # property is known (list value or known-absent None),
                # unknown when the property is DONT_KNOW
                for ref in getattr(node, prop) or []:
                    total += 1
                    if is_dont_know(value):
                        continue
                    if value is None:
                        known += 1  # known-absent whole property
                    elif isinstance(value, list) and concept_t2l.get(
                        ref.concept_id
                    ) in {r.concept_id for r in value}:
                        known += 1
    for eid, edge in truth.edges.items():
        local = edge_t2l.get(eid)
        ke = kg.edges.get(local) if local is not None else None
        total += 1
        if ke is not None:
            known += 1  # edge existence is known (removed edges are not)
        total += 1  # the condition slot itself
        if ke is not None and not is_dont_know(ke.condition):
            known += 1
    return known / total if total else 0.0


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate(
    db: InterviewDB,
    knowledge: StakeholderKnowledge,
    spec: EvaluationSpec,
    stakeholder=None,
    *,
    truth=None,
    annotations: Optional[dict[int, list[SemanticAnnotation]]] = None,
    alignments: Optional[dict[int, list[ConceptAlignmentAssertion]]] = None,
    terminology: Optional[dict[int, list[TerminologyConfirmation]]] = None,
) -> EvaluationResult:
    """Evaluate the inferred AgentGraph against the stakeholder's world model.

    ``knowledge`` is the scenario's ``StakeholderKnowledge``;
    ``annotations`` / ``alignments`` / ``terminology`` are the private
    sidecar ledgers (``SemanticLedger`` accessors). All are evaluator-only;
    the Agent never sees them. Invalid private metadata is rejected
    deterministically. ``truth`` (optional) is only used for the separate
    knowledge-coverage metric, never for Agent scoring.
    """
    agent = db.graph if db.graph is not None else AgentGraph()
    protocol = db.interview_complete
    graph_created = len(agent.nodes) > 0
    graph_valid = agent.is_valid
    annotation_ledger = annotations or {}
    alignment_ledger = alignments or {}
    terminology_ledger = terminology or {}
    _validate_annotations(annotation_ledger, knowledge, db.messages)
    _validate_events(alignment_ledger, terminology_ledger, knowledge, db.messages)

    # ---- knowledge coverage (Truth vs StakeholderKnowledge; informational) --
    knowledge_coverage = (
        _knowledge_coverage(truth, knowledge) if truth is not None else 0.0
    )

    # ---- node/edge correspondence ------------------------------------------
    mapping, edge_map = _match_nodes_and_edges(agent, knowledge, db, annotation_ledger)
    knowledge_node_ids = list(knowledge.graph.nodes)
    agent_node_ids = list(agent.nodes)
    matched_knowledge = set(mapping.values())
    matched_agent = set(mapping.keys())
    node_recall = (
        len(matched_knowledge) / len(knowledge_node_ids) if knowledge_node_ids else 0.0
    )
    node_precision = len(matched_agent) / len(agent_node_ids) if agent_node_ids else 0.0
    fabricated_node_count = len(agent_node_ids) - len(matched_agent)

    # ---- edges -------------------------------------------------------------
    knowledge_edge_list = list(knowledge.graph.edges.values())
    agent_edge_list = list(agent.edges.values())
    matched_knowledge_edges = {teid for teid in edge_map.values()}
    edge_recall = (
        len(matched_knowledge_edges) / len(knowledge_edge_list)
        if knowledge_edge_list
        else 1.0
    )
    edge_precision = len(edge_map) / len(agent_edge_list) if agent_edge_list else 0.0
    fabricated_edge_count = len(agent_edge_list) - len(edge_map)

    # ---- start / end --------------------------------------------------------
    start_correct = bool(
        agent.start_node_id is not None
        and mapping.get(agent.start_node_id) == knowledge.graph.start_node_id
    )
    agent_ends = {mapping.get(eid) for eid in agent.end_node_ids if mapping.get(eid)}
    knowledge_ends = set(knowledge.graph.end_node_ids)
    end_recall = (
        len(agent_ends & knowledge_ends) / len(knowledge_ends)
        if knowledge_ends
        else 1.0
    )
    end_precision = (
        len(agent_ends & knowledge_ends) / len(agent_ends) if agent_ends else 0.0
    )

    # ---- concept identity (before property scoring: bindings needed) -------
    concept_recall, concept_precision, agent_to_knowledge = _concept_bindings(
        agent, knowledge, db, annotation_ledger, mapping, edge_map
    )
    # bounded correctness: missing expected concepts and extra/split/wrong
    # claims both reduce it (never vacuous for an empty AgentGraph)
    concept_correctness = concept_recall * concept_precision

    # ---- property correctness ----------------------------------------------
    hits = {p: 0.0 for p in _NODE_PROPS}
    unsupported_concept_ref_count = 0
    for anid, mnid in mapping.items():
        for prop in _NODE_PROPS:
            score, unsup = _property_score(
                agent,
                knowledge,
                db,
                annotation_ledger,
                mnid,
                anid,
                prop,
                agent_to_knowledge,
            )
            hits[prop] += score
            unsupported_concept_ref_count += unsup
    nm = len(mapping) or 1
    activity_correctness = hits["activity"] / nm
    actor_correctness = hits["actor"] / nm
    system_correctness = hits["system"] / nm
    read_correctness = hits["reads"] / nm
    write_correctness = hits["writes"] / nm
    rationale_correctness = hits["rationale"] / nm

    # ---- edge conditions ----------------------------------------------------
    cond_hits = cond_total = 0
    for me in knowledge_edge_list:
        cond_total += 1
        ae = next(
            (edge for edge in agent.edges.values() if edge_map.get(edge.id) == me.id),
            None,
        )
        value = me.condition
        if is_dont_know(value):
            # only an explicit DONT_KNOW condition whose evidence resolves
            # to the EXACT mapped condition slot is correct
            ok = False
            if ae is not None and isinstance(ae.condition, DontKnowType):
                sid = f"edge:{me.id}:condition"
                grounded, _i, _a = _grounded_ids(
                    db, annotation_ledger, ae.condition.evidence
                )
                ok = sid in grounded
            cond_hits += 1 if ok else 0
            continue
        if value is None:
            # known absent: only an explicit ABSENT condition whose evidence
            # resolves to the EXACT mapped condition slot is correct;
            # UNSET / DONT_KNOW / a concept are incorrect
            ok = False
            if ae is not None and isinstance(ae.condition, AbsentType):
                sid = f"edge:{me.id}:condition"
                grounded, _i, _a = _grounded_ids(
                    db, annotation_ledger, ae.condition.evidence
                )
                ok = sid in grounded
            cond_hits += 1 if ok else 0
            continue
        if not isinstance(value, ConceptRef):
            continue  # defensive: only known ConceptRef conditions remain
        if ae is None or not isinstance(ae.condition, ConceptRef):
            continue
        if not ae.condition.asserted:
            continue
        grounded, _i, _a = _grounded_ids(
            db, annotation_ledger, _ref_evidence(ae.condition)
        )
        sid = f"edge:{me.id}:condition"
        bound = agent_to_knowledge.get(ae.condition.concept_id)
        if sid in grounded and bound == value.concept_id:
            cond_hits += 1
    condition_correctness = cond_hits / cond_total if cond_total else 1.0

    # ---- glossary completion + genuine validation ---------------------------
    # completeness of the required concept reconstruction (vs the expected
    # StakeholderKnowledgeConcept set) is SEPARATE from validation
    # correctness of the referenced concepts
    glossary_complete = bool(concept_recall == 1.0 and concept_precision == 1.0)
    glossary_pass, hypothesized, glossary_errors = _glossary_validation(
        agent,
        db,
        alignment_ledger,
        terminology_ledger,
        agent_to_knowledge,
        knowledge,
        annotation_ledger,
    )

    # ---- evidence hygiene ---------------------------------------------------
    authentic_obs_ids: set[str] = set()
    invalid_observation_source_count = 0
    for o in db.observations:
        m = db.messages[o.turn] if 0 <= o.turn < len(db.messages) else None
        if (
            m is not None
            and m.get("role") == "user"
            and (m.get("content") or "") == (o.text or "")
        ):
            authentic_obs_ids.add(o.id)
        else:
            invalid_observation_source_count += 1
    authentic_observation_count = len(authentic_obs_ids)
    orphan_observation_count = sum(
        1 for o in db.observations if o.id not in _all_referenced_observation_ids(agent)
    )
    (
        invalid_evidence_ref_count,
        invalid_observation_reference_count,
        node_evidence_coverage,
        ref_evidence_coverage,
        edge_evidence_coverage,
    ) = _evidence_metrics(db, agent)

    # ambiguous evidence refs: GLOBAL span rule, counted across every agent
    # ref (mapped or not), every ABSENT/DONT_KNOW marker and every edge
    ambiguous_evidence_ref_count = 0
    for anid in agent.nodes:
        for prop in _NODE_PROPS:
            slot = agent.nodes[anid].slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                _g, _i, amb = _grounded_ids(db, annotation_ledger, slot.evidence)
                ambiguous_evidence_ref_count += amb
            for ref in agent.nodes[anid].asserted_refs(prop):
                _g, _i, amb = _grounded_ids(db, annotation_ledger, _ref_evidence(ref))
                ambiguous_evidence_ref_count += amb
    for edge in agent.edges.values():
        _g, _i, amb = _grounded_ids(db, annotation_ledger, edge.evidence)
        ambiguous_evidence_ref_count += amb
        cond = edge.condition
        if isinstance(cond, ConceptRef) and cond.asserted:
            _g, _i, amb2 = _grounded_ids(db, annotation_ledger, _ref_evidence(cond))
            ambiguous_evidence_ref_count += amb2
        elif isinstance(cond, (AbsentType, DontKnowType)):
            _g, _i, amb2 = _grounded_ids(db, annotation_ledger, cond.evidence)
            ambiguous_evidence_ref_count += amb2

    marker_evidence_errors = _marker_evidence_errors(
        agent, knowledge, db, annotation_ledger, mapping, edge_map
    )

    provenance_authenticity_pass = bool(
        invalid_evidence_ref_count == 0
        and invalid_observation_reference_count == 0
        and invalid_observation_source_count == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and ambiguous_evidence_ref_count == 0
        and marker_evidence_errors == 0
        and node_evidence_coverage == 1.0
        and ref_evidence_coverage == 1.0
        and edge_evidence_coverage == 1.0
    )

    # ---- gates -------------------------------------------------------------
    structural_pass = bool(
        graph_created
        and graph_valid
        and node_recall == 1.0
        and node_precision == 1.0
        and edge_recall == 1.0
        and edge_precision == 1.0
        and start_correct
        and end_recall == 1.0
        and end_precision == 1.0
        and activity_correctness == 1.0
        and actor_correctness == 1.0
        and system_correctness == 1.0
        and read_correctness == 1.0
        and write_correctness == 1.0
        and rationale_correctness == 1.0
        and condition_correctness == 1.0
        and concept_recall == 1.0
        and concept_precision == 1.0
        and concept_correctness == 1.0
        and glossary_pass
        and glossary_complete
    )
    protocol_pass = protocol
    quality_pass = bool(structural_pass and evidence_pass)

    return EvaluationResult(
        protocol_completed=protocol,
        graph_created=graph_created,
        graph_valid=graph_valid,
        node_recall=node_recall,
        node_precision=node_precision,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        start_correct=start_correct,
        end_recall=end_recall,
        end_precision=end_precision,
        activity_correctness=activity_correctness,
        actor_correctness=actor_correctness,
        system_correctness=system_correctness,
        read_correctness=read_correctness,
        write_correctness=write_correctness,
        rationale_correctness=rationale_correctness,
        condition_correctness=condition_correctness,
        concept_correctness=concept_correctness,
        concept_recall=concept_recall,
        concept_precision=concept_precision,
        unsupported_ref_count=unsupported_concept_ref_count,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_pass=glossary_pass,
        glossary_complete=glossary_complete,
        referenced_hypothesized_concepts=hypothesized,
        glossary_validation_errors=glossary_errors,
        node_evidence_coverage=node_evidence_coverage,
        ref_evidence_coverage=ref_evidence_coverage,
        edge_evidence_coverage=edge_evidence_coverage,
        invalid_evidence_ref_count=invalid_evidence_ref_count,
        ambiguous_evidence_ref_count=ambiguous_evidence_ref_count,
        marker_evidence_errors=marker_evidence_errors,
        invalid_observation_reference_count=invalid_observation_reference_count,
        authentic_observation_count=authentic_observation_count,
        invalid_observation_source_count=invalid_observation_source_count,
        orphan_observation_count=orphan_observation_count,
        provenance_authenticity_pass=provenance_authenticity_pass,
        evidence_pass=evidence_pass,
        structural_pass=structural_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
        knowledge_coverage=knowledge_coverage,
    )
