"""Evaluator for the graph-native business_interview benchmark (v10).

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
matching, no aliases, no embeddings.

Scoring rules:

- **Node/edge correspondence** falls out of the semantic IDs: an agent
  property ref whose evidence grounds ``node:m:activity`` etc. binds its node
  to stakeholder node ``m`` (deterministic assignment maximizing matches);
  an agent edge whose evidence grounds ``edge:e`` maps to stakeholder edge
  ``e`` when the mapped endpoints match.
- **Property scoring uses property evidence ONLY** (never mentions, never
  validation evidence). For a known slot (``ConceptRef`` value) the agent's
  refs must resolve to the slot/element and their concepts must bind to the
  slot's knowledge concept; for ``None`` (known absent) or ``DONT_KNOW``
  slots, ANY assertion is wrong (epistemic restraint). Asserting an element
  the stakeholder does not know exists is a precision error.
- **Concept identity may use mentions** (plus the mapped property refs):
  all authentic bindings of one Agent concept must agree on one knowledge
  concept of the same kind; per kind the mapping is a bijection over the
  knowledge concepts referenced by the knowledge graph.
- **Validation uses explicit validation/dialogue evidence ONLY**:
  confirmed/partially_confirmed/unknown/disputed statuses and terminology
  agreements must be backed by the corresponding private dialogue events
  addressing the concept's bound knowledge concept.

Truth vs StakeholderKnowledge coverage is reported separately
(``knowledge_coverage``) and never mixed into Agent performance.
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
    AgentGraph,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    is_dont_know,
    spans_correspond,
)
from tau2.domains.business_interview.knowledge import StakeholderKnowledge

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

    v10: the spec carries NO semantic matchers. Everything semantic lives in
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
    unsupported_ref_count: int
    fabricated_node_count: int
    fabricated_edge_count: int

    # glossary completion + genuine validation
    glossary_pass: bool
    referenced_hypothesized_concepts: list[str]
    glossary_validation_errors: list[str]

    # evidence hygiene
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
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
    graph_ids = knowledge.graph.semantic_ids()
    for turn, turn_annotations in (annotations or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for annotation in turn_annotations:
            if annotation.semantic_id not in graph_ids:
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
    concept_ids = set(knowledge.graph.concepts)
    for turn, turn_events in (alignments or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for event in turn_events:
            if event.semantic_id not in concept_ids:
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
            if event.semantic_id not in concept_ids:
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
# Span-based grounding (provenance only)
# ---------------------------------------------------------------------------


def _obs_by_id(db: InterviewDB, obs_id: str):
    for obs in db.observations:
        if obs.id == obs_id:
            return obs
    return None


def _resolve_span_text(text: str, quote: str, occurrence: int):
    if not quote:
        return None
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start == -1:
            return None
    return (start, start + len(quote))


def _covered_ids_for_span(
    annotations: list[SemanticAnnotation],
    text: str,
    ev_span: tuple[int, int],
) -> set[str]:
    """The semantic IDs an evidence span covers, globally (across ALL
    elements).

    Deterministic containment rule:
    - equal-span annotations win: when the evidence span is exactly an
      annotation span, it covers exactly the semantic ids of those
      annotations;
    - otherwise it covers the ids of the maximal annotation spans it
      contains plus the ids of annotation spans strictly containing it.

    A span covering several DISTINCT semantic ids is globally ambiguous and
    must not be reused across slots.
    """
    resolved: list[tuple[SemanticAnnotation, tuple[int, int]]] = []
    for annotation in annotations:
        span = _resolve_span_text(text, annotation.quote, annotation.occurrence)
        if span is not None:
            resolved.append((annotation, span))
    equal = {a.semantic_id for a, s in resolved if s == ev_span}
    if equal:
        return equal
    contained = [
        (a, s) for a, s in resolved if s[0] >= ev_span[0] and s[1] <= ev_span[1]
    ]
    containing = [
        (a, s)
        for a, s in resolved
        if s[0] <= ev_span[0] and s[1] >= ev_span[1] and s != ev_span
    ]
    maximal = [
        (a, s)
        for a, s in contained
        if not any(s2 != s and s2[0] <= s[0] and s[1] <= s2[1] for _, s2 in contained)
    ]
    return {a.semantic_id for a, _ in maximal} | {
        a.semantic_id for a, _ in containing
    }


def _grounded_ids(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    evidence: list[EvidenceRef],
) -> tuple[set[str], int, int]:
    """Semantic IDs grounded by a list of EvidenceRefs via the GLOBAL span
    rule. Returns (grounded ids, invalid count, ambiguous count)."""
    grounded: set[str] = set()
    invalid = 0
    ambiguous = 0
    for ev in evidence:
        obs = _obs_by_id(db, ev.observation_id)
        if obs is None:
            invalid += 1
            continue
        ev_span = ev.resolve_span(obs.text)
        if ev_span is None:
            invalid += 1
            continue
        covered = _covered_ids_for_span(annotations.get(obs.turn, []), obs.text, ev_span)
        if len(covered) > 1:
            ambiguous += 1
            continue
        if len(covered) == 1:
            grounded.update(covered)
    return grounded, invalid, ambiguous


def _ref_evidence(ref: ConceptRef) -> list[EvidenceRef]:
    """The property evidence of a ref — property evidence ONLY (mentions and
    validation evidence never enter property scoring)."""
    return list(ref.evidence)


# ---------------------------------------------------------------------------
# Node / edge correspondence (provenance + semantic IDs)
# ---------------------------------------------------------------------------


def _candidate_node_ids(grounded: set[str]) -> set[str]:
    """Stakeholder node ids appearing in grounded ``node:*`` semantic ids."""
    out: set[str] = set()
    for sid in grounded:
        if sid.startswith("node:"):
            parts = sid.split(":")
            if len(parts) >= 2:
                out.add(parts[1])
    return out


def _candidate_edge_ids(grounded: set[str]) -> set[str]:
    """Stakeholder edge ids appearing in grounded ``edge:*`` semantic ids."""
    out: set[str] = set()
    for sid in grounded:
        if sid.startswith("edge:"):
            parts = sid.split(":")
            if len(parts) >= 2:
                out.add(parts[1])
    return out


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
    kn_ids = knowledge.graph.semantic_ids()
    tn: dict[str, set[str]] = {}
    for nid, node in agent.nodes.items():
        cand: set[str] = set()
        for prop in _NODE_PROPS:
            for ref in node.asserted_refs(prop):
                grounded, _i, _a = _grounded_ids(
                    db, annotations, _ref_evidence(ref)
                )
                cand.update(_candidate_node_ids(grounded))
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
        candidates = sorted(_candidate_edge_ids(grounded))
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


def _slot_value_concepts(knowledge: StakeholderKnowledge, mnid: str, prop: str) -> set[str]:
    """The knowledge concept ids a correct ref must bind to for this slot."""
    value = _slot_value(knowledge, mnid, prop)
    if isinstance(value, list):
        return {r.concept_id for r in value}
    if isinstance(value, ConceptRef):
        return {value.concept_id}
    return set()


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
    """Score one node property (activity/actor/system/reads/writes/rationale).

    Uses property evidence ONLY. DONT_KNOW / known-absent slots: any
    assertion is wrong (epistemic restraint). Known slots: recall over the
    slot's knowledge concepts (each covered by a ref resolving to the
    element and binding that concept) times precision over the agent refs
    (each must resolve to the slot and bind one of its concepts). Returns
    (score, unsupported_ref_count).
    """
    refs = agent.nodes[anid].asserted_refs(prop)
    value = _slot_value(knowledge, mnid, prop)
    if is_dont_know(value) or value is None:
        return (1.0 if not refs else 0.0), 0
    expected = _expected_ids(knowledge, mnid, prop)
    concepts = _slot_value_concepts(knowledge, mnid, prop)
    if not refs:
        return 0.0, 0
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


def _knowledge_value_concepts(
    knowledge: StakeholderKnowledge, sid: str
) -> set[str]:
    """The knowledge concept(s) a grounded element id stands for.

    Element ids (``node:<nid>:<axis>:<kcid>``) pin exactly one concept;
    slot ids resolve to the whole property's concepts; the condition slot
    resolves to the condition concept."""
    if sid.startswith("node:"):
        parts = sid.split(":")
        if len(parts) == 4 and parts[2] in ("reads", "writes"):
            return {parts[3]}
        node = knowledge.graph._parse_node_slot(sid)
        if node is not None:
            mnid, prop = node
            return _slot_value_concepts(knowledge, mnid, prop)
    if sid.startswith("edge:"):
        parts = sid.split(":")
        if len(parts) == 3 and parts[2] == "condition":
            edge = knowledge.graph.edges.get(parts[1])
            if edge is not None and isinstance(edge.condition, ConceptRef):
                return {edge.condition.concept_id}
    return set()


def _concept_bindings(
    agent: AgentGraph,
    knowledge: StakeholderKnowledge,
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    mapping: dict[str, str],
    edge_map: dict[str, str],
) -> tuple[float, dict[str, str]]:
    """Concept-level binding integrity across all matched visible slots.

    For every ConceptKind:
    - one Agent concept may bind to exactly ONE knowledge concept of its own
      kind (bindings from mapped property refs AND mentions must agree);
    - one knowledge concept referenced by the knowledge graph must be
      represented by exactly one Agent concept (splits fail; missing
      concepts fail).

    Returns (concept_correctness, agent concept id -> knowledge concept id).
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
                grounded, _i, _a = _grounded_ids(
                    db, annotations, _ref_evidence(ref)
                )
                expected = _expected_ids(knowledge, mnid, prop)
                for sid in grounded & expected:
                    add_binding(ref.concept_id, _knowledge_value_concepts(knowledge, sid))
    for eid, meid in edge_map.items():
        edge = agent.edges[eid]
        if edge.condition is not None and edge.condition.asserted:
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
    # mentions may participate in concept identity (never in property scoring)
    for cid, concept in agent.concepts.items():
        for mention in concept.mentions:
            grounded, _i, _a = _grounded_ids(db, annotations, [mention])
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

    agent_to_knowledge: dict[str, str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts.get(cid)
        if concept is None:
            return 0.0, {}
        cand = candidates.get(cid, set())
        if len(cand) != 1:
            return 0.0, {}
        kid = next(iter(cand))
        if knowledge_kinds.get(kid) != concept.kind:
            return 0.0, {}
        agent_to_knowledge[cid] = kid

    by_kind: dict[str, list[str]] = {}
    for cid, kid in agent_to_knowledge.items():
        by_kind.setdefault(agent.concepts[cid].kind, []).append(cid)
    for kind, agents in by_kind.items():
        covered = {agent_to_knowledge[c] for c in agents}
        if len(covered) != len(agents):
            return 0.0, {}
        expected_kind = {k for k, kk in knowledge_kinds.items() if kk == kind}
        if covered != expected_kind:
            return 0.0, {}
    return 1.0, agent_to_knowledge


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


def _glossary_validation(
    agent: AgentGraph,
    db: InterviewDB,
    alignments: dict[int, list[ConceptAlignmentAssertion]],
    terminology: dict[int, list[TerminologyConfirmation]],
    agent_to_knowledge: dict[str, str],
) -> tuple[bool, list[str], list[str]]:
    """Completion + genuine-validation rules.

    - every referenced concept must be resolved (not hypothesized);
    - ``grounded``: the concept must have an authentic binding (it appears
      in ``agent_to_knowledge``);
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
            if bound is None:
                errors.append(
                    f"concept {cid}: grounded requires an authentic provenance "
                    f"binding (concept is not bound to any knowledge concept)"
                )
            continue
        if concept.validation_status in ("confirmed", "partially_confirmed", "unknown", "disputed"):
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
                if acts and bound is not None and not _events_corresponding_at_span(
                    events_at_turn, obs.text, ev_span, acts, bound
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
        for prop in _NODE_PROPS:
            node_refs.extend(node.refs(prop))
        node_total += 1
        if any(
            r.asserted and any(span_ok(ev) for ev in _ref_evidence(r))
            for r in node_refs
        ):
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
        if edge.condition is not None:
            check_evidence(_ref_evidence(edge.condition))
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
    for edge in agent.edges.values():
        for ev in edge.evidence:
            ids.add(ev.observation_id)
        if edge.condition is not None:
            for ev in _ref_evidence(edge.condition):
                ids.add(ev.observation_id)
    return ids


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
    knowledge_coverage = 0.0
    if truth is not None:
        from tau2.domains.business_interview.graph import graph_semantic_ids

        truth_ids = graph_semantic_ids(truth.nodes, truth.edges)
        if truth_ids:
            known = 0
            for sid in truth_ids:
                value = knowledge.graph.element_value(sid)
                if value is not None and not is_dont_know(value):
                    known += 1
            knowledge_coverage = known / len(truth_ids)

    # ---- node/edge correspondence ------------------------------------------
    mapping, edge_map = _match_nodes_and_edges(agent, knowledge, db, annotation_ledger)
    knowledge_node_ids = list(knowledge.graph.nodes)
    agent_node_ids = list(agent.nodes)
    matched_knowledge = set(mapping.values())
    matched_agent = set(mapping.keys())
    node_recall = (
        len(matched_knowledge) / len(knowledge_node_ids)
        if knowledge_node_ids
        else 0.0
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
    concept_correctness, agent_to_knowledge = _concept_bindings(
        agent, knowledge, db, annotation_ledger, mapping, edge_map
    )

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
        if is_dont_know(value) or value is None:
            cond_hits += 1 if (ae is None or ae.condition is None or not ae.condition.asserted) else 0
            continue
        if ae is None or ae.condition is None or not ae.condition.asserted:
            continue
        grounded, _i, _a = _grounded_ids(
            db, annotation_ledger, _ref_evidence(ae.condition)
        )
        sid = f"edge:{me.id}:condition"
        bound = agent_to_knowledge.get(ae.condition.concept_id)
        if (
            isinstance(value, ConceptRef)
            and sid in grounded
            and bound == value.concept_id
        ):
            cond_hits += 1
    condition_correctness = cond_hits / cond_total if cond_total else 1.0

    # ---- glossary completion + genuine validation ---------------------------
    glossary_pass, hypothesized, glossary_errors = _glossary_validation(
        agent, db, alignment_ledger, terminology_ledger, agent_to_knowledge
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
    # ref (mapped or not) and every edge
    ambiguous_evidence_ref_count = 0
    for anid in agent.nodes:
        for prop in _NODE_PROPS:
            for ref in agent.nodes[anid].asserted_refs(prop):
                _g, _i, amb = _grounded_ids(
                    db, annotation_ledger, _ref_evidence(ref)
                )
                ambiguous_evidence_ref_count += amb
    for edge in agent.edges.values():
        _g, _i, amb = _grounded_ids(db, annotation_ledger, edge.evidence)
        ambiguous_evidence_ref_count += amb
        if edge.condition is not None and edge.condition.asserted:
            _g, _i, amb2 = _grounded_ids(
                db, annotation_ledger, _ref_evidence(edge.condition)
            )
            ambiguous_evidence_ref_count += amb2

    provenance_authenticity_pass = bool(
        invalid_evidence_ref_count == 0
        and invalid_observation_reference_count == 0
        and invalid_observation_source_count == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and ambiguous_evidence_ref_count == 0
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
        and concept_correctness == 1.0
        and glossary_pass
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
        unsupported_ref_count=unsupported_concept_ref_count,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_pass=glossary_pass,
        referenced_hypothesized_concepts=hypothesized,
        glossary_validation_errors=glossary_errors,
        node_evidence_coverage=node_evidence_coverage,
        ref_evidence_coverage=ref_evidence_coverage,
        edge_evidence_coverage=edge_evidence_coverage,
        invalid_evidence_ref_count=invalid_evidence_ref_count,
        ambiguous_evidence_ref_count=ambiguous_evidence_ref_count,
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
