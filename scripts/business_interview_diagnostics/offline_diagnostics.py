"""Deterministic offline analysis for business-interview artifacts.

This module consumes evaluator-private artifacts only.  It never calls an LLM
and never runs during an interview.  The root-cause classifier is intentionally
conservative: an attribution is emitted only when the stored knowledge,
accepted Observation sidecar, final AgentGraph, and (where needed) a clear
Agent question provide deterministic evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Optional, cast

from tau2.domains.business_interview.evaluation_diagnostics import (
    ConceptDiagnostics,
    FailureAttribution,
    SlotDiagnostic,
)
from tau2.domains.business_interview.graph import (
    UNSET,
    AbsentType,
    AgentConcept,
    AgentGraph,
    AgentListSlot,
    AgentSlot,
    ConceptKind,
    ConceptRef,
    DontKnowType,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    TerminologyAgreement,
    is_dont_know,
)


class ArtifactDecodeError(ValueError):
    """Raised when a persisted smoke-artifact value is ambiguous or invalid."""


_MARKER_NAMES = frozenset({"unset", "absent", "dont_know"})


def _decode_mapping(value, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ArtifactDecodeError(
            f"{path}: expected an object, got {type(value).__name__}"
        )
    return value


def _required_string(value: Mapping, name: str, path: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ArtifactDecodeError(f"{path}.{name}: expected a non-empty string")
    return item


def _decode_evidence(value, path: str) -> list[EvidenceRef]:
    if not isinstance(value, list):
        raise ArtifactDecodeError(f"{path}: expected an evidence list")
    decoded: list[EvidenceRef] = []
    for index, raw in enumerate(value):
        item = _decode_mapping(raw, f"{path}[{index}]")
        unknown = set(item) - {"observation_id", "quote", "occurrence"}
        if unknown:
            raise ArtifactDecodeError(
                f"{path}[{index}]: unknown evidence fields {sorted(unknown)!r}"
            )
        try:
            decoded.append(EvidenceRef.model_validate(item))
        except ValueError as exc:
            raise ArtifactDecodeError(
                f"{path}[{index}]: invalid evidence: {exc}"
            ) from exc
    return decoded


def _decode_concept_ref(value: Mapping, path: str) -> ConceptRef:
    unknown = set(value) - {"concept_id", "confidence", "evidence"}
    if unknown:
        raise ArtifactDecodeError(
            f"{path}: unknown ConceptRef fields {sorted(unknown)!r}"
        )
    concept_id = _required_string(value, "concept_id", path)
    evidence = (
        _decode_evidence(value["evidence"], f"{path}.evidence")
        if "evidence" in value
        else []
    )
    try:
        return ConceptRef(
            concept_id=concept_id,
            confidence=value.get("confidence", 1.0),
            evidence=evidence,
        )
    except ValueError as exc:
        raise ArtifactDecodeError(f"{path}: invalid ConceptRef: {exc}") from exc


def _decode_single_agent_slot(value, path: str) -> AgentSlot:
    item = _decode_mapping(value, path)
    marker_names = _MARKER_NAMES.intersection(item)
    if marker_names:
        if len(marker_names) != 1:
            raise ArtifactDecodeError(
                f"{path}: multiple epistemic markers {sorted(marker_names)!r}"
            )
        marker = next(iter(marker_names))
        if not isinstance(item[marker], bool) or not item[marker]:
            raise ArtifactDecodeError(f"{path}.{marker}: marker must be true")
        if marker == "unset":
            if set(item) != {"unset"}:
                raise ArtifactDecodeError(
                    f"{path}: UNSET marker cannot contain additional fields"
                )
            return UNSET
        unknown = set(item) - {marker, "evidence"}
        if unknown or "evidence" not in item:
            if unknown:
                detail = f"unknown fields {sorted(unknown)!r}"
            else:
                detail = "missing evidence list"
            raise ArtifactDecodeError(f"{path}: malformed {marker} marker ({detail})")
        evidence = _decode_evidence(item["evidence"], f"{path}.evidence")
        return (
            AbsentType(evidence=evidence)
            if marker == "absent"
            else DontKnowType(evidence=evidence)
        )
    if "concept_id" in item:
        return _decode_concept_ref(item, path)
    raise ArtifactDecodeError(
        f"{path}: expected exactly one supported marker or concept_id"
    )


def decode_agent_slot(
    value, *, path: str = "slot", list_slot: bool = False
) -> AgentSlot | AgentListSlot:
    """Decode one slot from the smoke script's explicit JSON representation.

    The runtime AgentSlot union is intentionally not used here: its marker
    models have no discriminator fields, so generic Pydantic validation can
    turn ``{"dont_know": true, ...}`` into ``AbsentType``.  A list slot may
    contain only ConceptRefs; whole-property markers are decoded explicitly.
    """
    if list_slot and isinstance(value, list):
        decoded: list[ConceptRef] = []
        for index, item in enumerate(value):
            ref = _decode_single_agent_slot(item, f"{path}[{index}]")
            if not isinstance(ref, ConceptRef):
                raise ArtifactDecodeError(
                    f"{path}[{index}]: list items must be ConceptRef objects"
                )
            decoded.append(ref)
        return decoded
    if list_slot and not isinstance(value, list):
        return _decode_single_agent_slot(value, path)
    if isinstance(value, list):
        raise ArtifactDecodeError(f"{path}: scalar slot cannot be a list")
    return _decode_single_agent_slot(value, path)


def _decode_agent_concept(value, path: str) -> AgentConcept:
    item = _decode_mapping(value, path)
    mentions = _decode_evidence(item.get("mentions", []), f"{path}.mentions")
    kind = item.get("kind")
    description = item.get("description", "")
    if not isinstance(kind, str):
        raise ArtifactDecodeError(f"{path}.kind: expected a string")
    if not isinstance(description, str):
        raise ArtifactDecodeError(f"{path}.description: expected a string")
    try:
        return AgentConcept(
            id=_required_string(item, "id", path),
            kind=cast(ConceptKind, kind),
            display_label=_required_string(item, "display_label", path),
            description=description,
            mentions=mentions,
        )
    except ValueError as exc:
        raise ArtifactDecodeError(f"{path}: invalid AgentConcept: {exc}") from exc


def _decode_agent_node(value, path: str) -> Node:
    item = _decode_mapping(value, path)
    required = ("activity", "actor", "system", "reads", "writes", "necessity_rationale")
    missing = [name for name in required if name not in item]
    if missing:
        raise ArtifactDecodeError(f"{path}: missing slot fields {missing!r}")
    try:
        return Node(
            id=_required_string(item, "id", path),
            activity=cast(
                AgentSlot,
                decode_agent_slot(item["activity"], path=f"{path}.activity"),
            ),
            actor=cast(
                AgentSlot,
                decode_agent_slot(item["actor"], path=f"{path}.actor"),
            ),
            system=cast(
                AgentSlot,
                decode_agent_slot(item["system"], path=f"{path}.system"),
            ),
            reads=cast(
                AgentListSlot,
                decode_agent_slot(item["reads"], path=f"{path}.reads", list_slot=True),
            ),
            writes=cast(
                AgentListSlot,
                decode_agent_slot(
                    item["writes"], path=f"{path}.writes", list_slot=True
                ),
            ),
            necessity_rationale=cast(
                AgentSlot,
                decode_agent_slot(
                    item["necessity_rationale"],
                    path=f"{path}.necessity_rationale",
                ),
            ),
        )
    except (ArtifactDecodeError, ValueError) as exc:
        if isinstance(exc, ArtifactDecodeError):
            raise
        raise ArtifactDecodeError(f"{path}: invalid Agent node: {exc}") from exc


def _decode_agent_edge(value, path: str) -> Edge:
    item = _decode_mapping(value, path)
    try:
        return Edge(
            id=_required_string(item, "id", path),
            from_node=_required_string(item, "from_node", path),
            to_node=_required_string(item, "to_node", path),
            condition=cast(
                AgentSlot,
                decode_agent_slot(item["condition"], path=f"{path}.condition"),
            ),
            evidence=_decode_evidence(item.get("evidence", []), f"{path}.evidence"),
        )
    except KeyError as exc:
        raise ArtifactDecodeError(
            f"{path}: missing edge field {exc.args[0]!r}"
        ) from exc
    except (ArtifactDecodeError, ValueError) as exc:
        if isinstance(exc, ArtifactDecodeError):
            raise
        raise ArtifactDecodeError(f"{path}: invalid Agent edge: {exc}") from exc


def _decode_terminology(value, path: str) -> list[TerminologyAgreement]:
    if not isinstance(value, list):
        raise ArtifactDecodeError(f"{path}: expected a terminology-agreement list")
    decoded: list[TerminologyAgreement] = []
    for index, raw in enumerate(value):
        item = _decode_mapping(raw, f"{path}[{index}]")
        term = item.get("term")
        stakeholder_id = item.get("stakeholder_id", "stakeholder")
        if not isinstance(term, str) or not term:
            raise ArtifactDecodeError(
                f"{path}[{index}].term: expected a non-empty string"
            )
        if not isinstance(stakeholder_id, str) or not stakeholder_id:
            raise ArtifactDecodeError(
                f"{path}[{index}].stakeholder_id: expected a non-empty string"
            )
        try:
            decoded.append(
                TerminologyAgreement(
                    concept_id=_required_string(item, "concept_id", f"{path}[{index}]"),
                    term=term,
                    stakeholder_id=stakeholder_id,
                    evidence=_decode_evidence(
                        item.get("evidence", []), f"{path}[{index}].evidence"
                    ),
                )
            )
        except (ArtifactDecodeError, ValueError) as exc:
            if isinstance(exc, ArtifactDecodeError):
                raise
            raise ArtifactDecodeError(
                f"{path}[{index}]: invalid terminology agreement: {exc}"
            ) from exc
    return decoded


def decode_agent_graph(value) -> AgentGraph:
    """Restore an AgentGraph from a smoke artifact without union guessing.

    The smoke serializer is a compatibility format, not a general Pydantic
    serialization contract.  This decoder deliberately rejects ambiguous
    markers and constructs every Agent slot before building the graph.  A
    future discriminated AgentSlot schema would make generic round-tripping
    safer, but changing that runtime schema is outside this offline fix.
    """
    item = _decode_mapping(value, "final_graph")
    concepts_raw = _decode_mapping(item.get("concepts"), "final_graph.concepts")
    nodes_raw = _decode_mapping(item.get("nodes"), "final_graph.nodes")
    edges_raw = _decode_mapping(item.get("edges"), "final_graph.edges")
    concepts: dict[str, AgentConcept] = {}
    for concept_id, raw in concepts_raw.items():
        if not isinstance(concept_id, str):
            raise ArtifactDecodeError("final_graph.concepts: ids must be strings")
        concept = _decode_agent_concept(raw, f"final_graph.concepts[{concept_id!r}]")
        if concept.id != concept_id:
            raise ArtifactDecodeError(
                f"final_graph.concepts[{concept_id!r}].id does not match its map key"
            )
        concepts[concept_id] = concept
    nodes: dict[str, Node] = {}
    for node_id, raw in nodes_raw.items():
        if not isinstance(node_id, str):
            raise ArtifactDecodeError("final_graph.nodes: ids must be strings")
        node = _decode_agent_node(raw, f"final_graph.nodes[{node_id!r}]")
        if node.id != node_id:
            raise ArtifactDecodeError(
                f"final_graph.nodes[{node_id!r}].id does not match its map key"
            )
        nodes[node_id] = node
    edges: dict[str, Edge] = {}
    for edge_id, raw in edges_raw.items():
        if not isinstance(edge_id, str):
            raise ArtifactDecodeError("final_graph.edges: ids must be strings")
        edge = _decode_agent_edge(raw, f"final_graph.edges[{edge_id!r}]")
        if edge.id != edge_id:
            raise ArtifactDecodeError(
                f"final_graph.edges[{edge_id!r}].id does not match its map key"
            )
        edges[edge_id] = edge
    end_node_ids = item.get("end_node_ids", [])
    if not isinstance(end_node_ids, list) or not all(
        isinstance(node_id, str) for node_id in end_node_ids
    ):
        raise ArtifactDecodeError("final_graph.end_node_ids: expected a string list")
    start_node_id = item.get("start_node_id")
    if start_node_id is not None and not isinstance(start_node_id, str):
        raise ArtifactDecodeError("final_graph.start_node_id: expected string or null")
    try:
        return AgentGraph(
            id=_required_string(item, "id", "final_graph"),
            name=item.get("name", ""),
            nodes=nodes,
            edges=edges,
            concepts=concepts,
            start_node_id=start_node_id,
            end_node_ids=end_node_ids,
            terminology_agreements=_decode_terminology(
                item.get("terminology_agreements", []),
                "final_graph.terminology_agreements",
            ),
        )
    except (ArtifactDecodeError, ValueError) as exc:
        if isinstance(exc, ArtifactDecodeError):
            raise
        raise ArtifactDecodeError(f"final_graph: invalid graph: {exc}") from exc


_WORD_RE = re.compile(r"[a-z0-9]+")


def _field(value, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _truth_semantic_id(semantic_id: str, knowledge_graph) -> Optional[str]:
    """Translate one private stakeholder-local semantic id to Truth form."""
    if not semantic_id:
        return None
    if semantic_id in knowledge_graph.concepts:
        return knowledge_graph.concepts[semantic_id].truth_concept_id
    parts = semantic_id.split(":")
    if parts[0] == "node":
        if len(parts) < 2:
            return None
        truth_node = knowledge_graph.node_truth_ids.get(parts[1])
        if truth_node is None:
            return None
        out = ["node", truth_node, *parts[2:]]
        if len(parts) == 4:
            local_concept = knowledge_graph.concepts.get(parts[3])
            if local_concept is None:
                return None
            out[3] = local_concept.truth_concept_id
        return ":".join(out)
    if parts[0] == "edge" and len(parts) >= 2:
        truth_edge = knowledge_graph.edge_truth_ids.get(parts[1])
        if truth_edge is None:
            return None
        return ":".join(["edge", truth_edge, *parts[2:]])
    return None


def _observation_by_turn(db: InterviewDB) -> dict[int, list]:
    out: dict[int, list] = {}
    for observation in db.observations:
        message = (
            db.messages[observation.turn]
            if 0 <= observation.turn < len(db.messages)
            else None
        )
        if (
            message is None
            or message.get("role") != "user"
            or (message.get("content") or "") != observation.text
        ):
            continue
        out.setdefault(observation.turn, []).append(observation)
    return out


def accepted_truth_annotations(
    db: InterviewDB,
    knowledge,
    annotations: Mapping[int | str, Iterable],
) -> list[dict]:
    """Return valid sidecar annotations translated to Truth semantic ids.

    The environment creates one immutable Observation for each accepted user
    message.  An annotation is considered public here only when its turn has
    such an Observation and its stored quote/occurrence resolves in that
    Observation.  Invalid or orphaned sidecar records are ignored.
    """
    knowledge_graph = knowledge.graph
    observations = _observation_by_turn(db)
    accepted: list[dict] = []
    for raw_turn, records in annotations.items():
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError):
            continue
        turn_observations = observations.get(turn, [])
        if not turn_observations:
            continue
        for record in records or ():
            local_id = _field(record, "semantic_id", "")
            truth_id = _truth_semantic_id(local_id, knowledge_graph)
            if truth_id is None:
                continue
            quote = _field(record, "quote", "") or ""
            occurrence = _field(record, "occurrence", 0) or 0
            if not quote:
                continue
            matched_observation = None
            for observation in turn_observations:
                if observation.has_span(quote, occurrence):
                    matched_observation = observation
                    break
            if matched_observation is None:
                continue
            accepted.append(
                {
                    "truth_semantic_id": truth_id,
                    "mode": _field(record, "mode"),
                    "turn": turn,
                    "observation_id": matched_observation.id,
                }
            )
    return sorted(
        accepted,
        key=lambda item: (
            item["truth_semantic_id"],
            item["turn"],
            item["observation_id"],
            item["mode"] or "",
        ),
    )


def _node_slot_value(node, prop: str):
    attr = "necessity_rationale" if prop == "necessity_rationale" else prop
    return getattr(node, attr, None)


def _knowledge_state(knowledge, truth, target_ids: list[str]) -> str:
    """Classify attainability of the exact Truth facts named by target ids."""
    if knowledge is None:
        return "unknown"
    kg = knowledge.graph
    statuses: list[str] = []
    for target_id in target_ids:
        parts = target_id.split(":")
        if parts[0] == "node" and len(parts) >= 3:
            local_node = next(
                (local for local, tid in kg.node_truth_ids.items() if tid == parts[1]),
                None,
            )
            node = kg.nodes.get(local_node) if local_node is not None else None
            if node is None:
                statuses.append("hidden_element")
                continue
            prop = parts[2]
            value = _node_slot_value(node, prop)
            if len(parts) == 4:
                local_concept = next(
                    (
                        local
                        for local, concept in kg.concepts.items()
                        if concept.truth_concept_id == parts[3]
                    ),
                    None,
                )
                if is_dont_know(value):
                    statuses.append("dont_know")
                elif isinstance(value, list) and any(
                    ref.concept_id == local_concept for ref in value
                ):
                    statuses.append("known_value")
                else:
                    statuses.append("known_absence")
            elif is_dont_know(value):
                statuses.append("dont_know")
            else:
                statuses.append("known_value")
        elif parts[0] == "edge" and len(parts) == 3:
            local_edge = next(
                (local for local, eid in kg.edge_truth_ids.items() if eid == parts[1]),
                None,
            )
            edge = kg.edges.get(local_edge) if local_edge is not None else None
            if edge is None:
                statuses.append("hidden_element")
            elif is_dont_know(edge.condition):
                statuses.append("dont_know")
            else:
                statuses.append("known_value")
        else:
            statuses.append("unknown")
    if not statuses:
        return "unknown"
    if "hidden_element" in statuses:
        return "hidden_element"
    if "dont_know" in statuses:
        return "dont_know"
    if all(status.startswith("known_") for status in statuses):
        return "known"
    return "unknown"


def _assistant_texts(db: InterviewDB) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in db.messages
        if message.get("role") == "assistant" and message.get("content")
    ]


def _question_asked(
    db: InterviewDB,
    property_name: str,
    truth_concept_labels: Iterable[str],
    context_labels: Iterable[str],
) -> Optional[bool]:
    """Detect a clear, deterministic question relevant to one property.

    The activity/endpoint context is required as well as a slot cue.  This
    prevents a question about one workflow step (for example, a month-end
    summary) from being credited to every node that shares a generic word such
    as ``quotation``.
    """
    cues = {
        "activity": {"what", "happen", "step", "trigger", "process"},
        "actor": {"who", "role", "actor", "responsible"},
        "system": {"system", "tool", "software", "crm", "excel"},
        "reads": {"read", "use", "draw", "input", "data", "source"},
        "writes": {"write", "produce", "output", "record", "send", "create"},
        "necessity_rationale": {"why", "necessary", "purpose", "reason"},
        "condition": {"condition", "when", "if", "amount", "branch", "case"},
    }.get(property_name, set())
    label_tokens = set().union(*(_tokens(label) for label in truth_concept_labels))
    context_tokens = set().union(*(_tokens(label) for label in context_labels))
    context_heads = {
        words[0]
        for label in context_labels
        if (words := _WORD_RE.findall((label or "").lower()))
    }
    ambiguous = False
    for text in _assistant_texts(db):
        lower = text.lower()
        if "?" not in lower:
            continue
        words = _tokens(lower)
        context_overlap = words & context_tokens
        if context_tokens and not context_overlap:
            continue
        # Two generic context words (for example ``quotation document``) are
        # not enough when the question never names the target activity head.
        if len(context_overlap) >= 2 and not (context_overlap & context_heads):
            ambiguous = True
            continue
        if words & cues:
            return True
        # A concept-specific question is enough even when the wording does
        # not use the canonical slot cue (e.g. "Does it draw on pricing?").
        if len(words & label_tokens) >= 1 and property_name in {
            "system",
            "reads",
            "writes",
            "condition",
        }:
            return True
    return None if ambiguous else False


def _target_ids_for_slot(
    target_id: str,
    property_name: str,
    slot: SlotDiagnostic,
) -> list[str]:
    base = f"node:{target_id}:{property_name}"
    if target_id.startswith("edge:"):
        base = target_id
    if property_name in ("reads", "writes") and slot.missing_truth_concept_ids:
        return [f"{base}:{concept_id}" for concept_id in slot.missing_truth_concept_ids]
    return [base]


def _context_labels(truth, target_id: str) -> list[str]:
    if target_id.startswith("edge:"):
        edge_id = target_id.split(":")[1]
        edge = truth.edges.get(edge_id)
        if edge is None:
            return []
        node_ids = [edge.from_node, edge.to_node]
    else:
        node_ids = [target_id]
    labels: list[str] = []
    for node_id in node_ids:
        node = truth.nodes.get(node_id)
        if node is None or not isinstance(node.activity, ConceptRef):
            continue
        concept = truth.concepts.get(node.activity.concept_id)
        labels.extend(concept.canonical_terms if concept is not None else [])
    return labels


def _accepted_for_targets(accepted: list[dict], target_ids: list[str]) -> list[dict]:
    target_set = set(target_ids)
    return [item for item in accepted if item["truth_semantic_id"] in target_set]


def _matcher_plausible(
    slot: SlotDiagnostic,
    target_ids: list[str],
    accepted: list[dict],
    concept_trace: Optional[ConceptDiagnostics],
    agent_evidence_observation_ids: set[str],
) -> bool:
    """Only call a miss matcher-caused when public evidence ties the ref to it."""
    if concept_trace is None or not slot.agent_concept_ids:
        return False
    disclosed_observations = {
        item["observation_id"] for item in _accepted_for_targets(accepted, target_ids)
    }
    if not disclosed_observations or not (
        disclosed_observations & agent_evidence_observation_ids
    ):
        return False
    expected_concepts = set(slot.truth_concept_ids)
    expected_concepts.update(
        target_id.rsplit(":", 1)[-1]
        for target_id in target_ids
        if target_id.count(":") == 4
    )
    for pair in concept_trace.candidate_pairs:
        if pair.agent_concept_id not in slot.agent_concept_ids:
            continue
        if pair.truth_concept_id not in expected_concepts:
            continue
        if pair.selected_mapping:
            continue
        # Candidate trace proves that the deterministic matcher considered
        # the pair but did not select it.  The accepted semantic annotation
        # proves the hidden value was the one being discussed.  We do not
        # infer semantic equivalence from text alone.
        if pair.lexical_similarity_score < pair.threshold:
            return True
    return False


def classify_failed_slot(
    *,
    target_id: str,
    property_name: str,
    slot: SlotDiagnostic,
    truth,
    knowledge,
    db: InterviewDB,
    annotations: Mapping[int | str, Iterable],
    concept_trace: Optional[ConceptDiagnostics] = None,
    agent_evidence_observation_ids: Optional[set[str]] = None,
) -> FailureAttribution:
    """Conservatively attribute one failed slot using stored evidence only."""
    accepted = accepted_truth_annotations(db, knowledge, annotations)
    target_ids = _target_ids_for_slot(target_id, property_name, slot)
    disclosed = _accepted_for_targets(accepted, target_ids)
    knowledge_state = _knowledge_state(knowledge, truth, target_ids)
    evidence = [f"score_reason:{slot.reason}", f"knowledge:{knowledge_state}"]
    if slot.missing_truth_concept_ids:
        evidence.append(
            "missing_truth_concepts:" + ",".join(slot.missing_truth_concept_ids)
        )
    if disclosed:
        evidence.append(
            "accepted_observations:"
            + ",".join(sorted({item["observation_id"] for item in disclosed}))
        )

    if slot.reason in {"unmatched_node", "unmatched_edge"}:
        category = "insufficient_evidence_to_classify"
        reason = "structural target was not aligned"
    elif knowledge_state in {"hidden_element", "dont_know", "unknown"}:
        category = "insufficient_evidence_to_classify"
        reason = "stored stakeholder knowledge cannot establish attainability"
    elif disclosed:
        if _matcher_plausible(
            slot,
            target_ids,
            accepted,
            concept_trace,
            agent_evidence_observation_ids or set(),
        ):
            category = "evaluator_matching"
            reason = "accepted semantic evidence exists but deterministic concept matching did not select the Agent concept"
        else:
            category = "agent_recording"
            reason = "accepted public semantic evidence exists but final Agent state does not score it"
    else:
        labels = slot.truth_labels
        asked = _question_asked(
            db,
            property_name,
            labels,
            _context_labels(truth, target_id),
        )
        evidence.append(f"clear_agent_question:{asked}")
        if asked:
            category = "stakeholder_disclosure"
            reason = "knowledge was available, but no accepted public annotation disclosed the needed value"
        elif asked is None:
            category = "insufficient_evidence_to_classify"
            reason = "stored questions were contextually ambiguous"
        else:
            category = "agent_elicitation"
            reason = "knowledge was available, but no clear relevant Agent question was stored"
    return FailureAttribution(
        target_id=target_id,
        dimension="edge_condition" if target_id.startswith("edge:") else "node_slot",
        property=property_name,
        category=category,
        reason=reason,
        evidence=evidence,
    )


def classify_failed_slots(
    result,
    *,
    truth,
    knowledge,
    db: InterviewDB,
    annotations: Mapping[int | str, Iterable],
) -> list[FailureAttribution]:
    """Classify every failed node slot and edge condition in an evaluation."""
    output: list[FailureAttribution] = []
    trace = getattr(getattr(result, "diagnostics", None), "concepts", None)
    for node in result.diagnostics.node_diagnostics:
        agent_node = (
            db.graph.nodes.get(node.agent_node_id)
            if db.graph is not None and node.agent_node_id is not None
            else None
        )
        for property_name, slot in node.slots.items():
            evidence_ids = set()
            if agent_node is not None:
                value = agent_node.slot_value(
                    "rationale"
                    if property_name == "necessity_rationale"
                    else property_name
                )
                refs = value if isinstance(value, list) else [value]
                evidence_ids = {
                    evidence.observation_id
                    for ref in refs
                    if isinstance(ref, ConceptRef)
                    for evidence in (ref.evidence or [])
                }
            if slot.matched:
                continue
            output.append(
                classify_failed_slot(
                    target_id=node.truth_node_id,
                    property_name=property_name,
                    slot=slot,
                    truth=truth,
                    knowledge=knowledge,
                    db=db,
                    annotations=annotations,
                    concept_trace=trace,
                    agent_evidence_observation_ids=evidence_ids,
                )
            )
    for edge in result.diagnostics.edge_diagnostics:
        if edge.condition.matched:
            continue
        edge_agent = (
            db.graph.edges.get(edge.agent_edge_id)
            if db.graph is not None and edge.agent_edge_id is not None
            else None
        )
        evidence_ids = set()
        if edge_agent is not None and isinstance(edge_agent.condition, ConceptRef):
            evidence_ids = {
                evidence.observation_id
                for evidence in (edge_agent.condition.evidence or [])
            }
        output.append(
            classify_failed_slot(
                target_id=f"edge:{edge.truth_edge_id}:condition",
                property_name="condition",
                slot=edge.condition,
                truth=truth,
                knowledge=knowledge,
                db=db,
                annotations=annotations,
                concept_trace=trace,
                agent_evidence_observation_ids=evidence_ids,
            )
        )
    return output
