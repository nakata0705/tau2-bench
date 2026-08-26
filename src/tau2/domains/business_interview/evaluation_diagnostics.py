"""Detailed diagnostic DTOs and builders for business_interview evaluation.

These are evaluator-only reconstruction diagnostics.  They reuse the primary
scoring helpers without feeding back into any score or pass/fail field.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .comparison import (
    _concept_similarity,
    _exact_label_match_path,
    _score_list_slot,
    _score_scalar_slot,
    agent_referenced_concept_ids,
    slot_value,
    truth_referenced_concept_ids,
)
from .graph import (
    AgentGraph,
    ConceptRef,
    business_edge_ids,
    business_node_ids,
    canonical_structure_errors,
    edge_is_structural,
    is_absent,
    is_dont_know,
    is_unset,
    node_is_structural,
)

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")
_DIAGNOSTIC_NODE_PROPS = (
    ("activity", "activity"),
    ("actor", "actor"),
    ("system", "system"),
    ("reads", "reads"),
    ("writes", "writes"),
    ("necessity_rationale", "rationale"),
)
_NO_VALUE: object = object()


class ConceptSummary(BaseModel):
    concept_id: str
    kind: str
    label: Optional[str] = None
    labels: list[str] = Field(default_factory=list)


class ConceptPairDiagnostic(BaseModel):
    truth_concept_id: str
    truth_kind: str
    truth_label: Optional[str] = None
    truth_labels: list[str] = Field(default_factory=list)
    agent_concept_id: str
    agent_kind: str
    agent_label: Optional[str] = None
    exact_label_match: bool = False
    exact_label_match_path: Optional[str] = None
    lexical_similarity_score: float = 0.0
    threshold: float
    eligible: bool = False
    selected_mapping: bool = False


class ConceptMappingDiagnostic(BaseModel):
    agent_concept_id: str
    truth_concept_id: str
    agent_kind: str
    truth_kind: str
    agent_label: Optional[str] = None
    truth_label: Optional[str] = None
    exact_label_match_path: Optional[str] = None
    lexical_similarity_score: float = 0.0


class ConceptDiagnostics(BaseModel):
    threshold: float
    expected_truth_concept_ids: list[str] = Field(default_factory=list)
    attempted_agent_concept_ids: list[str] = Field(default_factory=list)
    agent_to_truth: dict[str, str] = Field(default_factory=dict)
    candidate_pairs: list[ConceptPairDiagnostic] = Field(default_factory=list)
    selected_mappings: list[ConceptMappingDiagnostic] = Field(default_factory=list)
    unmatched_truth_concepts: list[ConceptSummary] = Field(default_factory=list)
    unmatched_agent_concepts: list[ConceptSummary] = Field(default_factory=list)


class SlotItemDiagnostic(BaseModel):
    item_type: str
    truth_concept_id: Optional[str] = None
    agent_concept_id: Optional[str] = None
    mapped_truth_concept_id: Optional[str] = None
    truth_label: Optional[str] = None
    agent_label: Optional[str] = None
    matched: bool = False
    reason: str


class SlotDiagnostic(BaseModel):
    property: str
    slot_kind: str
    truth_state: str
    agent_state: str
    truth_concept_id: Optional[str] = None
    agent_concept_id: Optional[str] = None
    truth_concept_ids: list[str] = Field(default_factory=list)
    agent_concept_ids: list[str] = Field(default_factory=list)
    mapped_truth_concept_ids: list[str] = Field(default_factory=list)
    truth_label: Optional[str] = None
    agent_label: Optional[str] = None
    truth_labels: list[str] = Field(default_factory=list)
    agent_labels: list[str] = Field(default_factory=list)
    matched: bool = False
    score_contribution: float = 0.0
    reason: str
    reason_codes: list[str] = Field(default_factory=list)
    missing_truth_concept_ids: list[str] = Field(default_factory=list)
    extra_agent_concept_ids: list[str] = Field(default_factory=list)
    items: list[SlotItemDiagnostic] = Field(default_factory=list)


class NodeDiagnostic(BaseModel):
    truth_node_id: str
    agent_node_id: Optional[str] = None
    matched: bool = False
    reason: str
    slots: dict[str, SlotDiagnostic] = Field(default_factory=dict)


class EdgeDiagnostic(BaseModel):
    truth_edge_id: str
    agent_edge_id: Optional[str] = None
    matched: bool = False
    reason: str
    truth_from_node: str
    truth_to_node: str
    agent_from_node: Optional[str] = None
    agent_to_node: Optional[str] = None
    from_node_match: bool = False
    to_node_match: bool = False
    structural_match: bool = False
    condition: SlotDiagnostic


class FailureAttribution(BaseModel):
    target_id: str
    dimension: str
    property: str
    category: Literal[
        "stakeholder_disclosure",
        "agent_elicitation",
        "agent_recording",
        "evaluator_matching",
        "insufficient_evidence_to_classify",
    ]
    reason: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationDiagnostics(BaseModel):
    schema_version: str = "business_interview.evaluation_diagnostics.v5"
    canonical_contract: dict[str, object] = Field(default_factory=dict)
    score_fields_unchanged: bool = True
    node_diagnostics: list[NodeDiagnostic] = Field(default_factory=list)
    unmatched_agent_nodes: list[str] = Field(default_factory=list)
    edge_diagnostics: list[EdgeDiagnostic] = Field(default_factory=list)
    unmatched_agent_edges: list[str] = Field(default_factory=list)
    concepts: ConceptDiagnostics = Field(
        default_factory=lambda: ConceptDiagnostics(threshold=0.0)
    )


# Utility functions for diagnostic builders
# These are kept in this module since they are diagnostic-only.


def concept_labels(concept, is_truth: bool) -> list[str]:
    if concept is None:
        return []
    if is_truth:
        return [str(label) for label in (concept.canonical_terms or [])]
    label = getattr(concept, "display_label", None)
    return [str(label)] if label else []


def concept_label(concept, is_truth: bool) -> Optional[str]:
    labels = concept_labels(concept, is_truth)
    return labels[0] if labels else None


def agent_state(value) -> str:
    if value is _NO_VALUE:
        return "missing"
    if value is None or is_unset(value):
        return "unset"
    if is_absent(value):
        return "absent"
    if is_dont_know(value):
        return "dont_know"
    if isinstance(value, ConceptRef):
        return "value" if value.asserted else "value_unasserted"
    if isinstance(value, list):
        return "value" if value else "value_empty"
    return type(value).__name__


def truth_state(value) -> str:
    if isinstance(value, ConceptRef):
        return "value"
    if isinstance(value, list) and value:
        return "value"
    return "absent"


def concept_for_ref(ref, concepts):
    if isinstance(ref, ConceptRef):
        return concepts.get(ref.concept_id)
    return None


def reason_for_epistemic_states(truth_state_val: str, agent_state_val: str) -> str:
    if truth_state_val == "value":
        return {
            "unset": "truth_value_agent_unset",
            "absent": "truth_value_agent_absent",
            "dont_know": "truth_value_agent_dont_know",
            "missing": "unmatched_node",
            "value_unasserted": "truth_value_agent_unasserted",
        }.get(agent_state_val, "wrong_concept")
    return {
        "absent": "truth_absent_agent_absent",
        "unset": "truth_absent_agent_unset",
        "dont_know": "truth_absent_agent_dont_know",
        "missing": "unmatched_node",
    }.get(agent_state_val, "truth_absent_agent_value")


def slot_diagnostic(
    property_name: str,
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
    agent_concepts,
    truth_concepts,
    *,
    unmatched_reason: Optional[str] = None,
) -> SlotDiagnostic:
    slot_kind = "list" if property_name in ("reads", "writes") else "scalar"
    truth_refs = list(truth_value) if isinstance(truth_value, list) else [truth_value]
    truth_refs = [ref for ref in truth_refs if isinstance(ref, ConceptRef)]
    agent_refs = list(agent_value) if isinstance(agent_value, list) else [agent_value]
    agent_refs = [ref for ref in agent_refs if isinstance(ref, ConceptRef)]
    truth_ids = sorted({ref.concept_id for ref in truth_refs})
    agent_ids = sorted({ref.concept_id for ref in agent_refs})
    mapped_ids = sorted(
        {
            agent_to_truth[ref.concept_id]
            for ref in agent_refs
            if ref.asserted and ref.concept_id in agent_to_truth
        }
    )
    t_state = truth_state(truth_value)
    a_state = agent_state(agent_value)
    truth_ref = truth_refs[0] if slot_kind == "scalar" and truth_refs else None
    agent_ref = agent_refs[0] if slot_kind == "scalar" and agent_refs else None
    truth_concept_obj = concept_for_ref(truth_ref, truth_concepts)
    agent_concept_obj = concept_for_ref(agent_ref, agent_concepts)

    if unmatched_reason is not None:
        return SlotDiagnostic(
            property=property_name,
            slot_kind=slot_kind,
            truth_state=t_state,
            agent_state="missing",
            truth_concept_id=truth_ref.concept_id if truth_ref else None,
            truth_concept_ids=truth_ids,
            truth_label=concept_label(truth_concept_obj, True),
            truth_labels=concept_labels(truth_concept_obj, True),
            matched=False,
            score_contribution=0.0,
            reason=unmatched_reason,
            reason_codes=[unmatched_reason],
            missing_truth_concept_ids=truth_ids,
        )

    if slot_kind == "list":
        score, _unsupported = _score_list_slot(agent_value, truth_value, agent_to_truth)
        expected = set(truth_ids)
        claimed = {
            agent_to_truth[ref.concept_id]
            for ref in agent_refs
            if ref.asserted and ref.concept_id in agent_to_truth
        }
        missing = sorted(expected - claimed)
        extra_refs_list = [
            ref
            for ref in agent_refs
            if ref.asserted and agent_to_truth.get(ref.concept_id) not in expected
        ]
        extra_ids = sorted({ref.concept_id for ref in extra_refs_list})
        items: list[SlotItemDiagnostic] = []
        for tid in sorted(expected):
            tconcept = truth_concepts.get(tid)
            items.append(
                SlotItemDiagnostic(
                    item_type="truth",
                    truth_concept_id=tid,
                    truth_label=concept_label(tconcept, True),
                    matched=tid in claimed,
                    reason="correct_value" if tid in claimed else "missing_list_item",
                )
            )
        for idx, ref in sorted(
            enumerate(agent_refs), key=lambda pair: (pair[1].concept_id, pair[0])
        ):
            mapped = agent_to_truth.get(ref.concept_id)
            aconcept = agent_concepts.get(ref.concept_id)
            matched_item = bool(ref.asserted and mapped in expected)
            item_reason = "correct_value" if matched_item else "extra_list_item"
            if not ref.asserted:
                item_reason = "truth_value_agent_unasserted"
            items.append(
                SlotItemDiagnostic(
                    item_type="agent",
                    agent_concept_id=ref.concept_id,
                    mapped_truth_concept_id=mapped,
                    agent_label=concept_label(aconcept, False),
                    truth_label=(
                        concept_label(truth_concepts.get(mapped), True)
                        if mapped is not None
                        else None
                    ),
                    matched=matched_item,
                    reason=item_reason,
                )
            )

        if not truth_ids:
            if score == 1.0:
                reason_str = "truth_absent_agent_absent"
                codes = [reason_str]
            elif extra_ids:
                reason_str = "extra_list_item"
                codes = [reason_str]
            else:
                reason_str = reason_for_epistemic_states(t_state, a_state)
                codes = [reason_str]
        elif not isinstance(agent_value, list):
            reason_str = reason_for_epistemic_states(t_state, a_state)
            codes = [reason_str, "missing_list_item"]
        elif score == 1.0 and not missing and not extra_ids:
            reason_str = "correct_value"
            codes = [reason_str]
        else:
            codes = []
            if missing:
                codes.append("missing_list_item")
            if extra_ids:
                codes.append("extra_list_item")
            if not codes:
                codes.append(reason_for_epistemic_states(t_state, a_state))
            if len(codes) > 1 and set(codes) == {
                "missing_list_item",
                "extra_list_item",
            }:
                reason_str = "missing_and_extra_list_items"
            else:
                reason_str = codes[0]
        if extra_ids and any(
            agent_to_truth.get(ref.concept_id) is None for ref in extra_refs_list
        ):
            codes.append("wrong_concept")
        return SlotDiagnostic(
            property=property_name,
            slot_kind=slot_kind,
            truth_state=t_state,
            agent_state=a_state,
            truth_concept_ids=truth_ids,
            agent_concept_ids=agent_ids,
            mapped_truth_concept_ids=mapped_ids,
            truth_labels=[
                label
                for tid in truth_ids
                for label in concept_labels(truth_concepts.get(tid), True)
            ],
            agent_labels=[
                label
                for aid in agent_ids
                for label in concept_labels(agent_concepts.get(aid), False)
            ],
            matched=score == 1.0,
            score_contribution=score,
            reason=reason_str,
            reason_codes=list(dict.fromkeys(codes)),
            missing_truth_concept_ids=missing,
            extra_agent_concept_ids=extra_ids,
            items=items,
        )

    score = _score_scalar_slot(agent_value, truth_value, agent_to_truth)
    if score == 1:
        reason_str = (
            "truth_absent_agent_absent"
            if t_state == "absent" and a_state == "absent"
            else "correct_value"
        )
        codes = [reason_str]
    elif isinstance(truth_value, ConceptRef) and isinstance(agent_value, ConceptRef):
        reason_str = "wrong_concept"
        codes = [reason_str]
    else:
        reason_str = reason_for_epistemic_states(t_state, a_state)
        codes = [reason_str]
        if isinstance(agent_value, ConceptRef) and t_state == "absent":
            codes.append("wrong_concept")
    return SlotDiagnostic(
        property=property_name,
        slot_kind=slot_kind,
        truth_state=t_state,
        agent_state=a_state,
        truth_concept_id=truth_ref.concept_id if truth_ref else None,
        agent_concept_id=agent_ref.concept_id if agent_ref else None,
        truth_concept_ids=truth_ids,
        agent_concept_ids=agent_ids,
        mapped_truth_concept_ids=mapped_ids,
        truth_label=concept_label(truth_concept_obj, True),
        agent_label=concept_label(agent_concept_obj, False),
        truth_labels=concept_labels(truth_concept_obj, True),
        agent_labels=concept_labels(agent_concept_obj, False),
        matched=score == 1,
        score_contribution=score,
        reason=reason_str,
        reason_codes=list(dict.fromkeys(codes)),
    )


def concept_diagnostics(
    agent: AgentGraph,
    truth,
    term_extras: dict[str, list[str]],
    agent_to_truth: dict[str, str],
) -> ConceptDiagnostics:
    """Trace the already-selected concept mapping without changing it."""
    from .comparison import (
        _CONCEPT_MATCH_THRESHOLD,
    )
    from .evaluation_diagnostics import (
        concept_label,
        concept_labels,
    )

    expected = sorted(truth_referenced_concept_ids(truth))
    attempted = sorted(agent_referenced_concept_ids(agent))
    pairs: list[ConceptPairDiagnostic] = []
    for acid in attempted:
        aconcept = agent.concepts.get(acid)
        if aconcept is None:
            continue
        for tid in expected:
            tconcept = truth.concepts.get(tid)
            if tconcept is None or aconcept.kind != tconcept.kind:
                continue
            path = _exact_label_match_path(aconcept, tconcept, term_extras.get(tid))
            similarity = _concept_similarity(aconcept, tconcept, term_extras.get(tid))
            pairs.append(
                ConceptPairDiagnostic(
                    truth_concept_id=tid,
                    truth_kind=tconcept.kind,
                    truth_label=concept_label(tconcept, True),
                    truth_labels=concept_labels(tconcept, True),
                    agent_concept_id=acid,
                    agent_kind=aconcept.kind,
                    agent_label=concept_label(aconcept, False),
                    exact_label_match=path is not None,
                    exact_label_match_path=path,
                    lexical_similarity_score=similarity,
                    threshold=_CONCEPT_MATCH_THRESHOLD,
                    eligible=similarity >= _CONCEPT_MATCH_THRESHOLD,
                    selected_mapping=agent_to_truth.get(acid) == tid,
                )
            )

    selected: list[ConceptMappingDiagnostic] = []
    for acid, tid in sorted(agent_to_truth.items()):
        aconcept = agent.concepts.get(acid)
        tconcept = truth.concepts.get(tid)
        if aconcept is None or tconcept is None:
            continue
        selected.append(
            ConceptMappingDiagnostic(
                agent_concept_id=acid,
                truth_concept_id=tid,
                agent_kind=aconcept.kind,
                truth_kind=tconcept.kind,
                agent_label=concept_label(aconcept, False),
                truth_label=concept_label(tconcept, True),
                exact_label_match_path=_exact_label_match_path(
                    aconcept, tconcept, term_extras.get(tid)
                ),
                lexical_similarity_score=_concept_similarity(
                    aconcept, tconcept, term_extras.get(tid)
                ),
            )
        )

    unmatched_truth = [
        ConceptSummary(
            concept_id=tid,
            kind=truth.concepts[tid].kind,
            label=concept_label(truth.concepts[tid], True),
            labels=concept_labels(truth.concepts[tid], True),
        )
        for tid in expected
        if tid not in set(agent_to_truth.values()) and tid in truth.concepts
    ]
    unmatched_agent = [
        ConceptSummary(
            concept_id=acid,
            kind=agent.concepts[acid].kind,
            label=concept_label(agent.concepts[acid], False),
            labels=concept_labels(agent.concepts[acid], False),
        )
        for acid in attempted
        if acid not in agent_to_truth and acid in agent.concepts
    ]
    return ConceptDiagnostics(
        threshold=_CONCEPT_MATCH_THRESHOLD,
        expected_truth_concept_ids=expected,
        attempted_agent_concept_ids=attempted,
        agent_to_truth={acid: agent_to_truth[acid] for acid in sorted(agent_to_truth)},
        candidate_pairs=pairs,
        selected_mappings=selected,
        unmatched_truth_concepts=unmatched_truth,
        unmatched_agent_concepts=unmatched_agent,
    )


def build_evaluation_diagnostics(
    agent: AgentGraph,
    target,
    term_extras: dict[str, list[str]],
    agent_to_truth: dict[str, str],
    node_mapping: dict[str, str],
    edge_mapping: dict[str, str],
) -> EvaluationDiagnostics:
    """Build the detailed diagnostic trace without usage/joint experiments."""
    concept_trace = concept_diagnostics(agent, target, term_extras, agent_to_truth)
    truth_to_agent = {tid: aid for aid, tid in node_mapping.items()}
    nodes: list[NodeDiagnostic] = []
    for tid in sorted(target.nodes):
        aid = truth_to_agent.get(tid)
        truth_node = target.nodes[tid]
        slots: dict[str, SlotDiagnostic] = {}
        for output_prop, score_prop in _DIAGNOSTIC_NODE_PROPS:
            truth_val = slot_value(truth_node, score_prop)
            if aid is None:
                slots[output_prop] = slot_diagnostic(
                    output_prop,
                    _NO_VALUE,
                    truth_val,
                    agent_to_truth,
                    agent.concepts,
                    target.concepts,
                    unmatched_reason="unmatched_node",
                )
            else:
                slots[output_prop] = slot_diagnostic(
                    output_prop,
                    agent.nodes[aid].slot_value(score_prop),
                    truth_val,
                    agent_to_truth,
                    agent.concepts,
                    target.concepts,
                )
        nodes.append(
            NodeDiagnostic(
                truth_node_id=tid,
                agent_node_id=aid,
                matched=aid is not None,
                reason="matched_node" if aid is not None else "unmatched_node",
                slots=slots,
            )
        )

    edges: list[EdgeDiagnostic] = []
    truth_to_agent_edge = {
        truth_id: agent_id for agent_id, truth_id in edge_mapping.items()
    }
    for tid in sorted(target.edges):
        truth_edge = target.edges[tid]
        aid = truth_to_agent_edge.get(tid)
        if aid is not None:
            agent_edge = agent.edges[aid]
            from_match = node_mapping.get(agent_edge.from_node) == truth_edge.from_node
            to_match = node_mapping.get(agent_edge.to_node) == truth_edge.to_node
            condition_diag = slot_diagnostic(
                "condition",
                agent_edge.condition,
                truth_edge.condition,
                agent_to_truth,
                agent.concepts,
                target.concepts,
            )
            str_match = from_match and to_match
        else:
            agent_edge = None
            from_match = to_match = str_match = False
            condition_diag = slot_diagnostic(
                "condition",
                _NO_VALUE,
                truth_edge.condition,
                agent_to_truth,
                agent.concepts,
                target.concepts,
                unmatched_reason="unmatched_edge",
            )
        edges.append(
            EdgeDiagnostic(
                truth_edge_id=tid,
                agent_edge_id=aid,
                matched=str_match,
                reason="matched_edge" if str_match else "unmatched_edge",
                truth_from_node=truth_edge.from_node,
                truth_to_node=truth_edge.to_node,
                agent_from_node=(agent_edge.from_node if agent_edge else None),
                agent_to_node=(agent_edge.to_node if agent_edge else None),
                from_node_match=from_match,
                to_node_match=to_match,
                structural_match=str_match,
                condition=condition_diag,
            )
        )

    return EvaluationDiagnostics(
        node_diagnostics=nodes,
        unmatched_agent_nodes=sorted(
            aid for aid in business_node_ids(agent) if aid not in node_mapping
        ),
        edge_diagnostics=edges,
        unmatched_agent_edges=sorted(
            eid for eid in business_edge_ids(agent) if eid not in edge_mapping
        ),
        concepts=concept_trace,
    )


def canonical_contract_diagnostic(graph, knowledge=None) -> dict[str, object]:
    """Describe boundary/shortcut handling without affecting any score."""
    has_explicit_boundary = bool(
        getattr(graph, "source_node_id", None)
        and getattr(graph, "sink_node_id", None)
        and any(node_is_structural(node) for node in graph.nodes.values())
    )
    structural_nodes = [
        node_id for node_id, node in graph.nodes.items() if node_is_structural(node)
    ]
    structural_edges = [
        edge_id for edge_id, edge in graph.edges.items() if edge_is_structural(edge)
    ]
    shortcuts = [
        {
            "edge_id": edge_id,
            "contracted_nodes": list(getattr(edge, "contracted_nodes", [])),
            "derived_from_edges": list(getattr(edge, "derived_from_edges", [])),
        }
        for edge_id, edge in sorted(graph.edges.items())
        if getattr(edge, "is_shortcut", False)
    ]
    errors = canonical_structure_errors(graph) if has_explicit_boundary else []
    stakeholder_graph = getattr(knowledge, "graph", None)
    stakeholder_shortcuts = []
    if stakeholder_graph is not None:
        stakeholder_shortcuts = [
            {
                "edge_id": edge_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "contracted_nodes": list(getattr(edge, "contracted_nodes", [])),
                "derived_from_edges": list(getattr(edge, "derived_from_edges", [])),
            }
            for edge_id, edge in sorted(stakeholder_graph.edges.items())
            if getattr(edge, "is_shortcut", False)
        ]
    return {
        "explicit_source_node_id": getattr(graph, "source_node_id", None),
        "explicit_sink_node_id": getattr(graph, "sink_node_id", None),
        "canonical": has_explicit_boundary and not errors,
        "invariant_errors": errors,
        "structural_node_count": len(structural_nodes),
        "structural_edge_count": len(structural_edges),
        "business_node_count": len(business_node_ids(graph)),
        "business_edge_count": len(business_edge_ids(graph)),
        "structural_node_ids": sorted(structural_nodes),
        "structural_edge_ids": sorted(structural_edges),
        "truth_shortcut_edges": shortcuts,
        "stakeholder_shortcut_edges": stakeholder_shortcuts,
        "existing_evaluator_shortcut_policy": (
            "shortcut edges are currently ordinary Agent business edges; "
            "provenance is diagnostic and no score credit is granted"
        ),
        "scoring_excludes_structural_elements": True,
        "topology_derived_end_inference_used": False,
    }
