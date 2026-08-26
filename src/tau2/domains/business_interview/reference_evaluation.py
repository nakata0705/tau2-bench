"""Reference-only StakeholderKnowledge-to-Truth evaluation.

This lane adapts evaluator-private local-to-Truth mappings to the shared aligned
comparison primitive.  It never feeds Agent scoring or reward.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from .comparison import (
    GraphAlignment,
    compare_aligned_graphs,
    knowledge_coverage,
    reconstruction_complete,
    truth_referenced_concept_ids,
)
from .graph import (
    business_edge_ids,
    business_entry_node_ids,
    business_graph_projection,
    business_node_ids,
    edge_is_structural,
)
from .knowledge import StakeholderKnowledge
from .stakeholder import StakeholderForgettingConfig


class StakeholderReferenceInput(BaseModel):
    stakeholder_id: str
    stakeholder_name: str = ""
    stakeholder_role: Optional[str] = None
    forgetting_configuration: dict[str, Any] = Field(default_factory=dict)
    knowledge: StakeholderKnowledge


class StakeholderTruthReferenceMetrics(BaseModel):
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
    glossary_complete: bool
    knowledge_coverage: float
    structural_component_score: float
    quality_component_score: float
    aggregate_score: float
    structural_pass: bool
    reconstruction_pass: bool
    quality_pass: bool


class StakeholderTruthReferenceDiagnostics(BaseModel):
    schema_version: str = "business_interview.stakeholder_truth_reference.v1"
    reference_only: bool = True
    primary_score_untouched: bool = True
    scoring_excludes_structural_elements: bool = True
    node_alignment: dict[str, str] = Field(default_factory=dict)
    edge_alignment: dict[str, str] = Field(default_factory=dict)
    concept_alignment: dict[str, str] = Field(default_factory=dict)
    missing_truth_node_ids: list[str] = Field(default_factory=list)
    missing_truth_edge_ids: list[str] = Field(default_factory=list)
    unmatched_stakeholder_node_ids: list[str] = Field(default_factory=list)
    unmatched_stakeholder_edge_ids: list[str] = Field(default_factory=list)
    missing_truth_concept_ids: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    forgetting_configuration: dict[str, Any] = Field(default_factory=dict)
    contracted_node_count: int = 0
    shortcut_edge_count: int = 0
    shortcut_provenance: list[dict[str, Any]] = Field(default_factory=list)
    canonical_contract: dict[str, object] = Field(default_factory=dict)


class StakeholderTruthReferenceEvaluation(BaseModel):
    stakeholder_id: str
    stakeholder_name: str = ""
    stakeholder_role: Optional[str] = None
    truth_reconstruction: StakeholderTruthReferenceMetrics
    diagnostics: StakeholderTruthReferenceDiagnostics


class StakeholderTruthReferenceAggregate(BaseModel):
    reference_only: bool = True
    stakeholder_count: int = 0
    min_stakeholder_truth_score: Optional[float] = None
    max_stakeholder_truth_score: Optional[float] = None
    mean_stakeholder_truth_score: Optional[float] = None


def _reference_stakeholder_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "stakeholder"


def _knowledge_fallback_identity(
    knowledge: StakeholderKnowledge,
) -> tuple[str, str]:
    graph = knowledge.graph
    name = str(getattr(graph, "name", "") or "")
    graph_id = str(getattr(graph, "id", "") or "")
    return graph_id or _reference_stakeholder_id(name), name


def _forgetting_configuration(filter_: Any) -> dict[str, Any]:
    forgetting = getattr(filter_, "forgetting", None)
    if forgetting is None:
        return {}
    if isinstance(forgetting, StakeholderForgettingConfig):
        return forgetting.model_dump(mode="json")
    if isinstance(forgetting, dict):
        return dict(forgetting)
    return {}


def coerce_stakeholder_reference(item: Any) -> StakeholderReferenceInput:
    """Normalize current scenario profile/reference inputs for offline use."""
    if isinstance(item, StakeholderReferenceInput):
        return item
    if isinstance(item, StakeholderKnowledge):
        stakeholder_id, stakeholder_name = _knowledge_fallback_identity(item)
        return StakeholderReferenceInput(
            stakeholder_id=stakeholder_id,
            stakeholder_name=stakeholder_name,
            knowledge=item,
        )
    if isinstance(item, dict):
        return StakeholderReferenceInput.model_validate(item)
    knowledge = getattr(item, "knowledge", None)
    if isinstance(knowledge, dict):
        knowledge = StakeholderKnowledge.model_validate(knowledge)
    if not isinstance(knowledge, StakeholderKnowledge):
        raise TypeError(
            "stakeholder reference must contain a StakeholderKnowledge object"
        )
    filter_ = getattr(item, "stakeholder", None)
    name = (
        getattr(item, "stakeholder_name", None)
        or getattr(item, "name", None)
        or getattr(filter_, "name", None)
        or ""
    )
    stakeholder_id = (
        getattr(item, "stakeholder_id", None)
        or getattr(filter_, "stakeholder_id", None)
        or _reference_stakeholder_id(name)
    )
    role = getattr(item, "stakeholder_role", None) or getattr(filter_, "role", None)
    return StakeholderReferenceInput(
        stakeholder_id=str(stakeholder_id),
        stakeholder_name=str(name),
        stakeholder_role=role,
        forgetting_configuration=_forgetting_configuration(filter_),
        knowledge=knowledge,
    )


def normalize_stakeholder_references(
    references: list[Any],
) -> list[StakeholderReferenceInput]:
    """Normalize an explicit, current multi-stakeholder reference list."""
    normalized = sorted(
        (coerce_stakeholder_reference(item) for item in references),
        key=lambda item: (
            item.stakeholder_id,
            item.stakeholder_name,
            item.stakeholder_role or "",
        ),
    )
    ids = [item.stakeholder_id for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "stakeholder reference ids must be unique; provide explicit stable "
            "ids for multiple StakeholderKnowledge views"
        )
    return normalized


def _reference_truth_mappings(
    truth: Any,
    knowledge: StakeholderKnowledge,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Map opaque local ids only through evaluator-private mappings."""
    target = business_graph_projection(truth)
    graph = knowledge.graph
    node_alignment: dict[str, str] = {}
    for local_id in sorted(business_node_ids(graph)):
        truth_id = graph.node_truth_ids.get(local_id)
        if isinstance(truth_id, str) and truth_id in target.nodes:
            node_alignment[local_id] = truth_id

    edge_alignment: dict[str, str] = {}
    for local_id in sorted(business_edge_ids(graph)):
        edge = graph.edges[local_id]
        if getattr(edge, "is_shortcut", False):
            continue
        truth_id = graph.edge_truth_ids.get(local_id)
        if not isinstance(truth_id, str) or truth_id not in target.edges:
            continue
        truth_edge = target.edges[truth_id]
        if (
            node_alignment.get(edge.from_node) == truth_edge.from_node
            and node_alignment.get(edge.to_node) == truth_edge.to_node
        ):
            edge_alignment[local_id] = truth_id

    expected_concepts = truth_referenced_concept_ids(target)
    concept_alignment: dict[str, str] = {}
    for local_id in sorted(graph.referenced_concept_ids()):
        concept = graph.concepts.get(local_id)
        truth_id = getattr(concept, "truth_concept_id", None)
        if (
            isinstance(truth_id, str)
            and truth_id in expected_concepts
            and concept is not None
            and getattr(concept, "kind", None)
            == getattr(target.concepts.get(truth_id), "kind", None)
        ):
            concept_alignment[local_id] = truth_id
    return node_alignment, edge_alignment, concept_alignment


def _shortcut_diagnostics(
    truth: Any,
    knowledge: StakeholderKnowledge,
    node_alignment: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    graph = knowledge.graph
    for local_id, edge in sorted(graph.edges.items()):
        if not getattr(edge, "is_shortcut", False):
            continue
        provenance = dict(graph.shortcut_provenance.get(local_id, {}))
        contracted = list(
            provenance.get("contracted_nodes") or getattr(edge, "contracted_nodes", [])
        )
        derived = list(
            provenance.get("derived_from_edges")
            or getattr(edge, "derived_from_edges", [])
        )
        records.append(
            {
                "stakeholder_edge_id": local_id,
                "from_node_id": edge.from_node,
                "to_node_id": edge.to_node,
                "truth_from_node_id": node_alignment.get(edge.from_node),
                "truth_to_node_id": node_alignment.get(edge.to_node),
                "is_shortcut": True,
                "contracted_nodes": contracted,
                "derived_from_edges": derived,
                "truth_path_edge_ids": [
                    edge_id for edge_id in derived if edge_id in truth.edges
                ],
                "matched_truth_edge_id": None,
                "direct_business_edge_credit": False,
                "reason": (
                    "safe serial contraction is diagnostic provenance; the "
                    "derived edge is not an exact Truth business edge"
                ),
            }
        )
    return records


def _component_score(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_stakeholder_truth_reference(
    truth: Any,
    reference: StakeholderReferenceInput,
) -> StakeholderTruthReferenceEvaluation:
    """Compare one three-state StakeholderKnowledge view to Truth."""
    target = business_graph_projection(truth)
    graph = reference.knowledge.graph
    validation_errors = graph.structure_errors()
    nodes, edges, concepts = _reference_truth_mappings(truth, reference.knowledge)
    expected_concepts = truth_referenced_concept_ids(target)
    attempted_concepts = set(graph.referenced_concept_ids())
    recalled_concepts = set(concepts.values()) & expected_concepts
    concept_recall = (
        len(recalled_concepts) / len(expected_concepts) if expected_concepts else 1.0
    )
    concept_precision = (
        len(concepts) / len(attempted_concepts) if attempted_concepts else 1.0
    )
    alignment = GraphAlignment(
        concept_to_truth=concepts,
        node_to_truth=nodes,
        edge_to_truth=edges,
        concept_recall=concept_recall,
        concept_precision=concept_precision,
    )

    stakeholder_starts: set[str] = set()
    if graph.start_node_id is not None:
        stakeholder_starts.add(graph.start_node_id)
    else:
        stakeholder_starts.update(
            edge.to_node
            for edge in graph.edges.values()
            if edge_is_structural(edge)
            and edge.from_node == graph.source_node_id
            and edge.to_node in graph.nodes
        )
    candidate_node_ids = sorted(business_node_ids(graph))
    candidate_edge_ids = sorted(business_edge_ids(graph))
    comparison = compare_aligned_graphs(
        candidate=graph,
        truth=target,
        alignment=alignment,
        candidate_node_ids=candidate_node_ids,
        candidate_edge_ids=candidate_edge_ids,
        candidate_start_node_ids=stakeholder_starts,
        candidate_end_node_ids=set(graph.end_node_ids),
        truth_entry_node_ids=set(business_entry_node_ids(truth)),
        graph_valid=not validation_errors,
        known_absent=lambda value: value is None,
        empty_node_recall=1.0,
    )
    complete = reconstruction_complete(comparison)
    structural_values = [
        1.0 if comparison.graph_valid else 0.0,
        comparison.node_recall,
        comparison.node_precision,
        comparison.edge_recall,
        comparison.edge_precision,
        1.0 if comparison.start_correct else 0.0,
        comparison.end_recall,
        comparison.end_precision,
    ]
    quality_values = [
        comparison.activity_correctness,
        comparison.actor_correctness,
        comparison.system_correctness,
        comparison.read_correctness,
        comparison.write_correctness,
        comparison.rationale_correctness,
        comparison.condition_correctness,
        comparison.concept_correctness,
        comparison.concept_recall,
        comparison.concept_precision,
    ]
    structural_component = _component_score(structural_values)
    quality_component = _component_score(quality_values)
    metrics = StakeholderTruthReferenceMetrics(
        graph_created=comparison.graph_created,
        graph_valid=comparison.graph_valid,
        node_recall=comparison.node_recall,
        node_precision=comparison.node_precision,
        edge_recall=comparison.edge_recall,
        edge_precision=comparison.edge_precision,
        start_correct=comparison.start_correct,
        end_recall=comparison.end_recall,
        end_precision=comparison.end_precision,
        activity_correctness=comparison.activity_correctness,
        actor_correctness=comparison.actor_correctness,
        system_correctness=comparison.system_correctness,
        read_correctness=comparison.read_correctness,
        write_correctness=comparison.write_correctness,
        rationale_correctness=comparison.rationale_correctness,
        condition_correctness=comparison.condition_correctness,
        concept_correctness=comparison.concept_correctness,
        concept_recall=comparison.concept_recall,
        concept_precision=comparison.concept_precision,
        unsupported_ref_count=comparison.unsupported_ref_count,
        fabricated_node_count=comparison.fabricated_node_count,
        fabricated_edge_count=comparison.fabricated_edge_count,
        glossary_complete=comparison.glossary_complete,
        knowledge_coverage=knowledge_coverage(truth, reference.knowledge),
        structural_component_score=structural_component,
        quality_component_score=quality_component,
        aggregate_score=(structural_component + quality_component) / 2.0,
        structural_pass=complete,
        reconstruction_pass=complete,
        quality_pass=complete,
    )

    shortcut_records = _shortcut_diagnostics(truth, reference.knowledge, nodes)
    from .evaluation_diagnostics import canonical_contract_diagnostic

    canonical_contract = canonical_contract_diagnostic(truth, reference.knowledge)
    canonical_contract.update(
        {
            "reference_only": True,
            "shortcut_edges_receive_direct_business_edge_credit": False,
            "reference_business_projection": True,
            "stakeholder_shortcut_provenance": shortcut_records,
        }
    )
    contracted_node_ids = {
        node_id for record in shortcut_records for node_id in record["contracted_nodes"]
    }
    diagnostics = StakeholderTruthReferenceDiagnostics(
        node_alignment=dict(sorted(nodes.items())),
        edge_alignment=dict(sorted(edges.items())),
        concept_alignment=dict(sorted(concepts.items())),
        missing_truth_node_ids=sorted(set(target.nodes) - set(nodes.values())),
        missing_truth_edge_ids=sorted(set(target.edges) - set(edges.values())),
        unmatched_stakeholder_node_ids=sorted(set(candidate_node_ids) - set(nodes)),
        unmatched_stakeholder_edge_ids=sorted(set(candidate_edge_ids) - set(edges)),
        missing_truth_concept_ids=sorted(expected_concepts - recalled_concepts),
        validation_errors=validation_errors,
        forgetting_configuration=dict(reference.forgetting_configuration),
        contracted_node_count=len(contracted_node_ids),
        shortcut_edge_count=len(shortcut_records),
        shortcut_provenance=shortcut_records,
        canonical_contract=canonical_contract,
    )
    return StakeholderTruthReferenceEvaluation(
        stakeholder_id=reference.stakeholder_id,
        stakeholder_name=reference.stakeholder_name,
        stakeholder_role=reference.stakeholder_role,
        truth_reconstruction=metrics,
        diagnostics=diagnostics,
    )


def aggregate_stakeholder_truth_references(
    references: list[StakeholderTruthReferenceEvaluation],
) -> StakeholderTruthReferenceAggregate:
    scores = [item.truth_reconstruction.aggregate_score for item in references]
    return StakeholderTruthReferenceAggregate(
        stakeholder_count=len(references),
        min_stakeholder_truth_score=min(scores) if scores else None,
        max_stakeholder_truth_score=max(scores) if scores else None,
        mean_stakeholder_truth_score=(sum(scores) / len(scores) if scores else None),
    )
