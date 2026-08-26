"""Lean primary evaluator for the business_interview benchmark.

The production score path is intentionally one-way::

    AgentGraph -> Agent-to-Truth alignment -> aligned comparison -> primary result

Reference evaluation, detailed traces, and research experiments consume this core
from separate modules.  This module does not import any of them.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .comparison import (
    AlignedGraphComparison,
    GraphAlignment,
    align_agent_to_truth,
    compare_aligned_graphs,
    knowledge_coverage,
    reconstruction_complete,
)
from .evaluation_diagnostics import EvaluationDiagnostics
from .graph import (
    AbsentType,
    AgentGraph,
    ConceptRef,
    DontKnowType,
    EvidenceRef,
    InterviewDB,
    business_edge_ids,
    business_entry_node_ids,
    business_graph_projection,
    business_node_ids,
)
from .reference_evaluation import (
    StakeholderTruthReferenceAggregate,
    StakeholderTruthReferenceEvaluation,
    aggregate_stakeholder_truth_references,
    evaluate_stakeholder_truth_reference,
    normalize_stakeholder_references,
)

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")


class PrimaryEvaluationResult(BaseModel):
    """All fields that participate in current score/reward semantics."""

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
    glossary_complete: bool
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
    marker_evidence_errors_surrogate: int
    invalid_observation_reference_count: int
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool
    evidence_pass: bool
    reconstruction_pass: bool
    structural_pass: bool
    protocol_pass: bool
    quality_pass: bool
    knowledge_coverage: float


class PrimaryEvaluation:
    """Primary result plus immutable comparison inputs for optional consumers."""

    def __init__(
        self,
        result: PrimaryEvaluationResult,
        agent: AgentGraph,
        truth: Any,
        target: Any,
        knowledge: Any,
        terminology_terms: dict[str, list[str]],
        alignment: GraphAlignment,
        comparison: AlignedGraphComparison,
    ):
        self.result = result
        self.agent = agent
        self.truth = truth
        self.target = target
        self.knowledge = knowledge
        self.terminology_terms = terminology_terms
        self.alignment = alignment
        self.comparison = comparison


# Backward compat aliases
class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    Kept as a model for API stability; scenarios may leave it empty.
    """


class EvaluationResult(BaseModel):
    """Full evaluation result including diagnostics and reference.

    This is built on top of PrimaryEvaluationResult for backward compatibility.
    """

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
    glossary_complete: bool
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
    marker_evidence_errors_surrogate: int
    invalid_observation_reference_count: int
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool
    evidence_pass: bool
    reconstruction_pass: bool
    structural_pass: bool
    protocol_pass: bool
    quality_pass: bool
    knowledge_coverage: float
    diagnostics: "EvaluationDiagnostics" = Field(default_factory=dict)  # type: ignore[assignment]
    stakeholder_truth_reference: list[StakeholderTruthReferenceEvaluation] = Field(
        default_factory=list
    )
    stakeholder_truth_reference_aggregate: StakeholderTruthReferenceAggregate = Field(
        default_factory=StakeholderTruthReferenceAggregate
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _terminology_terms(knowledge: Any) -> dict[str, list[str]]:
    terms: dict[str, list[str]] = {}
    if knowledge is None or not getattr(knowledge, "concepts", None):
        return terms
    for concept in knowledge.concepts.values():
        local_terms = getattr(concept, "terms", None)
        if isinstance(local_terms, list) and local_terms:
            terms.setdefault(concept.truth_concept_id, list(local_terms))
    return terms


def _evidence_metrics(
    db: InterviewDB,
    agent: AgentGraph,
) -> tuple[int, int, float, float, float]:
    """Return current invalid/count and node/ref/edge evidence coverage."""
    invalid = invalid_observation = 0
    ref_total = ref_hit = 0
    node_total = node_hit = 0
    edge_total = edge_hit = 0
    observation_text = {o.id: o.text for o in db.observations}

    def span_ok(ev: EvidenceRef) -> bool:
        nonlocal invalid, invalid_observation
        text = observation_text.get(ev.observation_id)
        if text is None:
            invalid += 1
            invalid_observation += 1
            return False
        if ev.quote and ev.resolve_span(text) is None:
            invalid += 1
            return False
        return True

    def all_ok(evs: list[EvidenceRef]) -> bool:
        ok = True
        for ev in evs:
            if not span_ok(ev):
                ok = False
        return ok

    for node in agent.nodes.values():
        refs: list[ConceptRef] = []
        markers: list[EvidenceRef] = []
        for prop in _NODE_PROPS:
            refs.extend(node.refs(prop))
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                markers.extend(slot.evidence)
        node_total += 1
        if any(
            r.asserted and any(span_ok(ev) for ev in r.evidence) for r in refs
        ) or any(span_ok(ev) for ev in markers):
            node_hit += 1
        for r in refs:
            if not r.asserted:
                continue
            ref_total += 1
            if r.evidence and all_ok(r.evidence):
                ref_hit += 1
    for edge in agent.edges.values():
        edge_total += 1
        if edge.evidence and all_ok(edge.evidence):
            edge_hit += 1
    return (
        invalid,
        invalid_observation,
        node_hit / node_total if node_total else 1.0,
        ref_hit / ref_total if ref_total else 1.0,
        edge_hit / edge_total if edge_total else 1.0,
    )


def _referenced_observations(agent: AgentGraph) -> set[str]:
    ids: set[str] = set()
    for node in agent.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                ids.update(ev.observation_id for ev in ref.evidence)
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                ids.update(ev.observation_id for ev in slot.evidence)
    for edge in agent.edges.values():
        ids.update(ev.observation_id for ev in edge.evidence)
        if isinstance(edge.condition, ConceptRef):
            ids.update(ev.observation_id for ev in edge.condition.evidence)
    return ids


def _build_primary_result(
    db: InterviewDB,
    truth: Any,
    knowledge_for_terms: Any = None,
) -> PrimaryEvaluation:
    """Build the primary evaluation for a given Agent graph and Truth."""

    target = business_graph_projection(truth)
    agent = db.graph if db.graph is not None else AgentGraph()
    terminology_terms = _terminology_terms(knowledge_for_terms)

    alignment = align_agent_to_truth(agent, target, terminology_terms)

    agent_start_ids = set(agent.start_node_ids)
    if not agent_start_ids and agent.start_node_id is not None:
        agent_start_ids = {agent.start_node_id}
    comparison = compare_aligned_graphs(
        candidate=agent,
        truth=target,
        alignment=alignment,
        candidate_node_ids=business_node_ids(agent),
        candidate_edge_ids=business_edge_ids(agent),
        candidate_start_node_ids=agent_start_ids,
        candidate_end_node_ids=set(agent.end_node_ids),
        truth_entry_node_ids=set(business_entry_node_ids(truth)),
        graph_valid=agent.is_valid,
    )

    authentic_ids: set[str] = set()
    invalid_source = 0
    for o in db.observations:
        m = db.messages[o.turn] if 0 <= o.turn < len(db.messages) else None
        if (
            m is not None
            and m.get("role") == "user"
            and (m.get("content") or "") == (o.text or "")
        ):
            authentic_ids.add(o.id)
        else:
            invalid_source += 1
    referenced_obs = _referenced_observations(agent)
    orphan_count = sum(1 for o in db.observations if o.id not in referenced_obs)
    invalid_refs, invalid_obs_refs, node_cov, ref_cov, edge_cov = _evidence_metrics(
        db, agent
    )
    authenticity_pass = bool(
        invalid_refs == 0 and invalid_obs_refs == 0 and invalid_source == 0
    )
    evidence_pass = bool(
        authenticity_pass and node_cov == 1.0 and ref_cov == 1.0 and edge_cov == 1.0
    )
    structural_pass = reconstruction_complete(comparison)
    protocol_pass = db.interview_complete

    result = PrimaryEvaluationResult(
        protocol_completed=db.interview_complete,
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
        node_evidence_coverage=node_cov,
        ref_evidence_coverage=ref_cov,
        edge_evidence_coverage=edge_cov,
        invalid_evidence_ref_count=invalid_refs,
        ambiguous_evidence_ref_count=0,
        marker_evidence_errors_surrogate=0,
        invalid_observation_reference_count=invalid_obs_refs,
        authentic_observation_count=len(authentic_ids),
        invalid_observation_source_count=invalid_source,
        orphan_observation_count=orphan_count,
        provenance_authenticity_pass=authenticity_pass,
        evidence_pass=evidence_pass,
        reconstruction_pass=structural_pass,
        structural_pass=structural_pass,
        protocol_pass=protocol_pass,
        quality_pass=structural_pass,
        knowledge_coverage=knowledge_coverage(truth, knowledge_for_terms),
    )

    return PrimaryEvaluation(
        result=result,
        agent=agent,
        truth=truth,
        target=target,
        knowledge=None,
        terminology_terms=terminology_terms,
        alignment=alignment,
        comparison=comparison,
    )


def evaluate(
    db: InterviewDB,
    knowledge: Any,
    spec: EvaluationSpec,
    stakeholder: Any = None,
    *,
    truth: Any,
    stakeholder_references: Optional[list[Any]] = None,
) -> EvaluationResult:
    """Evaluate one Agent graph against an explicit Truth graph.

    Builds the primary evaluation, then appends ordinary production diagnostics
    and Stakeholder-to-Truth reference evaluations. Usage and joint structural
    experiments run only in the offline tooling lane.
    """
    from .evaluation_diagnostics import (
        build_evaluation_diagnostics,
        canonical_contract_diagnostic,
    )

    if truth is None:
        raise ValueError("evaluate(): truth must be an explicit Truth graph")

    primary = _build_primary_result(db, truth, knowledge)

    # Reference evaluations
    from .reference_evaluation import (
        StakeholderReferenceInput,
        _forgetting_configuration,
        _knowledge_fallback_identity,
    )

    reference_evaluations: list[StakeholderTruthReferenceEvaluation] = []
    if stakeholder_references is not None:
        ref_inputs = normalize_stakeholder_references(stakeholder_references)
    elif knowledge is not None:
        # Current single-stakeholder input shape.
        fallback_id, fallback_name = _knowledge_fallback_identity(knowledge)
        name = getattr(stakeholder, "name", "") or fallback_name
        stakeholder_id = getattr(stakeholder, "stakeholder_id", None) or fallback_id
        ref_inputs = [
            StakeholderReferenceInput(
                stakeholder_id=str(stakeholder_id),
                stakeholder_name=str(name),
                stakeholder_role=getattr(stakeholder, "role", None),
                forgetting_configuration=_forgetting_configuration(stakeholder),
                knowledge=knowledge,
            )
        ]
    else:
        ref_inputs = []
    reference_evaluations = [
        evaluate_stakeholder_truth_reference(truth, ref) for ref in ref_inputs
    ]
    reference_aggregate = aggregate_stakeholder_truth_references(reference_evaluations)

    # Diagnostics (detailed trace, canonical contract only)
    canonical = canonical_contract_diagnostic(truth, knowledge)
    diagnostics = build_evaluation_diagnostics(
        primary.agent,
        primary.target,
        primary.terminology_terms,
        primary.alignment.concept_to_truth,
        primary.alignment.node_to_truth,
        primary.alignment.edge_to_truth,
    )
    diagnostics.canonical_contract = canonical

    result = primary.result
    return EvaluationResult(
        protocol_completed=result.protocol_completed,
        graph_created=result.graph_created,
        graph_valid=result.graph_valid,
        node_recall=result.node_recall,
        node_precision=result.node_precision,
        edge_recall=result.edge_recall,
        edge_precision=result.edge_precision,
        start_correct=result.start_correct,
        end_recall=result.end_recall,
        end_precision=result.end_precision,
        activity_correctness=result.activity_correctness,
        actor_correctness=result.actor_correctness,
        system_correctness=result.system_correctness,
        read_correctness=result.read_correctness,
        write_correctness=result.write_correctness,
        rationale_correctness=result.rationale_correctness,
        condition_correctness=result.condition_correctness,
        concept_correctness=result.concept_correctness,
        concept_recall=result.concept_recall,
        concept_precision=result.concept_precision,
        unsupported_ref_count=result.unsupported_ref_count,
        fabricated_node_count=result.fabricated_node_count,
        fabricated_edge_count=result.fabricated_edge_count,
        glossary_complete=result.glossary_complete,
        node_evidence_coverage=result.node_evidence_coverage,
        ref_evidence_coverage=result.ref_evidence_coverage,
        edge_evidence_coverage=result.edge_evidence_coverage,
        invalid_evidence_ref_count=result.invalid_evidence_ref_count,
        ambiguous_evidence_ref_count=result.ambiguous_evidence_ref_count,
        marker_evidence_errors_surrogate=result.marker_evidence_errors_surrogate,
        invalid_observation_reference_count=result.invalid_observation_reference_count,
        authentic_observation_count=result.authentic_observation_count,
        invalid_observation_source_count=result.invalid_observation_source_count,
        orphan_observation_count=result.orphan_observation_count,
        provenance_authenticity_pass=result.provenance_authenticity_pass,
        evidence_pass=result.evidence_pass,
        reconstruction_pass=result.reconstruction_pass,
        structural_pass=result.structural_pass,
        protocol_pass=result.protocol_pass,
        quality_pass=result.quality_pass,
        knowledge_coverage=result.knowledge_coverage,
        diagnostics=diagnostics,
        stakeholder_truth_reference=reference_evaluations,
        stakeholder_truth_reference_aggregate=reference_aggregate,
    )
