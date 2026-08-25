"""Deterministic tests for StakeholderKnowledge Truth reference scores.

The reference path is deliberately tested separately from Agent scoring: Truth
remains the primary target, while each stakeholder view is only an information
completeness diagnostic.
"""

from __future__ import annotations

import json

import pytest

from tau2.domains.business_interview.evaluation import (
    EvaluationResult,
    EvaluationSpec,
    evaluate,
)
from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentConcept,
    AgentGraph,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    InterviewDB,
    Node,
    TruthConcept,
    TruthEdge,
    TruthNode,
    business_edge_ids,
    business_node_ids,
    canonicalize_truth_graph,
    edge_is_structural,
    node_is_structural,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    project_knowledge,
    validate_stakeholder_knowledge_graph,
)
from tau2.domains.business_interview.scenario import Scenario, ScenarioStakeholder
from tau2.domains.business_interview.stakeholder import StakeholderFilter


def _truth() -> BusinessProcessGraph:
    concepts = {
        "act_a": TruthConcept(
            id="act_a", kind="activity", description="Do A", canonical_terms=["do A"]
        ),
        "act_b": TruthConcept(
            id="act_b", kind="activity", description="Do B", canonical_terms=["do B"]
        ),
        "act_c": TruthConcept(
            id="act_c", kind="activity", description="Do C", canonical_terms=["do C"]
        ),
        "actor": TruthConcept(
            id="actor",
            kind="actor",
            description="The operator",
            canonical_terms=["operator"],
        ),
        "system": TruthConcept(
            id="system",
            kind="system",
            description="The system",
            canonical_terms=["system"],
        ),
        "data_a": TruthConcept(
            id="data_a", kind="data", description="A data", canonical_terms=["A data"]
        ),
        "data_b": TruthConcept(
            id="data_b", kind="data", description="B data", canonical_terms=["B data"]
        ),
        "condition": TruthConcept(
            id="condition",
            kind="condition",
            description="When ready",
            canonical_terms=["ready"],
        ),
        "rationale": TruthConcept(
            id="rationale",
            kind="rationale",
            description="Business reason",
            canonical_terms=["business reason"],
        ),
    }
    truth = BusinessProcessGraph(
        id="reference_chain",
        name="Reference chain",
        concepts=concepts,
        nodes={
            "A": TruthNode(
                id="A",
                activity=ConceptRef(concept_id="act_a"),
                actor=ConceptRef(concept_id="actor"),
                system=ConceptRef(concept_id="system"),
                writes=[ConceptRef(concept_id="data_a")],
            ),
            "B": TruthNode(
                id="B",
                activity=ConceptRef(concept_id="act_b"),
                actor=ConceptRef(concept_id="actor"),
                system=ConceptRef(concept_id="system"),
                reads=[ConceptRef(concept_id="data_a")],
                writes=[ConceptRef(concept_id="data_b")],
                necessity_rationale=ConceptRef(concept_id="rationale"),
            ),
            "C": TruthNode(
                id="C",
                activity=ConceptRef(concept_id="act_c"),
                actor=ConceptRef(concept_id="actor"),
                system=ConceptRef(concept_id="system"),
                reads=[ConceptRef(concept_id="data_b")],
            ),
        },
        edges={
            "ab": TruthEdge(id="ab", from_node="A", to_node="B"),
            "bc": TruthEdge(id="bc", from_node="B", to_node="C"),
        },
        start_node_id="A",
        end_node_ids=["C"],
    )
    return canonicalize_truth_graph(truth, entry_node_ids=["A"], exit_node_ids=["C"])


_ALL_PROPS = ["activity", "actor", "system", "reads", "writes", "rationale"]


def _full_filter(truth: BusinessProcessGraph, *, name: str = "full", sid: str = "full"):
    return StakeholderFilter(
        name=name,
        stakeholder_id=sid,
        role=name,
        visible_node_ids=business_node_ids(truth),
        visible_edge_ids=business_edge_ids(truth),
        visible_node_attributes={
            node_id: list(_ALL_PROPS) for node_id in business_node_ids(truth)
        },
        visible_edge_attributes={
            edge_id: ["condition"] for edge_id in business_edge_ids(truth)
        },
    )


def _knowledge(truth: BusinessProcessGraph, filter_: StakeholderFilter):
    return project_knowledge(truth, filter_)


def _reference(result: EvaluationResult, stakeholder_id: str = "full"):
    return next(
        item
        for item in result.stakeholder_truth_reference
        if item.stakeholder_id == stakeholder_id
    )


def _agent_from_truth(truth: BusinessProcessGraph) -> AgentGraph:
    concepts = {
        f"agent_{concept_id}": AgentConcept(
            id=f"agent_{concept_id}",
            kind=concept.kind,
            display_label=(
                concept.canonical_terms or [concept.description or concept_id]
            )[0],
            description=concept.description,
        )
        for concept_id, concept in truth.concepts.items()
    }

    def ref(value):
        if isinstance(value, ConceptRef):
            return ConceptRef(concept_id=f"agent_{value.concept_id}")
        return AbsentType()

    def list_refs(values):
        return [ConceptRef(concept_id=f"agent_{value.concept_id}") for value in values]

    nodes = {
        node_id: Node(
            id=node_id,
            activity=ref(node.activity),
            actor=ref(node.actor),
            system=ref(node.system),
            reads=(list_refs(node.reads) if node.reads else AbsentType()),
            writes=(list_refs(node.writes) if node.writes else AbsentType()),
            necessity_rationale=ref(node.necessity_rationale),
        )
        for node_id, node in truth.nodes.items()
        if not node_is_structural(node)
    }
    edges = {
        edge_id: Edge(
            id=edge_id,
            from_node=edge.from_node,
            to_node=edge.to_node,
            condition=ref(edge.condition),
        )
        for edge_id, edge in truth.edges.items()
        if not edge_is_structural(edge)
    }
    return AgentGraph(
        concepts=concepts,
        nodes=nodes,
        edges=edges,
        start_node_id=truth.business_entry_node_ids[0],
        end_node_ids=list(truth.business_exit_node_ids),
    )


def _evaluate(
    truth: BusinessProcessGraph,
    knowledge: StakeholderKnowledge,
    filter_: StakeholderFilter,
    *,
    agent: AgentGraph | None = None,
    references=None,
):
    return evaluate(
        InterviewDB(graph=agent or AgentGraph()),
        knowledge,
        EvaluationSpec(),
        filter_,
        truth=truth,
        stakeholder_references=references,
    )


def test_no_forgetting_stakeholder_reference_is_exactly_one():
    truth = _truth()
    filter_ = _full_filter(truth)
    result = _evaluate(truth, _knowledge(truth, filter_), filter_)
    metrics = _reference(result).truth_reconstruction
    assert metrics.aggregate_score == pytest.approx(1.0)
    for field in (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "activity_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "rationale_correctness",
        "condition_correctness",
        "concept_recall",
        "concept_precision",
    ):
        assert getattr(metrics, field) == pytest.approx(1.0), field


def test_semantic_dont_know_lowers_reference_component_and_is_not_correct():
    truth = _truth()
    full = _full_filter(truth)
    attrs = {node_id: list(_ALL_PROPS) for node_id in business_node_ids(truth)}
    attrs["B"].remove("activity")
    partial = full.model_copy(update={"visible_node_attributes": attrs})
    result = _evaluate(truth, _knowledge(truth, partial), partial)
    metrics = _reference(result).truth_reconstruction
    assert metrics.activity_correctness < 1.0
    assert metrics.concept_recall < 1.0
    assert metrics.quality_pass is False


def test_serial_contraction_lowers_reference_and_keeps_shortcut_provenance():
    truth = _truth()
    partial = StakeholderFilter(
        name="forgetful",
        stakeholder_id="forgetful",
        visible_node_ids=["A", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": list(_ALL_PROPS), "C": list(_ALL_PROPS)},
        visible_edge_attributes={"ab": [], "bc": []},
    )
    knowledge = _knowledge(truth, partial)
    result = _evaluate(truth, knowledge, partial)
    reference = _reference(result, "forgetful")
    metrics = reference.truth_reconstruction
    assert metrics.node_recall == pytest.approx(2 / 3)
    assert metrics.edge_recall == pytest.approx(0.0)
    assert metrics.edge_precision == pytest.approx(0.0)
    assert metrics.aggregate_score < 1.0
    shortcuts = reference.diagnostics.shortcut_provenance
    assert len(shortcuts) == 1
    shortcut = shortcuts[0]
    assert shortcut["is_shortcut"] is True
    assert shortcut["contracted_nodes"] == ["B"]
    assert shortcut["derived_from_edges"] == ["ab", "bc"]
    assert shortcut["direct_business_edge_credit"] is False
    assert reference.diagnostics.contracted_node_count == 1
    assert reference.diagnostics.shortcut_edge_count == 1
    assert (
        reference.diagnostics.forgetting_configuration["allow_shortcut_contraction"]
        is True
    )
    assert (
        reference.diagnostics.canonical_contract["scoring_excludes_structural_elements"]
        is True
    )


def test_shortcut_never_receives_exact_credit_even_if_truth_has_direct_edge():
    original_truth = _truth()
    partial = StakeholderFilter(
        name="forgetful",
        stakeholder_id="forgetful",
        visible_node_ids=["A", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": list(_ALL_PROPS), "C": list(_ALL_PROPS)},
        visible_edge_attributes={"ab": [], "bc": []},
    )
    knowledge = _knowledge(original_truth, partial)
    truth_with_direct_edge = original_truth.model_copy(deep=True)
    truth_with_direct_edge.edges["ac"] = TruthEdge(id="ac", from_node="A", to_node="C")
    result = _evaluate(
        truth_with_direct_edge,
        knowledge,
        partial,
        references=[
            {
                "stakeholder_id": "forgetful",
                "stakeholder_name": "Forgetful",
                "knowledge": knowledge,
            }
        ],
    )
    reference = _reference(result, "forgetful")
    assert reference.truth_reconstruction.edge_recall == pytest.approx(0.0)
    assert reference.diagnostics.edge_alignment == {}
    assert reference.diagnostics.shortcut_provenance[0]["matched_truth_edge_id"] is None


def test_complete_multi_entry_stakeholder_reference_recovers_all_starts():
    truth = _truth()
    raw = truth.model_copy(deep=True)
    raw.nodes = {
        node_id: node
        for node_id, node in raw.nodes.items()
        if not node_is_structural(node)
    }
    raw.edges = {
        edge_id: edge
        for edge_id, edge in raw.edges.items()
        if not edge_is_structural(edge)
    }
    raw.nodes["D"] = TruthNode(
        id="D",
        activity=ConceptRef(concept_id="act_a"),
        actor=ConceptRef(concept_id="actor"),
        system=ConceptRef(concept_id="system"),
    )
    raw.edges["dc"] = TruthEdge(id="dc", from_node="D", to_node="C")
    multi_entry_truth = canonicalize_truth_graph(
        raw, entry_node_ids=["A", "D"], exit_node_ids=["C"]
    )
    filter_ = _full_filter(multi_entry_truth)
    knowledge = _knowledge(multi_entry_truth, filter_)
    assert knowledge.graph.start_node_id is None
    result = _evaluate(multi_entry_truth, knowledge, filter_)
    metrics = _reference(result).truth_reconstruction
    assert metrics.start_correct is True
    assert metrics.aggregate_score == pytest.approx(1.0)


def test_source_sink_and_boundary_edges_are_not_business_denominators():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    result = _evaluate(truth, knowledge, filter_)
    reference = _reference(result)
    contract = reference.diagnostics.canonical_contract
    assert contract["structural_node_count"] == 2
    assert contract["structural_edge_count"] == 2
    assert contract["business_node_count"] == 3
    assert contract["business_edge_count"] == 2
    assert reference.truth_reconstruction.node_recall == pytest.approx(1.0)
    assert reference.truth_reconstruction.edge_recall == pytest.approx(1.0)


def test_one_and_multiple_stakeholder_references_are_named_and_order_invariant():
    truth = _truth()
    full_filter = _full_filter(truth, name="Full", sid="stakeholder_full")
    forget_filter = StakeholderFilter(
        name="Forgetful",
        stakeholder_id="stakeholder_forgetful",
        visible_node_ids=["A", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": list(_ALL_PROPS), "C": list(_ALL_PROPS)},
        visible_edge_attributes={"ab": [], "bc": []},
    )
    full = _knowledge(truth, full_filter)
    forgetful = _knowledge(truth, forget_filter)
    one = _evaluate(truth, full, full_filter)
    assert len(one.stakeholder_truth_reference) == 1
    refs = [
        {
            "stakeholder_id": "stakeholder_forgetful",
            "stakeholder_name": "Forgetful",
            "stakeholder_role": "analyst",
            "knowledge": forgetful,
        },
        {
            "stakeholder_id": "stakeholder_full",
            "stakeholder_name": "Full",
            "stakeholder_role": "operator",
            "knowledge": full,
        },
    ]
    result = _evaluate(truth, full, full_filter, references=refs)
    assert [item.stakeholder_id for item in result.stakeholder_truth_reference] == [
        "stakeholder_forgetful",
        "stakeholder_full",
    ]
    assert result.stakeholder_truth_reference_aggregate.stakeholder_count == 2
    by_id = {item.stakeholder_id: item for item in result.stakeholder_truth_reference}
    assert by_id[
        "stakeholder_full"
    ].truth_reconstruction.aggregate_score == pytest.approx(1.0)
    reversed_result = _evaluate(
        truth, full, full_filter, references=list(reversed(refs))
    )
    assert {
        item.stakeholder_id: item.truth_reconstruction.aggregate_score
        for item in reversed_result.stakeholder_truth_reference
    } == {
        item.stakeholder_id: item.truth_reconstruction.aggregate_score
        for item in result.stakeholder_truth_reference
    }


def test_primary_agent_truth_score_is_unchanged_by_reference_metadata():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    agent = _agent_from_truth(truth)
    partial_filter = StakeholderFilter(
        name="partial",
        stakeholder_id="partial",
        visible_node_ids=business_node_ids(truth),
        visible_edge_ids=business_edge_ids(truth),
        visible_node_attributes={
            node_id: ["activity"] for node_id in business_node_ids(truth)
        },
        visible_edge_attributes={edge_id: [] for edge_id in business_edge_ids(truth)},
    )
    partial_knowledge = _knowledge(truth, partial_filter)
    without = _evaluate(truth, knowledge, filter_, agent=agent, references=[])
    with_reference = _evaluate(
        truth,
        knowledge,
        filter_,
        agent=agent,
        references=[
            {
                "stakeholder_id": "partial",
                "stakeholder_name": "Partial",
                "knowledge": partial_knowledge,
            }
        ],
    )
    primary_fields = (
        "protocol_completed",
        "graph_created",
        "graph_valid",
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "start_correct",
        "end_recall",
        "end_precision",
        "activity_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "rationale_correctness",
        "condition_correctness",
        "concept_correctness",
        "concept_recall",
        "concept_precision",
        "quality_pass",
        "structural_pass",
        "reconstruction_pass",
    )
    for field in primary_fields:
        assert getattr(with_reference, field) == getattr(without, field), field
    assert with_reference.quality_pass is True
    assert with_reference.stakeholder_truth_reference_aggregate.reference_only is True


def test_reference_quality_does_not_relax_primary_pass_fail():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    result = _evaluate(truth, knowledge, filter_, agent=AgentGraph())
    assert _reference(result).truth_reconstruction.quality_pass is True
    assert result.quality_pass is False
    assert result.reconstruction_pass is False


def test_reference_json_round_trip_preserves_identifiers_and_scores():
    truth = _truth()
    filter_ = _full_filter(truth)
    result = _evaluate(truth, _knowledge(truth, filter_), filter_)
    restored = EvaluationResult.model_validate_json(result.model_dump_json())
    assert restored.stakeholder_truth_reference[0].stakeholder_id == "full"
    assert restored.stakeholder_truth_reference[
        0
    ].truth_reconstruction.aggregate_score == pytest.approx(1.0)
    assert (
        restored.stakeholder_truth_reference_aggregate.model_dump()
        == result.stakeholder_truth_reference_aggregate.model_dump()
    )


def test_reference_and_primary_components_agree_for_no_forgetting_graph():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    result = _evaluate(truth, knowledge, filter_, agent=_agent_from_truth(truth))
    primary = result
    reference = _reference(result).truth_reconstruction
    for primary_field, reference_field in (
        ("node_recall", "node_recall"),
        ("node_precision", "node_precision"),
        ("edge_recall", "edge_recall"),
        ("edge_precision", "edge_precision"),
        ("start_correct", "start_correct"),
        ("end_recall", "end_recall"),
        ("end_precision", "end_precision"),
        ("activity_correctness", "activity_correctness"),
        ("actor_correctness", "actor_correctness"),
        ("system_correctness", "system_correctness"),
        ("read_correctness", "read_correctness"),
        ("write_correctness", "write_correctness"),
        ("rationale_correctness", "rationale_correctness"),
        ("condition_correctness", "condition_correctness"),
        ("concept_recall", "concept_recall"),
        ("concept_precision", "concept_precision"),
    ):
        assert getattr(primary, primary_field) == pytest.approx(
            getattr(reference, reference_field)
        )


def test_invalid_stakeholder_graph_is_rejected_by_canonical_validator():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    graph = knowledge.graph.model_copy(deep=True)
    structural_edge_id = next(
        edge_id for edge_id, edge in graph.edges.items() if edge_is_structural(edge)
    )
    graph.edges.pop(structural_edge_id)
    with pytest.raises(ValueError, match="Invalid stakeholder knowledge graph"):
        validate_stakeholder_knowledge_graph(graph)
    invalid_result = _evaluate(
        truth, knowledge.model_copy(update={"graph": graph}), filter_
    )
    assert _reference(invalid_result).truth_reconstruction.graph_valid is False
    assert invalid_result.stakeholder_truth_reference[0].diagnostics.validation_errors


def _rename_local_ids(knowledge: StakeholderKnowledge) -> StakeholderKnowledge:
    """Rename every non-structural local id while preserving Truth mappings."""
    graph = knowledge.graph
    node_ids = {
        old: f"opaque_node_{index}"
        for index, old in enumerate(sorted(business_node_ids(graph)), start=1)
    }
    edge_ids = {
        old: f"opaque_edge_{index}"
        for index, old in enumerate(sorted(business_edge_ids(graph)), start=1)
    }
    concept_ids = {
        old: f"opaque_concept_{index}"
        for index, old in enumerate(sorted(graph.concepts), start=1)
    }

    def rename_ref(value):
        if isinstance(value, ConceptRef):
            return value.model_copy(
                update={"concept_id": concept_ids[value.concept_id]}
            )
        if isinstance(value, list):
            return [rename_ref(item) for item in value]
        return value

    nodes = {}
    for old, node in graph.nodes.items():
        new_id = node_ids.get(old, old)
        nodes[new_id] = node.model_copy(
            update={
                "id": new_id,
                "activity": rename_ref(node.activity),
                "actor": rename_ref(node.actor),
                "system": rename_ref(node.system),
                "reads": rename_ref(node.reads),
                "writes": rename_ref(node.writes),
                "necessity_rationale": rename_ref(node.necessity_rationale),
            }
        )
    edges = {}
    for old, edge in graph.edges.items():
        edges[edge_ids.get(old, old)] = edge.model_copy(
            update={
                "id": edge_ids.get(old, old),
                "from_node": node_ids.get(edge.from_node, edge.from_node),
                "to_node": node_ids.get(edge.to_node, edge.to_node),
                "condition": rename_ref(edge.condition),
            }
        )
    concepts = {
        concept_ids[old]: concept.model_copy(update={"id": concept_ids[old]})
        for old, concept in graph.concepts.items()
    }
    renamed_start = (
        node_ids.get(graph.start_node_id) if graph.start_node_id is not None else None
    )
    renamed = graph.model_copy(
        update={
            "nodes": nodes,
            "edges": edges,
            "concepts": concepts,
            "node_truth_ids": {
                node_ids.get(local, local): truth_id
                for local, truth_id in graph.node_truth_ids.items()
            },
            "edge_truth_ids": {
                edge_ids.get(local, local): truth_id
                for local, truth_id in graph.edge_truth_ids.items()
            },
            "start_node_id": renamed_start,
            "end_node_ids": [
                node_ids.get(node_id, node_id) for node_id in graph.end_node_ids
            ],
            "shortcut_provenance": {
                edge_ids.get(local, local): value
                for local, value in graph.shortcut_provenance.items()
            },
        }
    )
    return knowledge.model_copy(update={"graph": renamed})


def test_opaque_local_ids_do_not_change_truth_mapping_or_reference_score():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    renamed = _rename_local_ids(knowledge)
    first = _reference(_evaluate(truth, knowledge, filter_)).truth_reconstruction
    second = _reference(_evaluate(truth, renamed, filter_)).truth_reconstruction
    for field in (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "activity_correctness",
        "read_correctness",
        "write_correctness",
        "condition_correctness",
        "concept_recall",
        "concept_precision",
        "aggregate_score",
    ):
        assert getattr(first, field) == pytest.approx(getattr(second, field)), field


def test_reference_output_makes_information_quality_interpretable():
    truth = _truth()
    full_filter = _full_filter(truth, name="S1", sid="s1")
    partial_filter = StakeholderFilter(
        name="S2",
        stakeholder_id="s2",
        visible_node_ids=["A", "B", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={
            "A": ["activity"],
            "B": ["activity"],
            "C": ["activity"],
        },
        visible_edge_attributes={"ab": [], "bc": []},
    )
    full = _knowledge(truth, full_filter)
    partial = _knowledge(truth, partial_filter)
    result = _evaluate(
        truth,
        full,
        full_filter,
        agent=AgentGraph(),
        references=[
            {"stakeholder_id": "s2", "stakeholder_name": "S2", "knowledge": partial},
            {"stakeholder_id": "s1", "stakeholder_name": "S1", "knowledge": full},
        ],
    )
    by_id = {item.stakeholder_id: item for item in result.stakeholder_truth_reference}
    assert (
        by_id["s1"].truth_reconstruction.aggregate_score
        > by_id["s2"].truth_reconstruction.aggregate_score
    )
    assert result.quality_pass is False
    assert (
        result.stakeholder_truth_reference_aggregate.mean_stakeholder_truth_score
        is not None
    )


# Keep the import checker from treating this test module as an untested schema
# fixture: all serialized references must be JSON-compatible.
def test_reference_model_dump_is_json_serializable():
    truth = _truth()
    filter_ = _full_filter(truth)
    result = _evaluate(truth, _knowledge(truth, filter_), filter_)
    json.dumps(result.model_dump(mode="json"))


def test_known_absence_scores_but_dont_know_does_not_score_absence():
    truth = _truth()
    full_filter = _full_filter(truth)
    full = _evaluate(truth, _knowledge(truth, full_filter), full_filter)
    attrs = {node_id: list(_ALL_PROPS) for node_id in business_node_ids(truth)}
    attrs["A"].remove("reads")
    unknown_filter = full_filter.model_copy(update={"visible_node_attributes": attrs})
    unknown = _evaluate(truth, _knowledge(truth, unknown_filter), unknown_filter)
    assert _reference(full).truth_reconstruction.read_correctness == pytest.approx(1.0)
    assert _reference(unknown).truth_reconstruction.read_correctness < 1.0


def test_shortcut_reference_diagnostics_do_not_mutate_truth_graph():
    truth = _truth()
    before = truth.model_dump(mode="json")
    partial = StakeholderFilter(
        name="forgetful",
        stakeholder_id="forgetful",
        visible_node_ids=["A", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": list(_ALL_PROPS), "C": list(_ALL_PROPS)},
        visible_edge_attributes={"ab": [], "bc": []},
    )
    _evaluate(truth, _knowledge(truth, partial), partial)
    assert truth.model_dump(mode="json") == before
    assert set(business_node_ids(truth)) == {"A", "B", "C"}
    assert set(business_edge_ids(truth)) == {"ab", "bc"}


def test_duplicate_stakeholder_ids_are_rejected_instead_of_being_positional():
    truth = _truth()
    filter_ = _full_filter(truth)
    knowledge = _knowledge(truth, filter_)
    refs = [
        {"stakeholder_id": "same", "stakeholder_name": "one", "knowledge": knowledge},
        {"stakeholder_id": "same", "stakeholder_name": "two", "knowledge": knowledge},
    ]
    with pytest.raises(ValueError, match="ids must be unique"):
        _evaluate(truth, knowledge, filter_, references=refs)


def test_scenario_multiple_stakeholder_profile_is_named_for_reference_evaluation():
    truth = _truth()
    filter_ = _full_filter(truth, name="Sales", sid="sales_1")
    knowledge = _knowledge(truth, filter_)
    profile = ScenarioStakeholder(
        stakeholder_id="sales_1",
        stakeholder=filter_,
        knowledge=knowledge,
        stakeholder_name="Sales",
        stakeholder_role="accounting",
    )
    scenario = Scenario(
        scenario_id="multi",
        truth=truth,
        stakeholder=filter_,
        knowledge=knowledge,
        stakeholders=(profile,),
    )
    assert scenario.stakeholder_references[0].stakeholder_id == "sales_1"
    result = _evaluate(
        truth,
        knowledge,
        filter_,
        references=list(scenario.stakeholder_references),
    )
    reference = _reference(result, "sales_1")
    assert reference.stakeholder_name == "Sales"
    assert reference.stakeholder_role == "accounting"


def test_good_agent_reconstruction_can_exceed_each_incomplete_stakeholder_view():
    truth = _truth()
    partial_filter = StakeholderFilter(
        name="partial",
        stakeholder_id="partial",
        visible_node_ids=business_node_ids(truth),
        visible_edge_ids=business_edge_ids(truth),
        visible_node_attributes={
            node_id: ["activity"] for node_id in business_node_ids(truth)
        },
        visible_edge_attributes={edge_id: [] for edge_id in business_edge_ids(truth)},
    )
    knowledge = _knowledge(truth, partial_filter)
    result = _evaluate(
        truth,
        knowledge,
        partial_filter,
        agent=_agent_from_truth(truth),
        references=[
            {
                "stakeholder_id": "partial",
                "stakeholder_name": "Partial",
                "knowledge": knowledge,
            }
        ],
    )
    assert result.quality_pass is True
    assert _reference(result, "partial").truth_reconstruction.aggregate_score < 1.0


def test_primary_result_has_no_agent_stakeholder_similarity_metric():
    assert "agent_stakeholder_similarity" not in EvaluationResult.model_fields
    assert "stakeholder_truth_reference" in EvaluationResult.model_fields
