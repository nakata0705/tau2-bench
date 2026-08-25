"""Deterministic tests for business_interview primary-input artifacts."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tau2.domains.business_interview.artifact_provenance import (
    build_evaluation_inputs,
    deserialize_evaluation_inputs,
    fingerprint,
    recompute_stakeholder_truth_reference,
    serialize_evaluation_inputs,
    serialize_stakeholder_knowledge,
    serialize_truth_graph,
)
from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    AgentGraph,
    BusinessProcessGraph,
    ConceptRef,
    InterviewDB,
    TruthConcept,
    TruthEdge,
    TruthNode,
    is_dont_know,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    project_knowledge,
)
from tau2.domains.business_interview.stakeholder import StakeholderFilter

_ALL_PROPERTIES = ["activity", "actor", "system", "reads", "writes", "rationale"]


def _truth() -> BusinessProcessGraph:
    raw = BusinessProcessGraph(
        id="artifact_chain",
        name="Artifact chain",
        concepts={
            "act_a": TruthConcept(
                id="act_a",
                kind="activity",
                description="Do A",
                canonical_terms=["do A"],
            ),
            "act_b": TruthConcept(
                id="act_b",
                kind="activity",
                description="Do B",
                canonical_terms=["do B"],
            ),
            "act_c": TruthConcept(
                id="act_c",
                kind="activity",
                description="Do C",
                canonical_terms=["do C"],
            ),
            "actor": TruthConcept(
                id="actor",
                kind="actor",
                description="Operator",
                canonical_terms=["operator"],
            ),
            "data": TruthConcept(
                id="data",
                kind="data",
                description="Payload",
                canonical_terms=["payload"],
            ),
        },
        nodes={
            "A": TruthNode(
                id="A",
                activity=ConceptRef(concept_id="act_a"),
                actor=ConceptRef(concept_id="actor"),
                writes=[ConceptRef(concept_id="data")],
            ),
            "B": TruthNode(
                id="B",
                activity=ConceptRef(concept_id="act_b"),
                actor=ConceptRef(concept_id="actor"),
                reads=[ConceptRef(concept_id="data")],
            ),
            "C": TruthNode(
                id="C",
                activity=ConceptRef(concept_id="act_c"),
                actor=ConceptRef(concept_id="actor"),
            ),
        },
        edges={
            "ab": TruthEdge(id="ab", from_node="A", to_node="B"),
            "bc": TruthEdge(id="bc", from_node="B", to_node="C"),
        },
    )
    from tau2.domains.business_interview.graph import canonicalize_truth_graph

    return canonicalize_truth_graph(raw, entry_node_ids=["A"], exit_node_ids=["C"])


def _filter(truth: BusinessProcessGraph) -> StakeholderFilter:
    return StakeholderFilter(
        name="Operations",
        stakeholder_id="operations",
        role="operator",
        visible_node_ids=["A", "C"],
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": ["activity"], "C": ["activity"]},
        visible_edge_attributes={"ab": [], "bc": []},
    )


def _reference_payload(profile):
    return {
        "stakeholder_id": profile.stakeholder_id,
        "stakeholder_name": profile.stakeholder_name,
        "stakeholder_role": profile.stakeholder_role,
        "forgetting_configuration": profile.forgetting_configuration,
        "knowledge": profile.knowledge,
    }


def test_primary_inputs_store_full_truth_knowledge_seed_and_provenance():
    truth = _truth()
    filter_ = _filter(truth)
    knowledge = project_knowledge(truth, filter_, seed=44)
    inputs = build_evaluation_inputs(
        truth,
        [
            {
                "stakeholder_id": "operations",
                "stakeholder_name": "Operations",
                "stakeholder_role": "operator",
                "stakeholder": filter_,
                "knowledge": knowledge,
                "seed_provenance": {
                    "stakeholder_generation_seed": 101,
                    "forgetting_seed": 202,
                },
            }
        ],
        simulation_seed=9002,
    )

    assert inputs.stakeholders[0].knowledge is knowledge
    payload = serialize_evaluation_inputs(inputs)
    assert payload["seed"] == 9002
    assert payload["seed_provenance"]["simulation_seed"] == 9002
    assert payload["stakeholders"][0]["stakeholder_id"] == "operations"
    assert (
        payload["stakeholders"][0]["seed_provenance"]["stakeholder_generation_seed"]
        == 101
    )
    assert payload["stakeholders"][0]["seed_provenance"]["forgetting_seed"] == 202

    truth_payload = payload["truth_graph"]
    assert truth_payload["source_node_id"]
    assert truth_payload["sink_node_id"]
    assert set(truth_payload["nodes"]) == set(truth.nodes)
    assert set(truth_payload["edges"]) == set(truth.edges)
    assert set(truth_payload["concepts"]) == set(truth.concepts)
    assert any(
        edge["structural_only"] and edge["protected"]
        for edge in truth_payload["edges"].values()
    )

    knowledge_payload = payload["stakeholders"][0]["knowledge"]["graph"]
    assert knowledge_payload["source_node_id"]
    assert knowledge_payload["sink_node_id"]
    assert knowledge_payload["node_truth_ids"]
    assert knowledge_payload["edge_truth_ids"]
    assert knowledge_payload["shortcut_provenance"]
    assert any(
        node.get("structural_role") == "source"
        for node in knowledge_payload["nodes"].values()
    )
    shortcut = next(
        edge for edge in knowledge_payload["edges"].values() if edge["is_shortcut"]
    )
    assert shortcut["contracted_nodes"] == ["B"]
    assert shortcut["derived_from_edges"] == ["ab", "bc"]
    assert any(
        is_dont_know(node.actor)
        for node in knowledge.graph.nodes.values()
        if not node.is_structural
    )


def test_round_trip_preserves_states_boundaries_shortcuts_and_reference_metrics():
    truth = _truth()
    filter_ = _filter(truth)
    knowledge = project_knowledge(truth, filter_, seed=44)
    inputs = build_evaluation_inputs(
        truth,
        [
            {
                "stakeholder_id": "operations",
                "stakeholder": filter_,
                "knowledge": knowledge,
            }
        ],
        simulation_seed=9002,
    )
    restored = deserialize_evaluation_inputs(serialize_evaluation_inputs(inputs))
    restored_knowledge = restored.stakeholders[0].knowledge

    assert serialize_truth_graph(restored.truth_graph) == serialize_truth_graph(truth)
    assert serialize_stakeholder_knowledge(
        restored_knowledge
    ) == serialize_stakeholder_knowledge(knowledge)
    assert any(
        is_dont_know(node.actor)
        for node in restored_knowledge.graph.nodes.values()
        if not node.is_structural
    )
    assert any(
        edge.is_shortcut and edge.contracted_nodes and edge.derived_from_edges
        for edge in restored_knowledge.graph.edges.values()
    )
    assert all(
        edge.protected and edge.structural_only
        for edge in restored_knowledge.graph.edges.values()
        if edge.is_structural
    )

    original_result = evaluate(
        InterviewDB(graph=AgentGraph()),
        knowledge,
        EvaluationSpec(),
        truth=truth,
        stakeholder_references=[_reference_payload(inputs.stakeholders[0])],
    )
    restored_result = evaluate(
        InterviewDB(graph=AgentGraph()),
        restored_knowledge,
        EvaluationSpec(),
        truth=restored.truth_graph,
        stakeholder_references=[_reference_payload(restored.stakeholders[0])],
    )
    original_metrics = original_result.stakeholder_truth_reference[
        0
    ].truth_reconstruction
    restored_metrics = restored_result.stakeholder_truth_reference[
        0
    ].truth_reconstruction
    assert restored_metrics.model_dump(mode="json") == original_metrics.model_dump(
        mode="json"
    )
    assert restored_result.stakeholder_truth_reference[0].diagnostics.model_dump(
        mode="json"
    ) == original_result.stakeholder_truth_reference[0].diagnostics.model_dump(
        mode="json"
    )

    serialized_inputs = serialize_evaluation_inputs(inputs)
    recomputed = recompute_stakeholder_truth_reference(
        {"evaluation_inputs": serialized_inputs}
    )
    info_recomputed = recompute_stakeholder_truth_reference(
        {"info": {"evaluation_inputs": serialized_inputs}}
    )
    assert (
        info_recomputed["stakeholder_truth_reference_aggregate"]
        == recomputed["stakeholder_truth_reference_aggregate"]
    )
    assert recomputed["stakeholder_truth_reference"][0]["truth_reconstruction"] == (
        original_metrics.model_dump(mode="json")
    )
    assert recomputed["stakeholder_truth_reference_aggregate"][
        "mean_stakeholder_truth_score"
    ] == pytest.approx(original_metrics.aggregate_score)


def test_fingerprints_ignore_mapping_and_unordered_collection_insertion_order():
    truth = _truth()
    knowledge = project_knowledge(truth, _filter(truth), seed=44)
    truth_reordered = truth.model_copy(deep=True)
    truth_reordered.nodes = dict(reversed(list(truth_reordered.nodes.items())))
    truth_reordered.edges = dict(reversed(list(truth_reordered.edges.items())))
    truth_reordered.concepts = dict(reversed(list(truth_reordered.concepts.items())))
    knowledge_reordered = knowledge.model_copy(deep=True)
    graph = knowledge_reordered.graph
    graph.nodes = dict(reversed(list(graph.nodes.items())))
    graph.edges = dict(reversed(list(graph.edges.items())))
    graph.concepts = dict(reversed(list(graph.concepts.items())))
    graph.node_truth_ids = dict(reversed(list(graph.node_truth_ids.items())))
    graph.edge_truth_ids = dict(reversed(list(graph.edge_truth_ids.items())))
    graph.shortcut_provenance = dict(reversed(list(graph.shortcut_provenance.items())))

    assert fingerprint(serialize_truth_graph(truth_reordered)) == fingerprint(
        serialize_truth_graph(truth)
    )
    assert fingerprint(
        serialize_stakeholder_knowledge(knowledge_reordered)
    ) == fingerprint(serialize_stakeholder_knowledge(knowledge))


def test_stakeholder_order_is_identifier_based_and_separate_seeds_are_saved():
    truth = _truth()
    filter_ = _filter(truth)
    knowledge = project_knowledge(truth, filter_, seed=44)
    first = {
        "stakeholder_id": "zeta",
        "stakeholder_name": "Zeta",
        "knowledge": knowledge,
        "seed_provenance": {"forgetting_seed": 501},
    }
    second = {
        "stakeholder_id": "alpha",
        "stakeholder_name": "Alpha",
        "knowledge": knowledge,
        "seed_provenance": {"forgetting_seed": 502},
    }
    left = build_evaluation_inputs(truth, [first, second], simulation_seed=9002)
    right = build_evaluation_inputs(truth, [second, first], simulation_seed=9002)
    left_result = recompute_stakeholder_truth_reference(left.model_dump(mode="json"))
    right_result = recompute_stakeholder_truth_reference(right.model_dump(mode="json"))
    left_scores = {
        item["stakeholder_id"]: item["truth_reconstruction"]["aggregate_score"]
        for item in left_result["stakeholder_truth_reference"]
    }
    right_scores = {
        item["stakeholder_id"]: item["truth_reconstruction"]["aggregate_score"]
        for item in right_result["stakeholder_truth_reference"]
    }
    assert left_scores == right_scores
    by_id = left.stakeholder_by_id()
    assert by_id["zeta"].seed_provenance.forgetting_seed == 501
    assert by_id["alpha"].seed_provenance.forgetting_seed == 502
    assert left.seed_provenance.simulation_seed == 9002
    assert left.seed_provenance.simulation_seed_drives_stakeholder_knowledge is False


def test_simulator_knowledge_block_does_not_expose_private_truth_mapping():
    from tau2.domains.business_interview.scenario import get_scenario
    from tau2.domains.business_interview.user_simulator import StakeholderUserSimulator

    scenario = get_scenario("quotation_workflow_1")
    assert scenario is not None
    simulator = object.__new__(StakeholderUserSimulator)
    simulator._scenario = scenario
    block = simulator._knowledge_block()
    for truth_id in scenario.knowledge.graph.node_truth_ids.values():
        if truth_id not in {
            scenario.knowledge.graph.source_node_id,
            scenario.knowledge.graph.sink_node_id,
        }:
            assert f"node:{truth_id}" not in block
    for truth_id in scenario.knowledge.graph.edge_truth_ids.values():
        if not truth_id.startswith("__tau2_structural_boundary__"):
            assert f"edge:{truth_id}" not in block
    for concept in scenario.knowledge.graph.concepts.values():
        assert concept.truth_concept_id not in block
    assert "node_truth_ids" not in block
    assert "edge_truth_ids" not in block
    assert "shortcut_provenance" not in block


def test_knowledge_serializer_rejects_invalid_noncanonical_graph():
    truth = _truth()
    knowledge = project_knowledge(truth, _filter(truth), seed=44)
    invalid = deepcopy(knowledge.model_dump(mode="json"))
    invalid["graph"]["nodes"].pop("__tau2_structural_source__")
    with pytest.raises(ValueError, match="not canonical"):
        serialize_stakeholder_knowledge(StakeholderKnowledge.model_validate(invalid))
