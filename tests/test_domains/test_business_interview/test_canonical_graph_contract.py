"""Deterministic tests for the canonical Truth/stakeholder graph contract."""

from __future__ import annotations

import pytest

from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    STRUCTURAL_SINK_ID,
    STRUCTURAL_SOURCE_ID,
    AgentGraph,
    BusinessProcessGraph,
    ConceptRef,
    InterviewDB,
    TruthConcept,
    TruthEdge,
    TruthNode,
    business_edge_ids,
    business_node_ids,
    canonical_structure_errors,
    canonicalize_truth_graph,
    edge_is_structural,
    is_dont_know,
    validate_canonical_graph,
)
from tau2.domains.business_interview.knowledge import (
    KnowledgeProjectionError,
    contract_serial_node,
    project_knowledge,
)
from tau2.domains.business_interview.stakeholder import (
    StakeholderFilter,
    StakeholderForgettingConfig,
)


def _truth(node_ids: list[str], edge_pairs: list[tuple[str, str]], *, conditions=None):
    conditions = conditions or {}
    concepts = {
        node_id: TruthConcept(
            id=node_id,
            kind="activity",
            canonical_terms=[f"activity {node_id}"],
        )
        for node_id in node_ids
    }
    nodes = {
        node_id: TruthNode(
            id=node_id,
            activity=ConceptRef(concept_id=node_id),
        )
        for node_id in node_ids
    }
    edges = {
        f"e{index}": TruthEdge(
            id=f"e{index}",
            from_node=source,
            to_node=target,
            condition=conditions.get((source, target)),
        )
        for index, (source, target) in enumerate(edge_pairs, 1)
    }
    return BusinessProcessGraph(
        id="fixture",
        nodes=nodes,
        edges=edges,
        concepts=concepts,
    )


def _full_filter(
    truth, *, visible_nodes=None, visible_edges=None, condition_known=True
):
    nodes = visible_nodes or list(business_node_ids(truth))
    edges = visible_edges or list(business_edge_ids(truth))
    props = {node_id: ["activity"] for node_id in nodes}
    edge_props = {
        edge_id: ["condition"] if condition_known else [] for edge_id in edges
    }
    return StakeholderFilter(
        name="fixture stakeholder",
        visible_node_ids=nodes,
        visible_edge_ids=edges,
        visible_node_attributes=props,
        visible_edge_attributes=edge_props,
    )


def test_truth_has_exactly_one_explicit_source_and_sink():
    truth = canonicalize_truth_graph(
        _truth(["A", "B"], [("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["B"],
    )
    assert canonical_structure_errors(truth) == []
    assert truth.source_node_id == STRUCTURAL_SOURCE_ID
    assert truth.sink_node_id == STRUCTURAL_SINK_ID
    assert [
        node_id
        for node_id, node in truth.nodes.items()
        if node.structural_role == "source"
    ] == [STRUCTURAL_SOURCE_ID]
    assert [
        node_id
        for node_id, node in truth.nodes.items()
        if node.structural_role == "sink"
    ] == [STRUCTURAL_SINK_ID]
    assert len(truth.successors(STRUCTURAL_SOURCE_ID)) == 1
    assert truth.incoming_edges(STRUCTURAL_SOURCE_ID) == []
    assert truth.successors(STRUCTURAL_SINK_ID) == []


def test_multiple_entries_and_exits_are_boundary_fanout_fanin():
    truth = canonicalize_truth_graph(
        _truth(
            ["A", "B", "C", "D", "E"],
            [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")],
        ),
        entry_node_ids=["A", "B"],
        exit_node_ids=["D", "E"],
    )
    assert canonical_structure_errors(truth) == []
    assert set(truth.successors(STRUCTURAL_SOURCE_ID)) == {"A", "B"}
    assert {
        edge.from_node
        for edge in truth.edges.values()
        if edge.to_node == STRUCTURAL_SINK_ID
    } == {"D", "E"}
    assert all(truth.successors(node_id) for node_id in ("D", "E"))


def test_structural_elements_are_excluded_from_business_projection_and_score():
    truth = canonicalize_truth_graph(
        _truth(["A", "B"], [("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["B"],
    )
    knowledge = project_knowledge(truth, _full_filter(truth))
    result = evaluate(
        InterviewDB(graph=AgentGraph()),
        knowledge,
        EvaluationSpec(),
        truth=truth,
    )
    assert len(business_node_ids(truth)) == 2
    assert len(business_edge_ids(truth)) == 1
    assert result.diagnostics.canonical_contract["structural_node_count"] == 2
    assert result.diagnostics.canonical_contract["structural_edge_count"] == 2
    assert result.diagnostics.canonical_contract["business_node_count"] == 2
    assert result.diagnostics.canonical_contract["business_edge_count"] == 1


def test_structural_nodes_and_boundary_edges_are_always_protected():
    truth = canonicalize_truth_graph(
        _truth(["A", "B"], [("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["B"],
    )
    knowledge = project_knowledge(
        truth,
        StakeholderFilter(
            name="semantic-only",
            visible_node_ids=["A", "B"],
            visible_edge_ids=["e1"],
            visible_node_attributes={},
            visible_edge_attributes={},
            forgetting=StakeholderForgettingConfig(property_forget_probability=1.0),
        ),
    )
    graph = knowledge.graph
    assert graph.is_valid
    assert graph.nodes[STRUCTURAL_SOURCE_ID].protected
    assert graph.nodes[STRUCTURAL_SINK_ID].protected
    assert all(edge.protected for edge in graph.edges.values() if edge.structural_only)
    assert all(
        edge.condition is None for edge in graph.edges.values() if edge.structural_only
    )


def test_semantic_forgetting_keeps_topology_intact():
    truth = canonicalize_truth_graph(
        _truth(["A", "B"], [("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["B"],
    )
    knowledge = project_knowledge(
        truth,
        StakeholderFilter(
            name="forget semantics",
            visible_node_ids=["A", "B"],
            visible_edge_ids=["e1"],
            visible_node_attributes={"A": ["activity"], "B": ["activity"]},
            visible_edge_attributes={"e1": ["condition"]},
            forgetting=StakeholderForgettingConfig(property_forget_probability=1.0),
        ),
    )
    assert knowledge.graph.is_valid
    assert all(
        getattr(node, "activity") is not None
        for node in knowledge.graph.nodes.values()
        if not node.is_structural
    )
    assert all(
        getattr(node, "activity").__class__.__name__ == "DontKnowType"
        for node in knowledge.graph.nodes.values()
        if not node.is_structural
    )
    assert all(
        is_dont_know(edge.condition)
        for edge in knowledge.graph.edges.values()
        if not edge_is_structural(edge)
    )


def test_boundary_node_forgetting_rebuilds_only_a_protected_boundary():
    truth = canonicalize_truth_graph(
        _truth(["A", "B", "C"], [("A", "B"), ("B", "C")]),
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    graph = project_knowledge(
        truth,
        _full_filter(truth, visible_nodes=["B", "C"]),
    ).graph
    assert graph.is_valid
    shortcuts = [edge for edge in graph.edges.values() if edge.is_shortcut]
    assert len(shortcuts) == 1
    assert shortcuts[0].structural_only is True
    assert shortcuts[0].protected is True
    assert shortcuts[0].from_node == STRUCTURAL_SOURCE_ID
    assert shortcuts[0].to_node != STRUCTURAL_SINK_ID


def test_serial_forgetting_contracts_node_and_retains_provenance():
    truth = canonicalize_truth_graph(
        _truth(["A", "B", "C"], [("A", "B"), ("B", "C")]),
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    filt = _full_filter(truth, visible_nodes=["A", "C"])
    knowledge = project_knowledge(truth, filt)
    graph = knowledge.graph
    assert graph.is_valid
    assert len([node for node in graph.nodes.values() if not node.is_structural]) == 2
    shortcuts = [edge for edge in graph.edges.values() if edge.is_shortcut]
    assert len(shortcuts) == 1
    assert shortcuts[0].contracted_nodes == ["B"]
    assert shortcuts[0].derived_from_edges == ["e1", "e2"]
    assert list(graph.shortcut_provenance.values())[0]["is_shortcut"] is True


def test_branch_node_is_rejected_instead_of_repaired():
    truth = canonicalize_truth_graph(
        _truth(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]),
        entry_node_ids=["A"],
        exit_node_ids=["D"],
    )
    filt = _full_filter(truth, visible_nodes=["B", "C", "D"])
    with pytest.raises(KnowledgeProjectionError, match="indegree=1, outdegree=2"):
        project_knowledge(truth, filt, max_retries=2)


def test_conditioned_path_is_not_contracted_without_condition_composition():
    condition = ConceptRef(concept_id="condition")
    truth_base = _truth(
        ["A", "B", "C"], [("A", "B"), ("B", "C")], conditions={("A", "B"): condition}
    )
    truth_base.concepts["condition"] = TruthConcept(
        id="condition", kind="condition", canonical_terms=["condition"]
    )
    truth = canonicalize_truth_graph(
        truth_base,
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    filt = _full_filter(truth, visible_nodes=["A", "C"])
    with pytest.raises(KnowledgeProjectionError, match="unsafe incoming condition"):
        project_knowledge(truth, filt, max_retries=2)


def test_public_contraction_rejects_protected_or_unknown_condition_paths():
    truth = canonicalize_truth_graph(
        _truth(["A", "B", "C"], [("A", "B"), ("B", "C")]),
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    graph = project_knowledge(truth, _full_filter(truth)).graph
    with pytest.raises(ValueError, match="protected"):
        contract_serial_node(graph, STRUCTURAL_SOURCE_ID)


def test_max_retry_failure_reports_configuration_and_validation_reason():
    truth = canonicalize_truth_graph(
        _truth(["A", "B", "C", "D"], [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]),
        entry_node_ids=["A"],
        exit_node_ids=["D"],
    )
    filt = _full_filter(truth, visible_nodes=["B", "C", "D"])
    with pytest.raises(KnowledgeProjectionError) as exc_info:
        project_knowledge(truth, filt, max_retries=1)
    message = str(exc_info.value)
    assert "max_retries" in message or "configuration" in message
    assert "indegree" in message
    assert exc_info.value.attempts == 1


def test_opaque_ids_are_insertion_order_invariant():
    first = canonicalize_truth_graph(
        _truth(["A", "B", "C"], [("A", "B"), ("B", "C")]),
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    second = canonicalize_truth_graph(
        _truth(["C", "B", "A"], [("B", "C"), ("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["C"],
    )
    f1 = _full_filter(first)
    f2 = _full_filter(second)
    k1 = project_knowledge(first, f1).graph
    k2 = project_knowledge(second, f2).graph
    assert sorted(k1.nodes) == sorted(k2.nodes)
    assert sorted(k1.edges) == sorted(k2.edges)
    assert [k1.node_truth_ids[n] for n in sorted(k1.node_truth_ids)] == [
        k2.node_truth_ids[n] for n in sorted(k2.node_truth_ids)
    ]


def test_canonicalization_preserves_an_existing_boundary_contract():
    truth = canonicalize_truth_graph(
        _truth(["A", "B"], [("A", "B")]),
        entry_node_ids=["A"],
        exit_node_ids=["B"],
    )
    again = canonicalize_truth_graph(truth)
    assert canonical_structure_errors(again) == []
    assert again.nodes.keys() == truth.nodes.keys()
    assert again.edges.keys() == truth.edges.keys()
    assert again.business_entry_node_ids == ("A",)
    assert again.business_exit_node_ids == ("B",)


def test_invalid_canonical_graph_is_rejected_by_validator_and_projection():
    invalid = _truth(["A", "B"], [("A", "B")])
    with pytest.raises(ValueError, match="canonical graph"):
        validate_canonical_graph(invalid)
    with pytest.raises(ValueError, match="not canonical"):
        project_knowledge(invalid, _full_filter(invalid))
