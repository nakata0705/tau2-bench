from __future__ import annotations

from pathlib import Path

from tau2.domains.business_interview.boundary_diagnostics import (  # pyright: ignore[reportMissingImports]
    STRUCTURAL_END_NODE_ID,
    STRUCTURAL_START_NODE_ID,
    audit_boundary,
    deterministic_fixture_pairs,
    deterministic_invalid_fixture_pairs,
    load_saved_seed_pair,
    normalize_graph,
    run_alignment_experiment,
    trace_boundary_generation,
)


def _fixture(name: str):
    return next(item for item in deterministic_fixture_pairs() if item[0] == name)


def _invalid_fixture(name: str):
    return next(
        item for item in deterministic_invalid_fixture_pairs() if item[0] == name
    )


def test_multiple_apparent_starts_normalize_to_one_structural_start():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    before = audit_boundary(agent)

    result = normalize_graph(agent)

    assert result.possible
    assert len(before["topology_sources"]) == 2
    assert result.canonical is not None
    after = audit_boundary(result.canonical.graph)
    assert after["topology_sources"] == [STRUCTURAL_START_NODE_ID]
    assert result.canonical.invariant["exactly_one_structural_start"]
    assert (
        len(
            [
                edge
                for edge in result.canonical.synthetic_edges
                if edge["from_node"] == STRUCTURAL_START_NODE_ID
            ]
        )
        == 2
    )


def test_multiple_apparent_ends_normalize_to_one_structural_end():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    before = audit_boundary(truth)

    result = normalize_graph(truth)

    assert result.possible
    assert len(before["topology_sinks"]) == 2
    assert result.canonical is not None
    after = audit_boundary(result.canonical.graph)
    assert after["topology_sinks"] == [STRUCTURAL_END_NODE_ID]
    assert result.canonical.invariant["exactly_one_structural_end"]
    assert (
        len(
            [
                edge
                for edge in result.canonical.synthetic_edges
                if edge["to_node"] == STRUCTURAL_END_NODE_ID
            ]
        )
        == 2
    )


def test_normalization_does_not_mutate_original_or_use_labels_or_ids():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    original_agent_nodes = tuple(sorted(agent.nodes))
    original_agent_edges = tuple(sorted(agent.edges))

    agent_result = normalize_graph(agent)
    truth_result = normalize_graph(truth)

    assert tuple(sorted(agent.nodes)) == original_agent_nodes
    assert tuple(sorted(agent.edges)) == original_agent_edges
    assert agent_result.canonical is not None
    assert truth_result.canonical is not None
    assert agent_result.canonical.graph.concepts.keys() == agent.concepts.keys()
    assert truth_result.canonical.graph.concepts.keys() == truth.concepts.keys()
    assert STRUCTURAL_START_NODE_ID not in agent_result.canonical.graph.concepts
    assert STRUCTURAL_END_NODE_ID not in truth_result.canonical.graph.concepts


def test_insertion_order_does_not_change_canonical_boundary_copy():
    _name, _truth, agent = _fixture("multiple_entry_and_exit")
    reversed_agent = agent.model_copy(
        update={
            "nodes": dict(reversed(list(agent.nodes.items()))),
            "edges": dict(reversed(list(agent.edges.items()))),
            "concepts": dict(reversed(list(agent.concepts.items()))),
        },
        deep=True,
    )

    baseline = normalize_graph(agent)
    reversed_result = normalize_graph(reversed_agent)

    assert baseline.canonical is not None
    assert reversed_result.canonical is not None
    assert baseline.canonical.to_dict() == reversed_result.canonical.to_dict()


def test_normalized_graph_keeps_all_business_nodes_reachable_from_both_anchors():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    for graph in (agent, truth):
        result = normalize_graph(graph)
        assert result.canonical is not None
        invariant = result.canonical.invariant
        assert invariant["all_business_nodes_start_reachable"]
        assert invariant["all_business_nodes_end_reachable"]
        assert invariant["structural_start_has_no_incoming"]
        assert invariant["structural_end_has_no_outgoing"]


def test_synthetic_edges_are_not_business_process_edges_or_concepts():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    result = run_alignment_experiment(agent, truth, name="fixture")
    variant = next(
        item
        for item in result["alignment_variants"]
        if item["name"] == "D_virtual_START_and_END"
    )

    assert variant["process_topology_agreement"]["synthetic_edges_excluded"]
    assert (
        variant["objective_component_policy"]["synthetic_boundary_edges_included"]
        is False
    )
    assert variant["process_topology_agreement"]["agent_business_edge_count"] == len(
        agent.edges
    )
    assert variant["process_topology_agreement"]["truth_business_edge_count"] == len(
        truth.edges
    )


def test_start_end_anchors_reduce_search_without_erasing_symmetric_ambiguity():
    _name, truth, agent = _fixture("symmetric_entry_exit_roles")
    result = run_alignment_experiment(agent, truth, name="symmetric")
    current, normalized = (
        result["alignment_variants"][0],
        result["alignment_variants"][-1],
    )

    assert (
        normalized["search"]["node_candidate_pair_count"]
        < current["search"]["node_candidate_pair_count"]
    )
    assert (
        normalized["search"]["node_search_states"]
        < current["search"]["node_search_states"]
    )
    assert not normalized["search"]["optimum_is_unique"]
    assert normalized["ambiguity_summary"]["node_class_count"] > 0
    assert normalized["ambiguity_summary"]["concept_class_count"] > 0


def test_renamed_ids_have_same_role_constraints_and_anchor_ids():
    _name, truth, agent = _fixture("multiple_entry_and_exit")
    result = run_alignment_experiment(agent, truth, name="renamed")
    variant = result["alignment_variants"][-1]

    assert variant["fixed_structural_anchors"]["start"]["agent"] == (
        STRUCTURAL_START_NODE_ID
    )
    assert variant["fixed_structural_anchors"]["end"]["truth"] == (
        STRUCTURAL_END_NODE_ID
    )
    assert variant["candidate_policy"].startswith("topology source/sink")


def test_disconnected_graph_is_not_collapsed_into_one_process():
    _name, truth, agent = _invalid_fixture("disconnected_component")
    result = normalize_graph(agent)

    assert not result.possible
    assert result.classification == "invalid_disconnected_component"
    assert normalize_graph(truth).classification == result.classification


def test_multiple_independent_processes_require_separate_representation():
    _name, truth, agent = _invalid_fixture("multiple_independent_processes")
    result = normalize_graph(agent)

    assert not result.possible
    assert result.classification == "multiple_independent_processes"
    assert normalize_graph(truth).classification == result.classification


def test_cycles_and_self_loops_are_preserved_when_source_and_sink_exist():
    cycle = normalize_graph(_invalid_fixture("cycle")[2])
    self_loop = normalize_graph(_invalid_fixture("self_loop")[2])

    assert cycle.possible
    assert cycle.classification == "normalizable_with_cycle"
    assert self_loop.possible
    assert self_loop.classification == "normalizable_with_self_loop"


def test_missing_source_sink_is_invalid():
    source_missing = normalize_graph(_invalid_fixture("source_missing")[2])
    sink_missing = normalize_graph(_invalid_fixture("sink_missing")[2])

    assert not source_missing.possible
    assert source_missing.classification == "invalid_source_missing"
    assert not sink_missing.possible
    assert sink_missing.classification == "invalid_sink_missing"


def test_reserved_existing_structural_boundary_ids_are_rejected():
    result = normalize_graph(
        _invalid_fixture("existing_explicit_structural_boundary_ids")[2]
    )

    assert not result.possible
    assert result.classification == "invalid_reserved_boundary_id_collision"


def test_seed_9003_metadata_omission_is_visible_from_saved_protocol_trace():
    artifact_dir = Path("artifacts/business_interview_real_llm")
    public, agent, truth, _observations = load_saved_seed_pair(artifact_dir, 9003)
    agent_audit = audit_boundary(agent)
    truth_audit = audit_boundary(truth)
    trace = trace_boundary_generation(public)

    assert agent_audit["declared_start_node"] is None
    assert agent_audit["declared_end_nodes"] == []
    assert agent_audit["topology_sources"] == ["node_receive_request"]
    assert set(agent_audit["topology_sinks"]) == {
        "node_send_quotation",
        "node_send_summary",
    }
    assert truth_audit["declared_start_node"] == "r"
    assert set(truth_audit["declared_end_nodes"]) == {"sq", "me"}
    assert trace["endpoint_declaration_call_count"] == 0
    assert trace["termination_reason"] == "max_steps"
    assert trace["serialized_start_node"] is None
    assert trace["serialized_end_nodes"] == []
