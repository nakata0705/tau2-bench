"""Deterministic tests for the label-independent joint graph matcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.business_interview import (
    joint_structural_alignment as joint_alignment_module,
)
from experiments.business_interview.joint_structural_alignment import (
    build_joint_structural_alignment_diagnostics,
)
from scripts.business_interview_evaluation_diagnostics import evaluate_artifact
from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    DONT_KNOW,
    UNSET,
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
)


def _ref(concept_id: str) -> ConceptRef:
    return ConceptRef(concept_id=concept_id)


def _synthetic_graphs(
    *,
    agent_label_prefix: str = "agent",
    reverse_insertion: bool = False,
    renamed_agent_ids: bool = False,
) -> tuple[BusinessProcessGraph, AgentGraph]:
    """Two structurally identical graphs with deliberately unrelated text."""
    truth_concepts = {
        "t_activity_start": TruthConcept(
            id="t_activity_start", kind="activity", canonical_terms=["truth start"]
        ),
        "t_activity_finish": TruthConcept(
            id="t_activity_finish", kind="activity", canonical_terms=["truth finish"]
        ),
        "t_actor": TruthConcept(
            id="t_actor", kind="actor", canonical_terms=["truth actor"]
        ),
        "t_system": TruthConcept(
            id="t_system", kind="system", canonical_terms=["truth system"]
        ),
        "t_input": TruthConcept(id="t_input", kind="data"),
        "t_output": TruthConcept(id="t_output", kind="data"),
        "t_reason": TruthConcept(id="t_reason", kind="rationale"),
        "t_condition": TruthConcept(id="t_condition", kind="condition"),
    }
    truth_nodes = {
        "t_start": TruthNode(
            id="t_start",
            activity=_ref("t_activity_start"),
            actor=_ref("t_actor"),
            system=_ref("t_system"),
            reads=[_ref("t_input")],
            writes=[_ref("t_output")],
            necessity_rationale=_ref("t_reason"),
        ),
        "t_finish": TruthNode(
            id="t_finish",
            activity=_ref("t_activity_finish"),
            actor=_ref("t_actor"),
            system=_ref("t_system"),
            reads=[_ref("t_input")],
            writes=[_ref("t_output")],
        ),
    }
    truth_edges = {
        "t_edge": TruthEdge(
            id="t_edge",
            from_node="t_start",
            to_node="t_finish",
            condition=_ref("t_condition"),
        )
    }

    if renamed_agent_ids:
        activity_start = "a_act_alpha"
        activity_finish = "a_act_omega"
        actor = "a_actor_z"
        system = "a_system_q"
        input_data = "a_data_left"
        output_data = "a_data_right"
        reason = "a_reason"
        condition = "a_condition"
        start_node = "a_node_beta"
        finish_node = "a_node_alpha"
        edge_id = "a_edge_q"
    else:
        activity_start = "a_activity_start"
        activity_finish = "a_activity_finish"
        actor = "a_actor"
        system = "a_system"
        input_data = "a_input"
        output_data = "a_output"
        reason = "a_reason"
        condition = "a_condition"
        start_node = "a_start"
        finish_node = "a_finish"
        edge_id = "a_edge"

    agent_concepts = {
        activity_start: AgentConcept(
            id=activity_start,
            kind="activity",
            display_label=f"{agent_label_prefix}-one-{activity_start}",
            description="text must not be read",
        ),
        activity_finish: AgentConcept(
            id=activity_finish,
            kind="activity",
            display_label=f"{agent_label_prefix}-two-{activity_finish}",
        ),
        actor: AgentConcept(
            id=actor, kind="actor", display_label=f"{agent_label_prefix}-actor"
        ),
        system: AgentConcept(
            id=system, kind="system", display_label=f"{agent_label_prefix}-system"
        ),
        input_data: AgentConcept(
            id=input_data, kind="data", display_label=f"{agent_label_prefix}-input"
        ),
        output_data: AgentConcept(
            id=output_data, kind="data", display_label=f"{agent_label_prefix}-output"
        ),
        reason: AgentConcept(
            id=reason, kind="rationale", display_label=f"{agent_label_prefix}-reason"
        ),
        condition: AgentConcept(
            id=condition,
            kind="condition",
            display_label=f"{agent_label_prefix}-condition",
        ),
    }
    agent_nodes = {
        start_node: Node(
            id=start_node,
            activity=_ref(activity_start),
            actor=_ref(actor),
            system=_ref(system),
            reads=[_ref(input_data)],
            writes=[_ref(output_data)],
            necessity_rationale=_ref(reason),
        ),
        finish_node: Node(
            id=finish_node,
            activity=_ref(activity_finish),
            actor=_ref(actor),
            system=_ref(system),
            reads=[_ref(input_data)],
            writes=[_ref(output_data)],
        ),
    }
    agent_edges = {
        edge_id: Edge(
            id=edge_id,
            from_node=start_node,
            to_node=finish_node,
            condition=_ref(condition),
        )
    }

    if reverse_insertion:
        truth_concepts = dict(reversed(list(truth_concepts.items())))
        truth_nodes = dict(reversed(list(truth_nodes.items())))
        truth_edges = dict(reversed(list(truth_edges.items())))
        agent_concepts = dict(reversed(list(agent_concepts.items())))
        agent_nodes = dict(reversed(list(agent_nodes.items())))
        agent_edges = dict(reversed(list(agent_edges.items())))

    truth = BusinessProcessGraph(
        concepts=truth_concepts,
        nodes=truth_nodes,
        edges=truth_edges,
        start_node_id="t_start",
        end_node_ids=["t_finish"],
    )
    agent = AgentGraph(
        concepts=agent_concepts,
        nodes=agent_nodes,
        edges=agent_edges,
        start_node_id=start_node,
        end_node_ids=[finish_node],
    )
    return truth, agent


def _joint(truth: BusinessProcessGraph, agent: AgentGraph):
    return build_joint_structural_alignment_diagnostics(agent, truth)


def _component(trace, name: str):
    return trace.objective.components[name]


def test_identical_graphs_align_perfectly_without_text():
    truth, agent = _synthetic_graphs()
    trace = _joint(truth, agent)

    assert trace.status == "ok"
    assert trace.search.exact_search
    assert trace.objective.total_score == pytest.approx(trace.objective.max_score)
    assert trace.objective.normalized_score == pytest.approx(1.0)
    assert trace.representative_agent_node_to_truth_node == {
        "a_finish": "t_finish",
        "a_start": "t_start",
    }
    assert trace.invariant_agent_concept_to_truth_concept["a_input"] == "t_input"
    assert trace.invariant_agent_concept_to_truth_concept["a_output"] == "t_output"
    assert trace.node_ambiguity_classes == []
    assert trace.concept_ambiguity_classes == []


def test_replacing_all_agent_labels_changes_nothing():
    truth, agent = _synthetic_graphs(agent_label_prefix="random unrelated strings")
    baseline = _joint(truth, agent)
    truth_again, relabeled = _synthetic_graphs(agent_label_prefix="ZZZ not semantic")
    changed = _joint(truth_again, relabeled)

    assert baseline.model_dump(mode="json") == changed.model_dump(mode="json")


def test_renaming_and_shuffling_agent_ids_preserves_structural_result():
    truth, baseline_agent = _synthetic_graphs()
    _, renamed_agent = _synthetic_graphs(renamed_agent_ids=True)
    baseline = _joint(truth, baseline_agent)
    renamed = _joint(truth, renamed_agent)

    assert renamed.objective == baseline.objective
    assert (
        renamed.search.node_candidate_pair_count
        == baseline.search.node_candidate_pair_count
    )
    assert renamed.objective.components == baseline.objective.components
    assert renamed.objective.normalized_score == baseline.objective.normalized_score


def test_insertion_order_does_not_change_serialized_diagnostic():
    truth, agent = _synthetic_graphs()
    reverse_truth, reverse_agent = _synthetic_graphs(reverse_insertion=True)

    assert _joint(truth, agent).model_dump(mode="json") == _joint(
        reverse_truth, reverse_agent
    ).model_dump(mode="json")


def _one_node_data_graph(
    *,
    truth_reads: list[str] | None = None,
    truth_writes: list[str] | None = None,
    agent_reads: list[str] | None = None,
    agent_writes: list[str] | None = None,
    extra_agent_node: bool = False,
):
    truth_concepts = {
        "t_activity": TruthConcept(id="t_activity", kind="activity"),
        "t_read": TruthConcept(id="t_read", kind="data"),
        "t_write": TruthConcept(id="t_write", kind="data"),
        "t_extra": TruthConcept(id="t_extra", kind="data"),
    }
    agent_concepts = {
        "a_activity": AgentConcept(
            id="a_activity", kind="activity", display_label="wrong label"
        ),
        "a_read": AgentConcept(id="a_read", kind="data", display_label="WRITE"),
        "a_write": AgentConcept(id="a_write", kind="data", display_label="READ"),
        "a_extra": AgentConcept(id="a_extra", kind="data", display_label="fabricated"),
    }
    truth_node = TruthNode(
        id="t_node",
        activity=_ref("t_activity"),
        reads=[_ref(cid) for cid in truth_reads] if truth_reads is not None else None,
        writes=[_ref(cid) for cid in truth_writes]
        if truth_writes is not None
        else None,
    )
    agent_node = Node(
        id="a_node",
        activity=_ref("a_activity"),
        reads=[_ref(cid) for cid in agent_reads] if agent_reads is not None else UNSET,
        writes=[_ref(cid) for cid in agent_writes]
        if agent_writes is not None
        else UNSET,
    )
    agent_nodes = {"a_node": agent_node}
    if extra_agent_node:
        agent_nodes["a_fabricated"] = Node(id="a_fabricated", activity=_ref("a_extra"))
    return (
        BusinessProcessGraph(
            concepts=truth_concepts,
            nodes={"t_node": truth_node},
            start_node_id="t_node",
            end_node_ids=["t_node"],
        ),
        AgentGraph(
            concepts=agent_concepts,
            nodes=agent_nodes,
            start_node_id="a_node",
            end_node_ids=["a_node"],
        ),
    )


def test_missing_usage_lowers_structural_agreement():
    truth, agent = _one_node_data_graph(
        truth_reads=["t_read"],
        truth_writes=["t_write"],
        agent_reads=[],
        agent_writes=["a_write"],
    )
    trace = _joint(truth, agent)

    assert _component(trace, "reads").agreement == pytest.approx(0.0)
    assert trace.objective.normalized_score < 1.0


def test_extra_usage_is_not_treated_as_exact():
    truth, agent = _one_node_data_graph(
        truth_reads=["t_read"], agent_reads=["a_read", "a_extra"]
    )
    trace = _joint(truth, agent)

    assert _component(trace, "reads").agreement < 1.0
    assert _component(trace, "reads").matched_count == 1
    assert trace.objective.normalized_score < 1.0


def test_fabricated_nodes_can_remain_unmatched():
    truth, agent = _one_node_data_graph(
        truth_reads=["t_read"], agent_reads=["a_read"], extra_agent_node=True
    )
    trace = _joint(truth, agent)

    assert "a_fabricated" in trace.unmatched_representative_agent_nodes
    assert trace.representative_agent_node_to_truth_node["a_node"] == "t_node"
    # Same-kind but zero-support fabricated concepts are not arbitrary matches.
    assert "a_extra" not in trace.representative_agent_concept_to_truth_concept


def test_reads_and_writes_are_distinct_even_when_labels_are_wrong():
    truth, agent = _one_node_data_graph(
        truth_reads=["t_read"],
        truth_writes=["t_write"],
        agent_reads=["a_read"],
        agent_writes=["a_write"],
    )
    trace = _joint(truth, agent)

    assert trace.representative_agent_concept_to_truth_concept["a_read"] == "t_read"
    assert trace.representative_agent_concept_to_truth_concept["a_write"] == "t_write"
    assert _component(trace, "reads").agreement == pytest.approx(1.0)
    assert _component(trace, "writes").agreement == pytest.approx(1.0)


def test_different_concept_kinds_never_map():
    truth = BusinessProcessGraph(
        concepts={"t_data": TruthConcept(id="t_data", kind="data")},
        nodes={"t": TruthNode(id="t", reads=[_ref("t_data")])},
    )
    agent = AgentGraph(
        concepts={
            "a_activity": AgentConcept(
                id="a_activity", kind="activity", display_label="x"
            )
        },
        nodes={"a": Node(id="a", reads=[_ref("a_activity")])},
    )
    trace = _joint(truth, agent)

    assert trace.representative_agent_concept_to_truth_concept == {}
    assert _component(trace, "reads").matched_count == 0
    assert _component(trace, "start_node").matched_count == 0
    assert _component(trace, "start_node").agent_count == 0
    assert _component(trace, "start_node").truth_count == 0


def test_topology_disambiguates_similar_nodes():
    truth = BusinessProcessGraph(
        concepts={"t_act": TruthConcept(id="t_act", kind="activity")},
        nodes={
            "t_first": TruthNode(id="t_first", activity=_ref("t_act")),
            "t_second": TruthNode(id="t_second", activity=_ref("t_act")),
        },
        edges={
            "t_edge": TruthEdge(id="t_edge", from_node="t_first", to_node="t_second")
        },
        start_node_id="t_first",
        end_node_ids=["t_second"],
    )
    agent = AgentGraph(
        concepts={
            "a_act": AgentConcept(
                id="a_act", kind="activity", display_label="misleading"
            )
        },
        nodes={
            "a_first": Node(id="a_first", activity=_ref("a_act")),
            "a_second": Node(id="a_second", activity=_ref("a_act")),
        },
        edges={"a_edge": Edge(id="a_edge", from_node="a_first", to_node="a_second")},
        start_node_id="a_first",
        end_node_ids=["a_second"],
    )
    trace = _joint(truth, agent)

    assert trace.invariant_agent_node_to_truth_node == {
        "a_first": "t_first",
        "a_second": "t_second",
    }
    assert _component(trace, "process_edges").agreement == pytest.approx(1.0)


def test_repeated_usage_strengthens_concept_alignment():
    truth = BusinessProcessGraph(
        concepts={
            "t_act_1": TruthConcept(id="t_act_1", kind="activity"),
            "t_act_2": TruthConcept(id="t_act_2", kind="activity"),
            "t_shared": TruthConcept(id="t_shared", kind="data"),
            "t_other": TruthConcept(id="t_other", kind="data"),
        },
        nodes={
            "t_one": TruthNode(
                id="t_one", activity=_ref("t_act_1"), reads=[_ref("t_shared")]
            ),
            "t_two": TruthNode(
                id="t_two", activity=_ref("t_act_2"), reads=[_ref("t_shared")]
            ),
            "t_three": TruthNode(id="t_three", reads=[_ref("t_other")]),
        },
    )
    agent = AgentGraph(
        concepts={
            "a_act_1": AgentConcept(id="a_act_1", kind="activity", display_label="x"),
            "a_act_2": AgentConcept(id="a_act_2", kind="activity", display_label="y"),
            "a_shared": AgentConcept(
                id="a_shared", kind="data", display_label="strongly wrong"
            ),
            "a_other": AgentConcept(
                id="a_other", kind="data", display_label="also wrong"
            ),
        },
        nodes={
            "a_one": Node(
                id="a_one", activity=_ref("a_act_1"), reads=[_ref("a_shared")]
            ),
            "a_two": Node(
                id="a_two", activity=_ref("a_act_2"), reads=[_ref("a_shared")]
            ),
            "a_three": Node(id="a_three", reads=[_ref("a_other")]),
        },
    )
    trace = _joint(truth, agent)

    assert trace.representative_agent_concept_to_truth_concept["a_shared"] == "t_shared"
    assert trace.representative_agent_concept_to_truth_concept["a_other"] == "t_other"


def test_symmetric_structures_report_ambiguity_not_identity():
    truth = BusinessProcessGraph(
        concepts={
            "t_act": TruthConcept(id="t_act", kind="activity"),
            "t_x": TruthConcept(id="t_x", kind="data"),
            "t_y": TruthConcept(id="t_y", kind="data"),
        },
        nodes={
            "t_one": TruthNode(
                id="t_one", activity=_ref("t_act"), reads=[_ref("t_x"), _ref("t_y")]
            ),
            "t_two": TruthNode(
                id="t_two", activity=_ref("t_act"), reads=[_ref("t_x"), _ref("t_y")]
            ),
        },
    )
    agent = AgentGraph(
        concepts={
            "a_act": AgentConcept(id="a_act", kind="activity", display_label="x"),
            "a_left": AgentConcept(id="a_left", kind="data", display_label="left"),
            "a_right": AgentConcept(id="a_right", kind="data", display_label="right"),
        },
        nodes={
            "a_one": Node(
                id="a_one",
                activity=_ref("a_act"),
                reads=[_ref("a_left"), _ref("a_right")],
            ),
            "a_two": Node(
                id="a_two",
                activity=_ref("a_act"),
                reads=[_ref("a_left"), _ref("a_right")],
            ),
        },
    )
    trace = _joint(truth, agent)

    assert not trace.search.optimal_solution_count_is_lower_bound
    assert trace.search.optimal_solution_count >= 4
    assert trace.invariants_proven
    assert any(
        set(item.agent_ids) == {"a_one", "a_two"}
        and set(item.truth_ids) == {"t_one", "t_two"}
        for item in trace.node_ambiguity_classes
    )
    assert any(
        set(item.agent_ids) == {"a_left", "a_right"}
        and set(item.truth_ids) == {"t_x", "t_y"}
        for item in trace.concept_ambiguity_classes
    )


def test_one_to_two_symmetric_mapping_reports_variably_unmatched_truth():
    truth = BusinessProcessGraph(
        concepts={"t_act": TruthConcept(id="t_act", kind="activity")},
        nodes={
            "t_left": TruthNode(id="t_left", activity=_ref("t_act")),
            "t_right": TruthNode(id="t_right", activity=_ref("t_act")),
        },
    )
    agent = AgentGraph(
        concepts={
            "a_act": AgentConcept(id="a_act", kind="activity", display_label="x")
        },
        nodes={"a_only": Node(id="a_only", activity=_ref("a_act"))},
    )
    trace = _joint(truth, agent)

    ambiguity = next(
        item for item in trace.node_ambiguity_classes if item.agent_ids == ["a_only"]
    )
    assert ambiguity.truth_ids == ["t_left", "t_right"]
    assert ambiguity.unmatched_truth_ids == ["t_left", "t_right"]


def test_epistemic_markers_do_not_become_concept_nodes():
    truth = BusinessProcessGraph(
        concepts={"t_data": TruthConcept(id="t_data", kind="data")},
        nodes={"t": TruthNode(id="t", reads=[_ref("t_data")])},
    )
    agent = AgentGraph(
        concepts={},
        nodes={
            "a": Node(
                id="a",
                reads=UNSET,
                writes=AbsentType(),
                system=DONT_KNOW,
            )
        },
    )
    trace = _joint(truth, agent)

    assert trace.representative_agent_concept_to_truth_concept == {}
    assert trace.objective.components["reads"].matched_count == 0
    assert trace.epistemic_markers_are_not_concepts
    assert isinstance(DONT_KNOW, type(UNSET)) is False


def test_disabling_joint_diagnostics_preserves_scores_and_production_mapping(
    monkeypatch,
):
    truth, agent = _synthetic_graphs()
    first = evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)

    def broken_experiment(*args, **kwargs):
        raise RuntimeError("offline joint experiment failure")

    monkeypatch.setattr(
        joint_alignment_module,
        "build_joint_structural_alignment_diagnostics",
        broken_experiment,
    )
    second = evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert not hasattr(first.diagnostics, "joint_structural_alignment")


def test_joint_diagnostics_are_deterministic_when_rebuilt():
    truth, agent = _synthetic_graphs()
    first = _joint(truth, agent).model_dump(mode="json")
    second = _joint(truth, agent).model_dump(mode="json")
    assert first == second


@pytest.mark.parametrize("seed", [9002, 9003, 9004])
def test_offline_joint_diagnostics_preserve_stored_metrics_and_are_separate(
    seed: int,
):
    root = Path(__file__).resolve().parents[3]
    stem = root / "artifacts" / "business_interview_real_llm" / f"run_00_seed{seed}"
    trace = evaluate_artifact(
        stem.with_suffix(".json"), stem.with_suffix(".private.json")
    )

    expected_status = "historical_drift" if seed == 9002 else "matched"
    assert trace["metric_parity"]["status"] == expected_status
    if expected_status == "matched":
        assert trace["metric_parity"]["differences"] == []
    else:
        assert trace["metric_parity"]["differences"]
    assert trace["experiments"]["joint_structural_alignment"]["status"] == "ok"
    assert "joint_structural_alignment" not in trace["evaluation"]["diagnostics"]


def test_search_bound_is_reported_without_claiming_uniqueness():
    truth, agent = _synthetic_graphs()
    trace = build_joint_structural_alignment_diagnostics(
        agent,
        truth,
        max_node_search_states=1,
        max_concept_search_states=1,
        max_optimal_alternatives=1,
    )

    assert trace.search.bound_hit
    assert not trace.search.exact_search
    assert not trace.invariants_proven
    assert not trace.search.optimal_solution_count_is_exact
    assert not trace.search.optimal_solution_count_is_lower_bound
    assert trace.search.objective_lower_bound <= trace.search.objective_upper_bound
