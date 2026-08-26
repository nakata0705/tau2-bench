"""Deterministic tests for the diagnostic-only usage alignment experiment."""

from __future__ import annotations

from pathlib import Path

import pytest

import tau2.domains.business_interview.evaluation as evaluation_module
from scripts.business_interview_evaluation_diagnostics import evaluate_artifact
from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
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
from tau2.domains.business_interview.usage_alignment import (  # pyright: ignore[reportMissingImports]
    UsageAlignmentDiagnostics,
    build_usage_alignment_diagnostics,
)


def _diagnostics(
    truth: BusinessProcessGraph,
    agent: AgentGraph,
    *,
    node_mapping: dict[str, str],
    current_mapping: dict[str, str] | None = None,
) -> UsageAlignmentDiagnostics:
    return build_usage_alignment_diagnostics(
        agent,
        truth,
        node_mapping=node_mapping,
        edge_mapping={},
        current_agent_to_truth=current_mapping or {},
    )


def _single_data_graphs(
    *,
    truth_nodes: dict[str, TruthNode],
    agent_nodes: dict[str, Node],
    truth_data: dict[str, TruthConcept],
    agent_data: dict[str, AgentConcept],
) -> tuple[BusinessProcessGraph, AgentGraph]:
    return (
        BusinessProcessGraph(concepts=truth_data, nodes=truth_nodes),
        AgentGraph(concepts=agent_data, nodes=agent_nodes),
    )


def _pair(trace: UsageAlignmentDiagnostics, agent_id: str, truth_id: str):
    return next(
        item
        for item in trace.candidate_pairs
        if item.agent_concept_id == agent_id and item.truth_concept_id == truth_id
    )


def test_exact_same_usage_matches_unrelated_labels():
    truth, agent = _single_data_graphs(
        truth_nodes={
            "n_customer": TruthNode(
                id="n_customer",
                reads=[ConceptRef(concept_id="tc_customer")],
            ),
            "n_quote": TruthNode(
                id="n_quote",
                reads=[ConceptRef(concept_id="tc_customer")],
            ),
        },
        agent_nodes={
            "a_customer": Node(
                id="a_customer",
                reads=[ConceptRef(concept_id="data_customer")],
            ),
            "a_quote": Node(
                id="a_quote",
                reads=[ConceptRef(concept_id="data_customer")],
            ),
        },
        truth_data={
            "tc_customer": TruthConcept(
                id="tc_customer",
                kind="data",
                canonical_terms=["customer information"],
            )
        },
        agent_data={
            "data_customer": AgentConcept(
                id="data_customer",
                kind="data",
                display_label="顧客マスタ",
            )
        },
    )
    trace = _diagnostics(
        truth,
        agent,
        node_mapping={"a_customer": "n_customer", "a_quote": "n_quote"},
    )
    pair = _pair(trace, "data_customer", "tc_customer")
    assert pair.exact_usage_match
    assert pair.truth_usage_addresses == [
        "node:n_customer:reads",
        "node:n_quote:reads",
    ]
    assert pair.agent_usage_addresses == pair.truth_usage_addresses
    assert trace.assignments[0].exact_usage_match
    assert trace.assignments[0].truth_concept_id == "tc_customer"


def test_same_label_different_usage_is_not_an_exact_usage_match():
    truth, agent = _single_data_graphs(
        truth_nodes={
            "n_customer": TruthNode(
                id="n_customer",
                reads=[ConceptRef(concept_id="tc_customer")],
            ),
            "n_quote": TruthNode(
                id="n_quote",
                reads=[ConceptRef(concept_id="tc_quote")],
            ),
        },
        agent_nodes={
            "a_quote": Node(
                id="a_quote",
                reads=[ConceptRef(concept_id="data_customer")],
            )
        },
        truth_data={
            "tc_customer": TruthConcept(
                id="tc_customer",
                kind="data",
                canonical_terms=["customer information"],
            ),
            "tc_quote": TruthConcept(
                id="tc_quote",
                kind="data",
                canonical_terms=["customer information"],
            ),
        },
        agent_data={
            "data_customer": AgentConcept(
                id="data_customer",
                kind="data",
                display_label="customer information",
            )
        },
    )
    trace = _diagnostics(
        truth,
        agent,
        node_mapping={"a_quote": "n_quote"},
        current_mapping={"data_customer": "tc_customer"},
    )
    assert not _pair(trace, "data_customer", "tc_customer").exact_usage_match
    assert _pair(trace, "data_customer", "tc_quote").exact_usage_match
    assert trace.assignments[0].truth_concept_id == "tc_quote"
    assert trace.comparisons[0].classification == "usage_exact_but_current_different"


def test_partial_usage_reports_expected_precision_recall_f1_and_jaccard():
    truth, agent = _single_data_graphs(
        truth_nodes={
            "n1": TruthNode(id="n1", reads=[ConceptRef(concept_id="truth_data")]),
            "n2": TruthNode(id="n2", reads=[ConceptRef(concept_id="truth_data")]),
        },
        agent_nodes={
            "a1": Node(id="a1", reads=[ConceptRef(concept_id="agent_data")]),
        },
        truth_data={
            "truth_data": TruthConcept(id="truth_data", kind="data"),
        },
        agent_data={
            "agent_data": AgentConcept(
                id="agent_data", kind="data", display_label="unrelated"
            ),
        },
    )
    trace = _diagnostics(truth, agent, node_mapping={"a1": "n1"})
    pair = _pair(trace, "agent_data", "truth_data")
    assert pair.overlap_count == 1
    assert pair.truth_usage_count == 2
    assert pair.agent_usage_count == 1
    assert pair.usage_recall == pytest.approx(0.5)
    assert pair.usage_precision == pytest.approx(1.0)
    assert pair.usage_f1 == pytest.approx(2 / 3)
    assert pair.usage_jaccard == pytest.approx(0.5)
    assert pair.usages_only_in_truth == ["node:n2:reads"]


def test_extra_agent_usage_reduces_precision_and_missing_usage_reduces_recall():
    truth, extra_agent = _single_data_graphs(
        truth_nodes={
            "n1": TruthNode(id="n1", reads=[ConceptRef(concept_id="truth_data")]),
        },
        agent_nodes={
            "a1": Node(id="a1", reads=[ConceptRef(concept_id="agent_data")]),
            "a2": Node(id="a2", reads=[ConceptRef(concept_id="agent_data")]),
        },
        truth_data={"truth_data": TruthConcept(id="truth_data", kind="data")},
        agent_data={
            "agent_data": AgentConcept(
                id="agent_data", kind="data", display_label="data"
            )
        },
    )
    extra = _diagnostics(
        truth,
        extra_agent,
        node_mapping={"a1": "n1", "a2": "n2"},
    )
    extra_pair = _pair(extra, "agent_data", "truth_data")
    assert extra_pair.usage_precision == pytest.approx(0.5)
    assert extra_pair.usage_recall == pytest.approx(1.0)
    assert extra_pair.usages_only_in_agent == ["node:n2:reads"]

    truth_with_two, missing_agent = _single_data_graphs(
        truth_nodes={
            "n1": TruthNode(id="n1", reads=[ConceptRef(concept_id="truth_data")]),
            "n2": TruthNode(id="n2", reads=[ConceptRef(concept_id="truth_data")]),
        },
        agent_nodes={
            "a1": Node(id="a1", reads=[ConceptRef(concept_id="agent_data")]),
        },
        truth_data={"truth_data": TruthConcept(id="truth_data", kind="data")},
        agent_data={
            "agent_data": AgentConcept(
                id="agent_data", kind="data", display_label="data"
            )
        },
    )
    missing = _diagnostics(
        truth_with_two,
        missing_agent,
        node_mapping={"a1": "n1"},
    )
    missing_pair = _pair(missing, "agent_data", "truth_data")
    assert missing_pair.usage_recall == pytest.approx(0.5)
    assert missing_pair.usage_precision == pytest.approx(1.0)


def test_different_concept_kinds_never_compete():
    truth = BusinessProcessGraph(
        concepts={
            "truth_activity": TruthConcept(id="truth_activity", kind="activity"),
            "truth_data": TruthConcept(id="truth_data", kind="data"),
        },
        nodes={
            "n": TruthNode(
                id="n",
                activity=ConceptRef(concept_id="truth_activity"),
                reads=[ConceptRef(concept_id="truth_data")],
            )
        },
    )
    agent = AgentGraph(
        concepts={
            "agent_activity": AgentConcept(
                id="agent_activity", kind="activity", display_label="x"
            ),
            "agent_data": AgentConcept(id="agent_data", kind="data", display_label="x"),
        },
        nodes={
            "a": Node(
                id="a",
                activity=ConceptRef(concept_id="agent_data"),
                reads=[ConceptRef(concept_id="agent_activity")],
            )
        },
    )
    trace = _diagnostics(truth, agent, node_mapping={"a": "n"})
    assert {
        (pair.agent_concept_id, pair.truth_concept_id) for pair in trace.candidate_pairs
    } == {
        ("agent_activity", "truth_activity"),
        ("agent_data", "truth_data"),
    }
    assert trace.assignments == []


def test_edge_condition_usage_is_translated_through_edge_mapping():
    truth = BusinessProcessGraph(
        concepts={
            "truth_condition": TruthConcept(
                id="truth_condition", kind="condition", canonical_terms=["month end"]
            )
        },
        nodes={
            "truth_from": TruthNode(id="truth_from"),
            "truth_to": TruthNode(id="truth_to"),
        },
        edges={
            "truth_edge": TruthEdge(
                id="truth_edge",
                from_node="truth_from",
                to_node="truth_to",
                condition=ConceptRef(concept_id="truth_condition"),
            )
        },
    )
    agent = AgentGraph(
        concepts={
            "agent_condition": AgentConcept(
                id="agent_condition", kind="condition", display_label="different words"
            )
        },
        nodes={
            "agent_from": Node(id="agent_from"),
            "agent_to": Node(id="agent_to"),
        },
        edges={
            "agent_edge": Edge(
                id="agent_edge",
                from_node="agent_from",
                to_node="agent_to",
                condition=ConceptRef(concept_id="agent_condition"),
            )
        },
    )
    trace = build_usage_alignment_diagnostics(
        agent,
        truth,
        node_mapping={"agent_from": "truth_from", "agent_to": "truth_to"},
        edge_mapping={"agent_edge": "truth_edge"},
        current_agent_to_truth={},
    )
    pair = _pair(trace, "agent_condition", "truth_condition")
    assert pair.exact_usage_match
    assert pair.truth_usage_addresses == ["edge:truth_edge:condition"]
    assert pair.agent_usage_addresses == ["edge:truth_edge:condition"]


def test_repeated_usage_on_multiple_nodes_uses_set_semantics():
    truth, agent = _single_data_graphs(
        truth_nodes={
            "n1": TruthNode(
                id="n1",
                reads=[
                    ConceptRef(concept_id="truth_data"),
                    ConceptRef(concept_id="truth_data"),
                ],
            ),
            "n2": TruthNode(id="n2", reads=[ConceptRef(concept_id="truth_data")]),
        },
        agent_nodes={
            "a1": Node(
                id="a1",
                reads=[
                    ConceptRef(concept_id="agent_data"),
                    ConceptRef(concept_id="agent_data"),
                ],
            ),
            "a2": Node(id="a2", reads=[ConceptRef(concept_id="agent_data")]),
        },
        truth_data={"truth_data": TruthConcept(id="truth_data", kind="data")},
        agent_data={
            "agent_data": AgentConcept(
                id="agent_data", kind="data", display_label="data"
            )
        },
    )
    trace = _diagnostics(
        truth,
        agent,
        node_mapping={"a1": "n1", "a2": "n2"},
    )
    pair = _pair(trace, "agent_data", "truth_data")
    assert pair.truth_usage_count == 2
    assert pair.agent_usage_count == 2
    assert pair.overlap_count == 2
    assert pair.exact_usage_match


def test_identical_usage_is_reported_as_ambiguous_not_resolved_by_labels():
    truth = BusinessProcessGraph(
        concepts={
            "truth_a": TruthConcept(
                id="truth_a", kind="data", canonical_terms=["alpha"]
            ),
            "truth_b": TruthConcept(
                id="truth_b", kind="data", canonical_terms=["beta"]
            ),
        },
        nodes={
            "n": TruthNode(
                id="n",
                reads=[
                    ConceptRef(concept_id="truth_a"),
                    ConceptRef(concept_id="truth_b"),
                ],
            )
        },
    )
    agent = AgentGraph(
        concepts={
            "agent_x": AgentConcept(id="agent_x", kind="data", display_label="x-label"),
            "agent_y": AgentConcept(id="agent_y", kind="data", display_label="y-label"),
        },
        nodes={
            "a": Node(
                id="a",
                reads=[
                    ConceptRef(concept_id="agent_x"),
                    ConceptRef(concept_id="agent_y"),
                ],
            )
        },
    )
    trace = _diagnostics(
        truth,
        agent,
        node_mapping={"a": "n"},
        current_mapping={"agent_x": "truth_b", "agent_y": "truth_a"},
    )
    assert len(trace.ambiguity_classes) == 1
    ambiguity = trace.ambiguity_classes[0]
    assert ambiguity.truth_concept_ids == ["truth_a", "truth_b"]
    assert ambiguity.agent_concept_ids == ["agent_x", "agent_y"]
    assert all(item.ambiguous for item in trace.assignments)
    assert {item.classification for item in trace.comparisons} == {"usage_ambiguous"}


def test_partial_tied_usage_candidates_are_reported_ambiguous():
    truth = BusinessProcessGraph(
        concepts={
            "truth_a": TruthConcept(id="truth_a", kind="data"),
            "truth_b": TruthConcept(id="truth_b", kind="data"),
        },
        nodes={
            "shared": TruthNode(
                id="shared",
                reads=[
                    ConceptRef(concept_id="truth_a"),
                    ConceptRef(concept_id="truth_b"),
                ],
            ),
            "only_a": TruthNode(id="only_a", reads=[ConceptRef(concept_id="truth_a")]),
            "only_b": TruthNode(id="only_b", reads=[ConceptRef(concept_id="truth_b")]),
        },
    )
    agent = AgentGraph(
        concepts={
            "agent_x": AgentConcept(id="agent_x", kind="data", display_label="x"),
            "agent_y": AgentConcept(id="agent_y", kind="data", display_label="y"),
        },
        nodes={
            "agent_shared": Node(
                id="agent_shared",
                reads=[
                    ConceptRef(concept_id="agent_x"),
                    ConceptRef(concept_id="agent_y"),
                ],
            )
        },
    )
    trace = _diagnostics(
        truth,
        agent,
        node_mapping={"agent_shared": "shared"},
    )
    pair_x_a = _pair(trace, "agent_x", "truth_a")
    pair_x_b = _pair(trace, "agent_x", "truth_b")
    assert pair_x_a.usage_f1 == pytest.approx(pair_x_b.usage_f1)
    assert pair_x_a.usage_f1 > 0.0
    assert len(trace.ambiguity_classes) == 1
    ambiguity = trace.ambiguity_classes[0]
    assert ambiguity.truth_concept_ids == ["truth_a", "truth_b"]
    assert ambiguity.agent_concept_ids == ["agent_x", "agent_y"]
    assert "multiple_candidates" in ambiguity.reason
    assert all(item.ambiguous for item in trace.assignments)
    assert {item.classification for item in trace.comparisons} == {"usage_ambiguous"}


def test_usage_assignment_is_deterministic_and_label_independent():
    truth = BusinessProcessGraph(
        concepts={
            "truth_b": TruthConcept(id="truth_b", kind="data"),
            "truth_a": TruthConcept(id="truth_a", kind="data"),
        },
        nodes={
            "n2": TruthNode(id="n2", reads=[ConceptRef(concept_id="truth_b")]),
            "n1": TruthNode(id="n1", reads=[ConceptRef(concept_id="truth_a")]),
        },
    )
    agent = AgentGraph(
        concepts={
            "agent_y": AgentConcept(
                id="agent_y", kind="data", display_label="totally different y"
            ),
            "agent_x": AgentConcept(
                id="agent_x", kind="data", display_label="totally different x"
            ),
        },
        nodes={
            "a2": Node(id="a2", reads=[ConceptRef(concept_id="agent_y")]),
            "a1": Node(id="a1", reads=[ConceptRef(concept_id="agent_x")]),
        },
    )
    kwargs = {
        "node_mapping": {"a1": "n1", "a2": "n2"},
        "current_mapping": {},
    }
    first = _diagnostics(truth, agent, **kwargs)
    second = _diagnostics(truth, agent, **kwargs)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert {
        item.agent_concept_id: item.truth_concept_id for item in first.assignments
    } == {"agent_x": "truth_a", "agent_y": "truth_b"}


def test_usage_diagnostics_do_not_change_score_fields_or_current_mapping(
    monkeypatch,
):
    truth, agent = _single_data_graphs(
        truth_nodes={
            "n": TruthNode(
                id="n",
                reads=[ConceptRef(concept_id="truth_data")],
            )
        },
        agent_nodes={
            "a": Node(
                id="a",
                reads=[ConceptRef(concept_id="agent_data")],
            )
        },
        truth_data={
            "truth_data": TruthConcept(
                id="truth_data", kind="data", canonical_terms=["data"]
            )
        },
        agent_data={
            "agent_data": AgentConcept(
                id="agent_data", kind="data", display_label="data"
            )
        },
    )
    first = evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)
    monkeypatch.setattr(
        evaluation_module,
        "build_usage_alignment_diagnostics",
        lambda *args, **kwargs: UsageAlignmentDiagnostics(),
    )
    second = evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)
    assert first.model_dump(mode="json", exclude={"diagnostics"}) == second.model_dump(
        mode="json", exclude={"diagnostics"}
    )
    assert (
        first.diagnostics.concepts.agent_to_truth
        == second.diagnostics.concepts.agent_to_truth
    )
    assert first.diagnostics.usage_alignment.assignment_uses_labels is False
    assert first.diagnostics.usage_alignment.method == (
        "usage_alignment_conditioned_on_current_node_mapping"
    )


@pytest.mark.parametrize("seed", [9002, 9003, 9004])
def test_stored_metric_parity_survives_usage_diagnostics(seed: int):
    root = Path(__file__).resolve().parents[3]
    stem = root / "artifacts" / "business_interview_real_llm" / f"run_00_seed{seed}"
    trace = evaluate_artifact(
        stem.with_suffix(".json"), stem.with_suffix(".private.json")
    )
    assert trace["metric_parity"]["status"] == "matched"
    assert trace["metric_parity"]["differences"] == []
    assert (
        trace["evaluation"]["diagnostics"]["usage_alignment"]["assignment_uses_labels"]
        is False
    )
