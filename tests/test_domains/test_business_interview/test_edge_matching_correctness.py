"""Adversarial primary Agent-to-Truth edge matching contracts.

These fixtures deliberately use parallel business edges.  They are kept
separate from the large scenario tests so that edge identity and score
arithmetic can be audited without relying on an LLM trajectory.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest

from tau2.domains.business_interview.comparison import (
    GraphAlignment,
    align_agent_to_truth,
    compare_aligned_graphs,
)
from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
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
    business_entry_node_ids,
    business_graph_projection,
    canonicalize_truth_graph,
)


def _truth(*, parallel_conditions: bool = False) -> BusinessProcessGraph:
    concepts = {
        "truth_a": TruthConcept(
            id="truth_a",
            kind="activity",
            description="Activity A",
            canonical_terms=["Activity A"],
        ),
        "truth_b": TruthConcept(
            id="truth_b",
            kind="activity",
            description="Activity B",
            canonical_terms=["Activity B"],
        ),
    }
    edges: dict[str, TruthEdge] = {}
    if parallel_conditions:
        concepts.update(
            {
                "truth_approved": TruthConcept(
                    id="truth_approved",
                    kind="condition",
                    description="Approved",
                    canonical_terms=["approved"],
                ),
                "truth_rejected": TruthConcept(
                    id="truth_rejected",
                    kind="condition",
                    description="Rejected",
                    canonical_terms=["rejected"],
                ),
            }
        )
        edges = {
            "truth_approved_edge": TruthEdge(
                id="truth_approved_edge",
                from_node="A",
                to_node="B",
                condition=ConceptRef(concept_id="truth_approved"),
            ),
            "truth_rejected_edge": TruthEdge(
                id="truth_rejected_edge",
                from_node="A",
                to_node="B",
                condition=ConceptRef(concept_id="truth_rejected"),
            ),
        }
    else:
        edges = {"truth_edge": TruthEdge(id="truth_edge", from_node="A", to_node="B")}
    raw = BusinessProcessGraph(
        id="edge_matching_fixture",
        name="Edge matching fixture",
        concepts=concepts,
        nodes={
            "A": TruthNode(id="A", activity=ConceptRef(concept_id="truth_a")),
            "B": TruthNode(id="B", activity=ConceptRef(concept_id="truth_b")),
        },
        edges=edges,
    )
    return canonicalize_truth_graph(raw, entry_node_ids=["A"], exit_node_ids=["B"])


def _agent(
    edge_specs: list[tuple[str, str | None]],
    *,
    reverse_edges: bool = False,
) -> AgentGraph:
    concepts = {
        "agent_a": AgentConcept(
            id="agent_a", kind="activity", display_label="Activity A"
        ),
        "agent_b": AgentConcept(
            id="agent_b", kind="activity", display_label="Activity B"
        ),
    }
    condition_ids = {condition for _, condition in edge_specs if condition}
    for condition_id in sorted(condition_ids):
        label = condition_id.removeprefix("agent_")
        concepts[condition_id] = AgentConcept(
            id=condition_id, kind="condition", display_label=label
        )
    edge_items = []
    for edge_id, condition_id in edge_specs:
        if condition_id is None:
            # Truth None means a known unconditional edge; UNSET is not a
            # correct assertion for that slot.
            condition = AbsentType()
        else:
            condition = ConceptRef(concept_id=condition_id)
        edge_items.append(
            (
                edge_id,
                Edge(
                    id=edge_id,
                    from_node="a",
                    to_node="b",
                    condition=condition,
                ),
            )
        )
    if reverse_edges:
        edge_items.reverse()
    return AgentGraph(
        concepts=concepts,
        nodes={
            "a": Node(id="a", activity=ConceptRef(concept_id="agent_a")),
            "b": Node(id="b", activity=ConceptRef(concept_id="agent_b")),
        },
        edges=OrderedDict(edge_items),
        start_node_id="a",
        end_node_ids=["b"],
    )


def _evaluate(truth: BusinessProcessGraph, agent: AgentGraph):
    return evaluate(
        InterviewDB(graph=agent),
        None,
        EvaluationSpec(),
        truth=truth,
        stakeholder_references=[],
    )


def _edge_metrics(result):
    return (
        result.edge_recall,
        result.edge_precision,
        result.condition_correctness,
        result.fabricated_edge_count,
        result.structural_pass,
        result.quality_pass,
    )


def test_duplicate_agent_edge_is_not_reused():
    """One Truth fact can credit at most one duplicate Agent edge."""
    truth = _truth()
    agent = _agent([("agent_edge_1", None), ("agent_edge_2", None)])
    result = _evaluate(truth, agent)
    alignment = align_agent_to_truth(agent, business_graph_projection(truth))

    assert len(alignment.edge_to_truth) == 1
    assert len(set(alignment.edge_to_truth.values())) == 1
    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(0.5)
    assert result.fabricated_edge_count == 1
    assert len(result.diagnostics.unmatched_agent_edges) == 1


def test_duplicate_truth_edge_leaves_one_truth_fact_unmatched():
    truth = _truth()
    truth.edges["truth_edge_2"] = TruthEdge(
        id="truth_edge_2", from_node="A", to_node="B"
    )
    result = _evaluate(truth, _agent([("agent_edge", None)]))

    assert result.edge_precision == pytest.approx(1.0)
    assert result.edge_recall == pytest.approx(0.5)
    assert result.fabricated_edge_count == 0


def test_parallel_condition_edges_pair_by_condition_compatibility():
    result = _evaluate(
        _truth(parallel_conditions=True),
        _agent(
            [
                ("agent_rejected", "agent_rejected"),
                ("agent_approved", "agent_approved"),
            ]
        ),
    )

    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(1.0)
    assert result.condition_correctness == pytest.approx(1.0)


def test_correct_edge_wins_over_fabricated_condition_duplicate():
    """A wrong-condition duplicate must not steal the Truth edge slot."""
    truth = _truth(parallel_conditions=True)
    truth.edges = {"truth_approved_edge": truth.edges["truth_approved_edge"]}
    result = _evaluate(
        truth,
        _agent(
            [
                # Deliberately put the fabricated edge first lexically.
                ("a_fabricated_duplicate", "agent_rejected"),
                ("z_correct_edge", "agent_approved"),
            ]
        ),
    )

    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(0.5)
    assert result.condition_correctness == pytest.approx(1.0)
    assert result.fabricated_edge_count == 1


def test_correct_condition_is_selected_when_duplicate_has_wrong_condition():
    truth = _truth(parallel_conditions=True)
    truth.edges = {"truth_approved_edge": truth.edges["truth_approved_edge"]}
    result = _evaluate(
        truth,
        _agent(
            [
                ("a_fabricated_duplicate", "agent_rejected"),
                ("z_correct_edge", "agent_approved"),
            ]
        ),
    )

    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(0.5)
    assert result.condition_correctness == pytest.approx(1.0)
    assert result.fabricated_edge_count == 1
    assert result.diagnostics.unmatched_agent_edges == ["a_fabricated_duplicate"]


def test_edge_dictionary_insertion_order_does_not_change_metrics():
    truth = _truth(parallel_conditions=True)
    edge_specs: list[tuple[str, str | None]] = [
        ("agent_rejected", "agent_rejected"),
        ("agent_approved", "agent_approved"),
    ]
    first = _evaluate(truth, _agent(edge_specs))
    reversed_agent = _evaluate(
        truth.model_copy(deep=True),
        _agent(edge_specs, reverse_edges=True),
    )
    reversed_truth = _truth(parallel_conditions=True)
    reversed_truth.edges = OrderedDict(reversed(list(reversed_truth.edges.items())))
    reversed_truth_result = _evaluate(reversed_truth, _agent(edge_specs))

    assert _edge_metrics(first) == _edge_metrics(reversed_agent)
    assert _edge_metrics(first) == _edge_metrics(reversed_truth_result)


def test_agent_edge_id_rename_does_not_change_metrics():
    truth = _truth(parallel_conditions=True)
    original = _evaluate(
        truth,
        _agent(
            [
                ("agent_rejected", "agent_rejected"),
                ("agent_approved", "agent_approved"),
            ]
        ),
    )
    renamed = _evaluate(
        truth,
        _agent(
            [
                ("local_x", "agent_rejected"),
                ("local_y", "agent_approved"),
            ]
        ),
    )

    assert _edge_metrics(original) == _edge_metrics(renamed)


def test_single_nonparallel_edge_preserves_exact_score():
    result = _evaluate(_truth(), _agent([("agent_edge", None)]))

    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(1.0)
    assert result.condition_correctness == pytest.approx(1.0)
    assert result.fabricated_edge_count == 0


def test_canonical_structural_edges_are_not_business_edges():
    truth = _truth()
    result = _evaluate(truth, _agent([("agent_edge", None)]))

    assert len(truth.edges) == 3
    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(1.0)
    contract = result.diagnostics.canonical_contract
    assert contract["structural_edge_count"] == 2
    assert contract["business_edge_count"] == 1
    assert contract["scoring_excludes_structural_elements"] is True


def test_tool_schema_and_artifact_provenance_contracts_remain_separate():
    """The edge matcher only consumes graphs; it does not alter public APIs."""
    from tau2.domains.business_interview.artifact_provenance import (
        fingerprint,
        serialize_truth_graph,
    )
    from tau2.domains.business_interview.environment import get_environment

    environment = get_environment()
    names = {tool.name for tool in environment.get_tools()}
    assert len(names) == 21
    assert "add_edge" in names
    assert "update_edge" in names
    assert "remove_edge" in names

    truth = _truth()
    before_fingerprint = fingerprint(serialize_truth_graph(truth))
    result = _evaluate(truth, _agent([("agent_edge", None)]))
    assert fingerprint(serialize_truth_graph(truth)) == before_fingerprint
    assert result.diagnostics.schema_version == (
        "business_interview.evaluation_diagnostics.v5"
    )
    assert result.diagnostics.canonical_contract["canonical"] is True


def test_shared_comparator_rejects_duplicate_truth_edge_alignment():
    truth = _truth()
    agent = _agent([("agent_edge_1", None), ("agent_edge_2", None)])
    valid = align_agent_to_truth(agent, business_graph_projection(truth))
    invalid = GraphAlignment(
        concept_to_truth=valid.concept_to_truth,
        node_to_truth=valid.node_to_truth,
        edge_to_truth={
            "agent_edge_1": "truth_edge",
            "agent_edge_2": "truth_edge",
        },
        concept_recall=valid.concept_recall,
        concept_precision=valid.concept_precision,
    )

    with pytest.raises(ValueError, match="edge alignment must be one-to-one"):
        compare_aligned_graphs(
            candidate=agent,
            truth=business_graph_projection(truth),
            alignment=invalid,
            candidate_node_ids=["a", "b"],
            candidate_edge_ids=["agent_edge_1", "agent_edge_2"],
            candidate_start_node_ids={"a"},
            candidate_end_node_ids={"b"},
            truth_entry_node_ids=set(business_entry_node_ids(truth)),
            graph_valid=True,
        )
