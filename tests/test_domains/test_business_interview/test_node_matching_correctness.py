"""Adversarial deterministic AgentGraph-to-TruthGraph node matching tests."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, cast

import pytest

from tau2.domains.business_interview.comparison import align_agent_to_truth
from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentConcept,
    AgentGraph,
    BusinessProcessGraph,
    ConceptKind,
    ConceptRef,
    DontKnowType,
    Edge,
    InterviewDB,
    Node,
    TruthConcept,
    TruthEdge,
    TruthNode,
    business_graph_projection,
    canonicalize_truth_graph,
)


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _truth_graph(
    node_specs: dict[str, dict[str, str]],
    edge_pairs: list[tuple[str, str]],
    *,
    entry: str,
    exit: str,
) -> BusinessProcessGraph:
    concepts: dict[str, TruthConcept] = {}
    nodes: dict[str, TruthNode] = {}
    for node_id, spec in node_specs.items():
        kwargs: dict[str, Any] = {}
        for prop, label in spec.items():
            concept_id = f"truth_{prop}_{_slug(label)}"
            kind = cast(
                ConceptKind,
                "activity" if prop == "activity" else prop,
            )
            concepts.setdefault(
                concept_id,
                TruthConcept(
                    id=concept_id,
                    kind=kind,
                    description=label,
                    canonical_terms=[label],
                ),
            )
            attribute = "necessity_rationale" if prop == "rationale" else prop
            kwargs[attribute] = ConceptRef(concept_id=concept_id)
        nodes[node_id] = TruthNode(id=node_id, **kwargs)
    edges = {
        f"truth_edge_{index}": TruthEdge(
            id=f"truth_edge_{index}",
            from_node=from_node,
            to_node=to_node,
        )
        for index, (from_node, to_node) in enumerate(edge_pairs)
    }
    raw = BusinessProcessGraph(
        id="node_matching_truth",
        name="Node matching truth",
        concepts=concepts,
        nodes=nodes,
        edges=edges,
    )
    return canonicalize_truth_graph(raw, entry_node_ids=[entry], exit_node_ids=[exit])


def _agent_graph(
    node_specs: dict[str, dict[str, str]],
    edge_pairs: list[tuple[str, str]],
    *,
    entry: str | None = None,
    exits: list[str] | None = None,
    reverse: bool = False,
) -> AgentGraph:
    concepts: dict[str, AgentConcept] = {}
    nodes: dict[str, Node] = {}
    for node_id, spec in node_specs.items():
        kwargs: dict[str, Any] = {}
        for prop, label in spec.items():
            concept_id = f"agent_{prop}_{_slug(label)}"
            kind = cast(
                ConceptKind,
                "activity" if prop == "activity" else prop,
            )
            concepts.setdefault(
                concept_id,
                AgentConcept(
                    id=concept_id,
                    kind=kind,
                    display_label=label,
                ),
            )
            attribute = "necessity_rationale" if prop == "rationale" else prop
            kwargs[attribute] = ConceptRef(concept_id=concept_id)
        nodes[node_id] = Node(id=node_id, **kwargs)
    edge_items = [
        (
            f"agent_edge_{index}",
            Edge(
                id=f"agent_edge_{index}",
                from_node=from_node,
                to_node=to_node,
            ),
        )
        for index, (from_node, to_node) in enumerate(edge_pairs)
    ]
    if reverse:
        edge_items.reverse()
        nodes = dict(reversed(list(nodes.items())))
        concepts = dict(reversed(list(concepts.items())))
    return AgentGraph(
        concepts=concepts,
        nodes=nodes,
        edges=OrderedDict(edge_items),
        start_node_id=entry,
        end_node_ids=exits or [],
    )


def _alignment(truth: BusinessProcessGraph, agent: AgentGraph):
    return align_agent_to_truth(agent, business_graph_projection(truth))


def _evaluate(truth: BusinessProcessGraph, agent: AgentGraph):
    return evaluate(
        InterviewDB(graph=agent),
        None,
        EvaluationSpec(),
        truth=truth,
        stakeholder_references=[],
    )


def _metric_tuple(result):
    return tuple(
        getattr(result, field)
        for field in (
            "node_recall",
            "node_precision",
            "edge_recall",
            "edge_precision",
            "fabricated_node_count",
            "fabricated_edge_count",
            "structural_pass",
            "quality_pass",
        )
    )


def test_same_activity_wrong_topology_maps_to_the_structural_match():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "branch_review": {"activity": "review"},
            "serial_review": {"activity": "review"},
            "left": {"activity": "left"},
            "tail": {"activity": "tail"},
        },
        [
            ("entry", "branch_review"),
            ("entry", "serial_review"),
            ("branch_review", "left"),
            ("branch_review", "tail"),
            ("left", "tail"),
            ("serial_review", "tail"),
        ],
        entry="entry",
        exit="tail",
    )
    agent = _agent_graph(
        {
            "a_entry": {"activity": "entry"},
            "a_review": {"activity": "review"},
            "a_other": {"activity": "other"},
            "a_tail": {"activity": "tail"},
        },
        [
            ("a_entry", "a_review"),
            ("a_entry", "a_other"),
            ("a_review", "a_tail"),
            ("a_other", "a_tail"),
        ],
        entry="a_entry",
        exits=["a_tail"],
    )

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth["a_review"] == "serial_review"
    assert alignment.node_to_truth["a_review"] != "branch_review"


def test_wl_refinement_disambiguates_same_local_role_nodes():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "a_right_review": {"activity": "review"},
            "z_left_review": {"activity": "review"},
            "a_right_mid": {"activity": "right mid"},
            "z_left_mid": {"activity": "left mid"},
            "right_side": {"activity": "right side"},
            "exit": {"activity": "exit"},
        },
        [
            ("entry", "a_right_review"),
            ("entry", "z_left_review"),
            ("a_right_review", "a_right_mid"),
            ("a_right_mid", "right_side"),
            ("a_right_mid", "exit"),
            ("right_side", "exit"),
            ("z_left_review", "z_left_mid"),
            ("z_left_mid", "exit"),
        ],
        entry="entry",
        exit="exit",
    )
    agent = _agent_graph(
        {
            "a_entry": {"activity": "entry"},
            "a_left_review": {"activity": "review"},
            "z_right_review": {"activity": "review"},
            "a_left_mid": {"activity": "left mid"},
            "z_right_mid": {"activity": "right mid"},
            "z_right_side": {"activity": "right side"},
            "a_exit": {"activity": "exit"},
        },
        [
            ("a_entry", "a_left_review"),
            ("a_entry", "z_right_review"),
            ("a_left_review", "a_left_mid"),
            ("a_left_mid", "a_exit"),
            ("z_right_review", "z_right_mid"),
            ("z_right_mid", "z_right_side"),
            ("z_right_mid", "a_exit"),
            ("z_right_side", "a_exit"),
        ],
        entry="a_entry",
        exits=["a_exit"],
    )

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth["a_left_review"] == "z_left_review"
    assert alignment.node_to_truth["z_right_review"] == "a_right_review"


def test_shared_actor_and_system_without_activity_do_not_create_identity():
    truth = _truth_graph(
        {
            "approve": {
                "activity": "approve",
                "actor": "manager",
                "system": "crm",
            },
            "review": {
                "activity": "review",
                "actor": "manager",
                "system": "crm",
            },
        },
        [("approve", "review")],
        entry="approve",
        exit="review",
    )
    agent = _agent_graph(
        {
            "unknown": {
                "activity": "unrecorded activity",
                "actor": "manager",
                "system": "crm",
            }
        },
        [],
        entry="unknown",
        exits=["unknown"],
    )

    alignment = _alignment(truth, agent)

    assert "unknown" not in alignment.node_to_truth


def test_activity_absence_or_dont_know_never_creates_identity():
    truth = _truth_graph(
        {
            "approve": {
                "activity": "approve",
                "actor": "manager",
                "system": "crm",
            }
        },
        [],
        entry="approve",
        exit="approve",
    )
    for marker in (AbsentType(), DontKnowType()):
        agent = _agent_graph(
            {
                "unknown": {
                    "activity": "approve",
                    "actor": "manager",
                    "system": "crm",
                }
            },
            [],
            entry="unknown",
            exits=["unknown"],
        )
        agent.nodes["unknown"].activity = marker

        alignment = _alignment(truth, agent)

        assert "unknown" not in alignment.node_to_truth


def test_unique_activity_maps_despite_branch_serial_mismatch():
    truth = _truth_graph(
        {
            "before": {"activity": "before"},
            "branch": {"activity": "checkpoint"},
            "left": {"activity": "left"},
            "right": {"activity": "right"},
            "after": {"activity": "after"},
        },
        [
            ("before", "branch"),
            ("branch", "left"),
            ("branch", "right"),
            ("left", "after"),
            ("right", "after"),
        ],
        entry="before",
        exit="after",
    )
    agent = _agent_graph(
        {
            "a_before": {"activity": "before"},
            "a_serial": {"activity": "checkpoint"},
            "a_after": {"activity": "after"},
        },
        [("a_before", "a_serial"), ("a_serial", "a_after")],
        entry="a_before",
        exits=["a_after"],
    )

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth["a_serial"] == "branch"


def test_unique_activity_maps_despite_merge_serial_mismatch():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "before_left": {"activity": "before left"},
            "before_right": {"activity": "before right"},
            "merge": {"activity": "checkpoint"},
            "after": {"activity": "after"},
        },
        [
            ("entry", "before_left"),
            ("entry", "before_right"),
            ("before_left", "merge"),
            ("before_right", "merge"),
            ("merge", "after"),
        ],
        entry="entry",
        exit="after",
    )
    agent = _agent_graph(
        {
            "a_before": {"activity": "before left"},
            "a_serial": {"activity": "checkpoint"},
            "a_after": {"activity": "after"},
        },
        [("a_before", "a_serial"), ("a_serial", "a_after")],
        entry="a_before",
        exits=["a_after"],
    )

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth["a_serial"] == "merge"


def test_symmetric_branch_nodes_are_conservatively_unmatched():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "left_review": {"activity": "review"},
            "right_review": {"activity": "review"},
            "exit": {"activity": "exit"},
        },
        [
            ("entry", "left_review"),
            ("entry", "right_review"),
            ("left_review", "exit"),
            ("right_review", "exit"),
        ],
        entry="entry",
        exit="exit",
    )
    agent = _agent_graph(
        {
            "a_entry": {"activity": "entry"},
            "a_left": {"activity": "review"},
            "a_right": {"activity": "review"},
            "a_exit": {"activity": "exit"},
        },
        [
            ("a_entry", "a_left"),
            ("a_entry", "a_right"),
            ("a_left", "a_exit"),
            ("a_right", "a_exit"),
        ],
        entry="a_entry",
        exits=["a_exit"],
    )

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth == {
        "a_entry": "entry",
        "a_exit": "exit",
    }


def test_partial_graph_does_not_map_isolated_activity_to_internal_truth_node():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "review": {"activity": "review"},
            "exit": {"activity": "exit"},
        },
        [("entry", "review"), ("review", "exit")],
        entry="entry",
        exit="exit",
    )
    agent = _agent_graph(
        {"a_review": {"activity": "review"}},
        [],
        entry="a_review",
        exits=["a_review"],
    )

    alignment = _alignment(truth, agent)
    result = _evaluate(truth, agent)

    assert alignment.node_to_truth == {"a_review": "review"}
    assert result.node_recall == pytest.approx(1 / 3)
    assert result.node_precision == pytest.approx(1.0)
    assert result.fabricated_node_count == 0


def test_partial_graph_ambiguous_activity_stays_unmatched_without_topology():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "left_review": {"activity": "review"},
            "right_review": {"activity": "review"},
            "exit": {"activity": "exit"},
        },
        [
            ("entry", "left_review"),
            ("entry", "right_review"),
            ("left_review", "exit"),
            ("right_review", "exit"),
        ],
        entry="entry",
        exit="exit",
    )
    agent = _agent_graph({"a_review": {"activity": "review"}}, [])

    alignment = _alignment(truth, agent)

    assert alignment.node_to_truth == {}


def test_unique_topology_maps_normally_and_preserves_edges():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "review": {"activity": "review"},
            "exit": {"activity": "exit"},
        },
        [("entry", "review"), ("review", "exit")],
        entry="entry",
        exit="exit",
    )
    agent = _agent_graph(
        {
            "a_entry": {"activity": "entry"},
            "a_review": {"activity": "review"},
            "a_exit": {"activity": "exit"},
        },
        [("a_entry", "a_review"), ("a_review", "a_exit")],
        entry="a_entry",
        exits=["a_exit"],
    )

    alignment = _alignment(truth, agent)
    result = _evaluate(truth, agent)

    assert alignment.node_to_truth == {
        "a_entry": "entry",
        "a_review": "review",
        "a_exit": "exit",
    }
    assert result.node_recall == pytest.approx(1.0)
    assert result.node_precision == pytest.approx(1.0)
    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(1.0)


def test_unique_activity_survives_extra_edge_topology_mismatch():
    """An edge-only reconstruction error must not erase Node identity."""
    truth = _truth_graph(
        {
            "receive": {"activity": "receive request"},
            "create": {"activity": "create quotation"},
            "send": {"activity": "send quotation"},
        },
        [("receive", "create"), ("create", "send")],
        entry="receive",
        exit="send",
    )
    agent = _agent_graph(
        {
            "a_receive": {"activity": "receive request"},
            "a_create": {"activity": "create quotation"},
            "a_send": {"activity": "send quotation"},
        },
        [
            ("a_receive", "a_create"),
            ("a_create", "a_send"),
            ("a_receive", "a_send"),
        ],
        entry="a_receive",
        exits=["a_send"],
    )

    alignment = _alignment(truth, agent)
    result = _evaluate(truth, agent)

    assert alignment.node_to_truth == {
        "a_receive": "receive",
        "a_create": "create",
        "a_send": "send",
    }
    assert result.node_recall == pytest.approx(1.0)
    assert result.node_precision == pytest.approx(1.0)
    assert result.edge_recall == pytest.approx(1.0)
    assert result.edge_precision == pytest.approx(2 / 3)
    assert result.fabricated_edge_count == 1


def test_unique_activity_survives_missing_edge_topology_mismatch():
    truth = _truth_graph(
        {
            "receive": {"activity": "receive request"},
            "create": {"activity": "create quotation"},
            "send": {"activity": "send quotation"},
        },
        [("receive", "create"), ("create", "send")],
        entry="receive",
        exit="send",
    )
    agent = _agent_graph(
        {
            "a_receive": {"activity": "receive request"},
            "a_create": {"activity": "create quotation"},
            "a_send": {"activity": "send quotation"},
        },
        [("a_receive", "a_create")],
        entry="a_receive",
        exits=["a_send"],
    )

    alignment = _alignment(truth, agent)
    result = _evaluate(truth, agent)

    assert len(alignment.node_to_truth) == 3
    assert result.node_recall == pytest.approx(1.0)
    assert result.node_precision == pytest.approx(1.0)
    assert result.edge_recall == pytest.approx(0.5)
    assert result.edge_precision == pytest.approx(1.0)
    assert result.fabricated_edge_count == 0


def test_unique_activity_survives_wrong_downstream_target():
    truth = _truth_graph(
        {
            "receive": {"activity": "receive request"},
            "create": {"activity": "create quotation"},
            "send": {"activity": "send quotation"},
        },
        [("receive", "create"), ("create", "send")],
        entry="receive",
        exit="send",
    )
    agent = _agent_graph(
        {
            "a_receive": {"activity": "receive request"},
            "a_create": {"activity": "create quotation"},
            "a_send": {"activity": "send quotation"},
        },
        [("a_receive", "a_send"), ("a_create", "a_send")],
        entry="a_receive",
        exits=["a_send"],
    )

    alignment = _alignment(truth, agent)
    result = _evaluate(truth, agent)

    assert alignment.node_to_truth == {
        "a_receive": "receive",
        "a_create": "create",
        "a_send": "send",
    }
    assert result.node_recall == pytest.approx(1.0)
    assert result.node_precision == pytest.approx(1.0)
    assert result.edge_recall == pytest.approx(0.5)
    assert result.edge_precision == pytest.approx(0.5)
    assert result.fabricated_edge_count == 1


def test_node_metrics_are_insertion_order_and_local_id_invariant():
    truth = _truth_graph(
        {
            "entry": {"activity": "entry"},
            "review": {"activity": "review"},
            "exit": {"activity": "exit"},
        },
        [("entry", "review"), ("review", "exit")],
        entry="entry",
        exit="exit",
    )
    original = _agent_graph(
        {
            "a_entry": {"activity": "entry"},
            "a_review": {"activity": "review"},
            "a_exit": {"activity": "exit"},
        },
        [("a_entry", "a_review"), ("a_review", "a_exit")],
        entry="a_entry",
        exits=["a_exit"],
    )
    renamed = _agent_graph(
        {
            "local_3": {"activity": "entry"},
            "local_1": {"activity": "review"},
            "local_2": {"activity": "exit"},
        },
        [("local_3", "local_1"), ("local_1", "local_2")],
        entry="local_3",
        exits=["local_2"],
        reverse=True,
    )

    reversed_truth = truth.model_copy(deep=True)
    reversed_truth.nodes = dict(reversed(list(reversed_truth.nodes.items())))
    reversed_truth.edges = dict(reversed(list(reversed_truth.edges.items())))
    reversed_truth.concepts = dict(reversed(list(reversed_truth.concepts.items())))

    first = _evaluate(truth, original)
    second = _evaluate(truth.model_copy(deep=True), renamed)
    third = _evaluate(reversed_truth, original)

    assert _metric_tuple(first) == _metric_tuple(second)
    assert _metric_tuple(first) == _metric_tuple(third)
