from __future__ import annotations

import json

from experiments.business_interview.joint_structural_alignment import (
    build_joint_structural_alignment_diagnostics,
    evaluate_joint_structural_mapping_objective,
)
from scripts.business_interview_joint_alignment_audit import (  # pyright: ignore[reportMissingImports]
    build_joint_concept_disagreement_audit,
)
from tau2.domains.business_interview.graph import (
    AgentConcept,
    AgentGraph,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    Node,
    TruthConcept,
    TruthEdge,
    TruthNode,
)


def _ref(concept_id: str) -> ConceptRef:
    return ConceptRef(concept_id=concept_id)


def _single_disagreement_fixture(
    *,
    prefix: str = "",
    reverse_insertion: bool = False,
    lexical_label: str = "quotation",
) -> tuple[BusinessProcessGraph, AgentGraph, dict[str, str]]:
    truth_concept_id = f"{prefix}truth_data"
    truth_node_id = f"{prefix}truth_node"
    structural_concept_id = f"{prefix}agent_structural"
    unsupported_concept_id = f"{prefix}agent_lexical"
    agent_node_id = f"{prefix}agent_node"
    truth_concepts = {
        truth_concept_id: TruthConcept(
            id=truth_concept_id, kind="data", canonical_terms=["truth text"]
        )
    }
    truth_nodes = {
        truth_node_id: TruthNode(id=truth_node_id, writes=[_ref(truth_concept_id)])
    }
    agent_concepts = {
        structural_concept_id: AgentConcept(
            id=structural_concept_id,
            kind="data",
            display_label="unrelated structural wording",
        ),
        unsupported_concept_id: AgentConcept(
            id=unsupported_concept_id,
            kind="data",
            display_label=lexical_label,
        ),
    }
    agent_nodes = {
        agent_node_id: Node(
            id=agent_node_id,
            writes=[_ref(structural_concept_id)],
            reads=[_ref(unsupported_concept_id)],
        )
    }
    if reverse_insertion:
        truth_concepts = dict(reversed(list(truth_concepts.items())))
        truth_nodes = dict(reversed(list(truth_nodes.items())))
        agent_concepts = dict(reversed(list(agent_concepts.items())))
        agent_nodes = dict(reversed(list(agent_nodes.items())))
    truth = BusinessProcessGraph(
        concepts=truth_concepts,
        nodes=truth_nodes,
        start_node_id=truth_node_id,
        end_node_ids=[truth_node_id],
    )
    agent = AgentGraph(
        concepts=agent_concepts,
        nodes=agent_nodes,
        start_node_id=agent_node_id,
        end_node_ids=[agent_node_id],
    )
    production = {unsupported_concept_id: truth_concept_id}
    return truth, agent, production


def _audit(
    truth: BusinessProcessGraph,
    agent: AgentGraph,
    production_concepts: dict[str, str],
    *,
    production_nodes: dict[str, str] | None = None,
):
    production_nodes = production_nodes or {
        next(iter(agent.nodes)): next(iter(truth.nodes))
    }
    joint = build_joint_structural_alignment_diagnostics(
        agent,
        truth,
        production_node_to_truth=production_nodes,
        production_concept_to_truth=production_concepts,
    )
    return build_joint_concept_disagreement_audit(agent, truth, joint)


def _identity_free_records(audit: dict) -> list[str]:
    return sorted(
        json.dumps(
            {
                "classification": record["classification"],
                "production": record["production_candidate_evidence"][
                    "structural_fingerprint"
                ],
                "joint": record["joint_candidate_evidence"]["structural_fingerprint"],
                "local_delta": record["classification_signals"][
                    "local_objective_delta_joint_minus_production"
                ],
            },
            sort_keys=True,
        )
        for record in audit["records"]
    )


def test_labels_and_ids_do_not_change_identity_free_audit_evidence():
    truth, agent, production = _single_disagreement_fixture(lexical_label="quotation")
    renamed_truth, renamed_agent, renamed_production = _single_disagreement_fixture(
        prefix="renamed_", lexical_label="completely different text"
    )
    baseline = _audit(truth, agent, production)
    renamed = _audit(renamed_truth, renamed_agent, renamed_production)

    assert _identity_free_records(baseline) == _identity_free_records(renamed)
    assert baseline["label_independent_structural_calculation"]
    assert all(
        record["observation_context"]["display_only"] for record in baseline["records"]
    )


def test_insertion_order_does_not_change_audit_evidence():
    truth, agent, production = _single_disagreement_fixture()
    reversed_truth, reversed_agent, reversed_production = _single_disagreement_fixture(
        reverse_insertion=True
    )
    baseline = _audit(truth, agent, production)
    reversed_audit = _audit(reversed_truth, reversed_agent, reversed_production)

    assert _identity_free_records(baseline) == _identity_free_records(reversed_audit)
    assert baseline["classification_counts"] == reversed_audit["classification_counts"]


def test_lexically_similar_unsupported_production_candidate_has_no_support():
    truth, agent, production = _single_disagreement_fixture(lexical_label="truth text")
    audit = _audit(truth, agent, production)
    unsupported = next(
        record
        for record in audit["records"]
        if record["agent_concept_id"].endswith("agent_lexical")
    )

    evidence = unsupported["production_candidate_evidence"]
    assert evidence["candidate_support"]["positive_structural_support"] is False
    assert evidence["candidate_support"]["unsupported_candidate"] is True
    assert unsupported["classification"] != "production_strongly_supported"


def _repeated_usage_fixture():
    truth_concept = "truth_shared"
    agent_concept = "agent_shared"
    truth = BusinessProcessGraph(
        concepts={truth_concept: TruthConcept(id=truth_concept, kind="data")},
        nodes={
            "truth_start": TruthNode(id="truth_start", writes=[_ref(truth_concept)]),
            "truth_end": TruthNode(id="truth_end", writes=[_ref(truth_concept)]),
        },
        edges={
            "truth_edge": TruthEdge(
                id="truth_edge", from_node="truth_start", to_node="truth_end"
            )
        },
        start_node_id="truth_start",
        end_node_ids=["truth_end"],
    )
    agent = AgentGraph(
        concepts={
            agent_concept: AgentConcept(
                id=agent_concept, kind="data", display_label="misleading"
            )
        },
        nodes={
            "agent_start": Node(id="agent_start", writes=[_ref(agent_concept)]),
            "agent_end": Node(id="agent_end", writes=[_ref(agent_concept)]),
        },
        edges={
            "agent_edge": Edge(
                id="agent_edge", from_node="agent_start", to_node="agent_end"
            )
        },
        start_node_id="agent_start",
        end_node_ids=["agent_end"],
    )
    return truth, agent, {"agent_start": "truth_start", "agent_end": "truth_end"}


def test_repeated_structural_usage_is_recorded_as_support():
    truth, agent, production_nodes = _repeated_usage_fixture()
    # Production is deliberately empty, so the structurally selected concept
    # is a disagreement and can be inspected without a lexical signal.
    joint = build_joint_structural_alignment_diagnostics(
        agent,
        truth,
        production_node_to_truth=production_nodes,
        production_concept_to_truth={},
    )
    audit = build_joint_concept_disagreement_audit(agent, truth, joint)
    record = audit["records"][0]
    repeated = record["joint_candidate_evidence"]["repeated_usage_support"]
    assert repeated["overlap_occurrence_count"] == 2
    assert repeated["repeated_overlap"]
    assert record["classification"] == "joint_strongly_supported"


def test_symmetric_optima_are_reported_as_ambiguous():
    truth = BusinessProcessGraph(
        concepts={
            "truth_left": TruthConcept(id="truth_left", kind="data"),
            "truth_right": TruthConcept(id="truth_right", kind="data"),
        },
        nodes={
            "truth_one": TruthNode(
                id="truth_one",
                writes=[_ref("truth_left"), _ref("truth_right")],
            ),
            "truth_two": TruthNode(
                id="truth_two",
                writes=[_ref("truth_left"), _ref("truth_right")],
            ),
        },
        edges={
            "truth_edge": TruthEdge(
                id="truth_edge", from_node="truth_one", to_node="truth_two"
            )
        },
        start_node_id="truth_one",
        end_node_ids=["truth_two"],
    )
    agent = AgentGraph(
        concepts={
            "agent_left": AgentConcept(
                id="agent_left", kind="data", display_label="left"
            ),
            "agent_right": AgentConcept(
                id="agent_right", kind="data", display_label="right"
            ),
        },
        nodes={
            "agent_one": Node(
                id="agent_one",
                writes=[_ref("agent_left"), _ref("agent_right")],
            ),
            "agent_two": Node(
                id="agent_two",
                writes=[_ref("agent_left"), _ref("agent_right")],
            ),
        },
        edges={
            "agent_edge": Edge(
                id="agent_edge", from_node="agent_one", to_node="agent_two"
            )
        },
        start_node_id="agent_one",
        end_node_ids=["agent_two"],
    )
    production = {"agent_left": "truth_right", "agent_right": "truth_left"}
    joint = build_joint_structural_alignment_diagnostics(
        agent,
        truth,
        production_node_to_truth={"agent_one": "truth_one", "agent_two": "truth_two"},
        production_concept_to_truth=production,
    )
    audit = build_joint_concept_disagreement_audit(agent, truth, joint)

    assert joint.search.optimum_is_unique is False
    assert any(joint.concept_ambiguity_classes)
    assert any(
        record["alternative_optimal_mapping"]["alternative_optimal_mapping_exists"]
        for record in audit["records"]
    )
    assert all(
        record["classification"] == "structurally_ambiguous"
        for record in audit["records"]
    )


def test_counterfactual_delta_matches_objective_difference():
    truth, agent, production = _single_disagreement_fixture()
    audit = _audit(truth, agent, production)
    record = next(
        item
        for item in audit["records"]
        if item["agent_concept_id"].endswith("agent_lexical")
    )
    comparison = record["counterfactual"]["local_candidate_counterfactual"]
    expected = (
        comparison["joint_objective"]["total_score"]
        - comparison["production_forced_objective"]["total_score"]
    )
    assert comparison["objective_delta_joint_minus_production"] == expected
    for name, delta in comparison["component_deltas"].items():
        assert delta["agreement_delta_joint_minus_production"] == (
            comparison["joint_objective"]["components"]
            .get(name, {})
            .get("agreement", 0.0)
            - comparison["production_forced_objective"]["components"]
            .get(name, {})
            .get("agreement", 0.0)
        )
    direct = evaluate_joint_structural_mapping_objective(
        agent,
        truth,
        node_mapping=audit["joint_mapping_basis"]["node_mapping"],
        concept_mapping=comparison["joint_candidate_concept_mapping"],
    )
    assert direct.total_score == comparison["joint_objective"]["total_score"]
