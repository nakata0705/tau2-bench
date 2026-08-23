"""Deterministic tests for evaluator-only reconstruction diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    DONT_KNOW,
    UNSET,
    AbsentType,
    AgentConcept,
    AgentGraph,
    BusinessProcessGraph,
    ConceptKind,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    Observation,
    TruthConcept,
    TruthEdge,
    TruthNode,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    StakeholderKnowledgeConcept,
    StakeholderKnowledgeGraph,
    StakeholderNode,
)
from tau2.domains.business_interview.offline_diagnostics import (
    classify_failed_slots,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.environment.toolkit import get_tool_signatures


def _scalar_result(truth_value, agent_value, *, prop: str):
    kind = cast(
        ConceptKind,
        {"activity": "activity", "actor": "actor", "system": "system"}[prop],
    )
    truth_concepts = {
        "act": TruthConcept(
            id="act", kind="activity", canonical_terms=["check customer information"]
        )
    }
    agent_concepts = {
        "a_act": AgentConcept(
            id="a_act", kind="activity", display_label="check customer information"
        )
    }
    node_kwargs = {prop: truth_value}
    if isinstance(truth_value, ConceptRef):
        truth_concepts[truth_value.concept_id] = TruthConcept(
            id=truth_value.concept_id,
            kind=kind,
            canonical_terms=["sales" if kind == "actor" else "CRM"],
        )
    agent_kwargs = {prop: agent_value}
    if isinstance(agent_value, ConceptRef):
        agent_concepts[agent_value.concept_id] = AgentConcept(
            id=agent_value.concept_id,
            kind=kind,
            display_label="sales" if kind == "actor" else "CRM",
        )
    truth = BusinessProcessGraph(
        id="scalar",
        concepts=truth_concepts,
        nodes={
            "n": TruthNode(id="n", activity=ConceptRef(concept_id="act"), **node_kwargs)
        },
    )
    agent = AgentGraph(
        concepts=agent_concepts,
        nodes={
            "a": Node(id="a", activity=ConceptRef(concept_id="a_act"), **agent_kwargs)
        },
    )
    return evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)


def _list_result(
    *,
    truth_reads: list[ConceptRef] | None,
    agent_reads: list[ConceptRef] | None,
    truth_writes: list[ConceptRef] | None = None,
    agent_writes: list[ConceptRef] | None = None,
):
    truth_concepts = {
        "act": TruthConcept(
            id="act", kind="activity", canonical_terms=["check customer information"]
        )
    }
    agent_concepts = {
        "a_act": AgentConcept(
            id="a_act", kind="activity", display_label="check customer information"
        )
    }
    for concept_id, label in (("d1", "customer data"), ("d2", "pricing data")):
        if any(ref.concept_id == concept_id for ref in (truth_reads or [])) or any(
            ref.concept_id == concept_id for ref in (truth_writes or [])
        ):
            truth_concepts[concept_id] = TruthConcept(
                id=concept_id, kind="data", canonical_terms=[label]
            )
        if any(
            ref.concept_id in {concept_id, f"a_{concept_id}"}
            for ref in (agent_reads or [])
        ) or any(
            ref.concept_id in {concept_id, f"a_{concept_id}"}
            for ref in (agent_writes or [])
        ):
            agent_concepts[f"a_{concept_id}"] = AgentConcept(
                id=f"a_{concept_id}", kind="data", display_label=label
            )
    truth = BusinessProcessGraph(
        id="lists",
        concepts=truth_concepts,
        nodes={
            "n": TruthNode(
                id="n",
                activity=ConceptRef(concept_id="act"),
                reads=truth_reads,
                writes=truth_writes,
            )
        },
    )
    agent = AgentGraph(
        concepts=agent_concepts,
        nodes={
            "a": Node(
                id="a",
                activity=ConceptRef(concept_id="a_act"),
                reads=agent_reads if agent_reads is not None else UNSET,
                writes=agent_writes if agent_writes is not None else UNSET,
            )
        },
    )
    return evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)


def test_scalar_reason_codes_preserve_epistemic_rules():
    truth_ref = ConceptRef(concept_id="sales")
    for value, reason in (
        (UNSET, "truth_value_agent_unset"),
        (DONT_KNOW, "truth_value_agent_dont_know"),
        (AbsentType(), "truth_value_agent_absent"),
    ):
        result = _scalar_result(truth_ref, value, prop="actor")
        slot = result.diagnostics.node_diagnostics[0].slots["actor"]
        assert not slot.matched
        assert slot.reason == reason
        assert result.actor_correctness == 0.0


def test_truth_none_reason_codes_require_explicit_absent():
    truth_none = None
    result = _scalar_result(truth_none, AbsentType(), prop="system")
    slot = result.diagnostics.node_diagnostics[0].slots["system"]
    assert slot.matched and slot.reason == "truth_absent_agent_absent"
    assert result.system_correctness == 1.0
    for value, reason in (
        (UNSET, "truth_absent_agent_unset"),
        (DONT_KNOW, "truth_absent_agent_dont_know"),
    ):
        result = _scalar_result(truth_none, value, prop="system")
        slot = result.diagnostics.node_diagnostics[0].slots["system"]
        assert not slot.matched and slot.reason == reason
        assert result.system_correctness == 0.0


def test_reads_and_writes_diagnostics_show_missing_and_extra_items():
    missing = _list_result(
        truth_reads=[ConceptRef(concept_id="d1"), ConceptRef(concept_id="d2")],
        agent_reads=[ConceptRef(concept_id="a_d1")],
    )
    reads = missing.diagnostics.node_diagnostics[0].slots["reads"]
    assert reads.reason == "missing_list_item"
    assert reads.missing_truth_concept_ids == ["d2"]
    assert missing.read_correctness == 0.5

    extra = _list_result(
        truth_reads=[ConceptRef(concept_id="d1")],
        agent_reads=[ConceptRef(concept_id="a_d1"), ConceptRef(concept_id="a_d2")],
        truth_writes=[ConceptRef(concept_id="d1")],
        agent_writes=[ConceptRef(concept_id="a_d1")],
    )
    reads = extra.diagnostics.node_diagnostics[0].slots["reads"]
    assert reads.reason == "extra_list_item"
    assert reads.extra_agent_concept_ids == ["a_d2"]
    assert "extra_list_item" in reads.reason_codes

    writes_missing = _list_result(
        truth_reads=None,
        agent_reads=None,
        truth_writes=[ConceptRef(concept_id="d1"), ConceptRef(concept_id="d2")],
        agent_writes=[ConceptRef(concept_id="a_d1")],
    )
    writes = writes_missing.diagnostics.node_diagnostics[0].slots["writes"]
    assert writes.reason == "missing_list_item"
    assert writes.missing_truth_concept_ids == ["d2"]

    writes_extra = _list_result(
        truth_reads=None,
        agent_reads=None,
        truth_writes=[ConceptRef(concept_id="d1")],
        agent_writes=[ConceptRef(concept_id="a_d1"), ConceptRef(concept_id="a_d2")],
    )
    writes = writes_extra.diagnostics.node_diagnostics[0].slots["writes"]
    assert writes.reason == "extra_list_item"
    assert writes.extra_agent_concept_ids == ["a_d2"]


def test_condition_diagnostic_reports_structural_and_epistemic_match():
    truth = BusinessProcessGraph(
        id="condition",
        concepts={
            "a1": TruthConcept(id="a1", kind="activity", canonical_terms=["start"]),
            "a2": TruthConcept(id="a2", kind="activity", canonical_terms=["finish"]),
            "cond": TruthConcept(
                id="cond", kind="condition", canonical_terms=["over threshold"]
            ),
        },
        nodes={
            "A": TruthNode(id="A", activity=ConceptRef(concept_id="a1")),
            "B": TruthNode(id="B", activity=ConceptRef(concept_id="a2")),
        },
        edges={
            "e": TruthEdge(
                id="e",
                from_node="A",
                to_node="B",
                condition=ConceptRef(concept_id="cond"),
            )
        },
    )
    agent = AgentGraph(
        concepts={
            "aa1": AgentConcept(id="aa1", kind="activity", display_label="start"),
            "aa2": AgentConcept(id="aa2", kind="activity", display_label="finish"),
        },
        nodes={
            "a": Node(id="a", activity=ConceptRef(concept_id="aa1")),
            "b": Node(id="b", activity=ConceptRef(concept_id="aa2")),
        },
        edges={"ae": Edge(id="ae", from_node="a", to_node="b")},
    )
    result = evaluate(InterviewDB(graph=agent), None, EvaluationSpec(), truth=truth)
    edge = result.diagnostics.edge_diagnostics[0]
    assert edge.structural_match
    assert edge.from_node_match and edge.to_node_match
    assert edge.condition.reason == "truth_value_agent_unset"
    assert not edge.condition.matched
    assert result.condition_correctness == 0.0


def test_mapping_trace_does_not_change_mapping_score_or_repeated_output():
    first = _list_result(
        truth_reads=[ConceptRef(concept_id="d1")],
        agent_reads=[ConceptRef(concept_id="a_d1")],
    )
    second = _list_result(
        truth_reads=[ConceptRef(concept_id="d1")],
        agent_reads=[ConceptRef(concept_id="a_d1")],
    )
    first_scores = first.model_dump(mode="json", exclude={"diagnostics"})
    second_scores = second.model_dump(mode="json", exclude={"diagnostics"})
    assert first_scores == second_scores
    assert (
        first.diagnostics.concepts.agent_to_truth
        == second.diagnostics.concepts.agent_to_truth
    )
    selected = first.diagnostics.concepts.selected_mappings
    assert selected and selected[0].exact_label_match_path == "canonical_term"
    assert selected[0].lexical_similarity_score == 1.0
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_truth_diagnostics_are_not_in_agent_graph_or_db():
    result = _scalar_result(
        ConceptRef(concept_id="sales"),
        ConceptRef(concept_id="a_sales"),
        prop="actor",
    )
    db_text = str(result.diagnostics.model_dump(mode="json"))
    agent_text = str(
        {
            "graph": result.model_dump(mode="json", exclude={"diagnostics"}),
        }
    )
    assert "truth_concept_id" in db_text
    assert (
        result.diagnostics.node_diagnostics[0].slots["actor"].truth_concept_id
        == "sales"
    )
    assert "truth_concept_id" not in agent_text
    assert '"sales"' not in agent_text
    assert "tc_secret" not in agent_text
    assert "Truth" not in agent_text
    assert "get_eval_diagnostics" not in get_tool_signatures(
        InterviewTools(InterviewDB())
    )


def _classification_fixture(
    *, knowledge_system, agent_system, question=None, annotation=None
):
    truth = BusinessProcessGraph(
        id="classification",
        concepts={
            "act": TruthConcept(id="act", kind="activity", canonical_terms=["check"]),
            "sys": TruthConcept(id="sys", kind="system", canonical_terms=["CRM"]),
        },
        nodes={
            "n": TruthNode(
                id="n",
                activity=ConceptRef(concept_id="act"),
                system=ConceptRef(concept_id="sys"),
            )
        },
    )
    knowledge = StakeholderKnowledge(
        graph=StakeholderKnowledgeGraph(
            nodes={
                "skn_001": StakeholderNode(
                    id="skn_001",
                    activity=ConceptRef(concept_id="skc_act"),
                    system=knowledge_system,
                )
            },
            concepts={
                "skc_act": StakeholderKnowledgeConcept(
                    id="skc_act",
                    truth_concept_id="act",
                    kind="activity",
                    terms=["check"],
                    description="check",
                ),
                "skc_sys": StakeholderKnowledgeConcept(
                    id="skc_sys",
                    truth_concept_id="sys",
                    kind="system",
                    terms=["CRM"],
                    description="CRM",
                ),
            },
            node_truth_ids={"skn_001": "n"},
        )
    )
    agent_concepts = {
        "a_act": AgentConcept(id="a_act", kind="activity", display_label="check")
    }
    if isinstance(agent_system, ConceptRef):
        agent_concepts[agent_system.concept_id] = AgentConcept(
            id=agent_system.concept_id,
            kind="system",
            display_label="unmatched artifact",
        )
    agent = AgentGraph(
        concepts=agent_concepts,
        nodes={
            "a": Node(
                id="a",
                activity=ConceptRef(concept_id="a_act"),
                system=agent_system,
            )
        },
    )
    messages = []
    observations = []
    if question is not None:
        messages = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": "I cannot say."},
        ]
        observations = [
            Observation(
                id="obs_1",
                source_id="stakeholder",
                text="I cannot say.",
                order=0,
                turn=1,
            )
        ]
    if annotation is not None:
        messages = [
            {"role": "assistant", "content": "What system does the check use?"},
            {"role": "user", "content": "CRM"},
        ]
        observations = [
            Observation(
                id="obs_1", source_id="stakeholder", text="CRM", order=0, turn=1
            )
        ]
    db = InterviewDB(graph=agent, messages=messages, observations=observations)
    annotations = {"1": [annotation]} if annotation is not None else {}
    result = evaluate(db, knowledge, EvaluationSpec(), truth=truth)
    attributions = classify_failed_slots(
        result,
        truth=truth,
        knowledge=knowledge,
        db=db,
        annotations=cast(Mapping[int | str, Iterable[object]], annotations),
    )
    return [item for item in attributions if item.property == "system"]


def test_offline_classification_examples():
    known_system = ConceptRef(concept_id="skc_sys")
    recording = _classification_fixture(
        knowledge_system=known_system,
        agent_system=UNSET,
        annotation={
            "semantic_id": "node:skn_001:system",
            "quote": "CRM",
            "occurrence": 0,
            "mode": "value",
        },
    )
    assert recording[0].category == "agent_recording"

    disclosure = _classification_fixture(
        knowledge_system=known_system,
        agent_system=UNSET,
        question="What system does the check use?",
    )
    assert disclosure[0].category == "stakeholder_disclosure"

    elicitation = _classification_fixture(
        knowledge_system=known_system,
        agent_system=UNSET,
    )
    assert elicitation[0].category == "agent_elicitation"

    unknown = _classification_fixture(
        knowledge_system=DONT_KNOW,
        agent_system=UNSET,
    )
    assert unknown[0].category == "insufficient_evidence_to_classify"

    evaluator = _classification_fixture(
        knowledge_system=known_system,
        agent_system=ConceptRef(
            concept_id="a_sys",
            evidence=[EvidenceRef(observation_id="obs_1", quote="CRM")],
        ),
        annotation={
            "semantic_id": "node:skn_001:system",
            "quote": "CRM",
            "occurrence": 0,
            "mode": "value",
        },
    )
    assert evaluator[0].category == "evaluator_matching"
