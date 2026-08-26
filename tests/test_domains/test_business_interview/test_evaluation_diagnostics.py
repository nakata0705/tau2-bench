"""Deterministic tests for evaluator-only reconstruction diagnostics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import pytest

from scripts.business_interview_diagnostics.offline_diagnostics import (
    ArtifactDecodeError,
    classify_failed_slots,
    decode_agent_graph,
    decode_agent_slot,
)
from scripts.business_interview_evaluation_diagnostics import (
    MetricParityError,
    _check_metric_parity,
    evaluate_artifact,
    load_artifact,
    render_report,
)
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
    is_absent,
    is_dont_know,
    is_unset,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    StakeholderKnowledgeConcept,
    StakeholderKnowledgeGraph,
    StakeholderNode,
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


def _smoke_graph_payload() -> dict:
    evidence = {
        "observation_id": "obs_1",
        "quote": "CRM",
        "occurrence": 0,
    }
    return {
        "id": "artifact-graph",
        "name": "quotation",
        "start_node_id": "n1",
        "end_node_ids": ["n1"],
        "concepts": {
            "a_activity": {
                "id": "a_activity",
                "kind": "activity",
                "display_label": "Check customer",
                "description": "",
                "canonical_terms": None,
                "mentions": [evidence],
            },
            "a_data": {
                "id": "a_data",
                "kind": "data",
                "display_label": "CRM data",
                "description": "",
                "canonical_terms": None,
                "mentions": [],
            },
        },
        "nodes": {
            "n1": {
                "id": "n1",
                "activity": {
                    "concept_id": "a_activity",
                    "confidence": 0.75,
                    "evidence": [evidence],
                },
                "actor": {"unset": True},
                "system": {"absent": True, "evidence": [evidence]},
                "reads": [
                    {
                        "concept_id": "a_data",
                        "confidence": 0.5,
                        "evidence": [evidence],
                    }
                ],
                "writes": {"dont_know": True, "evidence": [evidence]},
                "necessity_rationale": {"dont_know": True, "evidence": []},
            }
        },
        "edges": {
            "e1": {
                "id": "e1",
                "from_node": "n1",
                "to_node": "n1",
                "condition": {"absent": True, "evidence": [evidence]},
                "evidence": [evidence],
            }
        },
        "terminology_agreements": [
            {
                "concept_id": "a_data",
                "term": "CRM data",
                "stakeholder_id": "stakeholder",
                "evidence": [evidence],
            }
        ],
        "validation_errors": [],
        "is_valid": True,
    }


def test_smoke_marker_decoder_round_trips_all_slot_shapes():
    graph = decode_agent_graph(_smoke_graph_payload())
    assert graph.id == "artifact-graph"
    assert graph.start_node_id == "n1"
    assert graph.end_node_ids == ["n1"]
    assert graph.concepts["a_activity"].id == "a_activity"
    node = graph.nodes["n1"]
    assert isinstance(node.activity, ConceptRef)
    assert node.activity.concept_id == "a_activity"
    assert node.activity.confidence == 0.75
    assert node.activity.evidence == [
        EvidenceRef.model_validate(
            {"observation_id": "obs_1", "quote": "CRM", "occurrence": 0}
        )
    ]
    assert is_unset(node.actor)
    assert is_absent(node.system)
    assert isinstance(node.system, AbsentType)
    assert node.system.evidence[0].observation_id == "obs_1"
    assert isinstance(node.reads, list)
    assert node.reads[0].concept_id == "a_data"
    assert node.reads[0].confidence == 0.5
    assert is_dont_know(node.writes)
    assert is_dont_know(node.necessity_rationale)
    assert is_absent(graph.edges["e1"].condition)
    assert graph.edges["e1"].id == "e1"
    assert graph.edges["e1"].from_node == "n1"
    assert graph.edges["e1"].to_node == "n1"
    assert graph.edges["e1"].evidence[0].occurrence == 0
    assert graph.terminology_agreements[0].evidence[0].quote == "CRM"


@pytest.mark.parametrize(
    "raw",
    [
        {"absent": True, "dont_know": True},
        {"unset": True, "evidence": []},
        {"foo": True},
    ],
)
def test_smoke_marker_decoder_rejects_ambiguous_or_malformed_json(raw):
    with pytest.raises(ArtifactDecodeError):
        decode_agent_slot(raw)


@pytest.mark.parametrize("marker", ["unset", "absent", "dont_know"])
def test_smoke_list_property_markers_decode_explicitly(marker):
    raw: dict[str, object] = {marker: True}
    if marker != "unset":
        raw["evidence"] = []
    decoded = decode_agent_slot(raw, list_slot=True)
    checks = {
        "unset": is_unset,
        "absent": is_absent,
        "dont_know": is_dont_know,
    }
    assert checks[marker](decoded)


def _artifact_paths(seed: int) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[3]
    stem = root / "artifacts" / "business_interview_real_llm" / f"run_00_seed{seed}"
    return stem.with_suffix(".json"), stem.with_suffix(".private.json")


@pytest.mark.parametrize("seed", [9002, 9003, 9004])
def test_offline_artifact_metrics_match_stored_metrics(seed):
    public_path, private_path = _artifact_paths(seed)
    trace = evaluate_artifact(public_path, private_path)
    stored = json.loads(public_path.read_text(encoding="utf-8"))["evaluator_metrics"]
    expected_status = "historical_drift" if seed == 9002 else "matched"
    assert trace["metric_parity"]["status"] == expected_status
    if expected_status == "matched":
        assert trace["metric_parity"]["differences"] == []
    else:
        assert "node_recall" in {
            item["field"] for item in trace["metric_parity"]["differences"]
        }
        assert trace["stored_metrics"] == stored
    assert trace["experiments"]["usage_alignment"]["method"] == (
        "usage_alignment_conditioned_on_current_node_mapping"
    )
    assert trace["experiments"]["joint_structural_alignment"]["status"] == "ok"
    assert "usage_alignment" not in trace["evaluation"]["diagnostics"]
    assert "joint_structural_alignment" not in trace["evaluation"]["diagnostics"]
    if expected_status == "matched":
        for field in trace["metric_parity"]["checked_fields"]:
            if isinstance(stored[field], float):
                assert trace["current_metrics"][field] == pytest.approx(stored[field])
            else:
                assert trace["current_metrics"][field] == stored[field]
    if seed == 9004:
        assert trace["metrics"]["rationale_correctness"] == pytest.approx(1 / 6)


def test_node_matching_reevaluation_sidecar_preserves_original_metrics():
    root = Path(__file__).resolve().parents[3]
    sidecar = root / (
        "artifacts/business_interview_real_llm/"
        "seed_9002_9003_9004_node_matching_comparison.json"
    )
    payload = json.loads(sidecar.read_text(encoding="utf-8"))

    assert payload["historical_metric_policy"]["source_artifacts_are_unmodified"]
    rows = {row["seed"]: row for row in payload["seeds"]}
    assert rows[9002]["original_stored_metrics"]["node_recall"] == pytest.approx(1 / 3)
    assert rows[9002]["current_reevaluated_metrics"]["node_recall"] == pytest.approx(
        1.0
    )
    assert rows[9002]["old_node_mapping"] != rows[9002]["new_node_mapping"]
    assert rows[9003]["new_node_mapping"] == rows[9003]["old_node_mapping"]
    assert rows[9004]["new_node_mapping"] == rows[9004]["old_node_mapping"]
    manager = next(
        node
        for node in rows[9003]["nodes"]
        if node["agent_node_id"] == "node_manager_approve"
    )
    assert manager["new_mapping"] is None
    assert manager["business_identity_evidence"]["activity_aligned"] is False


def test_offline_artifact_report_generation(tmp_path, monkeypatch):
    public_path, private_path = _artifact_paths(9002)
    trace = evaluate_artifact(public_path, private_path)
    monkeypatch.setattr(
        "scripts.business_interview_evaluation_diagnostics.REPO_ROOT", tmp_path
    )
    output_path = tmp_path / "business-interview-diagnostics.md"
    render_report([trace], output_path)
    report = output_path.read_text(encoding="utf-8")
    assert "# Business-interview evaluation diagnostics" in report
    assert "## Usage-based concept alignment" in report
    assert "## Joint structural alignment" in report


def test_seed_9004_dont_know_rationale_is_not_restored_as_absent():
    public_path, private_path = _artifact_paths(9004)
    public, private, db, truth, knowledge = load_artifact(public_path, private_path)
    agent_graph = db.graph
    assert agent_graph is not None
    rationale = agent_graph.nodes["node_receive_request"].necessity_rationale
    assert is_dont_know(rationale)
    assert not is_absent(rationale)
    result = evaluate(db, knowledge, EvaluationSpec(), truth=truth)
    assert result.rationale_correctness == pytest.approx(1 / 6)
    assert public["evaluator_metrics"]["rationale_correctness"] == pytest.approx(1 / 6)


def test_metric_parity_fails_closed_on_semantic_drift():
    public_path, private_path = _artifact_paths(9004)
    public, private, db, truth, knowledge = load_artifact(public_path, private_path)
    result = evaluate(db, knowledge, EvaluationSpec(), truth=truth)
    stored = dict(public["evaluator_metrics"])
    stored["rationale_correctness"] = 1.0
    with pytest.raises(MetricParityError, match="rationale_correctness"):
        _check_metric_parity(result, stored)
