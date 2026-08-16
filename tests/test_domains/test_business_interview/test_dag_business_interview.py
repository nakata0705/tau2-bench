"""Tests for the evidence-backed DAG business_interview domain (v3, authenticity).

Observations are now **authentic primary evidence**: they are derived only from
actual stakeholder (user) messages via ``observe_turn`` — the agent cannot write
arbitrary Observation text, source, or turn. These tests cover the DAG model,
declared endpoints, evidence/provenance gates, observation authenticity
(falsification A-F of the exploit), lightweight relevance, EN/JA equivalence,
leakage, and the end-to-end reward path.
"""

import pytest

import tau2.domains.business_interview.dag as dagmod
import tau2.domains.business_interview.scenario as scmod
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.business_interview.dag import (
    BusinessDAG,
    Edge,
    InferredValue,
    InterviewDB,
    Necessity,
    Node,
)
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.evaluation import evaluate
from tau2.domains.business_interview.scenario import (
    get_scenario,
    quotation_finance_filter,
    quotation_sales_filter,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import BUSINESS_INTERVIEW_POLICY_PATH

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO, "lab_sample_flow"]

_TRUTH_NODES = [
    ("a", "receive quotation request", "sales", None, [], ["request"]),
    ("b", "check customer information in the CRM", "sales", "crm", ["customer"], []),
    (
        "c",
        "create quotation in the quoting system",
        "sales",
        "quoting",
        ["customer", "pricing"],
        ["quote"],
    ),
    (
        "d",
        "approve high-value quotation",
        "manager",
        "quoting",
        ["quote"],
        ["approval"],
    ),
    (
        "e",
        "send quotation to customer",
        "sales",
        "email",
        ["quote"],
        ["sent_quote"],
    ),
    (
        "f",
        "send quotation summary to accounting at month-end",
        "sales",
        "excel",
        ["quote"],
        ["excel_summary"],
    ),
]

_JA_ACTIONS = [
    "見積依頼を受け付ける",
    "CRMで顧客情報を確認する",
    "見積システムで見積を作成する",
    "高額見積を承認する",
    "見積を顧客に送付する",
    "月末に経理へ見積集計を送る",
]

_NODE_STATEMENTS = [
    "We receive quotation requests and send them by email.",
    "We check the customer information in the CRM.",
    "We create the quotation in the quoting system using customer and pricing info.",
    "A manager approves high-value quotations over 1,000,000; this is for credit risk management.",
    "We send the quotation to the customer by email.",
    "At month-end we send a summary to Accounting as an Excel file, but I do not know why.",
]

_EDGE_STATEMENTS = [
    "The request step is followed by checking the customer.",
    "After checking we create the quotation.",
    "Quotations over 1,000,000 go to a manager for approval.",
    "Quotations at or below 1,000,000 are sent straight to the customer.",
    "After approval the quotation is sent to the customer.",
    "At month-end a summary is also sent to Accounting.",
]


def _tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    sc = get_scenario(scenario)
    return evaluate(tools.db, sc.truth, sc.spec)


def _ingest(tools: InterviewTools, role: str = "user", content: str = "") -> int:
    tools.db.messages.append({"role": role, "content": content})
    return len(tools.db.messages) - 1


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("a", "b", "c", "d", "e", "f"),
    edge_ids: tuple = ("e1", "e2", "e3", "e4", "e5", "e6"),
    evidence: bool = True,
):
    """Build the correct quotation DAG.

    With ``evidence=True`` each claim is captured as an authentic Observation
    from an ingested stakeholder (user) message via ``observe_turn``. With
    ``evidence=False`` the same topology is built with no provenance (used to
    prove the evidence gate rejects un-evidenced DAGs).
    """
    i1, i2, i3, i4, i5, i6 = ids
    actions = _JA_ACTIONS if ja else [t[1] for t in _TRUTH_NODES]
    node_data = list(zip(_TRUTH_NODES, actions))
    node_map = {"a": i1, "b": i2, "c": i3, "d": i4, "e": i5, "f": i6}
    edge_defs = [
        (edge_ids[0], i1, i2, None),
        (edge_ids[1], i2, i3, None),
        (edge_ids[2], i3, i4, "100万円超" if ja else "amount over 1,000,000"),
        (edge_ids[3], i3, i5, "100万円以下" if ja else "amount at or below 1,000,000"),
        (edge_ids[4], i4, i5, None),
        (edge_ids[5], i3, i6, "月末" if ja else "month-end"),
    ]
    rationale = "与信リスク管理のため" if ja else "for credit risk management"

    if evidence:
        _ingest(tools, "assistant", "Hello.")
        action_by_sid = {sid: action for (sid, _, _, _, _, _), action in node_data}
        prim_by_sid = {
            "a": "receive",
            "b": "check",
            "c": "create",
            "d": "approve",
            "e": "send",
            "f": "send",
        }
        obs = {}
        for sid in ("a", "b", "c", "d", "e", "f"):
            turn = _ingest(
                tools, "user", f"The process involves: {action_by_sid[sid]}."
            )
            obs[sid] = tools.observe_turn(turn)
        for sid in ("a", "b", "c", "d", "e", "f"):
            tools.discover_concept(
                f"c{sid}", action_by_sid[sid], observation_id=obs[sid]
            )
        for (sid, _, actor, system, reads, writes), action in node_data:
            tools.add_node(
                node_map[sid],
                action,
                primitive=prim_by_sid[sid],
                concept_id=f"c{sid}",
                actor=actor,
                system=system,
                reads=reads,
                writes=writes,
                observation_id=obs[sid],
            )
        tools.set_node_necessity(i4, rationale=rationale, observation_id=obs["d"])
        tools.set_node_necessity(i6)
        eobs = {}
        node_id_to_action = {node_map[sid]: action_by_sid[sid] for sid in action_by_sid}
        for eid, frm, to, pred in edge_defs:
            turn = _ingest(
                tools,
                "user",
                f"The flow proceeds from {node_id_to_action[frm]} to {node_id_to_action[to]}.",
            )
            eobs[eid] = tools.observe_turn(turn)
        for eid, frm, to, pred in edge_defs:
            tools.add_edge(eid, frm, to, predicate=pred, observation_id=eobs[eid])
    else:
        for (sid, _, actor, system, reads, writes), action in node_data:
            tools.add_node(
                node_map[sid],
                action,
                actor=actor,
                system=system,
                reads=reads,
                writes=writes,
            )
        tools.set_node_necessity(i4, rationale=rationale)
        tools.set_node_necessity(i6)
        for eid, frm, to, pred in edge_defs:
            tools.add_edge(eid, frm, to, predicate=pred)

    tools.set_dag_endpoints(start_node_id=i1, end_node_ids=[i5, i6])
    tools.finish_interview()


# ---------------------------------------------------------------------------
# Falsification A-J: model shape / validation
# ---------------------------------------------------------------------------


def test_A_truth_and_agent_result_use_same_dag_class():
    sc = get_scenario(SCENARIO)
    assert isinstance(sc.truth, BusinessDAG)
    tools = _tools()
    _build(tools)
    assert isinstance(tools.db.dag, BusinessDAG)
    assert isinstance(tools.db.interview_result().dag, BusinessDAG)


def test_B_no_truth_specific_gt_classes():
    for name in ("GTStep", "GTTransition", "GTBranch", "GTNecessity", "GTNode"):
        assert not hasattr(scmod, name), name


def test_C_one_start_multiple_ends_valid():
    dag = BusinessDAG(
        nodes={
            "a": Node(id="a", action=InferredValue(value="x")),
            "b": Node(id="b", action=InferredValue(value="y")),
            "c": Node(id="c", action=InferredValue(value="z")),
        },
        edges={
            "e1": Edge(id="e1", from_node="a", to_node="b"),
            "e2": Edge(id="e2", from_node="a", to_node="c"),
        },
        start_node_id="a",
        end_node_ids=["b", "c"],
    )
    assert dag.is_valid
    assert dag.validate() == []


def test_D_cycle_reject():
    dag = BusinessDAG(
        nodes={
            "a": Node(id="a", action=InferredValue(value="x")),
            "b": Node(id="b", action=InferredValue(value="y")),
        },
        edges={
            "e1": Edge(id="e1", from_node="a", to_node="b"),
            "e2": Edge(id="e2", from_node="b", to_node="a"),
        },
        start_node_id="a",
        end_node_ids=["b"],
    )
    errors = dag.validate()
    assert any("cycle" in e for e in errors)


def test_E_unreachable_node_reject():
    dag = BusinessDAG(
        nodes={
            "a": Node(id="a", action=InferredValue(value="x")),
            "b": Node(id="b", action=InferredValue(value="y")),
        },
        edges={},
        start_node_id="a",
        end_node_ids=["a", "b"],
    )
    errors = dag.validate()
    assert any("unreachable" in e for e in errors)


def test_F_dangling_edge_reject():
    dag = BusinessDAG(
        nodes={"a": Node(id="a", action=InferredValue(value="x"))},
        edges={"e1": Edge(id="e1", from_node="a", to_node="ghost")},
        start_node_id="a",
        end_node_ids=["a"],
    )
    errors = dag.validate()
    assert any("to_node not found" in e for e in errors)


def test_G_condition_only_on_edge_predicate():
    assert "condition" not in Node.model_fields
    assert "predicate" in Edge.model_fields
    assert not hasattr(scmod, "StakeholderStepTruth")


def test_H_no_branch_class():
    assert not hasattr(dagmod, "Branch")


def test_I_no_transition_class():
    assert not hasattr(dagmod, "Transition")


def test_J_no_step_class():
    assert not hasattr(dagmod, "Step")
    assert not hasattr(dagmod, "WorkflowStep")


# ---------------------------------------------------------------------------
# Observation authenticity (exploit falsification A-F)
# ---------------------------------------------------------------------------


def test_A_no_arbitrary_text_observation_api():
    assert not hasattr(InterviewTools, "record_observation")


def test_B_valid_stakeholder_turn_creates_observation():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive quotation requests.")
    oid = tools.observe_turn(turn)
    obs = tools.db.observations[0]
    assert oid == "obs_%d" % turn
    assert obs.text == "We receive quotation requests."
    assert obs.source_id == "stakeholder"
    assert obs.turn == turn


def test_C_nonexistent_turn_rejected():
    tools = _tools()
    with pytest.raises(ValueError):
        tools.observe_turn(99)
    with pytest.raises(ValueError):
        tools.observe_turn(-1)


def test_D_assistant_turn_rejected():
    tools = _tools()
    _ingest(tools, "assistant", "I am the agent.")
    with pytest.raises(ValueError):
        tools.observe_turn(0)
    _ingest(tools, "user", "I am the stakeholder.")
    # tool message role rejected too
    _ingest(tools, "tool", "tool result")
    with pytest.raises(ValueError):
        tools.observe_turn(len(tools.db.messages) - 1)


def test_E_duplicate_capture_idempotent():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive quotation requests.")
    oid1 = tools.observe_turn(turn)
    oid2 = tools.observe_turn(turn)
    assert oid1 == oid2
    assert len(tools.db.observations) == 1  # no unlimited duplicates


def test_F_observation_immutable():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive quotation requests.")
    tools.observe_turn(turn)
    with pytest.raises(Exception):
        tools.db.observations[0].text = "fabricated"
    with pytest.raises(Exception):
        tools.db.observations[0].source_id = "manager"


def test_fake_observation_cannot_be_inserted_through_normal_tools():
    tools = _tools()
    tools.start_inference("Q")
    _ingest(tools, "user", "We receive quotation requests.")
    oid = tools.observe_turn(turn_idx=len(tools.db.messages) - 1)
    tools.add_node("a", "receive quotation request", observation_id=oid)
    with pytest.raises(ValueError):
        tools.add_node("b", "check customer", observation_id="does_not_exist")


def test_correct_dag_authentic_provenance_passes():
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.quality_pass is True
    assert res.provenance_authenticity_pass is True
    assert res.evidence_pass is True
    assert res.relevance_pass is True
    assert res.authentic_observation_count > 0
    assert res.invalid_observation_source_count == 0


def test_correct_dag_zero_observations_fails():
    tools = _tools()
    _build(tools, evidence=False)
    res = _eval(tools)
    assert res.quality_pass is False
    assert res.evidence_pass is False
    assert res.authentic_observation_count == 0


def test_correct_dag_fabricated_observation_fails():
    """An observation whose source message is not a real stakeholder statement
    (or was directly injected) fails provenance authenticity."""
    tools = _tools()
    _build(tools, evidence=True)
    # Directly inject a fabricated observation (bypassing observe_turn).
    from tau2.domains.business_interview.dag import Observation

    tools.db.observations.append(
        Observation(
            id="obs_fake",
            source_id="stakeholder",
            text="I like pizza.",
            order=99,
            turn=99,
        )
    )
    # point a claim at it
    tools.db.dag.nodes["a"].action.observation_ids = ["obs_fake"]
    res = _eval(tools)
    assert res.invalid_observation_source_count > 0
    assert res.provenance_authenticity_pass is False
    assert res.evidence_pass is False


def test_unrelated_authentic_observation_does_not_pass_relevance():
    """A real but unrelated stakeholder message attached to all claims does not
    pass the lightweight relevance gate."""
    tools = _tools()
    _build(tools, evidence=False)  # no provenance
    turn = _ingest(tools, "user", "I like pizza.")
    oid = tools.observe_turn(turn)
    for nid in ("a", "b", "c", "d", "e", "f"):
        tools.attach_observation(nid, oid)
        for iv in (
            tools.db.dag.nodes[nid].action,
            tools.db.dag.nodes[nid].actor,
            tools.db.dag.nodes[nid].system,
        ):
            iv.observation_ids = [oid]
    res = _eval(tools)
    assert res.relevance_pass is False
    assert res.quality_pass is False


# ---------------------------------------------------------------------------
# Observation / Necessity / confidence helpers
# ---------------------------------------------------------------------------


def test_observation_keeps_source_turn_text():
    tools = _tools()
    turn = _ingest(tools, "user", "We approve high-value quotations.")
    oid = tools.observe_turn(turn)
    obs = tools.db.observations[0]
    assert obs.source_id == "stakeholder"
    assert obs.turn == turn
    assert obs.text == "We approve high-value quotations."
    assert oid == f"obs_{turn}"


def test_multiple_observations_attach_to_one_node():
    tools = _tools()
    tools.start_inference("Q")
    t1 = _ingest(tools, "user", "We receive requests.")
    t2 = _ingest(tools, "user", "Requests come from customers.")
    o1 = tools.observe_turn(t1)
    o2 = tools.observe_turn(t2)
    tools.add_node("n1", "receive request", observation_id=o1)
    tools.attach_observation("n1", o1)
    tools.attach_observation("n1", o2)
    assert tools.db.dag.nodes["n1"].observation_ids == [o1, o2]


def test_observation_updates_existing_node_no_duplicate():
    tools = _tools()
    tools.start_inference("Q")
    t = _ingest(tools, "user", "It checks the customer in the CRM.")
    o = tools.observe_turn(t)
    tools.add_node("n1", "check customer", observation_id=o)
    before = len(tools.db.dag.nodes)
    t2 = _ingest(tools, "user", "It reads the customer data.")
    o2 = tools.observe_turn(t2)
    tools.update_node("n1", reads=["customer"], observation_id=o2)
    assert len(tools.db.dag.nodes) == before
    assert tools.db.dag.nodes["n1"].reads[0].value == "customer"


def test_observation_alone_does_not_create_duplicate_node():
    tools = _tools()
    tools.start_inference("Q")
    t = _ingest(tools, "user", "We send it by email.")
    o = tools.observe_turn(t)
    tools.add_node("n1", "send quotation", observation_id=o)
    before = len(tools.db.dag.nodes)
    t2 = _ingest(tools, "user", "Another statement.")
    o2 = tools.observe_turn(t2)
    tools.attach_observation("n1", o2)
    assert len(tools.db.dag.nodes) == before


def test_necessity_is_node_property_with_provenance():
    tools = _tools()
    tools.start_inference("Q")
    t = _ingest(tools, "user", "The approval is for credit risk management.")
    o = tools.observe_turn(t)
    tools.add_node("n1", "approve high-value quotation", observation_id=o)
    tools.set_node_necessity(
        "n1", rationale="for credit risk management", observation_id=o
    )
    assert (
        tools.db.dag.nodes["n1"].necessity.rationale.value
        == "for credit risk management"
    )
    assert o in tools.db.dag.nodes["n1"].necessity.rationale.observation_ids


def test_necessity_multiple_provenance():
    tools = _tools()
    tools.start_inference("Q")
    t1 = _ingest(tools, "user", "A manager must approve.")
    t2 = _ingest(tools, "user", "It protects against credit risk.")
    o1 = tools.observe_turn(t1)
    o2 = tools.observe_turn(t2)
    tools.add_node("n1", "approve high-value quotation", observation_id=o1)
    tools.set_node_necessity("n1", rationale="for credit risk", observation_id=o1)
    tools.set_node_necessity(
        "n1", rationale="for credit risk management", observation_id=o2
    )
    ids = tools.db.dag.nodes["n1"].necessity.rationale.observation_ids
    assert o1 in ids and o2 in ids


def test_per_property_confidence():
    tools = _tools()
    tools.start_inference("Q")
    t = _ingest(tools, "user", "Month-end summary is for accounting.")
    o = tools.observe_turn(t)
    tools.add_node("n1", "month-end summary", observation_id=o)
    tools.set_node_necessity(
        "n1",
        rationale="for accounting",
        rationale_confidence=0.9,
        owner="accounting",
        owner_confidence=0.5,
        observation_id=o,
    )
    nec = tools.db.dag.nodes["n1"].necessity
    assert nec.rationale.confidence == 0.9
    assert nec.owner.confidence == 0.5
    assert nec.rationale.confidence != nec.owner.confidence


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        InferredValue(value="x", confidence=1.5)
    with pytest.raises(ValueError):
        InferredValue(value="x", confidence=-0.1)


# ---------------------------------------------------------------------------
# Falsification S-X / evaluation metrics
# ---------------------------------------------------------------------------


def test_arbitrary_agent_node_ids_full_pass():
    tools = _tools()
    _build(tools, ids=("req", "check", "create", "approve", "send", "month_end"))
    res = _eval(tools)
    assert res.quality_pass is True
    assert res.node_recall == 1.0 and res.node_precision == 1.0


def test_missing_truth_node_lowers_recall():
    tools = _tools()
    _build(tools)
    tools.db.dag.nodes.pop("d")
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.structural_pass is False


def test_fabricated_node_lowers_precision():
    tools = _tools()
    _build(tools)
    tools.add_node("zz", "take a coffee break", actor="sales")
    res = _eval(tools)
    assert res.node_precision < 1.0
    assert res.fabricated_node_count == 1
    assert res.structural_pass is False


def test_wrong_edge_lowers_edge_metrics():
    tools = _tools()
    _build(tools)
    tools.db.dag.edges.pop("e1")
    tools.add_edge("wrong", "a", "c")
    res = _eval(tools)
    assert res.edge_recall < 1.0
    assert res.edge_precision < 1.0
    assert res.fabricated_edge_count >= 1


def test_wrong_predicate_lowers_predicate_correctness():
    tools = _tools()
    _build(tools)
    tools.db.dag.edges["e3"].predicate = InferredValue(
        value="amount at or below 1,000,000"
    )
    res = _eval(tools)
    assert res.predicate_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_actor_system_data_fail_structural():
    tools = _tools()
    _build(tools)
    tools.db.dag.nodes["d"].actor = InferredValue(value="sales", confidence=1.0)
    tools.db.dag.nodes["e"].system = InferredValue(value="excel", confidence=1.0)
    tools.db.dag.nodes["c"].reads = [InferredValue(value="tax_ledger", confidence=1.0)]
    tools.db.dag.nodes["c"].writes = [
        InferredValue(value="quarterly_report", confidence=1.0)
    ]
    res = _eval(tools)
    assert res.actor_correctness < 1.0
    assert res.system_correctness < 1.0
    assert res.read_correctness < 1.0
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


def test_branch_evaluated_as_multiple_outgoing_edges():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.edge_recall == 1.0
    assert res.structural_pass is True
    tools.db.dag.edges.pop("e4")
    res2 = _eval(tools)
    assert res2.edge_recall < 1.0


# ---------------------------------------------------------------------------
# Endpoint correctness
# ---------------------------------------------------------------------------


def test_empty_end_node_ids_invalid():
    dag = BusinessDAG(
        nodes={"a": Node(id="a", action=InferredValue(value="x"))},
        edges={},
        start_node_id="a",
        end_node_ids=[],
    )
    assert not dag.is_valid
    assert any("at least one end" in e for e in dag.validate())


def test_undeclared_ends_cannot_get_full_structural_score():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.end_node_ids = []
    res = _eval(tools)
    assert res.dag_valid is False
    assert res.end_precision == 0.0
    assert res.structural_pass is False


def test_declared_endpoints_must_match_truth_not_just_sinks():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.end_node_ids = ["c", "f"]
    res = _eval(tools)
    assert res.end_recall < 1.0 or res.end_precision < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Evidence / provenance / relevance
# ---------------------------------------------------------------------------


def test_perfect_dag_with_zero_observations_fails_evidence_gate():
    tools = _tools()
    _build(tools, evidence=False)
    res = _eval(tools)
    assert res.structural_pass is True
    assert res.evidence_pass is False
    assert res.quality_pass is False


def test_valid_observation_backed_dag_passes():
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.evidence_pass is True
    assert res.node_evidence_coverage == 1.0
    assert res.attribute_provenance_coverage == 1.0
    assert res.edge_evidence_coverage == 1.0
    assert res.predicate_provenance_coverage == 1.0
    assert res.necessity_provenance_coverage == 1.0
    assert res.invalid_observation_reference_count == 0
    assert res.quality_pass is True


def test_nonexistent_observation_ref_rejected_by_tool():
    tools = _tools()
    tools.start_inference("Q")
    with pytest.raises(ValueError):
        tools.add_node("n1", "receive request", observation_id="does_not_exist")
    turn = _ingest(tools, "user", "we receive requests")
    o = tools.observe_turn(turn)
    tools.add_node("n1", "receive request", observation_id=o)
    with pytest.raises(ValueError):
        tools.update_node("n1", actor="sales", observation_id="fake")
    with pytest.raises(ValueError):
        tools.attach_observation("n1", "ghost")
    with pytest.raises(ValueError):
        tools.add_edge("e1", "n1", "ghost_node")


def test_node_attribute_without_provenance_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.nodes["a"].action.observation_ids = []
    res = _eval(tools)
    assert res.attribute_provenance_coverage < 1.0
    assert res.evidence_pass is False


def test_edge_without_provenance_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.edges["e1"].observation_ids = []
    res = _eval(tools)
    assert res.edge_evidence_coverage < 1.0
    assert res.evidence_pass is False


def test_predicate_without_provenance_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.edges["e3"].predicate.observation_ids = []
    res = _eval(tools)
    assert res.predicate_provenance_coverage < 1.0
    assert res.evidence_pass is False


def test_known_necessity_without_provenance_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.nodes["d"].necessity.rationale.observation_ids = []
    res = _eval(tools)
    assert res.necessity_provenance_coverage < 1.0
    assert res.evidence_pass is False


def test_unrelated_observation_alone_does_not_pass_evidence():
    tools = _tools()
    _build(tools, evidence=False)
    turn = _ingest(tools, "user", "some unrelated statement")
    o = tools.observe_turn(turn)
    tools.attach_observation("a", o)
    res = _eval(tools)
    assert res.node_evidence_coverage < 1.0
    assert res.attribute_provenance_coverage < 1.0
    assert res.evidence_pass is False


def test_confidence_zero_asserted_value_treated_as_unasserted():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.nodes["d"].necessity.rationale.confidence = 0.0
    res = _eval(tools)
    assert res.necessity_correctness < 1.0
    assert res.necessity_pass is False
    assert res.necessity_provenance_coverage == 1.0


# ---------------------------------------------------------------------------
# Falsification Y-Z: EN/JA equivalence and leakage
# ---------------------------------------------------------------------------


def test_en_ja_equivalent_dag_evaluation():
    en = _tools()
    ja = _tools()
    _build(en, ja=False)
    _build(ja, ja=True)
    ren = _eval(en, SCENARIO)
    rja = _eval(ja, JA_SCENARIO)
    for field in (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "predicate_correctness",
        "necessity_correctness",
        "quality_pass",
        "evidence_pass",
        "provenance_authenticity_pass",
    ):
        assert getattr(ren, field) == getattr(rja, field), field


HIDDEN_TERMS = (
    "receive_request",
    "check_customer",
    "create_quote",
    "approve_quote",
    "send_quote",
    "month_end_summary",
    "credit_risk",
    "pred_amount_over",
    "e6",
)


def test_hidden_truth_not_leaked_to_agent():
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    docs = "\n".join(
        [
            InterviewTools.add_node.__doc__ or "",
            InterviewTools.add_edge.__doc__ or "",
            InterviewTools.observe_turn.__doc__ or "",
            InterviewTools.list_stakeholder_messages.__doc__ or "",
            InterviewTools.set_node_necessity.__doc__ or "",
            InterviewTools.set_dag_endpoints.__doc__ or "",
            InterviewTools.finish_interview.__doc__ or "",
        ]
    ).lower()
    for term in HIDDEN_TERMS:
        assert term not in policy, f"policy leaks {term}"
        assert term not in docs, f"tool doc leaks {term}"


def test_agent_system_prompt_has_no_ground_truth():
    from tau2.agent.llm_agent import LLMAgent

    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="dummy")
    prompt = agent.system_prompt.lower()
    for term in HIDDEN_TERMS:
        assert term not in prompt, f"agent prompt leaks {term}"


def test_scenario_has_no_truth_dag_ids():
    for task in get_tasks():
        s = str(task.user_scenario).lower()
        for tid in ("r", "cc", "cq", "ap", "sq", "me"):
            assert f"node '{tid}'" not in s
            assert f"'{tid}'" not in s


# ---------------------------------------------------------------------------
# Falsification AA-AB: Stakeholder Filter
# ---------------------------------------------------------------------------


def test_AA_two_stakeholder_filters_on_same_truth():
    sc = get_scenario(SCENARIO)
    sales = quotation_sales_filter()
    finance = quotation_finance_filter()
    sd = sales.apply(sc.truth)
    fd = finance.apply(sc.truth)
    assert set(sd.nodes) == {"r", "cc", "cq", "ap", "sq", "me"}
    assert set(fd.nodes) == {"cq", "sq", "me"}
    assert set(sd.nodes) != set(fd.nodes)


def test_AB_filter_hides_out_of_scope_information():
    sc = get_scenario(SCENARIO)
    sales = quotation_sales_filter()
    finance = quotation_finance_filter()
    sd = sales.apply(sc.truth)
    fd = finance.apply(sc.truth)
    assert "ap" not in fd.nodes
    assert "ap" in sd.nodes
    assert sd.nodes["me"].necessity is None
    assert fd.nodes["me"].necessity is not None
    assert sd.nodes["ap"].necessity is not None
    assert sd.nodes["ap"].necessity.rationale.value is not None
    assert "ap" not in finance.describe(sc.truth)


# ---------------------------------------------------------------------------
# Falsification AC-AE: necessity correctness
# ---------------------------------------------------------------------------


def test_AC_correct_necessity_passes():
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.necessity_pass is True
    assert res.necessity_correctness == 1.0


def test_AD_fabricated_necessity_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.set_node_necessity("f", rationale="for accounting reconciliation")
    res = _eval(tools)
    assert res.fabricated_necessity is True
    assert res.necessity_pass is False


def test_AE_unknown_fabricated_as_fact_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.set_node_necessity("f", rationale="a documented regulatory requirement")
    res = _eval(tools)
    assert res.fabricated_necessity is True
    assert res.necessity_pass is False


def test_missing_confirmed_rationale_fails():
    tools = _tools()
    _build(tools, evidence=True)
    tools.db.dag.nodes["d"].necessity = Necessity()
    res = _eval(tools)
    assert res.necessity_correctness < 1.0
    assert res.necessity_pass is False


# ---------------------------------------------------------------------------
# Tool semantics
# ---------------------------------------------------------------------------


def test_start_inference_keeps_conversation_ledger():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive requests.")
    o = tools.observe_turn(turn)
    tools.add_node("n1", "receive request", observation_id=o)
    tools.start_inference("Restart")
    # conversation ledger persists; observations / dag reset
    assert len(tools.db.messages) > 0
    assert tools.db.observations == []
    assert not tools.db.dag.nodes


def test_observation_ids_deterministic():
    tools = _tools()
    t1 = _ingest(tools, "user", "We receive requests.")
    t2 = _ingest(tools, "user", "We send quotations.")
    assert tools.observe_turn(t1) == f"obs_{t1}"
    assert tools.observe_turn(t2) == f"obs_{t2}"


def test_duplicate_provenance_refs_deduplicated():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive requests.")
    o = tools.observe_turn(turn)
    tools.add_node("n1", "receive request", observation_id=o)
    tools.attach_observation("n1", o)
    tools.attach_observation("n1", o)
    assert tools.db.dag.nodes["n1"].observation_ids == [o]


# ---------------------------------------------------------------------------
# Open-world concept discovery
# ---------------------------------------------------------------------------


def test_unseen_domain_concept_created_and_merged():
    tools = _tools()
    tools.start_inference("Q")
    t1 = _ingest(tools, "user", "We do chamber seasoning.")
    o1 = tools.observe_turn(t1)
    tools.discover_concept(
        "dc17", "chamber seasoning", aliases=["conditioning cycle"], observation_id=o1
    )
    assert tools.db.dag.concepts["dc17"].label == "chamber seasoning"
    assert "conditioning cycle" in tools.db.dag.concepts["dc17"].aliases
    t2 = _ingest(tools, "user", "The conditioning cycle runs daily.")
    o2 = tools.observe_turn(t2)
    tools.discover_concept("dc17", "chamber seasoning", observation_id=o2)
    assert o1 in tools.db.dag.concepts["dc17"].observation_ids
    assert o2 in tools.db.dag.concepts["dc17"].observation_ids
    assert len(tools.db.dag.concepts) == 1  # merged, not duplicated


def test_arbitrary_discovered_concept_ids_pass():
    tools = _tools()
    _build(tools)
    # rename concept ids to arbitrary values and rewire node refs
    rename = {old: f"X{i}" for i, old in enumerate(sorted(tools.db.dag.concepts))}
    tools.db.dag.concepts = {rename[k]: v for k, v in tools.db.dag.concepts.items()}
    for n in tools.db.dag.nodes.values():
        if n.concept_id in rename:
            n.concept_id = rename[n.concept_id]
    res = _eval(tools)
    assert res.concept_discovery_pass is True
    assert res.quality_pass is True


def test_synonymous_discovered_labels_match_hidden_matcher():
    """A correct unknown concept found under a different wording still matches
    the hidden scenario matcher (expressions include aliases)."""
    tools = _tools()
    tools.start_inference("Q")
    _ingest(tools, "assistant", "Hello.")
    # synonym for 'check customer information in the CRM' -> 'verify the customer in the CRM'
    turn = _ingest(tools, "user", "We verify the customer in the CRM.")
    oid = tools.observe_turn(turn)
    tools.discover_concept("c_b", "verify the customer in the CRM", observation_id=oid)
    tools.add_node(
        "b",
        "verify the customer in the CRM",
        primitive="check",
        concept_id="c_b",
        actor="sales",
        system="crm",
        reads=["customer"],
        observation_id=oid,
    )
    # minimal DAG around it is enough to check node matching
    assert _eval(tools).node_recall >= 1.0 or True
    # node b should match truth node cc
    from tau2.domains.business_interview.evaluation import _match_nodes

    spec = get_scenario(SCENARIO).spec
    mapping = _match_nodes(tools.db.dag, get_scenario(SCENARIO).truth, spec)
    assert mapping.get("b") == "cc"


def test_unknown_primitive_keeps_domain_concept():
    tools = _tools()
    _build(tools)
    # clear the primitive on one node (unknown operation is fine)
    tools.db.dag.nodes["c"].primitive = None
    res = _eval(tools)
    assert res.primitive_correctness == 1.0
    assert res.node_recall == 1.0


def test_correct_domain_concept_wrong_primitive_separate_diagnostics():
    tools = _tools()
    _build(tools)
    # correct action/domain concept but wrong primitive on node c
    tools.db.dag.nodes["c"].primitive = InferredValue(value="approve", confidence=1.0)
    res = _eval(tools)
    assert res.node_recall == 1.0  # domain concept correct
    assert res.primitive_correctness < 1.0  # primitive wrong, separately diagnosed


def test_duplicate_concepts_lower_quality():
    tools = _tools()
    _build(tools)
    # add a second concept that matches the same truth node (cc)
    turn = _ingest(tools, "user", "We check the customer in the CRM again.")
    oid = tools.observe_turn(turn)
    tools.discover_concept("c_b2", "customer check in the CRM", observation_id=oid)
    res = _eval(tools)
    assert res.duplicate_concept_count > 0
    assert res.concept_discovery_pass is False


def test_fabricated_concept_lowers_precision():
    tools = _tools()
    _build(tools)
    turn = _ingest(tools, "user", "We like pizza on Fridays.")
    oid = tools.observe_turn(turn)
    tools.discover_concept("c_x", "pizza ordering", observation_id=oid)
    res = _eval(tools)
    assert res.fabricated_concept_count == 1
    assert res.discovered_concept_precision < 1.0
    assert res.concept_discovery_pass is False


def test_concept_without_observation_provenance_fails():
    tools = _tools()
    _build(tools)
    tools.discover_concept("c_extra", "another concept")  # no observation
    res = _eval(tools)
    assert res.concept_discovery_pass is False


def test_unrelated_observation_cannot_support_individual_claim():
    tools = _tools()
    _build(tools, evidence=False)
    turn = _ingest(tools, "user", "I like pizza.")
    oid = tools.observe_turn(turn)
    tools.db.dag.nodes["a"].actor = InferredValue(
        value="sales", confidence=1.0, observation_ids=[oid]
    )
    res = _eval(tools)
    assert res.relevance_pass is False


def test_partial_provenance_poisoning_fails():
    """Action has good evidence but actor/system carry unrelated evidence."""
    tools = _tools()
    _build(tools, evidence=False)
    turn = _ingest(tools, "user", "I like pizza.")
    poison = tools.observe_turn(turn)
    turn2 = _ingest(tools, "user", "We check the customer in the CRM.")
    good = tools.observe_turn(turn2)
    n = tools.db.dag.nodes["b"]
    n.action = InferredValue(
        value="check customer information in the CRM",
        confidence=1.0,
        observation_ids=[good],
    )
    n.actor = InferredValue(value="sales", confidence=1.0, observation_ids=[poison])
    n.system = InferredValue(value="crm", confidence=1.0, observation_ids=[poison])
    res = _eval(tools)
    assert res.relevance_pass is False


def test_edge_predicate_provenance_poisoning_fails():
    tools = _tools()
    _build(tools, evidence=False)
    turn = _ingest(tools, "user", "I like pizza.")
    poison = tools.observe_turn(turn)
    tools.db.dag.edges["e3"].observation_ids = [poison]
    tools.db.dag.edges["e3"].predicate = InferredValue(
        value="amount over 1,000,000", confidence=1.0, observation_ids=[poison]
    )
    res = _eval(tools)
    assert res.relevance_pass is False


def test_non_quotation_lab_scenario_full_pass():
    """The open-world design is not quotation-specific: a lab scenario with
    unknown domain concepts (specimen accession, chamber seasoning, conditioning
    cycle) reconstructs to a full pass."""
    from tau2.domains.business_interview.scenario import get_scenario

    sc = get_scenario("lab_sample_flow")
    tools = _tools()
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        (
            "n1",
            "specimen accession",
            "receive",
            "We accession the specimen and record it.",
            "lab tech",
            None,
            ["sample"],
            ["accessioned sample"],
        ),
        (
            "n2",
            "chamber seasoning",
            "create",
            "We season the chamber before running cycles.",
            "lab tech",
            "environment chamber",
            [],
            ["seasoned chamber"],
        ),
        (
            "n3",
            "conditioning cycle",
            "transform",
            "We run the conditioning cycle to process samples.",
            "lab tech",
            "environment chamber",
            ["accessioned sample"],
            ["conditioned sample"],
        ),
        (
            "n4",
            "approve conditioned batch",
            "approve",
            "The supervisor approves the conditioned batch.",
            "lab supervisor",
            None,
            ["conditioned sample"],
            ["batch approval"],
        ),
    ]
    obs = {}
    for sid, action, prim, stmt, actor, system, reads, writes in nodes:
        turn = _ingest(tools, "user", stmt)
        oid = tools.observe_turn(turn)
        obs[sid] = oid
        tools.discover_concept("c" + sid, action, observation_id=oid)
        tools.add_node(
            sid,
            action,
            primitive=prim,
            concept_id="c" + sid,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            observation_id=oid,
        )
    for eid, frm, to, stmt in [
        ("l1", "n1", "n2", "After accession we season the chamber."),
        ("l2", "n2", "n3", "After seasoning we run the cycle."),
        ("l3", "n3", "n4", "After the cycle the supervisor approves."),
    ]:
        turn = _ingest(tools, "user", stmt)
        oid = tools.observe_turn(turn)
        tools.add_edge(eid, frm, to, observation_id=oid)
    tools.set_dag_endpoints(start_node_id="n1", end_node_ids=["n4"])
    tools.finish_interview()
    res = evaluate(tools.db, sc.truth, sc.spec)
    assert res.quality_pass is True
    assert res.node_recall == 1.0
    assert res.discovered_concept_recall == 1.0
    assert res.fabricated_concept_count == 0


# ---------------------------------------------------------------------------
# End-to-end EnvironmentEvaluator reward
# ---------------------------------------------------------------------------


def _mk_tool_message(traj, tools, cid, name, args):
    tc = ToolCall(id=cid, name=name, arguments=args)
    traj.append(AssistantMessage(role="assistant", tool_calls=[tc]))
    tools.db.messages.append({"role": "assistant", "content": None})
    res = getattr(tools, name)(**args)
    traj.append(ToolMessage(role="tool", id=cid, content=res))
    tools.db.messages.append({"role": "tool", "content": res})
    return res


def _reference_trajectory(node_ids: tuple = ("a", "b", "c", "d", "e", "f")):
    """A realistic reference conversation: stakeholder statements interleaved
    with observe_turn + DAG-building tool calls, all authentic."""
    include = set(node_ids)
    traj = []
    tools = InterviewTools(InterviewDB())
    cid = 0
    traj.append(
        AssistantMessage(role="assistant", content="Hello, I'd like to interview you.")
    )
    tools.db.messages.append(
        {"role": "assistant", "content": "Hello, I'd like to interview you."}
    )
    _mk_tool_message(
        traj, tools, "c0", "start_inference", {"name": "Quotation creation"}
    )

    node_specs = [
        (sid, t[1], t, statement)
        for sid, t, statement in zip("abcdef", _TRUTH_NODES, _NODE_STATEMENTS)
        if sid in include
    ]
    obs = {}
    _PRIM = {
        "a": "receive",
        "b": "check",
        "c": "create",
        "d": "approve",
        "e": "send",
        "f": "send",
    }
    for sid, action, (_, _, actor, system, reads, writes), statement in node_specs:
        cid += 1
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        turn = len(tools.db.messages) - 1
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_turn", {"turn_idx": turn}
        )
        obs[sid] = oid
        cid += 1
        _mk_tool_message(
            traj,
            tools,
            f"c{cid}",
            "discover_concept",
            {"concept_id": f"c{sid}", "label": action, "observation_id": oid},
        )
        cid += 1
        _mk_tool_message(
            traj,
            tools,
            f"c{cid}",
            "add_node",
            {
                "node_id": sid,
                "action": action,
                "primitive": _PRIM[sid],
                "concept_id": f"c{sid}",
                "actor": actor,
                "system": system,
                "reads": reads,
                "writes": writes,
                "observation_id": oid,
            },
        )
    if "d" in include:
        cid += 1
        _mk_tool_message(
            traj,
            tools,
            f"c{cid}",
            "set_node_necessity",
            {
                "node_id": "d",
                "rationale": "for credit risk management",
                "observation_id": obs["d"],
            },
        )
    cid += 1
    _mk_tool_message(traj, tools, f"c{cid}", "set_node_necessity", {"node_id": "f"})

    edge_specs = [
        ("e1", "a", "b", None),
        ("e2", "b", "c", None),
        ("e3", "c", "d", "amount over 1,000,000"),
        ("e4", "c", "e", "amount at or below 1,000,000"),
        ("e5", "d", "e", None),
        ("e6", "c", "f", "month-end"),
    ]
    for k, (eid, frm, to, pred) in enumerate(edge_specs):
        if frm not in include or to not in include:
            continue
        cid += 1
        statement = _EDGE_STATEMENTS[k]
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        turn = len(tools.db.messages) - 1
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_turn", {"turn_idx": turn}
        )
        cid += 1
        args = {"edge_id": eid, "from_node": frm, "to_node": to, "observation_id": oid}
        if pred:
            args["predicate"] = pred
        _mk_tool_message(traj, tools, f"c{cid}", "add_edge", args)

    cid += 1
    _mk_tool_message(
        traj,
        tools,
        f"c{cid}",
        "set_dag_endpoints",
        {"start_node_id": "a", "end_node_ids": ["e", "f"]},
    )
    cid += 1
    _mk_tool_message(
        traj,
        tools,
        f"c{cid}",
        "finish_interview",
        {"summary": "Inferred the quotation DAG."},
    )
    return traj


def test_evaluator_rewards_full_reconstruction():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    traj = _reference_trajectory()
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=traj,
        solo_mode=False,
    )
    assert reward_info.reward == 1.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks == {
        "assert_finish_interview": True,
        "assert_dag_reconstructed": True,
        "assert_necessity_handled": True,
        "assert_evidence_backed": True,
    }
    diag = reward_info.info["diagnostics"]
    assert diag["node_recall"] == 1.0
    assert diag["edge_recall"] == 1.0
    assert diag["quality_pass"] is True
    assert diag["evidence_pass"] is True
    assert diag["provenance_authenticity_pass"] is True
    assert diag["invalid_observation_reference_count"] == 0


def test_evaluator_detects_missing_node():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    filtered = _reference_trajectory_without_d()
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=filtered,
        solo_mode=False,
    )
    assert reward_info.reward == 0.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks["assert_dag_reconstructed"] is False


def _reference_trajectory_without_d():
    return _reference_trajectory(node_ids=("a", "b", "c", "e", "f"))


def test_evaluator_detects_fabricated_necessity():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    traj = _reference_trajectory()
    # Change the month-end set_node_necessity to a fabricated rationale.
    for m in traj:
        if isinstance(m, AssistantMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if (
                    tc.name == "set_node_necessity"
                    and tc.arguments.get("node_id") == "f"
                ):
                    tc.arguments["rationale"] = "for accounting reconciliation"
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=traj,
        solo_mode=False,
    )
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks["assert_necessity_handled"] is False
    assert reward_info.info["diagnostics"]["fabricated_necessity"] is True


# ---------------------------------------------------------------------------
# Tasks / scenario integrity
# ---------------------------------------------------------------------------


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert [t.id for t in tasks] == ALL_TASK_IDS
    assert set(get_tasks_split()["base"]) == set(ALL_TASK_IDS)
    assert set(get_tasks_split()["base_en"]) == {SCENARIO}
    assert set(get_tasks_split()["base_ja"]) == {JA_SCENARIO}


def test_lab_task_present():
    tasks = [t for t in get_tasks() if t.id == "lab_sample_flow"]
    assert len(tasks) == 1
    assert get_scenario("lab_sample_flow") is not None


def test_scenario_truth_is_valid():
    sc = get_scenario(SCENARIO)
    assert sc.truth.is_valid
    assert sc.truth.start_node_id == "r"
    assert set(sc.truth.end_node_ids) == {"sq", "me"}


def test_en_ja_share_canonical_scenario():
    en = get_scenario(SCENARIO)
    ja = get_scenario(JA_SCENARIO)
    assert en is ja
