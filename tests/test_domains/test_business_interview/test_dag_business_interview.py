"""Tests for the evidence-backed DAG business_interview domain (v3).

This replaces the old Step/Transition/Branch workflow tests. It covers the
BusinessDAG model (shared by Truth and Agent result), Observation-as-evidence,
incremental node update, per-attribute confidence / provenance, necessity as a
node property, the Stakeholder filter, the evaluator metrics, EN/JA equivalence,
leakage, and the falsification suite A-AE.
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
from tau2.data_model.tasks import Task
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
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO]

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


def _tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    sc = get_scenario(scenario)
    return evaluate(tools.db, sc.truth, sc.spec)


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("a", "b", "c", "d", "e", "f"),
    edge_ids: tuple = ("e1", "e2", "e3", "e4", "e5", "e6"),
):
    """Build the correct quotation DAG using (possibly arbitrary) ids."""
    i1, i2, i3, i4, i5, i6 = ids
    actions = _JA_ACTIONS if ja else [t[1] for t in _TRUTH_NODES]
    for (sid, _, actor, system, reads, writes), action in zip(_TRUTH_NODES, actions):
        tools.add_node(
            {"a": i1, "b": i2, "c": i3, "d": i4, "e": i5, "f": i6}[sid],
            action,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
        )
    tools.set_node_necessity(
        i4, rationale="与信リスク管理のため" if ja else "for credit risk management"
    )
    tools.set_node_necessity(i6)  # all unknown
    te1, te2, te3, te4, te5, te6 = edge_ids
    tools.add_edge(te1, i1, i2)
    tools.add_edge(te2, i2, i3)
    tools.add_edge(
        te3, i3, i4, predicate="100万円超" if ja else "amount over 1,000,000"
    )
    tools.add_edge(
        te4, i3, i5, predicate="100万円以下" if ja else "amount at or below 1,000,000"
    )
    tools.add_edge(te5, i4, i5)
    tools.add_edge(te6, i3, i6, predicate="月末" if ja else "month-end")
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
    # no Node-level condition concept
    assert not hasattr(scmod, "StakeholderStepTruth")


def test_H_no_branch_class():
    assert not hasattr(dagmod, "Branch")


def test_I_no_transition_class():
    assert not hasattr(dagmod, "Transition")


def test_J_no_step_class():
    assert not hasattr(dagmod, "Step")
    assert not hasattr(dagmod, "WorkflowStep")


# ---------------------------------------------------------------------------
# Falsification K-N: Observation
# ---------------------------------------------------------------------------


def test_K_multiple_observations_attach_to_one_node():
    tools = _tools()
    tools.start_inference("Q")
    tools.record_observation("We receive requests.")
    tools.record_observation("Requests come from customers.")
    tools.add_node("n1", "receive request", observation_id="o1")
    tools.attach_observation("n1", "o1")
    tools.attach_observation("n1", "o2")
    assert tools.db.dag.nodes["n1"].observation_ids == ["o1", "o2"]
    assert len([o for o in tools.db.observations if o.id in ("o1", "o2")]) == 2


def test_L_observation_update_existing_node_no_duplicate():
    tools = _tools()
    tools.start_inference("Q")
    tools.record_observation("The step checks the customer in the CRM.")
    tools.add_node("n1", "check customer", actor="sales", system="crm")
    before = len(tools.db.dag.nodes)
    tools.record_observation("It reads the customer data.")
    tools.update_node("n1", reads=["customer"], observation_id="o2")
    after = len(tools.db.dag.nodes)
    assert before == after == 1  # no duplicate node
    assert tools.db.dag.nodes["n1"].reads[0].value == "customer"


def test_M_observation_alone_does_not_create_duplicate_node():
    tools = _tools()
    tools.start_inference("Q")
    tools.add_node("n1", "send quotation")
    before = len(tools.db.dag.nodes)
    tools.record_observation("We send it by email.")
    tools.attach_observation("n1", "o1")
    assert len(tools.db.dag.nodes) == before  # no new node created
    assert "o1" in tools.db.dag.nodes["n1"].observation_ids


def test_N_observation_keeps_source_and_text():
    tools = _tools()
    tools.start_inference("Q")
    tools.record_observation(
        "We approve high-value quotes.", source_id="manager", locale="en"
    )
    obs = tools.db.observations[0]
    assert obs.source_id == "manager"
    assert obs.text == "We approve high-value quotes."
    assert obs.locale == "en"


# ---------------------------------------------------------------------------
# Falsification O-R: Necessity / confidence
# ---------------------------------------------------------------------------


def test_O_necessity_is_node_property():
    tools = _tools()
    tools.start_inference("Q")
    tools.add_node("n1", "approve high-value quotation")
    tools.set_node_necessity("n1", rationale="for credit risk management")
    assert (
        tools.db.dag.nodes["n1"].necessity.rationale.value
        == "for credit risk management"
    )


def test_P_necessity_has_multiple_observation_provenance():
    tools = _tools()
    tools.start_inference("Q")
    tools.record_observation("Manager must approve over 1M.")
    tools.record_observation("The approval protects against credit risk.")
    tools.add_node("n1", "approve high-value quotation")
    tools.set_node_necessity("n1", rationale="for credit risk", observation_id="o1")
    tools.set_node_necessity(
        "n1", rationale="for credit risk management", observation_id="o2"
    )
    ids = tools.db.dag.nodes["n1"].necessity.rationale.observation_ids
    assert "o1" in ids and "o2" in ids


def test_Q_per_property_confidence():
    tools = _tools()
    tools.start_inference("Q")
    tools.add_node("n1", "month-end summary")
    tools.set_node_necessity(
        "n1",
        rationale="for accounting",
        rationale_confidence=0.9,
        owner="accounting",
        owner_confidence=0.5,
    )
    nec = tools.db.dag.nodes["n1"].necessity
    assert nec.rationale.confidence == 0.9
    assert nec.owner.confidence == 0.5
    assert nec.rationale.confidence != nec.owner.confidence


def test_R_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        InferredValue(value="x", confidence=1.5)
    with pytest.raises(ValueError):
        InferredValue(value="x", confidence=-0.1)


# ---------------------------------------------------------------------------
# Falsification S-X: DAG evaluation
# ---------------------------------------------------------------------------


def test_S_arbitrary_agent_node_ids_full_pass():
    tools = _tools()
    _build(tools, ids=("req", "check", "create", "approve", "send", "month_end"))
    res = _eval(tools)
    assert res.quality_pass is True
    assert res.node_recall == 1.0 and res.node_precision == 1.0


def test_T_missing_truth_node_lowers_recall():
    tools = _tools()
    _build(tools)
    tools.db.dag.nodes.pop("d")  # drop approve node
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.structural_pass is False


def test_U_fabricated_node_lowers_precision():
    tools = _tools()
    _build(tools)
    tools.add_node("zz", "take a coffee break", actor="sales")
    res = _eval(tools)
    assert res.node_precision < 1.0
    assert res.fabricated_node_count == 1
    assert res.structural_pass is False


def test_V_wrong_edge_lowers_edge_metrics():
    tools = _tools()
    _build(tools)
    # wrong edge: a -> c (skip check)
    tools.db.dag.edges.pop("e1")
    tools.add_edge("wrong", "a", "c")
    res = _eval(tools)
    assert res.edge_recall < 1.0
    assert res.edge_precision < 1.0
    assert res.fabricated_edge_count >= 1


def test_W_wrong_predicate_lowers_predicate_correctness():
    tools = _tools()
    _build(tools)
    tools.db.dag.edges["e3"].predicate = InferredValue(
        value="amount at or below 1,000,000"
    )
    res = _eval(tools)
    assert res.predicate_correctness < 1.0
    assert res.structural_pass is False


def test_X_branch_evaluated_as_multiple_outgoing_edges():
    tools = _tools()
    _build(tools)
    # create (c) has two conditional outgoing edges to approve (d) and send (e)
    res = _eval(tools)
    assert res.edge_recall == 1.0
    assert res.structural_pass is True
    # dropping the low-value path lowers edge recall
    tools.db.dag.edges.pop("e4")
    res2 = _eval(tools)
    assert res2.edge_recall < 1.0


# ---------------------------------------------------------------------------
# Falsification Y-Z: EN/JA equivalence and leakage
# ---------------------------------------------------------------------------


def test_Y_en_ja_equivalent_dag_evaluation():
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


def test_Z_hidden_truth_not_leaked_to_agent():
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    docs = "\n".join(
        [
            InterviewTools.add_node.__doc__ or "",
            InterviewTools.add_edge.__doc__ or "",
            InterviewTools.record_observation.__doc__ or "",
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
            # node ids are short letters; check they don't appear as identifiers
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
    # finance filter drops the approval node
    assert "ap" not in fd.nodes
    assert "ap" in sd.nodes
    # month-end necessity is hidden from the sales stakeholder but visible to finance
    assert sd.nodes["me"].necessity is None
    assert fd.nodes["me"].necessity is not None
    # approval rationale is visible to sales
    assert sd.nodes["ap"].necessity is not None
    assert sd.nodes["ap"].necessity.rationale.value is not None
    # finance does not leak the approval rationale
    assert "ap" not in finance.describe(sc.truth)


# ---------------------------------------------------------------------------
# Falsification AC-AE: necessity correctness
# ---------------------------------------------------------------------------


def test_AC_correct_necessity_passes():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.necessity_pass is True
    assert res.necessity_correctness == 1.0


def test_AD_fabricated_necessity_fails():
    tools = _tools()
    _build(tools)
    # fabricate a rationale for the month-end node whose truth is unknown
    tools.set_node_necessity("f", rationale="for accounting reconciliation")
    res = _eval(tools)
    assert res.fabricated_necessity is True
    assert res.necessity_pass is False


def test_AE_unknown_fabricated_as_fact_fails():
    tools = _tools()
    _build(tools)
    tools.set_node_necessity(
        "f", rationale="a documented regulatory requirement", rationale_confidence=1.0
    )
    res = _eval(tools)
    assert res.fabricated_necessity is True
    assert res.necessity_pass is False


def test_missing_confirmed_rationale_fails():
    tools = _tools()
    _build(tools)
    # approve node records no rationale (all properties unset)
    tools.db.dag.nodes["d"].necessity = Necessity()
    res = _eval(tools)
    assert res.necessity_correctness < 1.0
    assert res.necessity_pass is False


# ---------------------------------------------------------------------------
# Confidence / provenance in diagnostics
# ---------------------------------------------------------------------------


def test_confidence_stored_and_validated_in_tools():
    tools = _tools()
    tools.start_inference("Q")
    tools.add_node(
        "n1", "approve high-value quotation", actor="manager", confidence=0.7
    )
    assert tools.db.dag.nodes["n1"].action.confidence == 0.7
    assert tools.db.dag.nodes["n1"].actor.confidence == 0.7
    with pytest.raises(ValueError):
        tools.add_node("n2", "x", confidence=2.0)


def test_observation_provenance_present():
    tools = _tools()
    tools.start_inference("Q")
    o1 = tools.record_observation("We receive requests from customers.")
    tools.add_node("a", "receive quotation request", observation_id=o1)
    assert tools.db.dag.nodes["a"].observation_ids == [o1]
    assert tools.db.dag.nodes["a"].action.observation_ids == [o1]


# ---------------------------------------------------------------------------
# End-to-end EnvironmentEvaluator reward
# ---------------------------------------------------------------------------


def _tool_call(cid: str, name: str, args: dict, result: str) -> list:
    return [
        AssistantMessage(
            role="assistant", tool_calls=[ToolCall(id=cid, name=name, arguments=args)]
        ),
        ToolMessage(role="tool", id=cid, content=result),
    ]


def _trajectory_from_reference(task: Task):
    traj = [
        AssistantMessage(role="assistant", content="Hello, I'd like to interview you."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(InterviewDB())
    for i, a in enumerate(task.evaluation_criteria.actions):
        res = getattr(tools, a.name)(**a.arguments)
        traj += _tool_call(f"c{i}", a.name, a.arguments, res)
    traj += [
        AssistantMessage(role="assistant", content="That completes the interview.")
    ]
    return traj


def test_evaluator_rewards_full_reconstruction():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_trajectory_from_reference(task),
        solo_mode=False,
    )
    assert reward_info.reward == 1.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks == {
        "assert_finish_interview": True,
        "assert_dag_reconstructed": True,
        "assert_necessity_handled": True,
    }
    diag = reward_info.info["diagnostics"]
    assert diag["node_recall"] == 1.0
    assert diag["edge_recall"] == 1.0
    assert diag["quality_pass"] is True


def test_evaluator_detects_missing_node():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    keep = []
    drop_ids = {"inf_9", "inf_10", "inf_18", "inf_20"}  # drop approve node + its refs
    for a in task.evaluation_criteria.actions:
        if a.action_id not in drop_ids:
            keep.append((a.name, a.arguments))
    traj = [
        AssistantMessage(role="assistant", content="Hello."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(InterviewDB())
    for i, (name, args) in enumerate(keep):
        res = getattr(tools, name)(**args)
        traj += _tool_call(f"c{i}", name, args, res)
    traj += [AssistantMessage(role="assistant", content="done")]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=traj,
        solo_mode=False,
    )
    assert reward_info.reward == 0.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks["assert_dag_reconstructed"] is False


def test_evaluator_detects_fabricated_necessity():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    actions = [a for a in task.evaluation_criteria.actions]
    # replace month-end necessity with a fabricated rationale
    for a in actions:
        if a.name == "set_node_necessity" and a.arguments.get("node_id") == "f":
            a.arguments["rationale"] = "for accounting reconciliation"
    traj = [
        AssistantMessage(role="assistant", content="Hello."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(InterviewDB())
    for i, a in enumerate(actions):
        res = getattr(tools, a.name)(**a.arguments)
        traj += _tool_call(f"c{i}", a.name, a.arguments, res)
    traj += [AssistantMessage(role="assistant", content="done")]
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


def test_scenario_truth_is_valid():
    sc = get_scenario(SCENARIO)
    assert sc.truth.is_valid
    assert sc.truth.start_node_id == "r"
    assert set(sc.truth.end_node_ids) == {"sq", "me"}


def test_en_ja_share_canonical_scenario():
    en = get_scenario(SCENARIO)
    ja = get_scenario(JA_SCENARIO)
    assert en is ja
