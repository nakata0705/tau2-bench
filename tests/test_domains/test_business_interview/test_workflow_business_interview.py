"""Tests for the Workflow-first business_interview domain (v2).

Covers the workflow-reconstruction tools, the structural/epistemic/challenge
evaluation, the leakage-free agent-visible context, the workflow falsification
suite (A-L), and an end-to-end EnvironmentEvaluator check.
"""

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import WorkflowDB
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.ground_truth import (
    get_ground_truth,
)
from tau2.domains.business_interview.semantic import evaluate
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
)

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO]


def _tools() -> InterviewTools:
    return InterviewTools(WorkflowDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    return evaluate(tools.db, scenario)


# ---------------------------------------------------------------------------
# Good reconstruction (reference behaviour)
# ---------------------------------------------------------------------------


def _good_workflow() -> InterviewTools:
    """A correct reconstruction: all steps, actors, systems, data, transitions,
    branches, the confirmed rationale, the UNKNOWN rationale, and a challenge."""
    tools = _tools()
    tools.create_workflow(
        "Quotation creation",
        trigger="customer requests a quotation",
        purpose="produce an accurate quotation",
        outcome="customer receives a quotation",
    )
    steps = [
        ("s1", "receive quotation request", "sales", None, [], ["request"], None),
        (
            "s2",
            "check customer information in the CRM",
            "sales",
            "crm",
            ["customer"],
            [],
            None,
        ),
        (
            "s3",
            "create quotation in the quoting system",
            "sales",
            "quoting",
            ["customer", "pricing"],
            ["quote"],
            None,
        ),
        (
            "s4",
            "approve high-value quotation",
            "manager",
            "quoting",
            ["quote"],
            ["approval"],
            "amount over 1,000,000",
        ),
        (
            "s5",
            "send quotation to customer",
            "sales",
            "email",
            ["quote"],
            ["sent_quote"],
            None,
        ),
        (
            "s6",
            "send quotation summary to accounting at month-end",
            "sales",
            "excel",
            ["quote"],
            ["excel_summary"],
            "month-end",
        ),
    ]
    for sid, act, actor, system, reads, writes, cond in steps:
        tools.add_step(
            sid,
            act,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            condition=cond,
        )
    for a, b, c in [
        ("s1", "s2", None),
        ("s2", "s3", None),
        ("s3", "s4", "amount over 1,000,000"),
        ("s3", "s5", "amount at or below 1,000,000"),
        ("s4", "s5", None),
        ("s5", "s6", "month-end"),
    ]:
        tools.connect_steps(a, b, c)
    tools.add_branch("s3", "amount threshold", ["s4", "s5"])
    tools.set_step_rationale("s4", "for credit risk management", "FACT", source="sales")
    tools.set_step_unknown("s6", "the stakeholder does not know why")
    tools.challenge_step("s6", "is this month-end excel step actually necessary?")
    tools.finish_interview()
    return tools


def test_good_reconstruction_passes_all_axes():
    tools = _good_workflow()
    ev = _eval(tools)
    assert ev.protocol_completed is True
    assert ev.step_recall == 1.0
    assert ev.unexpected_step_count == 0
    assert ev.actor_accuracy == 1.0
    assert ev.system_accuracy == 1.0
    assert ev.data_read_accuracy == 1.0
    assert ev.data_write_accuracy == 1.0
    assert ev.transition_accuracy == 1.0
    assert ev.branch_recall == 1.0
    assert ev.branch_condition_accuracy == 1.0
    assert ev.rationale_coverage == 1.0
    assert ev.uncertainty_handling is True
    assert ev.fabricated_rationale is False
    assert ev.challenge_done is True
    assert ev.improvement_order_ok is True
    assert ev.structural_pass is True
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.quality_pass is True
    assert ev.protocol_pass is True


# ---------------------------------------------------------------------------
# Leakage: agent-visible policy / tools must not reveal the hidden workflow
# ---------------------------------------------------------------------------

HIDDEN_TERMS = (
    "month_end_excel",
    "high_value_quote",
    "credit risk",
    "accounting need",
    "approve high-value",
    "month-end excel",
    "1,000,000",
    "amount over",
)


def test_L1_policy_has_no_hidden_workflow_identity():
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    for term in HIDDEN_TERMS:
        assert term not in policy, f"policy leaks hidden term: {term}"
    # General workflow guidance is present.
    assert "workflow" in policy
    assert "branch" in policy
    assert "necessity" in policy


def test_L2_tool_descriptions_have_no_hidden_workflow_identity():
    docs = "\n".join(
        [
            InterviewTools.create_workflow.__doc__ or "",
            InterviewTools.add_step.__doc__ or "",
            InterviewTools.connect_steps.__doc__ or "",
            InterviewTools.add_branch.__doc__ or "",
            InterviewTools.set_step_rationale.__doc__ or "",
            InterviewTools.set_step_unknown.__doc__ or "",
            InterviewTools.challenge_step.__doc__ or "",
            InterviewTools.propose_improvement.__doc__ or "",
            InterviewTools.finish_interview.__doc__ or "",
        ]
    ).lower()
    for term in HIDDEN_TERMS:
        assert term not in docs, f"tool description leaks hidden term: {term}"
    assert "condition" in docs


def test_ground_truth_is_evaluator_only_not_in_scenario():
    """The hidden workflow (steps, branch answer, rationale, challenge target)
    must not appear in the stakeholder scenario text the agent could see."""
    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    scenario = str(task.user_scenario).lower()
    # The confirmed rationale (credit risk) is something the stakeholder knows,
    # but the canonical step ids and the branch/step structure must not be given.
    assert "s1" not in scenario and "s4" not in scenario
    gt = get_ground_truth(SCENARIO)
    assert gt is not None
    # The stakeholder text should not spell out the full step list / branch answer.
    assert "amount over 1,000,000" not in scenario


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert [t.id for t in tasks] == ALL_TASK_IDS
    assert set(get_tasks_split()["base"]) == set(ALL_TASK_IDS)
    assert set(get_tasks_split()["base_en"]) == {SCENARIO}
    assert set(get_tasks_split()["base_ja"]) == {JA_SCENARIO}


def test_en_ja_share_canonical_ground_truth():
    en = get_ground_truth(SCENARIO)
    ja = get_ground_truth(JA_SCENARIO)
    assert en is not None and ja is not None
    assert en.scenario_id == "quotation_workflow_1"
    assert [s.id for s in en.steps] == [s.id for s in ja.steps]


# ---------------------------------------------------------------------------
# Workflow reconstruction falsification suite (A-L)
# ---------------------------------------------------------------------------


def _base_steps(tools: InterviewTools):
    """Add the six steps with correct attributes (no transitions/branches yet)."""
    steps = [
        ("s1", "receive quotation request", "sales", None, [], ["request"], None),
        (
            "s2",
            "check customer information in the CRM",
            "sales",
            "crm",
            ["customer"],
            [],
            None,
        ),
        (
            "s3",
            "create quotation in the quoting system",
            "sales",
            "quoting",
            ["customer", "pricing"],
            ["quote"],
            None,
        ),
        (
            "s4",
            "approve high-value quotation",
            "manager",
            "quoting",
            ["quote"],
            ["approval"],
            "amount over 1,000,000",
        ),
        (
            "s5",
            "send quotation to customer",
            "sales",
            "email",
            ["quote"],
            ["sent_quote"],
            None,
        ),
        (
            "s6",
            "send quotation summary to accounting at month-end",
            "sales",
            "excel",
            ["quote"],
            ["excel_summary"],
            "month-end",
        ),
    ]
    for sid, act, actor, system, reads, writes, cond in steps:
        tools.add_step(
            sid,
            act,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            condition=cond,
        )


def _good_edges(tools: InterviewTools):
    for a, b, c in [
        ("s1", "s2", None),
        ("s2", "s3", None),
        ("s3", "s4", "amount over 1,000,000"),
        ("s3", "s5", "amount at or below 1,000,000"),
        ("s4", "s5", None),
        ("s5", "s6", "month-end"),
    ]:
        tools.connect_steps(a, b, c)
    tools.add_branch("s3", "amount threshold", ["s4", "s5"])


def _good_rationale_challenge(tools: InterviewTools):
    tools.set_step_rationale("s4", "for credit risk management", "FACT", source="sales")
    tools.set_step_unknown("s6", "the stakeholder does not know why")
    tools.challenge_step("s6", "is this step actually necessary?")


def test_falsification_A_structural_pass():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.quality_pass is True
    assert ev.structural_pass is True


def test_falsification_B_step_missing_fails_step_recall():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    # Drop step s2 entirely.
    tools.db.workflow.steps = [s for s in tools.db.workflow.steps if s.id != "s2"]
    _good_edges(tools)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.step_recall < 1.0  # <-- step recall FAIL
    assert ev.structural_pass is False


def test_falsification_C_wrong_order_fails_transition():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    # Wrong order: s3 -> s1 (instead of s1 -> s2 -> s3).
    for a, b, c in [
        ("s3", "s1", None),
        ("s1", "s2", None),
        ("s2", "s4", "amount over 1,000,000"),
        ("s2", "s5", "amount at or below 1,000,000"),
        ("s4", "s5", None),
        ("s5", "s6", "month-end"),
    ]:
        tools.connect_steps(a, b, c)
    tools.add_branch("s3", "amount threshold", ["s4", "s5"])
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0  # <-- transition FAIL
    assert ev.structural_pass is False


def test_falsification_D_branch_as_linear_fails_branch():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    # Record the flow as linear: s3 -> s4 only; the other branch path is missing.
    for a, b, c in [
        ("s1", "s2", None),
        ("s2", "s3", None),
        ("s3", "s4", None),
        ("s4", "s5", None),
        ("s5", "s6", "month-end"),
    ]:
        tools.connect_steps(a, b, c)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.branch_recall < 1.0  # <-- branch FAIL
    assert ev.structural_pass is False


def test_falsification_E_wrong_actor_fails_actor():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    # s4 is approved by sales instead of manager.
    _base_steps(tools)
    tools.db.workflow.steps = [
        s if s.id != "s4" else s.model_copy(update={"actor": "sales"})
        for s in tools.db.workflow.steps
    ]
    _good_edges(tools)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.actor_accuracy < 1.0  # <-- actor FAIL
    assert ev.structural_pass is False


def test_falsification_F_wrong_system_fails_system():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    # s3 uses the wrong system (email instead of quoting).
    tools.db.workflow.steps = [
        s if s.id != "s3" else s.model_copy(update={"system": "email"})
        for s in tools.db.workflow.steps
    ]
    _good_edges(tools)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.system_accuracy < 1.0  # <-- system FAIL
    assert ev.structural_pass is False


def test_falsification_G_wrong_data_fails_data():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    # s3 writes 'approval' instead of 'quote' (read/write mix-up).
    tools.db.workflow.steps = [
        s if s.id != "s3" else s.model_copy(update={"writes": ["approval"]})
        for s in tools.db.workflow.steps
    ]
    _good_edges(tools)
    _good_rationale_challenge(tools)
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.data_write_accuracy < 1.0  # <-- data FAIL
    assert ev.structural_pass is False


def test_falsification_H_rationale_not_checked_fails_coverage():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    # The confirmed rationale (s4) is never recorded.
    tools.set_step_unknown("s6", "the stakeholder does not know why")
    tools.challenge_step("s6", "is this step actually necessary?")
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.rationale_coverage < 1.0  # <-- rationale coverage FAIL
    assert ev.rationale_pass is False


def test_falsification_I_unknown_rationale_fabricated_fails_epistemic():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    tools.set_step_rationale("s4", "for credit risk management", "FACT", source="sales")
    # WRONG: the UNKNOWN rationale of s6 is fabricated as a FACT.
    tools.set_step_rationale("s6", "for accounting reconciliation", "FACT")
    tools.challenge_step("s6", "is this step actually necessary?")
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.fabricated_rationale is True  # <-- epistemic FAIL
    assert ev.uncertainty_handling is False
    assert ev.rationale_pass is False


def test_falsification_J_legacy_not_challenged_fails_challenge():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    _good_rationale_challenge(tools)
    # Remove the challenge on the questionable step s6.
    for s in tools.db.workflow.steps:
        s.necessity.challenged = False
        s.necessity.challenges = []
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.challenge_done is False  # <-- challenge FAIL
    assert ev.challenge_pass is False


def test_falsification_K_automate_without_questioning_fails_improvement_order():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    _good_rationale_challenge(tools)
    # No challenge on s6, and an automation proposal without questioning first.
    for s in tools.db.workflow.steps:
        s.necessity.challenged = False
        s.necessity.challenges = []
    tools.propose_improvement("s6", "automate", "let us RPA the excel step")
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.improvement_order_ok is False  # <-- improvement-order FAIL
    assert ev.challenge_pass is False


def test_falsification_L_semantic_ok_protocol_fail():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    _good_rationale_challenge(tools)
    # No finish_interview.
    ev = _eval(tools)
    assert ev.quality_pass is True
    assert ev.protocol_pass is False
    assert ev.protocol_completed is False


def test_improvement_order_question_before_automate_ok():
    tools = _tools()
    tools.create_workflow("Quotation creation")
    _base_steps(tools)
    _good_edges(tools)
    _good_rationale_challenge(tools)
    # Question first, then automate: valid order.
    tools.propose_improvement("s6", "question", "is this step needed?")
    tools.propose_improvement("s6", "automate", "then automate if needed")
    tools.finish_interview()
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


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
    """Execute the reference actions to build a trajectory with real tool results."""
    traj = [
        AssistantMessage(role="assistant", content="Hello, I'd like to interview you."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(WorkflowDB())
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
        "assert_workflow_reconstructed": True,
        "assert_rationale_handled": True,
        "assert_necessity_challenged": True,
    }
    assert reward_info.info is not None
    diag = reward_info.info["diagnostics"]
    assert diag["step_recall"] == 1.0
    assert diag["quality_pass"] is True


def test_evaluator_detects_missing_step():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    # Drop the s2 step and the s1->s2 edge by removing the relevant actions.
    keep = []
    drop_ids = {"wf_3", "wf_8"}  # add_step s2, connect s1->s2
    for a in task.evaluation_criteria.actions:
        if a.action_id not in drop_ids:
            keep.append((a.name, a.arguments))
    traj = [
        AssistantMessage(role="assistant", content="Hello."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(WorkflowDB())
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
    assert checks["assert_workflow_reconstructed"] is False


def test_japanese_scenario_loads_and_ground_truth_matches():
    ja_task = [t for t in get_tasks() if t.id == JA_SCENARIO][0]
    assert ja_task.initial_state is not None
    assert "お世話になっております" in ja_task.initial_state.message_history[0].content
    # A JA reconstruction (Japanese actions) is matched via content tokens.
    tools = _tools()
    tools.create_workflow("見積作成")
    for sid, act, actor, system, reads, writes, cond in [
        ("s1", "見積依頼を受け付ける", "営業", None, [], ["依頼"], None),
        ("s2", "CRMで顧客情報を確認する", "営業", "crm", ["顧客"], [], None),
        (
            "s3",
            "見積システムで見積を作成する",
            "営業",
            "quoting",
            ["顧客", "価格"],
            ["見積"],
            None,
        ),
        (
            "s4",
            "高額見積を承認する",
            "manager",
            "quoting",
            ["見積"],
            ["承認"],
            "100万円超",
        ),
        ("s5", "見積を顧客に送付する", "営業", "email", ["見積"], ["送付済み"], None),
        (
            "s6",
            "月末に経理へ見積集計を送る",
            "営業",
            "excel",
            ["見積"],
            ["excel_summary"],
            "月末",
        ),
    ]:
        tools.add_step(
            sid,
            act,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            condition=cond,
        )
    for a, b, c in [
        ("s1", "s2", None),
        ("s2", "s3", None),
        ("s3", "s4", "100万円超"),
        ("s3", "s5", "100万円以下"),
        ("s4", "s5", None),
        ("s5", "s6", "月末"),
    ]:
        tools.connect_steps(a, b, c)
    tools.add_branch("s3", "金額閾値", ["s4", "s5"])
    tools.set_step_rationale("s4", "与信リスク管理のため", "FACT", source="営業")
    tools.set_step_unknown("s6", "理由は不明")
    tools.challenge_step("s6", "このステップは本当に必要ですか？")
    tools.finish_interview()

    ev = _eval(tools, JA_SCENARIO)
    # Bilingual matching partially reconstructs the steps (actor/system carry
    # steps whose free-text shares no latin tokens with the EN ground truth).
    assert ev.step_recall >= 0.5
    # The rationale and challenge handling are language-independent and hold.
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.fabricated_rationale is False
