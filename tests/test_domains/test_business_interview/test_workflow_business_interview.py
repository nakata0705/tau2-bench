"""Tests for the hardened Workflow-first business_interview domain (v2).

Covers the workflow-reconstruction tools, the concept-based structural /
epistemic / challenge evaluation, arbitrary step-id canonicalisation, NONE vs
UNKNOWN separation, confirmed-rationale content evaluation, EN/JA semantic
equivalence, precision-aware data, condition-aware transitions, workflow
metadata, necessity-challenge quality, leakage, stakeholder truth
completeness, and the full falsification suite (A-R).
"""

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import EpistemicStatus, WorkflowDB
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.ground_truth import (
    get_ground_truth,
    stakeholder_knowledge_requirements,
)
from tau2.domains.business_interview.semantic import evaluate
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
)

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO]

_EN_STEPS = [
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

_EN_EDGES = [
    ("s1", "s2", None),
    ("s2", "s3", None),
    ("s3", "s4", "amount over 1,000,000"),
    ("s3", "s5", "amount at or below 1,000,000"),
    ("s4", "s5", None),
    ("s5", "s6", "month-end"),
]

_JA_STEPS = [
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
    ("s4", "高額見積を承認する", "manager", "quoting", ["見積"], ["承認"], "100万円超"),
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
]

_JA_EDGES = [
    ("s1", "s2", None),
    ("s2", "s3", None),
    ("s3", "s4", "100万円超"),
    ("s3", "s5", "100万円以下"),
    ("s4", "s5", None),
    ("s5", "s6", "月末"),
]

_CHALLENGE_DIMS = ["why", "owner", "evidence", "removal", "deletion"]


def _tools() -> InterviewTools:
    return InterviewTools(WorkflowDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    return evaluate(tools.db, scenario)


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("s1", "s2", "s3", "s4", "s5", "s6"),
):
    """Build a correct workflow using the given step ids (may be arbitrary)."""
    steps = _JA_STEPS if ja else _EN_STEPS
    edges = _JA_EDGES if ja else _EN_EDGES
    i1, i2, i3, i4, i5, i6 = ids
    if ja:
        tools.create_workflow(
            "見積作成",
            trigger="顧客が見積を依頼する",
            purpose="正確な見積を作成する",
            outcome="顧客が見積を受け取る",
        )
    else:
        tools.create_workflow(
            "Quotation creation",
            trigger="customer requests a quotation",
            purpose="produce an accurate quotation",
            outcome="customer receives a quotation",
        )
    remap = {"s1": i1, "s2": i2, "s3": i3, "s4": i4, "s5": i5, "s6": i6}
    for sid, act, actor, system, reads, writes, cond in steps:
        tools.add_step(
            remap[sid],
            act,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            condition=cond,
        )
    for a, b, c in edges:
        tools.connect_steps(remap[a], remap[b], c)
    tools.add_branch(
        remap["s3"],
        "金額閾値" if ja else "amount threshold",
        [remap["s4"], remap["s5"]],
    )
    tools.set_step_rationale(
        remap["s4"],
        "与信リスク管理のため" if ja else "for credit risk management",
        "FACT",
        source="営業" if ja else "sales",
    )
    tools.set_step_unknown(remap["s6"], "理由は不明" if ja else "unknown")
    for dim in _CHALLENGE_DIMS:
        tools.challenge_step(remap["s6"], dim, f"investigate {dim}")
    tools.finish_interview()


# ---------------------------------------------------------------------------
# Good reconstruction (reference behaviour)
# ---------------------------------------------------------------------------


def test_good_reconstruction_passes_all_axes():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.protocol_completed is True
    assert ev.trigger_accuracy == 1.0
    assert ev.purpose_accuracy == 1.0
    assert ev.outcome_accuracy == 1.0
    assert ev.step_recall == 1.0
    assert ev.unexpected_step_count == 0
    assert ev.actor_accuracy == 1.0
    assert ev.system_accuracy == 1.0
    assert ev.data_read_recall == 1.0
    assert ev.data_read_precision == 1.0
    assert ev.data_write_recall == 1.0
    assert ev.data_write_precision == 1.0
    assert ev.transition_accuracy == 1.0
    assert ev.branch_recall == 1.0
    assert ev.branch_condition_accuracy == 1.0
    assert ev.rationale_coverage == 1.0
    assert ev.confirmed_rationale_ok is True
    assert ev.uncertainty_handling is True
    assert ev.fabricated_rationale is False
    assert ev.challenge_target_identified is True
    assert ev.why_investigated is True
    assert ev.owner_investigated is True
    assert ev.evidence_investigated is True
    assert ev.removal_investigated is True
    assert ev.deletion_considered is True
    assert ev.challenge_done is True
    assert ev.improvement_order_ok is True
    assert ev.structural_pass is True
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.protocol_pass is True
    assert ev.quality_pass is True


# ---------------------------------------------------------------------------
# P0: arbitrary step ids
# ---------------------------------------------------------------------------


def test_arbitrary_step_ids_full_pass():
    tools = _tools()
    _build(
        tools,
        ids=(
            "request",
            "check_customer",
            "create_quote",
            "approval",
            "send_quote",
            "month_end_export",
        ),
    )
    ev = _eval(tools)
    assert ev.step_recall == 1.0
    assert ev.unexpected_step_count == 0
    assert ev.transition_accuracy == 1.0
    assert ev.branch_recall == 1.0
    assert ev.challenge_target_identified is True
    assert ev.challenge_done is True
    assert ev.structural_pass is True
    assert ev.quality_pass is True


def test_arbitrary_ids_canonicalise_graph_not_raw_ids():
    """A full pass must not depend on the agent using s1..s6."""
    tools = _tools()
    _build(tools, ids=("a", "b", "c", "d", "e", "f"))
    ev = _eval(tools)
    assert ev.transition_accuracy == 1.0
    assert ev.branch_recall == 1.0
    assert ev.quality_pass is True


# ---------------------------------------------------------------------------
# P1: NONE / NOT_INVESTIGATED vs UNKNOWN
# ---------------------------------------------------------------------------


def test_uninvestigated_rationale_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        s.necessity.investigated = False
    ev = _eval(tools)
    assert ev.rationale_coverage == 0.0  # NOT_INVESTIGATED -> FAIL
    assert ev.rationale_pass is False


def test_investigated_unknown_passes():
    tools = _tools()
    _build(tools)  # s6 investigated and recorded UNKNOWN
    ev = _eval(tools)
    assert ev.rationale_coverage == 1.0
    assert ev.rationale_pass is True
    s6 = next(s for s in tools.db.workflow.steps if s.id == "s6")
    assert s6.necessity.investigated is True
    assert s6.necessity.epistemic_status == EpistemicStatus.UNKNOWN


def test_unknown_fabricated_as_fact_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.rationale_known = True
            s.necessity.epistemic_status = EpistemicStatus.FACT
            s.necessity.rationale = "for accounting reconciliation"
    ev = _eval(tools)
    assert ev.fabricated_rationale is True
    assert ev.uncertainty_handling is False
    assert ev.rationale_pass is False


# ---------------------------------------------------------------------------
# P1: confirmed rationale content / status / source
# ---------------------------------------------------------------------------


def test_confirmed_rationale_content_must_match():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s4":
            s.necessity.rationale = "for tax reporting"  # wrong content
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_confirmed_rationale_wrong_source_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s4":
            s.necessity.source = "manager"  # wrong source
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_correct_content_as_belief_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s4":
            s.necessity.epistemic_status = (
                EpistemicStatus.BELIEF
            )  # correct content, wrong status
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


# ---------------------------------------------------------------------------
# P2: step matching robustness (action-first)
# ---------------------------------------------------------------------------


def test_same_actor_system_wrong_action_no_match():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s2":  # sales/crm, but totally different action
            s.action = "take a coffee break"
    ev = _eval(tools)
    assert ev.step_recall < 1.0  # missing + unexpected
    assert ev.unexpected_step_count == 1
    assert ev.structural_pass is False


# ---------------------------------------------------------------------------
# P2: precision-aware data
# ---------------------------------------------------------------------------


def test_exact_data_passes():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.data_read_recall == 1.0 and ev.data_read_precision == 1.0
    assert ev.data_write_recall == 1.0 and ev.data_write_precision == 1.0
    assert ev.structural_pass is True


def test_missing_required_data_fails_recall():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s3":
            s.reads = ["customer"]  # pricing missing
    ev = _eval(tools)
    assert ev.data_read_recall < 1.0
    assert ev.structural_pass is False


def test_invented_extra_data_fails_precision():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s3":
            s.writes = ["quote", "quarterly_report", "tax_ledger"]  # invented extras
    ev = _eval(tools)
    assert ev.data_write_precision < 1.0
    assert ev.structural_pass is False


# ---------------------------------------------------------------------------
# P2: condition-aware transitions
# ---------------------------------------------------------------------------


def test_wrong_edge_fails_transition():
    tools = _tools()
    _build(tools)
    # Rewire s1->s2 into a wrong edge.
    tools.db.workflow.transitions = [
        t
        for t in tools.db.workflow.transitions
        if (t.from_step, t.to_step) != ("s1", "s2")
    ]
    tools.connect_steps("s1", "s3", None)  # skip the check step
    # (keep all other edges)
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0
    assert ev.structural_pass is False


def test_reversed_condition_fails():
    tools = _tools()
    _build(tools)
    for t in tools.db.workflow.transitions:
        if (t.from_step, t.to_step) == ("s3", "s4"):
            t.condition = "amount at or below 1,000,000"  # reversed
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0
    assert ev.structural_pass is False


def test_missing_condition_fails():
    tools = _tools()
    _build(tools)
    for t in tools.db.workflow.transitions:
        if (t.from_step, t.to_step) == ("s5", "s6"):
            t.condition = None  # month-end condition dropped
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0
    assert ev.structural_pass is False


def test_missing_branch_path_fails_branch_recall():
    tools = _tools()
    _build(tools)
    # Record the flow as linear: drop the low-value path from both the branch
    # record and the transitions (no divergence reconstructed).
    tools.db.workflow.transitions = [
        t
        for t in tools.db.workflow.transitions
        if (t.from_step, t.to_step) != ("s3", "s5")
    ]
    tools.db.workflow.branches = [
        b for b in tools.db.workflow.branches if not (b.from_step == "s3")
    ]
    ev = _eval(tools)
    assert ev.branch_recall < 1.0
    assert ev.structural_pass is False


# ---------------------------------------------------------------------------
# P2: workflow metadata
# ---------------------------------------------------------------------------


def test_workflow_metadata_accuracy():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.trigger_accuracy == 1.0
    assert ev.purpose_accuracy == 1.0
    assert ev.outcome_accuracy == 1.0
    assert ev.structural_pass is True


def test_wrong_purpose_fails_metadata():
    tools = _tools()
    _build(tools)
    tools.db.workflow.purpose = "track inventory levels"  # wrong purpose
    ev = _eval(tools)
    assert ev.purpose_accuracy == 0.0
    assert ev.structural_pass is False


# ---------------------------------------------------------------------------
# P1: EN / JA semantic equivalence
# ---------------------------------------------------------------------------


def test_en_full_structural_pass():
    tools = _tools()
    _build(tools, ja=False)
    ev = _eval(tools, SCENARIO)
    assert ev.step_recall == 1.0
    assert ev.transition_accuracy == 1.0
    assert ev.structural_pass is True
    assert ev.quality_pass is True


def test_ja_equivalent_full_structural_pass():
    tools = _tools()
    _build(tools, ja=True)
    ev = _eval(tools, JA_SCENARIO)
    assert ev.step_recall == 1.0
    assert ev.transition_accuracy == 1.0
    assert ev.branch_recall == 1.0
    assert ev.structural_pass is True
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.quality_pass is True


def test_en_ja_structural_metrics_equivalent():
    en_tools = _tools()
    ja_tools = _tools()
    _build(en_tools, ja=False)
    _build(ja_tools, ja=True)
    en = _eval(en_tools, SCENARIO)
    ja = _eval(ja_tools, JA_SCENARIO)
    for field in (
        "step_recall",
        "transition_accuracy",
        "branch_recall",
        "branch_condition_accuracy",
        "actor_accuracy",
        "system_accuracy",
        "data_read_recall",
        "data_read_precision",
        "data_write_recall",
        "data_write_precision",
        "rationale_coverage",
    ):
        assert getattr(en, field) == getattr(ja, field), field
    assert en.structural_pass == ja.structural_pass is True
    assert en.quality_pass == ja.quality_pass is True


# ---------------------------------------------------------------------------
# P2: necessity challenge quality
# ---------------------------------------------------------------------------


def test_owner_not_investigated_fails_challenge():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.owner_investigated = False
    ev = _eval(tools)
    assert ev.owner_investigated is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_evidence_not_investigated_fails_challenge():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.evidence_investigated = False
    ev = _eval(tools)
    assert ev.evidence_investigated is False
    assert ev.challenge_done is False


def test_removal_impact_not_investigated_fails_challenge():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.removal_investigated = False
    ev = _eval(tools)
    assert ev.removal_investigated is False
    assert ev.challenge_done is False


def test_challenge_target_found_via_id_mapping():
    """An arbitrary-id challenge still targets the questionable step."""
    tools = _tools()
    _build(tools, ids=("r", "cc", "cq", "ap", "sq", "me"))
    ev = _eval(tools)
    assert ev.challenge_target_identified is True
    assert ev.challenge_done is True


def test_no_challenge_at_all_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.investigated = False
            s.necessity.owner_investigated = False
            s.necessity.evidence_investigated = False
            s.necessity.removal_investigated = False
            s.necessity.deletion_considered = False
    ev = _eval(tools)
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


# ---------------------------------------------------------------------------
# P2: improvement order
# ---------------------------------------------------------------------------


def test_automate_before_investigation_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.investigated = False
            s.necessity.owner_investigated = False
            s.necessity.evidence_investigated = False
            s.necessity.challenged = False
    tools.propose_improvement("s6", "automate", "RPA it now")
    ev = _eval(tools)
    assert ev.improvement_order_ok is False
    assert ev.challenge_pass is False


def test_question_before_automate_ok():
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "question", "is this needed?")
    tools.propose_improvement("s6", "automate", "then automate")
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


def test_delete_after_investigation_ok():
    """Investigated unknown/no-evidence step treated as a deletion candidate is good."""
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "question", "is this step needed?")
    tools.propose_improvement("s6", "delete", "reason unknown, no evidence")
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


# ---------------------------------------------------------------------------
# Protocol vs quality
# ---------------------------------------------------------------------------


def test_quality_pass_protocol_fail():
    tools = _tools()
    _build(tools)
    tools.db.interview_complete = False
    ev = _eval(tools)
    assert ev.quality_pass is True
    assert ev.protocol_pass is False
    assert ev.protocol_completed is False


# ---------------------------------------------------------------------------
# Stakeholder truth completeness (structure test)
# ---------------------------------------------------------------------------


def test_scenario_covers_stakeholder_knowledge_requirements():
    """Every fact the evaluator grades that the stakeholder knows must be
    answerable from the scenario's known_info (truth completeness)."""
    for task in get_tasks():
        known = task.user_scenario.instructions.known_info or ""
        known_l = known.lower()
        for req in stakeholder_knowledge_requirements():
            matched = any(sig.lower() in known_l for sig in req.signals)
            assert matched, (
                f"{task.id}: scenario missing knowledge '{req.key}' ({req.signals})"
            )
            del req  # noqa: PLW0127


# ---------------------------------------------------------------------------
# Leakage (R)
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


def test_policy_has_no_hidden_workflow_identity():
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    for term in HIDDEN_TERMS:
        assert term not in policy, f"policy leaks hidden term: {term}"
    assert "workflow" in policy
    assert "branch" in policy
    assert "necessity" in policy


def test_tool_descriptions_have_no_hidden_workflow_identity():
    docs = "\n".join(
        [
            InterviewTools.create_workflow.__doc__ or "",
            InterviewTools.add_step.__doc__ or "",
            InterviewTools.connect_steps.__doc__ or "",
            InterviewTools.add_branch.__doc__ or "",
            InterviewTools.set_step_rationale.__doc__ or "",
            InterviewTools.set_step_unknown.__doc__ or "",
            InterviewTools.challenge_step.__doc__ or "",
            InterviewTools.record_necessity_detail.__doc__ or "",
            InterviewTools.propose_improvement.__doc__ or "",
            InterviewTools.finish_interview.__doc__ or "",
        ]
    ).lower()
    for term in HIDDEN_TERMS:
        assert term not in docs, f"tool description leaks hidden term: {term}"


def test_agent_system_prompt_has_no_ground_truth():
    """The agent's system prompt is built only from the domain policy, so the
    evaluator-only ground truth (in description.notes and the reference
    evaluation_criteria.actions) never reaches the agent."""
    from tau2.agent.llm_agent import LLMAgent

    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="dummy")
    prompt = agent.system_prompt.lower()
    for term in HIDDEN_TERMS:
        assert term not in prompt, f"agent system prompt leaks hidden term: {term}"
    # The task description / evaluation criteria are evaluator-only and must not
    # be referenced by the agent prompt.
    assert "description" not in prompt and "evaluation_criteria" not in prompt


def test_ground_truth_is_evaluator_only_not_in_scenario():
    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    scenario = str(task.user_scenario).lower()
    assert "s1" not in scenario and "s4" not in scenario
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
    assert [s.concept for s in en.steps] == [s.concept for s in ja.steps]


def test_diagnostics_explain_challenge_failure():
    """Diagnostics surface which challenge dimension was not investigated."""
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id == "s6":
            s.necessity.evidence_investigated = False
    ev = _eval(tools)
    diag = ev.model_dump(mode="json")
    assert diag["evidence_investigated"] is False
    assert diag["challenge_done"] is False
    assert diag["challenge_pass"] is False


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
    diag = reward_info.info["diagnostics"]
    assert diag["step_recall"] == 1.0
    assert diag["quality_pass"] is True


def test_evaluator_detects_missing_step():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
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
