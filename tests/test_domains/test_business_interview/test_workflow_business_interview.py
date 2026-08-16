"""Tests for the hardened Workflow-first business_interview domain (v2).

Covers the workflow-reconstruction tools, the concept-based structural /
epistemic / challenge evaluation, the ASKED-vs-RECORDED-vs-CORRECT separation,
the UNKNOWN / NOT_RECORDED / NONE_FOUND distinction, KNOWN-value validation,
ground-truth observation correctness (owner / evidence / removal), structured
step-unit stakeholder truth and structural completeness, the same-concept
matcher None fix, arbitrary step-id canonicalisation, EN/JA semantic
equivalence, leakage, and the falsification suite A-V.
"""

import copy

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import (
    NecessityResult,
    RationaleResult,
    WorkflowDB,
)
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.ground_truth import (
    GTStep,
    get_ground_truth,
    get_stakeholder_truth,
    missing_from_instructions,
    stakeholder_truth_completeness,
)
from tau2.domains.business_interview.semantic import _match_steps, evaluate
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
)

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
SAME_CONCEPT_SCENARIO = "same_concept_workflow"
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


def _step(tools: InterviewTools, sid: str):
    return next(s for s in tools.db.workflow.steps if s.id == sid)


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("s1", "s2", "s3", "s4", "s5", "s6"),
    record_observations: bool = True,
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
    tools.set_step_unknown(remap["s6"], "理由は不明" if ja else "does not know")
    for dim in _CHALLENGE_DIMS:
        tools.challenge_step(remap["s6"], dim, f"investigate {dim}")
    if record_observations:
        tools.record_necessity_detail(
            remap["s6"],
            owner="経理（不確か）" if ja else "accounting (uncertain)",
            owner_state="UNKNOWN",
            evidence="文書なし" if ja else "no documentation",
            evidence_state="NONE_FOUND",
            removal_impact="不明" if ja else "unknown",
            removal_state="UNKNOWN",
        )
    tools.finish_interview()


# ---------------------------------------------------------------------------
# Good reconstruction (reference behaviour) = falsification L / K
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
    assert ev.why_asked is True and ev.why_recorded is True and ev.why_correct is True
    assert (
        ev.owner_asked is True
        and ev.owner_recorded is True
        and ev.owner_correct is True
    )
    assert ev.evidence_asked is True and ev.evidence_recorded is True
    assert ev.evidence_correct is True
    assert ev.removal_asked is True and ev.removal_recorded is True
    assert ev.removal_correct is True
    assert ev.deletion_considered is True
    assert ev.challenge_done is True
    assert ev.improvement_order_ok is True
    assert ev.structural_pass is True
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.protocol_pass is True
    assert ev.quality_pass is True


# ---------------------------------------------------------------------------
# Falsification A-F: observation correctness vs ground truth
# ---------------------------------------------------------------------------


def test_A_owner_unknown_expected_unknown_recorded_passes():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.owner_result == "UNKNOWN"
    assert ev.owner_recorded is True
    assert ev.owner_correct is True
    assert ev.challenge_pass is True


def test_B_owner_unknown_expected_known_fake_value_fails():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.owner_result = NecessityResult.KNOWN
    s6.necessity.owner = "CEO"
    ev = _eval(tools)
    assert ev.owner_recorded is True
    assert ev.owner_result == "KNOWN"
    assert ev.owner_correct is False  # fabricated KNOWN vs expected UNKNOWN
    assert ev.challenge_pass is False


def test_C_evidence_none_found_expected_none_found_passes():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.evidence_result == "NONE_FOUND"
    assert ev.evidence_recorded is True
    assert ev.evidence_correct is True
    assert ev.challenge_pass is True


def test_D_evidence_none_found_expected_known_fake_fails():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.evidence_result = NecessityResult.KNOWN
    s6.necessity.evidence = "law"
    ev = _eval(tools)
    assert ev.evidence_recorded is True
    assert ev.evidence_result == "KNOWN"
    assert ev.evidence_correct is False
    assert ev.challenge_pass is False


def test_E_removal_unknown_expected_unknown_passes():
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.removal_result == "UNKNOWN"
    assert ev.removal_recorded is True
    assert ev.removal_correct is True
    assert ev.challenge_pass is True


def test_F_removal_unknown_expected_known_fabricated_fails():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.removal_result = NecessityResult.KNOWN
    s6.necessity.removal_impact = "sales cannot be booked"
    ev = _eval(tools)
    assert ev.removal_recorded is True
    assert ev.removal_result == "KNOWN"
    assert ev.removal_correct is False
    assert ev.challenge_pass is False


# ---------------------------------------------------------------------------
# Falsification G-I: KNOWN requires a value
# ---------------------------------------------------------------------------


def test_G_known_owner_without_value_tool_rejects():
    tools = _tools()
    _build(tools)
    with pytest.raises(ValueError):
        tools.record_necessity_detail("s6", owner_state="KNOWN")  # no owner value


def test_G_known_owner_without_value_evaluator_fails_defensively():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.owner_result = NecessityResult.KNOWN
    s6.necessity.owner = None  # KNOWN without a value -> invalid
    ev = _eval(tools)
    assert ev.owner_recorded is False
    assert ev.owner_correct is False
    assert ev.challenge_pass is False


def test_H_known_evidence_without_value_fails():
    tools = _tools()
    _build(tools)
    with pytest.raises(ValueError):
        tools.record_necessity_detail("s6", evidence_state="KNOWN")
    s6 = _step(tools, "s6")
    s6.necessity.evidence_result = NecessityResult.KNOWN
    s6.necessity.evidence = None
    ev = _eval(tools)
    assert ev.evidence_recorded is False
    assert ev.evidence_correct is False
    assert ev.challenge_pass is False


def test_I_known_removal_without_value_fails():
    tools = _tools()
    _build(tools)
    with pytest.raises(ValueError):
        tools.record_necessity_detail("s6", removal_state="KNOWN")
    s6 = _step(tools, "s6")
    s6.necessity.removal_result = NecessityResult.KNOWN
    s6.necessity.removal_impact = None
    ev = _eval(tools)
    assert ev.removal_recorded is False
    assert ev.removal_correct is False
    assert ev.challenge_pass is False


# ---------------------------------------------------------------------------
# Falsification J-L: asked / recorded / correct three tiers
# ---------------------------------------------------------------------------


def test_J_asked_only_fails():
    """Asked a question but recorded no result -> FAIL."""
    tools = _tools()
    _build(tools, record_observations=False)  # only asks, records nothing
    s6 = _step(tools, "s6")
    assert s6.necessity.owner_asked is True
    ev = _eval(tools)
    assert ev.owner_recorded is False
    assert ev.evidence_recorded is False
    assert ev.removal_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_K_recorded_but_incorrect_fails():
    """Recorded a result but it contradicts the ground truth -> FAIL."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.owner_result = NecessityResult.KNOWN
    s6.necessity.owner = "CEO"  # recorded but wrong (expected UNKNOWN)
    ev = _eval(tools)
    assert ev.owner_recorded is True
    assert ev.owner_correct is False
    assert ev.challenge_pass is False


def test_L_recorded_and_correct_passes():
    """Recorded a result that matches the ground truth -> PASS."""
    tools = _tools()
    _build(tools)
    ev = _eval(tools)
    assert ev.owner_recorded is True and ev.owner_correct is True
    assert ev.evidence_recorded is True and ev.evidence_correct is True
    assert ev.removal_recorded is True and ev.removal_correct is True
    assert ev.challenge_pass is True


def test_diagnostics_distinguish_asked_recorded_correct():
    """Diagnostics surface the three tiers for each observation."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.owner_result = NecessityResult.KNOWN
    s6.necessity.owner = "CEO"
    ev = _eval(tools)
    diag = ev.model_dump(mode="json")
    assert diag["owner_asked"] is True
    assert diag["owner_recorded"] is True
    assert diag["owner_result"] == "KNOWN"
    assert diag["owner_correct"] is False
    assert diag["evidence_result"] == "NONE_FOUND"
    assert diag["evidence_correct"] is True


# ---------------------------------------------------------------------------
# Falsification M-O: structured step/axis stakeholder-truth completeness
# ---------------------------------------------------------------------------


def test_M_step_axis_stakeholder_truth_completeness_passes():
    """Every evaluator-required field is present in the step-unit truth."""
    assert stakeholder_truth_completeness(SCENARIO) == []
    assert stakeholder_truth_completeness(JA_SCENARIO) == []


def test_N_removing_approve_system_from_truth_fails_completeness():
    """If approve_quote.system is required but missing from its own truth -> FAIL."""
    truth = copy.deepcopy(get_stakeholder_truth(SCENARIO))
    truth.steps["approve_quote"].system = None
    missing = stakeholder_truth_completeness(SCENARIO, stakeholder_truth=truth)
    assert ("approve_quote", "system") in missing


def test_O_other_step_same_system_does_not_fix_completeness():
    """A 'quoting' string in another step must NOT satisfy approve_quote.system."""
    truth = copy.deepcopy(get_stakeholder_truth(SCENARIO))
    truth.steps["approve_quote"].system = None
    assert truth.steps["create_quote"].system == "quoting"  # same string elsewhere
    missing = stakeholder_truth_completeness(SCENARIO, stakeholder_truth=truth)
    assert ("approve_quote", "system") in missing


def test_P_scenario_id_truth_completeness_is_scenario_aware():
    """Completeness uses each scenario's own truth; EN and JA share the canonical."""
    en = get_stakeholder_truth(SCENARIO)
    ja = get_stakeholder_truth(JA_SCENARIO)
    assert en is not None and en is ja  # canonical resolution for EN/JA
    # A scenario without a defined stakeholder truth is not graded (no missing).
    assert stakeholder_truth_completeness("unknown_scenario") == []
    # The check keys on the provided truth, not a global copy.
    truth = copy.deepcopy(en)
    truth.steps["send_quote"].system = None
    assert ("send_quote", "system") in stakeholder_truth_completeness(
        SCENARIO, stakeholder_truth=truth
    )
    assert stakeholder_truth_completeness(SCENARIO) == []  # global truth unchanged


def test_stakeholder_truth_is_answerable_by_simulator():
    """The simulator prompt can answer every structured truth value (no gap)."""
    for task in get_tasks():
        inst = task.user_scenario.instructions
        text = " ".join(
            [
                inst.known_info or "",
                inst.unknown_info or "",
                inst.task_instructions or "",
            ]
        )
        assert missing_from_instructions(task.id, text) == [], task.id


# ---------------------------------------------------------------------------
# Falsification Q: same-concept matcher None fix
# ---------------------------------------------------------------------------


def test_Q_unresolved_none_data_is_not_matching_evidence():
    """Two completely different unknown data items (both resolve to None) must
    not count as a data match in same-concept disambiguation."""
    g1 = GTStep(
        id="g1",
        concept="approve_quote",
        action="approve a",
        actor="sales",
        system="email",
        reads=["quote"],  # resolved data
    )
    g2 = GTStep(
        id="g2",
        concept="approve_quote",
        action="approve b",
        actor="sales",
        system="email",
        reads=["mysterywidget"],  # unresolved -> None
    )
    rec = GTStep(
        id="r1",
        concept="approve_quote",
        action="approve the quotation",
        actor="sales",
        system="email",
        reads=["completelydifferentunknown"],  # unresolved -> None
    )
    matches = _match_steps([rec], [g1, g2])
    # rec must NOT be pulled to g2 by None&None data overlap; tie-break to g1.
    assert matches[0][1] is g1
    assert matches[0][1].id == "g1"


def test_same_concept_resolved_data_still_disambiguates():
    """Resolved data still distinguishes same-concept steps (priority intact)."""
    g1 = GTStep(
        id="g1",
        concept="approve_quote",
        action="approve a",
        actor="sales",
        system="email",
        reads=["quote"],
    )
    g2 = GTStep(
        id="g2",
        concept="approve_quote",
        action="approve b",
        actor="sales",
        system="email",
        reads=["customer"],
    )
    rec = GTStep(
        id="r1",
        concept="approve_quote",
        action="approve the quotation",
        actor="sales",
        system="email",
        reads=["quote"],
    )
    matches = _match_steps([rec], [g1, g2])
    assert matches[0][1] is g1  # matches the step whose read data matches


# ---------------------------------------------------------------------------
# Falsification R / S / T / U: arbitrary ids, EN/JA, quality vs protocol
# ---------------------------------------------------------------------------


def test_R_arbitrary_step_ids_full_pass():
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
    assert ev.challenge_target_identified is True
    assert ev.challenge_done is True
    assert ev.structural_pass is True
    assert ev.quality_pass is True


def test_S_en_full_structural_pass():
    tools = _tools()
    _build(tools, ja=False)
    ev = _eval(tools, SCENARIO)
    assert ev.step_recall == 1.0
    assert ev.structural_pass is True
    assert ev.quality_pass is True


def test_T_ja_equivalent_full_structural_pass():
    tools = _tools()
    _build(tools, ja=True)
    ev = _eval(tools, JA_SCENARIO)
    assert ev.step_recall == 1.0
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
        "actor_accuracy",
        "system_accuracy",
        "rationale_coverage",
        "owner_correct",
        "evidence_correct",
        "removal_correct",
    ):
        assert getattr(en, field) == getattr(ja, field), field


def test_U_quality_correct_protocol_fail():
    tools = _tools()
    _build(tools)
    tools.db.interview_complete = False  # never called finish_interview
    ev = _eval(tools)
    assert ev.quality_pass is True
    assert ev.protocol_pass is False
    assert ev.protocol_completed is False


# ---------------------------------------------------------------------------
# Structural reconstruction robustness (kept from v2)
# ---------------------------------------------------------------------------


def test_wrong_action_no_match():
    tools = _tools()
    _build(tools)
    _step(tools, "s2").action = "take a coffee break"  # sales/crm, wrong action
    ev = _eval(tools)
    assert ev.step_recall < 1.0
    assert ev.unexpected_step_count == 1
    assert ev.structural_pass is False


def test_missing_required_data_fails_recall():
    tools = _tools()
    _build(tools)
    _step(tools, "s3").reads = ["customer"]  # pricing missing
    ev = _eval(tools)
    assert ev.data_read_recall < 1.0
    assert ev.structural_pass is False


def test_invented_extra_data_fails_precision():
    tools = _tools()
    _build(tools)
    _step(tools, "s3").writes = ["quote", "quarterly_report", "tax_ledger"]
    ev = _eval(tools)
    assert ev.data_write_precision < 1.0
    assert ev.structural_pass is False


def test_reversed_condition_fails():
    tools = _tools()
    _build(tools)
    for t in tools.db.workflow.transitions:
        if (t.from_step, t.to_step) == ("s3", "s4"):
            t.condition = "amount at or below 1,000,000"
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0
    assert ev.structural_pass is False


def test_missing_condition_fails():
    tools = _tools()
    _build(tools)
    for t in tools.db.workflow.transitions:
        if (t.from_step, t.to_step) == ("s5", "s6"):
            t.condition = None
    ev = _eval(tools)
    assert ev.transition_accuracy < 1.0
    assert ev.structural_pass is False


def test_missing_branch_path_fails_branch_recall():
    tools = _tools()
    _build(tools)
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


def test_wrong_purpose_fails_metadata():
    tools = _tools()
    _build(tools)
    tools.db.workflow.purpose = "track inventory levels"
    ev = _eval(tools)
    assert ev.purpose_accuracy == 0.0
    assert ev.structural_pass is False


def test_confirmed_rationale_content_must_match():
    tools = _tools()
    _build(tools)
    _step(tools, "s4").necessity.rationale = "for tax reporting"
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_confirmed_rationale_wrong_source_fails():
    tools = _tools()
    _build(tools)
    _step(tools, "s4").necessity.source = "manager"
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_unknown_fabricated_as_fact_fails():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.rationale_result = RationaleResult.FACT
    s6.necessity.rationale = "for accounting reconciliation"
    ev = _eval(tools)
    assert ev.fabricated_rationale is True
    assert ev.rationale_pass is False


def test_unrecorded_rationale_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id in ("s4", "s6"):
            s.necessity.rationale_result = RationaleResult.NOT_RECORDED
    ev = _eval(tools)
    assert ev.rationale_coverage == 0.0
    assert ev.rationale_pass is False


# ---------------------------------------------------------------------------
# Improvement order
# ---------------------------------------------------------------------------


def test_automate_before_necessity_recorded_fails():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.rationale_result = RationaleResult.NOT_RECORDED
    s6.necessity.owner_result = NecessityResult.NOT_RECORDED
    s6.necessity.evidence_result = NecessityResult.NOT_RECORDED
    s6.necessity.removal_result = NecessityResult.NOT_RECORDED
    tools.propose_improvement("s6", "automate", "RPA it now")
    ev = _eval(tools)
    assert ev.improvement_order_ok is False
    assert ev.challenge_pass is False


def test_delete_after_necessity_recorded_passes():
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "question", "is this step needed?")
    tools.propose_improvement("s6", "delete", "reason unknown, no evidence")
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


# ---------------------------------------------------------------------------
# Falsification V: agent-visible leakage
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
    from tau2.agent.llm_agent import LLMAgent

    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="dummy")
    prompt = agent.system_prompt.lower()
    for term in HIDDEN_TERMS:
        assert term not in prompt, f"agent system prompt leaks hidden term: {term}"
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
    assert diag["quality_pass"] is True
    assert diag["owner_result"] == "UNKNOWN"
    assert diag["evidence_result"] == "NONE_FOUND"
    assert diag["owner_correct"] is True
    assert diag["evidence_correct"] is True


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


def test_evaluator_detects_fabricated_observation():
    """A reference-like run that records a fabricated owner (KNOWN vs expected
    UNKNOWN) fails the necessity challenge even though the result was recorded."""
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    # Replace the record_necessity_detail owner with a fabricated KNOWN.
    actions = copy.deepcopy(task.evaluation_criteria.actions)
    for a in actions:
        if a.name == "record_necessity_detail":
            a.arguments["owner_state"] = "KNOWN"
            a.arguments["owner"] = "CEO"
    traj = [
        AssistantMessage(role="assistant", content="Hello."),
        UserMessage(role="user", content="Sure."),
    ]
    tools = InterviewTools(WorkflowDB())
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
    assert checks["assert_necessity_challenged"] is False
    diag = reward_info.info["diagnostics"]
    assert diag["owner_result"] == "KNOWN"
    assert diag["owner_correct"] is False
    assert diag["challenge_done"] is False
