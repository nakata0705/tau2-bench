"""Tests for the hardened Workflow-first business_interview domain (v2).

Covers the workflow-reconstruction tools, the concept-based structural /
epistemic / challenge evaluation, the ASKED-vs-RESULT-RECORDED separation, the
UNKNOWN / NOT_RECORDED / NONE_FOUND distinction, step-unit stakeholder truth
completeness, same-concept multi-step matching, arbitrary step-id
canonicalisation, confirmed-rationale content/source/status evaluation, EN/JA
semantic equivalence, precision-aware data, condition-aware transitions,
workflow metadata, leakage, and the full falsification suite A-V.
"""

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
    get_ground_truth,
    missing_stakeholder_truth,
)
from tau2.domains.business_interview.semantic import evaluate
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
# Good reconstruction (reference behaviour) = falsification K
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
    assert ev.why_asked is True and ev.why_recorded is True
    assert ev.owner_asked is True and ev.owner_recorded is True
    assert ev.evidence_asked is True and ev.evidence_recorded is True
    assert ev.removal_asked is True and ev.removal_recorded is True
    assert ev.deletion_considered is True
    assert ev.challenge_done is True
    assert ev.improvement_order_ok is True
    assert ev.structural_pass is True
    assert ev.rationale_pass is True
    assert ev.challenge_pass is True
    assert ev.protocol_pass is True
    assert ev.quality_pass is True


# ---------------------------------------------------------------------------
# Falsification A-D: the "why" dimension, ASKED vs RESULT RECORDED
# ---------------------------------------------------------------------------


def test_A_why_not_asked_fails():
    """A: the 'why is this needed' question was never asked / answered."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.why_asked = False
    s6.necessity.rationale_result = RationaleResult.NOT_RECORDED
    ev = _eval(tools)
    assert ev.why_asked is False
    assert ev.why_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_B_why_asked_but_result_not_recorded_fails():
    """B: the question was asked, but no answer was recorded -> NOT recorded.

    This is the core hardening: challenge_step('why') alone must NOT pass. An
    asked-but-unanswered 'why' is treated as NOT_RECORDED, never as UNKNOWN.
    """
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.why_asked = True  # the question was asked
    s6.necessity.rationale_result = RationaleResult.NOT_RECORDED  # no answer recorded
    ev = _eval(tools)
    assert ev.why_asked is True
    assert ev.why_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_C_explicit_unknown_recorded_passes():
    """C: an explicit UNKNOWN (stakeholder confirmed they don't know) passes."""
    tools = _tools()
    _build(tools)  # s6 why is recorded as UNKNOWN via set_step_unknown
    s6 = _step(tools, "s6")
    assert s6.necessity.rationale_result == RationaleResult.UNKNOWN
    ev = _eval(tools)
    assert ev.why_recorded is True
    assert ev.challenge_done is True


def test_D_unknown_fabricated_as_fact_fails():
    """D: promoting an UNKNOWN to a FACT is an epistemic fail."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.rationale_result = RationaleResult.FACT
    s6.necessity.rationale = "for accounting reconciliation"
    ev = _eval(tools)
    assert ev.fabricated_rationale is True
    assert ev.uncertainty_handling is False
    assert ev.rationale_pass is False


# ---------------------------------------------------------------------------
# Falsification E-H: the "owner" and "evidence" dimensions
# ---------------------------------------------------------------------------


def test_E_owner_asked_but_not_recorded_fails():
    """E: asking about the owner without recording a result -> FAIL."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.owner_asked = True  # the question was asked
    s6.necessity.owner_result = NecessityResult.NOT_RECORDED  # no answer recorded
    ev = _eval(tools)
    assert ev.owner_asked is True
    assert ev.owner_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_F_owner_unknown_or_none_found_recorded_passes():
    """F: recording an owner UNKNOWN or NONE_FOUND result is a valid finding."""
    for state in ("UNKNOWN", "NONE_FOUND", "KNOWN"):
        tools = _tools()
        _build(tools)
        s6 = _step(tools, "s6")
        s6.necessity.owner_asked = True
        s6.necessity.owner_result = NecessityResult(state)
        if state == "KNOWN":
            s6.necessity.owner = "sales"
        ev = _eval(tools)
        assert ev.owner_recorded is True, state
        assert ev.owner_result == state, state
        assert ev.challenge_done is True, state


def test_G_evidence_asked_but_not_recorded_fails():
    """G: asking about evidence without recording a result -> FAIL."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.evidence_asked = True
    s6.necessity.evidence_result = NecessityResult.NOT_RECORDED
    ev = _eval(tools)
    assert ev.evidence_asked is True
    assert ev.evidence_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_H_evidence_unknown_or_none_found_recorded_passes():
    """H: recording an evidence UNKNOWN / NONE_FOUND result is a valid finding."""
    for state in ("UNKNOWN", "NONE_FOUND", "KNOWN"):
        tools = _tools()
        _build(tools)
        s6 = _step(tools, "s6")
        s6.necessity.evidence_asked = True
        s6.necessity.evidence_result = NecessityResult(state)
        if state == "KNOWN":
            s6.necessity.evidence = "a written process document"
        ev = _eval(tools)
        assert ev.evidence_recorded is True, state
        assert ev.evidence_result == state, state
        assert ev.challenge_done is True, state


# ---------------------------------------------------------------------------
# Falsification I-J: the "removal" dimension
# ---------------------------------------------------------------------------


def test_I_removal_asked_but_not_recorded_fails():
    """I: asking about removal impact without recording a result -> FAIL."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.removal_asked = True
    s6.necessity.removal_result = NecessityResult.NOT_RECORDED
    ev = _eval(tools)
    assert ev.removal_asked is True
    assert ev.removal_recorded is False
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_J_removal_result_recorded_passes():
    """J: recording a removal result (KNOWN / UNKNOWN) is a valid finding."""
    for state in ("UNKNOWN", "KNOWN"):
        tools = _tools()
        _build(tools)
        s6 = _step(tools, "s6")
        s6.necessity.removal_asked = True
        s6.necessity.removal_result = NecessityResult(state)
        if state == "KNOWN":
            s6.necessity.removal_impact = "accounting would miss its summary"
        ev = _eval(tools)
        assert ev.removal_recorded is True, state
        assert ev.challenge_done is True, state


def test_asked_but_unrecorded_is_never_unknown_or_none_found():
    """An asked-but-unrecorded dimension is NOT_RECORDED, never UNKNOWN/NONE."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    # The agent only asks; records nothing for owner.
    s6.necessity.owner_asked = True
    s6.necessity.owner_result = NecessityResult.NOT_RECORDED
    s6.necessity.owner = None
    ev = _eval(tools)
    assert ev.owner_result == "NOT_RECORDED"
    assert ev.owner_recorded is False


# ---------------------------------------------------------------------------
# Falsification L-M: improvement order
# ---------------------------------------------------------------------------


def test_L_automate_before_necessity_recorded_fails():
    """L: automating an uninvestigated step (nothing recorded) -> FAIL."""
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


def test_automate_after_question_recorded_ok():
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "question", "is this needed?")
    tools.propose_improvement("s6", "automate", "then automate")
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


def test_M_delete_after_necessity_recorded_passes():
    """M: after the necessity is recorded, considering deletion is good."""
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "question", "is this step needed?")
    tools.propose_improvement("s6", "delete", "reason unknown, no evidence")
    ev = _eval(tools)
    assert ev.improvement_order_ok is True


def test_wrong_kind_order_fails():
    tools = _tools()
    _build(tools)
    tools.propose_improvement("s6", "automate", "automate first")
    tools.propose_improvement("s6", "question", "question later")
    ev = _eval(tools)
    assert ev.improvement_order_ok is False


# ---------------------------------------------------------------------------
# Falsification P: arbitrary step ids
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
# Falsification Q-R: same-concept multi-step matching
# ---------------------------------------------------------------------------


def _build_same_concept(tools, swapped=False):
    tools.create_workflow(
        "Approval flow",
        trigger="customer requests a quote approval",
        purpose="produce an accurate approved quote",
        outcome="the customer receives the final approved quote",
    )
    if not swapped:
        # r1 should map to the manager approval step (g1), r2 to the sales step (g2).
        tools.add_step(
            "r1",
            "approve the high-value quotation",
            actor="manager",
            system="quoting",
            reads=["quote"],
            writes=["approval"],
        )
        tools.add_step(
            "r2",
            "approve the revised quotation",
            actor="sales",
            system="quoting",
            reads=["quote"],
            writes=["revision"],
        )
    else:
        # Genuine confusion: each recorded step blends an attribute of the other.
        tools.add_step(
            "r1",
            "approve the high-value quotation",
            actor="sales",
            system="quoting",
            reads=["quote"],
            writes=["approval"],
        )
        tools.add_step(
            "r2",
            "approve the revised quotation",
            actor="manager",
            system="quoting",
            reads=["quote"],
            writes=["revision"],
        )
    tools.connect_steps("r1", "r2")
    tools.finish_interview()


def test_Q_same_concept_two_steps_both_match():
    """Q: two steps sharing a concept are both matched correctly."""
    tools = _tools()
    _build_same_concept(tools, swapped=False)
    ev = _eval(tools, SAME_CONCEPT_SCENARIO)
    assert ev.step_recall == 1.0
    assert ev.unexpected_step_count == 0
    assert ev.actor_accuracy == 1.0
    assert ev.data_write_recall == 1.0
    assert ev.data_write_precision == 1.0
    assert ev.structural_pass is True


def test_R_same_concept_two_steps_swapped_fails_structural():
    """R: swapping the same-concept steps' distinguishing attributes is a
    structural failure (the matcher does not 'fix' the confusion)."""
    tools = _tools()
    _build_same_concept(tools, swapped=True)
    ev = _eval(tools, SAME_CONCEPT_SCENARIO)
    assert ev.step_recall == 1.0  # both still found by concept
    assert ev.data_write_recall < 1.0  # but the written data is misaligned
    assert ev.structural_pass is False


def test_same_concept_not_duplicated_into_one_step():
    """A single reconstructed same-concept step only covers one GT step."""
    tools = _tools()
    tools.create_workflow(
        "Approval flow",
        trigger="customer requests a quote approval",
        purpose="produce an accurate approved quote",
        outcome="the customer receives the final approved quote",
    )
    tools.add_step(
        "only",
        "approve the quotation",
        actor="manager",
        system="quoting",
        reads=["quote"],
        writes=["approval"],
    )
    tools.finish_interview()
    ev = _eval(tools, SAME_CONCEPT_SCENARIO)
    assert ev.step_recall == 0.5  # only one of the two same-concept steps found


# ---------------------------------------------------------------------------
# P1: NONE / NOT_RECORDED vs UNKNOWN (rationale)
# ---------------------------------------------------------------------------


def test_unrecorded_rationale_fails():
    tools = _tools()
    _build(tools)
    for s in tools.db.workflow.steps:
        if s.id in ("s4", "s6"):
            s.necessity.rationale_result = RationaleResult.NOT_RECORDED
    ev = _eval(tools)
    assert ev.rationale_coverage == 0.0  # NOT_RECORDED -> FAIL
    assert ev.rationale_pass is False


def test_unknown_recorded_passes_rationale():
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    assert s6.necessity.rationale_result == RationaleResult.UNKNOWN
    ev = _eval(tools)
    assert ev.rationale_coverage == 1.0
    assert ev.rationale_pass is True


# ---------------------------------------------------------------------------
# P1: confirmed rationale content / status / source
# ---------------------------------------------------------------------------


def test_confirmed_rationale_content_must_match():
    tools = _tools()
    _build(tools)
    _step(tools, "s4").necessity.rationale = "for tax reporting"  # wrong content
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_confirmed_rationale_wrong_source_fails():
    tools = _tools()
    _build(tools)
    _step(tools, "s4").necessity.source = "manager"  # wrong source
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


def test_correct_content_as_belief_fails():
    tools = _tools()
    _build(tools)
    _step(tools, "s4").necessity.rationale_result = RationaleResult.BELIEF
    ev = _eval(tools)
    assert ev.confirmed_rationale_ok is False
    assert ev.rationale_pass is False


# ---------------------------------------------------------------------------
# P2: step matching robustness (action-first)
# ---------------------------------------------------------------------------


def test_same_actor_system_wrong_action_no_match():
    tools = _tools()
    _build(tools)
    _step(tools, "s2").action = "take a coffee break"  # sales/crm, wrong action
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


# ---------------------------------------------------------------------------
# P2: condition-aware transitions
# ---------------------------------------------------------------------------


def test_wrong_edge_fails_transition():
    tools = _tools()
    _build(tools)
    tools.db.workflow.transitions = [
        t
        for t in tools.db.workflow.transitions
        if (t.from_step, t.to_step) != ("s1", "s2")
    ]
    tools.connect_steps("s1", "s3", None)  # skip the check step
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
    tools.db.workflow.purpose = "track inventory levels"
    ev = _eval(tools)
    assert ev.purpose_accuracy == 0.0
    assert ev.structural_pass is False


# ---------------------------------------------------------------------------
# Falsification S-T: EN / JA semantic equivalence
# ---------------------------------------------------------------------------


def test_S_en_full_structural_pass():
    tools = _tools()
    _build(tools, ja=False)
    ev = _eval(tools, SCENARIO)
    assert ev.step_recall == 1.0
    assert ev.transition_accuracy == 1.0
    assert ev.structural_pass is True
    assert ev.quality_pass is True


def test_T_ja_equivalent_full_structural_pass():
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
# Falsification U: quality vs protocol
# ---------------------------------------------------------------------------


def test_U_quality_correct_protocol_fail():
    tools = _tools()
    _build(tools)
    tools.db.interview_complete = False  # never called finish_interview
    ev = _eval(tools)
    assert ev.quality_pass is True
    assert ev.protocol_pass is False
    assert ev.protocol_completed is False


# ---------------------------------------------------------------------------
# Challenge targeting via id mapping
# ---------------------------------------------------------------------------


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
    s6 = _step(tools, "s6")
    s6.necessity.rationale_result = RationaleResult.NOT_RECORDED
    s6.necessity.owner_result = NecessityResult.NOT_RECORDED
    s6.necessity.evidence_result = NecessityResult.NOT_RECORDED
    s6.necessity.removal_result = NecessityResult.NOT_RECORDED
    s6.necessity.deletion_considered = False
    s6.necessity.challenged = False
    ev = _eval(tools)
    assert ev.challenge_done is False
    assert ev.challenge_pass is False


def test_deletion_assessment_separate_from_observations():
    """Deletion is an analyst assessment; observations alone don't set it."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.deletion_considered = False  # never assessed
    ev = _eval(tools)
    assert ev.deletion_considered is False
    assert ev.challenge_done is False  # deletion assessment required


# ---------------------------------------------------------------------------
# Falsification N-O: step-unit stakeholder truth completeness
# ---------------------------------------------------------------------------


def test_N_step_unit_stakeholder_truth_completeness():
    """Every evaluator-graded stakeholder fact is answerable from the scenario."""
    for task in get_tasks():
        known = task.user_scenario.instructions.known_info or ""
        missing = missing_stakeholder_truth(known)
        assert missing == [], (
            f"{task.id}: scenario missing step-unit knowledge: "
            f"{[(m.step_concept, m.key) for m in missing]}"
        )


def test_O_missing_step_truth_fails_completeness():
    """If a step's required truth is missing, completeness FAILS."""
    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    known = task.user_scenario.instructions.known_info or ""
    # Simulate a scenario that never tells the stakeholder about the manager.
    stripped = known.replace("manager", "").replace("approval", "").replace("上司", "")
    missing = missing_stakeholder_truth(stripped)
    keys = {(m.step_concept, m.key) for m in missing}
    assert ("approve_quote", "actor") in keys
    # And the whole suite is not empty: completeness has failed.
    assert len(missing) > 0


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


def test_diagnostics_explain_challenge_failure():
    """Diagnostics surface ASKED vs RESULT RECORDED for a failed challenge."""
    tools = _tools()
    _build(tools)
    s6 = _step(tools, "s6")
    s6.necessity.evidence_asked = True
    s6.necessity.evidence_result = NecessityResult.NOT_RECORDED
    ev = _eval(tools)
    diag = ev.model_dump(mode="json")
    assert diag["evidence_asked"] is True
    assert diag["evidence_recorded"] is False
    assert diag["evidence_result"] == "NOT_RECORDED"
    assert diag["challenge_done"] is False
    assert diag["challenge_pass"] is False


def test_diagnostics_report_result_states():
    tools = _tools()
    _build(tools)
    diag = _eval(tools).model_dump(mode="json")
    assert diag["owner_result"] == "UNKNOWN"
    assert diag["evidence_result"] == "NONE_FOUND"
    assert diag["removal_result"] == "UNKNOWN"


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
    assert diag["owner_result"] == "UNKNOWN"
    assert diag["evidence_result"] == "NONE_FOUND"


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


def test_evaluator_detects_asked_but_unrecorded_challenge():
    """A reference-like run that only asks (no observation recorded) fails the
    necessity challenge even though the question was asked."""
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    drop_ids = {"wf_21"}  # drop record_necessity_detail
    keep = []
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
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks["assert_necessity_challenged"] is False
    diag = reward_info.info["diagnostics"]
    assert diag["owner_asked"] is True
    assert diag["owner_recorded"] is False
    assert diag["challenge_done"] is False
