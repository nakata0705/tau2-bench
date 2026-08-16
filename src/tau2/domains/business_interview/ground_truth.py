"""Evaluator-only ground truth for the business_interview workflow scenarios.

This is the canonical workflow the agent is expected to reconstruct by
interviewing a stakeholder. It is NEVER exposed to the agent (policy / tools /
task description) or to the stakeholder (user scenario). The stakeholder knows
only what the scenario tells them; the agent must discover the workflow, its
actor/system/data, branches, rationale, and the questionable legacy step on its
own.

The design keeps the *workflow structure* (Workflow/Step/Branch), the
*language-independent concepts* (``concepts.py``) that let an equivalent EN/JA
reconstruction be scored identically, and the *epistemic layer* (a source's
FACT/BELIEF/UNKNOWN about a step's necessity) separate.

``FACT`` here means a source asserted it as certain — not objective truth; the
objective rationale state is given by ``GTNecessity.objective_status``.
"""

from dataclasses import dataclass, field
from typing import Optional

from tau2.domains.business_interview.data_model import EpistemicStatus

# Suffix marking a Japanese pre-localized scenario variant. EN and JA variants
# share the same canonical ground truth.
JA_SCENARIO_SUFFIX = "_ja"


@dataclass
class GTNecessity:
    """Ground-truth expectation about a step's necessity / rationale.

    - ``objective_status``: the objective state (FACT = confirmed rationale,
      UNKNOWN = the rationale genuinely is not known).
    - ``investigated_required``: the agent must have investigated this step's
      necessity before the interview is complete.
    - ``rationale_concept``: concept id the confirmed rationale must resolve to
      (content match). Only for FACT steps.
    - ``expected_source``: who the stakeholder (source) is for the rationale.
    - ``expected_status``: how the rationale must be recorded (FACT / BELIEF /
      UNKNOWN).
    - ``questionable``: this is the legacy step whose necessity must be
      challenged.
    """

    objective_status: EpistemicStatus = EpistemicStatus.UNKNOWN
    investigated_required: bool = True
    rationale_concept: Optional[str] = None
    expected_source: Optional[str] = None
    expected_status: Optional[EpistemicStatus] = None
    questionable: bool = False


@dataclass
class GTStep:
    id: str
    concept: str  # language-independent concept id (see concepts.py)
    action: str  # canonical display text (EN reference only)
    actor: str
    system: Optional[str] = None
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    condition: Optional[str] = None
    necessity: Optional[GTNecessity] = None


@dataclass
class GTTransition:
    from_step: str
    to_step: str
    condition: Optional[str] = None
    condition_concept: Optional[str] = None  # language-independent condition id


@dataclass
class GTBranch:
    from_step: str
    condition: str
    paths: list[str] = field(default_factory=list)


@dataclass
class WorkflowGroundTruth:
    scenario_id: str
    workflow_name: str
    trigger: str
    trigger_concept: str
    purpose: str
    purpose_concept: str
    outcome: str
    outcome_concept: str
    steps: list[GTStep]
    transitions: list[GTTransition]
    branches: list[GTBranch]
    questionable_step: Optional[str] = None


def _quotation_workflow() -> WorkflowGroundTruth:
    """The single workflow scenario: quotation creation with a confirmed-rationale
    approval branch and a questionable legacy month-end Excel step."""
    confirmed = GTNecessity(
        objective_status=EpistemicStatus.FACT,
        investigated_required=True,
        rationale_concept="credit_risk",
        expected_source="sales",
        expected_status=EpistemicStatus.FACT,
    )
    unknown = GTNecessity(
        objective_status=EpistemicStatus.UNKNOWN,
        investigated_required=True,
        questionable=True,
    )
    return WorkflowGroundTruth(
        scenario_id="quotation_workflow_1",
        workflow_name="Quotation creation",
        trigger="a customer requests a quotation",
        trigger_concept="trigger",
        purpose="produce an accurate quotation",
        purpose_concept="purpose",
        outcome="the customer receives a completed quotation",
        outcome_concept="outcome",
        steps=[
            GTStep(
                id="s1",
                concept="receive_request",
                action="receive quotation request",
                actor="sales",
                reads=[],
                writes=["request"],
            ),
            GTStep(
                id="s2",
                concept="check_customer",
                action="check customer information in the CRM",
                actor="sales",
                system="crm",
                reads=["customer"],
                writes=[],
            ),
            GTStep(
                id="s3",
                concept="create_quote",
                action="create quotation in the quoting system",
                actor="sales",
                system="quoting",
                reads=["customer", "pricing"],
                writes=["quote"],
            ),
            GTStep(
                id="s4",
                concept="approve_quote",
                action="approve high-value quotation",
                actor="manager",
                system="quoting",
                reads=["quote"],
                writes=["approval"],
                condition="amount over 1,000,000",
                necessity=confirmed,
            ),
            GTStep(
                id="s5",
                concept="send_quote",
                action="send quotation to customer",
                actor="sales",
                system="email",
                reads=["quote"],
                writes=["sent_quote"],
            ),
            GTStep(
                id="s6",
                concept="month_end_summary",
                action="send quotation summary to accounting at month-end",
                actor="sales",
                system="excel",
                reads=["quote"],
                writes=["excel_summary"],
                condition="month-end",
                necessity=unknown,
            ),
        ],
        transitions=[
            GTTransition("s1", "s2"),
            GTTransition("s2", "s3"),
            GTTransition(
                "s3",
                "s4",
                "amount over 1,000,000",
                condition_concept="cond_amount_over",
            ),
            GTTransition(
                "s3",
                "s5",
                "amount at or below 1,000,000",
                condition_concept="cond_amount_below",
            ),
            GTTransition("s4", "s5"),
            GTTransition("s5", "s6", "month-end", condition_concept="cond_month_end"),
        ],
        branches=[
            GTBranch(from_step="s3", condition="amount threshold", paths=["s4", "s5"]),
        ],
        questionable_step="s6",
    )


_SCENARIOS: dict[str, WorkflowGroundTruth] = {
    "quotation_workflow_1": _quotation_workflow(),
}


def canonical_scenario_id(scenario_id: Optional[str]) -> Optional[str]:
    """Map a Japanese variant to its canonical (English) scenario id."""
    if scenario_id is None:
        return None
    if scenario_id.endswith(JA_SCENARIO_SUFFIX):
        return scenario_id[: -len(JA_SCENARIO_SUFFIX)]
    return scenario_id


def get_ground_truth(scenario_id: Optional[str]) -> Optional[WorkflowGroundTruth]:
    """Return the ground truth for a scenario (EN or JA variant), or None."""
    canonical = canonical_scenario_id(scenario_id)
    if canonical is None:
        return None
    return _SCENARIOS.get(canonical)


# ---------------------------------------------------------------------------
# Stakeholder-visible truth requirements
#
# These are the facts the evaluator grades that the stakeholder KNOWS (so the
# user simulator must be able to answer them when asked). The scenario provides
# them as *known* information; difficulty comes from the stakeholder not
# volunteering them, never from the simulator not knowing them. A structural
# test cross-checks that the scenario covers every entry here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StakeholderKnowledge:
    """A fact the stakeholder knows and the evaluator grades.

    ``key`` names the axis (for diagnostics / tests). ``signals`` are
    substrings that must appear in the scenario's known_info so the simulator
    can truthfully answer a question about this fact.
    """

    key: str
    signals: tuple[str, ...]


def stakeholder_knowledge_requirements() -> list[StakeholderKnowledge]:
    """Facts the stakeholder knows that the evaluator requires, keyed by axis.

    Kept as a data contract so a structural test can assert the scenario's
    ``known_info`` covers every entry (stakeholder truth completeness). This
    guarantees the agent is never penalised because the simulator *could not*
    know an answer — only because it did not volunteer it.
    """
    return [
        # workflow metadata
        StakeholderKnowledge("trigger", ("request", "依頼")),
        StakeholderKnowledge("purpose", ("accurate", "正確")),
        StakeholderKnowledge(
            "outcome", ("receive", "受け取", "send the quotation", "送付")
        ),
        # actors
        StakeholderKnowledge("actor_sales", ("sales", "営業", "you", "自分")),
        StakeholderKnowledge(
            "actor_manager", ("manager", "approval", "上司", "承認者", "approver")
        ),
        # systems / tools
        StakeholderKnowledge(
            "system_crm", ("crm", "customer relationship management", "CRM")
        ),
        StakeholderKnowledge(
            "system_quoting", ("quoting", "quotation system", "見積", "見積システム")
        ),
        StakeholderKnowledge("system_email", ("email", "メール")),
        StakeholderKnowledge(
            "system_excel", ("excel", "spreadsheet", "エクセル", "Excel")
        ),
        # data read / written
        StakeholderKnowledge("data_request", ("request", "依頼")),
        StakeholderKnowledge("data_customer", ("customer", "顧客")),
        StakeholderKnowledge("data_pricing", ("pricing", "price", "価格", "料金")),
        StakeholderKnowledge("data_quote", ("quotation", "quote", "見積")),
        StakeholderKnowledge("data_sent", ("sent", "send", "送付", "送信")),
        StakeholderKnowledge("data_summary", ("summary", "集計")),
        # branch / transition condition
        StakeholderKnowledge(
            "cond_amount", ("1,000,000", "100万", "over", "exceed", "超")
        ),
        StakeholderKnowledge(
            "cond_month_end", ("month-end", "monthly", "month end", "月末")
        ),
        # approval owner / confirmed rationale
        StakeholderKnowledge(
            "rationale_credit_risk", ("credit risk", "credit", "与信", "与信リスク")
        ),
    ]
