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

Stakeholder truth is **step-unit**: each requirement is tied to a specific
ground-truth step (by concept) and axis, so completeness is judged per step,
not by a keyword search over the whole scenario. A structural test asserts the
scenario's ``known_info`` can answer every requirement the evaluator grades.
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
    - ``investigated_required``: the agent must have recorded a result for this
      step's necessity before the interview is complete.
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


def _same_concept_workflow() -> WorkflowGroundTruth:
    """A small scenario exercising the same-concept multi-step matcher.

    Two steps share the ``approve_quote`` concept but are distinguished
    deterministically by actor (manager vs sales) and written data (approval vs
    revision). Used by the falsification Q/R tests only — it is not a task and
    is never in the agent-visible task set.
    """
    return WorkflowGroundTruth(
        scenario_id="same_concept_workflow",
        workflow_name="Quote approval flow",
        trigger="a quote requires approval",
        trigger_concept="trigger",
        purpose="approve quotes before they are used",
        purpose_concept="purpose",
        outcome="a final approved quote is recorded",
        outcome_concept="outcome",
        steps=[
            GTStep(
                id="g1",
                concept="approve_quote",
                action="approve the high-value quotation",
                actor="manager",
                system="quoting",
                reads=["quote"],
                writes=["approval"],
            ),
            GTStep(
                id="g2",
                concept="approve_quote",
                action="approve the revised quotation",
                actor="sales",
                system="quoting",
                reads=["quote"],
                writes=["revision"],
            ),
        ],
        transitions=[
            GTTransition("g1", "g2"),
        ],
        branches=[],
        questionable_step=None,
    )


_SCENARIOS: dict[str, WorkflowGroundTruth] = {
    "quotation_workflow_1": _quotation_workflow(),
    "same_concept_workflow": _same_concept_workflow(),
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
# Step-unit stakeholder-visible truth requirements
#
# These are the facts the evaluator grades that the stakeholder KNOWS about a
# specific step (so the user simulator must be able to answer them when asked).
# They are expressed **per step** (by concept) and per axis — not as a keyword
# search over the whole scenario — so completeness is judged step by step.
#
# The scenario provides them as *known* information; difficulty comes from the
# stakeholder not volunteering them, never from the simulator not knowing them.
# A structural test asserts the scenario's ``known_info`` covers every entry
# here, and that the ground truth never requires a truth the stakeholder cannot
# answer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepTruthRequirement:
    """A fact the stakeholder knows about one specific ground-truth step.

    ``step_concept`` ties the requirement to a step; ``key`` names the axis;
    ``signals`` are substrings that must appear in the scenario's ``known_info``
    so the simulator can truthfully answer a question about this fact.
    """

    step_concept: str
    key: str
    signals: tuple[str, ...]


def stakeholder_step_truth_requirements() -> list[StepTruthRequirement]:
    """Facts the stakeholder knows, keyed by step (concept) and axis.

    Every evaluator-graded fact must be answerable from the scenario's
    ``known_info``; otherwise the agent would be penalised because the
    simulator *could not* know an answer rather than because it did not
    volunteer it.
    """
    return [
        # receive_request
        StepTruthRequirement(
            "receive_request", "actor", ("sales", "営業", "you", "自分")
        ),
        StepTruthRequirement(
            "receive_request",
            "writes",
            ("request", "record the quotation request", "依頼"),
        ),
        # check_customer
        StepTruthRequirement(
            "check_customer", "actor", ("sales", "営業", "you", "自分")
        ),
        StepTruthRequirement(
            "check_customer",
            "system",
            ("crm", "customer relationship management", "CRM"),
        ),
        StepTruthRequirement("check_customer", "reads", ("customer", "顧客")),
        # create_quote
        StepTruthRequirement(
            "create_quote", "system", ("quoting", "見積", "見積システム")
        ),
        StepTruthRequirement(
            "create_quote", "reads", ("customer", "顧客", "pricing", "価格", "料金")
        ),
        # approve_quote (confirmed rationale)
        StepTruthRequirement(
            "approve_quote",
            "actor",
            ("manager", "approval", "上司", "承認者", "approver"),
        ),
        StepTruthRequirement(
            "approve_quote", "condition", ("1,000,000", "100万", "over", "exceed", "超")
        ),
        StepTruthRequirement(
            "approve_quote",
            "rationale",
            ("credit risk", "credit", "与信", "与信リスク"),
        ),
        # send_quote
        StepTruthRequirement("send_quote", "system", ("email", "メール")),
        StepTruthRequirement("send_quote", "reads", ("quotation", "quote", "見積")),
        # month_end_summary (UNKNOWN rationale)
        StepTruthRequirement(
            "month_end_summary", "system", ("excel", "spreadsheet", "エクセル", "Excel")
        ),
        StepTruthRequirement(
            "month_end_summary",
            "condition",
            ("month-end", "monthly", "month end", "月末"),
        ),
        StepTruthRequirement(
            "month_end_summary",
            "rationale_unknown",
            (
                "do not know",
                "does not know",
                "not know why",
                "知りません",
                "理由は知らない",
                "理由は分からない",
            ),
        ),
    ]


def missing_stakeholder_truth(
    known_info: str, scenario_id: Optional[str] = None
) -> list[StepTruthRequirement]:
    """Return the stakeholder-truth requirements the scenario cannot answer.

    Used by the completeness tests: an empty result means the stakeholder can
    truthfully answer every fact the evaluator grades for this scenario. The
    optional ``scenario_id`` can restrict the check to a scenario's steps, but
    the quotation scenario is the only task scenario with an agent-visible task.
    """
    known_l = (known_info or "").lower()
    missing: list[StepTruthRequirement] = []
    for req in stakeholder_step_truth_requirements():
        if not any(sig.lower() in known_l for sig in req.signals):
            missing.append(req)
    return missing
