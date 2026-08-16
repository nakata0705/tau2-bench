"""Evaluator-only ground truth for the business_interview workflow scenarios.

This is the canonical workflow the agent is expected to reconstruct by
interviewing a stakeholder. It is NEVER exposed to the agent (policy / tools) or
to the stakeholder (user scenario): the stakeholder knows only what the scenario
tells them, and the agent must discover the workflow, its actor/system/data,
branches, rationale, and the questionable legacy step on its own.

The design keeps the *workflow structure* (Workflow/Step/Branch) and the
*epistemic layer* (a source's FACT/BELIEF/UNKNOWN about a step's necessity)
separate. ``FACT`` here means a source asserted it as certain — not objective
truth; the objective rationale state is given by ``GTNecessity.objective_status``.
"""

from dataclasses import dataclass, field
from typing import Optional

from tau2.domains.business_interview.data_model import EpistemicStatus

# Suffix marking a Japanese pre-localized scenario variant. EN and JA variants
# share the same canonical ground truth.
JA_SCENARIO_SUFFIX = "_ja"


@dataclass
class GTNecessity:
    """Ground-truth expectation about a step's necessity / rationale."""

    objective_status: EpistemicStatus = EpistemicStatus.UNKNOWN
    # Optional expected source claim: (source, epistemic_status, claim_text).
    expected_claim: Optional[tuple[str, EpistemicStatus, str]] = None
    questionable: bool = False


@dataclass
class GTStep:
    id: str
    action: str
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
    purpose: str
    outcome: str
    steps: list[GTStep]
    transitions: list[GTTransition]
    branches: list[GTBranch]
    questionable_step: Optional[str] = None


def _quotation_workflow() -> WorkflowGroundTruth:
    """The single workflow scenario: quotation creation with a confirmed-rationale
    approval branch and a questionable legacy month-end Excel step."""
    confirmed = GTNecessity(
        objective_status=EpistemicStatus.FACT,
        expected_claim=("sales", EpistemicStatus.FACT, "credit risk"),
    )
    unknown = GTNecessity(objective_status=EpistemicStatus.UNKNOWN, questionable=True)
    return WorkflowGroundTruth(
        scenario_id="quotation_workflow_1",
        workflow_name="Quotation creation",
        trigger="A customer requests a quotation",
        purpose="Produce an accurate quotation for the customer",
        outcome="The customer receives a completed quotation",
        steps=[
            GTStep(
                id="s1",
                action="receive quotation request",
                actor="sales",
                reads=[],
                writes=["request"],
            ),
            GTStep(
                id="s2",
                action="check customer information in the CRM",
                actor="sales",
                system="crm",
                reads=["customer"],
                writes=[],
            ),
            GTStep(
                id="s3",
                action="create quotation in the quoting system",
                actor="sales",
                system="quoting",
                reads=["customer", "pricing"],
                writes=["quote"],
            ),
            GTStep(
                id="s4",
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
                action="send quotation to customer",
                actor="sales",
                system="email",
                reads=["quote"],
                writes=["sent_quote"],
            ),
            GTStep(
                id="s6",
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
            GTTransition("s3", "s4", "amount over 1,000,000"),
            GTTransition("s3", "s5", "amount at or below 1,000,000"),
            GTTransition("s4", "s5"),
            GTTransition("s5", "s6", "month-end"),
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
