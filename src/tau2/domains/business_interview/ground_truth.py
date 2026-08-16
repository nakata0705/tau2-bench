"""Evaluator-only ground truth for the business_interview workflow scenarios.

This is the canonical workflow the agent is expected to reconstruct by
interviewing a stakeholder. It is NEVER exposed to the agent (policy / tools /
task description) or to the stakeholder (user scenario). The stakeholder knows
only what the scenario tells them; the agent must discover the workflow, its
actor/system/data, branches, rationale, and the questionable legacy step on its
own.

Three layers are kept distinct:

- **Evaluator Ground Truth** (``WorkflowGroundTruth``) — the objective, complete
  workflow the evaluator grades against.
- **Stakeholder Truth** (``StakeholderTruth``) — the facts the stakeholder can
  answer when asked, expressed **per step (by concept) and per axis** as
  structured data. It is not a keyword search over the scenario text.
- **Agent Reconstruction** — the workflow the agent records during the
  interview.

Completeness is a **structural comparison**: every axis the evaluator requires
for a step must be present in that step's Stakeholder Truth (``stakeholder_truth_completeness``),
so a word that happens to appear in another step can never satisfy it. A
separate consistency check (``missing_from_instructions``) verifies the user
simulator's prompt can actually answer every Stakeholder Truth value, so we never
create a state where the evaluator has truth the simulator cannot answer.

``FACT`` here means a source asserted it as certain — not objective truth; the
objective rationale state is given by ``GTNecessity.objective_status``. The
questionable step's owner / evidence / removal observations carry explicit
ground-truth expectations (``expected_owner_result`` etc.) used to judge whether
a recorded observation is *correct*, not merely *recorded*.
"""

from dataclasses import dataclass, field
from typing import Optional

from tau2.domains.business_interview.aliases import value_signals
from tau2.domains.business_interview.data_model import EpistemicStatus, NecessityResult

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
    - ``expected_owner_result`` / ``expected_evidence_result`` /
      ``expected_removal_result``: the ground-truth owner / evidence / removal
      observation results for a questionable step (KNOWN / UNKNOWN / NONE_FOUND).
      Used to judge whether a recorded observation is *correct*.
    """

    objective_status: EpistemicStatus = EpistemicStatus.UNKNOWN
    investigated_required: bool = True
    rationale_concept: Optional[str] = None
    expected_source: Optional[str] = None
    expected_status: Optional[EpistemicStatus] = None
    questionable: bool = False
    expected_owner_result: Optional[NecessityResult] = None
    expected_evidence_result: Optional[NecessityResult] = None
    expected_removal_result: Optional[NecessityResult] = None


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
        expected_owner_result=NecessityResult.UNKNOWN,
        expected_evidence_result=NecessityResult.NONE_FOUND,
        expected_removal_result=NecessityResult.UNKNOWN,
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
# Stakeholder Truth (structured, step-unit, per scenario)
#
# These are the facts the stakeholder can answer when asked, expressed per step
# (by concept) and per axis. The evaluator grades against them structurally;
# the user simulator must be able to answer them (checked by
# ``missing_from_instructions``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StakeholderObservation:
    """A stakeholder's stated observation for an axis.

    ``result`` is the state the stakeholder can confirm: for rationale
    ``FACT`` / ``BELIEF`` / ``UNKNOWN``; for owner/evidence/removal ``KNOWN`` /
    ``UNKNOWN`` / ``NONE_FOUND``. ``value`` is an optional concrete value /
    belief label.
    """

    result: str
    value: Optional[str] = None


@dataclass
class StakeholderStepTruth:
    """The stakeholder-known facts about one step (by concept)."""

    concept: str
    actor: Optional[str] = None
    system: Optional[str] = None
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    condition: Optional[str] = None
    rationale: Optional[StakeholderObservation] = None
    owner: Optional[StakeholderObservation] = None
    evidence: Optional[StakeholderObservation] = None
    removal: Optional[StakeholderObservation] = None


@dataclass
class StakeholderTruth:
    scenario_id: str
    steps: dict[str, StakeholderStepTruth]


def _quotation_stakeholder_truth() -> StakeholderTruth:
    return StakeholderTruth(
        scenario_id="quotation_workflow_1",
        steps={
            "receive_request": StakeholderStepTruth(
                concept="receive_request", actor="sales", writes=["request"]
            ),
            "check_customer": StakeholderStepTruth(
                concept="check_customer",
                actor="sales",
                system="crm",
                reads=["customer"],
            ),
            "create_quote": StakeholderStepTruth(
                concept="create_quote",
                actor="sales",
                system="quoting",
                reads=["customer", "pricing"],
                writes=["quote"],
            ),
            "approve_quote": StakeholderStepTruth(
                concept="approve_quote",
                actor="manager",
                system="quoting",
                reads=["quote"],
                writes=["approval"],
                condition="1,000,000",
                rationale=StakeholderObservation(result="FACT", value="credit risk"),
            ),
            "send_quote": StakeholderStepTruth(
                concept="send_quote",
                actor="sales",
                system="email",
                reads=["quote"],
                writes=["sent_quote"],
            ),
            "month_end_summary": StakeholderStepTruth(
                concept="month_end_summary",
                actor="sales",
                system="excel",
                reads=["quote"],
                writes=["excel_summary"],
                condition="month-end",
                rationale=StakeholderObservation(result="UNKNOWN"),
                owner=StakeholderObservation(result="UNKNOWN", value="accounting"),
                evidence=StakeholderObservation(result="NONE_FOUND"),
                removal=StakeholderObservation(result="UNKNOWN"),
            ),
        },
    )


_STAKEHOLDER_TRUTHS: dict[str, StakeholderTruth] = {
    "quotation_workflow_1": _quotation_stakeholder_truth(),
}


def get_stakeholder_truth(scenario_id: Optional[str]) -> Optional[StakeholderTruth]:
    """Return the structured Stakeholder Truth for a scenario (or None)."""
    canonical = canonical_scenario_id(scenario_id)
    if canonical is None:
        return None
    return _STAKEHOLDER_TRUTHS.get(canonical)


# ---------------------------------------------------------------------------
# Structural completeness (evaluator-required fields vs Stakeholder Truth)
# ---------------------------------------------------------------------------


def required_truth_axes(gt_step: GTStep) -> list[str]:
    """The axes the evaluator grades for a step that the stakeholder must know."""
    axes = ["actor"]
    if gt_step.system:
        axes.append("system")
    if gt_step.reads:
        axes.append("reads")
    if gt_step.writes:
        axes.append("writes")
    if gt_step.condition:
        axes.append("condition")
    if gt_step.necessity is not None:
        axes.append("rationale")
        if gt_step.necessity.questionable:
            axes.extend(["owner", "evidence", "removal"])
    return axes


def stakeholder_truth_completeness(
    scenario_id: Optional[str],
    stakeholder_truth: Optional[StakeholderTruth] = None,
) -> list[tuple[str, str]]:
    """Return ``(step_concept, axis)`` pairs the scenario's Stakeholder Truth
    is missing for axes the evaluator requires.

    This is a **structural** per-step comparison: a value appearing in another
    step's truth can never satisfy a missing axis here. An empty result means
    the evaluator requires nothing the stakeholder could not know.

    ``stakeholder_truth`` may be overridden (e.g. a modified copy) to test
    completeness failure; by default the scenario's own structured truth is used.
    """
    gt = get_ground_truth(scenario_id)
    st = (
        stakeholder_truth
        if stakeholder_truth is not None
        else get_stakeholder_truth(scenario_id)
    )
    if gt is None or st is None:
        return []
    missing: list[tuple[str, str]] = []
    for gs in gt.steps:
        present: set[str] = set()
        t = st.steps.get(gs.concept)
        if t is not None:
            if t.actor is not None:
                present.add("actor")
            if t.system is not None:
                present.add("system")
            if t.reads:
                present.add("reads")
            if t.writes:
                present.add("writes")
            if t.condition is not None:
                present.add("condition")
            if t.rationale is not None:
                present.add("rationale")
            if t.owner is not None:
                present.add("owner")
            if t.evidence is not None:
                present.add("evidence")
            if t.removal is not None:
                present.add("removal")
        for axis in required_truth_axes(gs):
            if axis not in present:
                missing.append((gs.concept, axis))
    return missing


# ---------------------------------------------------------------------------
# Simulator-answerability consistency (Stakeholder Truth vs instructions text)
#
# Guarantees we never create a state where the evaluator has truth the user
# simulator cannot answer. The simulator prompt (known_info + unknown_info +
# task_instructions) must be able to produce every Stakeholder Truth value.
# ---------------------------------------------------------------------------

_UNKNOWN_SIGNALS = (
    "do not know",
    "does not know",
    "don't know",
    "not know why",
    "知りません",
    "知らない",
    "理由は分からない",
    "理由は知らない",
    "分かりません",
)
_NO_FINDINGS_SIGNALS = (
    "no documentation",
    "no evidence",
    "no formal evidence",
    "文書もありません",
    "文書",
    "根拠はない",
)


def missing_from_instructions(
    scenario_id: Optional[str], instructions_text: str
) -> list[str]:
    """Return labels of Stakeholder Truth values the instructions cannot answer.

    ``instructions_text`` is the concatenated simulator-facing instructions
    (known_info + unknown_info + task_instructions). Each stakeholder-known value
    must be expressible from it (using alias / concept expansion), and each
    ``UNKNOWN`` / ``NONE_FOUND`` observation must have a matching explicit
    unknown / no-findings signal.
    """
    st = get_stakeholder_truth(scenario_id)
    if st is None:
        return []
    known_l = (instructions_text or "").lower()
    missing: list[str] = []

    def check(value: Optional[str], axis: str, label: str) -> None:
        if not value:
            return
        signals = value_signals(value, axis)
        if not any(s in known_l for s in signals):
            missing.append(label)

    def check_unknown(result: str, label: str) -> None:
        if result == "UNKNOWN" and not any(s in known_l for s in _UNKNOWN_SIGNALS):
            missing.append(label)
        elif result == "NONE_FOUND" and not any(
            s in known_l for s in _NO_FINDINGS_SIGNALS
        ):
            missing.append(label)

    for concept, t in st.steps.items():
        check(t.actor, "actor", f"{concept}.actor")
        check(t.system, "system", f"{concept}.system")
        for r in t.reads:
            check(r, "data", f"{concept}.reads")
        for w in t.writes:
            check(w, "data", f"{concept}.writes")
        check(t.condition, "data", f"{concept}.condition")
        if t.rationale is not None:
            check(t.rationale.value, "data", f"{concept}.rationale.value")
            if t.rationale.result in ("UNKNOWN", "NONE_FOUND"):
                check_unknown(
                    t.rationale.result, f"{concept}.rationale.{t.rationale.result}"
                )
        if t.owner is not None:
            check(t.owner.value, "data", f"{concept}.owner.value")
            if t.owner.result in ("UNKNOWN", "NONE_FOUND"):
                check_unknown(t.owner.result, f"{concept}.owner.{t.owner.result}")
        if t.evidence is not None:
            check(t.evidence.value, "data", f"{concept}.evidence.value")
            if t.evidence.result in ("UNKNOWN", "NONE_FOUND"):
                check_unknown(
                    t.evidence.result, f"{concept}.evidence.{t.evidence.result}"
                )
        if t.removal is not None:
            check(t.removal.value, "data", f"{concept}.removal.value")
            if t.removal.result in ("UNKNOWN", "NONE_FOUND"):
                check_unknown(t.removal.result, f"{concept}.removal.{t.removal.result}")
    return missing
