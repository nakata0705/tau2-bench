"""Workflow-first data model for the business_interview domain (v2).

The central concept is a reconstructed *workflow graph*, not a bag of facts.
The agent records the business process it learns as a Workflow made of Steps,
Transitions and Branches. Each Step carries its structural attributes (actor,
system, reads, writes, condition) and its *necessity* (why the step is needed,
who requires it, what evidence exists, and whether it was challenged).

Epistemic information (a source's FACT / BELIEF / UNKNOWN about a step's
necessity) is a secondary layer attached to a Step's ``Necessity``. ``FACT``
means the source asserted it as certain — it is NOT objective truth; objective
correctness is judged by the evaluator against the benchmark ground truth.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class EpistemicStatus(str, Enum):
    """How a source presented a claim about a step's necessity.

    - ``FACT``: the source asserted it as certain / definitive.
    - ``BELIEF``: the source presented it as their own opinion / guess.
    - ``UNKNOWN``: the matter is unconfirmed / not known.

    This describes the *source's* claim. Objective correctness is a separate
    concern decided by the evaluator against the scenario ground truth.
    """

    FACT = "FACT"
    BELIEF = "BELIEF"
    UNKNOWN = "UNKNOWN"


class Necessity(BaseModel):
    """Why a step is needed and how strongly it is grounded.

    Captures the answers to the necessity questions a good BA asks:

    - why is this step needed?  (``rationale`` / ``rationale_known``)
    - who requires it?          (``source`` / ``owner_identified``)
    - what is the evidence?     (``evidence_identified``)
    - what happens if removed?  (part of ``challenges``)
    - can it be deleted / simplified?  (``challenged`` / ``deletion_candidate``)

    ``epistemic_status`` is how the source presented the rationale (FACT /
    BELIEF / UNKNOWN); it is separate from the objective state the evaluator
    holds in the ground truth.
    """

    rationale_known: bool = Field(
        default=False,
        description="True if a rationale for the step's necessity was stated.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="The stated reason the step is needed (paraphrase OK).",
    )
    source: Optional[str] = Field(
        default=None,
        description="Who stated the rationale / who requires the step.",
    )
    epistemic_status: Optional[EpistemicStatus] = Field(
        default=None,
        description=(
            "How the source presented the rationale: FACT (asserted as certain), "
            "BELIEF (opinion/guess), or UNKNOWN (not known)."
        ),
    )
    owner_identified: bool = Field(
        default=False,
        description="True if the owner / who-requires-it was asked and recorded.",
    )
    evidence_identified: bool = Field(
        default=False,
        description="True if concrete evidence for the requirement was asked and recorded.",
    )
    requirement_type: Optional[str] = Field(
        default=None,
        description="Optional type of the requirement (e.g. 'customer', 'regulatory', 'internal').",
    )
    challenged: bool = Field(
        default=False,
        description="True if the agent questioned this step's necessity.",
    )
    challenges: list[str] = Field(
        default_factory=list,
        description=(
            "The necessity questions the agent asked (e.g. 'why is this needed?', "
            "'who requires it?', 'what if removed?')."
        ),
    )
    deletion_candidate: bool = Field(
        default=False,
        description=(
            "True if the agent identified the step as a candidate for deletion / "
            "simplification (questioned before automating)."
        ),
    )


class WorkflowStep(BaseModel):
    """A single step in the reconstructed workflow."""

    id: str = Field(
        description="Agent-assigned step identifier (not hidden ground truth)."
    )
    action: str = Field(description="What is done in this step.")
    actor: Optional[str] = Field(
        default=None, description="Who performs the step (role/person)."
    )
    system: Optional[str] = Field(
        default=None, description="Which system / tool is used."
    )
    reads: list[str] = Field(default_factory=list, description="Data the step reads.")
    writes: list[str] = Field(
        default_factory=list, description="Data the step creates / writes."
    )
    condition: Optional[str] = Field(
        default=None, description="When / under what condition the step happens."
    )
    necessity: Necessity = Field(
        default_factory=Necessity,
        description="Why the step is needed (necessity / requirement info).",
    )


class Transition(BaseModel):
    """An edge in the workflow graph."""

    from_step: str = Field(description="Source step id.")
    to_step: str = Field(description="Destination step id.")
    condition: Optional[str] = Field(
        default=None, description="Optional condition on this transition."
    )


class Branch(BaseModel):
    """An explicit conditional divergence: a step whose outgoing paths depend on
    a condition (e.g. amount threshold)."""

    from_step: str = Field(description="The step at which the flow diverges.")
    condition: str = Field(description="The branching condition.")
    paths: list[str] = Field(
        default_factory=list, description="The divergent next step ids."
    )


class Workflow(BaseModel):
    """The reconstructed business workflow."""

    id: str = Field(description="Agent-assigned workflow identifier.")
    name: str = Field(description="Workflow name.")
    trigger: Optional[str] = Field(
        default=None, description="What starts the workflow."
    )
    purpose: Optional[str] = Field(default=None, description="Why the workflow exists.")
    outcome: Optional[str] = Field(default=None, description="The intended outcome.")
    steps: list[WorkflowStep] = Field(
        default_factory=list, description="The steps in the workflow."
    )
    transitions: list[Transition] = Field(
        default_factory=list, description="The transitions between steps."
    )
    branches: list[Branch] = Field(
        default_factory=list, description="Explicit conditional branches."
    )


class WorkflowEvaluation(BaseModel):
    """Structural + epistemic + challenge diagnostics for a reconstruction.

    The scalar reward connects minimally to this; the detailed, human-readable
    axes live here (and are surfaced via ``get_eval_diagnostics``).

    Accuracy metrics are fractions in [0,1] over the ground-truth items
    (matched reconstructed steps to ground-truth steps by content).
    """

    protocol_completed: bool = Field(
        description="True if the agent called finish_interview."
    )
    workflow_created: bool = Field(
        description="True if the agent created a workflow with at least one step."
    )
    step_recall: float = Field(
        description="Fraction of ground-truth steps that were reconstructed."
    )
    unexpected_step_count: int = Field(
        description="Number of reconstructed steps that match no ground-truth step."
    )
    actor_accuracy: float = Field(
        description="Fraction of matched steps whose actor matches the ground truth."
    )
    system_accuracy: float = Field(
        description="Fraction of matched steps whose system matches the ground truth."
    )
    data_read_accuracy: float = Field(
        description="Fraction of matched steps whose read-data set matches."
    )
    data_write_accuracy: float = Field(
        description="Fraction of matched steps whose write-data set matches."
    )
    transition_accuracy: float = Field(
        description="Fraction of ground-truth transitions reconstructed (from/to)."
    )
    branch_recall: float = Field(
        description="Fraction of ground-truth branches reconstructed (both paths present)."
    )
    branch_condition_accuracy: float = Field(
        description="Fraction of reconstructed branches whose condition matches."
    )
    rationale_coverage: float = Field(
        description="Fraction of ground-truth steps whose necessity was covered (rationale or UNKNOWN preserved)."
    )
    uncertainty_handling: bool = Field(
        description="True if every ground-truth UNKNOWN-rationale step was preserved as UNKNOWN (not fabricated)."
    )
    fabricated_rationale: bool = Field(
        description="True if a rationale was asserted as FACT for a step whose objective rationale is UNKNOWN."
    )
    challenge_done: bool = Field(
        description="True if the questionable legacy step was challenged."
    )
    improvement_order_ok: bool = Field(
        description="True if the agent questioned necessity before proposing deletion/automation."
    )
    # Convenience gates (used by the reward env assertions).
    structural_pass: bool = Field(
        description="True if steps + transitions + branches + actor/system/data reconstruct the ground truth."
    )
    rationale_pass: bool = Field(
        description="True if confirmed rationale captured and UNKNOWN preserved (no fabrication)."
    )
    challenge_pass: bool = Field(
        description="True if the questionable step was challenged with correct improvement order."
    )
    protocol_pass: bool = Field(
        description="Equal to protocol_completed (named for protocol-vs-quality comparison)."
    )
    quality_pass: bool = Field(
        description="True if structural_pass AND rationale_pass AND challenge_pass (independent of protocol)."
    )


class Improvement(BaseModel):
    """An improvement proposal recorded by the agent.

    ``kind`` follows the benchmark improvement order: question, delete, simplify,
    accelerate, automate. A good BA questions the requirement before automating.
    """

    step_id: Optional[str] = Field(
        default=None, description="Step the proposal concerns (optional)."
    )
    kind: str = Field(
        description="question | delete | simplify | accelerate | automate"
    )
    note: str = Field(default="", description="The proposal.")
    order: int = Field(
        default=0,
        description="Sequence order (auto-incrementing as proposals are made).",
    )


class WorkflowDB(DB):
    """State of a business interview: the reconstructed workflow.

    The benchmark ground truth (the canonical workflow, hidden steps, correct
    actor/system/data, objective rationale, expected claims, challenge target)
    lives evaluator-side in the domain code — never in the agent-visible policy
    or the stakeholder scenario. This database only stores what the agent
    explicitly records during the interview.
    """

    workflow: Optional[Workflow] = Field(
        default=None,
        description="The reconstructed workflow built by the agent.",
    )
    improvements: list[Improvement] = Field(
        default_factory=list,
        description="Improvement proposals, in the order made.",
    )
    interview_complete: bool = Field(
        default=False,
        description="Whether the agent called finish_interview.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional wrap-up summary provided to finish_interview.",
    )
