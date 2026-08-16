"""Workflow-first data model for the business_interview domain (v2, hardened).

The central concept is a reconstructed *workflow graph*, not a bag of facts.
The agent records the business process it learns as a Workflow made of Steps,
Transitions and Branches. Each Step carries its structural attributes (actor,
system, reads, writes, condition) and its *necessity* (why the step is needed,
who requires it, what evidence exists, what happens if removed, and whether it
was challenged).

Design principles of the hardened model:

- **ASKED vs RESULT RECORDED are separate axes.** A question being asked
  (``*_asked``) is never conflated with an answer being recorded (the
  ``*_result`` / ``rationale_result`` fields). The benchmark evaluates whether
  a *result was recorded*, not merely whether a question was posed.

- **UNKNOWN is an explicit recorded outcome.** ``UNKNOWN`` means the
  stakeholder was asked and confirmed they do not know — and that outcome was
  recorded. An unasked / unanswered dimension is ``NOT_RECORDED``, never
  ``UNKNOWN``. ``NONE_FOUND`` means the investigation happened and nothing
  exists.

- **Observations vs analyst assessment.** ``why`` / ``owner`` / ``evidence`` /
  ``removal`` are *observations* obtained from the stakeholder and recorded via
  ``record_necessity_detail`` / ``set_step_rationale`` / ``set_step_unknown``.
  ``deletion_considered`` / ``deletion_candidate`` are the analyst's own
  assessment and are recorded separately. ``challenge_pass`` requires the
  necessary observations to be recorded.

- **No hidden canonical ids and no ``concept -> single step`` constraint.** The
  agent uses its own step ids and its own action text; the evaluator resolves
  actions to language-independent *concepts* and supports multiple steps
  sharing the same concept, distinguishing them deterministically by
  actor/system, then data.
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


class RationaleResult(str, Enum):
    """The recorded outcome of the "why is this step needed?" investigation.

    - ``NOT_RECORDED``: the question was not asked or no answer was recorded.
    - ``FACT``: a reason was recorded as asserted certain.
    - ``BELIEF``: a reason was recorded as the stakeholder's opinion / guess.
    - ``UNKNOWN``: the stakeholder confirmed they do not know, recorded as such.
    """

    NOT_RECORDED = "NOT_RECORDED"
    FACT = "FACT"
    BELIEF = "BELIEF"
    UNKNOWN = "UNKNOWN"


class NecessityResult(str, Enum):
    """The recorded outcome of an owner / evidence / removal investigation.

    - ``NOT_RECORDED``: not asked or no answer recorded.
    - ``KNOWN``: a concrete value was found and recorded.
    - ``UNKNOWN``: the stakeholder confirmed they do not know (recorded).
    - ``NONE_FOUND``: investigated; nothing exists / no one / no evidence.
    """

    NOT_RECORDED = "NOT_RECORDED"
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    NONE_FOUND = "NONE_FOUND"


class Necessity(BaseModel):
    """Why a step is needed and how strongly it is grounded.

    The model separates *ASKED* from *RESULT RECORDED* on every necessity
    dimension:

    - ``*_asked`` flags record that a question was posed (from ``challenge_step``).
    - ``*_result`` / ``rationale_result`` fields record the outcome that was
      explicitly recorded. A result may be ``NOT_RECORDED`` even when the
      question was asked — an asked-but-unanswered dimension is never treated
      as ``UNKNOWN`` / ``NONE_FOUND``.

    ``deletion_considered`` / ``deletion_candidate`` are the analyst's own
    assessment (separate from the stakeholder observations).
    """

    # --- ASKED (was the question posed to the stakeholder?) -----------------
    why_asked: bool = Field(
        default=False,
        description="True if the 'why is this step needed?' question was asked.",
    )
    owner_asked: bool = Field(
        default=False,
        description="True if who-requires-it / owner was asked.",
    )
    evidence_asked: bool = Field(
        default=False,
        description="True if supporting evidence was asked about.",
    )
    removal_asked: bool = Field(
        default=False,
        description="True if 'what happens if removed?' was asked.",
    )

    # --- analyst assessment (NOT a stakeholder observation) -----------------
    deletion_considered: bool = Field(
        default=False,
        description="True if deletion/simplification of this step was considered (analyst assessment).",
    )
    deletion_candidate: bool = Field(
        default=False,
        description=(
            "True if the agent identified the step as a candidate for deletion / "
            "simplification (questioned before automating)."
        ),
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

    # --- why / rationale observation ----------------------------------------
    rationale_result: RationaleResult = Field(
        default=RationaleResult.NOT_RECORDED,
        description="The recorded outcome of the why-investigation.",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="The stated reason the step is needed (paraphrase OK).",
    )
    source: Optional[str] = Field(
        default=None,
        description="Who stated the rationale (the stakeholder).",
    )

    # --- owner observation ---------------------------------------------------
    owner_result: NecessityResult = Field(
        default=NecessityResult.NOT_RECORDED,
        description="The recorded outcome of the owner investigation.",
    )
    owner: Optional[str] = Field(
        default=None, description="Who requires / owns the requirement (may be none)."
    )

    # --- evidence observation ------------------------------------------------
    evidence_result: NecessityResult = Field(
        default=NecessityResult.NOT_RECORDED,
        description="The recorded outcome of the evidence investigation.",
    )
    evidence: Optional[str] = Field(
        default=None, description="Evidence for the requirement (may be none)."
    )

    # --- removal-impact observation ------------------------------------------
    removal_result: NecessityResult = Field(
        default=NecessityResult.NOT_RECORDED,
        description="The recorded outcome of the removal-impact investigation.",
    )
    removal_impact: Optional[str] = Field(
        default=None, description="What happens if the step is removed (may be none)."
    )
    requirement_type: Optional[str] = Field(
        default=None,
        description="Optional type of the requirement (e.g. 'customer', 'regulatory', 'internal').",
    )

    # --- convenience ----------------------------------------------------------
    @property
    def why_recorded(self) -> bool:
        return self.rationale_result != RationaleResult.NOT_RECORDED

    @property
    def owner_recorded(self) -> bool:
        return self.owner_result != NecessityResult.NOT_RECORDED

    @property
    def evidence_recorded(self) -> bool:
        return self.evidence_result != NecessityResult.NOT_RECORDED

    @property
    def removal_recorded(self) -> bool:
        return self.removal_result != NecessityResult.NOT_RECORDED


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

    All graph evaluations (transitions / branches / challenge / improvement)
    are computed over the *matched-step mapping* — the reconstructed step ids
    are first canonicalised to ground-truth step ids, so the agent's arbitrary
    step ids never leak into the score.

    Accuracy metrics are fractions in [0,1] over the ground-truth items.

    Challenge diagnostics separate the *asked* axis from the *result recorded*
    axis for each dimension, so a failure reason is human-readable (e.g.
    ``owner_asked: true`` / ``owner_recorded: false``).
    """

    protocol_completed: bool = Field(
        description="True if the agent called finish_interview."
    )
    workflow_created: bool = Field(
        description="True if the agent created a workflow with at least one step."
    )

    # --- workflow metadata ---------------------------------------------------
    trigger_accuracy: float = Field(
        description="1.0 if the recorded trigger resolves to the ground-truth trigger concept."
    )
    purpose_accuracy: float = Field(
        description="1.0 if the recorded purpose resolves to the ground-truth purpose concept."
    )
    outcome_accuracy: float = Field(
        description="1.0 if the recorded outcome resolves to the ground-truth outcome concept."
    )

    # --- steps (content / concept matching) ----------------------------------
    step_recall: float = Field(
        description="Fraction of ground-truth steps that were reconstructed (by concept)."
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

    # --- data (recall + precision) -------------------------------------------
    data_read_recall: float = Field(
        description="Fraction of ground-truth read-data items covered by the recorded reads."
    )
    data_read_precision: float = Field(
        description="Fraction of recorded read-data items that are grounded in the ground truth."
    )
    data_write_recall: float = Field(
        description="Fraction of ground-truth write-data items covered by the recorded writes."
    )
    data_write_precision: float = Field(
        description="Fraction of recorded write-data items that are grounded in the ground truth."
    )

    # --- graph: transitions + branches (condition-aware) ---------------------
    transition_accuracy: float = Field(
        description="Fraction of ground-truth transitions reconstructed (from/to AND condition)."
    )
    branch_recall: float = Field(
        description="Fraction of ground-truth branches reconstructed (all paths present)."
    )
    branch_condition_accuracy: float = Field(
        description="Fraction of reconstructed branches that are conditioned (not a plain linear split)."
    )

    # --- rationale / uncertainty ---------------------------------------------
    rationale_coverage: float = Field(
        description="Fraction of ground-truth necessity steps fully handled (recorded + correct)."
    )
    confirmed_rationale_ok: bool = Field(
        description="True if the confirmed rationale's epistemic status, source and content all match."
    )
    uncertainty_handling: bool = Field(
        description="True if every ground-truth UNKNOWN-rationale step was preserved as UNKNOWN (not fabricated)."
    )
    fabricated_rationale: bool = Field(
        description="True if a rationale was asserted as FACT for a step whose objective rationale is UNKNOWN."
    )

    # --- necessity challenge (ASKED vs RESULT RECORDED per dimension) --------
    challenge_target_identified: bool = Field(
        description="True if the questionable step was found among the reconstructed steps."
    )
    why_asked: bool = Field(
        description="True if the 'why' of the questionable step was asked."
    )
    why_recorded: bool = Field(
        description="True if a 'why' result was recorded for the questionable step."
    )
    owner_asked: bool = Field(
        description="True if the owner of the questionable step was asked."
    )
    owner_recorded: bool = Field(
        description="True if an owner result was recorded for the questionable step."
    )
    owner_result: str = Field(
        description="Recorded owner outcome (KNOWN / UNKNOWN / NONE_FOUND / NOT_RECORDED)."
    )
    evidence_asked: bool = Field(
        description="True if evidence for the questionable step was asked."
    )
    evidence_recorded: bool = Field(
        description="True if an evidence result was recorded for the questionable step."
    )
    evidence_result: str = Field(
        description="Recorded evidence outcome (KNOWN / UNKNOWN / NONE_FOUND / NOT_RECORDED)."
    )
    removal_asked: bool = Field(
        description="True if removal-impact of the questionable step was asked."
    )
    removal_recorded: bool = Field(
        description="True if a removal result was recorded for the questionable step."
    )
    removal_result: str = Field(
        description="Recorded removal outcome (KNOWN / UNKNOWN / NONE_FOUND / NOT_RECORDED)."
    )
    deletion_considered: bool = Field(
        description="True if deletion of the questionable step was considered (analyst assessment)."
    )
    challenge_done: bool = Field(
        description="True if every required observation was recorded and deletion was considered."
    )
    improvement_order_ok: bool = Field(
        description="True if necessity was recorded before proposing delete/simplify/accelerate/automate."
    )

    # --- gates ---------------------------------------------------------------
    structural_pass: bool = Field(
        description="True if metadata + steps + actor/system + data + transitions + branches reconstruct the ground truth."
    )
    rationale_pass: bool = Field(
        description="True if confirmed rationale captured correctly and UNKNOWN preserved (no fabrication)."
    )
    challenge_pass: bool = Field(
        description="True if the questionable step's observations were all recorded with correct improvement order."
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
