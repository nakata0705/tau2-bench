from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class Topic(str, Enum):
    """Canonical, language-independent business-process topic identifiers.

    A finding is associated with a topic when the agent records a canonical
    ``topic`` on it (preferred) or, as a backward-compatible fallback, when the
    finding's content matches a topic's bilingual identity signals. Keeping the
    identity canonical (rather than relying on surface-language keyword
    synonyms) is what lets the evaluator attribute a Fact / Belief / Unknown to
    the *right* business element even when two topics coexist in one interview.

    These identifiers are scenario-agnostic so the same topic can be reused
    (and later shared across stakeholders) in future multi-stakeholder
    scenarios.
    """

    MONTH_END_EXCEL = "month_end_excel"
    HIGH_VALUE_QUOTE = "high_value_quote"


class Source(str, Enum):
    """Who stated a claim (source provenance).

    This is distinct from ``subject`` (what / whom a finding is *about*):
    ``source`` is the stakeholder / source that *said* the claim. In a future
    multi-stakeholder benchmark a claim like ``source=sales, about=accounting``
    is possible, so ``source`` and the claim's target are kept as separate
    concepts.
    """

    SALES = "sales"
    ACCOUNTING = "accounting"
    UNKNOWN = "unknown"


class RationaleValue(str, Enum):
    """A minimal canonical value for a rationale claim.

    Free text alone is hard to compare across stakeholders later, so a claim
    may carry a small canonical value (e.g. ``accounting_need``, ``credit_risk``).
    This is deliberately a tiny, bounded vocabulary — not a generic proposition
    engine.
    """

    ACCOUNTING_NEED = "accounting_need"
    CREDIT_RISK = "credit_risk"
    UNKNOWN = "unknown"


class EpistemicStatus(str, Enum):
    """The epistemic status of a recorded finding.

    These describe how the *source* (the stakeholder) presented the claim;
    objective correctness is a separate concern decided by the evaluator by
    comparing findings against the benchmark ground truth.

    - ``FACT``: the stakeholder asserted the claim as certain / definitive.
    - ``BELIEF``: the stakeholder presented the claim as their own opinion,
      guess, or impression (not certain).
    - ``UNKNOWN``: the matter is unconfirmed / not known (recorded as an
      uncertainty rather than guessed).
    - ``EXCEPTION``: the finding records a process exception (not an epistemic
      claim about it).

    This is the canonical, language-agnostic state that the semantic evaluator
    reasons about. Recording `epistemic_status` lets a well-behaved agent
    separate belief from fact without relying on surface-language keywords.
    Whether a recorded claim is *objectively correct* is judged by the
    evaluator against the task's ground truth, not by the status alone.
    """

    FACT = "FACT"
    BELIEF = "BELIEF"
    UNKNOWN = "UNKNOWN"
    EXCEPTION = "EXCEPTION"


class InterviewFact(BaseModel):
    """A fact about the current process that the interviewee stated."""

    content: str = Field(
        description="The fact as stated by the interviewee (paraphrase OK)."
    )
    epistemic_status: EpistemicStatus = Field(
        default=EpistemicStatus.FACT,
        description=(
            "Epistemic status of the finding. Use BELIEF for statements the "
            "interviewee qualified as their own opinion/guess (e.g. 'I think ...'), "
            "never promote those to FACT. Use FACT only for statements the "
            "interviewee asserted as certain."
        ),
    )
    subject: Optional[str] = Field(
        default=None,
        description=(
            "Optional subject/role the finding is about (e.g. 'sales_employee'). "
            "Reserved for future multi-stakeholder scenarios; unused today."
        ),
    )
    topic: Optional[Topic] = Field(
        default=None,
        description=(
            "Canonical topic the finding is about (e.g. 'month_end_excel', "
            "'high_value_quote'). When set, evaluation attributes the finding to "
            "that business element regardless of surface wording. Optional for "
            "backward compatibility: if omitted, the evaluator falls back to "
            "content-based topic association."
        ),
    )
    source: Optional[Source] = Field(
        default=None,
        description=(
            "Who stated this claim (source provenance), e.g. 'sales'. Distinct "
            "from subject (what the finding is about). Optional; when omitted the "
            "single stakeholder (sales) is assumed."
        ),
    )
    value: Optional[RationaleValue] = Field(
        default=None,
        description=(
            "Optional minimal canonical value of a rationale claim (e.g. "
            "'accounting_need', 'credit_risk'). This is a small structured value "
            "that lets the evaluator compare claims across stakeholders later. "
            "Optional: content-based value detection is used as a fallback."
        ),
    )


class InterviewException(BaseModel):
    """An exception / process variation the interviewee described."""

    content: str = Field(
        description="The exception process as described by the interviewee."
    )
    subject: Optional[str] = Field(
        default=None,
        description=(
            "Optional subject/role the finding is about (e.g. 'sales_employee'). "
            "Reserved for future multi-stakeholder scenarios; unused today."
        ),
    )
    topic: Optional[Topic] = Field(
        default=None,
        description=(
            "Canonical topic the exception is about (e.g. 'month_end_excel'). "
            "Optional for backward compatibility; see InterviewFact.topic."
        ),
    )
    source: Optional[Source] = Field(
        default=None,
        description=("Who described this exception (source provenance). Optional."),
    )


class InterviewUncertainty(BaseModel):
    """Something the interviewee does not know."""

    content: str = Field(
        description="What the interviewee could not answer (recorded as uncertainty, never guessed)."
    )
    subject: Optional[str] = Field(
        default=None,
        description=(
            "Optional subject/role the finding is about (e.g. 'sales_employee'). "
            "Reserved for future multi-stakeholder scenarios; unused today."
        ),
    )
    topic: Optional[Topic] = Field(
        default=None,
        description=(
            "Canonical topic the uncertainty is about (e.g. 'month_end_excel'). "
            "Optional for backward compatibility; see InterviewFact.topic."
        ),
    )
    source: Optional[Source] = Field(
        default=None,
        description=("Who stated that they do not know (source provenance). Optional."),
    )


class ObjectiveRationale(BaseModel):
    """The objective state of a topic's rationale.

    This comes from the scenario ground truth (``TopicSpec.expected_rationale_``
    ``status``) and is NOT something the agent sets. ``correct`` reflects whether
    the agent *represented* that objective state correctly (e.g. an UNKNOWN
    objective must be preserved as an uncertainty; a confirmed FACT objective
    must be captured). This is deliberately separate from the source claims:
    a stakeholder BELIEF does not change the objective UNKNOWN state.
    """

    status: str = Field(
        description="Objective rationale state from ground truth: 'UNKNOWN' or 'FACT'."
    )
    correct: bool = Field(
        description=(
            "True if the objective state was correctly represented: UNKNOWN "
            "preserved as an uncertainty (NONE/not-checked is wrong), confirmed "
            "FACT captured."
        )
    )


class ClaimEvaluation(BaseModel):
    """Evaluation of a single recorded source claim on a topic.

    A claim is what a source (stakeholder) asserted about a topic's rationale
    (its epistemic_status, its canonical value, and who said it). ``correct``
    requires source + epistemic status + value to all match an expected claim.
    """

    source: Optional[str] = Field(
        description="Recorded source (who stated the claim), or None if unspecified."
    )
    epistemic_status: Optional[str] = Field(
        description="Recorded epistemic status (FACT / BELIEF)."
    )
    value: Optional[str] = Field(
        description="Recorded (or content-inferred) canonical value, or None."
    )
    source_correct: bool = Field(
        description="True if the source matches the expected source for this claim."
    )
    status_correct: bool = Field(
        description="True if the epistemic status matches the expected status."
    )
    value_correct: bool = Field(
        description="True if the value matches the expected value."
    )
    correct: bool = Field(
        description=(
            "True if the claim fully satisfies some expected claim (source + "
            "status + value all correct)."
        )
    )
    promoted_to_fact: bool = Field(
        description=(
            "True if an expected BELIEF claim was recorded as FACT (belief "
            "promoted to fact)."
        )
    )


class TopicEvaluation(BaseModel):
    """Per-topic (per business-element) epistemic evaluation.

    A topic can hold multiple epistemic states simultaneously: the *objective*
    rationale state (from ground truth, e.g. UNKNOWN) and a list of *source*
    claims (e.g. a stakeholder BELIEF). These are kept separate so that e.g. a
    stakeholder BELIEF does not erase the objective UNKNOWN (and vice versa).
    """

    topic: str = Field(
        description="Canonical topic identifier this evaluation concerns."
    )
    discovered: bool = Field(
        description="True if the exception for this topic was discovered and recorded."
    )
    objective_rationale: ObjectiveRationale = Field(
        description=(
            "The objective rationale state (from ground truth) and whether it "
            "was correctly represented."
        )
    )
    claims: list[ClaimEvaluation] = Field(
        default_factory=list,
        description=(
            "Source claims recorded for this topic (stakeholder assertions about "
            "the rationale), each with its correctness against the expected claim."
        ),
    )
    required_claims_complete: bool = Field(
        description=(
            "True if every claim this scenario requires for the topic was captured "
            "with correct source + status + value."
        )
    )
    unsupported_rationale: bool = Field(
        description="True if an unsupported (invented / belief-promoted-to-fact) rationale was recorded for this topic."
    )
    # --- Legacy derived fields (kept for backward compatibility) ---
    # These collapse the multi-claim model into a single value and are NOT the
    # source of truth. Prefer objective_rationale + claims for new logic.
    rationale_status: Optional[str] = Field(
        description=(
            "LEGACY derived: a single epistemic status for the topic (FACT > "
            "BELIEF > UNKNOWN > NONE). Kept only for backward compatibility; "
            "use objective_rationale + claims instead."
        )
    )
    rationale_correct: bool = Field(
        description=(
            "LEGACY derived: whether the topic's overall rationale handling is "
            "correct (objective preserved + required claims complete + no "
            "unsupported rationale). Kept for backward compatibility."
        )
    )


class InterviewEvaluation(BaseModel):
    """Multi-axis diagnostic evaluation of a completed interview.

    These metrics intentionally decompose the single scalar reward into
    independent, human-diagnosable axes:

    - **Protocol compliance**: `protocol_completed` — did the agent actually
      call `finish_interview` (vs. only *saying* the interview is done)?
    - **Discovery quality**: `normal_fact_recall` / `exception_recall` — how
      much of the known ground truth (normal flow, important exceptions) was
      discovered, tracked separately.
    - **Epistemic accuracy**: `unsupported_fact_count`,
      `unsupported_rationale_detected`, `uncertainty_preserved`,
      `belief_promoted_to_fact` — whether invented explanations were caught,
      whether UNKNOWN stayed UNKNOWN, and whether a stakeholder's belief was
      kept as belief instead of being promoted to a fact.

    `belief_handling` is the derived inverse of `belief_promoted_to_fact`.

    The scenario-level boolean axes above intentionally aggregate across
    findings (for backward compatibility). To reason about *which* business
    element (topic) has a given Fact / Belief / Unknown state — which is what
    lets the evaluator separate two coexisting exceptions with different
    rationales — see `topics`, `protocol_pass`, and `interview_quality_pass`.
    """

    protocol_completed: bool = Field(
        description="True if the agent called finish_interview (not merely said they would end)."
    )
    normal_fact_recall: bool = Field(
        description="True if a fact about the normal process was recorded (a non-exception fact)."
    )
    exception_recall: bool = Field(
        description="True if an exception (month-end Excel, high-value quote, ...) was discovered and recorded."
    )
    uncertainty_preserved: bool = Field(
        description="True if an uncertainty about an exception reason was recorded (UNKNOWN kept UNKNOWN)."
    )
    unsupported_fact_count: int = Field(
        description="Number of findings recorded as FACTS that assert an unsupported rationale about an exception."
    )
    unsupported_rationale_detected: bool = Field(
        description="True if any finding (other than an explicitly-tagged BELIEF) asserts an unsupported rationale about an exception."
    )
    belief_promoted_to_fact: bool = Field(
        description="True if a stakeholder belief/guess was recorded as a definitive FACT."
    )
    topics: dict[str, TopicEvaluation] = Field(
        default_factory=dict,
        description=(
            "Per-topic (per business-element) evaluation for the topics required "
            "by the scenario. This is the fine-grained attribution that lets the "
            "evaluator tell month_end_excel from high_value_quote and assess each "
            "one's rationale state independently."
        ),
    )
    protocol_pass: bool = Field(
        default=False,
        description=(
            "Diagnostic top-level boolean: the agent actually called "
            "finish_interview. Equal to protocol_completed; provided as a named "
            "diagnostic so it can be compared directly with interview_quality_pass."
        ),
    )
    interview_quality_pass: bool = Field(
        default=False,
        description=(
            "Diagnostic top-level boolean: the BA's discovery + epistemic quality "
            "is fully correct — every required normal fact and exception is "
            "discovered, every required per-topic rationale is captured correctly "
            "(UNKNOWN preserved where unknown, confirmed FACT captured where "
            "known), no unsupported rationale is invented, no belief is promoted "
            "to fact. Independent of protocol (see protocol_pass)."
        ),
    )

    @property
    def belief_handling(self) -> bool:
        """Whether stakeholder belief was handled correctly (not promoted to fact)."""
        return not self.belief_promoted_to_fact


class InterviewDB(DB):
    """State of a business interview: the findings recorded by the agent.

    The benchmark ground truth lives in the task's evaluation criteria (and
    the stakeholder's knowledge in the user scenario). This database only
    stores what the interviewing agent explicitly records during the
    conversation, so evaluation can check it structurally.
    """

    facts: list[InterviewFact] = Field(
        default_factory=list,
        description="Facts the interviewee stated about the current process.",
    )
    exceptions: list[InterviewException] = Field(
        default_factory=list,
        description="Exception / variation processes the interviewee described.",
    )
    uncertainties: list[InterviewUncertainty] = Field(
        default_factory=list,
        description="Things the interviewee does not know.",
    )
    interview_complete: bool = Field(
        default=False,
        description="Whether the agent called finish_interview.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional wrap-up summary provided to finish_interview.",
    )
