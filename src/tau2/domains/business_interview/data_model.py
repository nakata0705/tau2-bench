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


class EpistemicStatus(str, Enum):
    """The epistemic status of a recorded finding.

    Used to distinguish what is objectively true (FACT), what the stakeholder
    believes but does not know for certain (BELIEF), what is genuinely unknown
    (UNKNOWN / uncertainty), and what is a process exception (EXCEPTION).

    This is the canonical, language-agnostic state that the semantic evaluator
    reasons about. Recording `epistemic_status` lets a well-behaved agent
    separate belief from fact without relying on surface-language keywords.
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


class TopicEvaluation(BaseModel):
    """Per-topic (per business-element) epistemic evaluation.

    This is the *fine-grained* counterpart to the scenario-level metrics: it
    attributes discovery and the Fact / Belief / Unknown state to a specific
    canonical topic, so that e.g. the month-end-Excel reason being UNKNOWN is
    not conflated with the high-value-quote reason being FACT.
    """

    topic: str = Field(
        description="Canonical topic identifier this evaluation concerns."
    )
    discovered: bool = Field(
        description="True if the exception for this topic was discovered and recorded."
    )
    rationale_status: Optional[str] = Field(
        description=(
            "Epistemic status of the rationale recorded for this topic: "
            "'FACT', 'BELIEF', 'UNKNOWN' (preserved uncertainty), or 'NONE' "
            "(no rationale recorded)."
        )
    )
    rationale_correct: bool = Field(
        description=(
            "True if the recorded rationale matches the ground truth for this "
            "topic: UNKNOWN/NONE preserved for an unknown-rationale topic, and "
            "the confirmed FACT (with its expected value) captured for a "
            "known-rationale topic."
        )
    )
    unsupported_rationale: bool = Field(
        description="True if an unsupported (invented / belief-promoted-to-fact) rationale was recorded for this topic."
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
