from typing import Optional

from pydantic import BaseModel, Field

from tau2.environment.db import DB


class InterviewFact(BaseModel):
    """A fact about the current process that the interviewee stated."""

    content: str = Field(
        description="The fact as stated by the interviewee (paraphrase OK)."
    )


class InterviewException(BaseModel):
    """An exception / process variation the interviewee described."""

    content: str = Field(
        description="The exception process as described by the interviewee."
    )


class InterviewUncertainty(BaseModel):
    """Something the interviewee does not know."""

    content: str = Field(
        description="What the interviewee could not answer (recorded as uncertainty, never guessed)."
    )


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
