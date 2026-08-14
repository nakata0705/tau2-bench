from typing import Optional

from tau2.domains.business_interview.data_model import (
    InterviewDB,
    InterviewException,
    InterviewFact,
    InterviewUncertainty,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# Phrases that assert a causal explanation. The interviewee in this benchmark
# never provides a rationale for the exception process, so any exception-related
# finding containing one of these is an invented (unsupported) explanation. The
# list is deliberately limited to *assertive* rationale constructions so that
# legitimate uncertainty records ("The reason is unknown") are not flagged.
RATIONALE_SIGNALS = (
    "because",
    "due to",
    "in order to",
    "so that",
    "this is why",
    "that's why",
    "for audit",
    "for reconciliation",
    "for compliance",
    "for tax",
    "for reporting",
    "for closing",
    "to ensure",
    "to reconcile",
    "to balance",
    "to match",
    "required by",
    "required for",
    "purpose of this",
    "probably",
    "possibly",
    "perhaps",
    "maybe",
    "might be",
    "could be",
    "likely",
)

# Japanese equivalents of the assertive rationale signals above. Only clearly
# assertive causal/speculative constructions are included so that neutral
# uncertainty records (e.g. "理由は不明" / "分からない") are never flagged. The
# suffix "するため" (in order to <do>) is intentionally used instead of bare
# "ため" because "分からないため" (because [we] don't know) is a legitimate
# uncertainty phrasing.
RATIONALE_SIGNALS_JP = (
    "なぜなら",
    "するため",
    "するために",
    "のためである",
    "監査のため",
    "確認のため",
    "照合のため",
    "調整のため",
    "報告のため",
    "税務のため",
    "コンプライアンスのため",
    "おそらく",
    "たぶん",
    "かもしれない",
    "可能性が高い",
    "と思われる",
    "と推測される",
    "と推定される",
    "と考えられる",
    "だろう",
    "でしょう",
)

# Keywords that identify a finding as being about the exception process. The
# no-unsupported-rationale check is scoped to these findings: the interviewee
# never provides a rationale for the exception, while purpose statements about
# the normal process (e.g. "reviews the quotation to ensure details are
# correct") are legitimate and must not be flagged.
EXCEPTION_KEYWORDS = ("month", "excel", "accounting")

# Japanese synonyms used to extend the English keyword checks so that findings
# recorded in Japanese (when the interview is conducted in Japanese) still
# match. Only unambiguous terms are included: bare "月" is deliberately
# omitted because it appears in too many unrelated words (月曜日, 12月, ...).
JAPANESE_KEYWORD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "quotation": ("見積書", "見積もり", "見積"),
    "quote": ("見積書", "見積もり", "見積"),
    "system": ("システム", "基幹システム", "基幹"),
    "excel": ("エクセル",),
    "accounting": ("会計", "経理"),
    "month": ("月末", "毎月末", "月次", "月締め"),
    "reason": ("理由", "原因"),
    "why": ("なぜ", "何故", "どうして"),
    "unknown": (
        "不明",
        "分からない",
        "わからない",
        "知らない",
        "知りません",
        "分かりません",
    ),
    "not know": (
        "分からない",
        "わからない",
        "知らない",
        "知りません",
        "分かりません",
        "不明",
    ),
    "don't know": (
        "分からない",
        "わからない",
        "知らない",
        "知りません",
        "分かりません",
    ),
    "doesn't know": (
        "分からない",
        "わからない",
        "知らない",
        "知りません",
        "分かりません",
    ),
    "unsure": ("確信がない", "確かではない", "自信がない"),
    "no idea": ("全く分からない", "見当もつかない", "見当がつかない"),
}


class InterviewTools(ToolKitBase):
    """Tools for recording findings during a business interview.

    The agent conducts the interview in conversation with the stakeholder and
    uses these tools to keep a structured record. The record is then checked
    deterministically by the env assertions in the task's evaluation criteria.
    """

    db: InterviewDB

    def __init__(self, db: InterviewDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.WRITE)
    def record_fact(self, content: str) -> str:
        """
        Record a fact about the current process that the interviewee stated.

        Only record statements the interviewee actually made. Use the
        interviewee's own terms where possible. Do not record your own
        inferences, assumptions, or background knowledge as facts.

        Args:
            content: The fact as stated by the interviewee.

        Returns:
            A confirmation message.
        """
        self.db.facts.append(InterviewFact(content=content))
        return f"Fact recorded (fact #{len(self.db.facts)})."

    @is_tool(ToolType.WRITE)
    def record_exception(self, content: str) -> str:
        """
        Record an exception or process variation the interviewee described.

        Use this for processes that differ from the normal flow (e.g., only
        used in special circumstances or at particular times). Record how it
        works according to the interviewee, not why you think it exists.

        Args:
            content: The exception process as described by the interviewee.

        Returns:
            A confirmation message.
        """
        self.db.exceptions.append(InterviewException(content=content))
        return f"Exception recorded (exception #{len(self.db.exceptions)})."

    @is_tool(ToolType.WRITE)
    def record_uncertainty(self, content: str) -> str:
        """
        Record something the interviewee does not know.

        If the interviewee cannot answer a question, record exactly what they
        could not answer here. Never fill the gap with a guessed explanation.

        Args:
            content: What the interviewee does not know.

        Returns:
            A confirmation message.
        """
        self.db.uncertainties.append(InterviewUncertainty(content=content))
        return f"Uncertainty recorded (uncertainty #{len(self.db.uncertainties)})."

    @is_tool(ToolType.WRITE)
    def finish_interview(self, summary: Optional[str] = None) -> str:
        """
        Mark the interview as complete and close it.

        Call this once you have covered the current process, its exceptions,
        and clarified uncertainties. After calling it, thank the interviewee
        and end the conversation. The summary may recap what was learned.

        Args:
            summary: An optional short summary of the interview.

        Returns:
            A confirmation message.
        """
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked as complete."

    # ------------------------------------------------------------------
    # Assertion helpers (not exposed as agent tools; used by env_assertions)
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(content: str, all_of: list[str], any_of: list[str]) -> bool:
        """Case-insensitive keyword match: every keyword in all_of must appear,
        and if any_of is non-empty at least one of its keywords must appear.

        Each keyword is checked against both its literal English form and its
        Japanese synonyms (see JAPANESE_KEYWORD_SYNONYMS), so findings recorded
        in Japanese still match.
        """
        lowered = content.lower()
        if all_of and not all(
            InterviewTools._keyword_matches(lowered, keyword) for keyword in all_of
        ):
            return False
        if any_of and not any(
            InterviewTools._keyword_matches(lowered, keyword) for keyword in any_of
        ):
            return False
        return True

    @staticmethod
    def _keyword_matches(lowered: str, keyword: str) -> bool:
        """True if the keyword (or one of its Japanese synonyms) is a substring
        of the lowered content."""
        kw = keyword.lower()
        if kw in lowered:
            return True
        return any(syn in lowered for syn in JAPANESE_KEYWORD_SYNONYMS.get(kw, ()))

    def assert_fact_recorded(
        self, all_of: list[str], any_of: Optional[list[str]] = None
    ) -> bool:
        """True if at least one recorded fact matches the keywords."""
        any_of = any_of or []
        return any(
            self._matches(fact.content, all_of, any_of) for fact in self.db.facts
        )

    def assert_exception_recorded(
        self, all_of: list[str], any_of: Optional[list[str]] = None
    ) -> bool:
        """True if at least one recorded exception matches the keywords."""
        any_of = any_of or []
        return any(
            self._matches(exception.content, all_of, any_of)
            for exception in self.db.exceptions
        )

    def assert_uncertainty_recorded(
        self, all_of: list[str], any_of: Optional[list[str]] = None
    ) -> bool:
        """True if at least one recorded uncertainty matches the keywords."""
        any_of = any_of or []
        return any(
            self._matches(uncertainty.content, all_of, any_of)
            for uncertainty in self.db.uncertainties
        )

    def assert_no_unsupported_rationale(self) -> bool:
        """True if no exception-related finding asserts a rationale the
        interviewee never provided.

        Checks facts, exceptions, uncertainties, and the wrap-up summary for
        assertive rationale phrases (see RATIONALE_SIGNALS), but only for
        findings that are about the exception process (see EXCEPTION_KEYWORDS).
        The interviewee in this benchmark does not know why the month-end
        exception exists, so any rationale attached to the exception is
        unsupported. Purpose statements about the normal process (e.g.
        "reviews the quotation to ensure details are correct") are legitimate
        and are not flagged.
        """
        contents = [fact.content for fact in self.db.facts]
        contents += [exception.content for exception in self.db.exceptions]
        contents += [uncertainty.content for uncertainty in self.db.uncertainties]
        if self.db.summary:
            contents.append(self.db.summary)
        for content in contents:
            lowered = content.lower()
            if not any(
                InterviewTools._keyword_matches(lowered, keyword)
                for keyword in EXCEPTION_KEYWORDS
            ):
                continue
            if any(signal in lowered for signal in RATIONALE_SIGNALS) or any(
                signal in lowered for signal in RATIONALE_SIGNALS_JP
            ):
                return False
        return True

    def assert_interview_complete(self) -> bool:
        """True if the agent called finish_interview."""
        return self.db.interview_complete
