from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import (
    EpistemicStatus,
    InterviewDB,
    InterviewException,
    InterviewFact,
    InterviewUncertainty,
    Topic,
)
from tau2.domains.business_interview.semantic import (
    EXCEPTION_KEYWORDS,
    JAPANESE_KEYWORD_SYNONYMS,
    RATIONALE_SIGNALS,
    RATIONALE_SIGNALS_JP,
    TOPIC_SPECS,
    SemanticEvaluator,
    is_unsupported_rationale,
    keyword_matches,
    matches_all_any,
    topic_of_finding,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# Re-export the canonical signal lists so existing imports (and the README's
# reference to RATIONALE_SIGNALS living in tools.py) keep working.
__all__ = [
    "RATIONALE_SIGNALS",
    "RATIONALE_SIGNALS_JP",
    "EXCEPTION_KEYWORDS",
    "JAPANESE_KEYWORD_SYNONYMS",
]


class InterviewTools(ToolKitBase):
    """Tools for recording findings during a business interview.

    The agent conducts the interview in conversation with the stakeholder and
    uses these tools to keep a structured record. The record is then checked
    deterministically by the env assertions in the task's evaluation criteria,
    and (via the semantic evaluator) reported as multi-axis diagnostics.
    """

    db: InterviewDB

    def __init__(self, db: InterviewDB) -> None:
        super().__init__(db)

    @staticmethod
    def _parse_topic(topic: Optional[str]) -> Optional[Topic]:
        """Parse an optional canonical topic string into a Topic, best-effort."""
        if topic is None:
            return None
        try:
            return Topic(str(topic).strip().lower())
        except ValueError:
            return None

    @is_tool(ToolType.WRITE)
    def record_fact(
        self,
        content: str,
        epistemic_status: str = EpistemicStatus.FACT.value,
        topic: Optional[str] = None,
    ) -> str:
        """
        Record a fact about the current process that the interviewee stated.

        Only record statements the interviewee actually made. Use the
        interviewee's own terms where possible. Do not record your own
        inferences, assumptions, or background knowledge as facts.

        If the interviewee qualified a statement as their own opinion or guess
        (for example "I think it's because ..."), record it with
        epistemic_status="BELIEF" and do NOT promote it to a fact. If they
        said they do not know something, record it with record_uncertainty
        instead of guessing.

        Args:
            content: The fact as stated by the interviewee.
            epistemic_status: "FACT" for statements the interviewee asserted as
                certain, or "BELIEF" for statements they presented as their own
                opinion/guess. Defaults to "FACT".
            topic: Optional canonical topic this finding is about (e.g.
                "month_end_excel", "high_value_quote"). When you can identify
                which business element a finding belongs to, set it so the
                evaluation can attribute the finding correctly. Optional.

        Returns:
            A confirmation message.
        """
        status = EpistemicStatus.FACT
        if epistemic_status is not None:
            try:
                status = EpistemicStatus(str(epistemic_status).upper())
            except ValueError:
                # Unknown values fall back to FACT (best-effort, non-breaking).
                status = EpistemicStatus.FACT
        self.db.facts.append(
            InterviewFact(
                content=content,
                epistemic_status=status,
                topic=self._parse_topic(topic),
            )
        )
        return f"Fact recorded (fact #{len(self.db.facts)})."

    @is_tool(ToolType.WRITE)
    def record_exception(self, content: str, topic: Optional[str] = None) -> str:
        """
        Record an exception or process variation the interviewee described.

        Use this for processes that differ from the normal flow (e.g., only
        used in special circumstances or at particular times). Record how it
        works according to the interviewee, not why you think it exists.

        Args:
            content: The exception process as described by the interviewee.
            topic: Optional canonical topic this exception is about (e.g.
                "month_end_excel", "high_value_quote"). Set it when you can
                identify the business element. Optional.

        Returns:
            A confirmation message.
        """
        self.db.exceptions.append(
            InterviewException(content=content, topic=self._parse_topic(topic))
        )
        return f"Exception recorded (exception #{len(self.db.exceptions)})."

    @is_tool(ToolType.WRITE)
    def record_uncertainty(self, content: str, topic: Optional[str] = None) -> str:
        """
        Record something the interviewee does not know.

        If the interviewee cannot answer a question, record exactly what they
        could not answer here. Never fill the gap with a guessed explanation.

        Args:
            content: What the interviewee does not know.
            topic: Optional canonical topic this uncertainty is about (e.g.
                "month_end_excel"). Set it when you can identify the business
                element. Optional.

        Returns:
            A confirmation message.
        """
        self.db.uncertainties.append(
            InterviewUncertainty(content=content, topic=self._parse_topic(topic))
        )
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
        """Case-insensitive bilingual keyword match (backward-compatible helper)."""
        return matches_all_any(content, all_of, any_of or [])

    @staticmethod
    def _keyword_matches(lowered: str, keyword: str) -> bool:
        """Backward-compatible keyword match helper."""
        return keyword_matches(lowered, keyword)

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
        assertive rationale phrases (see RATIONALE_SIGNALS), scoped to the
        exception topics (either via an explicit canonical ``topic`` on the
        finding, or via the exception content keywords as a fallback).
        Unknown-rationale exception topics (e.g. the month-end Excel hand-off)
        must not have a rationale asserted as a certainty; known-rationale
        topics (e.g. the high-value quote's ``credit risk`` rationale) are
        exempt because asserting their confirmed reason is correct. Purpose
        statements about the normal process (e.g. "reviews the quotation to
        ensure details are correct") are legitimate and are not flagged.

        Findings explicitly recorded as BELIEF (epistemic_status="BELIEF") are
        exempt, and a rationale that is framed as the reason being unknown is
        also exempt (the agent preserved UNKNOWN rather than committing to a
        cause). Promoting a belief to a FACT, or inventing a reason, is flagged.
        """

        # Facts: only flag a rationale when it was recorded as a FACT.
        for fact in self.db.facts:
            if is_unsupported_rationale(
                fact.epistemic_status, fact.content, topic=fact.topic
            ):
                return False
        # Exceptions / uncertainties: never BELIEF, always subject to the rule.
        for exception in self.db.exceptions:
            if is_unsupported_rationale(
                EpistemicStatus.EXCEPTION, exception.content, topic=exception.topic
            ):
                return False
        for uncertainty in self.db.uncertainties:
            if is_unsupported_rationale(
                EpistemicStatus.UNKNOWN, uncertainty.content, topic=uncertainty.topic
            ):
                return False
        if self.db.summary and is_unsupported_rationale(
            EpistemicStatus.FACT, self.db.summary
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Topic-scoped assertion helpers (used by the multi-exception scenario).
    # These make the *canonical topic* the primary ground truth for the reward,
    # rather than surface-language keywords.
    # ------------------------------------------------------------------

    def assert_topic_exception_discovered(self, topic: str) -> bool:
        """True if the exception for the given canonical topic was recorded.

        Matches the semantic evaluator's discovery rule: a finding is
        discovery evidence if it is associated with the topic (explicit
        canonical topic, or content fallback) AND it is an EXCEPTION finding or
        describes the exception via the topic's identity signals. A bare
        rationale fact does not by itself count as discovery.
        """
        t = self._parse_topic(topic)
        spec = TOPIC_SPECS.get(t)
        if spec is None:
            return False
        findings: list[tuple[EpistemicStatus, str, Optional[Topic]]] = []
        for fact in self.db.facts:
            findings.append((fact.epistemic_status, fact.content, fact.topic))
        for exception in self.db.exceptions:
            findings.append(
                (EpistemicStatus.EXCEPTION, exception.content, exception.topic)
            )
        for uncertainty in self.db.uncertainties:
            findings.append(
                (EpistemicStatus.UNKNOWN, uncertainty.content, uncertainty.topic)
            )
        return any(
            topic_of_finding(topic_field, content) == t
            and (
                kind == EpistemicStatus.EXCEPTION
                or any(sig in content.lower() for sig in spec.content_signals)
            )
            for kind, content, topic_field in findings
        )

    def assert_topic_rationale(
        self,
        topic: str,
        expected_status: str,
        rationale_value: Optional[str] = None,
    ) -> bool:
        """True if the rationale recorded for a canonical topic matches the
        expected ground truth.

        ``expected_status`` is one of FACT / BELIEF / UNKNOWN / NONE. When a
        confirmed-rationale topic (FACT) is expected, ``rationale_value`` may
        be given (e.g. "credit_risk") to additionally require that the expected
        rationale was captured. The check is based on the semantic evaluator's
        per-topic state, which is computed over all findings attributed to the
        topic (facts, exceptions and uncertainties), consistent with the
        multi-axis diagnostics.
        """
        ev = SemanticEvaluator.evaluate(self.db)
        topic_eval = ev.topics.get(topic)
        if topic_eval is None:
            return False
        if topic_eval.rationale_status != expected_status:
            return False
        if rationale_value is not None:
            # For a confirmed-rationale topic, rationale_correct already encodes
            # that the expected rationale value was captured (and nothing was
            # improperly downgraded or invented).
            return topic_eval.rationale_correct
        return True

    def assert_interview_complete(self) -> bool:
        """True if the agent called finish_interview."""
        return self.db.interview_complete

    # ------------------------------------------------------------------
    # Multi-axis diagnostics hook (used by the generic evaluator to surface
    # structured metrics into reward_info.info, alongside the scalar reward).
    # ------------------------------------------------------------------

    def get_eval_diagnostics(self, task: Optional[Task] = None) -> Optional[dict]:
        """Return the multi-axis diagnostic metrics as a JSON-serializable dict.

        ``task`` is optional: when supplied, the evaluator knows which topics
        this scenario requires and can report per-topic discovery / epistemic
        state. Without it, topics are inferred from the recorded findings.

        This is consumed by the generic EnvironmentEvaluator extension point:
        the result is attached to the run's additional-info so humans can see
        protocol / discovery / epistemic outcomes separately instead of only a
        single scalar reward.
        """
        scenario_id = task.id if task is not None else None
        evaluation = SemanticEvaluator.evaluate(self.db, scenario_id=scenario_id)
        payload = evaluation.model_dump(mode="json")
        # Expose the derived belief_handling flag too.
        payload["belief_handling"] = evaluation.belief_handling
        return payload
