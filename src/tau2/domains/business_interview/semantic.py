"""Language-agnostic semantic evaluation for business_interview.

This module centralizes the *canonical* concept classification used by both the
English and Japanese evaluation paths. Instead of scattering per-keyword logic,
it reasons about a small set of canonical concepts (normal-process fact,
exception, rationale, uncertainty) and produces a structured multi-axis
diagnostic (`InterviewEvaluation`).

The signal lists below are deliberately small and bilingual (English + Japanese)
so that a finding with the same *meaning* is evaluated identically regardless of
surface language. This is the single shared semantic path; the keyword-synonym
fallbacks that predate this module remain available via
:func:`keyword_matches` for backward compatibility of the existing env
assertions.
"""

from typing import Optional

from tau2.domains.business_interview.data_model import (
    EpistemicStatus,
    InterviewDB,
    InterviewEvaluation,
    Topic,
    TopicEvaluation,
)

# ---------------------------------------------------------------------------
# Canonical signal lists (bilingual). Keep small; extend only with unambiguous,
# widely-used expressions. Never build an unbounded synonym dictionary.
# ---------------------------------------------------------------------------

# Phrases that assert a causal explanation. In this benchmark the interviewee
# never provides a rationale for the exception process, so any exception-related
# finding containing one of these is an invented (unsupported) explanation.
# The list is deliberately limited to *assertive* rationale constructions so
# that legitimate uncertainty records ("The reason is unknown") are not flagged.
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

# Additional invented rationale concepts for the *known-rationale* topic check.
# A known-rationale topic (e.g. the high-value-quote ``credit risk`` reason) is
# only correct when the expected rationale value is captured AND no *other*
# rationale is asserted. This is a deliberately small, bounded set of common
# business reasons the benchmark stakeholder never states; it is NOT an
# unbounded forbidden-word dictionary. It is only consulted for known-rationale
# topics, so legitimate exception discovery text never trips it.
EXTRA_RATIONALE_SIGNALS = (
    # English
    "audit",
    "reconciliation",
    "reconcile",
    "tax",
    "compliance",
    "comply",
    "reporting",
    "closing",
    "balance",
    "regulation",
    "regulatory",
    "legal",
    # Japanese
    "監査",
    "照合",
    "税務",
    "コンプライアンス",
    "法令",
    "報告",
    "決算",
    "規制",
    "法令遵守",
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

# Common Japanese expressions for "does not know / has not grasped" that the
# basic 分からない/知らない/不明 set does not cover. Kept to a small curated set
# of unambiguous epistemic markers (they all negate knowing), so that an
# uncertainty record like 「正確な理由は把握していない」 is recognised regardless
# of surface wording. These are added to the "not know"-family synonym tuples
# below so both the keyword assertions and the semantic evaluator accept them.
_JAPANESE_NOT_KNOW = (
    "分からない",
    "わからない",
    "知らない",
    "知りません",
    "分かりません",
    "不明",
    "把握していない",
    "把握しておらず",
    "把握できていない",
    "把握しておりません",
)
for _key in ("unknown", "not know", "don't know", "doesn't know"):
    JAPANESE_KEYWORD_SYNONYMS[_key] = _JAPANESE_NOT_KNOW

# Signals that a finding expresses genuine uncertainty (the reason is unknown).
# Used to decide whether UNKNOWN was *preserved* as UNKNOWN.
UNCERTAINTY_SIGNALS = (
    "don't know",
    "do not know",
    "doesn't know",
    "does not know",
    "unknown",
    "unclear",
    "not sure",
    "not certain",
    "unsure",
    "no idea",
    "cannot say",
    "can't say",
)

UNCERTAINTY_SIGNALS_JP = (
    "不明",
    "分からない",
    "わからない",
    "分かりません",
    "知らない",
    "知りません",
    "確信がない",
    "確かではない",
    "はっきりしない",
    "見当もつかない",
    "見当がつかない",
    "把握していない",
    "把握しておらず",
    "把握できていない",
    "把握しておりません",
)


# ---------------------------------------------------------------------------
# Bilingual keyword matching (shared by the existing env assertions and the
# semantic evaluator).
# ---------------------------------------------------------------------------


def keyword_matches(lowered: str, keyword: str) -> bool:
    """True if the keyword (or one of its Japanese synonyms) is a substring of
    the lowered content."""
    kw = keyword.lower()
    if kw in lowered:
        return True
    return any(syn in lowered for syn in JAPANESE_KEYWORD_SYNONYMS.get(kw, ()))


def matches_all_any(
    content: str, all_of: list[str], any_of: Optional[list[str]] = None
) -> bool:
    """Case-insensitive keyword match: every keyword in all_of must appear, and
    if any_of is non-empty at least one of its keywords must appear. Each
    keyword is checked against both its literal English form and its Japanese
    synonyms, so findings recorded in Japanese still match."""
    lowered = content.lower()
    if all_of and not all(keyword_matches(lowered, keyword) for keyword in all_of):
        return False
    if any_of and not any(keyword_matches(lowered, keyword) for keyword in any_of):
        return False
    return True


# ---------------------------------------------------------------------------
# Canonical topic model.
#
# A topic is a *business element* (e.g. the month-end Excel hand-off, the
# high-value quote check) that a finding can be about. Each topic carries its
# expected ground-truth rationale state:
#
# - ``month_end_excel``  -> the reason is UNKNOWN (must be preserved, never
#   invented)
# - ``high_value_quote`` -> the reason is a confirmed FACT ("credit risk")
#   (must be captured, never left UNKNOWN)
#
# The topic *identity* is canonical and language-independent. ``content_signals``
# are only a backward-compatible fallback used when a finding does not carry an
# explicit ``topic`` field; they are deliberately small, not an unbounded synonym
# dictionary. ``rationale_value_signals`` detect the expected confirmed rationale
# value for known-rationale topics.
# ---------------------------------------------------------------------------


class TopicSpec:
    def __init__(
        self,
        topic: Topic,
        expected_rationale_status: EpistemicStatus,
        content_signals: tuple[str, ...],
        rationale_value_signals: tuple[str, ...] = (),
    ) -> None:
        self.topic = topic
        self.expected_rationale_status = expected_rationale_status
        self.content_signals = content_signals
        self.rationale_value_signals = rationale_value_signals


TOPIC_SPECS: dict[Topic, TopicSpec] = {
    Topic.MONTH_END_EXCEL: TopicSpec(
        topic=Topic.MONTH_END_EXCEL,
        expected_rationale_status=EpistemicStatus.UNKNOWN,
        content_signals=(
            "month",
            "month-end",
            "excel",
            "accounting",
            "月末",
            "毎月末",
            "月次",
            "月締め",
            "エクセル",
            "経理",
            "会計",
        ),
    ),
    Topic.HIGH_VALUE_QUOTE: TopicSpec(
        topic=Topic.HIGH_VALUE_QUOTE,
        expected_rationale_status=EpistemicStatus.FACT,
        content_signals=(
            "million",
            "one million",
            "1,000,000",
            "1000000",
            "high-value",
            "high value",
            "large amount",
            "large quotation",
            "100万",
            "百万円",
            "高額",
            "大口",
            "金額が大きい",
        ),
        rationale_value_signals=("credit", "与信", "クレジット"),
    ),
}

# Which topics a scenario *requires* to be discovered and correctly reasoned
# about. This is the scenario's ground truth about what the interview must
# cover; it lives here (domain-side) rather than in the agent-visible scenario
# or policy so it is never leaked to the interviewing agent or the stakeholder.
SCENARIO_TOPICS: dict[str, tuple[Topic, ...]] = {
    "quotation_process_interview_1": (Topic.MONTH_END_EXCEL,),
    "quotation_belief_uncertainty_1": (Topic.MONTH_END_EXCEL,),
    "quotation_multi_exception_1": (
        Topic.MONTH_END_EXCEL,
        Topic.HIGH_VALUE_QUOTE,
    ),
}

# Suffix that marks a Japanese (pre-localized) scenario variant of an English
# scenario id. The EN and JA variants are the SAME canonical scenario with the
# SAME ground truth, so the evaluator canonicalizes the id before looking up its
# required topics. This is a minimal, domain-local canonicalization (no generic
# i18n framework).
JA_SCENARIO_SUFFIX = "_ja"


def canonical_scenario_id(scenario_id: Optional[str]) -> Optional[str]:
    """Map a Japanese scenario variant to its canonical (English) scenario id.

    EN/JA variants share one canonical scenario ground truth. This is the only
    place the language suffix is understood; it is intentionally minimal and
    domain-local so that e.g. ``quotation_multi_exception_1_ja`` resolves to the
    same required topics as ``quotation_multi_exception_1`` (and a missed topic
    stays required, never disappearing from the diagnostics).
    """
    if scenario_id is None:
        return None
    if scenario_id.endswith(JA_SCENARIO_SUFFIX):
        return scenario_id[: -len(JA_SCENARIO_SUFFIX)]
    return scenario_id


def _content_inferred_topics(content: str) -> list[Topic]:
    """The topics whose identity signals are unambiguous in ``content``."""
    lowered = content.lower()
    return [
        spec.topic
        for spec in TOPIC_SPECS.values()
        if any(sig in lowered for sig in spec.content_signals)
    ]


def topic_of_finding(topic: Optional[Topic], content: str) -> Optional[Topic]:
    """Resolve the canonical topic a finding is about.

    An explicit canonical ``topic`` field is the preferred identity, but it is
    only trusted when the finding's content does not unambiguously point at a
    *different* topic. If the content clearly identifies another topic, the
    reported topic is not reliable (the agent may be misattributing), so the
    finding is left unassociated (None) rather than credited to either topic.
    If the content matches no topic (or matches more than one, i.e. it is
    ambiguous), the reported topic is trusted and no guess is made from the
    content. When no explicit topic is given, the content is matched against
    each topic's bilingual identity signals as a backward-compatible fallback
    (single, unambiguous match only).
    """
    inferred = _content_inferred_topics(content)
    if topic is not None:
        if len(inferred) == 1 and inferred[0] != topic:
            # Content clearly indicates a different topic than the reported one:
            # do not unconditionally trust the reported topic (topic-attribution
            # robustness) and do not reward a misattribution to the content
            # topic either. Leave it unassociated so it satisfies neither.
            return None
        return topic
    if len(inferred) == 1:
        return inferred[0]
    return None


def is_exception_related(content: str, topic: Optional[Topic] = None) -> bool:
    """True if a finding concerns an exception process.

    A finding is exception-related if it is attributed to a canonical exception
    topic (via ``topic_of_finding``, which already cross-checks reported vs
    content topic) or, as a backward-compatible fallback, if its content matches
    the legacy month/excel/accounting exception keywords. This is broader than
    ``is_exception_content`` so that high-value-quote findings (whose surface
    text does not mention month/Excel/Accounting) are also treated as exceptions
    for the scenario-level epistemic checks.
    """
    return topic_of_finding(topic, content) is not None or is_exception_content(content)


def asserts_extra_rationale(content: str) -> bool:
    """True if the content asserts a rationale other than a topic's canonical
    value (see :data:`EXTRA_RATIONALE_SIGNALS`). Used to catch an unsupported
    *additional* rationale on a known-rationale topic (e.g. ``credit risk`` plus
    an invented ``tax reporting`` / ``audit reconciliation`` reason)."""
    lowered = content.lower()
    return any(sig in lowered for sig in EXTRA_RATIONALE_SIGNALS)


def topic_spec(topic: Optional[Topic]) -> Optional[TopicSpec]:
    """The spec for a topic, or None if it is not a known exception topic."""
    if topic is None:
        return None
    return TOPIC_SPECS.get(topic)


# ---------------------------------------------------------------------------
# Canonical concept classification
# ---------------------------------------------------------------------------


def is_exception_content(content: str) -> bool:
    """True if the finding concerns the exception process (month/excel/accounting)."""
    lowered = content.lower()
    return any(keyword_matches(lowered, keyword) for keyword in EXCEPTION_KEYWORDS)


def asserts_rationale(content: str) -> bool:
    """True if the content asserts a causal / speculative rationale."""
    lowered = content.lower()
    return any(signal in lowered for signal in RATIONALE_SIGNALS) or any(
        signal in lowered for signal in RATIONALE_SIGNALS_JP
    )


def expresses_uncertainty(content: str) -> bool:
    """True if the content expresses that something is not known."""
    lowered = content.lower()
    return any(signal in lowered for signal in UNCERTAINTY_SIGNALS) or any(
        signal in lowered for signal in UNCERTAINTY_SIGNALS_JP
    )


def _rationale_value_signals() -> tuple[str, ...]:
    """All rationale-value signals across every known topic.

    Used to detect when a finding asserts a specific reason *value* (e.g.
    ``credit`` / ``与信`` for the high-value-quote rationale) even when it does
    not use an explicit causal connective. Combined with the canonical topic,
    this lets the evaluator catch a wrongly-misattributed rationale (e.g. the
    month-end-Excel reason asserted as ``credit risk``).
    """
    return tuple(
        sig for spec in TOPIC_SPECS.values() for sig in spec.rationale_value_signals
    )


def asserts_rationale_or_value(content: str) -> bool:
    """True if the content asserts a rationale via a causal connective
    (``asserts_rationale``) or by naming a known rationale value."""
    lowered = content.lower()
    if any(sig in lowered for sig in _rationale_value_signals()):
        return True
    return asserts_rationale(content)


def is_unsupported_rationale(
    kind: EpistemicStatus, content: str, topic: Optional[Topic] = None
) -> bool:
    """True if a finding commits to a rationale about an exception whose reason
    is not known, and the finding asserts that reason as a certainty.

    This is the single, language-agnostic rule for the no-unsupported-rationale
    check. A rationale is *not* unsupported when:

    - it was recorded as a BELIEF (the stakeholder's own opinion, kept as
      opinion), or
    - the finding itself frames the reason as unknown/uncertain (the agent
      preserved UNKNOWN instead of committing to a cause), or
    - it concerns a topic whose reason IS known/confirmed (e.g. the
      high-value-quote ``credit risk`` rationale), so asserting it is correct.

    A rationale *is* unsupported when the agent asserts a causal explanation
    for an unknown-rationale exception as a certainty (e.g. "for audit",
    "because of ...") without keeping the reason unknown — that is either
    fabrication or promotion of a belief to fact.

    ``topic`` is the optional canonical topic of the finding. When provided it
    is the primary, language-independent signal: any assertive rationale on an
    unknown-rationale topic is unsupported regardless of surface wording. When
    omitted, the backward-compatible content-based rule (exception keywords +
    rationale signals) applies.
    """
    if kind == EpistemicStatus.BELIEF:
        return False
    if topic is not None:
        spec = topic_spec(topic)
        if spec is None:
            return False
        if spec.expected_rationale_status == EpistemicStatus.FACT:
            # Known-rationale topic: asserting its confirmed reason is supported,
            # but adding an unsupported *additional* rationale (a reason other
            # than the canonical value) is not. Kept to a small curated signal
            # set so e.g. "credit risk and tax reporting" is caught without a
            # huge forbidden-word dictionary.
            if asserts_rationale_or_value(content) and asserts_extra_rationale(content):
                return True
            return False
        if not asserts_rationale_or_value(content):
            return False
        if expresses_uncertainty(content):
            return False
        return True
    if not is_exception_content(content):
        return False
    if not asserts_rationale(content):
        return False
    if expresses_uncertainty(content):
        return False
    return True


# ---------------------------------------------------------------------------
# Semantic evaluator
# ---------------------------------------------------------------------------


class SemanticEvaluator:
    """Computes the multi-axis diagnostic for an interview from its DB state.

    The evaluator is deliberately *structural* and language-agnostic: it reads
    the canonical findings (with their `epistemic_status`) and classifies each
    against the bilingual concept signals above. English and Japanese findings
    that mean the same thing are evaluated the same way.
    """

    @classmethod
    def evaluate(
        cls, db: InterviewDB, scenario_id: Optional[str] = None
    ) -> InterviewEvaluation:
        # Canonical findings: (kind, content, topic). For facts the agent's
        # recorded epistemic_status decides FACT vs BELIEF. Each finding is
        # associated with a canonical topic (explicit `topic` field preferred,
        # content-signal fallback otherwise).
        canonical: list[tuple[EpistemicStatus, str, Optional[Topic]]] = []
        for fact in db.facts:
            canonical.append(
                (
                    fact.epistemic_status,
                    fact.content,
                    topic_of_finding(fact.topic, fact.content),
                )
            )
        for exception in db.exceptions:
            canonical.append(
                (
                    EpistemicStatus.EXCEPTION,
                    exception.content,
                    topic_of_finding(exception.topic, exception.content),
                )
            )
        for uncertainty in db.uncertainties:
            canonical.append(
                (
                    EpistemicStatus.UNKNOWN,
                    uncertainty.content,
                    topic_of_finding(uncertainty.topic, uncertainty.content),
                )
            )

        # ---- Scenario-level discovery / epistemic (backward compatible) ----
        # A finding that is attributed to a canonical *exception* topic (or that
        # matches the legacy exception keywords) is exception-related. It must
        # never count as a normal-process fact, and it is what the epistemic
        # (no-unsupported-rationale / uncertainty) checks scan. The resolved
        # topic already cross-checks reported vs content so a misattributed
        # finding is not counted as a normal fact either.
        normal_fact_recall = any(
            kind == EpistemicStatus.FACT and not is_exception_related(content, topic)
            for kind, content, topic in canonical
        )
        exception_recall = any(
            is_exception_related(content, topic) for _, content, topic in canonical
        )

        unsupported_fact_count = 0
        unsupported_rationale_detected = False
        belief_promoted_to_fact = False
        uncertainty_preserved = False

        for kind, content, topic in canonical:
            if not is_exception_related(content, topic):
                continue
            if kind == EpistemicStatus.UNKNOWN and expresses_uncertainty(content):
                # The unknown reason was preserved as an uncertainty.
                uncertainty_preserved = True
            if not is_unsupported_rationale(kind, content, topic):
                continue
            unsupported_rationale_detected = True
            if kind == EpistemicStatus.FACT:
                unsupported_fact_count += 1
                belief_promoted_to_fact = True

        # The wrap-up summary is not a canonical finding; it must not fabricate
        # a rationale either. It is never BELIEF, so the same certainty/unknown
        # rule applies (FACT is used only as a non-BELIEF sentinel here).
        if db.summary and is_unsupported_rationale(EpistemicStatus.FACT, db.summary):
            unsupported_rationale_detected = True

        # ---- Per-topic evaluation (the fine-grained attribution) -----------
        topics = cls._evaluate_topics(canonical, scenario_id)

        # ---- Diagnostic top-level booleans --------------------------------
        # protocol_pass: did the agent actually close the interview.
        protocol_pass = db.interview_complete
        # interview_quality_pass: full discovery + epistemic quality, judged
        # independently of protocol. Every required topic must be discovered
        # with a correct rationale; nothing may be invented or belief-promoted.
        topics_ok = bool(topics) and all(
            t.discovered and t.rationale_correct for t in topics.values()
        )
        interview_quality_pass = (
            normal_fact_recall
            and topics_ok
            and unsupported_fact_count == 0
            and not unsupported_rationale_detected
            and not belief_promoted_to_fact
        )

        return InterviewEvaluation(
            protocol_completed=db.interview_complete,
            normal_fact_recall=normal_fact_recall,
            exception_recall=exception_recall,
            uncertainty_preserved=uncertainty_preserved,
            unsupported_fact_count=unsupported_fact_count,
            unsupported_rationale_detected=unsupported_rationale_detected,
            belief_promoted_to_fact=belief_promoted_to_fact,
            topics=topics,
            protocol_pass=protocol_pass,
            interview_quality_pass=interview_quality_pass,
        )

    @classmethod
    def _evaluate_topics(
        cls,
        canonical: list[tuple[EpistemicStatus, str, Optional[Topic]]],
        scenario_id: Optional[str],
    ) -> dict[str, TopicEvaluation]:
        """Evaluate each topic required by the scenario (or inferred from the
        findings when no scenario id is supplied)."""
        required = list(SCENARIO_TOPICS.get(canonical_scenario_id(scenario_id), ()))
        if not required:
            inferred = {topic for _, _, topic in canonical if topic is not None}
            # Keep a deterministic order consistent with TOPIC_SPECS.
            required = [
                spec.topic for spec in TOPIC_SPECS.values() if spec.topic in inferred
            ]

        evaluations: dict[str, TopicEvaluation] = {}
        for topic in required:
            spec = TOPIC_SPECS.get(topic)
            if spec is None:
                continue
            topic_findings = [item for item in canonical if item[2] == spec.topic]
            evaluations[spec.topic.value] = cls._evaluate_topic(topic_findings, spec)
        return evaluations

    @classmethod
    def _evaluate_topic(
        cls,
        findings: list[tuple[EpistemicStatus, str, Optional[Topic]]],
        spec: TopicSpec,
    ) -> TopicEvaluation:
        """Evaluate a single topic from the findings attributed to it.

        - ``discovered``: the exception for this topic was recorded (an
          EXCEPTION finding, or any finding that describes the exception via the
          topic's identity signals).
        - ``rationale_status``: FACT if a rationale was asserted as a certainty,
          BELIEF if asserted as the stakeholder's opinion, UNKNOWN if the reason
          was preserved as an uncertainty, else NONE.
        - ``rationale_correct``: matches the topic's ground truth — for an
          unknown-rationale topic the reason must not be asserted (UNKNOWN/NONE
          preserved and nothing invented); for a known-rationale topic the
          confirmed FACT with its expected value must be captured.
        - ``unsupported_rationale``: any asserted rationale on this topic that
          the topic-aware rule flags as unsupported.
        """
        topic_id = spec.topic
        discovered = any(
            kind == EpistemicStatus.EXCEPTION
            or any(sig in content.lower() for sig in spec.content_signals)
            for kind, content, _ in findings
        )

        has_fact_rationale = any(
            kind in (EpistemicStatus.FACT, EpistemicStatus.EXCEPTION)
            and asserts_rationale_or_value(content)
            for kind, content, _ in findings
        )
        has_belief_rationale = any(
            kind == EpistemicStatus.BELIEF and asserts_rationale_or_value(content)
            for kind, content, _ in findings
        )
        has_unknown = any(
            kind == EpistemicStatus.UNKNOWN and expresses_uncertainty(content)
            for kind, content, _ in findings
        )
        unsupported = any(
            is_unsupported_rationale(kind, content, topic=spec.topic)
            for kind, content, _ in findings
        )

        if has_fact_rationale:
            rationale_status = EpistemicStatus.FACT.value
        elif has_belief_rationale:
            rationale_status = EpistemicStatus.BELIEF.value
        elif has_unknown:
            rationale_status = EpistemicStatus.UNKNOWN.value
        else:
            rationale_status = None

        if spec.expected_rationale_status == EpistemicStatus.UNKNOWN:
            # Unknown-rationale topic: correct iff the reason was explicitly
            # investigated and preserved as UNKNOWN, and nothing invented. A
            # finding that leaves the rationale unrecorded (NONE) is NOT correct:
            # NONE means the agent did not check, UNKNOWN means they checked and
            # it is unknown.
            rationale_correct = (
                rationale_status == EpistemicStatus.UNKNOWN.value and not unsupported
            )
        else:
            # Known-rationale (FACT) topic: correct iff a FACT rationale that
            # carries the expected rationale value was captured, nothing was
            # improperly downgraded to UNKNOWN, and no unsupported *additional*
            # rationale was invented (precision: expected rationale captured AND
            # unsupported extra absent).
            captured_value = any(
                kind in (EpistemicStatus.FACT, EpistemicStatus.EXCEPTION)
                and asserts_rationale_or_value(content)
                and any(sig in content.lower() for sig in spec.rationale_value_signals)
                for kind, content, _ in findings
            )
            rationale_correct = (
                rationale_status == EpistemicStatus.FACT.value
                and captured_value
                and not unsupported
            )

        return TopicEvaluation(
            topic=topic_id.value,
            discovered=discovered,
            rationale_status=rationale_status,
            rationale_correct=rationale_correct,
            unsupported_rationale=unsupported,
        )

    @classmethod
    def critical_pass(cls, db: InterviewDB) -> bool:
        """Whether every critical axis is satisfied (scenario-level).

        This is exactly the conjunction of the existing reward env assertions,
        so adding it never changes the reward of an already-correct interview.
        It is provided as a convenience diagnostic. Note that, like the scalar
        reward, it aggregates across findings and therefore cannot attribute a
        failure to a specific topic — use :attr:`InterviewEvaluation.topics`
        and :attr:`InterviewEvaluation.interview_quality_pass` for that.
        """
        ev = cls.evaluate(db)
        return (
            ev.protocol_completed
            and ev.normal_fact_recall
            and ev.exception_recall
            and ev.uncertainty_preserved
            and ev.unsupported_fact_count == 0
            and not ev.unsupported_rationale_detected
            and not ev.belief_promoted_to_fact
        )
