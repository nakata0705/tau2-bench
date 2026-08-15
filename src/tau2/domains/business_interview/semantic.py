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

from dataclasses import dataclass
from typing import Callable, Optional

from tau2.domains.business_interview.data_model import (
    ClaimEvaluation,
    EpistemicStatus,
    InterviewDB,
    InterviewEvaluation,
    ObjectiveRationale,
    RationaleValue,
    Source,
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
    # Hedged-belief markers (the stakeholder reasons tentatively, e.g.
    # "経理側の都合ではないかと思う"). These let the evaluator recognise a
    # belief that the agent records within an uncertainty/hedged phrasing.
    "ではないか",
    "と思う",
    "と思います",
    "と推測",
    "関係している",
    "関わっている",
    "による",
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
# Structured topic identity predicates.
#
# These implement the small structural rules used to associate a finding with a
# canonical topic from its content (when no explicit ``topic`` is given):
#   month_end_excel : excel AND (month_end OR accounting)
#   high_value_quote: amount_threshold OR explicit high-value expression
# This avoids weak single-word matches (e.g. ``accounting`` alone being read as
# a month-end Excel hand-off). Bilingual; bounded (no huge synonym dictionary).
# ---------------------------------------------------------------------------


def _has_excel(lowered: str) -> bool:
    return "excel" in lowered or "エクセル" in lowered or "エクセル" in lowered


def _has_month(lowered: str) -> bool:
    return any(
        k in lowered for k in ("month", "month-end", "月末", "毎月末", "月次", "月締め")
    )


def _has_accounting(lowered: str) -> bool:
    return any(k in lowered for k in ("accounting", "会計", "経理"))


def _has_amount(lowered: str) -> bool:
    return any(
        k in lowered
        for k in (
            "million",
            "one million",
            "1,000,000",
            "1000000",
            "100万",
            "百万円",
            "1,000,000yen",
        )
    )


def _has_explicit_high_value(lowered: str) -> bool:
    return any(
        k in lowered
        for k in (
            "high-value",
            "high value",
            "large amount",
            "large quotation",
            "高額",
            "大口",
            "金額が大きい",
        )
    )


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


# Minimal canonical rationale value signals used to infer a claim's value from
# free text (when the agent does not set the explicit ``value`` field). Small and
# bounded, matching the canonical values in ``RationaleValue``.
CLAIM_VALUE_SIGNALS: dict[str, tuple[str, ...]] = {
    RationaleValue.CREDIT_RISK.value: ("credit", "与信", "クレジット"),
    RationaleValue.ACCOUNTING_NEED.value: ("accounting", "会計", "経理"),
}


# Normalization for source labels recorded by the agent (free string) into a
# canonical source id. Kept small; unknown strings stay as-is.
_SOURCE_ALIASES: dict[str, str] = {
    "sales": Source.SALES.value,
    "sales employee": Source.SALES.value,
    "salesperson": Source.SALES.value,
    "営業": Source.SALES.value,
    "営業担当者": Source.SALES.value,
    "営業社員": Source.SALES.value,
    "sales team": Source.SALES.value,
    "accounting": Source.ACCOUNTING.value,
    "accounting team": Source.ACCOUNTING.value,
    "経理": Source.ACCOUNTING.value,
    "会計": Source.ACCOUNTING.value,
    "経理チーム": Source.ACCOUNTING.value,
}


@dataclass
class ExpectedClaim:
    """A claim a scenario requires a source to have made on a topic.

    This is the ground truth for source-claim coverage: the evaluator checks
    whether a recorded claim with matching source + epistemic status + value
    exists.
    """

    source: str
    epistemic_status: EpistemicStatus
    value: str


class TopicSpec:
    def __init__(
        self,
        topic: Topic,
        expected_rationale_status: EpistemicStatus,
        content_signals: tuple[str, ...],
        rationale_value_signals: tuple[str, ...] = (),
        matcher: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.topic = topic
        self.expected_rationale_status = expected_rationale_status
        self.content_signals = content_signals
        self.rationale_value_signals = rationale_value_signals
        self._matcher = matcher

    def content_matches(self, lowered: str) -> bool:
        """Structured content-based topic association.

        When a ``matcher`` is provided it is used (a small structural rule);
        otherwise the legacy word-OR ``content_signals`` fallback applies. The
        structured rules avoid weak single-word matches (e.g. ``accounting``
        alone is not enough to call something a month-end Excel hand-off).
        """
        if self._matcher is not None:
            return self._matcher(lowered)
        return any(sig in lowered for sig in self.content_signals)


def _matches_month_end_excel(lowered: str) -> bool:
    return _has_excel(lowered) and (_has_month(lowered) or _has_accounting(lowered))


def _matches_high_value_quote(lowered: str) -> bool:
    return _has_amount(lowered) or _has_explicit_high_value(lowered)


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
        # structured: excel AND (month_end OR accounting) — ``accounting`` alone
        # is not enough to claim a month-end Excel hand-off.
        matcher=_matches_month_end_excel,
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
        # structured: amount_threshold OR explicit high-value expression.
        matcher=_matches_high_value_quote,
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

# Claims a scenario *requires* to be captured, per topic. A claim is a specific
# source asserting a specific epistemic status with a canonical value. This is
# ground truth and lives domain-side (never in the agent-visible policy/tools).
# A scenario whose topic has no required claim means only discovery + objective
# state matter for that topic.
SCENARIO_CLAIMS: dict[str, dict[Topic, tuple[ExpectedClaim, ...]]] = {
    "quotation_belief_uncertainty_1": {
        Topic.MONTH_END_EXCEL: (
            ExpectedClaim(
                source=Source.SALES.value,
                epistemic_status=EpistemicStatus.BELIEF,
                value=RationaleValue.ACCOUNTING_NEED.value,
            ),
        ),
    },
    "quotation_multi_exception_1": {
        Topic.HIGH_VALUE_QUOTE: (
            ExpectedClaim(
                source=Source.SALES.value,
                epistemic_status=EpistemicStatus.FACT,
                value=RationaleValue.CREDIT_RISK.value,
            ),
        ),
    },
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


# Canonical mapping from a rationale claim's value to the topic it concerns.
# In this benchmark each canonical value belongs to exactly one business element
# (accounting_need -> the month-end Excel hand-off; credit_risk -> the
# high-value quote check). This lets a rationale claim be attributed to the
# right topic even when its free text only names the reason (e.g. a belief
# ``due to Accounting's circumstances``) and does not restate the exception's
# own identity (excel / month / amount). Bounded, canonical — not a synonym
# dictionary.
VALUE_TO_TOPIC: dict[str, Topic] = {
    RationaleValue.ACCOUNTING_NEED.value: Topic.MONTH_END_EXCEL,
    RationaleValue.CREDIT_RISK.value: Topic.HIGH_VALUE_QUOTE,
}


def _content_inferred_topics(content: str) -> list[Topic]:
    """The topics whose structured identity rule matches ``content``."""
    lowered = content.lower()
    return [
        spec.topic for spec in TOPIC_SPECS.values() if spec.content_matches(lowered)
    ]


def _candidate_topics(content: str, value: Optional[str]) -> list[Topic]:
    """All topics a finding could plausibly be about.

    Combines the structured content identity rule with value-based inference: a
    rationale claim (one that asserts a rationale or carries an explicit value)
    that names a canonical reason value is attributed to the topic that value
    belongs to. Value-based inference is gated on ``asserts_rationale`` so that
    a bare mention of a shared word (e.g. ``accounting`` in a non-rationale
    sentence) is NOT enough to claim the month-end Excel hand-off.
    """
    cands = set(_content_inferred_topics(content))
    if value is not None and value in VALUE_TO_TOPIC:
        cands.add(VALUE_TO_TOPIC[value])
    elif asserts_rationale(content):
        for v in detect_claim_values(content):
            t = VALUE_TO_TOPIC.get(v)
            if t is not None:
                cands.add(t)
    return list(cands)


def topic_of_finding(
    topic: Optional[Topic], content: str, value: Optional[str] = None
) -> Optional[Topic]:
    """Resolve the canonical topic a finding is about.

    An explicit canonical ``topic`` field is the preferred identity. It is only
    distrusted when the finding's *content* unambiguously points at a different
    topic (topic-misattribution robustness): the value field alone never
    overrides an explicit topic, so a claim attributed to a topic with a wrong
    value can still be evaluated on that topic (e.g. ``credit_risk`` recorded
    on the month-end topic).

    When no explicit topic is given, content candidates are combined with
    value-based inference (a rationale value maps to the topic it concerns),
    using a single unambiguous match only.
    """
    if topic is not None:
        content_cands = _content_inferred_topics(content)
        if len(content_cands) == 1 and content_cands[0] != topic:
            # Content clearly indicates a different topic than the reported one:
            # do not unconditionally trust the reported topic and do not reward
            # a misattribution to the content topic either. Leave it
            # unassociated so it satisfies neither.
            return None
        return topic
    cands = _candidate_topics(content, value)
    if len(cands) == 1:
        return cands[0]
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


def normalize_source(source: Optional[str]) -> Optional[str]:
    """Normalize a recorded source label into a canonical source id.

    Unknown labels are returned lowercased unchanged (so they still compare,
    but only match expected sources exactly). None stays None (unspecified ->
    the single stakeholder is assumed)."""
    if source is None:
        return None
    key = source.strip().lower().replace("_", " ").replace("-", " ")
    return _SOURCE_ALIASES.get(key, key)


def detect_claim_values(content: str) -> set[str]:
    """Infer the canonical rationale value(s) mentioned in free text.

    Used when the agent does not set the explicit ``value`` field. Bounded by
    :data:`CLAIM_VALUE_SIGNALS`."""
    lowered = content.lower()
    values = set()
    for value, signals in CLAIM_VALUE_SIGNALS.items():
        if any(sig in lowered for sig in signals):
            values.add(value)
    return values


def _source_matches(recorded_source: Optional[str], expected_source: str) -> bool:
    """Whether a recorded source satisfies an expected source.

    In the single-stakeholder benchmark the one interviewee is the implicit
    source, so an unspecified source (None) counts as correct. An explicitly
    wrong source does not."""
    if recorded_source is None:
        return True
    return recorded_source == expected_source


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
        # Canonical findings: (kind, content, topic, source, value). For facts
        # the agent's recorded epistemic_status decides FACT vs BELIEF; `source`
        # is who stated the claim; `value` is the optional canonical rationale
        # value. Each finding is associated with a canonical topic (explicit
        # `topic` field preferred, content / value inference otherwise).
        canonical = cls._canonical(db)

        # ---- Scenario-level discovery / epistemic (backward compatible) ----
        # A finding that is attributed to a canonical *exception* topic (or that
        # matches the legacy exception keywords) is exception-related. It must
        # never count as a normal-process fact, and it is what the epistemic
        # (no-unsupported-rationale / uncertainty) checks scan. The resolved
        # topic already cross-checks reported vs content so a misattributed
        # finding is not counted as a normal fact either.
        normal_fact_recall = any(
            kind == EpistemicStatus.FACT and not is_exception_related(content, topic)
            for kind, content, topic, _, _ in canonical
        )
        exception_recall = any(
            is_exception_related(content, topic)
            for _, content, topic, _, _ in canonical
        )

        unsupported_fact_count = 0
        unsupported_rationale_detected = False
        belief_promoted_to_fact = False
        uncertainty_preserved = False

        for kind, content, topic, _, _ in canonical:
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

        # A stakeholder belief promoted to a FACT may evade the unsupported-
        # rationale scan when its content still hedges (e.g. "... but not
        # certain"). Derive the scenario-level flag from the per-claim
        # promotion detection too, so a promotion is never reported as clean.
        claim_promoted = any(
            claim.promoted_to_fact
            for topic_eval in topics.values()
            for claim in topic_eval.claims
        )
        belief_promoted_to_fact = belief_promoted_to_fact or claim_promoted

        # ---- Diagnostic top-level booleans --------------------------------
        # protocol_pass: did the agent actually close the interview.
        protocol_pass = db.interview_complete
        # interview_quality_pass: full discovery + epistemic quality, judged
        # independently of protocol. Every required topic must be discovered
        # with the objective rationale preserved correctly and every required
        # source claim captured; nothing may be invented or belief-promoted.
        topics_ok = bool(topics) and all(
            t.discovered
            and t.objective_rationale.correct
            and t.required_claims_complete
            and not t.unsupported_rationale
            for t in topics.values()
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
    def _canonical(cls, db: InterviewDB):
        """Build the canonical findings: (kind, content, topic, source, value)."""
        canonical: list[
            tuple[EpistemicStatus, str, Optional[Topic], Optional[str], Optional[str]]
        ] = []
        for fact in db.facts:
            value = fact.value.value if fact.value is not None else None
            canonical.append(
                (
                    fact.epistemic_status,
                    fact.content,
                    topic_of_finding(fact.topic, fact.content, value),
                    normalize_source(
                        fact.source.value if fact.source is not None else None
                    ),
                    value,
                )
            )
        for exception in db.exceptions:
            canonical.append(
                (
                    EpistemicStatus.EXCEPTION,
                    exception.content,
                    topic_of_finding(exception.topic, exception.content),
                    normalize_source(
                        exception.source.value if exception.source is not None else None
                    ),
                    None,
                )
            )
        for uncertainty in db.uncertainties:
            canonical.append(
                (
                    EpistemicStatus.UNKNOWN,
                    uncertainty.content,
                    topic_of_finding(uncertainty.topic, uncertainty.content),
                    normalize_source(
                        uncertainty.source.value
                        if uncertainty.source is not None
                        else None
                    ),
                    None,
                )
            )
        return canonical

    @classmethod
    def _evaluate_topics(
        cls, canonical, scenario_id: Optional[str]
    ) -> dict[str, TopicEvaluation]:
        """Evaluate each topic required by the scenario (or inferred from the
        findings when no scenario id is supplied)."""
        required = list(SCENARIO_TOPICS.get(canonical_scenario_id(scenario_id), ()))
        if not required:
            inferred = {topic for _, _, topic, _, _ in canonical if topic is not None}
            # Keep a deterministic order consistent with TOPIC_SPECS.
            required = [
                spec.topic for spec in TOPIC_SPECS.values() if spec.topic in inferred
            ]

        scenario = canonical_scenario_id(scenario_id)
        expected_claims = SCENARIO_CLAIMS.get(scenario, {})

        evaluations: dict[str, TopicEvaluation] = {}
        for topic in required:
            spec = TOPIC_SPECS.get(topic)
            if spec is None:
                continue
            topic_findings = [item for item in canonical if item[2] == spec.topic]
            evaluations[spec.topic.value] = cls._evaluate_topic(
                topic_findings, spec, expected_claims.get(topic, ())
            )
        return evaluations

    @classmethod
    def _evaluate_topic(
        cls,
        findings,
        spec: TopicSpec,
        expected_claims: tuple[ExpectedClaim, ...] = (),
    ) -> TopicEvaluation:
        """Evaluate a single topic from the findings attributed to it.

        A topic can hold multiple epistemic states simultaneously:

        - ``discovered``: the exception for this topic was recorded.
        - ``objective_rationale``: the objective state (from ground truth) and
          whether it was correctly represented (UNKNOWN preserved as an
          uncertainty; a confirmed FACT captured).
        - ``claims``: the recorded source claims (stakeholder assertions about
          the rationale), each evaluated for source / status / value correctness.
        - ``required_claims_complete``: every required source claim was captured
          with correct source + status + value.

        ``rationale_status`` / ``rationale_correct`` are legacy derived fields
        (single-value collapse) kept for backward compatibility; they are not
        the source of truth.
        """
        topic_id = spec.topic
        discovered = any(
            kind == EpistemicStatus.EXCEPTION or spec.content_matches(content.lower())
            for kind, content, _, _, _ in findings
        )

        unsupported = any(
            is_unsupported_rationale(kind, content, topic=spec.topic)
            for kind, content, _, _, _ in findings
        )

        # ---- Objective rationale state (from ground truth) ----------------
        objective_status = spec.expected_rationale_status.value
        if spec.expected_rationale_status == EpistemicStatus.UNKNOWN:
            objective_unknown_preserved = any(
                kind == EpistemicStatus.UNKNOWN and expresses_uncertainty(content)
                for kind, content, _, _, _ in findings
            )
            objective_correct = objective_unknown_preserved and not unsupported
        else:
            captured_value = any(
                kind in (EpistemicStatus.FACT, EpistemicStatus.EXCEPTION)
                and (value is not None or asserts_rationale_or_value(content))
                and any(
                    sig in content.lower() or (value is not None and sig in value)
                    for sig in spec.rationale_value_signals
                )
                for kind, content, _, _, value in findings
            )
            objective_correct = captured_value and not unsupported

        objective_rationale = ObjectiveRationale(
            status=objective_status, correct=objective_correct
        )

        # ---- Source claims --------------------------------------------------
        claims, required_complete = cls._evaluate_claims(findings, expected_claims)

        # ---- Legacy derived fields (backward compatibility only) ----------
        has_fact_rationale = any(
            kind in (EpistemicStatus.FACT, EpistemicStatus.EXCEPTION)
            and (value is not None or asserts_rationale_or_value(content))
            for kind, content, _, _, value in findings
        )
        has_belief_rationale = any(
            kind == EpistemicStatus.BELIEF
            and (value is not None or asserts_rationale_or_value(content))
            for kind, content, _, _, value in findings
        )
        has_unknown = any(
            kind == EpistemicStatus.UNKNOWN and expresses_uncertainty(content)
            for kind, content, _, _, _ in findings
        )
        if has_fact_rationale:
            rationale_status = EpistemicStatus.FACT.value
        elif has_belief_rationale:
            rationale_status = EpistemicStatus.BELIEF.value
        elif has_unknown:
            rationale_status = EpistemicStatus.UNKNOWN.value
        else:
            rationale_status = None

        rationale_correct = (
            objective_rationale.correct and required_complete and not unsupported
        )

        return TopicEvaluation(
            topic=topic_id.value,
            discovered=discovered,
            objective_rationale=objective_rationale,
            claims=claims,
            required_claims_complete=required_complete,
            unsupported_rationale=unsupported,
            rationale_status=rationale_status,
            rationale_correct=rationale_correct,
        )

    @classmethod
    def _evaluate_claims(cls, findings, expected_claims):
        """Evaluate recorded source claims against the scenario's expected
        claims, and whether all expected claims were captured.

        A claim is a recorded assertion about the rationale: a FACT / BELIEF (or
        an EXCEPTION that itself asserts a rationale value), or a hedged belief
        embedded in an uncertainty record (e.g. "経理側の都合ではないかと思うが
        確信はない"). ``required_complete`` is True when every expected claim is
        satisfied by some recorded claim.
        """
        recorded: list[tuple[EpistemicStatus, str, Optional[str], Optional[str]]] = []
        for kind, content, _, source, value in findings:
            has_explicit = value is not None
            is_rationale = asserts_rationale_or_value(content)
            if kind == EpistemicStatus.UNKNOWN:
                # A hedged belief embedded in an uncertainty record is a captured
                # belief claim (not merely the objective unknown) IF it asserts a
                # rationale and names a value. A plain objective-unknown
                # ("does not know why...") stays with objective_rationale.
                if not (
                    is_rationale and (has_explicit or detect_claim_values(content))
                ):
                    continue
                presented = EpistemicStatus.BELIEF
            else:
                if kind == EpistemicStatus.BELIEF:
                    presented = EpistemicStatus.BELIEF
                else:  # FACT / EXCEPTION
                    presented = EpistemicStatus.FACT
                # A claim must assert a rationale (a causal/hedge connective or a
                # known value like "credit"), or carry an explicit value field.
                # A bare exception description (e.g. "sends Excel to Accounting")
                # names a shared word but is not a rationale claim.
                if not (has_explicit or is_rationale):
                    continue
            detected = detect_claim_values(content)
            resolved = (
                value
                if value is not None
                else (next(iter(detected)) if len(detected) == 1 else None)
            )
            recorded.append((presented, content, source, resolved))

        claim_evals: list[ClaimEvaluation] = []
        for presented, _content, source, resolved in recorded:
            exp = expected_claims[0] if expected_claims else None
            if exp is not None:
                source_correct = _source_matches(source, exp.source)
                status_correct = presented == exp.epistemic_status
                value_correct = resolved == exp.value
                correct = source_correct and status_correct and value_correct
                promoted = (
                    exp.epistemic_status == EpistemicStatus.BELIEF
                    and presented == EpistemicStatus.FACT
                    and source_correct
                    and value_correct
                )
            else:
                source_correct = status_correct = value_correct = correct = promoted = (
                    False
                )
            claim_evals.append(
                ClaimEvaluation(
                    source=source,
                    epistemic_status=presented.value,
                    value=resolved,
                    source_correct=source_correct,
                    status_correct=status_correct,
                    value_correct=value_correct,
                    correct=correct,
                    promoted_to_fact=promoted,
                )
            )

        required_complete = all(
            any(
                presented == exp.epistemic_status
                and _source_matches(source, exp.source)
                and resolved == exp.value
                for presented, _content, source, resolved in recorded
            )
            for exp in expected_claims
        )
        return claim_evals, required_complete

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
