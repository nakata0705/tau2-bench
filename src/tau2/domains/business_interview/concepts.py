"""Scenario-local, language-independent concepts for business_interview (v2).

The benchmark is not a hidden-id guessing game and not a surface-language
matching game: an equivalent English or Japanese reconstruction must be
evaluated identically. To achieve that, the evaluator resolves free-text
action / condition / metadata / rationale strings to a bounded set of
*concepts* using bilingual signal keywords.

The agent is never asked to output a concept id. Only the evaluator uses these
concepts (evaluator-only, in domain code). Resolution is deterministic
(substring signal scoring), so no LLM judge is needed and EN/JA are equivalent.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Concept:
    """A bounded semantic target that free text is resolved against.

    ``primary`` signals are the defining keywords (typically the verb /
    action word) and are weighted double. ``context`` signals add supporting
    evidence (object, tool, role). ``actor`` / ``system`` are used only to
    break ties between equally-scored concepts — never to force a match.
    """

    id: str
    primary: list[str]
    context: list[str] = ()
    actor: Optional[str] = None
    system: Optional[str] = None


# ---------------------------------------------------------------------------
# Step concepts (the 6 actions of the quotation workflow)
# ---------------------------------------------------------------------------

STEP_CONCEPTS = [
    Concept(
        "receive_request",
        ["receive", "request", "intake", "受け付", "受付", "依頼を受け"],
        ["customer", "quotation", "見積"],
        actor="sales",
    ),
    Concept(
        "check_customer",
        ["check", "verify", "確認", "照会"],
        ["customer", "crm", "顧客"],
        actor="sales",
        system="crm",
    ),
    Concept(
        "create_quote",
        ["create", "build", "prepare", "generate", "作成", "生成"],
        ["quotation", "quote", "見積", "quoting"],
        actor="sales",
        system="quoting",
    ),
    Concept(
        "approve_quote",
        ["approve", "authorize", "承認", "審査"],
        ["approval", "quotation", "見積", "manager"],
        actor="manager",
        system="quoting",
    ),
    Concept(
        "send_quote",
        ["send", "deliver", "transmit", "送付", "送信", "発送"],
        ["customer", "email", "見積"],
        actor="sales",
        system="email",
    ),
    Concept(
        "month_end_summary",
        ["month", "month-end", "monthly", "月末", "集計"],
        ["summary", "accounting", "excel", "経理", "見積"],
        actor="sales",
        system="excel",
    ),
]

# ---------------------------------------------------------------------------
# Read/write data-item concepts (for recall + precision scoring)
# ---------------------------------------------------------------------------

DATA_CONCEPTS = [
    Concept("d_request", ["request", "依頼"]),
    Concept("d_customer", ["customer", "顧客"]),
    Concept("d_pricing", ["pricing", "price", "価格", "料金"]),
    Concept("d_quote", ["quotation", "見積", "見積書"], ["quote"]),
    Concept("d_approval", ["approval", "承認", "批准"]),
    Concept("d_sent", ["sent", "send", "送付", "送信", "deliver"]),
    Concept("d_summary", ["excel", "summary", "集計", "excel_summary"]),
]

# ---------------------------------------------------------------------------
# Transition / branch condition concepts
# ---------------------------------------------------------------------------

CONDITION_CONCEPTS = [
    Concept(
        "cond_amount_over",
        ["over", "above", "exceed", "greater", "more than", "超", "超過"],
        ["amount", "1,000,000", "100万", "百万", "高額", "金額"],
    ),
    Concept(
        "cond_amount_below",
        ["below", "under", "less than", "at or below", "以下", "未満"],
        ["amount", "1,000,000", "100万", "百万", "金額"],
    ),
    Concept(
        "cond_month_end",
        ["month", "month-end", "monthly", "月末"],
        ["summary", "accounting", "excel", "経理"],
    ),
    Concept(
        "cond_threshold",
        ["threshold", "閾値", "しきい値"],
        ["amount", "金額"],
    ),
]

# ---------------------------------------------------------------------------
# Workflow metadata concepts (trigger / purpose / outcome)
# ---------------------------------------------------------------------------

METADATA_CONCEPTS = {
    "trigger": Concept(
        "trigger", ["request", "依頼"], ["customer", "quotation", "見積"]
    ),
    "purpose": Concept(
        "purpose",
        ["accurate", "正確"],
        ["produce", "create", "quotation", "見積", "作成"],
    ),
    "outcome": Concept(
        "outcome",
        ["receive", "receives", "受け取", "受け渡し"],
        ["customer", "quotation", "見積", "completed", "完了"],
    ),
}

# ---------------------------------------------------------------------------
# Confirmed-rationale content concepts
# ---------------------------------------------------------------------------

RATIONALE_CONCEPTS = {
    "credit_risk": Concept(
        "credit_risk",
        ["credit", "与信"],
        ["risk", "risk management", "creditworthiness", "リスク", "credit control"],
    ),
}


def resolve(
    text: Optional[str],
    concepts: list[Concept],
    actor: Optional[str] = None,
    system: Optional[str] = None,
) -> Optional[str]:
    """Resolve free text to the id of the best-matching concept, or None.

    Scoring is ``2 * primary_hits + context_hits`` (substring, case-insensitive).
    A concept must score > 0 to be a candidate. ``actor`` / ``system`` only
    break ties between equally-scored concepts.
    """
    if not text:
        return None
    t = text.lower()
    scored: list[tuple[int, Concept]] = []
    for c in concepts:
        prim = sum(1 for s in c.primary if s in t)
        ctx = sum(1 for s in c.context if s in t)
        sc = 2 * prim + ctx
        if sc > 0:
            scored.append((sc, c))
    if not scored:
        return None
    best_score = max(sc for sc, _ in scored)
    candidates = [c for sc, c in scored if sc == best_score]
    if len(candidates) == 1:
        return candidates[0].id
    if actor:
        a = actor.strip().lower()
        for c in candidates:
            if c.actor and c.actor in a or (a and c.actor and a in c.actor):
                return c.id
    if system:
        s = system.strip().lower()
        for c in candidates:
            if c.system and c.system in s:
                return c.id
    return candidates[0].id


def resolve_exact(
    text: Optional[str], concepts: list[Concept], expected_id: str
) -> bool:
    """True if free text resolves to exactly ``expected_id``."""
    return resolve(text, concepts) == expected_id


def resolve_metadata(text: Optional[str], expected_key: str) -> bool:
    return resolve_exact(text, list(METADATA_CONCEPTS.values()), expected_key)
