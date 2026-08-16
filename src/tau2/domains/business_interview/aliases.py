"""Bilingual role / system aliases and signal expansion for business_interview.

Used by the evaluator to normalise reconstructed actor/system labels, and by
the stakeholder-truth consistency check to verify the user simulator can answer
every truth the evaluator relies on. Keeping the alias tables in one place
avoids duplication between ``semantic.py`` (evaluation) and ``ground_truth.py``
(stakeholder truth consistency).
"""

from typing import Optional

_ROLE_ALIASES = {
    "sales": (
        "sales",
        "sales employee",
        "sales representative",
        "sales rep",
        "salesperson",
        "quotation handler",
        "quotation preparer",
        "handler",
        "interviewee",
        "you",
        "yourself",
        "営業",
        "営業担当者",
        "営業社員",
        "自分",
        "見積担当",
    ),
    "manager": (
        "manager",
        "approver",
        "supervisor",
        "approval",
        "authority",
        "承認者",
        "上司",
        "承認",
    ),
}

_SYSTEM_ALIASES = {
    "crm": ("crm", "customer relationship management"),
    "quoting": (
        "quoting",
        "quote system",
        "quoting system",
        "quotation system",
        "見積システム",
    ),
    "email": ("email", "e-mail", "mail", "メール"),
    "excel": ("excel", "spreadsheet", "エクセル"),
}


def _normalize_label(label: Optional[str], aliases: dict[str, tuple[str, ...]]) -> str:
    if label is None:
        return ""
    key = label.strip().lower().replace("_", " ").replace("-", " ")
    for canonical, variants in aliases.items():
        if key == canonical or any(v in key for v in variants):
            return canonical
    return key


def norm_role(s: Optional[str]) -> str:
    """Normalise a role label to a canonical role, or the lowercased raw label."""
    return _normalize_label(s, _ROLE_ALIASES)


def norm_system(s: Optional[str]) -> str:
    """Normalise a system label to a canonical system, or the lowercased raw label."""
    return _normalize_label(s, _SYSTEM_ALIASES)


# Bilingual variants for non-data stakeholder-truth values (condition labels,
# rationale / owner belief values) so the consistency check works for both EN
# and JA pre-localized instructions.
_VALUE_VARIANTS = {
    "1,000,000": ("1,000,000", "100万", "百万", "1,000,000 yen", "100万円"),
    "month-end": ("month-end", "monthly", "月末"),
    "credit risk": ("credit risk", "credit", "与信", "与信リスク"),
    "accounting": ("accounting", "account", "経理", "会計"),
}


def value_signals(value: Optional[str], axis: str) -> tuple[str, ...]:
    """Expanded substrings a simulator prompt may use to state ``value``.

    ``value`` is a canonical stakeholder-truth value; ``axis`` tells us how to
    expand it (e.g. an actor value ``sales`` may appear in the prompt as ``you``;
    a data value ``sent_quote`` may appear as ``send``). Returns a tuple of
    case-insensitive substrings; the raw lowercased value is always included.
    """
    v = (value or "").strip().lower()
    if not v:
        return ()
    if axis == "actor":
        return tuple({v} | set(_ROLE_ALIASES.get(v, ())))
    if axis == "system":
        return tuple({v} | set(_SYSTEM_ALIASES.get(v, ())))
    return tuple({v} | set(_VALUE_VARIANTS.get(v, ())))
