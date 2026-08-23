"""Tool-error accounting for business_interview runs (diagnostics).

A real run must not report success just because the top-level ``errors`` list
is empty: the trajectory may contain ``ToolMessage``\\ s with ``error=True``
(responses starting with ``Error: ...``) that top-level error capture never
sees. This module walks the trajectory, identifies failing tool calls, and
normalizes them into a small set of deterministic categories so repeated
identical failures are grouped.

Categories:

- ``tool_not_found``     — the model called a tool that does not exist
                          (``Tool 'x' not found.``); a policy/schema bug.
- ``missing_reference``  — a reference to an unknown observation/concept/node/
                          edge id (``... not found:`` but a real tool).
- ``invalid_argument``   — malformed / unusable call arguments (incl. JSON
                          parse errors and shape errors like ``must be ...``).
- ``validation_error``   — a semantic validation rejection (``already
                          exists``, ``kind ...``, ``cannot ...``).
- ``other_tool_error``   — anything else.
"""

import re
from typing import Optional

_KNOWN_CATEGORIES = (
    "tool_not_found",
    "missing_reference",
    "invalid_argument",
    "validation_error",
    "other_tool_error",
)

_MISSING_REF_RE = re.compile(
    r"\b(observation|concept|node|edge|graph|property|condition)\s+.*\bnot found\b"
)


def classify_tool_error(content: Optional[str], tool_name: Optional[str] = None) -> str:
    """Normalize one failing tool response into a deterministic category."""
    text = (content or "").strip()
    if not text:
        return "other_tool_error"
    if re.search(r"Tool '[\w.]+' not found", text) or (
        tool_name is not None and "not found" in text and "Tool" in text
    ):
        return "tool_not_found"
    if _MISSING_REF_RE.search(text) and "not found" in text:
        return "missing_reference"
    lowered = text.lower()
    if any(
        w in lowered
        for w in (
            "malformed",
            "parse_error",
            "must be",
            "expected",
            "requires at least",
            "requires a",
            "missing concept",
            "missing reference",
            "invalid evidence",
        )
    ):
        return "invalid_argument"
    if any(
        w in lowered
        for w in ("already exists", "cannot", "invalid", "rejected", "not allowed")
    ):
        return "validation_error"
    return "other_tool_error"


def _tool_name_by_id(messages) -> dict[str, str]:
    """Map tool-call ids -> tool name by walking assistant messages."""
    by_id: dict[str, str] = {}
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if getattr(tc, "id", None):
                by_id[tc.id] = tc.name
    return by_id


def account_tool_errors(messages) -> dict:
    """Account for every failing ``ToolMessage`` in the trajectory.

    ``messages`` is the flattened trajectory (Assistant/Tool/User messages).
    Returns

        tool_error_count          total failing tool messages
        tool_error_categories     sorted unique normalized categories
        tool_error_counts_by_tool {tool_name: count} for failing calls
        tool_error_counts_by_category {category: count}
    """
    by_id = _tool_name_by_id(messages)
    count = 0
    counts_by_tool: dict[str, int] = {}
    counts_by_cat: dict[str, int] = {c: 0 for c in _KNOWN_CATEGORIES}
    for m in messages:
        if not getattr(m, "error", False):
            continue
        count += 1
        tool_name = by_id.get(getattr(m, "id", "") or "") or getattr(
            m, "requestor", None
        )
        cat = classify_tool_error(getattr(m, "content", None), tool_name)
        counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1
        if tool_name:
            counts_by_tool[tool_name] = counts_by_tool.get(tool_name, 0) + 1
    return {
        "tool_error_count": count,
        "tool_error_categories": [
            c for c in _KNOWN_CATEGORIES if counts_by_cat.get(c, 0) > 0
        ],
        "tool_error_counts_by_tool": dict(sorted(counts_by_tool.items())),
        "tool_error_counts_by_category": {
            c: n for c, n in sorted(counts_by_cat.items()) if n > 0
        },
    }


def provider_error_count(errors: list) -> int:
    """Count top-level provider/runtime errors (the ``errors`` list)."""
    return len(errors or [])
