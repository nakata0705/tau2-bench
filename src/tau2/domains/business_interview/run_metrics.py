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

import json
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


_MODEL_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi\s+(?:cannot|can't)\s+(?:assist|help|comply|fulfill)\b",
        r"\bi\s+(?:am|'m)\s+unable\s+to\s+(?:assist|help|comply)\b",
        r"\bi\s+must\s+refuse\b",
        r"\b(?:unable|not able)\s+to\s+(?:assist|help)\s+with\s+(?:that|this)\b",
    )
)


def _message_side(message) -> Optional[str]:
    role = getattr(message, "role", None)
    if role == "assistant":
        return "Agent"
    if role == "user":
        return "Stakeholder"
    return None


def _provider_refusal_text(raw_data) -> Optional[str]:
    """Return an explicit provider refusal field, if one was supplied."""
    if not isinstance(raw_data, dict):
        return None
    choices = raw_data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    refusal = message.get("refusal")
    return str(refusal) if refusal else None


def _provider_metadata(raw_data) -> dict:
    """Extract bounded refusal-related metadata from a provider response."""
    if not isinstance(raw_data, dict):
        return {
            "finish_reason": None,
            "moderation_or_content_filter": False,
            "metadata_keys": [],
        }
    choices = raw_data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    metadata_blob = json.dumps(raw_data, ensure_ascii=False).lower()
    moderation = any(
        marker in metadata_blob
        for marker in (
            "content_filter",
            "content filtering",
            "moderation",
            '"safety":',
        )
    ) or bool(re.search(r'"blocked"\s*:\s*true', metadata_blob))
    keys = sorted(
        key
        for key in set(raw_data) | set(choice) | set(message)
        if any(
            marker in key.lower()
            for marker in ("moderation", "content_filter", "safety", "refusal")
        )
    )
    return {
        "finish_reason": choice.get("finish_reason"),
        "moderation_or_content_filter": moderation,
        "metadata_keys": keys,
    }


def _short_text(value, limit: int = 500) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def account_model_refusals(
    messages,
    *,
    agent_model: Optional[str] = None,
    stakeholder_model: Optional[str] = None,
) -> list[dict]:
    """Identify explicit model refusal text without conflating other errors.

    Only non-empty public Agent/Stakeholder message text is inspected. Normal
    ``I don't know`` answers, empty responses, provider exceptions, malformed
    JSON/tool errors and sidecar validation failures are not refusals unless a
    model actually emits one of the explicit refusal forms. Each finding keeps
    the side, per-side call index, model/provider, preceding public prompt,
    bounded response text and provider moderation metadata.
    """
    refusals: list[dict] = []
    side_call_counts: dict[str, int] = {"Agent": 0, "Stakeholder": 0}

    def model_for(side: str) -> Optional[str]:
        return agent_model if side == "Agent" else stakeholder_model

    def provider_for(model: Optional[str]) -> Optional[str]:
        return model.split("/", 1)[0] if model else None

    for index, message in enumerate(messages or []):
        side = _message_side(message)
        if side is None:
            continue
        side_index = side_call_counts[side]
        side_call_counts[side] += 1
        content = getattr(message, "content", None)
        response_text = str(content) if content and str(content).strip() else ""
        matched = next(
            (
                pattern.pattern
                for pattern in _MODEL_REFUSAL_PATTERNS
                if pattern.search(response_text)
            ),
            None,
        )
        provider_refusal = _provider_refusal_text(getattr(message, "raw_data", None))
        if matched is None and provider_refusal is None:
            continue
        if matched is None:
            matched = "provider_refusal_field"
            response_text = provider_refusal or ""

        preceding_prompt = None
        for previous in reversed((messages or [])[:index]):
            if _message_side(previous) == ("Stakeholder" if side == "Agent" else "Agent"):
                previous_content = getattr(previous, "content", None)
                if previous_content and str(previous_content).strip():
                    preceding_prompt = _short_text(previous_content)
                    break

        recovered = False
        for later in (messages or [])[index + 1 :]:
            if _message_side(later) != side:
                continue
            later_content = getattr(later, "content", None)
            if not later_content or not str(later_content).strip():
                continue
            if not any(pattern.search(str(later_content)) for pattern in _MODEL_REFUSAL_PATTERNS):
                recovered = True
                break

        model = model_for(side)
        refusals.append(
            {
                "side": side,
                "call_index": side_index,
                "message_index": index,
                "model": model,
                "provider": provider_for(model),
                "matched_pattern": matched,
                "response_excerpt": _short_text(response_text),
                "preceding_public_prompt": preceding_prompt,
                "provider_metadata": _provider_metadata(
                    getattr(message, "raw_data", None)
                ),
                "retry_recovered": recovered,
            }
        )
    return refusals


def provider_error_count(errors: list) -> int:
    """Count top-level provider/runtime errors supplied by the run harness."""
    return len(errors or [])
