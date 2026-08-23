"""LLM call metrics and bounded generation diagnostics.

The common ``generate()`` layer records one row per provider generation
attempt. Rows contain context-size / latency numbers plus the minimum bounded
response diagnostics needed for benchmark analysis: call name, provider,
status, finish reason, moderation markers, retry metadata, and explicit model
refusal evidence. Raw prompts, request bodies, headers, secrets, private
StakeholderKnowledge, and private semantic IDs are never persisted.

The refusal detector lives here so the call-level instrumentation and the
legacy public-trajectory compatibility diagnostic use exactly the same narrow
patterns. Ordinary uncertainty (``I don't know``), malformed output, and
provider exceptions are not refusals without explicit refusal evidence.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

__all__ = [
    "LLMCallRecord",
    "LLMCallMetricsCollector",
    "get_llm_call_metrics_collector",
    "set_llm_call_metrics_collector",
    "summarize_records",
    "record_to_dict",
    "model_refusal_records",
    "detect_model_refusal",
    "provider_refusal_text",
    "provider_metadata",
    "short_text",
]


_REFUSAL_EXCERPT_LIMIT = 500
_MODEL_REFUSAL_PATTERNS = (
    re.compile(
        r"\bi\s+(?:cannot|can't)\s+(?:assist|help|comply|fulfill)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi\s+(?:am|'m)\s+unable\s+to\s+(?:assist|help|comply)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bi\s+must\s+refuse\b", re.IGNORECASE),
    re.compile(
        r"\b(?:unable|not able)\s+to\s+(?:assist|help)\s+with\s+(?:that|this)\b",
        re.IGNORECASE,
    ),
)
_PROVIDER_METADATA_MARKERS = (
    "moderation",
    "content_filter",
    "content filtering",
    "safety",
    "refusal",
)
_MODERATION_METADATA_MARKERS = (
    "moderation",
    "content_filter",
    "content filtering",
    "safety",
)


def short_text(value: Any, limit: int = _REFUSAL_EXCERPT_LIMIT) -> Optional[str]:
    """Return a bounded text excerpt suitable for a run artifact."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _choice_and_message(raw_data: Any) -> tuple[dict, dict]:
    if not isinstance(raw_data, dict):
        return {}, {}
    choices = raw_data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, dict):
        choice = {}
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}
    return choice, message


def provider_refusal_text(raw_data: Any) -> Optional[str]:
    """Return the provider's explicit ``message.refusal`` field, if present."""
    _, message = _choice_and_message(raw_data)
    refusal = message.get("refusal")
    return short_text(refusal)


def _bounded_metadata_value(value: Any, depth: int = 0) -> Any:
    """Keep provider moderation metadata bounded and free of long payloads."""
    if depth >= 2:
        return short_text(value, 200)
    if isinstance(value, dict):
        return {
            str(key): _bounded_metadata_value(item, depth + 1)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [_bounded_metadata_value(item, depth + 1) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return short_text(value, 200) if isinstance(value, str) else value
    return short_text(value, 200)


def provider_metadata(raw_data: Any, finish_reason: Any = None) -> dict:
    """Extract bounded finish/moderation metadata from a provider response."""
    choice, message = _choice_and_message(raw_data)
    if finish_reason is None:
        finish_reason = choice.get("finish_reason")
    if not isinstance(raw_data, dict):
        return {
            "finish_reason": finish_reason,
            "moderation_or_content_filter": False,
            "metadata_keys": [],
            "moderation_metadata": {},
        }
    try:
        metadata_blob = json.dumps(raw_data, ensure_ascii=False, default=str).lower()
    except (TypeError, ValueError):
        metadata_blob = ""
    selected: dict[str, Any] = {}
    for source in (raw_data, choice, message):
        for key, value in source.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in _PROVIDER_METADATA_MARKERS):
                selected[key_text] = _bounded_metadata_value(value)
    keys = sorted(selected)
    moderation = any(
        any(marker in key.lower() for marker in _MODERATION_METADATA_MARKERS)
        for key in selected
    )
    moderation = moderation or bool(re.search(r'"blocked"\s*:\s*true', metadata_blob))
    moderation = moderation or str(finish_reason).lower() in {
        "content_filter",
        "content_filter_stop",
    }
    return {
        "finish_reason": finish_reason,
        "moderation_or_content_filter": moderation,
        "metadata_keys": keys,
        "moderation_metadata": selected,
    }


def detect_model_refusal(
    content: Any,
    *,
    raw_data: Any = None,
    provider_refusal: Any = None,
    finish_reason: Any = None,
) -> dict:
    """Classify one generation using explicit response/refusal evidence only.

    The result is safe to persist: text fields are bounded excerpts and
    provider metadata is reduced to bounded moderation-related fields.
    """
    response_text = str(content) if content is not None else ""
    matched_pattern = next(
        (
            pattern.pattern
            for pattern in _MODEL_REFUSAL_PATTERNS
            if pattern.search(response_text)
        ),
        None,
    )
    provider_refusal_value = short_text(provider_refusal) or provider_refusal_text(
        raw_data
    )
    if matched_pattern is None and provider_refusal_value is not None:
        matched_pattern = "provider_refusal_field"
    explicit_refusal = matched_pattern is not None
    refusal_excerpt = (
        short_text(response_text)
        if matched_pattern and matched_pattern != "provider_refusal_field"
        else None
    )
    if not refusal_excerpt and explicit_refusal:
        refusal_excerpt = provider_refusal_value
    metadata = provider_metadata(raw_data, finish_reason=finish_reason)
    return {
        "explicit_refusal": explicit_refusal,
        "matched_pattern": matched_pattern,
        "refusal_excerpt": refusal_excerpt,
        "provider_refusal": provider_refusal_value,
        "provider_metadata": metadata,
    }


# One row per real LLM call: numeric metrics plus bounded diagnostics.
# ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` are exact
# provider usage, or ``None`` when the provider reported none.
#
# Char components (numeric only; never the raw prompt body):
# - ``messages_chars``: serialized message payload (system + conversation).
# - ``conversation_chars``: conversation portion, excluding system messages.
# - ``tool_schema_chars``: serialized tool schemas; separate from messages.
# - ``total_input_chars``: all major serialized input components together
#   (messages + tool schemas; add any other serialized component here).
# - ``output_contract_chars``: for stakeholder calls, the fixed output-sidecar
#   contract appended to every request, kept separate from the conversation.
#
# Every provider generation records a row, including failures: ``status`` is
# ``"success"`` or ``"error"`` and ``error_type`` is a safe exception class
# name (or ``None``). Exception *messages* are never persisted (they may leak
# request/provider content).
@dataclass
class LLMCallRecord:
    side: str
    call_index: int
    model: str
    message_count: int
    messages_chars: int
    system_chars: int
    conversation_chars: int
    tool_schema_chars: int
    total_input_chars: int
    status: str = "success"
    error_type: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    output_contract_chars: Optional[int] = None
    latency_seconds: float = 0.0
    # Which input kind triggered this generation (Agent-side classification,
    # e.g. ``stakeholder_message`` / ``tool_result`` / ``multi_tool_result`` /
    # ``initial_turn`` / ``other``); None for unclassified / non-agent calls.
    trigger: Optional[str] = None
    # Generation-attempt diagnostics. ``attempt_index`` is a per-side,
    # per-call-name sequence number; ``retry_attempt`` is explicit caller
    # metadata because a later call may be a new turn rather than a retry.
    call_name: Optional[str] = None
    provider: Optional[str] = None
    attempt_index: int = 0
    retry_attempt: bool = False
    explicit_refusal: bool = False
    refusal_excerpt: Optional[str] = None
    provider_refusal: Optional[str] = None
    finish_reason: Optional[str] = None
    moderation_metadata: Optional[dict] = None


# Call-trigger classification for the Agent side (see requirement 3): which kind
# of input caused this generation. ``None`` means "not classified" (non-agent
# calls / unknown).
#
#   stakeholder_message    -- the Agent is generating a fresh question to the
#                              stakeholder (or the initial turn).
#   tool_result            -- a single tool result came back to the Agent.
#   multi_tool_result      -- several tool results returned together.
#   other                  -- anything else.
AgentTrigger = str  # one of the constants below (or None)


class LLMCallMetricsCollector:
    """Append-only log of LLM calls for one simulation run."""

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []
        self._index = 0
        self._attempt_indices: dict[tuple[str, Optional[str]], int] = {}

    def record(self, record: LLMCallRecord) -> None:
        record.call_index = self._index
        self._index += 1
        key = (record.side, record.call_name)
        record.attempt_index = self._attempt_indices.get(key, 0)
        self._attempt_indices[key] = record.attempt_index + 1
        self._records.append(record)

    def records(self) -> list[LLMCallRecord]:
        """Snapshot of all metrics and bounded generation diagnostics."""
        return list(self._records)

    def by_side(self) -> dict[str, list[LLMCallRecord]]:
        out: dict[str, list[LLMCallRecord]] = {}
        for r in self._records:
            out.setdefault(r.side, []).append(r)
        return out


# Context var so the collector is inherited by the same-thread synchronous
# orchestrator/agent/user calls without threading a parameter through every
# call site. Default: no collection.
_llm_call_metrics: ContextVar[Optional[LLMCallMetricsCollector]] = ContextVar(
    "llm_call_metrics", default=None
)


def get_llm_call_metrics_collector() -> Optional[LLMCallMetricsCollector]:
    return _llm_call_metrics.get()


def set_llm_call_metrics_collector(
    collector: Optional[LLMCallMetricsCollector],
) -> None:
    """Install (or clear) the collector for the current context."""
    _llm_call_metrics.set(collector)


def _p50(values: list[float]) -> Optional[float]:
    if not values:
        return None
    try:
        return float(median(sorted(values)))
    except (ValueError, TypeError):
        # Defensive: empty/unnormalized input never reaches this (guarded
        # above), but a broken record must not take the run down.
        return None


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    try:
        vals = sorted(values)
        idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
        return float(vals[idx])
    except (ValueError, TypeError):
        return None


def summarize_records(side_records: list[LLMCallRecord]) -> dict:
    """Summarize one side's calls, including refusal/retry diagnostics.

    Exact token totals are aggregated only over calls where exact usage exists,
    with ``token_usage_calls`` reporting that coverage. ``success``/``error``
    report call status; trigger, retry and refusal breakdowns are included when
    available.
    """
    calls = len(side_records)
    exact = [r for r in side_records if r.prompt_tokens is not None]
    exact_prompt = [r.prompt_tokens for r in exact if r.prompt_tokens is not None]
    exact_completion = [
        r.completion_tokens for r in exact if r.completion_tokens is not None
    ]
    token_usage_calls = len(exact)
    latencies = [r.latency_seconds for r in side_records]
    errors = [r for r in side_records if r.status == "error"]
    error_types: list[str] = sorted({r.error_type for r in errors if r.error_type})
    return {
        "calls": calls,
        "success": calls - len(errors),
        "error": len(errors),
        "error_types": error_types,
        "max_prompt_tokens": max(exact_prompt) if exact_prompt else None,
        "max_messages_chars": (
            max(r.messages_chars for r in side_records) if side_records else None
        ),
        "max_total_input_chars": (
            max(r.total_input_chars for r in side_records) if side_records else None
        ),
        "max_message_count": (
            max(r.message_count for r in side_records) if side_records else None
        ),
        "total_prompt_tokens": sum(exact_prompt) if exact_prompt else None,
        "total_completion_tokens": sum(exact_completion) if exact_completion else None,
        "total_input_chars": (
            sum(r.total_input_chars for r in side_records) if side_records else None
        ),
        "latency_p50": _p50(latencies),
        "latency_p95": _p95(latencies),
        "latency_max": max(latencies) if latencies else None,
        "token_usage_calls": token_usage_calls,
        "trigger_counts": _trigger_counts(side_records),
        "retry_attempts": sum(1 for r in side_records if r.retry_attempt),
        "explicit_refusal_count": sum(1 for r in side_records if r.explicit_refusal),
    }


def _trigger_counts(side_records: list[LLMCallRecord]) -> dict[str, int]:
    """Count Agent-generations by the input kind that triggered them. Rows with
    no ``trigger`` (e.g. stakeholder-side calls) are omitted."""
    out: dict[str, int] = {}
    for r in side_records:
        if r.trigger:
            out[r.trigger] = out.get(r.trigger, 0) + 1
    return out


def slowest_calls(side_records: list[LLMCallRecord], n: int = 5) -> list[dict]:
    """The ``n`` slowest calls with safe identifying metrics."""
    ordered = sorted(side_records, key=lambda r: r.latency_seconds, reverse=True)
    return [
        {
            "side": r.side,
            "call_index": r.call_index,
            "call_name": r.call_name,
            "prompt_tokens": r.prompt_tokens,
            "messages_chars": r.messages_chars,
            "message_count": r.message_count,
            "latency_seconds": r.latency_seconds,
        }
        for r in ordered[:n]
    ]


def record_to_dict(record: LLMCallRecord) -> dict:
    """JSON-safe metrics and bounded generation-diagnostic fields."""
    return {
        "side": record.side,
        "call_index": record.call_index,
        "attempt_index": record.attempt_index,
        "call_name": record.call_name,
        "model": record.model,
        "provider": record.provider,
        "message_count": record.message_count,
        "messages_chars": record.messages_chars,
        "system_chars": record.system_chars,
        "conversation_chars": record.conversation_chars,
        "tool_schema_chars": record.tool_schema_chars,
        "total_input_chars": record.total_input_chars,
        "status": record.status,
        "error_type": record.error_type,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
        "output_contract_chars": record.output_contract_chars,
        "latency_seconds": record.latency_seconds,
        "trigger": record.trigger,
        "retry_attempt": record.retry_attempt,
        "explicit_refusal": record.explicit_refusal,
        "refusal_excerpt": record.refusal_excerpt,
        "provider_refusal": record.provider_refusal,
        "finish_reason": record.finish_reason,
        "moderation_metadata": record.moderation_metadata,
    }


def model_refusal_records(records: list[LLMCallRecord]) -> list[dict]:
    """Return one artifact record per explicit call-level model refusal.

    This deliberately does not merge with public trajectory diagnostics: a
    public message may correspond to the same accepted generation, while an
    internal plan/realization retry has no public message at all.
    """
    return [record_to_dict(record) for record in records if record.explicit_refusal]
