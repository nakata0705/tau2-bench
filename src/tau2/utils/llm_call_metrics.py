"""LLM call metrics collection for context-size / latency measurement.

This module is **measurement only** — it records one numeric row per real LLM
generation (side, model, message/char/token counts, latency). It NEVER
persists raw prompts, provider request bodies, headers, secrets, private
StakeholderKnowledge, or private semantic IDs: records hold numbers plus the
safe ``side`` / ``model`` / ``call_index`` identifiers.

Design:
- ``LLMCallMetricsCollector`` is installed via a context var; the common LLM
  layer (``tau2.utils.llm_utils.generate``) records into whichever collector
  is active on the current thread/context, so Agent and Stakeholder calls in
  the same run are captured without duplicating provider logic.
- The caller explicitly declares which side made the call (``side="agent"`` /
  ``side="stakeholder"``); the side is never inferred from message text.
- Token counts come ONLY from provider/client usage metadata. ``None`` is
  stored when usage is missing — exact counts are never invented. Optional
  estimates must be named separately (not implemented by default).
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from statistics import median
from typing import Optional

__all__ = [
    "LLMCallRecord",
    "LLMCallMetricsCollector",
    "get_llm_call_metrics_collector",
    "set_llm_call_metrics_collector",
    "summarize_records",
]


# One row per real LLM call (numbers + safe identifiers only).
# ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` are exact
# provider usage, or ``None`` when the provider reported none.
@dataclass
class LLMCallRecord:
    side: str
    call_index: int
    model: str
    message_count: int
    request_chars: int
    system_chars: int
    conversation_chars: int
    tool_schema_chars: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_seconds: float = 0.0


class LLMCallMetricsCollector:
    """Append-only numeric log of LLM calls for one simulation run."""

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []
        self._index = 0

    def record(self, record: LLMCallRecord) -> None:
        record.call_index = self._index
        self._index += 1
        self._records.append(record)

    def records(self) -> list[LLMCallRecord]:
        """Snapshot of all records (safe numbers only)."""
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
    return float(median(sorted(values)))


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))
    return float(vals[idx])


def summarize_records(side_records: list[LLMCallRecord]) -> dict:
    """One side's summary (numeric only). Exact token totals are aggregated
    only over calls where exact usage exists, with ``token_usage_calls``
    reporting that coverage."""
    calls = len(side_records)
    exact = [r for r in side_records if r.prompt_tokens is not None]
    exact_prompt = [r.prompt_tokens for r in exact if r.prompt_tokens is not None]  # type: ignore[misc]
    exact_completion = [
        r.completion_tokens for r in exact if r.completion_tokens is not None
    ]  # type: ignore[misc]
    token_usage_calls = len(exact)
    latencies = [r.latency_seconds for r in side_records]
    return {
        "calls": calls,
        "max_prompt_tokens": max(exact_prompt) if exact_prompt else None,
        "max_request_chars": (
            max(r.request_chars for r in side_records) if side_records else None
        ),
        "max_message_count": (
            max(r.message_count for r in side_records) if side_records else None
        ),
        "total_prompt_tokens": sum(exact_prompt) if exact_prompt else None,
        "total_completion_tokens": (
            sum(exact_completion) if exact_completion else None
        ),
        "latency_p50": _p50(latencies),
        "latency_p95": _p95(latencies),
        "latency_max": max(latencies) if latencies else None,
        "token_usage_calls": token_usage_calls,
    }


def slowest_calls(side_records: list[LLMCallRecord], n: int = 5) -> list[dict]:
    """The ``n`` slowest calls as numeric-only dicts (side, call_index,
    prompt_tokens, request_chars, message_count, latency_seconds)."""
    ordered = sorted(side_records, key=lambda r: r.latency_seconds, reverse=True)
    return [
        {
            "side": r.side,
            "call_index": r.call_index,
            "prompt_tokens": r.prompt_tokens,
            "request_chars": r.request_chars,
            "message_count": r.message_count,
            "latency_seconds": r.latency_seconds,
        }
        for r in ordered[:n]
    ]


def record_to_dict(record: LLMCallRecord) -> dict:
    """Numeric-only dict form ({json.dumps}-serializable, no content)."""
    return {
        "side": record.side,
        "call_index": record.call_index,
        "model": record.model,
        "message_count": record.message_count,
        "request_chars": record.request_chars,
        "system_chars": record.system_chars,
        "conversation_chars": record.conversation_chars,
        "tool_schema_chars": record.tool_schema_chars,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
        "latency_seconds": record.latency_seconds,
    }
