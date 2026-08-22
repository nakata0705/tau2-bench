"""Deterministic tests for LLM call metrics instrumentation.

No network calls: provider ``completion`` is monkeypatched with fake
responses, or the pure record/summary helpers are exercised directly.

Covers:
- agent/stakeholder classification, latency, exact provider usage;
- char components (messages/system/conversation/tool_schema) and
  ``total_input_chars`` correctness;
- failed provider calls record a safe metrics row (``status``/``error_type``,
  never the exception message);
- no raw prompt / private semantic ids are persisted;
- Agent-side trigger classification;
- summary max/p50/p95 + nullable-token aggregation; slowest-calls ordering;
  call_index monotonic.
"""

import json

import pytest

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_call_metrics import (
    LLMCallMetricsCollector,
    LLMCallRecord,
    record_to_dict,
    set_llm_call_metrics_collector,
    slowest_calls,
    summarize_records,
)
from tau2.utils.llm_utils import generate

# ---------------------------------------------------------------------------
# Fake provider response (no network)
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    role = "assistant"
    content = "hello"
    tool_calls = []


class _FakeChoice:
    finish_reason = "stop"

    def __init__(self):
        self.message = _FakeMessage()


class _FakeResponse:
    model = "fake-model"

    def __init__(self, usage=None):
        self._usage = usage
        self.choices = [_FakeChoice()]

    def get(self, key):
        if key == "usage":
            return self._usage
        return None

    def to_dict(self):
        return {}


class _Boom:
    """Raises immediately when called (simulates a failed provider)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def __call__(self, **kwargs):
        raise self._exc


@pytest.fixture
def collector():
    c = LLMCallMetricsCollector()
    set_llm_call_metrics_collector(c)
    yield c
    set_llm_call_metrics_collector(None)


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.

        Args:
            x (int): The number to calculate the square of.

        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


def _monkeypatch_completion(monkeypatch, response):
    monkeypatch.setattr("tau2.utils.llm_utils.completion", lambda **kwargs: response)


# ---------------------------------------------------------------------------
# Classification + latency + exact usage via generate()
# ---------------------------------------------------------------------------


def test_side_classification_agent_and_stakeholder(monkeypatch, collector):
    _monkeypatch_completion(
        monkeypatch, _FakeResponse(usage=_FakeUsage(prompt=11, completion=3))
    )
    msgs = [
        SystemMessage(role="system", content="sys"),
        UserMessage(role="user", content="hello"),
    ]
    generate("fake-model", msgs, side="agent", trigger="stakeholder_message")
    generate("fake-model", msgs, side="stakeholder")
    generate("fake-model", msgs)  # no side -> unspecified
    by = collector.by_side()
    assert set(by) == {"agent", "stakeholder", "unspecified"}
    assert by["agent"][0].latency_seconds >= 0.0
    assert by["stakeholder"][0].latency_seconds >= 0.0
    # trigger is recorded on the agent row only
    assert by["agent"][0].trigger == "stakeholder_message"
    assert by["stakeholder"][0].trigger is None


def test_exact_provider_usage_preserved(monkeypatch, collector):
    _monkeypatch_completion(
        monkeypatch, _FakeResponse(usage=_FakeUsage(prompt=42, completion=7))
    )
    msgs = [UserMessage(role="user", content="hi")]
    generate("fake-model", msgs, side="agent")
    rec = collector.records()[0]
    assert rec.prompt_tokens == 42
    assert rec.completion_tokens == 7
    assert rec.total_tokens == 49


def test_missing_usage_remains_null(monkeypatch, collector):
    _monkeypatch_completion(monkeypatch, _FakeResponse(usage=None))
    msgs = [UserMessage(role="user", content="hi")]
    generate("fake-model", msgs, side="agent")
    rec = collector.records()[0]
    assert rec.prompt_tokens is None
    assert rec.completion_tokens is None
    assert rec.total_tokens is None


def test_component_counts_deterministic_and_tool_schema_separate(
    monkeypatch, collector, tool
):
    _monkeypatch_completion(
        monkeypatch, _FakeResponse(usage=_FakeUsage(prompt=5, completion=1))
    )
    msgs = [
        SystemMessage(role="system", content="you are a bot"),
        UserMessage(role="user", content="square 5"),
    ]
    # with tools
    generate("fake-model", msgs, tools=[tool], side="agent")
    # without tools
    generate("fake-model", msgs, side="agent")

    with_tool, without_tool = collector.records()
    assert with_tool.tool_schema_chars > 0
    assert without_tool.tool_schema_chars == 0
    # system_chars is the same fixed system message in both
    sys_chars = len(json.dumps({"role": "system", "content": "you are a bot"}))
    assert with_tool.system_chars == sys_chars
    assert without_tool.system_chars == sys_chars
    # conversation excludes system: messages_chars - system_chars
    assert with_tool.conversation_chars == (
        with_tool.messages_chars - with_tool.system_chars
    )
    assert without_tool.conversation_chars == (
        without_tool.messages_chars - without_tool.system_chars
    )
    # conversation excludes the tool schema (schema is a separate component)
    assert with_tool.conversation_chars == without_tool.conversation_chars
    # messages_chars is the message-only payload; total_input adds schemas
    assert with_tool.messages_chars == without_tool.messages_chars
    assert with_tool.total_input_chars == (
        with_tool.messages_chars + with_tool.tool_schema_chars
    )
    assert without_tool.total_input_chars == without_tool.messages_chars
    # the message-only payload is NOT called request_chars anymore
    assert not hasattr(with_tool, "request_chars")


def test_stakeholder_output_contract_chars_measured_but_not_stored(
    monkeypatch, collector
):
    _monkeypatch_completion(
        monkeypatch, _FakeResponse(usage=_FakeUsage(prompt=1, completion=1))
    )
    contract = 'Reply with a JSON object: {"message": "..."}' * 5
    msgs = [UserMessage(role="user", content="hi")]
    generate(
        "fake-model",
        msgs,
        side="stakeholder",
        output_contract_text=contract,
    )
    rec = collector.records()[0]
    assert rec.output_contract_chars == len(contract)
    # the contract TEXT is never persisted
    blob = json.dumps(record_to_dict(rec))
    assert contract not in blob


def test_no_raw_content_in_records(monkeypatch, collector):
    _monkeypatch_completion(
        monkeypatch, _FakeResponse(usage=_FakeUsage(prompt=1, completion=1))
    )
    secret = "super-secret-prompt-content-xyz"
    msgs = [SystemMessage(role="system", content=secret)]
    generate("fake-model", msgs, side="stakeholder", output_contract_text=secret)
    blob = json.dumps(record_to_dict(collector.records()[0]))
    assert secret not in blob


# ---------------------------------------------------------------------------
# Failed provider calls record a safe row (requirement 2)
# ---------------------------------------------------------------------------


def test_failed_provider_call_records_error_row(monkeypatch, collector):
    secret_msg = "provider said: authentication failed for token ABC123XYZ"
    monkeypatch.setattr(
        "tau2.utils.llm_utils.completion",
        _Boom(RuntimeError(secret_msg)),
    )
    msgs = [UserMessage(role="user", content="hi")]
    with pytest.raises(RuntimeError):
        generate("fake-model", msgs, side="agent", trigger="tool_result")
    recs = collector.records()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.status == "error"
    assert rec.error_type == "RuntimeError"
    assert rec.total_tokens is None
    # the exception MESSAGE (which may carry request/provider content) is never
    # persisted — only the safe class name
    blob = json.dumps(record_to_dict(rec))
    assert secret_msg not in blob
    assert rec.side == "agent"
    assert rec.trigger == "tool_result"
    # char components are still recorded for the failed attempt, and latency
    # covers the time spent before the failure
    assert rec.messages_chars > 0
    assert rec.latency_seconds >= 0.0


def test_failed_and_success_rows_side_by_side(monkeypatch, collector):
    from tau2.utils import llm_utils

    original = llm_utils.completion
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("provider timeout")
        return _FakeResponse(usage=_FakeUsage(prompt=3, completion=1))

    monkeypatch.setattr("tau2.utils.llm_utils.completion", flaky)
    msgs = [UserMessage(role="user", content="hi")]
    with pytest.raises(TimeoutError):
        generate("fake-model", msgs, side="stakeholder")
    generate("fake-model", msgs, side="stakeholder")
    recs = collector.records()
    assert [r.status for r in recs] == ["error", "success"]
    assert [r.error_type for r in recs] == ["TimeoutError", None]
    s = summarize_records(recs)
    assert s["success"] == 1
    assert s["error"] == 1
    assert s["error_types"] == ["TimeoutError"]
    assert original is not None  # unused guard; keep ref for clarity


# ---------------------------------------------------------------------------
# Summary + slowest calls
# ---------------------------------------------------------------------------


def _mk(
    side,
    i,
    prompt,
    lat,
    messages_chars=1000,
    msg_count=5,
    completion=10,
    status="success",
    error_type=None,
):
    return LLMCallRecord(
        side=side,
        call_index=i,
        model="m",
        message_count=msg_count,
        messages_chars=messages_chars,
        system_chars=500,
        conversation_chars=messages_chars - 500,
        tool_schema_chars=100,
        total_input_chars=messages_chars + 100,
        status=status,
        error_type=error_type,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if prompt is not None else None,
        latency_seconds=lat,
    )


def test_summary_aggregates_with_nullable_tokens():
    recs = [
        _mk("agent", 0, 10, 1.0),
        _mk("agent", 1, 20, 2.0),
        _mk("agent", 2, 30, 3.0),
        _mk("agent", 3, None, 100.0),  # no usage -> excluded from token totals
        _mk("agent", 4, 40, 5.0),
    ]
    s = summarize_records(recs)
    assert s["calls"] == 5
    assert s["success"] == 5
    assert s["error"] == 0
    assert s["error_types"] == []
    assert s["max_prompt_tokens"] == 40
    assert s["total_prompt_tokens"] == 100  # 10+20+30+40, null excluded
    assert s["total_completion_tokens"] == 40
    assert s["token_usage_calls"] == 4
    assert s["latency_p50"] == 3.0
    assert s["latency_p95"] == 100.0
    assert s["latency_max"] == 100.0
    assert s["max_messages_chars"] == 1000
    assert s["max_total_input_chars"] == 1100
    assert s["total_input_chars"] == 5500
    assert s["max_message_count"] == 5


def test_summary_trigger_counts():
    recs = [
        _mk("agent", 0, 1, 1.0),
        _mk("agent", 1, 1, 1.0),
        _mk("agent", 2, 1, 1.0),
        _mk("stakeholder", 3, 1, 1.0),
    ]
    recs[0].trigger = "stakeholder_message"
    recs[1].trigger = "tool_result"
    recs[2].trigger = "multi_tool_result"
    s = summarize_records(recs)
    assert s["trigger_counts"] == {
        "stakeholder_message": 1,
        "tool_result": 1,
        "multi_tool_result": 1,
    }
    # stakeholder-side row with no trigger is not counted
    assert s["trigger_counts"].get("other") is None


def test_summary_empty_side():
    s = summarize_records([])
    assert s["calls"] == 0
    assert s["success"] == 0
    assert s["error"] == 0
    for key in (
        "max_prompt_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "latency_p50",
        "latency_p95",
        "latency_max",
        "max_messages_chars",
        "max_total_input_chars",
        "total_input_chars",
        "max_message_count",
    ):
        assert s[key] is None, key


def test_slowest_calls_ordering():
    recs = [_mk("agent", i, i * 10, float(i)) for i in range(6)]
    slow = slowest_calls(recs, n=3)
    assert [r["call_index"] for r in slow] == [5, 4, 3]
    assert slow[0]["latency_seconds"] == 5.0
    assert "side" in slow[0] and "prompt_tokens" in slow[0]


def test_call_index_monotonic():
    c = LLMCallMetricsCollector()
    c.record(_mk("agent", 0, 1, 1.0))
    c.record(_mk("stakeholder", 0, 2, 1.0))
    c.record(_mk("agent", 0, 3, 1.0))
    idxs = [r.call_index for r in c.records()]
    assert idxs == [0, 1, 2]
