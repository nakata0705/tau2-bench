"""Provider-selection tests that never call an external model API."""

from __future__ import annotations

from types import SimpleNamespace

from tau2.config import (
    openai_client_kwargs,
    resolve_openai_compatible_model,
    resolve_openrouter_model,
)


def _openrouter_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAU2_USE_OPENROUTER_FOR_OPENAI", raising=False)


def test_bare_openai_chat_models_route_through_openrouter(monkeypatch):
    _openrouter_only(monkeypatch)
    assert resolve_openrouter_model("gpt-4o-mini") == ("openrouter/openai/gpt-4o-mini")
    # OpenRouter exposes the public undated model id for this dated OpenAI
    # default.
    assert resolve_openrouter_model("gpt-4.1-2025-04-14") == (
        "openrouter/openai/gpt-4.1"
    )
    assert resolve_openrouter_model("openrouter/openai/gpt-4o-mini") == (
        "openrouter/openai/gpt-4o-mini"
    )


def test_raw_openai_sdk_models_use_openrouter_endpoint(monkeypatch):
    _openrouter_only(monkeypatch)
    assert resolve_openai_compatible_model("text-embedding-3-large") == (
        "openai/text-embedding-3-large"
    )
    assert openai_client_kwargs() == {
        "api_key": "or-test-key",
        "base_url": "https://openrouter.ai/api/v1",
    }


def test_explicit_provider_and_direct_key_are_not_rewritten(monkeypatch):
    _openrouter_only(monkeypatch)
    assert resolve_openrouter_model("anthropic/claude-sonnet-4") == (
        "anthropic/claude-sonnet-4"
    )
    assert resolve_openrouter_model("gpt-4o-mini", explicit_api_key="direct") == (
        "gpt-4o-mini"
    )
    assert openai_client_kwargs(api_key="direct") == {"api_key": "direct"}


def test_explicit_environment_override_can_disable_routing(monkeypatch):
    _openrouter_only(monkeypatch)
    monkeypatch.setenv("TAU2_USE_OPENROUTER_FOR_OPENAI", "false")
    assert resolve_openrouter_model("gpt-4o-mini") == "gpt-4o-mini"


def test_text_generation_passes_the_qualified_model_to_litellm(monkeypatch):
    _openrouter_only(monkeypatch)
    from tau2.data_model.message import SystemMessage, UserMessage
    from tau2.utils import llm_utils

    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(role="assistant", content="ok", tool_calls=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=message)],
            to_dict=lambda: {"choices": []},
        )

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _response: 0.0)
    monkeypatch.setattr(llm_utils, "get_response_usage", lambda _response: None)
    monkeypatch.setattr(llm_utils, "_write_llm_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        llm_utils,
        "_record_llm_call_metrics",
        lambda **kwargs: None,
    )

    response = llm_utils.generate(
        "gpt-4o-mini",
        [
            SystemMessage(role="system", content="be concise"),
            UserMessage(role="user", content="hello"),
        ],
        num_retries=0,
    )
    assert response.content == "ok"
    assert calls[0]["model"] == "openrouter/openai/gpt-4o-mini"
