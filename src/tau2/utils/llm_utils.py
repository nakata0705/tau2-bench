import json
import logging
import os
import re
import time
import uuid
import warnings
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx
import litellm  # pyright: ignore[reportMissingImports]
from litellm import completion, completion_cost  # pyright: ignore[reportMissingImports]
from litellm.caching.caching import Cache  # pyright: ignore[reportMissingImports]
from litellm.main import ModelResponse, Usage  # pyright: ignore[reportMissingImports]
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
    resolve_openrouter_model,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ParticipantMessageBase,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# Suppress Pydantic serialization warnings from LiteLLM
# These occur due to type mismatches between streaming and non-streaming response types
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

# Configure httpx connection limits for LiteLLM
httpx_limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
litellm.client_session = httpx.Client(limits=httpx_limits)
litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)

# Context variable to store the directory where LLM debug logs should be written
llm_log_dir: ContextVar[Optional[Path]] = ContextVar("llm_log_dir", default=None)

# Context variable to store the LLM logging mode ("all" or "latest")
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")

# litellm._turn_on_debug()

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

if USE_LANGFUSE:
    litellm.success_callback = ["langfuse"]
else:
    litellm.success_callback = []

litellm.drop_params = True

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    match = re.match(
        r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)",
        model,
    )
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def _record_llm_call_metrics(
    *,
    side: Optional[str],
    model: str,
    messages: Sequence[Message],
    litellm_messages: list[dict],
    tools_schema: Optional[list[dict]],
    usage: Optional[dict],
    latency_seconds: float,
    status: str = "success",
    error_type: Optional[str] = None,
    output_contract_text: Optional[str] = None,
    trigger: Optional[str] = None,
    call_name: Optional[str] = None,
    response_content: Any = None,
    response_raw_data: Any = None,
    provider_refusal: Any = None,
    finish_reason: Any = None,
    retry_attempt: bool = False,
    public_prompt_context: Optional[str] = None,
) -> None:
    """Record one generation-attempt row into the active collector.

    Raw prompts / headers / private content are never persisted. Response
    content is passed only to the shared refusal classifier; explicit refusal
    records may also retain one bounded public opposite-side excerpt. The
    record keeps bounded refusal and provider moderation metadata.
    ``status``/``error_type`` record provider/runtime failures, including a
    row when the provider call raises.
    """
    from tau2.utils.llm_call_metrics import (
        LLMCallRecord,
        detect_model_refusal,
        get_llm_call_metrics_collector,
        preceding_public_prompt,
    )

    collector = get_llm_call_metrics_collector()
    if collector is None:
        return
    # messages_chars = serialized message payload (system + conversation).
    messages_chars = len(json.dumps(litellm_messages))
    system_chars = sum(
        len(json.dumps(m)) for m in litellm_messages if m.get("role") == "system"
    )
    conversation_chars = messages_chars - system_chars
    tool_schema_chars = len(json.dumps(tools_schema)) if tools_schema else 0
    # total_input_chars = every serialized major input component together.
    total_input_chars = messages_chars + tool_schema_chars
    output_contract_chars = len(output_contract_text) if output_contract_text else None
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None
    total_tokens = (
        prompt_tokens + completion_tokens
        if prompt_tokens is not None and completion_tokens is not None
        else None
    )
    refusal = detect_model_refusal(
        response_content,
        raw_data=response_raw_data,
        provider_refusal=provider_refusal,
        finish_reason=finish_reason,
    )
    provider_info = refusal["provider_metadata"]
    public_prompt = (
        preceding_public_prompt(
            messages,
            side,
            public_prompt_context=public_prompt_context,
        )
        if refusal["explicit_refusal"]
        else None
    )
    collector.record(
        LLMCallRecord(
            side=side or "unspecified",
            call_index=0,  # assigned by the collector
            model=model,
            provider=model.split("/", 1)[0] if model else None,
            message_count=len(messages),
            messages_chars=messages_chars,
            system_chars=system_chars,
            conversation_chars=conversation_chars,
            tool_schema_chars=tool_schema_chars,
            total_input_chars=total_input_chars,
            status=status,
            error_type=error_type,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            output_contract_chars=output_contract_chars,
            latency_seconds=latency_seconds,
            trigger=trigger,
            call_name=call_name,
            retry_attempt=retry_attempt,
            explicit_refusal=refusal["explicit_refusal"],
            refusal_excerpt=refusal["refusal_excerpt"],
            provider_refusal=refusal["provider_refusal"],
            finish_reason=provider_info["finish_reason"],
            moderation_metadata=provider_info,
            preceding_public_prompt=public_prompt,
        )
    )


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: Sequence[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in (message.tool_calls or [])
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def validate_message(message: Message) -> None:
    """
    Validate the message.
    """

    def has_text_content(message: SystemMessage) -> bool:
        """
        Check if the message has text content.
        """
        return message.content is not None and bool(message.content.strip())

    def has_content_or_tool_calls(message: ParticipantMessageBase) -> bool:
        """
        Check if the message has content or tool calls.
        """
        return message.has_content() or message.is_tool_call()

    if isinstance(message, SystemMessage):
        assert has_text_content(message), (
            f"System message must have content. got {message}"
        )
    if isinstance(message, ParticipantMessageBase):
        assert has_content_or_tool_calls(message), (
            f"Message must have content or tool calls. got {message}"
        )


def validate_message_history(messages: Sequence[Message]) -> None:
    """
    Validate the message history.
    """
    for message in messages:
        validate_message(message)


def set_llm_log_dir(log_dir: Optional[Path | str]) -> None:
    """
    Set the directory where LLM debug logs should be written.

    Args:
        log_dir: Path to the directory where logs should be saved, or None to disable file logging
    """
    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    llm_log_dir.set(log_dir)


def set_llm_log_mode(mode: str) -> None:
    """
    Set the LLM debug logging mode.

    Args:
        mode: Logging mode - "all" to save every LLM call, "latest" to keep only the most recent call of each type
    """
    if mode not in ("all", "latest"):
        raise ValueError(f"Invalid LLM log mode: {mode}. Must be 'all' or 'latest'")
    llm_log_mode.set(mode)


def _format_messages_for_logging(messages: list[dict]) -> list[dict]:
    """
    Format messages for debug logging by splitting content on newlines.

    Args:
        messages: List of litellm message dictionaries

    Returns:
        Modified message list with content split into lines for readability
    """
    formatted = []
    for msg in messages:
        msg_copy = msg.copy()
        if "content" in msg_copy and isinstance(msg_copy["content"], str):
            # Split content on newlines for better readability
            content_lines = msg_copy["content"].split("\n")
            if len(content_lines) > 1:
                msg_copy["content"] = content_lines
        formatted.append(msg_copy)
    return formatted


def _write_llm_log(
    request_data: dict, response_data: dict, call_name: Optional[str] = None
) -> None:
    """
    Write LLM call log to file if a log directory is set.
    Behavior depends on the current log mode:
    - "all": Saves every LLM call
    - "latest": Only keeps the most recent call of each call_name type

    Args:
        request_data: Dictionary containing request information
        response_data: Dictionary containing response information
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message")
    """
    log_dir = llm_log_dir.get()

    if log_dir is None:
        # No log directory set, skip logging
        return

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get current logging mode
    current_log_mode = llm_log_mode.get()

    # If mode is "latest" and call_name is provided, remove existing files with the same call_name
    if current_log_mode == "latest" and call_name:
        # Find and remove existing files with this call_name
        pattern = f"*_{call_name}_*.json"
        existing_files = list(log_dir.glob(pattern))
        for existing_file in existing_files:
            try:
                existing_file.unlink()
            except FileNotFoundError:
                # File might have been removed by another thread; continue
                continue

    # Create a new file for this LLM call
    call_id = str(uuid.uuid4())[:8]  # Use short UUID for readability
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds

    # Include call_name in filename if provided
    if call_name:
        log_file = log_dir / f"{timestamp}_{call_name}_{call_id}.json"
    else:
        log_file = log_dir / f"{timestamp}_{call_id}.json"

    # Create complete JSON structure with both request and response
    call_data = {
        "call_id": call_id,
        "call_name": call_name,
        "timestamp": datetime.now().isoformat(),
        "request": request_data,
        "response": response_data,
    }

    # Write to file with indentation. Logging must never turn a successful
    # provider generation into a failed simulation.
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(call_data, f, indent=2)
    except OSError as exc:
        logger.warning("Unable to write LLM debug log: {}", exc)


def generate(
    model: str,
    messages: Sequence[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    call_name: Optional[str] = None,
    side: Optional[str] = None,
    trigger: Optional[str] = None,
    output_contract_text: Optional[str] = None,
    retry_attempt: bool = False,
    public_prompt_context: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        call_name: Optional name identifying the purpose of this LLM call
                   (e.g., "detect_interrupt", "generate_agent_message").
                   Used for logging and debugging.
        side: Optional caller side (e.g. "agent" / "stakeholder") for
                   context-size/latency measurement. Never inferred from
                   message text.
        trigger: Optional Agent-side classification of the input that
                   triggered this generation (e.g. "stakeholder_message" /
                   "tool_result" / "multi_tool_result" / "initial_turn" /
                   "other"). Recorded into the metrics row when a collector
                   is active; never influences the provider call.
        output_contract_text: Optional stakeholder output-sidecar contract
                   text (the fixed block appended to every stakeholder
                   request). Only its length is recorded as
                   ``output_contract_chars``; the text itself never persists.
        retry_attempt: Whether this generate invocation is a caller-visible
                   retry of the same logical request. Provider-internal retry
                   attempts are not exposed as separate generate rows.
        public_prompt_context: Optional bounded public Agent utterance for a
                   Stakeholder call. It is used only for explicit refusal
                   diagnostics and is never sent to the provider or logged.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.

    Notes:
        A metrics row is recorded for EVERY provider generation attempt,
        including failures (``status="error"`` with a safe ``error_type``
        exception class name; ``latency_seconds`` covers the time spent
        before the failure). Exception messages are never persisted because
        they may contain request/provider content.
    """
    validate_message_history(messages)
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    requested_model = model
    provider_model = resolve_openrouter_model(
        model,
        explicit_api_key=kwargs.get("api_key"),
    )

    # Vertex AI Gemini 3 models require VERTEXAI_LOCATION="global"
    if provider_model.startswith("vertex_ai/gemini-3") and not os.environ.get(
        "VERTEXAI_LOCATION"
    ):
        os.environ["VERTEXAI_LOCATION"] = "global"

    litellm_messages = to_litellm_messages(messages)
    tools_schema = [tool.openai_schema for tool in tools] if tools else None
    if tools_schema and tool_choice is None:
        tool_choice = "auto"

    # Prepare request data for logging
    formatted_messages = _format_messages_for_logging(litellm_messages)
    request_data = {
        "model": provider_model,
        "requested_model": requested_model,
        "messages": formatted_messages,
        "tools": tools_schema,
        "tool_choice": tool_choice,
        "kwargs": {
            k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
            for k, v in kwargs.items()
        },
    }
    request_timestamp = datetime.now().isoformat()

    start_time = time.perf_counter()
    try:
        response = completion(
            model=provider_model,
            messages=litellm_messages,
            tools=tools_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        # Record the failed attempt (latency includes time before the
        # failure). Only the exception CLASS name is kept — the message can
        # carry provider/request content and must not be persisted.
        failed_latency = time.perf_counter() - start_time
        _record_llm_call_metrics(
            side=side,
            model=provider_model,
            messages=messages,
            litellm_messages=litellm_messages,
            tools_schema=tools_schema,
            usage=None,
            latency_seconds=failed_latency,
            status="error",
            error_type=type(e).__name__,
            output_contract_text=output_contract_text,
            trigger=trigger,
            call_name=call_name,
            retry_attempt=retry_attempt,
            public_prompt_context=public_prompt_context,
        )
        logger.error(e)
        raise e
    generation_time_seconds = time.perf_counter() - start_time
    cost = get_response_cost(response)
    usage = get_response_usage(response)

    response_choice = response.choices[0]
    try:
        finish_reason = response_choice.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response_choice.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response_choice.message.content
    provider_refusal = getattr(response_choice.message, "refusal", None)
    raw_response_data = response.to_dict()
    raw_tool_calls = response_choice.message.tool_calls or []
    tool_calls = []
    for tool_call in raw_tool_calls:
        tool_name = str(tool_call.function.name or "")
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            # RECOVERABLE: malformed tool-call JSON must not abort the run.
            # Keep the tool name, drop the malformed arguments and surface a
            # concise parse error through the ordinary tool-error path (the
            # agent consumes its error budget and may retry). Semantic
            # content is never silently repaired.
            logger.warning(
                "Malformed tool-call arguments JSON for %r: %s",
                tool_call.function.name,
                str(exc),
            )
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_name,
                    arguments={},
                    requestor="assistant",
                    parse_error=(
                        f"malformed tool-call arguments JSON for "
                        f"{tool_name!r}: {str(exc)}"
                    ),
                )
            )
            continue
        if not isinstance(arguments, dict):
            # RECOVERABLE: the JSON parses but ToolCall arguments must be a
            # JSON object. Keep the tool name, drop the invalid arguments and
            # surface a concise validation error through the ordinary
            # tool-error path (the agent consumes its error budget and may
            # retry). Semantic content is never silently repaired.
            logger.warning(
                "Tool-call arguments for %r must be a JSON object, got %s",
                tool_call.function.name,
                type(arguments).__name__,
            )
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_name,
                    arguments={},
                    requestor="assistant",
                    parse_error=(
                        f"tool-call arguments for {tool_name!r} must be a "
                        f"JSON object, got {type(arguments).__name__}"
                    ),
                )
            )
            continue
        tool_calls.append(
            ToolCall(
                id=tool_call.id,
                name=tool_name,
                arguments=arguments,
                requestor="assistant",
            )
        )
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=raw_response_data,
        generation_time_seconds=generation_time_seconds,
    )

    # Log complete LLM call (request + response)
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        "cost": cost,
        "usage": usage,
        "generation_time_seconds": generation_time_seconds,
    }
    # Add timestamp to request data
    request_data["timestamp"] = request_timestamp
    _write_llm_log(request_data, response_data, call_name=call_name)

    _record_llm_call_metrics(
        side=side,
        model=provider_model,
        messages=messages,
        litellm_messages=litellm_messages,
        tools_schema=tools_schema,
        usage=usage,
        latency_seconds=generation_time_seconds,
        status="success",
        error_type=None,
        output_contract_text=output_contract_text,
        trigger=trigger,
        call_name=call_name,
        response_content=content,
        response_raw_data=raw_response_data,
        provider_refusal=provider_refusal,
        finish_reason=finish_reason,
        retry_attempt=retry_attempt,
        public_prompt_context=public_prompt_context,
    )

    return message


def get_cost(messages: list[Message]) -> tuple[float | None, float | None]:
    """
    Get the (agent_cost, user_cost) of the interaction.

    Each side is computed independently: a side is None if any of its
    messages has no cost. This way an uncosted agent message (e.g. an
    audio-native provider without usage reporting) doesn't discard the
    user side's cost, and vice versa.
    """
    agent_cost: float | None = 0.0
    user_cost: float | None = 0.0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AssistantMessage):
            if message.cost is None:
                logger.warning(f"Agent message has no cost: {message.content}")
                agent_cost = None
            elif agent_cost is not None:
                agent_cost += message.cost
        elif isinstance(message, UserMessage):
            if message.cost is None:
                logger.warning(f"User message has no cost: {message.content}")
                user_cost = None
            elif user_cost is not None:
                user_cost += message.cost
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if not isinstance(message, (AssistantMessage, UserMessage)):
            continue
        message_usage = getattr(message, "usage", None)
        if message_usage is None:
            logger.warning(
                f"Message {message.role}: {getattr(message, 'content', None)} "
                "has no usage"
            )
            continue
        usage["completion_tokens"] += message_usage["completion_tokens"]
        usage["prompt_tokens"] += message_usage["prompt_tokens"]
    return usage


def extract_json_from_llm_response(response: str) -> str:
    """
    Extract JSON from an LLM response, handling markdown code blocks.
    """
    # Try to extract JSON from markdown code blocks
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content between first { and last }
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    # Return original response as fallback
    return response
