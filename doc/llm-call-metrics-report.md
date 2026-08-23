# business_interview: LLM call metrics and refusal diagnostics

## Purpose

Instrument the business_interview run so every Agent/Stakeholder
`generate()` attempt records one row with context-size/latency metrics and
bounded generation diagnostics. The smoke artifact summarizes both sides and
keeps explicit model refusals from internal Stakeholder plan/realization calls,
including calls whose later retry produces the accepted public answer. This is
not a prompt, evaluator, or Truth-reconstruction redesign.

## Metrics schema (one row per LLM call)

| field                | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| side                | agent | stakeholder (explicit at call site, never inferred)        |
| call_index          | monotonic per run                                               |
| attempt_index       | per-side, per-call-name generation sequence                     |
| call_name           | logical generation purpose (plan / realization / agent response) |
| model               | provider model id                                               |
| provider            | model provider prefix when available                             |
| message_count       | number of messages in the request                               |
| messages_chars      | serialized message payload chars (system + conversation)        |
| system_chars        | fixed system/instructions portion                              |
| conversation_chars  | accumulated conversation/history portion (excludes system+tools)|
| tool_schema_chars   | serialized tool schemas; 0 when no tools; separate component    |
| total_input_chars   | messages + tool schemas (+ any other serialized major component)|
| status               | `success` | `error` (a row is recorded even when generation raises)|
| error_type           | safe exception class name (or None); never the message          |
| prompt_tokens        | EXACT provider usage or null                                  |
| completion_tokens    | EXACT provider usage or null                                  |
| total_tokens         | prompt+completion when both known, else null                  |
| output_contract_chars | stakeholder output-sidecar contract length (separate; NULL for agent)|
| latency_seconds      | wall-clock time around the provider `completion()` call (incl. failures)|
| retry_attempt       | caller marked this generation as a retry                            |
| explicit_refusal    | explicit text-pattern or provider `message.refusal` evidence         |
| refusal_excerpt     | bounded refusal response excerpt, when present                       |
| provider_refusal    | bounded provider refusal field, when present                         |
| finish_reason       | provider finish reason                                                |
| moderation_metadata | bounded moderation/content-filter metadata and key summary           |
| preceding_public_prompt | bounded opposite-side public message, only for explicit refusals |

- `latency_seconds` = wall-clock time around the provider `completion()` call,
  including the time spent before a failure.
- Failed provider calls still produce a row: `status="error"` with
  `error_type` = exception class name (safe; the exception *message* is never
  persisted because it may contain provider/request content).
- Token counts come ONLY from `usage` in the provider response; missing usage
  → null (never invented).
- `conversation_chars` excludes system instructions and tool schemas.
- The earlier `request_chars` naming is GONE: the message-only payload is
  `messages_chars`, and the all-inclusive serialized input is
  `total_input_chars`.
- no raw prompts / headers / secrets / private semantic IDs ever persisted;
  refusal excerpts are the intentional bounded response diagnostic exception.

## Refusal classification scope

The shared detector in `llm_call_metrics.py` is used both by call-level rows
and the compatibility `account_model_refusals()` public-trajectory diagnostic.
Only explicit refusal evidence counts:

- `I don't know` / `I'm not sure`, including generic apologies followed by
  uncertainty, is **not** a refusal;
- malformed JSON, tool-argument parsing, sidecar validation, and provider or
  network exceptions are **not** refusals without explicit refusal evidence;
- explicit text such as `I can't assist with that request`, or a provider
  `message.refusal` field, is a refusal.

The real-run artifact's `model_refusal_count` / `model_refusals` are derived
only from the complete generation-attempt records. The optional
`public_trajectory_refusal_count` is secondary and is never added to the
call-level count, so an accepted public message cannot double-count its own
underlying generation and internal failed/retried calls remain visible.

For an explicit refusal only, `preceding_public_prompt` contains the latest
public opposite-side participant message, bounded to 500 characters. It is
selected by participant role rather than by taking the last raw prompt, because
Stakeholder requests append private plan/realization contracts. System
messages, hidden knowledge, contracts, sidecars, tool messages, and private
IDs are never selected; ordinary non-refusal rows store `null`.

## Call-trigger classification (Agent side)

Every Agent generation records the input kind that caused it (`trigger`):

- `initial_turn` — the first generation with no preceding input
- `stakeholder_message` — the Agent is answering a fresh stakeholder message
- `tool_result` — one tool result came back to the Agent
- `multi_tool_result` — several tool results returned together
- `other` — anything else

`summarize_records` reports `trigger_counts` per side. This is used to quantify
why the Agent calls the LLM and to verify round-trip reduction (fewer
`tool_result` / `multi_tool_result` generations; more batched tool calls per
generation).

## Implementation

- `src/tau2/utils/llm_call_metrics.py` — `LLMCallRecord`, `LLMCallMetricsCollector`
  (+ contextvar install/get), `summarize_records`, `slowest_calls`, `record_to_dict`.
- `src/tau2/utils/llm_utils.py` — `generate()` records into the active collector
  (`_record_llm_call_metrics`), including on provider failure; accepts `trigger`
  (Agent call-trigger), `call_name`, `retry_attempt`, and
  `output_contract_text` (stakeholder sidecar contract, measured as a length
  only), and classifies the provider response with the shared refusal detector.
- `src/tau2/agent/llm_agent.py` — agent call sites pass `side="agent"` and the
  `trigger` classification.
- `src/tau2/domains/business_interview/user_simulator.py` —
  stakeholder plan/realization call sites pass `side="stakeholder"`, distinct
  `call_name`s, `output_contract_text`, and retry metadata.
- `scripts/business_interview_real_llm_smoke.py` — installs the collector before
  the run and dumps `dump["llm_call_metrics"]` = {records, by_side_summary,
  slowest_calls} into the per-run artifact. The by-side summary now includes
  `trigger_counts`, `success`/`error`/`error_types`.

## Side summary (per side: agent / stakeholder)

`calls`, `success`, `error`, `error_types`, `max_prompt_tokens`,
`max_messages_chars`, `max_total_input_chars`, `max_message_count`,
`total_prompt_tokens`, `total_completion_tokens`, `total_input_chars`,
`latency_p50`, `latency_p95`, `latency_max`, `token_usage_calls` (exact-token
coverage), `trigger_counts`, `retry_attempts`, `explicit_refusal_count`.
Exact token totals aggregate ONLY over calls with exact usage.

## Tests

`tests/test_llm_call_metrics.py` (deterministic, no network — provider
`completion` monkeypatched with fake responses):

- agent/stakeholder classification; latency recorded;
- exact provider usage preserved; missing usage stays null;
- component counts deterministic; tool schema separate from messages;
  `total_input_chars = messages_chars + tool_schema_chars`; message payload is
  NOT named `request_chars`;
- stakeholder output_contract_chars measured, contract body not persisted;
- failed provider call records an error row (status/error_type, char components,
  latency); safe class name only, exception message never stored;
- records contain no raw prompt / private semantic ids;
- refusal patterns, provider refusal fields, retry flags, and bounded metadata
  are recorded without classifying uncertainty or provider exceptions;
- summary max/p50/p95, nullable-token aggregation, success/error/error_types,
  retry/refusal counts, and trigger_counts correct;
- slowest-calls ordering; call/attempt indexes monotonic;
- an internal Stakeholder refusal followed by a successful retry remains one
  call-level refusal while the accepted public uncertainty response has zero
  public-trajectory refusals;
- explicit refusals retain only bounded public opposite-side context; private
  system/contract text is absent, Agent context is captured, and normal
  uncertainty calls retain a null context.

`tests/test_business_interview_roundtrips.py` (deterministic, real
business_interview env, no LLM):

- independent tool calls in one Agent message execute as ONE batch (concepts
  created; one tool-call turn);
- dependent tool calls cannot bypass a required result (a node cannot reference
  a missing concept; no behind-the-back auto-creation);
- merged observation acquisition `observe_latest_stakeholder_message` returns
  the Observation id directly (no separate observe_message), preserving exact
  ids / text / order / turn binding (provenance).

`tests/test_domains/test_business_interview/` passes (104), including the
`remove_edge` policy/schema and graph-revision regressions.

## One real measurement run

`scripts/business_interview_real_llm_smoke.py` with the existing DeepSeek V4
Flash (OpenRouter) config. Artifact includes the per-side metrics table and the
Agent trigger classification.
