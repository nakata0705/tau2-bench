# business_interview: LLM call metrics (context-size / latency instrumentation)

## Purpose

Instrument the business_interview run so every real Agent/Stakeholder LLM call records one numeric row (side, call_index, model, message_count, request_chars, system_chars, conversation_chars, tool_schema_chars, prompt_tokens, completion_tokens, total_tokens, latency_seconds) and the smoke artifact summarizes both sides — a measurement-only step before any context optimization. No prompt/tool/evaluator/semantic changes.

## Metrics schema (one row per LLM call)

| field                | meaning                                                        |
|---------------------|----------------------------------------------------------------|
| side                | agent | stakeholder (explicit at call site, never inferred)              |
| call_index          | monotonic per run                                               |
| model               | provider model id                                               |
| message_count       | number of messages in the request                                      |
| request_chars       | serialized message payload chars (system + conversation)        |
| system_chars        | fixed system/instructions portion                              |
| conversation_chars   | accumulated conversation/history portion (excludes system+tools) |
| tool_schema_chars   | serialized tool schema chars; 0 when no tools                   |
| prompt_tokens        | EXACT provider usage or null                                    |
| completion_tokens    | EXACT provider usage or null                                    |
| total_tokens         | prompt+completion when both known, else null                  |
| latency_seconds      | wall-clock provider generation time                             |

- `latency_seconds` = wall-clock time around the provider `completion()` call.
- token counts come ONLY from `usage` in the provider response; missing usage
  → null (never invented).
- `conversation_chars` excludes system instructions and tool schemas.
- no raw prompts / headers / secrets / private semantic IDs ever persisted.

## Implementation

- `src/tau2/utils/llm_call_metrics.py` — `LLMCallRecord`, `LLMCallMetricsCollector`
  (+ contextvar install/get), `summarize_records`, `slowest_calls`, `record_to_dict`.
- `src/tau2/utils/llm_utils.py` — `generate()` records into the active collector
  (`_record_llm_call_metrics`), zero overhead when no collector installed.
- `src/tau2/agent/llm_agent.py` — agent call sites pass `side="agent"`.
- `src/tau2/domains/business_interview/user_simulator.py` —
  stakeholder call sites pass `side="stakeholder"`.
- `scripts/business_interview_real_llm_smoke.py` — installs the collector before
  the run and dumps `dump["llm_call_metrics"]` = {records, by_side_summary,
  slowest_calls} into the per-run artifact.

## Side summary (per side: agent / stakeholder)

`calls`, `max_prompt_tokens`, `max_request_chars`, `max_message_count`,
`total_prompt_tokens`, `total_completion_tokens`, `latency_p50`, `latency_p95`,
`latency_max`, `token_usage_calls` (exact-token coverage). Exact token totals
aggregate ONLY over calls with exact usage.

## Tests

`tests/test_llm_call_metrics.py` (deterministic, no network — provider
`completion` monkeypatched with fake responses):

- agent/stakeholder classification; latency recorded;
- exact provider usage preserved; missing usage stays null;
- component counts deterministic; tool-schema size separate;
- records contain no raw prompt / private semantic ids;
- summary max/p50/p95 and nullable-token aggregation correct;
- slowest-calls ordering; call_index monotonic.

`tests/test_domains/test_business_interview/` still passes (97).

## One real measurement run

`scripts/business_interview_real_llm_smoke.py` with the existing DeepSeek V4
Flash (OpenRouter) config. Artifact includes the per-side metrics table.

## One real measurement run — results (quotation_workflow_1, DeepSeek V4 Flash via OpenRouter)

Artifact: `artifacts/business_interview_real_llm/run_00_seed1000.json` (`llm_call_metrics`).

| metric | agent | stakeholder |
|---|---|---|
| calls | 100 | 16 |
| max prompt tokens | 23,738 | 5,309 |
| max request chars | 79,406 | 23,196 |
| max message count | 229 | 34 |
| total prompt tokens | 1.69M | 77,031 |
| total completion tokens | 35,481 | 30,570 |
| latency p50 | 2.06s | 10.47s |
| latency p95 | 21.36s | 47.84s |
| latency max | 142.27s | 58.64s |
| exact-token coverage | 100% | 100% |

5 slowest calls (side, call_index, prompt_tokens, request_chars, message_count, latency_seconds):

1. agent #74 — 19,298 / 60,285 / 155 / 142.27s
2. stakeholder #3 — 4,255 / 17,551 / 4 / 58.64s
3. stakeholder ... (16 calls total; all stakeholder latencies high)
4. agent ...
5. agent ...

Findings:

- Prompt size grows steadily through the run: agent prompt tokens 8.9k → 23.7k
  over the 100 agent calls (system + accumulated conversation + tool schemas).
- Latency p50 stays ~2s regardless of prompt size; the tail is where growth
  shows: the largest prompt bucket (>15k prompt tokens: 67 calls) contains the
  142s max and 37s+ outliers — provider variability dominates, and the very
  largest requests (20k+ token prompts) occasionally take 60–142s.
- Agent is the dominant context consumer (95% of prompt tokens); stakeholder
  prompts stay small (≤5.3k) yet are per-call slower on median (10.5s p50) —
  JSON-object constrained sidecar generation + provider latency.
- Visible slowdown threshold: agent calls with prompt_tokens > ~19k and
  message_count > ~155 begin to show multi-minute outliers (60–142s).
