# Session summary for investigating the Luna rejection

Date/context: continuation of the `tau2-bench` coding session. No secrets or API keys are included here.

## User request that was in progress

The active follow-up objective was to extend the `business-interview` benchmark with:

1. A runtime/orchestrator guard for repeated successful Agent write-tool operations that make no meaningful progress. It must catch repeated identical writes and short cycles such as `A B A B ...`, including the seed-9003 `UNSET <-> DONT_KNOW` rationale loop, without using Truth or semantic LLM matching.
2. Bounded public-context excerpts on call-level refusal records. A Stakeholder refusal should retain the preceding public Agent question; an Agent refusal should retain the preceding public Stakeholder message where safe. Hidden system prompts, StakeholderKnowledge, contracts, sidecars, and private IDs must never be persisted. Non-refusal calls should not persist context excerpts.

The completed Truth-reconstruction evaluator, provenance rules, `remove_edge`, and two-phase Stakeholder architecture must remain unchanged.

## Repository state around the rejection

- Repository: `/home/nakata0705/Projects/tau2-bench`
- Branch: `business-interview`
- HEAD at the time of the rejected operation: `da3eef4c76db1b15330e9a04f42e244c4eae1498`
- The preceding commit was `fix: support edge revision and call-level refusal diagnostics`, and it had already been pushed.
- The rejection was observed during a later coding-agent continuation; subsequent fixes were committed separately. This document records the evidence for the rejection cause, not a claim that the transient worktree state was itself the cause.

## What was inspected

Relevant existing code:

- `src/tau2/orchestrator/orchestrator.py`
  - `BaseOrchestrator` already has question/response/interaction loop guards.
  - `loop_guard_diagnostics` is persisted as `SimulationRun.info["loop_guard"]`.
  - Tool calls flow through `_execute_tool_calls()`.
  - Half-duplex `Orchestrator._check_termination()` checks loop guards before `max_steps`.
- `src/tau2/orchestrator/full_duplex_orchestrator.py`
  - Inherits `_execute_tool_calls()` but has its own termination check.
- `src/tau2/data_model/simulation.py`
  - `TerminationReason` currently includes `repeated_question`, `repeated_response`, and `stalled_interaction`, but not `stalled_tool_operation`.
- `src/tau2/data_model/message.py`
  - `ToolCall` has `name`, dict `arguments`, `requestor`, and `parse_error`.
  - `ToolMessage.error` identifies failed tool calls.
- `src/tau2/data_model/simulation.py` / `src/tau2/runner/build.py`
  - Existing repetition-guard settings are exposed through `BaseRunConfig` and threaded into text/voice orchestrators.
- `src/tau2/utils/llm_call_metrics.py`
  - `LLMCallRecord` already stores call-level refusal fields and `model_refusal_records()` returns refusal rows.
- `src/tau2/utils/llm_utils.py`
  - `_record_llm_call_metrics()` is the common instrumentation point.
- `src/tau2/agent/llm_agent.py`
  - Agent calls pass the public conversation plus system messages to `generate()`.
- `src/tau2/domains/business_interview/user_simulator.py`
  - Private plan/realization contracts are appended to the last user message, so context extraction must not simply persist the last raw prompt.
- `tests/test_loop_guards.py`
  - Existing deterministic scripted orchestrator guard tests and stub Agent/User helpers.
- `tests/test_llm_call_metrics.py`
  - Existing deterministic fake-provider refusal and retry tests.

## Evidence for the Luna rejection cause

The earlier session note attributed the stop to the coding harness/LSP message:

> `128 issue(s) must be fixed`

That was not the failure shown by the captured Pi screen. The direct screen evidence showed:

> `Codex error: Invalid prompt:`
> `your prompt was flagged as potentially violating our usage policy.`

The immediate rejection therefore came from the Codex/OpenAI input-policy layer. The diagnostic text about 128 issues was emitted in the coding workflow, but it is not evidence that the LSP diagnostics caused the request rejection.

## Follow-up context-injection experiment

Pi was using Codex through the Luna Max coding-agent workflow, with `pi-lens` context injection enabled when the rejection occurred. Running `/lens-context-toggle` disabled automatic lens context injection. The same `/goal` could then be completed and committed successfully.

This strongly suggests a false positive involving the composed/injected context, but the experiment did not isolate exactly which injected text triggered the filter. It does not prove that any particular pi-lens finding or excerpt was individually responsible.

This event is separate from benchmark `model_refusal_count`: it was a coding-agent Codex request rejection, not a DeepSeek Agent/Stakeholder refusal inside a `business_interview` real run.

## Implementation notes captured by this record

### Tool-operation guard

- Add `TerminationReason.STALLED_TOOL_OPERATION`.
- Add an optional `max_stalled_tool_operations` setting, likely defaulting to a small threshold (the design discussion favored detecting identical writes around 4 occurrences and a period-2 cycle around 6 operations; `0`/`None` should disable it).
- Record only successful Agent-requested mutating operations. Exclude read-only names/prefixes such as `list_*`, `validate_*`, `get_*`, and unknown non-mutation test tools.
- Build a canonical structural fingerprint from tool name + sorted arguments and retain only hashes in persisted diagnostics.
- Build a separate logical target key so `update_node` on different node/property targets breaks the candidate, while `update_node(node_id="x", unset=["necessity_rationale"])` and `update_node(node_id="x", necessity_rationale={"dont_know": true})` can be recognized as the same logical target with different state fingerprints.
- Reset the candidate history on a public Stakeholder response, target/property changes, and structural add/remove/create/merge/set operations.
- Persist only safe fields such as `type`, `reason`, `threshold`, `count`, `cycle_period`, `repetition_count`, `first_step`, `trigger_step`, and `fingerprint_hashes`.
- Thread the setting through `BaseRunConfig`, `build_text_orchestrator`, `build_voice_orchestrator`, `Orchestrator`, and `FullDuplexOrchestrator` as appropriate.

### Refusal public context

- `preceding_public_prompt` is populated only for explicit refusals.
- The Stakeholder path captures the bounded public Agent utterance before `UserState.flip_roles()` output is augmented with private plan/sidecar contracts, then passes it explicitly through `_generate_plan` / `_realize_sidecar` / `_call_llm` / `generate()` into call metrics.
- The common Agent path may continue deriving context from the ordinary public `UserMessage` conversation.
- System prompts, StakeholderKnowledge, contracts, sidecars, retry hints, raw provider prompts, hidden Truth, and private semantic IDs are never persisted.
- Deterministic tests cover the real role-flip path, retries, privacy, bounding, non-refusals, and Agent-side context.

## Prior completed verification (before this follow-up)

The previous commit had already passed:

- business-interview deterministic tests: 104 passed
- business-interview roundtrip tests: 16 passed
- LLM metrics tests: 18 passed
- Ruff check/format
- compile/static/LSP checks
- one fresh real run with seed 9003

Seed-9003 prior result: `max_steps`, 100 Agent calls, 9 accepted Observations, 17 Stakeholder generation attempts (one retry), provider/tool errors 0, call-level refusals 0, node and edge recall/precision 1.0/1.0, concept recall/precision 0.7619/0.9412, fabricated nodes/edges 0/0, reconstruction/structural/quality all false, elapsed 491.34 seconds. The observed issue was repeated rationale tool-state toggling without a new public question.

## Follow-up status

- The stalled successful-tool-operation guard was implemented in a separate commit and is treated as accepted; its thresholds and algorithm are not part of the Luna rejection diagnosis.
- The remaining refusal-diagnostic cleanup is deterministic: it must preserve only safe public context and does not require another live DeepSeek quotation run.
- The corrected rejection evidence and the context-injection experiment are recorded above; neither should be conflated with benchmark `model_refusal_count`.
