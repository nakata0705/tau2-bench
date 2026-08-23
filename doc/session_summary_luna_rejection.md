# Session summary for investigating the Luna rejection

Date/context: continuation of the `tau2-bench` coding session. No secrets or API keys are included here.

## User request that was in progress

The active follow-up objective was to extend the `business-interview` benchmark with:

1. A runtime/orchestrator guard for repeated successful Agent write-tool operations that make no meaningful progress. It must catch repeated identical writes and short cycles such as `A B A B ...`, including the seed-9003 `UNSET <-> DONT_KNOW` rationale loop, without using Truth or semantic LLM matching.
2. Bounded public-context excerpts on call-level refusal records. A Stakeholder refusal should retain the preceding public Agent question; an Agent refusal should retain the preceding public Stakeholder message where safe. Hidden system prompts, StakeholderKnowledge, contracts, sidecars, and private IDs must never be persisted. Non-refusal calls should not persist context excerpts.

The completed Truth-reconstruction evaluator, provenance rules, `remove_edge`, and two-phase Stakeholder architecture must remain unchanged.

## Repository state before/after the rejection

- Repository: `/home/nakata0705/Projects/tau2-bench`
- Branch: `business-interview`
- HEAD and origin before this follow-up: `da3eef4c76db1b15330e9a04f42e244c4eae1498`
- Previous commit: `fix: support edge revision and call-level refusal diagnostics`
- That previous commit was already pushed successfully.
- Current worktree after the rejected operation:
  - `M doc/business-interview-real-llm-smoke-report.md` (pre-existing Markdown autofix from the previous continuation: table separator alignment and blank lines)
  - `M src/tau2/orchestrator/orchestrator.py` (the only code edit made during this follow-up so far)

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

## Code change made immediately before Luna rejected processing

An edit added only these module-level constants near the top of `src/tau2/orchestrator/orchestrator.py`:

```python
_MUTATING_TOOL_PREFIXES = (
    "add_", "create_", "delete_", "finish_", "mark_", "merge_",
    "record_", "remove_", "reset_", "set_", "update_",
)
_STRUCTURAL_TOOL_PREFIXES = (
    "add_", "create_", "delete_", "merge_", "remove_", "set_",
)
```

No other code for the new guard or refusal context was written yet.

## Luna/tool rejection output

The edit tool returned a STOP message claiming:

> `128 issue(s) must be fixed`

The displayed diagnostics were mostly type diagnostics in `orchestrator.py`, including:

- `L1113`: `Message | None` passed where `ValidUserInputMessage` is expected
- `L1149`: `Message | None` passed where `ValidAgentInputMessage` is expected
- `L1174`: possible missing `is_tool_call` on `SystemMessage`/`None`
- `L489`/`L495`: `object*` may not have `.stop`
- `L761`: `MultiToolMessage` may not have `turn_idx`
- `L780`, `L790`, `L804`: inferred tuple/list types incompatible with message types
- The output said `... and 118 more`

These diagnostics appeared after the tiny constants edit; the edit itself did not touch the reported lines. They may be pre-existing or a broad/stale LSP/pyright scan rather than errors caused by the constants, but this is the main failure to investigate. The exact warning was emitted by the coding harness, not by a provider API.

## Intended implementation plan (not yet completed)

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

- Add `preceding_public_prompt: Optional[str]` to `LLMCallRecord` and `record_to_dict()`.
- In `_record_llm_call_metrics()`, derive context only from the opposite participant's public-role messages (`AssistantMessage` for Stakeholder calls, `UserMessage` for Agent calls), bounded to the existing ~500-character limit.
- Attach the excerpt only when `explicit_refusal` is true; otherwise persist `null`.
- Do not scan/persist system messages, private contracts, sidecars, raw prompts, hidden knowledge, or private semantic IDs.
- Add deterministic tests for Stakeholder and Agent context, privacy, bounding, non-refusals, and existing internal retry behavior.

## Prior completed verification (before this follow-up)

The previous commit had already passed:

- business-interview deterministic tests: 104 passed
- business-interview roundtrip tests: 16 passed
- LLM metrics tests: 18 passed
- Ruff check/format
- compile/static/LSP checks
- one fresh real run with seed 9003

Seed-9003 prior result: `max_steps`, 100 Agent calls, 9 accepted Observations, 17 Stakeholder generation attempts (one retry), provider/tool errors 0, call-level refusals 0, node and edge recall/precision 1.0/1.0, concept recall/precision 0.7619/0.9412, fabricated nodes/edges 0/0, reconstruction/structural/quality all false, elapsed 491.34 seconds. The observed issue was repeated rationale tool-state toggling without a new public question.

## Next investigation steps

1. Re-read the current `orchestrator.py` and run targeted LSP diagnostics to determine whether the 128 diagnostics are pre-existing/stale and whether the small constants edit is syntactically safe.
2. Implement the guard and refusal-context changes incrementally, preferably in separate edits.
3. Extend deterministic tests and run the existing complete business-interview, loop-guard, roundtrip, and LLM-metrics suites.
4. Run Ruff and compile/static checks.
5. Run one fresh real quotation seed different from 9000-9003, inspect `loop_guard`, and report metrics accurately.
6. Update documentation, commit, and push only after verification.
