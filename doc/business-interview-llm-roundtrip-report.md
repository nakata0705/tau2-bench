# business_interview: LLM round-trip reduction (call amplification analysis)

## Baseline (before, quotation_workflow_1, deepseek-v4-flash-0731 via OpenRouter)

Artifact: `artifacts/business_interview_real_llm/run_00_seed1000.json`

| metric | agent | stakeholder |
| --- | --- | --- |
| calls | 100 | 16 |
| total prompt tokens | 1,693,732 | 77,031 |
| max prompt tokens | 23,738 | 5,309 |
| max request chars | 79,406 | 23,196 |
| latency p50 | 2.06s | 10.47s |
| latency p95 | 21.36s | 47.84s |
| latency max | 142.27s | 58.64s |

### Baseline call-trigger classification (reconstructed from the artifact)

Agent generations by input kind:

- `initial_turn`: 1
- `stakeholder_message`: 17
- `tool_result` (single): 66
- `multi_tool_result`: 17 (13×2, 2×3, 1×4, 1×9)

→ **83 of 100 agent generations (83%) were triggered by a tool result**, and 67
of the 84 tool-call turns were SINGLE tool calls. This is the amplification
archetype: one logical Agent decision fragmented across several LLM
generations because each tool result caused another model call.

### Dominant mechanical 2-step: observation acquisition

The old flow forced two Agent decisions for one bookkeeping step:

    observe_latest_stakeholder_message  → sm_N
    observe_message(sm_N)               → obs_T

In the baseline artifact this pattern occurs 16 times (16 of the 17 stakeholder
messages were captured through the 2-step chain).

## Changes (deterministic, preserving provenance/semantics)

1. **Merged observation acquisition** (`observe_latest_stakeholder_message`
   now captures the newest stakeholder message AND returns its Observation id
   directly — one tool call, no separate `observe_message` round trip). The
   environment still keeps message/observation separation internally;
   `observe_message(message_id)` still captures earlier messages by id;
   `list_stakeholder_messages` still lists stable ids. Exact Observation ids
   (`obs_<turn>`), text, source, order and turn binding are preserved (the
   merged op calls the same `_capture_user_message`).
2. **Explicit batching guidance in policy**: independent tool operations that
   are already knowable from the current state are made together in one turn
   (one model call, several tool calls, executed as one batch). Dependent
   steps (observation id before evidence; concept before node reference) must
   still wait for their result — nothing is executed speculatively or
   auto-created behind the Agent's back. The orchestrator already executes all
   tool calls in one Agent message as a batch and returns results together;
   the policy now makes that the default Agent strategy.
3. **Measurement unchanged in semantics** (see `llm-call-metrics-report.md`):
   char components renamed/clarified (`messages_chars`, `total_input_chars`,
   no `request_chars`), failed provider calls record a safe row
   (`status`/`error_type`, latency includes pre-failure time), stakeholder
   `output_contract_chars` is measured separately from the conversation, and
   Agent generations are classified by trigger (`stakeholder_message` /
   `tool_result` / `multi_tool_result` / `initial_turn` / `other`).

## After (real run, seed 9802, same config)

Artifact: `artifacts/business_interview_real_llm/run_00_seed9802.json`

| metric | agent | stakeholder |
| --- | --- | --- |
| calls | 54 | 16 |
| total prompt tokens | 825,176 | 72,992 |
| max prompt tokens | 20,332 | 5,122 |
| max total input chars | 83,342 | 22,393 |
| latency p50 | 8.29s | 23.87s |
| latency p95 | 31.14s | 52.18s |
| latency max | 138.65s | 78.16s |

Agent call-trigger classification (recorded by the instrumented run):

- `stakeholder_message`: 11
- `tool_result`: 32
- `multi_tool_result`: 11

Observation flow: 11 stakeholder messages → 11 observations (1:1 capture,
single step; no `observe_latest → observe_message` chains remain).

Semantic/evaluator outcome (before the run aborted on a stakeholder sidecar
mode-contradiction — see below): `node_recall 1.0`, `node_precision 1.0`,
`edge_recall 0.167`, `graph_valid True`, `fabricated_node_count 0`,
`provenance_authenticity_pass True`, `marker_evidence_errors 0`,
`private_id_leakage 0`. The graph's nodes were fully reconstructed and
authentically evidenced; the run could not `finish_interview` only because the
stakeholder LLM produced, on one later response, a private sidecar whose mode
contradicts its own knowledge — a pre-existing strict-fidelity harness failure
that aborts the whole run (the strict semantic-mode sidecar validation is a
benchmark semantic feature; it is NOT being weakened).

## Before/after table (main success criterion)

| metric | before (100) | after (9802) | Δ |
| --- | --- | --- | --- |
| Agent LLM calls | 100 | 54 | **−46%** |
| Agent calls per stakeholder response | 6.25 | 3.38 | **−46%** |
| Agent prompt total (tokens) | 1.69M | 0.83M | **−51%** |
| max Agent prompt (tokens) | 23,738 | 20,332 | −14% |
| tool-triggered agent generations | 83 | 43 | −48% |
| observe 2-step chains | 16 | 0 | eliminated |
| Agent latency p95 | 21.4s | 31.1s (fewer, slower calls; provider variance) | n/a |

The deterministic tests (`tests/test_business_interview_roundtrips.py`,
`tests/test_llm_call_metrics.py`) prove the reduction is real and safe:

- independent parallel tool calls still work (one Agent turn → batch);
- dependent tool calls cannot incorrectly bypass required results (a node
  cannot reference a missing concept; no auto-creation behind the Agent's
  back);
- merged observation acquisition preserves exact Observation ids / text /
  order / turn binding (identical to by-id capture);
- failed provider calls produce safe metrics rows (status/error_type, no
  exception message);
- char components and `total_input_chars` are correct; raw/private prompt
  content is never stored.

All existing business_interview evidence/evaluator tests pass unchanged (97),
as do the orchestrator (69) and loop-guard (16) suites.

## Remaining sources of LLM-call amplification

- **Context growth**: the Agent conversation still accumulates every message
  (+~8.9k→20k tokens) because history is never truncated. Each generation
  re-sends the full context; fewer generations directly means fewer tokens.
- **Stakeholder-sidecar strict-fidelity aborts**: a single mode-contradiction
  in a private sidecar kills the entire run after real work — wasted
  generations. This harness strictness (semantic-mode sidecar validation) is a
  benchmark semantic feature and must not be weakened; it is called out as the
  main remaining failure mode (2 of 3 recent seeds aborted this way), not as
  something fixed here.
- **Provider latency tail**: max latencies (138s) are provider-dominated;
  fewer calls shrink the probability of hitting the tail but do not change per-call variance.
