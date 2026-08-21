# business_interview / runner: stop stalled conversation loops early

Goal: add deterministic conversation-loop guards so broken runs stop early
instead of consuming max_steps. This is a benchmark RUNTIME safeguard, not an
evaluator semantic feature. No LLM, embeddings, fuzzy similarity, or hidden
Truth is used to detect loops.

## Termination reasons

`TerminationReason` gained three dedicated values:

- `repeated_question` — same normalized conversational Agent question 3x
- `repeated_response` — same normalized stakeholder response 3x
- `stalled_interaction` — same (normalized question, semantic answer) 3x

These are never reported as `max_steps` / `too_many_errors`. The final
`SimulationRun` keeps the exact reason (`termination_reason`) and, when a
guard fires, `SimulationRun.info["loop_guard"]` carries
{type, threshold, count, fingerprint_hash, first_step, trigger_step}.

## Normalization (`src/tau2/utils/normalization.py`)

`normalize_text()` is a small shared normalizer used ONLY for loop
comparison: CRLF/CR -> LF, casefold, strip, collapse whitespace, optionally
strip trivial terminal punctuation (`.!?。！？`). Original text is always
preserved in artifacts. No semantic similarity, no synonyms, no stemming, no
LLMs — the purpose is catching effectively identical messages with cosmetic
formatting differences, never paraphrases.

## Generic orchestrator guards (`BaseOrchestrator` / half-duplex `step`)

- Agent **conversational** messages (non-tool, non-empty, non-solo) are
  normalized and counted; the third identical occurrence fires
  `repeated_question`. Repetitions need not be consecutive.
- Stakeholder natural-language responses (public text only) are normalized
  and counted; the third identical public response fires
  `repeated_response`. The third identical response is recorded in the
  trajectory, then the run terminates before another agent generation.
- Tool-only messages, empty content, private metadata, and internal env
  messages are ignored: they never affect the counters.
- Interaction guard: when the user implementation provides an optional
  private `interaction_signature(user_msg)` (e.g. the business_interview
  sidecar), the orchestrator pairs the normalized question with that
  signature; the third identical pair fires `stalled_interaction` even when
  surface wording differs slightly. The signature is NEVER exposed to the
  Agent (only a sha-256 hash is stored in diagnostics).
- Guards fire in `_check_termination` BEFORE max_steps/max_errors, so a
  clearly-repeating run terminates early with its own reason.
- Same question + different semantic answer (or same answer to different
  questions) is NOT a repeated interaction — legitimate clarification
  sequences never trigger `stalled_interaction` (in fact the pair includes
  the normalized question, so both sides must repeat).

## business_interview semantic fingerprint

`StakeholderUserSimulator.interaction_signature(user_msg)` derives a
deterministic private fingerprint from the message's private sidecar
annotations: sorted `(semantic_id, mode)` tuples. Two responses with the
same (semantic_id, mode) set are semantically the same answer; a different
set is a genuinely different answer. Returns `None` when no annotations
(greetings) so no interaction is claimed. This is the integration that uses
semantic mode once present; no NLP similarity substitute.

## Configuration

`BaseRunConfig` gained `max_repeated_questions` / `max_repeated_responses` /
`max_repeated_interactions` (default 3; `0`/`None` disables a guard),
threaded through `build_text_orchestrator` / `Orchestrator` /
`FullDuplexOrchestrator`. The business_interview smoke script enables all
three at 3.

## Retry separation

Stakeholder sidecar generation retries, malformed-tool-call retries, and
provider/API retries are untouched — a sidecar retry inside one generation
does not count as a public repeated response unless that response enters the
trajectory.

## Tests (`tests/test_loop_guards.py`)

15 deterministic tests (scripted stub participants, no LLM):

- same Agent question twice does NOT terminate;
- third identical Agent question → repeated_question;
- repetitions separated by other messages still count;
- cosmetic whitespace/case/punctuation normalize consistently;
- genuinely different questions do not collide;
- same stakeholder response twice does not terminate;
- third identical response → repeated_first_response;
- tool-only Agent messages do not affect counters;
- business_interview identical question + identical semantic sidecar 3x →
  stalled_interaction;
- same question with different semantic answers does not stall;
- same semantic answer to different questions does not stall;
- loop termination occurs BEFORE max_steps;
- guards disabled when 0 (max_steps still fires normally);
- Diagnostic hash preserved in finalized SimulationRun.info["loop_guard"].
- `test_normalize_whitespace_case_and_punctuation` covers the normalizer.

## Real-LLM verification

With the guards enabled (default 3) on `quotation_workflow_1`
(deepseek-v4-flash-0731 via OpenRouter, temperature 0):

- seed 6600: partial progress (node_recall 0.5, 0 marker errors, 0 leaks).
- seed 6800: **full reconstruction** — 6 nodes, 6 edges, 20 concepts,
  node_recall 1.0, node_precision 1.0, structurally valid, zero
  marker_evidence_errors, zero private-ID leakage; healthy run, no loop
  guard fired (`loop_guard: None`), termination via finish/stop.
- seed 6801 aborted early because deepseek-v4-flash stalled on a hung
  provider response (context-growth / provider latency), not a loop — the
  run log showed no repeated fingerprints before the hang.

Pathological repeated-response sessions (simulator failures) are caught by
the deterministic tests above at threshold 3; the third identical public
response is recorded in the trajectory and the run terminates before the
next agent generation, well before max_steps.
