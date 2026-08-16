# business_interview: Observation Authenticity — Completion Report

## Goal summary

Change `business_interview`'s Observation from "agent free-text notes" to
**authentic primary evidence derived from actual stakeholder conversation
messages**. The agent can no longer write Observation text / source / turn; it
captures real stakeholder (user) messages via `observe_turn` and references the
returned Observation ids as provenance. The evaluator gates `quality_pass` on
**authentic** provenance (an Observation must trace to a real stakeholder
message), not merely on "the id exists".

Core contract (now enforced):

```
Actual stakeholder message -> environment-controlled Observation -> agent references Observation -> InferredValue / Node / Edge / Necessity
```

## Previous exploit

Previously the agent called `record_observation(text="...")` and could fabricate
any statement the stakeholder never said; the evaluator only checked that an
`observation_id` existed, not that the Observation derived from the conversation.
This let a "correct DAG + fabricated Observation ids" pass the evidence gate.

## Final Observation lifecycle

1. The environment ingests every conversation message into `InterviewDB.messages`
   (a `{role, content}` ledger) via a new `Environment.on_message` hook — called
   by `set_state` during replay/evaluation and by the half-duplex orchestrator
   during a live run (deterministic in both paths).
2. The agent calls `list_stakeholder_messages()` to see stakeholder (user)
   statements and their turn indices.
3. The agent calls `observe_turn(turn_idx)`. The tool validates the message
   exists and is a `user` (stakeholder) message, then idempotently creates an
   Observation with `text = message.content`, `source_id = "stakeholder"`,
   `turn = turn_idx`, `id = obs_{turn_idx}`.
4. The agent references the returned id as provenance on nodes / edges /
   necessity / attributes.
5. `start_inference` is destructive to the DAG and captured Observations but
   **keeps** the conversation ledger (observations are re-captured from it).

## Tool API changes

- **Removed** `record_observation(text, ...)` — no arbitrary-text Observation API.
- **Added** `observe_turn(turn_idx)` (WRITE, idempotent, validates role/existence).
- **Added** `list_stakeholder_messages()` (READ).
- `add_node` / `update_node` / `set_node_necessity` / `add_edge` / `update_edge` /
  `attach_observation` keep validating `observation_id` against the authentic
  `db.observations` (a nonexistent id is rejected).
- `start_inference` now resets DAG + captured Observations but preserves the
  conversation ledger.
- `update_edge(..., clear_predicate=...)` and `set_node_necessity(..., unset=[...])`
  semantics retained.

## Evaluator / authenticity changes

`evaluation.py` now:
- Verifies each Observation traces to a real user message: `obs.turn` must index
  a `user` message in `db.messages` whose content matches `obs.text`.
- Computes `authentic_observation_count`, `invalid_observation_source_count`,
  `orphan_observation_count`, `provenance_authenticity_pass`.
- Treats a provenance reference to a **nonexistent** Observation as invalid and a
  reference to a **non-authentic** Observation as non-authentic; both fail
  `evidence_pass` and `provenance_authenticity_pass`.
- `quality_pass = structural AND necessity AND evidence AND provenance_authenticity
  AND relevance`.
- Adds a lightweight, deterministic **relevance** gate: every node with asserted
  claims must have at least one evidence Observation sharing a concept signal (or
  word) with the node's action, so `"I like pizza."` cannot support
  `actor=sales`. This is deliberately conservative (documented limitation).

## Relevance check (present, limited)

Implemented as `relevance_pass` using the existing bilingual concept resolver /
aliases: a node's evidence must share a signal with the node's action. It only
rejects clearly-unrelated evidence; heavy paraphrase may pass/fail on raw word
overlap. Limitation noted in §limitations.

## Invariants

- Observation `text` / `source_id` / `turn` are environment-determined (from the
  actual message) — the agent cannot set them.
- No arbitrary-text Observation API exists.
- `Observation` is immutable (frozen).
- Same message observed twice is idempotent (same id, no unlimited duplicates).
- Assistant / tool / system messages cannot be captured as stakeholder evidence.
- Nonexistent turn/message references are rejected.
- Invalid observation references are rejected by the tools and counted by the
  evaluator.

## Tests added / changed

`tests/test_domains/test_business_interview/test_dag_business_interview.py`
(67 tests) now covers, in addition to the retained endpoint / evidence /
necessity / EN-JA / filter / leakage suite:

- no `record_observation` API exists
- valid stakeholder turn -> Observation created
- nonexistent turn rejected; assistant turn rejected
- duplicate capture idempotent
- Observation immutable
- fake Observation cannot be inserted through normal tools
- correct DAG + authentic provenance passes
- correct DAG + zero Observations fails
- correct DAG + fabricated (directly-injected) Observation fails
- unrelated authentic Observation attached to all claims fails relevance
- `start_inference` keeps the conversation ledger
- deterministic observation ids; duplicate provenance refs deduped
- end-to-end `EnvironmentEvaluator` reward 1.0 via an authentic reference
  conversation (`_reference_trajectory`), plus missing-node and
  fabricated-necessity failure paths.

## Verification commands / results

- `uv run pytest tests/test_domains/test_business_interview/` → **67 passed**.
- `make check-all` (ruff lint + format) → **All checks passed**.
- `make test` (core, incl. shared-core `on_message`/orchestrator changes) →
  **297 passed, 1 xpassed, 0 failed**.
- Reference trajectory through `EnvironmentEvaluator` → **reward 1.0** (unit
  test `test_evaluator_rewards_full_reconstruction`).
- Real-model smoke (DeepSeek): the live `on_message` hook + `observe_turn` path
  was verified directly (environment populates `db.messages` and captures
  stakeholder turns); the bundled weak agent did not finish a full run, which is
  not a completion requirement.

## Removed incompatible behavior

- `record_observation` (arbitrary-text Observation creation).
- Observations that are not traceable to a real stakeholder message.
- Evaluator accepting "the id exists" as sufficient provenance (now requires
  authenticity).
- `start_inference` clearing the conversation ledger.

## Remaining limitations

- **Relevance is lightweight** (concept-signal / word overlap with the node's
  action). It reliably rejects clearly-unrelated evidence but is not a semantic
  entailment check; a heavily paraphrased but genuinely-relevant statement could
  fail it. Authenticity is the hard gate; relevance is best-effort per the goal.
- The bundled `deepseek-chat` agent rarely calls the tool protocol, so real-run
  reward is low (weak-agent behavior, not a benchmark defect).
- `observe_turn` uses message **positions** (turn indices) which are stable only
  within a fixed conversation; filtering/reordering a trajectory changes them
  (the tests rebuild trajectories from scratch for that reason).
- The static `task.evaluation_criteria.actions` gold sequence uses illustrative
  `observe_turn` turn indices that are only reproducible in a matching reference
  conversation; the reward-1.0 reference is built in the test.

## Implementation commit SHA

See the git history for this change (committed and pushed to
`origin/business-interview`). The report itself is part of the same commit, so it
cannot reference its own SHA.
