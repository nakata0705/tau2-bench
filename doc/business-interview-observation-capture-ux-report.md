# business_interview — Observation Capture UX Report

**Date:** 2026-08-18
**Branch:** `business-interview`
**Scope:** Observation capture tool UX only. Evaluator, stakeholder simulator,
necessity semantics, node matching, and scenario data were **not** changed.

---

## 1. The problem with the old `observe_turn(turn_idx)` API

The previous Observation-capture API was `observe_turn(turn_idx: int)`, where the
agent had to supply the **index of the message in the conversation ledger**. In
the half-duplex ledger, `assistant`, `user`, and `tool` messages are all appended
to the same `db.messages` list, so the "latest stakeholder statement" is not a
stable number — it depends on how many assistant/tool turns have been logged.

In the real DeepSeek smoke runs (seeds 1000–1004) the Interview Agent repeatedly
mis-computed `turn_idx`:

| run | tool errors | termination |
|-----|-------------|-------------|
| run_00 | 2 | user_stop |
| run_01 | 10 | **too_many_errors** |
| run_02 | 2 | user_stop |
| run_03 | 2 | user_stop |
| run_04 | 2 | user_stop |

Every error was `turn N is not a stakeholder (user) message; role=tool` (or
similar) — the agent was guessing integer ledger indices. Run 01 consumed its
error budget (`too_many_errors`) entirely on turn-index bookkeeping. This was
benchmark noise: the task measures **business-interview skill**, not the ability
to count messages in a shared ledger.

## 2. New message-ID design

Stakeholder (user) messages now carry **stable, unique ids** `sm_1, sm_2, ...`
derived deterministically from the append-only conversation ledger (the Nth user
message is always `sm_N`). The agent never deals with ledger integers.

### Tool API (in `src/tau2/domains/business_interview/tools.py`)

- **`observe_message(message_id="sm_3")`** *(WRITE)* — capture the stakeholder
  message with that stable id as an authentic Observation. Idempotent. Fabricated
  / nonexistent ids are rejected.
- **`observe_latest_stakeholder_message()`** *(READ)* — return the id of the most
  recent stakeholder message (e.g. `"sm_3"`), so the agent can act on the newest
  statement without tracking indices. It is a **read** (does not itself create an
  Observation), which makes it deterministic under state replay; the actual
  capture is done by `observe_message(message_id=...)` on the returned id.
- **`list_stakeholder_messages()`** *(READ)* — now lists statements keyed by
  stable id (`sm_1: ...`, `sm_2: ...`), not turn indices.

`observe_turn` is **removed**. There is no compatibility shim.

**Why `observe_latest_stakeholder_message` is a read.** The evaluator
reconstructs state by replaying the trajectory via `environment.set_state`,
which first loads *all* messages into the ledger and *then* replays mutating
tools. A capture-style "latest at call time" tool would capture the wrong message
under replay (it would always see the final message), producing non-deterministic
results. Making it a read (return the latest id) and capturing via the explicit
`observe_message(message_id)` keeps every state-changing call deterministic and
replay-safe.

### Internal mapping (`_stakeholder_entries`)

`InterviewDB` keeps the conversation ledger; `_stakeholder_entries()` walks it
and assigns `sm_N` to the Nth `user` message. Because the ledger is append-only
and ids derive from message role + position, the mapping is deterministic and
stable across `set_state` / replay.

## 3. How authenticity is preserved

- `observe_message` only accepts ids present in `_stakeholder_entries()`, i.e.
  **only actual user/stakeholder messages** can be observed. A fabricated id
  (`sm_99`, `sm_0`, `not_a_message`) is rejected with `ValueError`.
- The `Observation` is built from the **real ledger message** (`text`, `source_id="stakeholder"`,
  `turn` = actual ledger index), so its text/turn always match the recorded
  conversation.
- The observation id format is unchanged (`obs_{ledger_turn}`), so the evaluator's
  provenance-authenticity checks (`db.messages[o.turn]` role == user and content
  matches) are untouched.
- `observe_latest_stakeholder_message` returns an id that itself only ever refers
  to a user message, so it cannot leak assistant/tool content into observations.

## 4. Reference actions / trajectory updated

- `data/tau2/domains/business_interview/tasks.json` — all golden `observe_turn`
  actions in **quotation_workflow_1**, **quotation_workflow_1_ja**, and
  **lab_sample_flow** were converted to `observe_message(message_id="sm_K")`.
  No `observe_turn` remains.
- `tests/.../test_dag_business_interview.py::_reference_trajectory` now captures
  via `observe_message(message_id="sm_K")` with a tracked counter (deterministic,
  replay-safe).
- `_claim_obs` / `_ingest_and_observe` helpers use the new API.
- `data/.../policy.md` and `README.md` updated to describe the new API (no
  turn-index language).

## 5. Tests (business_interview suite: 51 passed)

New / updated tests:

- `test_list_stakeholder_messages_returns_stable_ids` — returns `sm_1`/`sm_2`,
  no `turn` in output.
- `test_observe_message_by_id_creates_correct_observation` — correct text, real
  ledger turn, idempotent.
- `test_invalid_and_fabricated_message_ids_rejected` — `sm_0`, `sm_2` (absent),
  `sm_99`, `turn_3`, `foo` rejected; latest with no messages rejected.
- `test_assistant_tool_messages_cannot_be_observed` — only the user message gets
  an `sm_` id; observing it yields the user text.
- `test_observe_latest_returns_newest_user_message_id` — returns the newest `sm_N`.
- `test_multiple_stakeholder_messages_ids_stable_and_unique` — ids stable/unique.
- `test_observation_capture_survives_set_state_replay` — replaying a conversation
  via `environment.set_state` restores the same sm ids and observations.
- `test_observation_authenticity_invariants` — updated for the new API
  (fabricated ids rejected, assistant not observable, idempotent).
- Existing evidence-authenticity, EN/JA, quotation reference trajectory
  (`reward == 1.0`), and lab scenario tests all still pass.

Full result: `pytest tests/test_domains/test_business_interview/` → **51 passed**.
`make check-all` (ruff lint + format) → clean.

## 6. Real DeepSeek smoke — before / after

Re-ran the manual smoke experiment (2 runs, seeds 2000–2001) with the new API.
The agent's tool-call log shows the intended flow: `observe_latest_stakeholder_message`
returns `sm_N`, then `observe_message(message_id="sm_N")` captures it.

| metric | OLD (seeds 1000–1004) | NEW (seeds 2000–2001) |
|--------|------------------------|------------------------|
| `observe_turn` calls | many (turn_idx guessing) | **0** |
| tool errors per run | 2–10 | **0** |
| `not a stakeholder message` errors | yes (2–10/run) | **none** |
| too_many_errors termination | run_01 | **none** |
| termination | 4× user_stop, 1× too_many_errors | **2× user_stop** |
| `observe_message`/`observe_latest` used | n/a | 11 / 11 each run |
| evidence_pass | True | **True** |
| invalid observation refs | 0 | **0** |

Both new runs completed normally with **zero observation-capture tool errors**,
and the previously fatal `too_many_errors` turn-index retry loop is gone. (DAG
reconstruction quality still varies run-to-run — e.g. node_recall 0.83 / 0.67,
edge_recall 0.67 / 0.50 — but improving question strategy / DAG quality was
explicitly out of scope for this task.)

## 7. Remaining limitations

- `observe_latest_stakeholder_message` is intentionally a **read** (returns the
  latest id); capturing still requires `observe_message(message_id)`. This is the
  deliberate trade-off for deterministic state replay.
- An agent that never calls `list_stakeholder_messages` / `observe_latest` but
  guesses an `sm_` id still gets a clean rejection (no silent wrong capture), so
  fabrication remains impossible.
- The observation id remains `obs_{ledger_turn}` internally; the agent treats it
  as an opaque provenance string and never needs to interpret it.
- JA relies on the same message-id mapping (ids are language-agnostic), and the
  EN/JA equivalence test passes unchanged.

## 8. Deliverables

- Code: `src/tau2/domains/business_interview/tools.py`, `environment.py`, `dag.py`
- Policy / docs: `data/tau2/domains/business_interview/policy.md`, `README.md`
- Reference data: `data/tau2/domains/business_interview/tasks.json`
- Tests: `tests/test_domains/test_business_interview/test_dag_business_interview.py`
- Smoke artifacts: `artifacts/business_interview_real_llm/run_00_seed2000.json`,
  `run_01_seed2001.json`
- This report: `doc/business-interview-observation-capture-ux-report.md`
