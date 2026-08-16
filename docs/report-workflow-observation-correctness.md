# Report: Workflow Observation Correctness (business_interview)

This report documents the hardening of the `business_interview` single-stakeholder
Workflow-first benchmark to **validate recorded observations against ground
truth** — not merely whether a result was recorded, but whether owner / evidence /
removal observations are *correct*. It also makes Stakeholder Truth genuinely
step-unit (structured, structural completeness) and fixes a same-concept matcher
bug where two unresolved (None) data items were treated as matching evidence.

Scope: entirely domain / data / tests / docs. **The generic tau2 core is
unchanged** (verified via `git diff` — only `business_interview` files changed).

---

## 1. Problems before this change

1. **Observation correctness was not graded.** The evaluator judged whether
   owner / evidence / removal were *recorded*, but not whether the recorded value
   matched the ground truth. A fabricated owner (`KNOWN("CEO")`) or fabricated
   evidence (`KNOWN("law")`) passed as long as *something* was recorded.
2. **No ground-truth expectations for observations.** `GTNecessity` had no
   `expected_owner_result` / `expected_evidence_result` / `expected_removal_result`,
   so "correct" was undefined.
3. **`KNOWN` without a value was accepted.** `owner_result=KNOWN` with
   `owner=None` counted as a recorded (and passing) observation; the state was
   internally inconsistent.
4. **Stakeholder Truth was a keyword search over the whole scenario text.** The
   old `StepTruthRequirement` + `known_info` full-text signal search meant a word
   appearing in *any* step (or the general description) satisfied a *specific*
   step's requirement. Completeness was not structural or per-step.
5. **Same-concept matcher used None as matching evidence.** In `_pick_candidate`,
   `{resolve(i, DATA_CONCEPTS) for i in ...}` left unresolved `None` in the set;
   two different unknown data items that both resolved to `None` matched via set
   intersection and added a (spurious) data-match signal.

## 2. Breaking changes

- `GTNecessity` gained `expected_owner_result`, `expected_evidence_result`,
  `expected_removal_result` (and the values are graded as *correct*).
- `WorkflowEvaluation` gained `why_correct`, `owner_correct`, `evidence_correct`,
  `removal_correct`; `challenge_done` now requires all required observations to be
  **recorded AND correct** (plus deletion considered).
- The old `StepTruthRequirement` / `stakeholder_step_truth_requirements()` /
  `missing_stakeholder_truth(known_info)` were **removed** and replaced by
  structured `StakeholderTruth` / `StakeholderStepTruth` / `StakeholderObservation`
  plus `stakeholder_truth_completeness(scenario_id, stakeholder_truth?)` and
  `missing_from_instructions(scenario_id, instructions_text)`.
- `record_necessity_detail` now **raises** if `*_state="KNOWN"` is given without a
  non-empty value.
- The role/system alias tables were moved out of `semantic.py` into a new
  `aliases.py` (shared with `ground_truth.py`); `semantic.py` imports `norm_role` /
  `norm_system` from there.
- No compatibility shims were retained.

## 3. Ground Truth observation model

`GTNecessity` now declares the expected owner / evidence / removal observation
results for the questionable legacy step:

- `expected_owner_result = UNKNOWN` (belief: Accounting, uncertain)
- `expected_evidence_result = NONE_FOUND`
- `expected_removal_result = UNKNOWN`

These are minimal per-dimension expectations — not a large claim model.

## 4. Observation correctness evaluation

`challenge_pass` / `challenge_done` now require the recorded owner / evidence /
removal results to match the ground-truth expectations (`_observation_correct`),
in addition to being recorded. A recorded `KNOWN` is only ever correct when the
truth is also `KNOWN`; a fabricated `KNOWN` (e.g. an invented owner) is recorded
but **incorrect** and fails the challenge. The "why" rationale correctness is
reused from `_rationale_status` (`why_correct`).

Diagnostics surface the three tiers per dimension:

```
owner:   { asked: true, recorded: true, result: KNOWN,      correct: false }
evidence:{ asked: true, recorded: true, result: NONE_FOUND, correct: true }
removal: { asked: true, recorded: true, result: UNKNOWN,    correct: true }
```

## 5. KNOWN value validation

- **Tool side:** `record_necessity_detail(..., owner_state="KNOWN")` with no
  `owner` value raises `ValueError` (same for evidence / removal).
- **Evaluator side:** `_observation_correct` defensively treats a `KNOWN` with no
  value as **not recorded and not correct** (so even a direct DB write fails).
- `UNKNOWN` / `NONE_FOUND` do not require a value.

## 6. Stakeholder Truth structure

`StakeholderTruth` / `StakeholderStepTruth` / `StakeholderObservation` hold the
facts the stakeholder can answer, expressed **per step (by concept) and per axis**
(actor, system, reads, writes, condition, rationale, owner, evidence, removal).
It is stored evaluator-side in `ground_truth.py` and never shown to the agent.
The three layers remain distinct: Evaluator Ground Truth (objective workflow),
Stakeholder Truth (what the stakeholder knows), Agent Reconstruction (what the
agent records).

## 7. Completeness verification (structural, per step, scenario-aware)

`stakeholder_truth_completeness(scenario_id, stakeholder_truth=None)` returns the
`(step_concept, axis)` pairs the scenario's structured truth is missing for axes
the evaluator requires (`required_truth_axes`). It is a **structural per-step
comparison**: a value in another step can never satisfy a missing axis. It is
**scenario-aware**: it uses the canonical truth for the given scenario id (EN/JA
share the canonical `quotation_workflow_1` truth), and can be passed an overridden
truth for tests. For the bundled scenario the result is `[]` (complete).

`missing_from_instructions(scenario_id, instructions_text)` is a separate
**simulator-answerability** check: every Stakeholder Truth value must be
expressible from the simulator prompt (known_info + unknown_info + task_instructions),
with alias / bilingual expansion, and each `UNKNOWN` / `NONE_FOUND` observation
needs a matching explicit unknown / no-findings signal. This guarantees we never
create a state where the evaluator has truth the simulator cannot answer. For both
EN and JA it returns `[]`.

## 8. Same-concept matcher None fix

`_pick_candidate` (and a new `_data_concepts` helper) now **exclude `None`** from
resolved data-concept sets. Two different unresolved data items (both
`resolve -> None`) are no longer treated as a data match. Verified: reverting the
fix makes the matcher pull a reconstructed step toward a candidate purely by the
`None & None` overlap (old: matches `g2`; fixed: ties to `g1`).

The disambiguation priority is unchanged: **1. action concept, 2. actor/system,
3. data**. Graph-context matching is intentionally **not** added.

## 9. Falsification suite A–V

The test file (`test_workflow_business_interview.py`, 49 tests) covers:

| ID | Case | Result |
|----|------|--------|
| A | owner UNKNOWN expected / UNKNOWN recorded | PASS |
| B | owner UNKNOWN expected / KNOWN fake value | FAIL |
| C | evidence NONE_FOUND expected / NONE_FOUND | PASS |
| D | evidence NONE_FOUND expected / KNOWN fake evidence | FAIL |
| E | removal UNKNOWN expected / UNKNOWN | PASS |
| F | removal UNKNOWN expected / KNOWN fabricated impact | FAIL |
| G | KNOWN owner without value | FAIL (tool raises; evaluator defensive) |
| H | KNOWN evidence without value | FAIL |
| I | KNOWN removal without value | FAIL |
| J | asked only (no result) | FAIL |
| K | recorded but incorrect | FAIL |
| L | recorded and correct | PASS |
| M | structured step/axis completeness | PASS |
| N | approve_quote.system removed from truth | completeness FAIL |
| O | same system string in another step | still FAIL (per-step) |
| P | scenario-aware completeness (EN/JA canonical, injected truth) | PASS |
| Q | unresolved None data not matching evidence | PASS |
| R | arbitrary agent step ids | full PASS |
| S | EN reconstruction | full PASS |
| T | equivalent JA reconstruction | full PASS |
| U | quality correct / no finish | quality PASS / protocol FAIL |
| V | agent-visible leakage | none |

Additional tests keep data recall/precision, condition-aware transitions,
branches, metadata, confirmed-rationale content/source/status, improvement order,
diagnostics readability, and the end-to-end `EnvironmentEvaluator` reward
(including a run that fabricates an owner and fails the necessity challenge).

## 10. business_interview tests

`uv run pytest tests/test_domains/test_business_interview/` → **49 passed**.

## 11. core tests / make check-all

- `make check-all` (ruff lint + format) → **All checks passed**.
- `make test` (core) → **278 passed, 1 xpassed, 1 failed**. The single failure is
  `tests/test_run.py::test_run_tasks_nl_assertions`, which drives the **mock
  domain with live gpt-3.5-turbo** — unrelated to this change (only
  `business_interview` files changed) and flaky/network-dependent: it **passes in
  isolation** (`uv run pytest tests/test_run.py::test_run_tasks_nl_assertions` →
  1 passed). Not a regression.

## 12. EN / JA real run (DeepSeek)

DeepSeek (`deepseek/deepseek-chat`, `DEEPSEEK_API_KEY` set) ran the full
`base` split: **6 simulations** (3× EN, 3× JA). Reward scores were 0 because the
bundled deepseek agent records few steps, but the qualitative goals were confirmed
from diagnostics:

- **Observation-correctness diagnostics distinguish the tiers.** Runs that never
  asked show `owner_asked=False, owner_result=NOT_RECORDED, owner_correct=False`.
  Runs that recorded show `owner_result=UNKNOWN, owner_correct=True`;
  `evidence_result=NONE_FOUND, evidence_correct=True`;
  `removal_result=UNKNOWN, removal_correct=True`; `challenge_done=True` when
  deletion was also considered.
- **UNKNOWN / NONE_FOUND / KNOWN are distinguished** in the recorded results.
- **The simulator can answer the truth** (the structured truth is consistent with
  the simulator instructions; `missing_from_instructions` is empty for EN and JA).
- **The workflow evaluator runs normally** on both EN and JA.

## 13. Generic core diff

Zero. `git diff` touches only `business_interview` files (`data_model.py`,
`ground_truth.py`, `semantic.py`, `tools.py`, new `aliases.py`, `README.md`) plus
the test file. Registry/cli references are unchanged.

## 14. Remaining limitations

- **Observation value *content* is not concept-graded** for `KNOWN` results: a
  `KNOWN` is correct iff the expectation is also `KNOWN` (value present). The
  bundled scenario has no `KNOWN`-expected observation, so value-content matching
  is not wired (documented; `expected_*_concept` fields are not used).
- **Simulator-answerability is a consistency test, not generated text.** The
  structured truth is stored separately from the simulator instructions and tied
  together by `missing_from_instructions`. If a future scenario drifts, that test
  catches it, but there is no code generation.
- **Same-concept steps differing only by graph position** remain a known
  limitation (graph-context matching intentionally not added).
- **`deepseek-chat` is weak** at the workflow tool protocol, so real-run reward
  scores are low; the run verifies the evaluator, not agent skill.
- The `same_concept_workflow` scenario and `StakeholderTruth` are not in the task
  set (matching unit tests exercise the matcher directly).

## 15. Readiness for a second workflow scenario

The infrastructure is ready: the ground-truth observation model
(`expected_owner_result` etc.), the structured `StakeholderTruth` (per-step/per-axis,
scenario-aware completeness), the alias/bilingual `value_signals`, and the
observation-correctness evaluator are all scenario-generic. Adding a second
production scenario requires authoring its `WorkflowGroundTruth` + `StakeholderTruth`
(and ensuring `missing_from_instructions` passes for its localized instructions) —
no evaluator changes are expected.
