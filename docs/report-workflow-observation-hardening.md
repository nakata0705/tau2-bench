# Report: Workflow Observation Hardening (business_interview)

This report documents the hardening of the `business_interview` single-stakeholder
Workflow-first benchmark: separating *interview questions asked* from *observed
workflow evidence recorded*, making `UNKNOWN` an explicit recorded outcome,
requiring result recording (not just asking) for the necessity challenge, moving
stakeholder truth to a step-unit model, and supporting multiple steps that share
an action concept.

Scope: this is entirely domain / data / tests / docs. **The generic tau2 core is
unchanged** (verified via `git diff --stat` — only `business_interview` files and
its tests/data/README changed).

---

## 1. Problems before this change

The previous v2 implementation (commits `98ff1c1`, `b677482`) had these gaps:

1. **ASKED conflated with RESULT RECORDED.** `challenge_step(step_id, "why", …)`
   set `necessity.investigated = True` directly, and `challenge_step("owner")` /
   `"evidence"` / `"removal"` set the `*_investigated` flags. The evaluator's
   `_challenge_dimensions` treated those "asked" flags as sufficient, so **asking
   a question alone passed that dimension**. This directly violated falsification
   B / E / G / I (asked-only must FAIL).
2. **UNKNOWN conflated with "not answered".** Because only an "investigated"
   boolean was checked, an asked-but-unrecorded dimension scored the same as an
   explicitly recorded `UNKNOWN`. There was no `NOT_RECORDED` state distinct from
   `UNKNOWN` / `NONE_FOUND`.
3. **Necessity challenge gated on self-reported asking.** `challenge_pass`
   required only the `*_investigated` booleans, which the agent could set with a
   single `challenge_step` call each — no recorded observation was required.
4. **Stakeholder truth was flat, not step-unit.** `stakeholder_knowledge_requirements()`
   was a flat list of facts; the completeness test only checked each signal
   appeared somewhere in `known_info`, which was close to a whole-scenario keyword
   check rather than per-step completeness.
5. **`concept -> single step` constraint.** `_match_steps` used
   `gt_by_concept = {g.concept: g for g in gt_steps}` (a dict). A concept could
   map to only one ground-truth step, so same-concept multi-step workflows were
   impossible.

## 2. Old design removed

- `Necessity.investigated` / `rationale_known` / `epistemic_status` /
  `owner_investigated` / `evidence_investigated` / `removal_investigated` were
  removed (the single "asked ≈ recorded" boolean axis).
- The evaluator's `*_investigated` diagnostics fields were removed.
- `_match_steps`'s `concept -> single step` dict was replaced.
- Flat `stakeholder_knowledge_requirements()` was replaced with step-unit
  `stakeholder_step_truth_requirements()`.

No compatibility shims were retained (backward compatibility was explicitly not
required).

## 3. New data model

`src/tau2/domains/business_interview/data_model.py`:

- `RationaleResult`: `NOT_RECORDED | FACT | BELIEF | UNKNOWN` — the recorded
  outcome of the "why is this needed?" question.
- `NecessityResult`: `NOT_RECORDED | KNOWN | UNKNOWN | NONE_FOUND` — the recorded
  outcome of the owner / evidence / removal investigations.
- `Necessity` now has a separate **ASKED** axis (`why_asked`, `owner_asked`,
  `evidence_asked`, `removal_asked`) and a **RESULT RECORDED** axis
  (`rationale_result`, `owner_result`, `evidence_result`, `removal_result`,
  plus the value fields `rationale`/`source`/`owner`/`evidence`/`removal_impact`),
  with convenience properties `why_recorded` / `owner_recorded` /
  `evidence_recorded` / `removal_recorded`.
- `deletion_considered` / `deletion_candidate` remain the **analyst's own
  assessment**, explicitly separate from the stakeholder observations.

## 4. ASKED vs RESULT RECORDED

- `challenge_step` now records only that a question was **asked** (`*_asked`,
  and `deletion_considered` for the deletion assessment). It records no result.
- Results are recorded only via:
  - `set_step_rationale` / `set_step_unknown` → `rationale_result` (why).
  - `record_necessity_detail(…, owner_state/evidence_state/removal_state, …)`
    → `owner_result` / `evidence_result` / `removal_result`.
- The benchmark evaluates **result recorded**, never "question asked". Diagnostics
  expose both axes so a failure reason is human-readable:

  ```
  why:      { asked: true,  recorded: false }
  owner:    { asked: true,  recorded: true,  result: UNKNOWN }
  evidence: { asked: false, recorded: false, result: NOT_RECORDED }
  ```

## 5. UNKNOWN / NOT_RECORDED / NONE_FOUND

- `NOT_RECORDED`: the question was not asked or no answer was recorded. An
  asked-but-unanswered dimension is `NOT_RECORDED` — it is **never** treated as
  `UNKNOWN` / `NONE_FOUND`.
- `UNKNOWN`: the stakeholder was asked and confirmed they do not know, and that
  outcome was recorded explicitly.
- `NONE_FOUND`: investigated and nothing exists / no one / no evidence.
- These are single, explicit outcome enums (no duplicated state).

## 6. Necessity observation (challenge) requires result recording

`challenge_pass` (⇒ `challenge_done`) requires, on the questionable step, that
**why / owner / evidence / removal all have a result recorded** (FACT/BELIEF/
UNKNOWN for why; KNOWN/UNKNOWN/NONE_FOUND for owner/evidence/removal) **and**
that `deletion_considered` is set. `None` is never used to mean both "not
investigated" and "investigated but absent" — the explicit result enum covers
that.

`improvement_order_ok` now keys off recorded results: a non-`question` first
proposal requires `why_recorded`; `automate` requires `why_recorded` AND
`owner_recorded` AND `evidence_recorded`.

## 7. Stakeholder truth model (step-unit)

Three layers are kept separate:

- **Evaluator Ground Truth** — the complete workflow (`ground_truth.py`,
  evaluator-only).
- **Stakeholder Truth** — per-step facts the stakeholder knows
  (`stakeholder_step_truth_requirements()`), each tied to a ground-truth step
  (by concept) and axis (actor / system / reads / writes / condition /
  rationale / rationale_unknown).
- **Agent Reconstruction** — the workflow the agent records during the interview.

`missing_stakeholder_truth(known_info)` returns the step-unit requirements the
scenario's `known_info` cannot answer. Completeness is judged **per step**, not by
a whole-scenario keyword search. A structural test asserts every evaluator-graded
fact is answerable from the scenario (falsification N), and that a scenario
missing a step's required truth fails (falsification O).

## 8. Same-concept multi-step matching

`_match_steps` now groups ground-truth steps by concept and disambiguates
same-concept candidates deterministically by priority:

1. **action concept** (bilingual, `concepts.py`);
2. **actor / system**;
3. **data** (read/write concept overlap);

ties fall to the earliest ground-truth step (stable, explainable). No LLM judge,
no hidden step ids. The `same_concept_workflow` scenario (not a task) has two
steps sharing the `approve_quote` concept and is exercised by falsification Q
(both match correctly) and R (swapped attributes → structural FAIL).

## 9. Improvement order

`Question → Delete → Simplify → Accelerate → Automate` is enforced per step
(ranks 1–5), plus the recorded-necessity preconditions described in §6. No complex
state machine.

## 10. Leakage audit

- Agent-visible policy and tool descriptions contain no hidden workflow identity
  (`HIDDEN_TERMS` test), no scenario answers, no canonical ids, no `description` /
  `evaluation_criteria` references.
- The agent system prompt is built only from the general policy.
- The stakeholder scenario does not reveal `s1`…`s6`, the canonical ids, or the
  exact branch text.
- The full ground truth and the step-unit stakeholder requirements live only in
  evaluator-side domain code and the stakeholder scenario's `known_info`
  (stakeholder-only, not agent-visible).

## 11. Falsification suite A–V

The test file `tests/test_domains/test_business_interview/test_workflow_business_interview.py`
contains 56 tests, including the full falsification suite A–V:

| ID | Case | Result |
|----|------|--------|
| A | why not asked | FAIL |
| B | why asked, result not recorded | FAIL |
| C | explicit UNKNOWN recorded | PASS |
| D | UNKNOWN promoted to FACT | FAIL |
| E | owner asked only | FAIL |
| F | owner UNKNOWN / NONE_FOUND / KNOWN recorded | PASS |
| G | evidence asked only | FAIL |
| H | evidence UNKNOWN / NONE_FOUND / KNOWN recorded | PASS |
| I | removal asked only | FAIL |
| J | removal result recorded | PASS |
| K | all observations recorded | challenge PASS |
| L | automate before necessity recorded | FAIL |
| M | delete after necessity recorded | PASS |
| N | step-unit stakeholder truth completeness | PASS |
| O | a step's required truth missing | completeness FAIL |
| P | arbitrary agent step ids | full PASS |
| Q | same-concept 2 steps both match | PASS |
| R | same-concept 2 steps swapped | structural FAIL |
| S | EN reconstruction | full PASS |
| T | equivalent JA reconstruction | full PASS |
| U | quality correct / no finish | quality PASS / protocol FAIL |
| V | agent-visible leakage | none |

Additional tests cover data recall/precision, condition-aware transitions,
branches, metadata, confirmed-rationale content/source/status, deletion vs
observation separation, diagnostics readability, and end-to-end
`EnvironmentEvaluator` reward (including an end-to-end case where the reference
run drops `record_necessity_detail` and correctly fails the necessity challenge
with `owner_asked: true, owner_recorded: false`).

## 12. Verification run

- `uv run pytest tests/test_domains/test_business_interview/` → **56 passed**.
- `make check-all` (ruff lint + format) → **All checks passed**; 337 files
  unchanged after format.
- `make test` (core) → **285 passed, 1 xfailed, 1 failed**.

The single core failure is `tests/test_run.py::test_run_tasks_nl_assertions`,
which drives the **mock domain with live `gpt-3.5-turbo` LLM calls**. It is
unrelated to this change (only `business_interview` files were modified) and is
flaky/network-dependent: it **passes in isolation** (`uv run pytest
tests/test_run.py::test_run_tasks_nl_assertions` → 1 passed). This is a known
live-LLM flake, not a regression.

## 13. EN / JA real run (DeepSeek)

DeepSeek (`deepseek/deepseek-chat` via LiteLLM, `DEEPSEEK_API_KEY` set) was
available. `tau2 run --domain business_interview --task-split base --agent-llm
deepseek/deepseek-chat --user-llm deepseek/deepseek-chat --num-trials 3
--num-tasks 2` ran **6 simulations** (2 tasks × 3 trials) successfully through
the workflow evaluator.

Qualitative observations (the point of the run — not scores, which were 0 because
the weak `deepseek-chat` agent recorded few steps):

- **ASKED vs RESULT RECORDED are distinguishable.** One run recorded
  `why_asked: True, why_recorded: True`, `owner_result: UNKNOWN`,
  `evidence_result: NONE_FOUND`, `removal_result: UNKNOWN`. Runs where the agent
  never asked show `owner_asked: False, owner_result: NOT_RECORDED`.
- **UNKNOWN / NONE_FOUND are explicit recorded outcomes**, distinct from
  `NOT_RECORDED`.
- **Challenge failure reasons are human-readable** in diagnostics (e.g.
  `deletion_considered: False`, or missing observations).
- **The workflow evaluator runs normally on both EN and JA** through the same
  `evaluate()` path.

Limitation: the bundled `deepseek-chat` agent rarely followed the full tool
protocol, so no trial achieved `quality_pass`. This does not affect the evaluator
verification, which was the goal of the real run. No scores are reported as
"passes".

## 14. Generic core diff

Zero. `git diff` touches only:
`src/tau2/domains/business_interview/{data_model,ground_truth,semantic,tools}.py`,
`data/tau2/domains/business_interview/{policy.md,tasks.json}`,
`src/tau2/domains/business_interview/README.md`,
`tests/test_domains/test_business_interview/test_workflow_business_interview.py`,
and this report. The registry/cli references (`get_environment`, `get_tasks`,
`get_tasks_split`) are unchanged and still work.

## 15. Breaking changes

- `Necessity` field renames/removals (no backward compatibility): `investigated`,
  `rationale_known`, `epistemic_status`, `owner_investigated`,
  `evidence_investigated`, `removal_investigated` removed; replaced by the ASKED
  booleans and the `*_result` outcome enums.
- `record_necessity_detail` now takes explicit `*_state` params (KNOWN / UNKNOWN /
  NONE_FOUND); values default the state to KNOWN.
- `WorkflowEvaluation` challenge fields renamed from `*_investigated` to
  `*_asked` / `*_recorded` (+ `owner_result` / `evidence_result` /
  `removal_result`).
- Reference evaluation actions add a `record_necessity_detail` call (asking alone
  no longer suffices for `assert_necessity_challenged`).
- `stakeholder_knowledge_requirements()` → `stakeholder_step_truth_requirements()`
  (step-unit) plus `missing_stakeholder_truth()`.

## 16. Remaining weaknesses

- **Same-concept disambiguation is actor/system then data**; graph-context
  tie-breaking is not yet used. Two same-concept steps that share actor, system
  AND data (differing only by graph position) would tie-break to earliest GT and
  could be swapped without detection. The bundled scenario differs by actor and
  data, so this is not exercised end-to-end.
- **`deepseek-chat` is weak** at following the workflow tool protocol, so real-run
  reward scores are low; stronger agents (e.g. GPT-4.1) should be benchmarked for
  meaningful score signal.
- **Deterministic substring concept resolution** is a domain-local heuristic; very
  unusual wording for a step could fail to resolve. This is bounded and tested for
  the bundled scenarios.
- The deletion assessment is binary (`deletion_considered`); it does not track the
  specific deletion/simplification rationale the analyst gave.

## 17. Readiness for next steps

The single-stakeholder foundation is now hardened: ASKED vs RESULT RECORDED are
separate, UNKNOWN is an explicit recorded outcome, the necessity challenge
requires recorded observations, stakeholder truth is step-unit, arbitrary step
ids and same-concept multi-step workflows are supported, the evaluator is
deterministic, and EN/JA equivalence holds. This is a sound base to add more
scenarios (the `same_concept_workflow` infrastructure is already in place) and to
extend toward **multi-stakeholder** later — the stakeholder-truth model and the
observation-vs-assessment separation already anticipate per-stakeholder evidence,
which is the natural next step.
