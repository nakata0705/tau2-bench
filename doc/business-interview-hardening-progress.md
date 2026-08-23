# business_interview: Hardening Truth-reconstruction evaluation (progress report)

## Status

Completed hardening checkpoint. Primary benchmark goal:

    AgentConcepts + AgentGraph -> compare directly with TruthConcepts + TruthGraph

Conversational provenance remains diagnostic only; unsupported-but-correct
reconstruction still counts as correct. The deterministic business-interview
suite passes **113 tests** and ruff is clean.

## 1. Removed the obsolete concept grounding lifecycle

- `graph.py`: removed `ValidationStatus` literal and the
  `validation_status` / `validation_evidence` fields + `.resolved` from
  `AgentConcept` (concept = id, kind, label, description, optional mentions).
- `tools.py`: removed the `ground_concept`, `confirm_concept`,
  `mark_concept_unknown`, `mark_concept_disputed` tools and their
  docstrings; `merge_concepts` / `list_concepts` no longer touch
  validation fields; `finish_interview` docstring no longer mentions
  hypothesized concepts.
- `evaluation.py`: removed `glossary_pass`,
  `referenced_hypothesized_concepts`, `glossary_validation_errors` from the
  result; `glossary_complete` kept as concept reconstruction completeness.
- `policy.md`: replaced the "Validate the glossary" section with "Keep the
  glossary tidy" (create/update/merge/mention/terminology only); no
  unavailable tool names remain (regression test enforces this).
- `README.md`: replaced the "Concept status" section with "No concept
  validation lifecycle"; epistemic scoring section updated.
- Restored the missing `@is_tool` decorator on `record_dont_know` (it was
  exposed by policy but missing from the runtime schema — a real
  policy/schema inconsistency).
- Tests: removed obsolete lifecycle tests, added
  `test_obsolete_concept_lifecycle_tools_are_removed` and
  `test_policy_references_only_available_agent_tools` (the regression test
  proving the policy contains no unavailable Agent tool names).

## 2. Robust Truth-reconstruction content matcher

`evaluation.py` now uses a deterministic, language-tolerant matcher:

- **Unicode / Japanese**: NFKC normalize + lowercase; Latin runs tokenize to
  words with a small stop-word set removed; CJK runs (Hiragana/Katakana/
  ideographs) tokenize to character bigrams — Japanese labels never produce
  empty signatures.
- **Similarity**: Dice coefficient (the standard n-gram score), with a
  meaningful `_CONCEPT_MATCH_THRESHOLD = 0.4` so a single generic token does
  not equate unrelated concepts.
- **Concept label vs description**: `_concept_similarity` takes the max of
  label-only and label+description scores (a short label such as "CRM" still
  strongly matches the canonical term "CRM" even when the Truth description
  is long), computed against canonical terms AND canonical + stakeholder
  local terms (JA locale support).
- **Stable global matching**: `_max_weight_assignment` is a deterministic
  maximum-weight bipartite matching (Hungarian, rectangular-safe with dummy
  rows/columns at cost 0) per concept kind, with the threshold gating weak
  pairs — invariant to Agent-local concept/node ids.
- **Node matching**: `_map_nodes_and_edges` uses aligned `(prop,
  truth_concept_id)` signatures and max-weight assignment; a shared activity
  pair is the node's primary identity (sharing only a minor property is not
  identity). Edge mapping by matching endpoint pair on the Truth graph.
- **JA locale scenario**: `scenario.py` now projects Japanese local terms for
  the JA locale via `concept_overrides`; `evaluate` feeds those local terms
  as `term_extras` so a Japanese agent vocabulary matches deterministically.
- Tests added: id-invariance (`test_score_invariant_to_agent_concept_ids`),
  generic-words rejection
  (`test_generic_words_do_not_cause_false_concept_matches`),
  Japanese matching (`test_japanese_concept_matching`,
  `test_ja_scenario_reconstruction_scores`).

## 3. Epistemic scoring for Truth absence

`evaluation.py` slot scoring is now Truth-epistemic (no-answer is not a lucky
guess):

- scalar Truth `ConceptRef` -> only a matching asserted Agent ConceptRef;
- scalar Truth `None` -> only an explicit Agent `ABSENT` is correct
  (UNSET / DONT_KNOW / any concept are incorrect);
- list Truth `list` -> recall*precision over elements;
- list Truth `None`/known-empty -> explicit ABSENT required (UNSET /
  DONT_KNOW incomplete);
- edge condition `None` -> explicit ABSENT required (same rule).

Tests added covering the full matrix
(`test_truth_value_slot_scoring_matrix`,
`test_truth_absent_scalar_slot_scoring_matrix`,
`test_truth_absent_list_slot_requires_absent`,
`test_truth_absent_edge_condition_requires_absent`,
`test_truth_absent_slots_score_absent_in_full_build`).

## 4. Real-run tool-error accounting

New module `src/tau2/domains/business_interview/run_metrics.py`:

- `account_tool_errors(messages)` walks the trajectory, finds every failing
  `ToolMessage` (error=True), maps tool-call ids -> tool names, and reports
  `tool_error_count`, `tool_error_categories`, `tool_error_counts_by_tool`,
  `tool_error_counts_by_category`.
- `classify_tool_error` normalizes failures into deterministic categories:
  `tool_not_found` / `missing_reference` / `invalid_argument` /
  `validation_error` / `other_tool_error`.
- `provider_error_count(errors)` counts the top-level provider/runtime list.

`scripts/business_interview_real_llm_smoke.py` now persists
`provider_error_count`, `tool_error_count`, `tool_error_categories`,
`tool_error_counts_by_tool`, `tool_error_counts_by_category`, `agent_calls`
and `accepted_observations` in each run dump and in summary.json (no longer
infers success from `errors == []`). The summary also includes concept
recall/precision/correctness, every node-property correctness metric,
`reconstruction_pass`, fabricated counts, and elapsed time. The graph dump
drops the removed `validation_status`/`validation_evidence` and the old
`glossary_pass` name is replaced by `glossary_complete`.

Tests added: `test_tool_error_accounting_classifies_and_groups`,
`test_tool_error_accounting_empty_trajectory`.

## Verification

### Deterministic and shared tests

- Business-interview roundtrips + domain tests: **113 passed**.
- Relevant shared tests (`test_environment.py`,
  `test_evaluate_trajectories.py`, `test_tasks.py`, `test_results_format.py`,
  and the non-LLM portions of `test_run.py`): **188 passed**.
- Eight `test_run.py` tests attempted to invoke the existing OpenAI-backed
  user simulator and failed before simulation because this environment has no
  `OPENAI_API_KEY`. These are environment/authentication failures, not
  assertion failures from this hardening change.
- Ruff: **clean**. Changed Python modules compile successfully.

### One real DeepSeek/OpenRouter run

Artifact: `artifacts/business_interview_real_llm/run_00_seed9001.json` and its
private ledger. Configuration was the script default
`openrouter/deepseek/deepseek-v4-flash-0731` for both Agent and stakeholder,
temperature `0.0`, task `quotation_workflow_1`, seed `9001`.

| Metric | Result |
| --- | --- |
| termination reason | `episode_complete` |
| provider/runtime errors | `0` |
| Agent tool errors | `1` (`invalid_argument`: `add_node` × 1) |
| tool-error detail | `add_node writes: expected a list of concept refs, ...` |
| Agent calls | `56` |
| accepted Observations | `22` |
| elapsed | `483.01 s` |
| node recall / precision | `1.0 / 1.0` |
| edge recall / precision | `1.0 / 1.0` |
| concept recall / precision / correctness | `0.9048 / 0.9048 / 0.8186` |
| activity / actor / system correctness | `0.8333 / 1.0 / 0.6667` |
| read / write / rationale correctness | `0.3333 / 0.1667 / 0.1667` |
| condition correctness | `1.0` |
| fabricated nodes / edges | `0 / 0` |
| reconstruction pass | `false` |
| structural pass | `false` |
| quality pass | `false` |
| glossary complete | `false` |
| evidence pass | `false` |
| reward | `0.0` |
| private-ID leakage | none |

The run is **not benchmark success**: `episode_complete` only describes
termination, while `quality_pass=false` and `reconstruction_pass=false`.
The top-level `errors` list was empty, but the trajectory contained one
failing `ToolMessage`; the artifact correctly reports it rather than claiming
zero errors. The Agent called only available tools; the failure was a bad
`add_node` argument, not an obsolete/nonexistent lifecycle tool.

## Remaining evaluator weaknesses

- The content matcher is deterministic and thresholded, but its lexical
  signatures are not semantic understanding; language-specific paraphrases
  outside the tested canonical/local-term vocabularies can still be missed.
- The relevant shared `test_run.py` cases require an OpenAI credential in the
  execution environment and were not green here.
- The single live run is exploratory and non-deterministic; it demonstrates
  accounting correctness, not model quality.

## Final commit

- Branch: `business-interview`
- Message: `fix: harden truth reconstruction evaluation`
- The final commit SHA is reported in the handoff and can be obtained with
  `git log -1 --oneline`.
