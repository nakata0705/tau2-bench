# business_interview: Hardening Truth-reconstruction evaluation (progress report)

## Status

Work-in-progress checkpoint on the hardening task. Primary benchmark goal
unchanged:

    AgentConcepts + AgentGraph -> compare directly with TruthConcepts + TruthGraph

Conversational provenance remains diagnostic only; unsupported-but-correct
reconstruction still counts as correct. Deterministic suite: **113 passed**
(roundtrips + graph domain), ruff lint clean, all modules compile.

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
infers success from `errors == []`); it also drops the removed
`validation_status`/`validation_evidence` from the graph dump and renames
`glossary_pass` -> `glossary_complete`.

Tests added: `test_tool_error_accounting_classifies_and_groups`,
`test_tool_error_accounting_empty_trajectory`.

## Not yet done

- Full shared-test sweep after the scenario/matcher changes (was in flight).
- One real-LLM seed run with the hardened tool-error accounting.
- Handoff doc + final commit message.

## Commit

[filled at commit time]
