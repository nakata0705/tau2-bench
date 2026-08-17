# business_interview: Evaluation Simplification (Natural-Language Semantic Matching Removed)

## Goal summary

The `business_interview` benchmark expects that, in the future, the stakeholder LLM
may answer the same Ground Truth with different natural language on every run. It is
therefore infeasible for the evaluator to keep parsing Observation *text* with
token / substring / negation rules to decide whether `support(observation, claim) ->
SUPPORTED / CONTRADICTED / UNKNOWN` holds.

What we actually want to measure is *"can the Agent recover the correct business DAG
from varying natural-language interviews"* — not the evaluator's ability to do
natural-language entailment. This change therefore **removes the deterministic
semantic matcher** from the quality gate and **separates evaluation responsibility**
into (A) Result correctness (Ground Truth comparison) and (B) Evidence hygiene
(Observation references are real & authentic).

Scope was limited to the evaluator; no stakeholder-simulator rewrite, no Disclosure
Controller / DisclosureTrace / embeddings / LLM judge / ontology additions.

## What was removed

From `src/tau2/domains/business_interview/evaluation.py`:

- The whole **claim-level semantic relevance** block:
  - `support(obs_text, claim_kind, claim_value, context) -> SUPPORTED / CONTRADICTED / UNKNOWN`
  - `_obs_ids_supported`, `_relevance_pass`
  - token/substring matching via `_mentions` and `_match`-style overlap
  - relation-word edge support (`_relation_supported`, `_RELATION_WORDS`)
  - predicate text/direction support (`_predicate_supported`, `_DIRECTION_UP` / `_DIRECTION_DOWN`)
  - negation-based semantic judgement (`_is_negated`, `_NEGATION`)
- `relevance_pass` field removed from `EvaluationResult`.
- `relevance_pass` removed from the `quality_pass` conjunction.
- `obs_by_id` lookup (used only for relevance) removed from `evaluate`.

The only matcher-adjacent helpers retained are those still needed for **Result
correctness** scoring: hidden-`EvaluationSpec` node matching, `_data_recall` for
reads/writes, `_predicate_ok`, and `_necessity_value_ok` — all Ground Truth
comparison metrics, not Observation-body semantic relevance.

From the tests, removed (semantic relevance tests that no longer gate):
- `test_per_claim_poisoning_fails` (poisoned provenance -> relevance fail)
- `test_correct_action_evidence_cannot_substitute_actor`
- `test_from_node_evidence_alone_cannot_support_edge`
- `test_node_evidence_alone_cannot_support_predicate`
- `test_explicit_negation_is_contradicted`
- `test_unrelated_evidence_is_unsupported`
- the now-invalid `relevance_pass` assertions in remaining tests

The `doc/`/README text describing claim-level relevance was rewritten to describe
Result correctness vs Evidence hygiene.

## Result correctness vs evidence hygiene

**Result correctness (A).** The inferred DAG is compared against the hidden Ground
Truth. This is unchanged and is the sole arbiter of *semantic* correctness of the
final DAG: node/action correctness, primitive diagnostic, actor, system, reads /
writes, edges, predicates, necessity, start/end, arbitrary agent node IDs, hidden
scenario-local `EvaluationSpec`.

**Evidence hygiene (B).** Deterministically guaranteed per asserted claim:
- the referenced Observation **exists**;
- it was **captured from an actual stakeholder (user) message** (authentic, not
  fabricated);
- the claim carries **provenance** (every asserted attribute has an authentic
  Observation reference).

The evaluator does **not** re-interpret the Observation body to decide whether it
"really means" the claim value. Whether `actor="manager"` is correct is judged by
Ground Truth comparison, not by re-reading `obs_17`'s text.

## New `quality_pass` conditions

`quality_pass` is now:

```
structural_pass AND necessity_pass AND evidence_pass AND provenance_authenticity_pass
```

- `structural_pass` — Ground Truth structural/metric correctness (nodes, edges,
  start/end, predicate/actor/system/read/write correctness all == 1.0, valid DAG,
  endpoints match).
- `necessity_pass` — necessity correctness == 1.0 and no fabricated necessity.
- `evidence_pass` — provenance coverage == 1.0 across nodes, attributes, edges,
  predicates, necessity.
- `provenance_authenticity_pass` — no invalid observation references and no
  non-authentic observation references.

`relevance_pass` is gone from `EvaluationResult` and from the gate.

## New / changed tests (`test_dag_business_interview.py`)

Maintained (unchanged semantics): full valid DAG passes; Observation authenticity;
invalid / nonexistent observation rejected; missing provenance coverage fails
evidence gate; endpoints regression; arbitrary agent node IDs; EN/JA equivalent
behavior; `unclassified` primitive valid; correct action + wrong known primitive
lowers the diagnostic; necessity correct/fabricated; lab `lab_sample_flow` full
pass; lab `unclassified` primitive valid; leakage; quotation reference trajectory
reward 1.0; missing-node / fabricated-necessity detection.

Added to replace the removed relevance tests:
- `test_wrong_claim_passes_evidence_hygiene_but_fails_ground_truth` — for each of
  action / actor / system / read / write / predicate / edge / necessity, a **wrong
  claim backed by an authentic-but-unrelated Observation** still passes
  `provenance_authenticity_pass` and `evidence_pass` (hygiene), but fails
  `quality_pass` (wrong value caught by Ground Truth comparison).
- Per-metric drops: `test_wrong_action_drops_node_correctness`,
  `test_wrong_actor_drops_actor_correctness`,
  `test_wrong_system_drops_system_correctness`,
  `test_wrong_read_drops_read_correctness`,
  `test_wrong_write_drops_write_correctness`,
  `test_wrong_predicate_drops_predicate_correctness`,
  `test_wrong_edge_drops_edge_correctness`,
  `test_wrong_necessity_drops_necessity_correctness`.

This directly confirms the objective's key requirement: *wrong claim + authentic
but unrelated Observation -> evidence hygiene passes, but claim correctness fails
via Ground Truth comparison*.

## Remaining limitations

- Result-correctness matching still uses a hidden scenario-local `EvaluationSpec`
  with expression/alias lists and lightweight token/substring/role tie-breaking to
  map free-text agent actions onto Truth nodes. This is a *scoring* aid for the
  deterministic Ground Truth comparison (unchanged and in scope), not a semantic
  judgement of Observation text.
- Evidence hygiene is a proxy: it verifies provenance *references* are real and
  authentic but cannot itself tell whether a claim is true — truth is delegated to
  Ground Truth comparison.
- Primitive resolution remains a heuristic substring matcher (finite signal-word
  coverage); unknown operations resolve to `unclassified`, which is not a failure.
- The bundled `deepseek-chat` agent rarely follows the full tool protocol, so a
  real-model smoke reward is low (weak-agent behavior, not a benchmark defect).
- Out of scope (intentionally not implemented): Disclosure Controller,
  DisclosureTrace, new stakeholder LLM generation architecture, embeddings, LLM
  judge, ontology additions, simulator overhaul.

## Verification results

- `uv run pytest tests/test_domains/test_business_interview/` -> **33 passed**.
  (Includes quotation EN/JA, `lab_sample_flow`, arbitrary node IDs, reference
  trajectory reward 1.0 via `test_evaluator_rewards_full_reconstruction`,
  evidence/authenticity regressions, and the new wrong-claim tests.)
- `make check-all` (ruff lint + format) -> **passed**.
- `uv run python -m py_compile` on the changed modules -> **OK**.
- `make test` (core) -> **262 passed, 1 xpassed**; a single unrelated flaky
  live-API test (`tests/test_run.py::test_run_tasks_env_assertions`, **mock** domain
  with `gpt-3.5-turbo`) failed in the full-suite run but **passes in isolation**;
  its own docstring notes it "can fail if model is not good enough" and it does not
  touch `business_interview`.
- Grep for `relevance_pass` / `SUPPORTED` / `CONTRADICTED` / `_relevance_pass`
  across `src`/`tests` -> no business_interview references remain.

## Commit

Committed on the `business-interview` branch and pushed to `origin/business-interview`.
