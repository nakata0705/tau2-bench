# Business Interview primary edge-matching correctness fix

## Scope

This document records the deferred correctness fix identified in
`docs/business-interview-refactor-result.md`.  It changes only primary
AgentGraph → TruthGraph business-edge alignment.  It does not change concept
lexical matching, node alignment semantics, slot score matrices, epistemic
states, canonical boundaries, stakeholder behavior, evidence reward, or tool
signatures.

The implementation is in:

- `src/tau2/domains/business_interview/comparison.py`
- `src/tau2/domains/business_interview/evaluation.py`
- `src/tau2/domains/business_interview/evaluation_diagnostics.py`

The adversarial contract tests are in
`tests/test_domains/test_business_interview/test_edge_matching_correctness.py`.

## Old behavior and minimal reproduction

The old `_map_nodes_and_edges()` first mapped Agent nodes, then handled each
Agent edge independently:

```python
cands = [truth_edge_id for truth_edge_id in truth.edges ...]
if cands:
    edge_map[agent_edge_id] = sorted(cands)[0]
```

There was no reserved-Truth-edge set.  Thus, for:

```text
Truth: T1 = A -> B
Agent: E1 = A' -> B'
       E2 = A' -> B'
```

where `A' -> A` and `B' -> B`, the old alignment was:

```text
E1 -> T1
E2 -> T1
```

The old comparison then calculated:

```text
edge_recall          = |set(edge_mapping.values())| / |Truth edges|
edge_precision       = len(edge_mapping) / |Agent edges|
fabricated_edge_count = |Agent edges| - len(edge_mapping)
```

Therefore the minimal fixture incorrectly produced:

```text
edge_recall = 1.0
edge_precision = 1.0
fabricated_edge_count = 0
```

The regression test was added before the production change and was run against
the old implementation.  It failed with `edge_precision` observed as `1.0`
where the expected value was `0.5`.  This proves the issue was present in the
current HEAD implementation rather than inferred only from historical docs.

The same old behavior also made parallel-edge condition scoring depend on
Agent dictionary insertion order: both Agent edges were assigned to the first
lexical Truth edge, and the condition loop selected the first Agent edge for
that Truth edge.

## Current matching contract

For primary business edges:

1. Structural-only Truth nodes/edges are removed by
   `business_graph_projection()` and never enter the denominator.
2. Agent and Truth business edge ids are obtained through
   `business_edge_ids()`.
3. An Agent edge is eligible only if both endpoints have a node mapping and
   the mapped `(from_node, to_node)` pair is exactly equal to the Truth edge's
   endpoint pair.
4. Every Agent business edge maps to at most one Truth business edge.
5. Every Truth business edge maps to at most one Agent business edge.
6. An eligible but conditionally incorrect edge still counts as a structural
   edge match, preserving the historical edge/condition separation; its
   condition slot receives a zero score.
7. An Agent edge with no eligible Truth edge is unmatched and contributes to
   fabricated-edge/precision penalty.
8. A Truth edge with no assigned Agent edge is unmatched and contributes to
   recall penalty.

`compare_aligned_graphs()` now rejects an explicitly supplied duplicate Truth
edge value in an edge alignment rather than silently scoring an invalid
non-injective alignment.

## Assignment algorithm

The existing small `_max_weight_assignment()` Hungarian/Kuhn–Munkres
primitive is reused.  No experiment module is imported into production, and
no joint structural matcher was moved back into the score path.

Candidate edges are grouped by their already mapped endpoint pair.  The
candidate graph is disconnected across endpoint groups, so solving each group
independently is exactly equivalent to solving one global assignment over the
union of all eligible pairs.

For every eligible pair, the assignment weight is:

```text
2 + existing_condition_slot_score
```

The `2` is larger than the maximum condition score (`1`).  This gives the
lexicographic objective:

1. maximize the number of endpoint-compatible edge assignments;
2. among assignments with that cardinality, maximize the existing condition
   correctness count.

Thus the fix is a global maximum-weight one-to-one assignment (decomposed by
independent endpoint groups), not a greedy first-match algorithm.

The assignment reads only information already scored by the primary evaluator:

- mapped endpoint equality (hard eligibility);
- Agent/Truth condition slot compatibility through `_score_scalar_slot()`.

It does **not** inspect edge ids as business identity, labels, descriptions,
evidence text, LLM output, stakeholder private mappings, or any new semantic
signal.  It therefore does not artificially maximize node/slot scores by
looking through unrelated Truth fields.

## Parallel edges, ties, and invariance

For parallel Truth edges with the same endpoints, condition compatibility is
the tie-break/secondary weight.  For example:

```text
Truth: T-approved  A -[approved]-> B
       T-rejected  A -[rejected]-> B
Agent: E-approved  A' -[approved]-> B'
       E-rejected  A' -[rejected]-> B'
```

is paired by condition and obtains `condition_correctness = 1.0`.

IDs are sorted only for deterministic traversal and Hungarian tie
serialization.  They are not a business identity signal.  If two complete
assignments are genuinely symmetric, they have the same edge cardinality and
condition total, so aggregate business metrics are identical regardless of
which symmetric pairing is represented in diagnostics.  Reversing Agent or
Truth dictionary insertion order and renaming local Agent edge ids therefore
does not change metrics.

The regression suite covers:

- duplicate Agent edge;
- duplicate Truth edge;
- parallel conditioned edges;
- one correct plus one wrong-condition duplicate;
- Agent/Truth insertion-order reversal;
- Agent edge-id rename;
- non-parallel score parity;
- canonical structural-edge exclusion;
- public tool count/schema surface;
- artifact Truth fingerprint non-mutation.

## Metrics on the adversarial fixtures

The new expected metrics are:

| Fixture | edge recall | edge precision | condition correctness | fabricated edges |
| --- | ---: | ---: | ---: | ---: |
| one Truth edge, two duplicate Agent edges | `1.0` | `0.5` | `1.0` for known-unconditional duplicates | `1` |
| two duplicate Truth edges, one Agent edge | `0.5` | `1.0` | `1.0` for known-unconditional edge | `0` |
| two parallel conditioned edges, matching Agent pair | `1.0` | `1.0` | `1.0` | `0` |
| one conditioned Truth edge, correct + wrong-condition Agent duplicate | `1.0` | `0.5` | `1.0` | `1` |

The first fixture's old result was `recall=1.0`, `precision=1.0`,
`fabricated=0`; the new result correctly penalizes the unmatched duplicate.

## Stakeholder reference impact

The shared comparator is used by both primary Agent scoring and the
Stakeholder→Truth reference lane, so the reference path was audited separately.

Stakeholder edge mappings are not lexical.  `knowledge.py` constructs one
opaque local edge id per retained final edge and serializes
`edge_truth_ids={local_edge_id: truth_edge_id}` from the one-to-one
`edge_to_local` construction.  `_reference_truth_mappings()` then accepts a
mapping only when that private Truth id exists and its mapped endpoints agree.
Shortcut edges have no direct Truth edge id, are skipped by that adapter, and
remain in shortcut provenance with `direct_business_edge_credit = false`.

No valid generated Stakeholder mapping changed.  The shared comparator now
rejects a duplicate Truth value in an explicitly supplied edge alignment, which
hardens the common invariant without changing valid Stakeholder or shortcut
semantics.

The stored real-artifact reference scores remained:

```text
seed 9002: aggregate reference score 0.7958333333333334, edge R/P 1.0/1.0
seed 9003: aggregate reference score 0.7958333333333334, edge R/P 1.0/1.0
seed 9004: aggregate reference score 0.7958333333333334, edge R/P 1.0/1.0
```

The existing shortcut regression
`test_shortcut_never_receives_exact_credit_even_if_truth_has_direct_edge`
continues to pass.

## Real-LLM artifact impact

Sources:

- `artifacts/business_interview_real_llm/run_00_seed9002.json` and `.private.json`
- `artifacts/business_interview_real_llm/run_00_seed9003.json` and `.private.json`
- `artifacts/business_interview_real_llm/run_00_seed9004.json` and `.private.json`

These are legacy split artifacts without `evaluation_inputs`; their stored
Truth/Knowledge payloads and missing seed provenance remain unchanged.  The
`before` column below is the stored pre-fix `evaluator_metrics`; the `after`
column is a fresh evaluation of the same saved graph inputs with the fixed
matcher.  Offline `evaluate_artifact()` was also run for all three seeds and
reported `metric_parity.status = matched`.

| seed | primary aggregate field | reward before → after | edge recall before → after | edge precision before → after | condition before → after | fabricated edges before → after | structural pass before → after | quality pass before → after |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 9002 | no primary `aggregate_score` field; `structural_pass=False`, `quality_pass=False` | `0.0 → 0.0` | `0.8333333333 → 0.8333333333` | `0.7142857143 → 0.7142857143` | `0.5 → 0.5` | `2 → 2` | `False → False` | `False → False` |
| 9003 | no primary `aggregate_score` field; `structural_pass=False`, `quality_pass=False` | `0.0 → 0.0` | `1.0 → 1.0` | `1.0 → 1.0` | `0.5 → 0.5` | `0 → 0` | `False → False` | `False → False` |
| 9004 | no primary `aggregate_score` field; `structural_pass=False`, `quality_pass=False` | `0.0 → 0.0` | `1.0 → 1.0` | `1.0 → 1.0` | `1.0 → 1.0` | `0 → 0` | `False → False` | `False → False` |

`EvaluationResult` has no primary scalar `aggregate_score`; aggregate scoring
is a Stakeholder reference diagnostic.  The stored tau2 `reward_info.reward`
was `0.0` for all three artifacts.  As an additional after-check, the saved
conversation/tool calls were replayed through the current
`EnvironmentEvaluator` with `strict_replay=False`: seeds 9002, 9003, and 9004
all again produced scalar reward `0.0`, with `assert_graph_reconstructed=False`.
Thus the scalar reward did not change either.

The after edge mappings were one-to-one in every artifact.  Seed 9002 had two
unmatched Agent edges, while seeds 9003 and 9004 had none; this agrees with the
stored pre-fix results, so no real artifact contained a duplicate-credit case.
There was no unintended mapping drift.

## Golden migration and provenance

For the edge-only fix recorded here, no existing golden score was updated:
the real artifacts were byte/value-equal for the measured primary and
reference fields, and the corrected synthetic duplicate-edge scores were
encoded as regression expectations rather than compatibility hacks.  A later,
independent Node-matching change legitimately migrated the primary
`evaluator_metrics` values.  The current Node-matching comparison is stored
separately (without overwriting those legacy fields) and its impact is recorded
in `docs/business-interview-node-matching-correctness.md`.

`tools.py`, `graph.py`, `artifact_provenance.py`, and simulator code were not
modified.  The existing 21-tool schema tests and artifact round-trip/fingerprint
tests pass.  The edge tests also verify that evaluating a graph does not
change its serialized Truth fingerprint.  The current Node-matching change
leaves the legacy public `evaluator_metrics` fields untouched and records
stored/current Node and Edge comparisons in
`artifacts/business_interview_real_llm/seed_9002_9003_9004_node_matching_comparison.json`.
Truth, Knowledge, private provenance, and source artifacts are not rewritten.

## Validation and remaining risks

The following is the validation snapshot for the edge-only fix:

Targeted business-interview and experiment suites after the fix:

```text
234 passed
0 failed
0 xfailed / 0 unexpected xfail
0 collection errors
```

The additional affected generic LLM-metrics suite also passed independently:

```text
19 passed
0 failed
0 xfailed / 0 unexpected xfail
0 collection errors
```

The pre-fix duplicate-Agent regression intentionally failed with the old
`edge_precision=1.0` result before the production change.  Ruff and primary
LSP diagnostics are clean for changed files; only the repository's existing
`audioop` deprecation and unknown pytest config warnings remain.

The upstream Node-alignment risk described in the original edge-fix audit is
now addressed by the separate business-identity-first matcher documented
in `docs/business-interview-node-matching-correctness.md`.  This edge fix still
consumes the finalized Node mapping and guarantees finite Truth-edge usage
without using Edge conditions to choose Node identity.  Parallel edges with
genuinely identical endpoints and identical condition score remain
structurally ambiguous, but their aggregate metrics are intentionally
symmetric and invariant.
