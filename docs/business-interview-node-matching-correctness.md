# Business Interview Node-matching correctness

## Scope

This document records the current AgentGraph → TruthGraph Node matcher.  The
contract is:

```text
business identity first → topology/WL disambiguation → one-to-one Node mapping
                         → one-to-one Edge mapping → Edge condition scoring
```

The implementation is in:

- `src/tau2/domains/business_interview/comparison.py`

The deterministic contracts are in:

- `tests/test_domains/test_business_interview/test_node_matching_correctness.py`
- `tests/test_domains/test_business_interview/test_edge_matching_correctness.py`

The matcher does not change Concept lexical alignment, epistemic states,
canonical SOURCE/SINK handling, Stakeholder forgetting/reference semantics,
tool APIs, simulator behavior, evidence reward, or the one-to-one Edge
matcher.

## Minimal hard-topology regression

The former matcher rejected a candidate whenever the Agent and Truth base
profiles differed.  The smallest useful reproduction is:

```text
Truth: A(receive request) → B(create quotation) → C(send quotation)
Agent: same three activity identities plus
       A → B, B → C, and an extra A → C edge
```

The extra edge changes A's branch profile and C's entry distance.  Before this
change, the current `HEAD` implementation mapped only B.  The regression was
added before the production fix and failed with:

```text
expected: a_receive → A, a_create → B, a_send → C
observed: a_create → B
```

After the fix, all three Nodes map, while the Edge metrics are
`recall=1.0`, `precision=2/3`, `fabricated_edge_count=1`.

The corresponding missing-edge and wrong-downstream-target fixtures verify
that Node identity survives both Edge recall and endpoint errors.

## Business-identity-first candidate generation

1. Concept alignment is computed by the existing content-based bijective
   alignment.  Its semantics are unchanged.
2. A Node candidate requires an **asserted, aligned activity ConceptRef on both
   sides**.  An unaligned/mismatched activity, explicit absence, or
   `DONT_KNOW` cannot create a candidate.
3. Actor, system, reads, writes, and necessity-rationale scores are supporting
   evidence only.  Actor/system overlap without activity identity cannot map a
   Node.
4. Every activity-compatible pair is retained initially, even if its local
   topology differs.
5. The existing maximum-weight one-to-one assignment is then applied to the
   candidate bipartite graph.  Cardinality is primary; business attributes
   and topology are secondary weights.

The current weight is:

```text
100.0                         cardinality priority
+ 3.0 * topology/WL similarity
+ 8.0 * activity agreement
+ 2.0 * actor agreement
+ 2.0 * system agreement
+ 1.0 * reads agreement
+ 1.0 * writes agreement
+ 0.5 * necessity-rationale agreement
```

The `100.0` term preserves the cardinality-first behavior of the existing
assignment primitive.  No Edge is inspected while selecting a Node mapping.

## Topology and WL disambiguation

The existing topology data is retained:

- entry/exit role;
- predecessor/successor degree and neighbor structure;
- branch/merge role;
- self-loop presence;
- entry and exit distance;
- deterministic WL-style predecessor/successor color refinement.

The base profile and WL history now produce a **soft topology bonus**, not a
universal candidate gate.  A local mismatch therefore lowers the bonus but
cannot erase a unique business identity.

There is no topology-based Node candidate rejection in the current matcher.
Even in an ambiguous same-activity class, all activity-compatible pairs remain
available to the one-to-one assignment; topology/WL scores can select a unique
optimum, while equal optima remain unmatched.  This avoids reducing Node
cardinality merely because several Agent Nodes share an activity and one of
their local profiles is wrong.  The only hard identity gate is the asserted,
aligned activity requirement (plus one-to-one and the bounded ambiguity
policy).

This supports, for example, `review (branch)` versus `review (serial)` through
topology/WL bonus while still matching a unique `create quotation` Node whose
Agent graph has one wrong Edge.

## Assignment and ambiguity policy

- Node assignment is one-to-one.
- Candidate components are solved independently using the existing deterministic
  maximum-weight assignment.
- Every selected pair is checked by forbidding that pair and solving again.
- Equal optimum alternatives are not given meaning by sorted local IDs; the
  pair is discarded as ambiguous.
- Symmetric same-activity branches remain unmatched.
- A component larger than the existing ambiguity bound is conservatively left
  unmatched.
- Insertion order and Agent-local Node-ID renaming do not change aggregate
  metrics or create semantic meaning for a tie.

The final Edge mapping is still performed only by `_map_edges_one_to_one()`
after Node mapping is complete.  It remains endpoint-compatible,
one-to-one, and condition-aware.  It is not part of the Node objective, so
Edge correctness is not counted once in Node identity and again in Edge
matching through a joint objective.

## Required deterministic coverage

The Node/Edge suites cover:

1. unique activity + extra Edge;
2. unique activity + missing Edge;
3. unique activity + wrong downstream target;
4. same-activity branch versus serial disambiguation;
5. symmetric same-activity branches → unmatched;
6. actor/system-only overlap with activity mismatch → unmatched;
7. insertion-order invariance;
8. Agent local-ID rename invariance;
9. duplicate/parallel/conditioned Edge one-to-one correctness;
10. Stakeholder reference parity;
11. 21-tool schema and artifact Truth-fingerprint/provenance parity.

## Real-LLM artifacts

The archived public `evaluator_metrics` fields in the legacy artifacts are
not overwritten.  Current reevaluation and mapping decisions are stored in:

`artifacts/business_interview_real_llm/seed_9002_9003_9004_node_matching_comparison.json`

The sidecar is versioned `...v2`.  Each seed row carries
`baseline_evaluator_metrics` (the value archived in the legacy artifact's
`evaluator_metrics` field at the recorded `baseline_head` commit) alongside
`current_reevaluated_metrics`, per-Agent-Node business/topology evidence, old
and new mappings, and a provenance policy stating that the source artifacts
are unmodified.

### Baseline metrics are not necessarily run-time original

The legacy `run_00_seed9002.json` / `run_00_seed9003.json`
`evaluator_metrics` fields are **stored metrics at the baseline commit**, not
necessarily the original run-time scores.  For seed 9002 in particular, the
`92120c1` hard-topology commit rewrote the stored metrics once, so:

```text
pre-hard-topology evaluator (run-time)     Node R/P = 1.0 / 1.0
92120c1 hard-topology evaluator            Node R/P = 1/3 / 1/3  <- archived
8398c6d business-identity-first evaluator  Node R/P = 1.0 / 1.0   <- current
```

Readers should treat the archived `baseline_evaluator_metrics` as “the
metrics stored in the legacy artifact at `baseline_head`”, not as the
original run-time score.  No run-time score is reconstructed here.

### Metric impact

| seed | Node recall | Node precision | fabricated Nodes | Edge recall | Edge precision | Condition | structural / quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 9002 | `1/3 → 1` | `1/3 → 1` | `4 → 0` | `0 → 5/6` | `0 → 5/7` | `0 → 0.5` | `False → False` |
| 9003 | `5/6 → 5/6` | `5/6 → 5/6` | `1 → 1` | `2/3 → 2/3` | `2/3 → 2/3` | `1/3 → 1/3` | `False → False` |
| 9004 | `1 → 1` | `1 → 1` | `0 → 0` | `1 → 1` | `1 → 1` | `1 → 1` | `False → False` |

The 9002 Node property scores are intentionally still separate from Node
identity: activity and actor remain `1.0`; system, reads, writes, rationale,
and condition retain their own scores and expose the Agent's attribute/Edge
errors rather than hiding them behind a Node miss.

### Seed 9002 per-Node decisions

| Agent Node | business identity evidence | topology evidence | old mapping | new mapping | decision |
| --- | --- | --- | --- | --- | --- |
| `node_approve_high_value` | aligned unique `approve quotation` activity | compatible base profile | `ap` | `ap` | retained |
| `node_check_customer_info` | aligned unique `check customer information` activity | compatible base profile | `cc` | `cc` | retained |
| `node_create_quotation_doc` | aligned unique `create quotation` activity | base mismatch; soft similarity retained | — | `cq` | restored |
| `node_receive_request` | aligned unique `receive request` activity | branch/distance mismatch; soft similarity retained | — | `r` | restored |
| `node_send_month_end_summary` | aligned unique `send month-end summary` activity | entry-distance mismatch; soft similarity retained | — | `me` | restored |
| `node_send_quotation` | aligned unique `send quotation` activity | exit/degree mismatch; soft similarity retained | — | `sq` | restored |

Seed 9003's `node_manager_approve` remains unmatched because its activity
concept is not aligned.  Matching its manager/rationale or topology would
violate the activity-identity rule.  Seed 9004 has no mapping change.

The Stakeholder reference aggregate remains
`0.7958333333333334` for all three seeds.  The reference lane uses its private
Truth IDs and the shared comparison arithmetic; it is not fed the Agent Node
assignment.  Tool count/schema checks and Truth/Knowledge fingerprint checks
remain unchanged and pass.

## Historical artifact handling

`evaluate_artifact()` now reports `historical_drift` with explicit stored/current
metrics instead of requiring a direct rewrite of a legacy artifact.  Strict
parity checking remains available through `_check_metric_parity()` and still
fails closed for arbitrary drift.  This separates:

- the metrics **stored in the artifact** (`baseline_evaluator_metrics`; the
  archived `evaluator_metrics` field at the recorded `baseline_head`, not
  necessarily the original run-time score);
- the current reevaluated metrics under the current evaluator;
- the comparison evidence explaining intentional evaluator changes.

## Validation snapshot

The targeted correctness and artifact suites pass after this change.  The
repository-wide `make check-all` and the core `make test` are the final gates
for the completed implementation.  Known unrelated warnings are the existing
`audioop` deprecation and the repository's unknown pytest configuration option.

## Remaining risk and next priority

The matcher is not full graph isomorphism.  Wrong or missing activity concept
alignment still prevents a Node candidate, and large/symmetric ambiguity is
conservatively unmatched.  The next priority is richer evaluator-private
diagnostics that distinguish a missing business Node from an activity concept
that failed lexical alignment, without weakening the false-positive policy.
