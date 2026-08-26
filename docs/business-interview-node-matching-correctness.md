# Business Interview primary Node-matching correctness

## Scope

This document records the deterministic AgentGraph → TruthGraph Node-matching
hardening that follows the primary edge reservation fix.  The goal is to avoid
crediting a Truth Node merely because an Agent Node shares a weak lexical or
property overlap when its structural role is incompatible.

The implementation is in:

- `src/tau2/domains/business_interview/comparison.py`

The adversarial contract tests are in:

- `tests/test_domains/test_business_interview/test_node_matching_correctness.py`

The implementation does not change concept lexical-alignment semantics, slot
scoring, epistemic states, canonical SOURCE/SINK scoring, forgetting,
contraction, Stakeholder reference semantics, tool APIs, simulator behavior,
or evidence reward logic.  Edge matching still runs only after Node matching
has completed and remains the existing one-to-one matcher.

## Reproduced false positives in the previous matcher

Before the production change, the new adversarial suite reproduced these
current-HEAD failures:

- two Nodes with the same `activity` mapped by lexical overlap even when one
  was a branch Node and the other was serial;
- an Agent Node with a different/unmapped activity but shared `actor` and
  `system` mapped to a Truth Node;
- a serial Agent Node mapped to a Truth branch or merge Node;
- symmetric branch Nodes were resolved by sorted local IDs instead of being
  treated as ambiguous;
- an isolated partial-graph Node was mapped to an internal Truth Node solely
  because its activity label matched.

The old matcher built a set of aligned `(property, Truth-concept-id)` pairs and
then ran a global assignment over any non-empty overlap.  It had no structural
candidate constraint, and a two-property actor/system overlap could therefore
be enough to create identity.  It also did not distinguish equal-optimum
symmetric assignments from unique business identity.

The pre-fix run showed the expected failures in these valid fixtures; the
merge fixture itself was corrected to include a canonical entry before it was
used as a regression test.

## Topology-first structural fingerprint

Only `business_node_ids()` and `business_edge_ids()` participate.  Explicit
canonical SOURCE/SINK nodes and boundary edges are therefore excluded from the
Node matching denominator and cannot become business candidates.

For every business Node, the deterministic base profile contains:

- business entry and exit role;
- unique predecessor and successor degree;
- `branch` / `merge` role derived from those neighbor sets;
- explicit self-loop presence;
- shortest distance from an inferred/declared business entry;
- shortest distance to an inferred/declared business exit.

Distances are computed with sorted adjacency traversal.  Parallel conditioned
edges to the same neighbor are collapsed for the Node topology profile: they
are distinct Edge evidence, not a new Node branch or merge role.  Node IDs and
dictionary insertion order are never included in the profile.

A graph with no business edges and no explicit Agent boundary is treated as
structurally unspecified rather than as proof of an isolated business Node.
That compatibility fallback still requires a matched activity and is subject
to the same one-to-one and ambiguity rules.  An explicit isolated Agent
boundary remains a meaningful topology claim and cannot map to an internal
Truth Node.

## WL-style iterative refinement

The base profile is round zero.  Each subsequent round canonicalizes:

```text
(previous own color,
 sorted predecessor colors,
 sorted successor colors)
```

Colors are assigned from the union of Agent and Truth payloads in sorted
serialized order, so corresponding structural payloads receive the same
color without relying on local IDs.  Refinement runs for up to
`min(12, max(agent_node_count, truth_node_count))` rounds and stops early when
both color maps stabilize.  The complete color history is retained as the
refinement component of the fingerprint.

Base-profile equality is the hard candidate constraint for a topology-known
Agent graph.  WL history agreement is a deterministic topology-similarity
term inside that candidate set.  This lets local topology reject branch/
serial and merge/serial mismatches while still allowing a partial graph whose
known local role agrees but whose farther context is incomplete.

## Unique anchors and constraint propagation

A candidate pair is an anchor only when:

1. both graphs have known topology;
2. the complete base-plus-WL fingerprint is unique in each graph; and
3. the already-aligned activity slot agrees.

Anchors are fixed one-to-one.  Candidate pairs are then filtered by mapped
predecessor/successor relations in both directions.  If a mapped Agent
neighbor is not a corresponding Truth neighbor, or a mapped Truth neighbor is
not a corresponding Agent neighbor, that candidate is removed.  Newly unique
fingerprints are propagated until no new anchor is available.

A structural fingerprint alone never establishes business identity; the
activity agreement remains required.

## Business-attribute assignment

Only topology-compatible candidates are weighted using existing aligned slot
scoring.  The weight is:

```text
100.0                         cardinality priority
+ 3.0 * WL topology similarity
+ 8.0 * activity agreement
+ 2.0 * actor agreement
+ 2.0 * system agreement
+ 1.0 * reads agreement
+ 1.0 * writes agreement
+ 0.5 * necessity-rationale agreement
```

An activity mismatch or absent activity agreement produces no candidate.  This
makes actor/system-only overlap insufficient while keeping activity the most
important business attribute.

The existing `_max_weight_assignment()` primitive is reused.  Candidate
bipartite graphs are decomposed into disconnected components, then each
component is solved one-to-one with cardinality-first maximum weight.  A
component larger than the bounded ambiguity limit of eight Nodes is left
unmatched rather than allowing ID order to decide a potentially symmetric
class.

## Ambiguity policy

For every selected pair in a bounded component, the exact pair is forbidden
and the assignment is solved again.  If the alternative has the same optimum,
the selected pair is not forced by the evidence and is discarded.  Therefore:

- a unique optimum is retained;
- an equal optimum is conservatively unmatched;
- symmetric assignments cannot acquire meaning from sorted local IDs;
- assignment cardinality remains one-to-one.

This is a bounded alternative check, not an unbounded graph-isomorphism
search.

## Edge evaluation boundary

Node mapping is finalized before `_map_edges_one_to_one()` is called.  Edge
conditions are not inspected while choosing Node identity.  The existing Edge
matcher then applies mapped endpoint eligibility, one-to-one Truth-edge
reservation, and condition scoring.  This preserves the intended hierarchy:

```text
Node identity → Edge identity → Edge condition
```

## Adversarial coverage

The new Node suite covers:

- same activity with wrong topology;
- same actor/system without activity agreement;
- branch versus serial;
- merge versus serial;
- WL disambiguation of same local-role Nodes;
- symmetric branch ambiguity;
- partial-graph false-positive prevention;
- normal unique-topology mapping and Edge preservation;
- insertion-order and Agent local-Node-ID invariance.

The existing primary Edge correctness suite, Stakeholder reference tests,
artifact provenance tests, and 21-tool schema tests remain separate and are
run alongside this suite.

## Real-artifact impact

The stored pre-change metrics were compared with a fresh evaluation after the
Node matcher change.  The public legacy artifact `evaluator_metrics` sections
were migrated only where values changed (seeds 9002 and 9003); seed 9004 was
already identical.  This keeps the offline parity verifier fail-closed and
green.  Truth/Knowledge payloads, private payloads, provenance metadata, and
the stored historical diagnostics files were not rewritten.

| seed | Node recall | Node precision | fabricated Nodes | Edge recall | Edge precision | condition | structural | quality |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 9002 | `1.0 → 0.3333333333` | `1.0 → 0.3333333333` | `0 → 4` | `0.8333333333 → 0.0` | `0.7142857143 → 0.0` | `0.5 → 0.0` | `False → False` | `False → False` |
| 9003 | `1.0 → 0.8333333333` | `1.0 → 0.8333333333` | `0 → 1` | `1.0 → 0.6666666667` | `1.0 → 0.6666666667` | `0.5 → 0.3333333333` | `False → False` | `False → False` |
| 9004 | `1.0 → 1.0` | `1.0 → 1.0` | `0 → 0` | `1.0 → 1.0` | `1.0 → 1.0` | `1.0 → 1.0` | `False → False` | `False → False` |

The changed Node mappings were:

- seed 9002: the old mappings for `node_create_quotation_doc`,
  `node_send_month_end_summary`, `node_receive_request`, and
  `node_send_quotation` were rejected because their local structural roles
  were incompatible with the mapped Truth Nodes.  `node_approve_high_value →
  ap` and `node_check_customer_info → cc` remained.
- seed 9003: `node_manager_approve → ap` was rejected because the activity
  did not have an aligned activity identity; the five topology-and-activity
  supported mappings remained.
- seed 9004: no Node mapping changed.

These are conservative false-positive/unsupported-identity corrections, not
local-ID or insertion-order drift.  The reference aggregate remained
`0.7958333333333334` for all three seeds.  Scalar reward remained `0.0` in the
saved artifact replay; the Node matcher changes only the graph-comparison
inputs to the existing reward pipeline.

## Validation

The complete related suite passed:

```text
247 passed
0 failed
0 unexpected xfail
0 collection errors
```

This includes the business-interview domain suite, artifact provenance tests,
experiment tests, and the affected generic LLM-metrics tests.  `make
check-all` passed, Ruff reports no issues, and primary LSP diagnostics for the
changed Python files are clean.  Only the repository's existing `audioop`
deprecation and unknown pytest configuration warnings remain.

## Remaining risk

The matcher deliberately does not claim full graph isomorphism.  A wrong
activity concept alignment can still prevent a correct Node candidate, and a
large topology ambiguity class is conservatively discarded.  The next
correctness priority is improving diagnostics for distinguishing a genuinely
missing Node from a Node whose activity concept was not aligned, without
weakening the false-positive policy.
