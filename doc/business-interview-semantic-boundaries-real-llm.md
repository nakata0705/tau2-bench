# business_interview — semantic boundary tightening (real-LLM validation)

Goal report for the "tighten semantic boundaries after the graph-context
refactor" work: StakeholderKnowledge as a physical projection, private
semantic dialogue events, graph-context prerequisite for claim scoring,
global span ambiguity, cleaned task prose, and the Observation lifecycle.

All deterministic: **no evaluator semantic NLP**, no aliases, no lexical
matchers. Backward compatibility was not required and was not preserved.

## 1. Visibility-safe StakeholderKnowledge (physical projection)

`StakeholderKnowledge` is now built ONLY by `project_knowledge`
(`facts.py`), which projects the Truth graph through the stakeholder filter:

- `visible_claim_ids` — claims built by `build_claims` (hidden
  contexts/properties produce no claims);
- `contextual_knowledge` — contexts for exactly the nodes the stakeholder
  can talk about, with `incoming_edge_ids` restricted to visible relations
  and the start flag;
- `concept_views` — wordings for exactly the concepts those visible claims
  reference. Hidden concepts have **no lexical view**;
- `visible_node_ids` / `visible_edge_ids` — the projection's own node/edge
  sets (relations render only from these).

`StakeholderUserSimulator._knowledge_block` renders positions, relations and
views **from the knowledge only** (never from the Truth graph), so hidden
concepts/relations cannot leak into the stakeholder prompt. `_view` no longer
falls back to the private concept id.

Regression tests: `test_lab_knowledge_is_a_physical_projection_no_hidden_concepts`
(lab_sample_flow hidden read/write concepts and their labels absent from the
knowledge AND the rendered prompt), `test_quotation_knowledge_contexts_use_only_visible_relations`,
`test_lab_hidden_assertions_never_enter_visible_claim_ids`.

## 2. Private semantic dialogue events

Ordinary `TruthClaim`s express business facts; they never prove concept
identity or terminology. Two new **private dialogue events** live in the
sidecar ledger (`facts.py`), distinct from `StakeholderAssertion`:

    ConceptAlignmentAssertion  {truth_concept_id, quote, occurrence, act}
                               act: confirm | partial | unknown | dispute
    TerminologyConfirmation    {truth_concept_id, proposed_term, quote, occurrence}

The stakeholder simulator emits them **only when the reply genuinely
performs that dialogue act** (answering "Yes." to "do you mean X?" /
"can we call this Y?"). Merely using a word in workflow speech creates
neither event. The sidecar contract and retry hint spell this out; the
catalog validates the metadata deterministically (visible concept, exact
span).

## 3. Enforcement in tools + evaluator

- `confirm_concept(concept_id, evidence, partial=False)` requires the cited
  span to correspond to a `ConceptAlignmentAssertion` (act `confirm`, or
  `partial` when `partial=True`). The evaluator additionally requires the
  event's `truth_concept_id` to be the concept's **bound** Truth concept.
- `mark_concept_unknown` / `mark_concept_disputed` require `unknown` /
  `dispute` events (disputed: events from >= 2 distinct Observations).
- `record_terminology_agreement(concept_id, term, evidence)` requires a
  `TerminologyConfirmation` with `proposed_term == term` at a corresponding
  cited span; the evaluator also checks the event's Truth concept equals the
  bound concept and the cited Observation/span corresponds. The old behavior
  (any authentic mention authorizes an agreement) is deleted.
- The anti-bulk rule (one span backs at most one concept) is kept.

Regression tests: `test_ordinary_mention_cannot_confirm_concept`,
`test_mention_is_not_terminology` (ordinary mention "quotation" cannot
authorize an agreement for "the offer document"; a genuine event can),
`test_unknown_and_disputed_require_evidence_and_can_complete`,
`test_partially_confirmed_can_complete` (a plain confirm event cannot
authorize `partially_confirmed`), `test_confirmation_evidence_must_match_concepts_claims`.

## 4. Graph context prerequisite for claim scoring

`TruthClaim = graph position + property + value` — a node's visible property
claims are now credited only when the reconstructed incoming context covers
the **complete visible Truth incoming context**:

- every visible incoming Truth edge of the mapped node must be matched by an
  agent edge (with edge-existence provenance) mapped onto it;
- merge nodes therefore require every visible incoming relation;
- rework back-edges in cycles participate (they are part of the incoming
  context);
- the same activity at several positions stays distinguishable (topology
  assignment, plus the gate);
- hidden edges are never required.

`evaluate()` computes `context_ok` per mapped node and `_property_score`
returns no credit when it fails. Tests:
`test_missing_required_incoming_topology_blocks_contextual_claim_credit`,
`test_missing_incoming_edge_alone_does_not_void_other_nodes`,
`test_same_activity_distinct_positions_remain_distinguishable_with_gate`
(cycle with rework back-edge; dropping the back-edge voids credit).

## 5. Global span ambiguity (no cross-credit)

Ambiguity is no longer slot-local. For one EvidenceRef/span the evaluator
determines **all** claims its covered assertions stand for
(`_covered_claims_for_span`):

- equal-span assertions win (the span is the canonical span of those claims);
- otherwise: maximal assertion spans contained in the evidence span + spans
  strictly containing it.

A span covering several independent claims is **globally ambiguous and
grounds nothing at any slot** — the Agent must cite narrower evidence. The
exception is the sidecar representing the span as ONE assertion (then it
supports exactly that relation). Ambiguity is counted across every agent ref
(mapped or not).

Regression tests: `test_broad_clause_cannot_independently_ground_activity_system_data`
(the goal's "I check customer information in CRM" case: the clause grounds
the activity's own assertion but not system/data),
`test_broad_clause_over_several_independent_clauses_grounds_nothing`.

## 6. Task prose has no business facts

`known_info == ""` was not enough. Removed scenario-specific facts from
`unknown_info`, the disclosure instructions and `task_instructions`
("high-value approval", "month-end summary", "credit risk", "Accounting"
branch mappings) and from the task `description`; stale DAG terminology
replaced with "directed process graph". The agent policy now explains the
evidence contract (one span = one claim; edge relations vs. condition
phrases; confirmation spans) without hard-coding domain terms.
Test: `test_task_prose_contains_no_scenario_business_facts`.

## 7. Observation lifecycle

`start_inference()` resets the inferred graph, glossary and completion state
but **preserves already captured Observations, the conversation ledger, and
the private sidecar ledger**. Test:
`test_start_inference_preserves_observations_and_ledger` (observe ->
start_inference -> same observation still valid as evidence).

## 8. Deterministic suite

`tests/test_domains/test_business_interview/` — **85 passed**.

## 9. Real-LLM validation (DeepSeek, quotation_workflow_1)

Fourteen new seeds ran across five batches (9100-9103 pre-policy; 9200-9203
and 9300-9303 with the updated policy + edge-claim rendering; 9400-9401 and
9500-9501 with the explicit relation-claim contract). Inspection points:

- **Termination**: mostly `max_steps`/`too_many_errors`; several runs aborted
  on repeated invalid stakeholder sidecars (misquoted spans or invented
  claim ids). No run finished the protocol.
- **Tool errors**: the mention-only `confirm_concept` rejection fires in real
  runs (requirement-3 enforcement active); **no "observation not found"
  errors anywhere** (requirement-7 fixed the graph-context run's recurring
  obs-reset errors).
- **Node/edge recall**: node_recall reached 1.0 (seeds 9201, 9500) after the
  policy update; edge_recall first became non-zero (0.167) once the sidecar
  contract required relation claims — the stakeholder then asserted
  `e1/e2/e3/e4/e5.edge_exists` — but the agent still cites full clauses as
  edge evidence (e.g. "After I receive the quotation request, I check the
  customer information"), which the global span rule correctly rejects as
  covering the relation + activity + system claims. The stakeholder's edge
  assertion spans also tend to overlap the activity/condition phrases, so
  exact-span citation is required.
- **Concept correctness/glossary**: concepts stay hypothesized because
  confirmations stall (agent cites mention spans; or runs hit step/error
  caps); `glossary_validation_errors == 0` on every run (events are clean
  when used).
- **Terminology confirmation**: no agreements recorded (the agent did not
  propose-and-confirm terms).
- **Private-id leakage**: `[]` on every run.
- **Dialogue events**: the sidecar emitted `alignments` only for genuine
  identity questions ("Yes, that's correct", "the quotation I create ... is
  the same one I send to the customer") — the mention-vs-event separation
  works in the live pipeline.

## Remaining architectural issues

1. **Stakeholder edge-claim assertion completeness**: the stakeholder LLM
   often describes a relation and its condition in one sentence and asserts
   the condition + activity claims but omits the relation's own claim
   (``e3.edge_exists``) — so the agent's edge evidence covers only condition/
   activity phrases and grounds nothing. The sidecar contract now explicitly
   requires asserting relation claims anchored to the phrase expressing the
   relation; results in the 9500 batch.
2. **Agent evidence citation style**: LLM agents cite whole clauses; the
   global one-span-one-claim rule requires exact claim phrases. The policy
   now teaches this (nodes, edges/conditions, confirmations); a follow-up
   could add per-claim evidence retrieval guidance.
3. **Stakeholder sidecar precision**: the stakeholder LLM occasionally
   misquotes spans or invents claim ids ("me.condition"); strict ingestion
   rejects these and, after one retry, aborts the episode. Robustness
   follow-up: recoverable soft-retry or a second retry with a corrected
   example.
4. **Step budget**: interviews with strict evidence rules take longer;
   `max_steps`/`max_errors` caps were hit before completion. Not an
   evaluator issue.
