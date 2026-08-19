# business_interview (v7 — graph context, provenance-only)

The agent interviews a stakeholder to discover an unknown team's business
process. The stakeholder's knowledge is a **projection of the Truth graph**
(semantic claims at workflow positions + concept views); the stakeholder LLM
**realizes** the claims it chooses into natural language. The Interview Agent
builds its own typed **glossary** and an inferred **BusinessProcessGraph**
with declared start/end; the evaluator scores correctness **only through
private provenance** — no semantic NLP over Agent or Observation text.

    TruthClaim (graph position + property + value)
       ├─ public natural language (realized by the stakeholder) ─▶ Agent concept
       └─ private assertion (claim_id + exact message span) ─▶ grounding
                                        ▲
    ConceptRef ─▶ EvidenceRef ─▶ Observation span ┘

## Graph-contextual TruthClaims

`TruthClaim {context_id, property, concept_id}` — every claim belongs to a
workflow position:

    cq.activity      -> tc_activity_create_quotation
    cq.reads         -> tc_customer
    e3.edge_exists   (structural)
    e3.condition     -> tc_cond_over_1m

`TruthNodeContext {node_id, incoming_edge_ids, is_start}` is resolved from the
Truth graph and lists **ALL** incoming edges (including back edges in cycles);
the start flag restores explicit start semantics. Cycles remain valid.

## Glossary (mentions, not terminology)

`create_concept(concept_id, kind, label, evidence?)` — one concept per
business thing, with a kind: activity / actor / system / data / condition /
rationale. `label` is a working `display_label` (never evaluated).

- `add_concept_mention(concept_id, evidence)` — record Observation spans the
  Agent believes refer to the concept. **A mention is not terminology.**
- `record_terminology_agreement(concept_id, term, evidence)` — record an
  explicit agreement only when the stakeholder confirmed the proposed term.
- `merge_concepts(target, sources)` — repair a mistaken split (same-kind only).
- `confirm_concept(concept_id, evidence, partial=False)` — **genuine**
  confirmation: evidence must correspond to a private assertion of the
  concept's claims (the stakeholder actually asserted the concept's identity);
  one evidence span may back at most one concept (no bulk self-confirmation).
- `mark_concept_unknown(concept_id, evidence)` / `mark_concept_disputed(
  concept_id, evidence)` — also require stakeholder evidence (unknown: the
  stakeholder did not assert the concept; disputed: evidence from >= 2
  distinct Observations).

Concepts start `hypothesized`; `finish_interview` refuses while any
**referenced** concept is unresolved.

## Evidence (`EvidenceRef`)

`{"observation_id", "quote", "occurrence"}` resolves to a concrete character
span of the immutable Observation. Grounding uses **span correspondence
(containment)**: an evidence span that corresponds to exactly one expected
claim at its slot grounds it; a span corresponding to several expected claims
at one slot is **ambiguous and grounds nothing** (no cross-credit).

## Process graph

`add_node` / `update_node` / `remove_node` / `add_edge` / `update_edge` /
`set_graph_endpoints` / `validate_graph` / `finish_interview`. Structural
ConceptKind rules are enforced (activity->activity, actor->actor, system->
system, reads/writes->data, rationale->rationale, condition->condition).
Cycles are valid; a successful `finish_interview` terminates the episode
immediately (`EPISODE_COMPLETE`, distinct from `max_steps`).

## Node identity uses topology

An agent node maps to a Truth node through a deterministic consistent
assignment of **activity provenance + reconstructed incoming topology**: every
agent edge with edge-existence provenance must land on a Truth edge between
the mapped endpoints whose `edge_exists` claim is supported, with conditions
consistent. The same activity at several positions is disambiguated by
topology.

## Private provenance

The stakeholder LLM returns `{"message", "assertions": [{claim_id, quote,
occurrence}]}`; only the message enters the conversation. At ingestion the
environment validates deterministically (claim exists / visible /
quote+occurrence exactly match the message) and stores the assertions
privately against that exact turn. Provenance is never reconstructed from
text; claim ids never appear in Agent-visible messages, tools, state or
artifacts.

## Evaluation (`evaluation.py`)

Semantic grounding follows only:

    ConceptRef -> EvidenceRef -> Observation span
        -> private assertion -> TruthClaim -> Truth BusinessConcept

- Node/edge correspondence via provenance + topology (above).
- Concept identity per kind (conditions included): every referenced agent
  concept binds exactly one Truth concept of its own kind; one visible Truth
  concept is represented by one Agent concept; reuse across slots required.
- Visibility stays prior: hidden unset -> correct, hidden asserted ->
  incorrect; provenance never rescues a hidden assertion.
- Glossary validation: referenced concepts resolved; `confirmed` /
  `unknown` / `disputed` backed by appropriate stakeholder evidence; no bulk
  self-confirmation.
- Start/end correctness and edge conditions are scored.

Deleted machinery (no shims): `StakeholderFact` sentences, `ConceptTerm`
terminology semantics, activity-only node matching, semantic expression
matchers, primitive resolver, role/system aliases, duplicated task facts, DAG
terminology/acyclicity.

## Files

| Module | Purpose |
| -------- | --------- |
| `graph.py` | EvidenceRef / BusinessConcept (mentions) / TerminologyAgreement / Node / Edge / BusinessProcessGraph / TruthNodeContext / Observation / InterviewDB |
| `claims.py` | graph-contextual TruthClaim + `build_claims` (visibility-filtered) |
| `facts.py` | StakeholderKnowledge / StakeholderAssertion / catalog / private ledger |
| `stakeholder.py` | StakeholderFilter (per-node/per-edge property visibility) |
| `scenario.py` | Truth graphs (start/end) + concept views (EN/JA share ids) |
| `evaluation.py` | provenance-only evaluator (topology matching, span containment) |
| `tools.py` | glossary + graph tools (kinds enforced, genuine confirmation) |
| `user_simulator.py` | semantic stakeholder realization (claim-based sidecar) |
| `environment.py` | conversation ledger + private assertion binding + `episode_complete` |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/`.
