# business_interview (v8 — graph context, provenance-only)

The agent interviews a stakeholder to discover an unknown team's business
process. The stakeholder's knowledge is a **physical projection of the Truth
graph** (semantic claims at workflow positions + concept views for visible
concepts only); the stakeholder LLM **realizes** the claims it chooses into
natural language. The Interview Agent builds its own typed **glossary** and an
inferred **BusinessProcessGraph** with declared start/end; the evaluator scores
correctness **only through private provenance** — no semantic NLP over Agent
or Observation text.

    TruthClaim (graph position + property + value)
       ├─ public natural language (realized by the stakeholder) ─▶ Agent concept
       └─ private assertion (claim_id + exact message span) ─▶ grounding
                                        ▲
    ConceptRef ─▶ EvidenceRef ─▶ Observation span ┘

Private **semantic dialogue events** (concept-identity alignments and
terminology confirmations) are distinct from ordinary TruthClaims: the
stakeholder emits them only when the reply genuinely performs that dialogue
act, and they are the ONLY way to authorize ``confirm_concept`` /
``mark_concept_unknown`` / ``mark_concept_disputed`` /
``record_terminology_agreement``.

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

**Graph context is a scoring prerequisite**: a node's visible property claims
are credited only when the reconstructed incoming context covers the complete
visible Truth incoming context (merge nodes need every visible incoming
relation, rework back-edges participate; hidden edges are never required).
Missing required incoming topology prevents contextual claim credit.

## Visibility-safe StakeholderKnowledge

The knowledge is built by ``project_knowledge`` and contains ONLY:
- visible TruthClaims;
- graph contexts for nodes the stakeholder can talk about (incoming edges
  restricted to visible relations, start flag);
- concept views for exactly the concepts those visible claims reference.

Hidden concepts have no lexical view and hidden relations never appear in the
knowledge or in the rendered stakeholder prompt.

## Glossary (mentions, not terminology)

`create_concept(concept_id, kind, label, evidence?)` — one concept per
business thing, with a kind: activity / actor / system / data / condition /
rationale. `label` is a working `display_label` (never evaluated).

- `add_concept_mention(concept_id, evidence)` — record Observation spans the
  Agent believes refer to the concept. **A mention is not terminology.**
- `record_terminology_agreement(concept_id, term, evidence)` — record an
  explicit agreement ONLY when the stakeholder confirmed the proposed term:
  the evidence must correspond to a private ``TerminologyConfirmation`` event
  with the same proposed term (evaluator additionally requires the event's
  Truth concept == the concept's bound Truth concept and the cited span).
- `merge_concepts(target, sources)` — repair a mistaken split (same-kind only).
- `confirm_concept(concept_id, evidence, partial=False)` — **genuine**
  confirmation: evidence must correspond to a private `ConceptAlignmentAssertion`
  (act `confirm`, or `partial` when ``partial=True``) — an ordinary mention in
  workflow speech can never authorize confirmation. The evaluator additionally
  requires the event's Truth concept to be the concept's bound Truth concept.
- `mark_concept_unknown(concept_id, evidence)` / `mark_concept_disputed(
  concept_id, evidence)` — require private alignment events with act
  `unknown` / `dispute` (disputed: events from >= 2 distinct Observations).

Concepts start `hypothesized`; `finish_interview` refuses while any
**referenced** concept is unresolved.

## Evidence (`EvidenceRef`)

`{"observation_id", "quote", "occurrence"}` resolves to a concrete character
span of the immutable Observation. Grounding uses **span correspondence**
(containment) against the private assertions **globally**: for one evidence
span the evaluator determines ALL claims its covered assertions stand for
(equal spans win; otherwise maximal contained + strictly containing
assertions). If a span covers several **independent** claims it is **globally
ambiguous and grounds nothing at any slot** — a broad clause such as "I check
customer information in CRM" cannot independently ground activity + system +
data; the Agent must cite narrower evidence, unless the private sidecar
represents the clause as ONE inseparable assertion (then it supports exactly
that relation). No semantic similarity, deterministic only.

## Process graph

`add_node` / `update_node` / `remove_node` / `add_edge` / `update_edge` /
`set_graph_endpoints` / `validate_graph` / `finish_interview`. Structural
ConceptKind rules are enforced (activity->activity, actor->actor, system->
system, reads/writes->data, rationale->rationale, condition->condition).
Cycles are valid; a successful `finish_interview` terminates the episode
immediately (`EPISODE_COMPLETE`, distinct from `max_steps`).

`start_inference` may reset the inferred graph, the glossary and the
completion state, but **preserves already captured Observations and the
conversation ledger**: Observations are immutable primary evidence and stay
valid as evidence after a restart.

## Node identity uses topology

An agent node maps to a Truth node through a deterministic consistent
assignment of **activity provenance + reconstructed incoming topology**: every
agent edge with edge-existence provenance must land on a Truth edge between
the mapped endpoints whose `edge_exists` claim is supported, with conditions
consistent. The same activity at several positions is disambiguated by
topology.

## Private provenance

The stakeholder LLM returns `{"message", "assertions", "alignments",
"terminology"}`; only the message enters the conversation. At ingestion the
environment validates deterministically (claim exists / visible /
quote+occurrence exactly match the message; dialogue events reference a
visible concept) and stores everything privately against that exact turn.
Provenance is never reconstructed from text; claim ids, truth concept ids and
event metadata never appear in Agent-visible messages, tools, state or
artifacts.

## Evaluation (`evaluation.py`)

Semantic grounding follows only:

    ConceptRef -> EvidenceRef -> Observation span
        -> private assertion -> TruthClaim -> Truth BusinessConcept

- Node/edge correspondence via provenance + topology (above).
- **Graph context prerequisite**: property claims credited only when the
  complete visible incoming context is reconstructed (above).
- Concept identity per kind (conditions included): every referenced agent
  concept binds exactly one Truth concept of its own kind; one visible Truth
  concept is represented by one Agent concept; reuse across slots required.
- Visibility stays prior: hidden unset -> correct, hidden asserted ->
  incorrect; provenance never rescues a hidden assertion.
- Glossary validation: referenced concepts resolved; `confirmed` /
  `partially_confirmed` / `unknown` / `disputed` and terminology agreements
  backed by the appropriate private dialogue events for the concept's bound
  Truth concept; no bulk self-confirmation.
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
