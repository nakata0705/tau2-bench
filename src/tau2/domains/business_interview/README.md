# business_interview (v6 — unified glossary, provenance-only)

The agent interviews a stakeholder to discover an unknown team's business
process. The stakeholder simulator answers only from hidden, structured
**StakeholderFacts** (naturalizing them); the Interview Agent builds its own
typed **glossary** and an inferred **BusinessProcessGraph**; the evaluator
scores correctness **only through private provenance** — it performs no
semantic NLP over Agent or Observation text.

    StakeholderFact
       ├─ public natural language ─▶ Agent ─▶ Agent-local glossary concept
       └─ private assertion (fact_id + exact message span) ─▶ TruthClaim
                                        ▲
    ConceptRef ─▶ EvidenceRef ─▶ exact Observation span ┘

The stakeholder simulator naturalizes facts. The Agent LLM interprets
language and maintains concept identity. Private instrumentation records Truth
provenance. The evaluator binds provenance; it does no NLP semantic matching.

## Glossary (typed, Agent-local)

`create_concept(concept_id, kind, label, description?, evidence?)` — one
concept per business thing, with a **kind**:

| kind        | what it is                              |
|-------------|------------------------------------------|
| `activity`  | what is done (a step)                    |
| `actor`     | who does it (a role)                     |
| `system`    | which system / tool is used              |
| `data`      | a business object / artifact in/out      |
| `condition` | a branch condition / threshold           |
| `rationale` | why a step is needed                     |

- `add_concept_term(concept_id, term, evidence?)` — record a later, different
  wording the Agent judges to be the same thing.
- `update_concept_description(concept_id, description)` — working notes (never
  evaluated).
- `merge_concepts(target, sources)` — repair a mistaken split (same-kind only;
  re-points every reference; merging different kinds is rejected).
- `confirm_concept(concept_id, evidence?, partial=False)` — resolve a concept
  with authentic stakeholder evidence (`confirmed` / `partially_confirmed`).
- `mark_concept_unknown(concept_id)` / `mark_concept_disputed(concept_id)` —
  resolve without confirmation.
- `list_concepts()` — see the agent's own glossary.

Concepts start **`hypothesized`**. `finish_interview` refuses while any
**referenced** concept is still hypothesized — resolve them first
(confirmed / partially_confirmed / disputed / unknown).

## Evidence (`EvidenceRef`)

Every claim cites `{"observation_id", "quote", "occurrence"}`:

- `quote` must be an **exact substring occurrence** of the immutable
  Observation; `occurrence` selects which occurrence (0-based).
- Validity is checked deterministically; the evaluator never infers what a
  quote *means*.

Observations are captured from actual stakeholder messages via
`observe_message` (stable ids `sm_1`, ...) — never written by the agent.

## Process graph (`BusinessProcessGraph`)

- `start_inference(name?)` — begin an inferred graph (destructive).
- `add_node(node_id, activity, actor?, system?, reads?, writes?,
  necessity_rationale?, evidence?)` — a node; `activity` is required and all
  properties are glossary concept ids.
- `update_node(node_id, ...)` / `remove_node(node_id)` — refine the working
  hypothesis.
- `add_edge(edge_id, from_node, to_node, condition?, evidence?)` —
  `from_node`/`to_node` are structural identities; `condition` is a condition
  concept; **edge existence needs stakeholder evidence too**.
- `update_edge(edge_id, ...)` — endpoints / condition / evidence.
- `validate_graph()` — internal structural consistency (**cycles are valid**
  and never reported as errors; there is no acyclicity check and no LoopNode).
- `finish_interview(summary?)` — refuses a structurally invalid graph and any
  referenced `hypothesized` concept.

## Private provenance

`StakeholderFact {id, text, supported_claim_ids}` — atomic facts (one
TruthClaim per fact). The stakeholder LLM returns, alongside its public
message, a private assertion sidecar:

    {"message": "...", "assertions": [
      {"fact_id": "quotation.check.system", "quote": "CRM", "occurrence": 0}]}

Only the message enters the conversation. At ingestion the environment
validates deterministically (fact exists / belongs to the stakeholder /
supported claims exist and are visible / quote+occurrence exactly match the
message) and stores the assertions privately against that exact turn
(`StakeholderFactLedger`). Provenance is **never reconstructed from text**.

## Evaluation (`evaluation.py`)

`TruthClaim {id, subject_kind (node|edge), subject_id, property, concept_id}`
covers every scored semantic property:

    cq.activity -> tc_activity_create_quotation
    cq.actor    -> tc_actor_sales
    cq.reads    -> tc_customer
    e3.edge_exists
    e3.condition -> tc_cond_over_1m

Semantic grounding follows only:

    ConceptRef -> EvidenceRef -> exact Observation span
        -> private assertion -> StakeholderFact -> TruthClaim
        -> Truth BusinessConcept

- **Node identity** comes from the node's `activity` claim binding; **edge
  identity** from from/to structure + the private `edge_exists` claim.
- **Concept identity** (per kind): every referenced agent concept must bind to
  exactly one Truth concept of its own kind (singleton candidates), one
  visible Truth concept must be represented by exactly one Agent concept, and
  reuse across slots is required. Splits, ambiguous/merged bindings,
  incompatible kinds and missing concepts all fail.
- **Visibility stays prior**: hidden unset → correct, hidden asserted →
  incorrect; private provenance never rescues a hidden assertion.
- **Evidence hygiene**: every EvidenceRef must be an exact span of an
  authentic Observation; asserted refs must carry evidence.
- **Glossary completion**: `referenced_hypothesized_concepts` blocks the
  structural pass.

Deleted machinery (no shims kept): `DataConcept`, `BusinessDAG` +
acyclicity/endpoint validation, surface terms / stop phrases, action /
actor-system / predicate / necessity semantic expression matchers, primitive
resolver, role/system alias normalization, and duplicated business facts in
`known_info` (facts are the only business-fact source).

## Files

| Module | Purpose |
| -------- | --------- |
| `graph.py` | EvidenceRef / BusinessConcept / ConceptRef / Node / Edge / BusinessProcessGraph / Observation / InterviewDB |
| `claims.py` | TruthClaim + `build_claims` (visibility-filtered) |
| `facts.py` | StakeholderFact / StakeholderAssertion / catalog / private ledger |
| `stakeholder.py` | StakeholderFilter (per-node/per-edge property visibility) |
| `scenario.py` | Truth graphs + atomic facts (EN/JA share fact/claim ids) |
| `evaluation.py` | provenance-only evaluator |
| `tools.py` | glossary + graph tools |
| `user_simulator.py` | fact-grounded stakeholder (assertion sidecar) |
| `environment.py` | conversation ledger + private assertion binding |

Run `business_interview_user` as the user implementation; the deterministic
suite is `tests/test_domains/test_business_interview/`.
