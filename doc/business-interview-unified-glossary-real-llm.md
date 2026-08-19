# business_interview — Unified Glossary + Provenance-Only Evaluation: Real-LLM Validation

**Date:** 2026-08-19
**Branch:** `business-interview`
**Mechanism:** `refactor: unify business concepts and provenance`

## What was validated

The agent now builds a typed **glossary** (BusinessConcept of kind activity /
actor / system / data / condition / rationale, starting `hypothesized`) and an
inferred **BusinessProcessGraph**; the stakeholder answers only from hidden
**atomic StakeholderFacts** and returns a private **assertion sidecar**
(`[{fact_id, quote, occurrence}]`) whose spans must exactly match its own
message. The evaluator grounds everything through

    ConceptRef -> EvidenceRef -> exact Observation span
        -> private assertion -> StakeholderFact -> TruthClaim
        -> Truth BusinessConcept

and performs no semantic NLP: no action expressions, no predicate/necessity
matchers, no actor/system aliases, no label/description/term comparison.
Cycles are valid. Edge existence and conditions are provenance-grounded.
Interview completion requires every referenced concept to be resolved.

## Runs (new seeds, DeepSeek for agent + stakeholder)

`scripts/business_interview_real_llm_smoke.py --runs 4 --seed-base 7500`
(artifacts: `artifacts/business_interview_real_llm/run_0{0..3}_seed750{0..3}.json`

+ separate `.private.json` assertion ledgers).

| seed | node r/p | act | actor | sys | read | write | concept | glossary | evidence | leaks | errors |
| ------ | ---------- | ----- | ------- | ----- | ------ | ------- | --------- | ---------- | ---------- | ------- | -------- |
| 7500 | 0.17/0.14 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.0 | True | True | 0 | 0 |
| 7501 | 0.17/0.14 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.0 | True | True | 0 | 0 |
| 7502 | 0.33/0.29 | 1.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.0 | True | True | 0 | 0 |
| 7503 | 0.00/0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 | True | True | 0 | 0 |

All runs: zero mechanism errors, zero private-ID leakage (conversation, tool
calls, observations, glossary labels/descriptions/terms, DB ledger and
evaluator metrics scanned for every fact/claim id), `glossary_pass` and
`evidence_pass` True, protocol completed. Only **one** sidecar retry across all
four runs — the assertion contract is reliably produced. Runs terminate at
`max_steps` (the interviewer asks many questions).

## Checks against the objective

+ **Glossary quality/reuse** — the agent consistently created and reused the
  `sales` actor concept across nodes and maintained one activity concept per
  step; activity claims bind at 1.00 in three of four runs. Weakness: the
  agent mis-categorized some concepts (e.g. `customer` as an `actor` instead
  of `data`, invented `accounting_team` actor), which the per-kind concept
  identity rule correctly rejects (`concept_correctness` 0).
+ **Confirmations** — the agent resolved every referenced concept
  (`glossary_pass` True in all runs; no `referenced_hypothesized_concepts`),
  so `finish_interview` never blocked.
+ **Graph quality** — partial node mapping (0–2 of 6 nodes). The dominant
  limiter is exact-span coordination: the agent's `EvidenceRef` quotes must
  equal the stakeholder's private assertion quotes verbatim; the two LLMs
  rarely pick the same span for actor/system/data claims (activity spans align
  thanks to generous multi-span annotation).
+ **Leakage** — 0 leaks in every run; fact/claim ids appear only in the
  `.private.json` sidecar artifacts.
+ **Unsupported/hidden assertions** — unsupported refs are counted
  (`unsupported_ref_count`); the agent asserted hidden properties (e.g.
  `sq.reads`) and the hidden prior gate penalized them.
+ **Termination** — `max_steps` in all runs (interviewer over-asks).
+ **Paraphrase** — the stakeholder's wording differs materially from the fact
  texts, e.g. *"The process starts when we receive a quotation request from a
  customer. That's what triggers it, and we're the ones who receive it."* vs
  fact *"You receive quotation requests from customers."*; the sidecar still
  credits `quotation.receive.activity` / `.actor`.
+ **Multi-fact response** — one utterance carried four assertions across three
  facts, e.g. *"After checking the customer's information in the CRM, I create
  the quotation using the customer and pricing information."* asserting
  `create.activity`, `create.actor`, `create.reads.customer` and
  `create.reads.pricing`; exact-span matching keeps them distinct (a ref
  citing one span cannot cross-credit another fact's claim — proven
  deterministically in the suite).

## Remaining failures

+ Reward 0.0 in all runs: node/edge reconstruction is incomplete because of
  (a) exact-span coordination between the stakeholder's private assertion
  quotes and the agent's `EvidenceRef` quotes, and (b) interviewer glossary
  mis-categorization (customer-as-actor etc.). The mechanism is verified
  deterministically; these are agent-quality limitations of the architecture's
  exact-span contract, which the objective forbids weakening (no aliases,
  no semantic matching).

## Deterministic tests

`tests/test_domains/test_business_interview/` — 62 passed, covering every
objective proof: cycles are valid; all six ConceptKinds use Agent-local
concepts; arbitrary labels bind through provenance; paraphrasing
labels/descriptions does not change the semantic score; identical text with
different private provenance grounds differently; invalid quote/occurrence is
rejected at ingestion and at evaluation; multi-fact utterances cannot
cross-credit unrelated concepts; reuse passes; split identity and incompatible
merges fail; hidden assertions fail regardless of provenance; edge existence
and condition are provenance-grounded; private ids cannot leak; hypothesized
referenced concepts block completion; confirmed/unknown/disputed can complete
when grounded; EN/JA/lab use the same mechanism; the old semantic matcher
machinery (and `BusinessDAG`, `DataConcept`, primitives resolver, role/system
alias normalization) is gone.
