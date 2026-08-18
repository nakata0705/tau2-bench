# business_interview — Private Stakeholder-Fact Provenance: Real-LLM Validation

**Date:** 2026-08-19
**Branch:** `business-interview`
**Mechanism:** `refactor: use private stakeholder fact provenance`

## What was validated

The stakeholder simulator now answers **only from hidden structured
`StakeholderFact`s** and returns a **private `used_fact_ids` sidecar** alongside
its natural-language message. The environment stores the sidecar privately
against that exact message's turn; the evaluator grounds every visible
`ConceptRef` through

    ConceptRef -> Observation ids -> Observation.turn
        -> private used_fact_ids -> supported TruthClaims -> Truth data concept

No surface-term tables, stop phrases, or text parsing remain; the evaluator
never inspects `preferred_label`, `ConceptTerm.text`, or `Observation.text`.

## Runs (new seeds, DeepSeek for agent + stakeholder)

`scripts/business_interview_real_llm_smoke.py --runs 4 --seed-base 7000`
(artifacts: `artifacts/business_interview_real_llm/run_0{0..3}_seed700{0..3}.json`

+ separate `.private.json` sidecar ledgers).

| seed | node r/p | edge r | read | write | concept | necessity | leaks | errors |
| ------ | ---------- | -------- | ------ | ------- | --------- | ----------- | ------- | -------- |
| 7000 | 0.83 / 1.0 | 0.83 | 0.40 | 0.80 | 0.0 | True | 0 | 0 |
| 7001 | 1.00 / 1.0 | 0.83 | 0.50 | 1.00 | 1.0 | True | 0 | 0 |
| 7002 | 1.00 / 1.0 | 1.00 | 0.33 | 0.83 | 0.0 | False | 0 | 0 |
| 7003 | 0.83 / 1.0 | 0.83 | 0.40 | 0.80 | 0.0 | True | 0 | 0 |

All four runs: protocol completed, zero mechanism errors, zero private-ID
leakage (conversation, tool calls, observations, summaries, DAG, DB ledger and
evaluator metrics scanned for every fact/claim id). Runs terminate at
`max_steps` because the DeepSeek interviewer asks many clarifying questions
(211–216 messages) — an agent-strategy issue, not a mechanism issue.

## Checks against the objective

+ **Stable Agent-local concepts** — the agent created and reused local
  concepts consistently (e.g. `customer_information` at both `cc.reads` and
  `cq.reads`; seed 7001 reaches `concept_correctness = 1.0` with all six nodes
  reconstructed and all visible writes bound).
+ **Wording variation handled by the Agent LLM** — stakeholder replies differ
  materially from the fact texts, e.g. *"The process starts when we receive a
  quotation request from a customer. The goal is to produce an accurate
  quotation for them."* vs fact *"You receive quotation requests from
  customers."*; the private sidecars still record the right facts
  (`quotation.receive_request`, ...).
+ **No private-ID leakage** — every run's `private_id_leakage == []`; fact and
  claim ids appear only in the `.private.json` sidecar artifacts.
+ **Concept reuse works** — one `customer_information` concept reused across
  visible slots binds to one Truth concept (seed 7001).
+ **Unsupported refs remain meaningful** — refs whose citations support no
  claim at their slot are counted as unsupported (`unsupported_concept_ref_count`).
+ **Hidden-read epistemic errors remain detectable** — the agent asserted
  hidden reads (`ap.reads`, `sq.reads`, `me.reads`) in every run; the hidden
  prior gate penalized them (`read_correctness < 1.0`) even though the cited
  observations carried valid provenance — private provenance never rescues a
  hidden assertion.

## Remaining failures

+ Reward is 0.0 in all runs: `assert_dag_reconstructed` (structural_pass) is
  unmet because the interviewer agent (DeepSeek) over-asserts hidden reads,
  occasionally misses nodes/edges, and exhausts `max_steps`. This matches the
  pre-refactor smoke behavior (previous artifacts also show reward 0.0) and is
  an agent-quality issue, not a provenance-mechanism issue.
+ Seed 7002 fabricates the month-end necessity rationale (necessity False) —
  agent behavior; the evaluator detects it (`fabricated_necessity`).

## Deterministic tests

`tests/test_domains/test_business_interview/` — 114 passed, including the new
spec-mandated proofs: no surface-term machinery remains, paraphrased
Observation text with unchanged provenance grounds identically, identical text
with different private provenance grounds differently, unrelated Observations
cannot support a claim, label-agnostic binding ("quotation"/"document" bind
`tc_quote`), same-concept-across-slots passes, customer+pricing merged into one
concept fails, duplicate concepts for one Truth concept fail, hidden refs fail
regardless of provenance, private ids are unobtainable through Agent tools /
serialized state, and EN/JA/lab share one mechanism.
