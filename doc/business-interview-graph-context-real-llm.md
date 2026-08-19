# business_interview — Graph-Contextual TruthClaims + Semantic Realization: Real-LLM Validation

**Date:** 2026-08-19
**Branch:** `business-interview`
**Mechanism:** `refactor: ground interview semantics in graph context`

## What was validated

Business facts are **pure semantic structure** — graph-contextual TruthClaims
(position + property + value) with `TruthNodeContext` (ALL incoming edges +
start flag) resolved from the Truth graph. `StakeholderFact.text` sentences are
gone; the stakeholder LLM receives the semantic knowledge (positions, claims,
concept views) plus question/history and **realizes** the claims it chooses
into natural language, returning a private sidecar
`[{claim_id, quote, occurrence}]`. Mention != terminology (mentions +
`TerminologyAgreement`); `confirm_concept` requires genuine stakeholder
evidence (no bulk self-confirmation); ConceptKind rules are enforced; a
successful `finish_interview` terminates the episode immediately
(`EPISODE_COMPLETE`). Node identity uses provenance + reconstructed incoming
topology: the assignment maximizes satisfied topology constraints, so an agent
edge whose topology contradicts its own edge-existence provenance becomes an
edge miss instead of voiding the correspondence.

## Runs (new seeds, DeepSeek for agent + stakeholder)

`scripts/business_interview_real_llm_smoke.py --runs 4 --seed-base 7800`
(artifacts: `artifacts/business_interview_real_llm/run_0{0..3}_seed780{0..3}.json`

+ separate `.private.json` assertion ledgers).

| seed | term | node r | act | edge r | start | end r | glossary | evidence | leaks | errors |
| ------ | ------ | -------- | ----- | -------- | ------- | ------- | ---------- | ---------- | ------- | -------- |
| 7800 | episode_complete | 1.00 | 1.00 | 0.33 | True | 1.00 | False | True | 0 | 0 |
| 7801 | too_many_errors | 0.67 | 1.00 | 0.00 | True | 0.50 | False | True | 0 | 0 |
| 7802 | too_many_errors | 0.83 | 1.00 | 0.17 | True | 0.50 | False | True | 0 | 0 |
| 7803 | episode_complete | 0.67 | 1.00 | 0.17 | True | 0.50 | False | True | 0 | 0 |

## Checks against the objective

+ **Fake "record request" node disappears** — with sentence facts removed,
  the agent builds exactly the six Truth activities (receive / check / create
  / approve / send / month-end summary); no invented "record request"
  activity node in any run.
+ **Natural wording varies with questions** — e.g. *"After receiving the
  request, I check the customer information in the CRM."* realizes
  `cc.activity` / `cc.system` from semantic claims; wording differs per
  question and is not authored text.
+ **Selective disclosure** — early replies describe the normal flow; the
  approval branch and month-end tail are only realized when the question's
  scope calls for them (disclosure rules in the conversation instructions).
+ **Mention/concept reuse** — the agent created one actor concept per role
  and reused it across nodes; one `TerminologyAgreement` was recorded in the
  flagship run (the agent proposed a term and the stakeholder confirmed).
+ **Genuine confirmation** — `glossary_pass` False in every run: the
  evaluator rejected confirmations whose evidence does not correspond to a
  private assertion of the concept's own claims (mention-only speech or
  cross-concept spans are not confirmation). No bulk self-confirmation.
+ **Node/edge reconstruction** — node_recall 0.67–1.0 with activity
  correctness 1.0 in ALL runs (provenance + topology correspondence); edge
  recall is limited because the agent cites edge-existence evidence only for
  the branch edges (e3/e4) and its month-end edge topology was wrong in one
  run — correctly scored as edge misses.
+ **Provenance failures** — invalid assertion metadata (shortened claim ids,
  hallucinated quotes) is rejected at ingestion; runs with a double rejection
  terminate with `too_many_errors`; evidence hygiene passes in all runs.
+ **Start/end/cycles** — start_correct True in all runs (restored start
  semantics); end_recall 0.5–1.0; cycles validate cleanly (deterministic
  tests).
+ **Leakage and termination** — 0 leaks in all runs (claim ids only in the
  `.private.json` ledgers); 2/4 runs terminated with `episode_complete`
  immediately after `finish_interview` (distinct from max_steps).

## Remaining failures

+ Reward 0.0: full structural pass requires perfect node+edge+property
  grounding; the interviewer's actor/system/read evidence spans rarely equal
  the stakeholder's private assertion spans (exact-span coordination), edge
  evidence is sparse, and confirmations fail the genuine-evidence rule. The
  architecture intentionally forbids aliases/semantic matching to paper over
  this — the deterministic suite pins the mechanism precisely.
+ Two runs hit `too_many_errors` because the agent LLM called tools with
  fabricated observation ids; the environment correctly rejected them.

## Deterministic tests

`tests/test_domains/test_business_interview/` — 72 passed, covering every
objective proof: all incoming edges in `TruthNodeContext`; merge/rework/start
contexts (back edge included); the same activity at distinct positions
disambiguated by topology; no sentence-based facts (no `StakeholderFact`,
empty `known_info`); mention != terminology (mentions never establish
terminology; `record_terminology_agreement` separate); confirmation /
unknown / disputed require stakeholder evidence (bulk self-confirmation
rejected at tool and evaluator level); span containment grounds without
cross-credit (slot-scoped ambiguity fails); ConceptKind enforcement;
`finish_interview` terminates the episode (`EPISODE_COMPLETE`, distinct from
max_steps); private claim ids never leak.
