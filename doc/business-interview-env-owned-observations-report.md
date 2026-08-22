# business_interview — Env-owned Observations + Semantic Response Plan

**Date:** 2026-08-22
**Branch:** `business-interview`
**Commit:** `refactor: make stakeholder observations environment owned`

## What changed

### 1. Environment owns Observation creation

- An ACCEPTED stakeholder utterance automatically becomes one immutable
  Observation BEFORE the Agent sees it (`BusinessInterviewEnvironment.on_message`):
  the private sidecar is validated first (unknown ids / bad quotes / mode
  contradictions raise and create nothing), then the Observation is created
  (`obs_<turn>`, raw public text, source `stakeholder`).
- The Observation id is delivered inline with the public text:
  `[Observation obs_8] We do it to manage credit risk.` The raw text lives in
  the ledger and the immutable Observation; quotes in the sidecar are anchored
  to the raw text; the embed is deterministic so `set_state` replay recovers
  the raw text exactly and never double-embeds.
- Guarantee enforced: one accepted Stakeholder utterance <-> one Observation
  <-> one Agent-visible Observation id. A failed/retried Stakeholder
  generation consumes no id (simulator retries happen before any message
  reaches the environment).

### 2. Removed obsolete observation tools

- `observe_latest_stakeholder_message` and `observe_message` are deleted
  (no compatibility shim). The Agent can never create or mutate Observations.
- `list_stakeholder_messages` now lists the accepted Observations (obs ids).
- Agent policy updated: every accepted response already carries its
  Observation id; use it directly in EvidenceRefs; never guess or reuse an
  older Observation id for a new utterance.

### 3. Batching semantics

- Policy now states the batching rule precisely: every tool in a batch must
  have all required arguments known before the batch starts; independent
  evidence operations batch freely on the already-delivered Observation id;
  a same-batch FUTURE dependency (an id that does not exist when the batch
  starts) is rejected — never guessed or fabricated. No example implies one
  parallel tool depends on another tool's result from the same batch.

### 4–6. Semantic Response Plan (WHAT before HOW)

The stakeholder simulator now answers in two deterministic phases:

1. **Plan** — from the question + its own world model, the stakeholder LLM
   builds a private Semantic Response Plan: intended semantic addresses +
   modes (`node:skn_002:rationale` / `value`, `node:skn_004:system` /
   `dont_know`, ...), plan text is `{"plan": [...]}`.
2. **Validate** — every planned item is checked through the canonical
   resolver (`ConceptRef -> value`, `None -> absent`, `DONT_KNOW ->
   dont_know`, node/edge -> exists, StakeholderKnowledgeConcept -> mention).
   A plan that contradicts the StakeholderKnowledge is rejected before any
   wording (known value cannot become dont_know; DONT_KNOW cannot become
   value; a plan can never carry Truth-only information because only
   knowledge-local opaque ids resolve).
3. **Realize** — the validated plan is expressed in natural language; the
   private sidecar must contain, for EVERY planned item, an exact
   public-text span anchored to the SAME (semantic_id, mode), and nothing
   outside the plan. Missing assertions, unplanned assertions,
   value/absent/dont_know contradictions, quotes not in the public text are
   rejected (bounded retry). Ordinary terminology references stay `mention`;
   terminology agreement is never inferred.

### 7. Responsibility split (unchanged boundaries preserved)

- Environment: validates Stakeholder output, creates the accepted Observation,
  owns Observation ids + the private sidecar.
- Stakeholder simulator: chooses the Semantic Response Plan from its own
  knowledge and realizes it.
- Agent: receives Observation id + public text, interprets EvidenceRefs,
  builds AgentConcept / AgentGraph. Never creates or mutates Observations.
- Evaluator: resolves EvidenceRefs against the immutable
  Observation/private metadata.

## Deterministic tests

`tests/test_domains/test_business_interview/` (97) +
`tests/test_business_interview_roundtrips.py` (15). New coverage:

- accepted Stakeholder response auto-creates exactly one Observation
  (correct id/text/source/order) and the Agent receives
  `[Observation obs_N] <public text>`;
- Agent receives correct observation_id + public text (delivery equals the
  recorded Observation);
- failed sidecar retry creates no Observation and consumes no id (repeated
  failures leave zero Observations);
- accepted Stakeholder messages and Observations remain 1:1;
- obsolete observation tools are removed (attribute + tool-set assertion);
- EvidenceRef can immediately use the delivered Observation id;
- independent evidence tools batch in one Agent turn using that id;
- same-batch future dependency is rejected (ghost obs id raises, creates
  nothing);
- Semantic Response Plan cannot contradict StakeholderKnowledge;
- every planned semantic assertion must appear in the sidecar
  (missing / wrong-mode / unplanned annotations all rejected);
- known value cannot become dont_know; DONT_KNOW cannot become value;
  unknown/truth-only ids cannot be planned;
- plan-then-realize pipeline provable end-to-end with a stubbed LLM;
- existing provenance / evaluator / replay / loop-guard / round-trip tests
  still pass (233 tests in the affected suites).

## Real run (ONE seed, quotation_workflow_1)

Command: `make test-business-interview-smoke` (forces 1 run) with
DeepSeek V4 Flash 0731 for BOTH sides (same as all prior runs).
Artifact: `artifacts/business_interview_real_llm/run_00_seed8100.json` +
`.private.json`.

### Mechanic verification (all green)

- 14 accepted Stakeholder messages <-> 14 Observations (unique ids, correct
  order/turn); every delivered `[Observation obs_N]` text matches the recorded
  Observation text (0 mismatches);
- `invalid_observation_reference_count: 0`, `authentic_observation_count: 14`,
  `provenance_authenticity_pass: True`;
- 0 failed provider calls; private-id leakage scan: `[]` (no skn_/ske_/skc_
  id ever reached the Agent);
- Agent LLM calls triggered by stakeholder messages: **14 for 14 accepted
  Observations (1:1)** — no observation round trips. Stakeholder calls: 27
  (13 responses used plan+realize = 2 calls; 1 response used 3 calls:
  plan -> realize rejected by the sidecar-completeness check -> successful
  retry — exactly one completeness retry, bounded as designed).

### Evaluator metrics (ABORTED run — reported fully, not summarized as success)

| metric | value |
| --- | --- |
| termination_reason | `too_many_errors` (32 tool errors > max_errors 30) |
| reward | 0.0 (ENV_ASSERTION only) |
| protocol_completed / interview_complete | False / False |
| node_recall / node_precision | 0.167 / 0.167 |
| edge_recall / edge_precision | 0.0 / 0.0 |
| start_correct / end_recall | False / 0.0 |
| activity / actor / system / read / write / rationale / condition correctness | 0.0 / 0.0 / 1.0 / 1.0 / 0.0 / 1.0 / 0.0 |
| concept_correctness / recall / precision | 1.0 / 1.0 / 1.0 |
| glossary_pass / glossary_complete | True / True |
| unsupported_ref_count | 2 |
| fabricated_node_count / fabricated_edge_count | 5 / 6 |
| node/ref/edge evidence coverage | 1.0 / 0.76 / 1.0 |
| invalid / ambiguous evidence ref counts | 0 / 12 |
| marker_evidence_errors | 0 |
| orphan_observation_count | 8 (8 of 14 Observations never cited) |
| knowledge_coverage | 0.717 |

The final AgentGraph is structurally VALID (6 nodes, 6 edges, 22 concepts,
`validation_errors: []`) and semantically close to the Truth — the evaluator
binding failed because much of the property evidence was ambiguous /
unsupported (`unsupported_ref_count: 2`, `ambiguous_evidence_ref_count: 12`),
so the node/edge correspondence could not be established
(`node_recall: 0.167`, `edge_recall: 0`).

### Remaining failures (Agent-side skill, consistent with ALL prior runs on this model)

1. Repeated quote-resolution failures: the Agent cited quotes for the request
   statement against the WRONG observation id (`obs_1` = greeting) 3 times —
   the tools rejected every one with a clear error (no invalid reference ever
   reached the evaluator).
2. Repeated grounding failures: the Agent re-submitted the SAME ambiguous
   evidence (e.g. `at the end of the month`) for `ground_concept` 10+ times
   (obs_48 / obs_113 / obs_77) despite the policy forbidding re-submitting
   rejected evidence — this alone burned most of the error budget.
3. Ordering cascade: edges/nodes created before the concepts/nodes they
   depend on (`concept not found`, `node not found`,
   `not uniquely bindable` for `record_edge_condition_absent`).
4. No `finish_interview` was ever reached (protocol_completed False).

These are the same failure classes seen on this model before this refactor
(previous runs: 153 / 50 tool errors, obs 1–5, node_recall 0.0); every
rejection was produced by the deterministic tools, none by the new
mechanism. The refactor itself removed one Agent LLM/tool round trip per
Stakeholder response while keeping the delivered Observation ids exact.
