# business_interview: Simplified Truth-Reconstruction Scoring

## Goal summary

Re-scoped evaluation from "prove every graph element from exact utterance
spans" to **Truth reconstruction**: the final AgentConcepts + AgentGraph are
compared to the TruthConcepts + TruthGraph by content, without requiring
conversational provenance. Unsupported-but-correct reconstruction counts as
correct; fabricated/wrong elements are still penalised. StakeholderKnowledge
is retained as a simulator constraint, not the scored target.

## What changed

| File | Change |
| -------- | -------- |
| `graph.py` | `EvidenceRef` slimmed to `{observation_id, quote?, occurrence?}` (quote now optional diagnostic hint; exact quote matching no longer required) |
| `evaluation.py` | rewritten: content/Truth-reconstruction scorer (concept/node/edge/slot metrics vs Truth); provenance metrics kept as diagnostics that never gate `quality_pass` |
| `tools.py` | removed all provenance hard gates: `ground_concept`, `confirm_concept`, `mark_concept_unknown/disputed`, `record_terminology_agreement`, `record_dont_know`/`record_edge_condition_dont_know`, `record_absent`/`record_edge_condition_absent`, `add_node`/`update_node`/`add_edge`/`update_edge` marker handling, `finish_interview`; deleted 11 dead provenance-helper methods + unused imports |
| `policy.md` | rewritten to conversational/reconstruction policy (evidence optional & diagnostic; no exact-span burden; no grounding gate) |
| `README.md` | updated architecture/Evaluation/Concept-status sections |
| `scripts/business_interview_real_llm_smoke.py` | metric key renamed to the updated surrogate field |
| `tests/test_domains/test_business_interview/test_graph_business_interview.py` | replaced ~23 obsolete provenance/epistemic-restraint tests with the new reconstruction semantics |
| `tests/test_business_interview_roundtrips.py` | unchanged (Environment-owned Observation lifecycle, simulator-integrity, sidecar-plan tests all retained) |

## Deterministic suite

- `uv run pytest tests/test_business_interview.py::test_domain/test_graph_business_interview.py tests/test_business_interview_roundtrips.py` -> **114 passed** (was 97+15 with 23 obsolete failures; all re-scoped to reconstruction).
- ruff lint + format on all business_interview + smoke + tests: **All checks passed**.
- Core sweep (`pytest tests/` minus the unrelated flaky live-rule esc test) -> **372 passed, 1 deselected**.

## Real-LLM seed (quotation_workflow_1, deepseek-v4-flash-0731 via OpenRouter)

- termination: **episode_complete** (was: abort too_many_errors with 32 tool errors)
- exit errors: **0** (was 32 deterministic tool errors in the previous env-observation commit)
- `ground_concept` calls: **22/22 succeed** (no repeated ground_concept / quote-resolution / edge-binding retries)
- tool error categories: **none** (invalid_evidence_ref_count 0, ambiguous_evidence_ref_count 0, provenance_authenticity_pass True)
- Agent tool messages: **34** for **19** accepted Observations (~1.8 tool messages per Observation; includes graph build + 22 concepts + ground + 6 nodes + 6 edges)
- node recall/precision: **1.0 / 1.0**; edge recall/precision: **1.0 / 1.0**
- concept_correctness: **1.0**; start_correct **true**; end recall/precision **1.0/1.0**
- fabricated_node_count: **0**; fabricated_edge_count: **0**; private_id_leakage: **[]**
- read_correctness 0.667, write_correctness 0.833: the few misses are honest epistemic DONT_KNOW markers (stakeholder genuinely did not know some source artifacts where Truth has values) — the model did not invent them.

## Removed provenance requirements

- `ground_concept` no longer requires (or rejects on) ambiguous/unrelated/kind-wrong semantic annotations; evidence optional.
- ABSENT / DONT_KNOW markers no longer require exact-mapped private stakeholder-slot binding.
- Edge/property existence no longer requires a unique private edge/node binding via sidecar annotations.
- `finish_interview` no longer refuses hypothesized concepts.
- Evaluator no longer gates `quality_pass` on evidence coverage/authenticity (reported as diagnostics only).

## Simplified EvidenceRef / concept lifecycle

- `EvidenceRef = {observation_id, quote?, occurrence?}`.
- `AgentConcept.validation_status` is a belief record, not a provenance proof.
- Concept identity evaluated by content against Truth, never by private StakeholderKnowledge alignment.

## Retained simulator-integrity checks

- Stakeholder can't reveal outside its StakeholderKnowledge (plan/sidecar validated against the knowledge catalog; retained).
- Environment-owned Observation lifecycle (one accepted utterance -> one Observation -> one Agent-visible id) retained.
- Private StakeholderKnowledge / opaque semantic ids remain hidden from the Agent.
- Provenance sidecar / Semantic Response Plan kept as diagnostic + simulator-integrity metadata.

## Remaining unnecessary complexity

None known to block the benchmark; the provenance ledger is still collected (useful for debugging/leakage detection) but no longer gates. `grounded_semantic_ids`/`resolve_grounding_refs` re-exports kept in the evaluation module for diagnostic callers.
