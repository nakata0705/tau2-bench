# Business-interview evaluation diagnostics

This is an evaluator-private, deterministic re-evaluation of stored quotation artifacts. No LLM was called. The Truth labels, private StakeholderKnowledge mappings, and sidecar annotations in this report must not be copied into Agent or Stakeholder runtime surfaces.

## Method and score contract

Each seed was loaded from its saved `final_graph`, `truth_graph`, accepted `observations`, and evaluator-private `.private.json` knowledge/annotation sidecar. The smoke artifacts use a custom JSON marker encoding, so offline restoration explicitly decodes UNSET, ABSENT, DONT_KNOW, and ConceptRef values instead of passing the graph through the undiscriminated Pydantic union. Stored scalar evaluator metrics are compared with a floating-point representation tolerance before any attribution; a mismatch fails closed and no report is generated. The current evaluator was run twice per artifact during verification; a direct comparison with the pre-change `business-interview` HEAD evaluator matched every scalar score/pass field on full and partial deterministic graphs. Diagnostics are metadata only and do not alter score fields, thresholds, or matcher selection; the table below is the current-HEAD re-evaluation after stored-metric parity, not a copy of historical metrics. All diagnostic output remains evaluator-private/offline.

## Diagnostic schema and reason codes

`EvaluationResult.diagnostics` contains `node_diagnostics` (one Truth node with `slots` for `activity`, `actor`, `system`, `reads`, `writes`, and `necessity_rationale`), `edge_diagnostics` (endpoint structural match plus `condition`), and `concepts` (candidate pair scores, exact label path, selected mapping, and unmatched concepts). Each slot has Truth/Agent state, concept ids/labels, matched flag, score contribution, and reason codes. The deterministic codes used here include: `correct_value`, `truth_value_agent_unset`, `truth_value_agent_dont_know`, `truth_value_agent_absent`, `truth_absent_agent_absent`, `truth_absent_agent_unset`, `truth_absent_agent_dont_know`, `truth_absent_agent_value`, `truth_value_agent_unasserted`, `wrong_concept`, `missing_list_item`, `extra_list_item`, `missing_and_extra_list_items`, `unmatched_node`, and `unmatched_edge`.

## Per-seed metrics

| seed | nodes (R/P) | edges (R/P) | concepts (R/P) | activity | actor | system | reads | writes | rationale | condition | knowledge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9002 | 1.000/1.000 | 0.833/0.714 | 1.000/0.955 | 1.000 | 1.000 | 0.667 | 0.250 | 0.167 | 0.167 | 0.500 | 0.717 |
| 9003 | 1.000/1.000 | 1.000/1.000 | 0.762/0.941 | 0.833 | 1.000 | 0.167 | 0.167 | 0.333 | 0.167 | 0.500 | 0.717 |
| 9004 | 1.000/1.000 | 1.000/1.000 | 0.952/1.000 | 1.000 | 1.000 | 0.500 | 0.333 | 0.333 | 0.167 | 1.000 | 0.717 |

## Aggregate failed-slot attribution

Counts are failed scored slots, not token-level or LLM judgments. `unknown` includes hidden/DONT_KNOW stakeholder facts, structural unmatches, and any ambiguous evidence.

| slot | failures | disclosure | elicitation | recording | evaluator | unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| activity | 1 | 1 | 0 | 0 | 0 | 0 |
| actor | 0 | 0 | 0 | 0 | 0 | 0 |
| system | 10 | 3 | 0 | 0 | 0 | 7 |
| reads | 14 | 2 | 0 | 0 | 0 | 12 |
| writes | 13 | 1 | 0 | 3 | 0 | 9 |
| rationale | 15 | 0 | 0 | 0 | 0 | 15 |
| condition | 6 | 0 | 0 | 0 | 0 | 6 |

## Supported next intervention

Among classifiable failures, `stakeholder_disclosure` is largest (7). The highest-value next diagnostic intervention is to inspect accepted public disclosure/sidecar capture before changing Agent policy; this task does not implement it.

## Remaining attribution limitations

- A public disclosure is credited only when the evaluator-private sidecar annotation resolves to an authentic accepted Observation; an unannotated paraphrase is not treated as deterministic proof.
- `agent_elicitation` versus `stakeholder_disclosure` uses a lexical question/context check over stored Agent messages, not an LLM; ambiguous questions should be treated as unknown.
- `evaluator_matching` requires the Agent evidence Observation to overlap the accepted semantic evidence and an unselected below-threshold matcher candidate. No seed had enough evidence for that category.
- DONT_KNOW/hidden knowledge and unmatched structure remain `insufficient_evidence_to_classify`; they are not Agent failures.
- This task does not change existing node/edge or concept matcher tie-breaking; duplicate identical structural signatures remain a future evaluator investigation.

## Seed 9002

- source: `artifacts/business_interview_real_llm/run_00_seed9002.json`
- private sidecar: `artifacts/business_interview_real_llm/run_00_seed9002.private.json`
- stored-metric parity: `matched` (41 fields)
- quality/reconstruction pass: `False` / `False`

### Failed slots

- `ap:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `ap:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `ap:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `cq:reads` — reason `missing_list_item` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:missing_list_item; knowledge:known; missing_truth_concepts:tc_customer; clear_agent_question:True)
- `cq:writes` — reason `missing_and_extra_list_items` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:missing_and_extra_list_items; knowledge:known; missing_truth_concepts:tc_quote; clear_agent_question:True)
- `cq:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `me:reads` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know; accepted_observations:obs_56)
- `me:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `r:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `r:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `r:writes` — reason `truth_value_agent_unset` — category `agent_recording` — accepted public semantic evidence exists but final Agent state does not score it (score_reason:truth_value_agent_unset; knowledge:known; missing_truth_concepts:tc_request; accepted_observations:obs_3)
- `r:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `sq:reads` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `sq:writes` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `sq:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `edge:e1:condition` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `edge:e2:condition` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `edge:e6:condition` — reason `unmatched_edge` — category `insufficient_evidence_to_classify` — structural target was not aligned (score_reason:unmatched_edge; knowledge:known; missing_truth_concepts:tc_cond_month_end)

### Notable concept matching

- unmatched Truth concepts: `none`
- unmatched Agent concepts: `['data_quotation_document']`
- selected mappings: `21`

## Seed 9003

- source: `artifacts/business_interview_real_llm/run_00_seed9003.json`
- private sidecar: `artifacts/business_interview_real_llm/run_00_seed9003.private.json`
- stored-metric parity: `matched` (41 fields)
- quality/reconstruction pass: `False` / `False`

### Failed slots

- `ap:activity` — reason `wrong_concept` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:wrong_concept; knowledge:known; clear_agent_question:True)
- `ap:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `ap:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `ap:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:system` — reason `truth_value_agent_unset` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:truth_value_agent_unset; knowledge:known; clear_agent_question:True)
- `cc:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `cq:reads` — reason `truth_value_agent_unset` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:truth_value_agent_unset; knowledge:known; missing_truth_concepts:tc_customer,tc_pricing; clear_agent_question:True)
- `cq:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `me:system` — reason `truth_value_agent_unset` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:truth_value_agent_unset; knowledge:known; clear_agent_question:True)
- `me:reads` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `me:necessity_rationale` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `r:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `r:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `r:writes` — reason `truth_value_agent_unset` — category `agent_recording` — accepted public semantic evidence exists but final Agent state does not score it (score_reason:truth_value_agent_unset; knowledge:known; missing_truth_concepts:tc_request; accepted_observations:obs_3)
- `r:necessity_rationale` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `sq:system` — reason `truth_value_agent_unset` — category `insufficient_evidence_to_classify` — stored questions were contextually ambiguous (score_reason:truth_value_agent_unset; knowledge:known; clear_agent_question:None)
- `sq:reads` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `sq:writes` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `sq:necessity_rationale` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `edge:e1:condition` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `edge:e2:condition` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `edge:e5:condition` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)

### Notable concept matching

- unmatched Truth concepts: `['tc_activity_approve_quotation', 'tc_pricing', 'tc_system_crm', 'tc_system_email', 'tc_system_excel']`
- unmatched Agent concepts: `['act_manager_approve']`
- selected mappings: `16`

## Seed 9004

- source: `artifacts/business_interview_real_llm/run_00_seed9004.json`
- private sidecar: `artifacts/business_interview_real_llm/run_00_seed9004.private.json`
- stored-metric parity: `matched` (41 fields)
- quality/reconstruction pass: `False` / `False`

### Failed slots

- `ap:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `ap:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `ap:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:writes` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `cc:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `cq:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `me:system` — reason `truth_value_agent_unset` — category `stakeholder_disclosure` — knowledge was available, but no accepted public annotation disclosed the needed value (score_reason:truth_value_agent_unset; knowledge:known; clear_agent_question:True)
- `me:reads` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `me:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `r:system` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `r:reads` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `r:writes` — reason `truth_value_agent_unset` — category `agent_recording` — accepted public semantic evidence exists but final Agent state does not score it (score_reason:truth_value_agent_unset; knowledge:known; missing_truth_concepts:tc_request; accepted_observations:obs_3)
- `r:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)
- `sq:reads` — reason `truth_absent_agent_unset` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_unset; knowledge:dont_know)
- `sq:writes` — reason `extra_list_item` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:extra_list_item; knowledge:dont_know)
- `sq:necessity_rationale` — reason `truth_absent_agent_dont_know` — category `insufficient_evidence_to_classify` — stored stakeholder knowledge cannot establish attainability (score_reason:truth_absent_agent_dont_know; knowledge:dont_know)

### Notable concept matching

- unmatched Truth concepts: `['tc_system_excel']`
- unmatched Agent concepts: `none`
- selected mappings: `20`
