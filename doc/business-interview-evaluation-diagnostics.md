# Business-interview evaluation diagnostics

This is an evaluator-private, deterministic re-evaluation of stored quotation artifacts. No LLM was called. The Truth labels, private StakeholderKnowledge mappings, and sidecar annotations in this report must not be copied into Agent or Stakeholder runtime surfaces.

## Method and score contract

Each seed was loaded from its saved `final_graph`, `truth_graph`, accepted `observations`, and evaluator-private `.private.json` knowledge/annotation sidecar. The smoke artifacts use a custom JSON marker encoding, so offline restoration explicitly decodes UNSET, ABSENT, DONT_KNOW, and ConceptRef values instead of passing the graph through the undiscriminated Pydantic union. Stored scalar evaluator metrics are compared with a floating-point representation tolerance before any attribution; a mismatch fails closed and no report is generated. The current evaluator was run twice per artifact during verification; a direct comparison with the pre-change `business-interview` HEAD evaluator matched every scalar score/pass field on full and partial deterministic graphs. Diagnostics are metadata only and do not alter score fields, thresholds, or matcher selection; the table below is the current-HEAD re-evaluation after stored-metric parity, not a copy of historical metrics. All diagnostic output remains evaluator-private/offline.

## Diagnostic schema and reason codes

`EvaluationResult.diagnostics` contains `node_diagnostics` (one Truth node with `slots` for `activity`, `actor`, `system`, `reads`, `writes`, and `necessity_rationale`), `edge_diagnostics` (endpoint structural match plus `condition`), and `concepts` (candidate pair scores, exact label path, selected mapping, and unmatched concepts). The separate `usage_alignment` section contains usage candidate sets, per-kind assignments, ambiguity classes, and current-vs-usage comparisons. Each slot has Truth/Agent state, concept ids/labels, matched flag, score contribution, and reason codes. The deterministic codes used here include: `correct_value`, `truth_value_agent_unset`, `truth_value_agent_dont_know`, `truth_value_agent_absent`, `truth_absent_agent_absent`, `truth_absent_agent_unset`, `truth_absent_agent_dont_know`, `truth_absent_agent_value`, `truth_value_agent_unasserted`, `wrong_concept`, `missing_list_item`, `extra_list_item`, `missing_and_extra_list_items`, `unmatched_node`, and `unmatched_edge`.

## Per-seed metrics

| seed | nodes (R/P) | edges (R/P) | concepts (R/P) | activity | actor | system | reads | writes | rationale | condition | knowledge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9002 | 1.000/1.000 | 0.833/0.714 | 1.000/0.955 | 1.000 | 1.000 | 0.667 | 0.250 | 0.167 | 0.167 | 0.500 | 0.717 |
| 9003 | 1.000/1.000 | 1.000/1.000 | 0.762/0.941 | 0.833 | 1.000 | 0.167 | 0.167 | 0.333 | 0.167 | 0.500 | 0.717 |
| 9004 | 1.000/1.000 | 1.000/1.000 | 0.952/1.000 | 1.000 | 1.000 | 0.500 | 0.333 | 0.333 | 0.167 | 1.000 | 0.717 |

## Usage-based concept alignment (diagnostic only)

The usage experiment is explicitly named `usage_alignment_conditioned_on_current_node_mapping`. It translates Agent node/edge addresses through the existing production node/edge correspondence, then compares deterministic sets of `node:<id>:<property>` and `edge:<id>:condition` addresses. Empty mapped signatures are insufficient evidence, not exact matches. Concept kind is a hard constraint. Per-pair precision, recall, F1, Jaccard, exact equality, set differences, and strict broader/narrower relations are retained in the JSON traces.

The one-to-one assignment is per kind and uses only usage F1, with a deterministic priority bonus for non-empty exact usage equality. Labels, descriptions, canonical terms, translations, embeddings, and LLM judgment are not inputs to that assignment. Sorted opaque ids are used only for reproducible serialization; identical usage signatures and tied usage-candidate rows are reported as ambiguity classes rather than resolved by labels. The report-only `substantially stronger` flag means exact usage or positive usage F1 where the current mapping has zero usage support; it is not a production threshold. This is an exploratory comparison and introduces no score change.

| seed | Truth concepts | Agent concepts | exact usage | partial usage | ambiguous concepts | mapping disagreements | substantially stronger | labels differ + usage agrees | labels agree + usage differs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9002 | 21 | 22 | 17 | 2 | 0 | 4 | 1 | 7 | 2 |
| 9003 | 21 | 17 | 14 | 2 | 0 | 2 | 1 | 4 | 2 |
| 9004 | 21 | 20 | 18 | 1 | 0 | 1 | 0 | 4 | 1 |

### Compact per-kind summary

| seed | kind | Truth | Agent | exact | partial | ambiguous | disagreements | stronger | insufficient |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9002 | activity | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| 9002 | actor | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 9002 | condition | 3 | 3 | 2 | 0 | 0 | 1 | 0 | 1 |
| 9002 | data | 5 | 6 | 2 | 2 | 0 | 3 | 1 | 2 |
| 9002 | rationale | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 9002 | system | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 9003 | activity | 6 | 6 | 6 | 0 | 0 | 1 | 1 | 0 |
| 9003 | actor | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 9003 | condition | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 9003 | data | 5 | 4 | 1 | 2 | 0 | 1 | 0 | 1 |
| 9003 | rationale | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 9003 | system | 4 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 9004 | activity | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| 9004 | actor | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| 9004 | condition | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| 9004 | data | 5 | 5 | 3 | 1 | 0 | 1 | 0 | 1 |
| 9004 | rationale | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 9004 | system | 4 | 3 | 3 | 0 | 0 | 0 | 0 | 0 |

## Usage experiment conclusion

Across these stored seeds, the usage scaffold produced `49` exact and `5` partial assigned matches, with `0` structurally ambiguous concepts and `7` comparison disagreements (`2` where usage had exact or positive-vs-zero support substantially stronger than the current mapping).

**Does usage appear strong enough to replace lexical matching? No, not as a production replacement from these artifacts.** Exact usage is a strong and useful conditional signal, including cases where labels are not exact, but partial/missing usage, unmapped Agent addresses, and ambiguous equivalence classes prevent usage from resolving every concept. A disagreement is evidence to inspect, not proof that the usage assignment is correct.

Usage is insufficient when the current node/edge scaffold leaves a concept with no mapped addresses, when a concept is absent from some of its Truth locations, or when multiple concepts share the same translated address set. The JSON candidate records expose the exact Truth-only and Agent-only usages and strict broader/narrower relations for follow-up.

A fully label-independent joint matcher would first need a validated node/edge correspondence derived from label-independent topology, start/end roles, degree, and slot/co-occurrence structure; then a joint or confidence-aware iterative optimization over node, edge, and concept assignments; explicit handling for missing/extra structure and non-identifiability; and adversarial multilingual/duplicate-usage fixtures. This experiment intentionally does not attempt that large optimization.

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

### Usage alignment comparison

- referenced concepts: Truth `21`, Agent `22`
- exact/partial assigned usage matches: `17` / `2`
- structurally ambiguous concepts: `0`
- disagreements with current mapping: `4` (substantially stronger usage evidence: `1`)

- structural ambiguity classes: none


#### Concrete disagreements with the current lexical/content matcher

| Agent concept | label | current Truth | usage Truth | classification | lexical score | usage F1 | exact usage |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `cond_end_of_month` | `at the end of the month` | `tc_cond_month_end` | `—` | `current_mapping_has_no_usage_support` | 1.000 | 0.000 | False |
| `data_quotation` | `quotation` | `tc_quote` | `—` | `current_mapping_has_no_usage_support` | 1.000 | 0.000 | False |
| `data_quotation_document` | `quotation document` | `—` | `tc_quote` | `usage_prefers_different_mapping` | 0.000 | 0.667 | False |
| `data_quotation_request` | `customer's quotation request` | `tc_request` | `—` | `current_mapping_has_no_usage_support` | 1.000 | 0.000 | False |

#### Examples where labels differ but usage agrees

- Agent `act_check_customer_info` (check the customer information in the CRM) -> Truth `tc_activity_check_customer` with exact usage; the current lexical path was not an exact label match.
- Agent `act_create_quotation_doc` (create the quotation document in the quoting system using the pricing information) -> Truth `tc_activity_create_quotation` with exact usage; the current lexical path was not an exact label match.
- Agent `act_receive_request` (receive the customer's quotation request) -> Truth `tc_activity_receive_request` with exact usage; the current lexical path was not an exact label match.
- Agent `act_send_month_end_summary` (send the month-end summary of quotation information to Accounting) -> Truth `tc_activity_send_month_end_summary` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_amount_at_or_below_1m` (quotation amount is at or below 1,000,000 yen) -> Truth `tc_cond_at_or_below_1m` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_amount_over_1m` (quotation amount is over 1,000,000 yen) -> Truth `tc_cond_over_1m` with exact usage; the current lexical path was not an exact label match.
- Agent `data_month_end_summary` (month-end summary of quotation information) -> Truth `tc_excel_summary` with exact usage; the current lexical path was not an exact label match.

#### Examples where labels agree but usage does not

- Agent `data_customer_information` -> Truth `tc_customer`: exact label match, but usage exact=`False`, usage F1=`0.667`.
- Agent `data_quotation` -> Truth `tc_quote`: exact label match, but usage exact=`False`, usage F1=`0.000`.

#### Usage-only differences and broader/narrower evidence

- Agent `data_customer_information` -> Truth `tc_customer`: only Truth `['node:cq:reads']`, only Agent `none`; relation `truth_broader_agent_narrower`.
- Agent `data_quotation_document` -> Truth `tc_quote`: only Truth `none`, only Agent `['node:ap:reads']`; relation `agent_broader_truth_narrower`.

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

### Usage alignment comparison

- referenced concepts: Truth `21`, Agent `17`
- exact/partial assigned usage matches: `14` / `2`
- structurally ambiguous concepts: `0`
- disagreements with current mapping: `2` (substantially stronger usage evidence: `1`)

- structural ambiguity classes: none


#### Concrete disagreements with the current lexical/content matcher

| Agent concept | label | current Truth | usage Truth | classification | lexical score | usage F1 | exact usage |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `act_manager_approve` | `manager approves the quotation` | `—` | `tc_activity_approve_quotation` | `usage_exact_but_current_different` | 0.000 | 1.000 | True |
| `data_quotation_request` | `customer's quotation request` | `tc_request` | `—` | `current_mapping_has_no_usage_support` | 1.000 | 0.000 | False |

#### Examples where labels differ but usage agrees

- Agent `act_receive_request` (receive the customer's quotation request) -> Truth `tc_activity_receive_request` with exact usage; the current lexical path was not an exact label match.
- Agent `act_send_summary` (send the summary of the quotation information to Accounting) -> Truth `tc_activity_send_month_end_summary` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_at_or_below_1m` (amount at or below 1,000,000 yen) -> Truth `tc_cond_at_or_below_1m` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_over_1m` (amount over 1,000,000 yen) -> Truth `tc_cond_over_1m` with exact usage; the current lexical path was not an exact label match.

#### Examples where labels agree but usage does not

- Agent `data_customer_information` -> Truth `tc_customer`: exact label match, but usage exact=`False`, usage F1=`0.667`.
- Agent `data_quotation` -> Truth `tc_quote`: exact label match, but usage exact=`False`, usage F1=`0.500`.

#### Usage-only differences and broader/narrower evidence

- Agent `data_customer_information` -> Truth `tc_customer`: only Truth `['node:cq:reads']`, only Agent `none`; relation `truth_broader_agent_narrower`.
- Agent `data_quotation` -> Truth `tc_quote`: only Truth `none`, only Agent `['node:ap:reads', 'node:sq:writes']`; relation `agent_broader_truth_narrower`.

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

### Usage alignment comparison

- referenced concepts: Truth `21`, Agent `20`
- exact/partial assigned usage matches: `18` / `1`
- structurally ambiguous concepts: `0`
- disagreements with current mapping: `1` (substantially stronger usage evidence: `0`)

- structural ambiguity classes: none


#### Concrete disagreements with the current lexical/content matcher

| Agent concept | label | current Truth | usage Truth | classification | lexical score | usage F1 | exact usage |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| `data_quotation_request` | `customer's quotation request` | `tc_request` | `—` | `current_mapping_has_no_usage_support` | 1.000 | 0.000 | False |

#### Examples where labels differ but usage agrees

- Agent `act_receive_request` (receive the customer's quotation request) -> Truth `tc_activity_receive_request` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_high_value` (amount over 1,000,000 yen) -> Truth `tc_cond_over_1m` with exact usage; the current lexical path was not an exact label match.
- Agent `cond_low_value` (amount at or below 1,000,000 yen) -> Truth `tc_cond_at_or_below_1m` with exact usage; the current lexical path was not an exact label match.
- Agent `data_month_end_summary` (month-end summary) -> Truth `tc_excel_summary` with exact usage; the current lexical path was not an exact label match.

#### Examples where labels agree but usage does not

- Agent `data_quotation` -> Truth `tc_quote`: exact label match, but usage exact=`False`, usage F1=`0.500`.

#### Usage-only differences and broader/narrower evidence

- Agent `data_quotation` -> Truth `tc_quote`: only Truth `none`, only Agent `['node:ap:reads', 'node:sq:writes']`; relation `agent_broader_truth_narrower`.
