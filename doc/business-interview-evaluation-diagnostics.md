# Business-interview evaluation diagnostics

This is an evaluator-private, deterministic re-evaluation of stored quotation artifacts. No LLM was called. The Truth labels, private StakeholderKnowledge mappings, and sidecar annotations in this report must not be copied into Agent or Stakeholder runtime surfaces.

## Method and score contract

Each seed was loaded from its saved `final_graph`, `truth_graph`, accepted `observations`, and evaluator-private `.private.json` knowledge/annotation sidecar. The smoke artifacts use a custom JSON marker encoding, so offline restoration explicitly decodes UNSET, ABSENT, DONT_KNOW, and ConceptRef values instead of passing the graph through the undiscriminated Pydantic union. Stored scalar evaluator metrics are compared with a floating-point representation tolerance before any attribution; a mismatch fails closed and no report is generated. The current evaluator was run twice per artifact during verification; a direct comparison with the pre-change `business-interview` HEAD evaluator matched every scalar score/pass field on full and partial deterministic graphs. Diagnostics are metadata only and do not alter score fields, thresholds, or matcher selection; the table below is the current-HEAD re-evaluation after stored-metric parity, not a copy of historical metrics. All diagnostic output remains evaluator-private/offline.

## Diagnostic schema and reason codes

`EvaluationResult.diagnostics` contains `node_diagnostics` (one Truth node with `slots` for `activity`, `actor`, `system`, `reads`, `writes`, and `necessity_rationale`), `edge_diagnostics` (endpoint structural match plus `condition`), and `concepts` (candidate pair scores, exact label path, selected mapping, and unmatched concepts). The separate `usage_alignment` section contains usage candidate sets, per-kind assignments, ambiguity classes, and current-vs-usage comparisons. The `joint_structural_alignment` section is a separate label-independent joint Node/Concept search with typed incidence, process-edge, start, and end objective components; it is not a production matcher. Each slot has Truth/Agent state, concept ids/labels, matched flag, score contribution, and reason codes. The deterministic codes used here include: `correct_value`, `truth_value_agent_unset`, `truth_value_agent_dont_know`, `truth_value_agent_absent`, `truth_absent_agent_absent`, `truth_absent_agent_unset`, `truth_absent_agent_dont_know`, `truth_absent_agent_value`, `truth_value_agent_unasserted`, `wrong_concept`, `missing_list_item`, `extra_list_item`, `missing_and_extra_list_items`, `unmatched_node`, and `unmatched_edge`.

## Per-seed metrics

| seed | nodes (R/P) | edges (R/P) | concepts (R/P) | activity | actor | system | reads | writes | rationale | condition | knowledge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9002 | 1.000/1.000 | 0.833/0.714 | 1.000/0.955 | 1.000 | 1.000 | 0.667 | 0.250 | 0.167 | 0.167 | 0.500 | 0.717 |
| 9003 | 1.000/1.000 | 1.000/1.000 | 0.762/0.941 | 0.833 | 1.000 | 0.167 | 0.167 | 0.333 | 0.167 | 0.500 | 0.717 |
| 9004 | 1.000/1.000 | 1.000/1.000 | 0.952/1.000 | 1.000 | 1.000 | 0.500 | 0.333 | 0.333 | 0.167 | 1.000 | 0.717 |

## Usage-based concept alignment (diagnostic only)

The usage experiment is explicitly named `usage_alignment_conditioned_on_current_node_mapping`. It translates Agent node/edge addresses through the existing production node/edge correspondence, then compares deterministic sets of `node:<id>:<property>` and `edge:<id>:condition` addresses. Empty mapped signatures are insufficient evidence, not exact matches. Concept kind is a hard constraint. Per-pair precision, recall, F1, Jaccard, exact equality, set differences, and strict broader/narrower relations are retained in the JSON traces.

Interpretation: given the current node/edge correspondence, this asks whether graph usage is a better concept-identity signal than labels. Because that correspondence can itself depend partly on concept alignment, this does not prove a fully label-independent evaluator; the scaffold is a stated circularity limitation, not a production matcher change.

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

Usage alone is not a fully label-independent matcher because its address translation is conditioned on the production node/edge scaffold. The separate `joint_structural_alignment` section below tests a bounded joint search instead; it remains diagnostic-only and does not replace either production or usage mappings.

## Joint structural alignment (diagnostic only)

The joint experiment searches Node and asserted Concept mappings together. It uses only typed incidence, directed process topology, start/end roles, and one-to-one constraints; Concept kind is hard. A representative mapping is serialized for audit only; opaque IDs are used only for deterministic ordering/serialization. Equal optima are retained as ambiguity classes, and a bounded search never claims uniqueness. Same-kind pairs with no positive typed support remain unmatched rather than being assigned arbitrarily. Each component is F1-style `2*matched/(Agent+Truth)` (empty/empty is exact), and the total is the unweighted sum of the auditable components. `runtime estimate` is a deterministic work-unit estimate rather than wall-clock timing, allowing this report to be regenerated byte-for-byte.

| seed | objective | normalized | Nodes | Concepts | process edges | start | end | exact | unique | optimal alternatives | ambiguous Nodes | ambiguous Concepts | states | runtime estimate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 9002 | 10.558/12.000 | 0.880 | 1.000 | 0.884 | 0.769 | 1.000 | 1.000 | True | True | 1 | 0 | 0 | 577847 | 577 ms |
| 9003 | 8.242/12.000 | 0.687 | 1.000 | 0.842 | 1.000 | 0.000 | 0.000 | True | True | 1 | 0 | 0 | 495403 | 495 ms |
| 9004 | 11.201/12.000 | 0.933 | 1.000 | 0.927 | 1.000 | 1.000 | 1.000 | True | True | 1 | 0 | 0 | 581626 | 581 ms |

### Joint experiment conclusion

The three stored seeds completed an exact bounded-space search in `3/3` reports; production-vs-joint mapping differences numbered `7`. Node ambiguity classes appeared in `0` report(s), and concept ambiguity classes appeared in `0` report(s).

**Exact fixtures:** the deterministic synthetic suite shows that identical graphs align perfectly, arbitrary Agent labels/IDs and insertion order do not affect structural scores, reads and writes remain distinct, kind mismatches never map, topology can disambiguate similar nodes, repeated usage strengthens concept alignment, and symmetric structures are reported ambiguous. A deliberately misleading-label fixture is resolved by structure rather than text.

**Identifiability:** directed topology plus start/end roles makes the small seed Node skeletons identifiable. Concepts with repeated or slot-specific usage are usually identifiable; concepts whose usage is missing, extra, or structurally unsupported remain unmatched rather than being guessed. Symmetric duplicate subgraphs/concept usages remain valid ambiguity classes.

**Viability:** this is viable as an evaluator-private diagnostic and as a candidate for further experiments, not a production migration. Before production use, validate objective weighting and edge cases on larger adversarial graphs, retain explicit optimality bounds, and measure whether the structural mapping is stable under realistic missing/extra structure. Existing production scoring and mappings are unchanged.

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

### Joint structural alignment

- status: `ok`; method: `joint_structural_branch_and_bound`; assignment uses labels: `False`
- objective: `10.557714` / `12.000000` (normalized `0.879809`)
- search: exact=`True`, bound_hit=`False`, node states=`18703`, concept states=`524836`, edge states=`34308`, leaves=`13327`, optimal alternatives observed=`1`, count exact=`True`, optimum unique=`True`
- objective bounds: lower=`10.557714`, upper=`10.557714`
- deterministic search-runtime estimate: `577 ms` (work-unit estimate, not wall-clock time, so regenerated artifacts remain deterministic)

#### Objective components

| component | matched | Agent total | Truth total | agreement |
| --- | ---: | ---: | ---: | ---: |
| `nodes` | 6 | 6 | 6 | 1.000000 |
| `concepts` | 19 | 22 | 21 | 0.883721 |
| `activity` | 6 | 6 | 6 | 1.000000 |
| `actor` | 6 | 6 | 6 | 1.000000 |
| `system` | 4 | 4 | 4 | 1.000000 |
| `reads` | 2 | 4 | 3 | 0.571429 |
| `writes` | 2 | 3 | 3 | 0.666667 |
| `rationale` | 1 | 1 | 1 | 1.000000 |
| `condition` | 2 | 3 | 3 | 0.666667 |
| `process_edges` | 5 | 7 | 6 | 0.769231 |
| `start_node` | 1 | 1 | 1 | 1.000000 |
| `end_nodes` | 2 | 2 | 2 | 1.000000 |

#### Resolved versus ambiguous entities

- invariant Node mappings proven: `{'node_approve_high_value': 'ap', 'node_check_customer_info': 'cc', 'node_create_quotation_doc': 'cq', 'node_receive_request': 'r', 'node_send_month_end_summary': 'me', 'node_send_quotation': 'sq'}`
- invariant Concept mappings proven: `{'act_approve_high_value': 'tc_activity_approve_quotation', 'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation_doc': 'tc_activity_create_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_month_end_summary': 'tc_activity_send_month_end_summary', 'act_send_quotation': 'tc_activity_send_quotation', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_amount_at_or_below_1m': 'tc_cond_at_or_below_1m', 'cond_amount_over_1m': 'tc_cond_over_1m', 'data_customer_information': 'tc_customer', 'data_month_end_summary': 'tc_excel_summary', 'data_pricing_information': 'tc_pricing', 'data_quotation_document': 'tc_quote', 'rationale_credit_risk': 'tc_rationale_credit_risk', 'sys_crm': 'tc_system_crm', 'sys_email': 'tc_system_email', 'sys_excel': 'tc_system_excel', 'sys_quoting_system': 'tc_system_quoting'}`
- representative unmatched Agent Nodes: `none`
- representative unmatched Truth Nodes: `none`
- representative unmatched Agent Concepts: `['cond_end_of_month', 'data_quotation', 'data_quotation_request']`
- representative unmatched Truth Concepts: `['tc_cond_month_end', 'tc_request']`

##### Node ambiguity classes

- none

##### Concept ambiguity classes

- none

#### Representative structural mappings

- Agent Node -> Truth Node: `{'node_approve_high_value': 'ap', 'node_check_customer_info': 'cc', 'node_create_quotation_doc': 'cq', 'node_receive_request': 'r', 'node_send_month_end_summary': 'me', 'node_send_quotation': 'sq'}`
- Agent Concept -> Truth Concept: `{'act_approve_high_value': 'tc_activity_approve_quotation', 'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation_doc': 'tc_activity_create_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_month_end_summary': 'tc_activity_send_month_end_summary', 'act_send_quotation': 'tc_activity_send_quotation', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_amount_at_or_below_1m': 'tc_cond_at_or_below_1m', 'cond_amount_over_1m': 'tc_cond_over_1m', 'data_customer_information': 'tc_customer', 'data_month_end_summary': 'tc_excel_summary', 'data_pricing_information': 'tc_pricing', 'data_quotation_document': 'tc_quote', 'rationale_credit_risk': 'tc_rationale_credit_risk', 'sys_crm': 'tc_system_crm', 'sys_email': 'tc_system_email', 'sys_excel': 'tc_system_excel', 'sys_quoting_system': 'tc_system_quoting'}`
- Agent Edge -> Truth Edge: `{'edge_approve_to_send': 'e5', 'edge_check_to_create': 'e2', 'edge_create_to_approve': 'e3', 'edge_create_to_send': 'e4', 'edge_receive_to_check': 'e1'}`

#### Production versus joint mapping differences

| Agent entity | production Truth | joint Truth |
| --- | --- | --- |
| `cond_end_of_month` | `tc_cond_month_end` | `—` |
| `data_quotation` | `tc_quote` | `—` |
| `data_quotation_document` | `—` | `tc_quote` |
| `data_quotation_request` | `tc_request` | `—` |

#### Conditioned usage versus joint differences

| Agent Concept | usage Truth | joint Truth |
| --- | --- | --- |
| none | — | — |

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

### Joint structural alignment

- status: `ok`; method: `joint_structural_branch_and_bound`; assignment uses labels: `False`
- objective: `8.242105` / `12.000000` (normalized `0.686842`)
- search: exact=`True`, bound_hit=`False`, node states=`18703`, concept states=`437294`, edge states=`39406`, leaves=`13327`, optimal alternatives observed=`1`, count exact=`True`, optimum unique=`True`
- objective bounds: lower=`8.242105`, upper=`8.242105`
- deterministic search-runtime estimate: `495 ms` (work-unit estimate, not wall-clock time, so regenerated artifacts remain deterministic)

#### Objective components

| component | matched | Agent total | Truth total | agreement |
| --- | ---: | ---: | ---: | ---: |
| `nodes` | 6 | 6 | 6 | 1.000000 |
| `concepts` | 16 | 17 | 21 | 0.842105 |
| `activity` | 6 | 6 | 6 | 1.000000 |
| `actor` | 6 | 6 | 6 | 1.000000 |
| `system` | 1 | 1 | 4 | 0.400000 |
| `reads` | 1 | 3 | 3 | 0.333333 |
| `writes` | 2 | 3 | 3 | 0.666667 |
| `rationale` | 1 | 1 | 1 | 1.000000 |
| `condition` | 3 | 3 | 3 | 1.000000 |
| `process_edges` | 6 | 6 | 6 | 1.000000 |
| `start_node` | 0 | 0 | 1 | 0.000000 |
| `end_nodes` | 0 | 0 | 2 | 0.000000 |

#### Resolved versus ambiguous entities

- invariant Node mappings proven: `{'node_check_customer_info': 'cc', 'node_create_quotation': 'cq', 'node_manager_approve': 'ap', 'node_receive_request': 'r', 'node_send_quotation': 'sq', 'node_send_summary': 'me'}`
- invariant Concept mappings proven: `{'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation': 'tc_activity_create_quotation', 'act_manager_approve': 'tc_activity_approve_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_quotation': 'tc_activity_send_quotation', 'act_send_summary': 'tc_activity_send_month_end_summary', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_at_or_below_1m': 'tc_cond_at_or_below_1m', 'cond_month_end': 'tc_cond_month_end', 'cond_over_1m': 'tc_cond_over_1m', 'data_customer_information': 'tc_customer', 'data_quotation': 'tc_quote', 'data_summary': 'tc_excel_summary', 'rat_credit_risk': 'tc_rationale_credit_risk', 'system_quoting_system': 'tc_system_quoting'}`
- representative unmatched Agent Nodes: `none`
- representative unmatched Truth Nodes: `none`
- representative unmatched Agent Concepts: `['data_quotation_request']`
- representative unmatched Truth Concepts: `['tc_pricing', 'tc_request', 'tc_system_crm', 'tc_system_email', 'tc_system_excel']`

##### Node ambiguity classes

- none

##### Concept ambiguity classes

- none

#### Representative structural mappings

- Agent Node -> Truth Node: `{'node_check_customer_info': 'cc', 'node_create_quotation': 'cq', 'node_manager_approve': 'ap', 'node_receive_request': 'r', 'node_send_quotation': 'sq', 'node_send_summary': 'me'}`
- Agent Concept -> Truth Concept: `{'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation': 'tc_activity_create_quotation', 'act_manager_approve': 'tc_activity_approve_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_quotation': 'tc_activity_send_quotation', 'act_send_summary': 'tc_activity_send_month_end_summary', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_at_or_below_1m': 'tc_cond_at_or_below_1m', 'cond_month_end': 'tc_cond_month_end', 'cond_over_1m': 'tc_cond_over_1m', 'data_customer_information': 'tc_customer', 'data_quotation': 'tc_quote', 'data_summary': 'tc_excel_summary', 'rat_credit_risk': 'tc_rationale_credit_risk', 'system_quoting_system': 'tc_system_quoting'}`
- Agent Edge -> Truth Edge: `{'edge_approve_to_send': 'e5', 'edge_check_to_create': 'e2', 'edge_create_to_approve': 'e3', 'edge_create_to_send_direct': 'e4', 'edge_create_to_summary': 'e6', 'edge_receive_to_check': 'e1'}`

#### Production versus joint mapping differences

| Agent entity | production Truth | joint Truth |
| --- | --- | --- |
| `act_manager_approve` | `—` | `tc_activity_approve_quotation` |
| `data_quotation_request` | `tc_request` | `—` |

#### Conditioned usage versus joint differences

| Agent Concept | usage Truth | joint Truth |
| --- | --- | --- |
| none | — | — |

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

### Joint structural alignment

- status: `ok`; method: `joint_structural_branch_and_bound`; assignment uses labels: `False`
- objective: `11.200639` / `12.000000` (normalized `0.933387`)
- search: exact=`True`, bound_hit=`False`, node states=`18703`, concept states=`505703`, edge states=`57220`, leaves=`13327`, optimal alternatives observed=`1`, count exact=`True`, optimum unique=`True`
- objective bounds: lower=`11.200639`, upper=`11.200639`
- deterministic search-runtime estimate: `581 ms` (work-unit estimate, not wall-clock time, so regenerated artifacts remain deterministic)

#### Objective components

| component | matched | Agent total | Truth total | agreement |
| --- | ---: | ---: | ---: | ---: |
| `nodes` | 6 | 6 | 6 | 1.000000 |
| `concepts` | 19 | 20 | 21 | 0.926829 |
| `activity` | 6 | 6 | 6 | 1.000000 |
| `actor` | 6 | 6 | 6 | 1.000000 |
| `system` | 3 | 3 | 4 | 0.857143 |
| `reads` | 3 | 5 | 3 | 0.750000 |
| `writes` | 2 | 3 | 3 | 0.666667 |
| `rationale` | 1 | 1 | 1 | 1.000000 |
| `condition` | 3 | 3 | 3 | 1.000000 |
| `process_edges` | 6 | 6 | 6 | 1.000000 |
| `start_node` | 1 | 1 | 1 | 1.000000 |
| `end_nodes` | 2 | 2 | 2 | 1.000000 |

#### Resolved versus ambiguous entities

- invariant Node mappings proven: `{'node_approve_high_value': 'ap', 'node_check_customer_info': 'cc', 'node_create_quotation': 'cq', 'node_receive_request': 'r', 'node_send_month_end_summary': 'me', 'node_send_quotation_customer': 'sq'}`
- invariant Concept mappings proven: `{'act_approve_high_value': 'tc_activity_approve_quotation', 'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation': 'tc_activity_create_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_month_end_summary': 'tc_activity_send_month_end_summary', 'act_send_quotation_customer': 'tc_activity_send_quotation', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_high_value': 'tc_cond_over_1m', 'cond_low_value': 'tc_cond_at_or_below_1m', 'cond_month_end': 'tc_cond_month_end', 'data_customer_information': 'tc_customer', 'data_month_end_summary': 'tc_excel_summary', 'data_pricing_information': 'tc_pricing', 'data_quotation': 'tc_quote', 'rationale_credit_risk': 'tc_rationale_credit_risk', 'system_crm': 'tc_system_crm', 'system_email': 'tc_system_email', 'system_quoting_system': 'tc_system_quoting'}`
- representative unmatched Agent Nodes: `none`
- representative unmatched Truth Nodes: `none`
- representative unmatched Agent Concepts: `['data_quotation_request']`
- representative unmatched Truth Concepts: `['tc_request', 'tc_system_excel']`

##### Node ambiguity classes

- none

##### Concept ambiguity classes

- none

#### Representative structural mappings

- Agent Node -> Truth Node: `{'node_approve_high_value': 'ap', 'node_check_customer_info': 'cc', 'node_create_quotation': 'cq', 'node_receive_request': 'r', 'node_send_month_end_summary': 'me', 'node_send_quotation_customer': 'sq'}`
- Agent Concept -> Truth Concept: `{'act_approve_high_value': 'tc_activity_approve_quotation', 'act_check_customer_info': 'tc_activity_check_customer', 'act_create_quotation': 'tc_activity_create_quotation', 'act_receive_request': 'tc_activity_receive_request', 'act_send_month_end_summary': 'tc_activity_send_month_end_summary', 'act_send_quotation_customer': 'tc_activity_send_quotation', 'actor_manager': 'tc_actor_manager', 'actor_sales_employee': 'tc_actor_sales', 'cond_high_value': 'tc_cond_over_1m', 'cond_low_value': 'tc_cond_at_or_below_1m', 'cond_month_end': 'tc_cond_month_end', 'data_customer_information': 'tc_customer', 'data_month_end_summary': 'tc_excel_summary', 'data_pricing_information': 'tc_pricing', 'data_quotation': 'tc_quote', 'rationale_credit_risk': 'tc_rationale_credit_risk', 'system_crm': 'tc_system_crm', 'system_email': 'tc_system_email', 'system_quoting_system': 'tc_system_quoting'}`
- Agent Edge -> Truth Edge: `{'edge_approve_to_send_customer': 'e5', 'edge_check_to_create': 'e2', 'edge_create_to_approve': 'e3', 'edge_create_to_month_end': 'e6', 'edge_create_to_send_customer': 'e4', 'edge_receive_to_check': 'e1'}`

#### Production versus joint mapping differences

| Agent entity | production Truth | joint Truth |
| --- | --- | --- |
| `data_quotation_request` | `tc_request` | `—` |

#### Conditioned usage versus joint differences

| Agent Concept | usage Truth | joint Truth |
| --- | --- | --- |
| none | — | — |
