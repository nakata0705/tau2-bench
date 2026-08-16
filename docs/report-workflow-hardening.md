# business_interview Workflow-first hardening レポート (v2)

> 本変更は既存 v2 の Workflow-first architecture を **single-stakeholder
> Workflow Interview Benchmark として信頼できる形に hardening** するもので、
> 新機能追加（multi-stakeholder 等）は含まない。後方互換は要求せず、
> v2 設計上の不都合は breaking change で修正した。generic tau2 core への
> 変更はゼロ。

## 1. 発見した問題（修正前の現状）

既存 v2（`98ff1c1`）を authoritative source として監査し、以下が実際に存在する
ことを確認した。

- **P0 step ID**: 評価は content fuzzy matching（Jaccard）で step を対応付けるが、
  transition / branch / challenge / improvement は **agent の生 step id をそのまま**
  使っていた。agent が任意 id（`request` / `check_customer` 等）を使うと
  transition が一切一致しない。ground truth の `s1..s6` を当てるゲームになっていた。
- **P0 truth completeness**: 必須化した評価軸（actor_manager / data_pricing /
  data_sent / outcome / summary 等）が user_scenario の known_info に存在せず、
  simulator が「自発的に話さない」のではなく「質問されても答えられない」可能性。
- **P1 NONE vs UNKNOWN**: `rationale_known` が「聞いていない」と「UNKNOWN を記録」を
  区別できず、未調査が評価上 UNKNOWN 扱いになる。
- **P1 rationale 内容**: `rationale_known` しか見ず、epistemic status / source /
  content を ground truth と比較していない。`credit risk` でなく
  `tax reporting` でも PASS になり得た。
- **P1 EN/JA**: JA を `step_recall >= 0.5` で妥協。surface language 依存の
  token 一致で同等評価にならない。
- **P2 step matching**: `jaccard(action) + actor(0.3) + system(0.3)` で、
  actor/system が同一なら action が全く異なっても match（0.6 > 0.45）する誤match。
- **P2 data precision**: 再現率のみで、架空 data を大量追加しても満点。
- **P2 transition condition**: from/to のみ評価。condition 欠落・反転・別条件が FAIL にならない。
- **P2 workflow metadata**: trigger / purpose / outcome を診断で一切評価していない。
- **P2 challenge quality**: `challenged=True` だけで PASS。why/owner/evidence/removal/deletion
  の調査有無を分離していない。
- **P2 improvement order**: proposal kind 順序のみ。necessity 未調査のまま automate が通る。
- **Leakage**: policy / tool description は clean だが、task.description.notes と
  evaluation_criteria が agent-visible でないことを明示する test が無かった。

## 2. 変更ファイル

- `src/tau2/domains/business_interview/concepts.py`（**新規**）: bilingual な
  scenario-local concept と resolver。
- `src/tau2/domains/business_interview/data_model.py`: `Necessity` に investigation
  flags、`WorkflowEvaluation` を新指標へ拡張。
- `src/tau2/domains/business_interview/ground_truth.py`: concept 化、stakeholder
  knowledge requirements。
- `src/tau2/domains/business_interview/semantic.py`: 概念ベース評価・id canonicalization・
  precision-aware data・condition-aware transition・metadata・challenge 品質。
- `src/tau2/domains/business_interview/tools.py`: `challenge_step` の dimension 化、
  investigation flag の記録。
- `data/tau2/domains/business_interview/tasks.json`: 参照 trajectory（challenge を
  dimension 化）、scenario known_info を truth-complete に拡充。
- `tests/test_domains/test_business_interview/test_workflow_business_interview.py`:
  falsification A-R を含む 41 test へ全面改訂。
- `src/tau2/domains/business_interview/README.md`: 設計・指標・tool 仕様を更新。
- `docs/report-workflow-hardening.md`（本レポート）。

## 3. data model 変更

`Necessity` に investigation と outcome を分離するフラグを追加:

- `investigated`（why を聞き、結果を記録したか。UNKNOWN 記録も含む）
- `owner_investigated` / `evidence_investigated` / `removal_investigated` /
  `deletion_considered`
- `owner` / `evidence`（調査結果の値。無くても有効）
- 旧 `owner_identified` / `evidence_identified` を削除（`*_investigated` + `owner`/`evidence` に統合）

`WorkflowEvaluation` を拡張:

- metadata: `trigger_accuracy` / `purpose_accuracy` / `outcome_accuracy`
- data: `data_{read,write}_{recall,precision}`
- rationale: `confirmed_rationale_ok`（status/source/content）
- challenge: `challenge_target_identified` / `why_investigated` /
  `owner_investigated` / `evidence_investigated` / `removal_investigated` /
  `deletion_considered` / `challenge_done`
- transition は `transition_accuracy`（from/to **and** condition）

## 4. arbitrary step ID canonicalization

`semantic._match_steps` が各 reconstruct step の action を concept に解決し
`rec_step_id -> gt_step_id` の `id_map` を構築。`_rec_graph` は transition /
branch の from/to/path をこのマップで canonical 化してから評価する。これにより
transition・branch・condition・questionable-step challenge・improvement の全てが
matched-step mapping を使う。hidden ground-truth id（`s1`/`s2`...）を agent に
当てさせる必要は一切ない。

必須 test: `test_arbitrary_step_ids_full_pass`, `test_arbitrary_ids_canonicalise_graph_not_raw_ids`,
`test_challenge_target_found_via_id_mapping`。

## 5. stakeholder truth completeness

`ground_truth.stakeholder_knowledge_requirements()` が、evaluator が要求する事実の
うち stakeholder が知るべき事実（trigger/purpose/outcome、actor、system、
reads/writes、branch/transition condition、approval owner、confirmed rationale、
conditional step）を axis + bilingual signals のデータ契約で宣言。
`test_scenario_covers_stakeholder_knowledge_requirements` が各 scenario の
known_info が全 requirement を満たすことを構造的に保証する。

scenario の `known_info` を拡充し（manager 承認、pricing 利用、email 送付、
accurate outcome、summary 等を追加）、「自発的に話さない」ことで難易度を作り、
「質問されても答えを知らない simulator」にしない。Agent performance が user
simulator の hallucination に依存しない。Ground truth は agent に隠したまま、
stakeholder が知る truth は simulator に与える。

## 6. NONE vs UNKNOWN

- `investigated=False`（理由を聞いていない / NOT_INVESTIGATED）→ FAIL
- `investigated=True` + UNKNOWN 記録 → PASS
- UNKNOWN を FACT に推測で昇格 → epistemic FAIL（`fabricated_rationale`）

必須 test: `test_uninvestigated_rationale_fails`（D）, `test_investigated_unknown_passes`（E）,
`test_unknown_fabricated_as_fact_fails`（F）。

## 7. confirmed rationale 評価

`GTNecessity` に `rationale_concept`（例 `credit_risk`）、`expected_source`、
`expected_status` を追加。`semantic._confirmed_rationale_ok` が以下を比較:

- epistemic status（正解が FACT なのに BELIEF 記録 → FAIL）
- source（`sales`/`営業` でない → FAIL）
- rationale semantic content（`credit_risk` concept に解決しない → FAIL）

bounded scenario-local semantic matcher（`concepts.resolve`、substring 信号、
deterministic）。LLM judge は使わない。

必須 test: `test_confirmed_rationale_content_must_match`（G）, `test_confirmed_rationale_wrong_source_fails`（H）,
`test_correct_content_as_belief_fails`。

## 8. EN/JA semantic matching

`concepts.py` に language-independent concept（receive_request / check_customer /
create_quote / approve_quote / send_quote / month_end_summary、および data /
condition / metadata / rationale concept）を定義。agent は concept id を入力しない。
evaluator 側が EN/JA 自由文を bilingual signal で concept に解決するため、
意味的に同じ EN/JA reconstruction は主要 structural metrics が一致し、両方 full
structural PASS になる。JA を `step_recall >= 0.5` で妥協しない。

必須 test: `test_en_full_structural_pass`（K）, `test_ja_equivalent_full_structural_pass`（L）,
`test_en_ja_structural_metrics_equivalent`。

## 9. step matching

action を primary signal に concept 解決で match。actor/system は tie-break と
accuracy 指標にのみ使う。よって同じ actor/system でも action が全く異なれば
`missing + unexpected` になる。

必須 test: `test_same_actor_system_wrong_action_no_match`（I）。

## 10. data precision

read/write それぞれに recall と precision を算出（concept 解決 + token overlap
フォールバック）。必須 data を含んでも架空 data を追加すると precision が下がる。

必須 test: `test_exact_data_passes`, `test_missing_required_data_fails_recall`,
`test_invented_extra_data_fails_precision`（J）。

## 11. transition condition

`GTTransition` に `condition_concept` を追加。`transition_accuracy` は from/to
**and** condition concept を判定する。condition 欠落・反転・別条件は FAIL。
high-value / normal / month-end の各 path を condition 付きで評価。

branch の役割は「発散構造 + 分岐が condition 付きか」の grouping に整理し、
個別 path の routing condition は transition 側で重複評価させない
（`_rec_graph` で transition と branch を分離し、branch condition が transition
condition を上書きしない）。

必須 test: `test_wrong_edge_fails_transition`（B）, `test_reversed_condition_fails`（C）,
`test_missing_condition_fails`（C）, `test_missing_branch_path_fails_branch_recall`。

## 12. workflow metadata

trigger / purpose / outcome を `trigger_accuracy` / `purpose_accuracy` /
`outcome_accuracy` として概念解決で測定し、`structural_pass` に含める。
purpose は将来 Workflow 自体の necessity challenge に再利用可能。

必須 test: `test_workflow_metadata_accuracy`, `test_wrong_purpose_fails_metadata`。

## 13. necessity / challenge 評価

`challenge_step(step_id, dimension, question)` で why / owner / evidence /
removal / deletion を個別に記録。questionable step は id mapping で特定し、
5 次元を別々に評価（`why_investigated` 等）。`challenge_done` は 5 次元全ての
調査を要求。「owner が見つからない」「evidence がない」も調査結果なら有効
（value の有無でなく調査有無を評価）。未調査と UNKNOWN/NONE_FOUND を区別。
`challenge_pass` は単なる `challenged=True` より強い。

必須 test: M/N/O（`test_owner_not_investigated_fails_challenge` 等）,
`test_no_challenge_at_all_fails`, `test_challenge_target_found_via_id_mapping`,
`test_diagnostics_explain_challenge_failure`。

## 14. improvement order

Question→Delete→Simplify→Accelerate→Automate を維持しつつ、necessity 調査が先に
行われたかを判定。非 question の最初の提案は調査済みを要求、automate は
why+owner+evidence の full 調査を要求。「理由不明・証拠なしと確認して deletion
candidate として扱う」は良い挙動として PASS。

必須 test: `test_automate_before_investigation_fails`（P）, `test_question_before_automate_ok`,
`test_delete_after_investigation_ok`。

## 15. leakage audit

agent-visible は policy / tool description / initial assistant prompt / agent へ
渡る task metadata のみ。audit の結果:

- `LLMAgent.system_prompt` は domain policy のみで構成され（`llm_agent.py`）、
  task.description / evaluation_criteria は agent に渡らない。
- policy / tool description に scenario 固有 answer（hidden step / branch threshold /
  confirmed rationale / questionable step / canonical id）が無いことを test で保証。
- tasks.json の `description.notes` と `evaluation_criteria.actions` は evaluator-only
  （Task の docstring: "This can be sent to the evaluator"）で、agent prompt には
  含まれない。`test_agent_system_prompt_has_no_ground_truth` で明示。
- stakeholder-visible knowledge は agent に直接渡らないため leakage ではない。

必須 test: R（`test_policy_has_no_hidden_workflow_identity` /
`test_tool_descriptions_have_no_hidden_workflow_identity` /
`test_agent_system_prompt_has_no_ground_truth` /
`test_ground_truth_is_evaluator_only_not_in_scenario`）。

## 16. falsification A-R 結果

`tests/test_domains/test_business_interview/` の 41 test で全て検証。A-R の対応:

| Case | test | 結果 |
|------|------|------|
| A 任意 step ID で完全復元 | `test_arbitrary_step_ids_full_pass` | ✅ PASS |
| B correct steps / wrong edge | `test_wrong_edge_fails_transition` | ✅ transition FAIL |
| C correct edge / wrong condition | `test_reversed_condition_fails` / `test_missing_condition_fails` | ✅ condition FAIL |
| D rationale 未調査 | `test_uninvestigated_rationale_fails` | ✅ rationale FAIL |
| E UNKNOWN 確認 | `test_investigated_unknown_passes` | ✅ PASS |
| F UNKNOWN→FACT | `test_unknown_fabricated_as_fact_fails` | ✅ epistemic FAIL |
| G wrong confirmed rationale | `test_confirmed_rationale_content_must_match` | ✅ FAIL |
| H wrong rationale source | `test_confirmed_rationale_wrong_source_fails` | ✅ FAIL |
| I same actor/system + wrong action | `test_same_actor_system_wrong_action_no_match` | ✅ missing+unexpected FAIL |
| J invented read/write data | `test_invented_extra_data_fails_precision` | ✅ precision FAIL |
| K EN full structural | `test_en_full_structural_pass` | ✅ structural PASS |
| L equivalent JA full structural | `test_ja_equivalent_full_structural_pass` | ✅ structural PASS |
| M owner 未調査 | `test_owner_not_investigated_fails_challenge` | ✅ challenge FAIL |
| N evidence 未調査 | `test_evidence_not_investigated_fails_challenge` | ✅ challenge FAIL |
| O removal impact 未調査 | `test_removal_impact_not_investigated_fails_challenge` | ✅ challenge FAIL |
| P automate before investigation | `test_automate_before_investigation_fails` | ✅ improvement-order FAIL |
| Q quality 正しい / finish なし | `test_quality_pass_protocol_fail` | ✅ quality PASS, protocol FAIL |
| R agent-visible leakage なし | leakage tests | ✅ 検出なし |

加えて: good reconstruction 全軸 PASS、exact/precision data、EN/JA metrics 一致、
metadata、challenge diagnostics を test で担保。

## 17. business_interview test 結果

```
uv run pytest tests/test_domains/test_business_interview/ -q
41 passed, 2 warnings
```

## 18. core tests / make check-all

```
make check-all  (ruff check + ruff format)
All checks passed! / 337 files left unchanged

uv run pytest tests/ --ignore=tests/test_voice --ignore=tests/test_streaming \
  --ignore=tests/test_gym --ignore=tests/test_domains/test_banking_knowledge
269 passed, 1 xpassed, 2 failed
```

失敗 2 件は `tests/test_run.py` の `test_run_tasks_env_assertions` と
`test_run_tasks_nl_assertions`。いずれも gpt-3.5-turbo を実呼び出しし reward==1.0
を assert する LLM 依存の flaky テストで、本変更の `business_interview` とは無関係。
`test_run_tasks_env_assertions` は変更前の clean baseline（stash）でも失敗し、
`test_run_tasks_nl_assertions` は同一ブランチで pass/fail が揺れる（本変更と無関係な
モデル出力依存）。generic core は未変更のため、これらの失敗は本変更に起因しない。

## 19. EN/JA real run 結果（DeepSeek, concurrency=1）

`deepseek/deepseek-chat` を agent/user 双方に使用。データは gitignored の
`data/simulations/` に保存（レポート参照用）。

- EN `quotation_workflow_1`（3 trials）:
  `20260816_112810_business_interview_llm_agent_deepseek-chat_user_simulator_deepseek-chat`
- JA `quotation_workflow_1_ja`（3 trials）:
  `20260816_113333_business_interview_llm_agent_deepseek-chat_user_simulator_deepseek-chat`

主要指標（trial 毎 / 平均）:

| metric | EN (3) | JA (3) |
|---|---|---|
| step_recall | 1.0, 1.0, 0.83 → 0.94 | 0.67, 0.83, 0.83 → 0.78 |
| transition_accuracy | 0.83, 0.5, 0.67 → 0.67 | 0.17, 0.0, 0.17 → 0.11 |
| branch_recall | 1.0, 0.0, 1.0 → 0.67 | 0.0, 0.0, 0.0 → 0.0 |
| rationale_coverage | 0.5, 0.5, 0.0 → 0.33 | 0.5, 1.0, 0.5 → 0.67 |
| confirmed_rationale_ok | F,F,F → 0.00 | T,T,F → 0.67 |
| challenge_target_identified | T,T,F → 0.67 | F,T,T → 0.67 |
| challenge_done | 0.0 | 0.0 |
| structural_pass / rationale_pass / challenge_pass | 0.0 / 0.0 / 0.0 | 0.0 / 0.33 / 0.0 |
| protocol_pass | 1.0 | 1.0 |
| quality_pass | 0.0 | 0.0 |
| data_write_precision | 0.47 | 0.37 |

スコアの高さは DoD ではない。確認したいこと（DoD）:

- **arbitrary step ID で評価可能**: 実 run で agent が step を独自 id/順序で記録しても
  concept canonicalization により評価・diagnostics が正常に動作。
- **simulator が必要 truth を回答可能**: 実 run で user が credit risk / 1,000,000
  閾値 / month-end を答え、unknown も UNKNOWN として回答（truth completeness 成立）。
- **NONE と UNKNOWN を区別**: ユニット test（D/E/F）で担保。
- **rationale 内容/source まで評価**: EN run で credit risk を BELIEF 記録した trial が
  `confirmed_rationale_ok=False` で検出された。
- **EN/JA で同等評価**: 同一 metric セットで両言語を評価（`step_recall>=0.5` の妥協なし）。
- **challenge failure 理由を diagnostics で説明可能**: 実 run で
  `challenge_target_identified=False`（agent が manager 承認 step を challenge し、
  month-end Excel step を challenge していない）が diagnostics から判読可能。

EN trial 3 の診断例:
`step_recall=0.83, unexpected=1, transition=0.5, branch_recall=0, rationale_coverage=0,
challenge_target_identified=False, quality_pass=False, protocol_pass=True` —
「構造復元に失敗 + 正しい questionable step を challenge していない」ことが
deterministic に説明できる。

## 20. generic core 差分

**なし**。`src/tau2/environment` / `src/tau2/evaluator` / `src/tau2/agent` /
`src/tau2/runner` 等の generic core は未変更。`git diff` は
`business_interview` ドメイン・データ・test・docs のみに限定される。

## 21. 残る既知の弱点

- Step matching は bounded bilingual concept への substring 信号解決。表現が
  concept 信号から完全に外れる free text（例外的な言い回し）は recall 低下の余地。
- Data precision は概念解決 + token フォールバック。agent の data 表現が ground
  truth の data concept 信号と一致しない場合、precision が厳しめに出る。
- `improvement_order_ok` は提案 kind 順序 + necessity 調査有無の判定まで。
  To-Be 提案の深い意味評価はしない（非目標）。
- condition 評価は bounded condition concept のみ。閾値数値の厳密一致は
  方向キーワードに依存。
- real run で agent（deepseek-chat）が branch/month-end step を十分に発見できない
  のは benchmark の想定難易度どおりだが、モデル依存。

## 22. multi-stakeholder へ進める状態か

**進める土台は完成**。次の要素が整った:

- Workflow graph が ground truth の中心（step / transition / branch / condition /
  metadata / data recall+precision）で、任意 step id / EN/JA で deterministic 評価。
- `Necessity.source`（rationale source）と `owner` が「誰が言ったか / 誰の要求か」を保持。
- NONE vs UNKNOWN の分離により、複数 stakeholder の「知らない」を正確に扱える。
- scenario の truth completeness を構造 test で保証する仕組みを確立。

multi-stakeholder 化では次を予定: 各 step の necessity/rationale を stakeholder 別の
claim リスト化、stakeholder 間の belief 差・矛盾検出、`Necessity` の source 別
claim への拡張、conflicting beliefs の評価。これらは本タスクの非目標として未実装。

## Commit / Push 情報

- commit message: `fix: harden workflow benchmark evaluation and scenario truth`
- branch: `business-interview`（現行）→ `origin/business-interview`
- 本レポート: `docs/report-workflow-hardening.md`
