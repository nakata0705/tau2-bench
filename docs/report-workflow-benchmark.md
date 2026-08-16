# business_interview v2 — Workflow-first benchmark 実装レポート

## 1. 目的と方針

旧 `business_interview` は Fact / Exception / Rationale を個別に拾うことが中心で、
「隠れた業務 Workflow を復元できるか」を構造的に評価できなかった。本変更は
**中心概念を Fact から Workflow Graph に変更**する breaking redesign (v2) であり、
後方互換は要求しない。

## 2. 削除した旧設計

- 旧 `quotation_process_interview_1` / `quotation_belief_uncertainty_1` /
  `quotation_multi_exception_1`（および `_ja`）の全 scenario
- 旧 Fact-centric data model: `InterviewFact` / `InterviewException` /
  `InterviewUncertainty`
- legacy evaluator / metrics: `TopicEvaluation` / `ObjectiveRationale` /
  `ClaimEvaluation` / `rationale_status` / `rationale_correct`
- 旧 `Topic` enum、`RationaleValue` enum、旧 topic inference（`topic_of_finding` /
  content-signal フォールバック）
- 旧 `record_fact` / `record_exception` / `record_uncertainty` API
- 旧 scenario 固有 keyword assertions（`assert_fact_recorded` 等）
- 旧 backward-compatibility tests（`test_tools_business_interview.py` 全体）
- 旧 report docs（report-hardening / report-source-aware-model /
  report-multi-axis / report-topic-scoped）

## 3. 新 Workflow data model

`data_model.py`:

- `Workflow`: id / name / trigger / purpose / outcome / steps / transitions / branches
- `WorkflowStep`: id / action / actor / system / reads / writes / condition / necessity
- `Transition`: from_step / to_step / condition
- `Branch`: from_step / condition / paths
- `Necessity`: rationale_known / rationale / source / epistemic_status /
  owner_identified / evidence_identified / requirement_type / challenged /
  challenges / deletion_candidate
- `Improvement`: step_id / kind / note / order
- `WorkflowDB`: workflow / improvements / interview_complete / summary

`EpistemicStatus` = FACT / BELIEF / UNKNOWN。FACT は「source が確定的に述べた」で、
objective truth と同一視しない。

## 4. ground truth / reconstructed workflow の構造

- `ground_truth.py`: `WorkflowGroundTruth`（scenario_id / workflow メタ / steps /
  transitions / branches / questionable_step）。scenario 毎に canonical な workflow
  を evaluator-only で保持。EN/JA は同一 ground truth を共有（`_ja` canonicalization）。
- Agent は自分で step id / action を決めて workflow を構築。Evaluator は
  **content ベースの step matching**（action の言語非依存 signature + actor/system
  一致）で ground truth と比較。hidden canonical id 当てゲームにしない。

## 5. Agent tools

`tools.py`: `create_workflow` / `add_step` / `connect_steps` / `add_branch` /
`set_step_rationale` / `set_step_unknown` / `challenge_step` /
`record_necessity_detail` / `propose_improvement` / `finish_interview`。
assertion helpers（env_assertions）と `get_eval_diagnostics` は agent 非公開。

## 6. evaluation metrics

`semantic.py` の `evaluate(db, scenario_id)` が `WorkflowEvaluation` を返す:

- step_recall / unexpected_step_count
- actor_accuracy / system_accuracy
- data_read_accuracy / data_write_accuracy
- transition_accuracy
- branch_recall / branch_condition_accuracy
- rationale_coverage / uncertainty_handling / fabricated_rationale
- challenge_done / improvement_order_ok
- structural_pass / rationale_pass / challenge_pass / protocol_pass / quality_pass

reward は env_assertions（finish + structural + rationale + challenge）の積。
詳細は diagnostics（`get_eval_diagnostics` → `reward_info.info["diagnostics"]`）。

## 7. necessity / challenge model

各 step の `Necessity` で「なぜ必要か」「誰の要求か」「根拠は何か」「なくすと
どうなるか」を記録・評価。`challenge_step` で質問、`propose_improvement` で改善案
（question→delete→simplify→accelerate→automate）。automate を necessity 確認より
先に提案すると `improvement_order_ok=False`。

## 8. leakage 対策

- Agent-visible policy（`policy.md`）: 一般的な BA・workflow 発見・necessity
  challenge・改善順序のみ。scenario 固有の step / branch / rationale / canonical id
  なし。
- Agent tools の description: 汎用のみ。
- Stakeholder-visible（`tasks.json` の user_scenario）: その人が知る workflow と
  belief / 未知情報のみ。
- Evaluator-only: `ground_truth.py` の完全な workflow。
- leakage test: policy / tool description に hidden term が無いことを保証。

## 9. scenario 概要

`quotation_workflow_1`（+`_ja`）:
- 6 Step（依頼受付→CRM顧客確認→見積作成→高額承認→送付→月末Excel集計）
- 分岐1（金額閾値）、条件パス1（月末）
- 2+ systems（crm / quoting / email / excel）、複数 read/write
- confirmed rationale 1（高額承認 = 与信管理）
- UNKNOWN rationale 1（月末Excel = 理由不明、経理が必要らしいという BELIEF のみ・証拠なし）
- questionable legacy step 1（月末Excel）→ challenge 対象

## 10. falsification 結果（A-L）

| Case | 内容 | 期待 | 結果 |
|------|------|------|------|
| A | 全 Step・順序・branch 正しく復元 | structural PASS | ✅ |
| B | 1 Step 欠落 | step recall FAIL | ✅ |
| C | Step 揃うが順序誤り | transition FAIL | ✅ |
| D | branch を linear に記録 | branch FAIL | ✅ |
| E | actor 誤り | actor FAIL | ✅ |
| F | system 誤り | system FAIL | ✅ |
| G | read/write data 取り違え | data FAIL | ✅ |
| H | rationale 未確認 | rationale coverage FAIL | ✅ |
| I | UNKNOWN rationale を FACT 化 | epistemic FAIL | ✅ |
| J | legacy step を challenge しない | challenge FAIL | ✅ |
| K | 即 automation 提案 | improvement-order FAIL | ✅ |
| L | semantic 正しいが finish なし | quality PASS / protocol FAIL | ✅ |

加えて: leakage test、EN/JA 同一 ground truth、good reconstruction 全軸 PASS、
end-to-end EnvironmentEvaluator reward 1.0、missing-step reward 0。

## 11. Verification

- 新 test suite（`test_workflow_business_interview.py`）: **22 passed**
- `make check-all`（ruff lint + format）: clean
- core tests: 下記
- EN/JA representative run: 下記

## 12. generic core 差分

**なし**。`src/tau2/environment` / `src/tau2/evaluator` は未変更（既存の薄い
`get_eval_diagnostics` hook を利用）。

## 13. 残る弱点

- Step matching は content ベースの fuzzy で、agent の action 表現が ground truth と
  大きく異なる場合（特に JA で latin 語彙が無い action）は recall が下がる。
- branch / data の exact 照合は言語依存になり得る（JA は部分一致）。
- `improvement_order_ok` は proposal の kind 順序を見る最小実装（deep な To-Be
  判定はしない）。
- multi-stakeholder の source 比較は未実装（`Necessity.source` は単一）。

## 14. 次に multi-stakeholder へ進めるか

**進める土台はできた**。Workflow が ground truth の中心になり、`Necessity.source`
が「誰が言ったか」を持てる。次は:
1. 各 step の necessity / rationale に複数 stakeholder の claim を紐付ける。
2. stakeholder ごとの belief 差・矛盾検出を workflow 上で評価。
3. `Necessity` を source 別の claim リストに拡張。

現状は「single stakeholder + hidden workflow reconstruction + necessity
challenge」まで成立。
