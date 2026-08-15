# business_interview ハードニング（leak-free 発見 + evaluator 堅牢化）実装レポート

## 1. 目的

前回までのトピック単位評価（`docs/report-topic-scoped-evaluation.md`）では、評価が
正しくトピックに結び付くようになったが、以下の問題が残っていた。

1. **ground-truth leakage**: Agent-visible な policy / tool description に
   canonical topic ID（`month_end_excel` / `high_value_quote`）や具体的な
   exception（月末Excel・高額見積・credit risk）が例示されており、Agent が
   発見前に答えを知り得た。
2. **JA シナリオの canonical 化不足**: `quotation_multi_exception_1_ja` が
   `SCENARIO_TOPICS` に登録されておらず、`high_value_quote` を完全に見逃しても
   required topic から消えて `interview_quality_pass` が誤って真になれた。
3. **UNKNOWN と NONE の混同**: unknown-rationale topic で rationale 未記録
   （NONE）も correct 扱いになっていた。
4. **normal_fact_recall の false positive**: exception の rationale FACT だけで
   normal_fact_recall が真になった。
5. **topic misattribution への耐性不足**: explicit `topic` を無条件に信頼して
   いたため、捏造 rationale を別トピックへ偽装して evaluator を回避できた。
6. **known-rationale の精度不足**: `credit risk and tax reporting` のように期待値に
   余計な捏造を足しても PASS した。
7. **FACT/BELIEF/UNKNOWN の意味の曖昧さ**: 「objectively true」という表現と
   「stakeholder が確定的に述べた」が混在していた。
8. **EN-only / JA-only の実行が不便**: `base` しか split がなかった。

## 2. 変更ファイル

| ファイル | 種別 | 内容 |
|----------|------|------|
| `src/tau2/domains/business_interview/data_model.py` | 変更 | `EpistemicStatus` docstring を「source がどう述べたか」に統一、`rationale_correct` docstring を UNKNOWN/NONE・precision に更新 |
| `src/tau2/domains/business_interview/semantic.py` | 変更 | `EXTRA_RATIONALE_SIGNALS`、`canonical_scenario_id`、`topic_of_finding` の交差チェック、`is_exception_related`、`asserts_extra_rationale`、known-rationale over-claim、UNKNOWN/NONE、normal_fact_recall 修正、JA canonicalization |
| `src/tau2/domains/business_interview/tools.py` | 変更 | record 3 ツールの description から hidden identity 除去、`assert_no_unsupported_rationale` を topic 交差チェック使用に |
| `data/.../business_interview/policy.md` | 変更 | 「Attributing findings to topics」から scenario 固有 ID・例を除去し一般BA行動のみに |
| `data/.../business_interview/tasks.json` | 変更 | multi_exception EN/JA の stakeholder prompt を「確定的に把握」に明確化 |
| `data/.../business_interview/split_tasks.json` | 変更 | `base_en` / `base_ja` split を追加（`base` は不変） |
| `src/tau2/domains/business_interview/README.md` | 変更 | hardening 節・split・per-topic 意味を更新 |
| `tests/.../test_tools_business_interview.py` | 変更 | hardening falsification suite（L1/L2/J1/U1/U2/N1/T1/T2/T3/K1/K2 + split）を追加 |
| `docs/report-hardening.md` | **新規** | 本レポート |

generic core（`Environment.get_eval_diagnostics` / `EnvironmentEvaluator`）への
追加変更は **なし**（既存 hook を維持）。

## 3. leakage 箇所と除去方法

- **policy.md**: 「Attributing findings to topics」節に `month_end_excel` /
  `high_value_quote` と「月末Excel」「高額見積」の例があった → 除去し、
  「例外はどの業務要素に関わるか（誰・いつ・何に適用）を尋ね、canonical topic
  identifier を一貫して記録し、別トピックの理由を別トピックに付けない」という一般
  BA 行動のみに。
- **record_fact / record_exception / record_uncertainty の description**: `topic`
  引数の例（`month_end_excel` / `high_value_quote`）を除去し、canonical identifier
  の概念だけ記述。
- **guard テスト**: `test_L1_policy_has_no_hidden_exception_identity` /
  `test_L2_agent_tool_descriptions_have_no_hidden_exception_identity` が
  `month_end_excel` / `high_value_quote` / `month-end` / `high-value` /
  `credit risk` が policy・tool description に無いことを保証。
- **別経路**: canonical topic ID・required topics は domain 側
  （`semantic.py` / `data_model.py`）にのみ存在し、agent-visible の policy /
  scenario / tool description には漏れない。stakeholder が知っている内容
  （credit risk が高額見積の理由であること）は user_scenario に残るが、それは
  stakeholder 自身の知識であり発見対象。月末Excel の理由（UNKNOWN）はどの経路にも
  存在しない。

## 4. canonical topic の扱い

- `Topic` enum（`month_end_excel` / `high_value_quote`）は domain 側に保持。
- `TOPIC_SPECS` が各 topic の ground truth（`expected_rationale_status` /
  `content_signals` / `rationale_value_signals`）を持つ。
- `SCENARIO_TOPICS` が各 scenario の required topics を宣言。これらは evaluator /
  task evaluation criteria 側（agent 非可視）にのみ存在。
- Agent は `topic` 引数を任意に指定できるが、`topic_of_finding` が content と交差
  チェックするため、偽装・誤割当は報われない。

## 5. JA canonicalization

- `canonical_scenario_id()` が `_ja` 接尾辞を剥がして canonical scenario を解決。
  `quotation_multi_exception_1_ja -> quotation_multi_exception_1`。generic i18n
  framework は導入せず、domain 内の最小 canonicalization。
- 効果: JA multi-exception で `high_value_quote` を完全に見逃しても required topics
  から消えず、`interview_quality_pass=False`（テスト `test_J1_...`）。

## 6. UNKNOWN vs NONE

- unknown-rationale topic（`month_end_excel`）:
  - `UNKNOWN`（調査して不明を記録）= correct
  - `NONE` / None（rationale 未確認・未記録）= incorrect
- `_evaluate_topic` の `rationale_correct` を `rationale_status == "UNKNOWN"` のみに。
- テスト `test_U1_...`（NONE → incorrect）、`test_U2_...`（UNKNOWN → correct）。

## 7. normal_fact_recall 修正

- 旧: `not is_exception_content(content)` のみ（month/excel/accounting ベース）。
  high_value の credit-risk FACT はこれに該当せず normal fact 扱いだった。
- 新: `is_exception_related(content, topic)`（canonical exception topic に帰属する
  finding を含む）を除外。
- テスト `test_N1_...`: high_value の credit-risk FACT だけでは
  `normal_fact_recall=False`。positive control `test_normal_process_fact_sets_...`。

## 8. topic attribution 判定

- `topic_of_finding(topic, content)` を交差チェックに変更:
  - explicit topic と content の一意な content-inferred topic が一致 → topic を使用
  - content が別トピックを一意に示す → reported topic を無条件優先せず **None**
    （どちらのトピックにも credit しない）
  - content が曖昧（0 または複数マッチ）→ reported topic を信頼（勝手に決めない）
  - explicit topic なし → content の一意マッチのみ（曖昧なら None）
- `assert_no_unsupported_rationale` も `topic_of_finding` を経由。
- テスト:
  - `test_T1_...`: 月末Excel の捏造 rationale を `topic=high_value_quote` に偽装 →
    content が month-end を示すため高額に credit されず、捏造が検出され FAIL
  - `test_T2_...`: high-value credit-risk rationale を `topic=month_end_excel` に
    誤割当 → content が high-value を示すため month_end に credit されず、high_value
    の rationale が不足して FAIL
  - `test_T3_...`: content と reported topic が一致 → PASS

## 9. known-rationale precision

- high_value の正しい理由は credit risk。
- `rationale_correct` は「期待値（credit）を捕捉」かつ「unsupported additional
  rationale が無い」の両方を要求。
- `is_unsupported_rationale` が known-rationale topic で
  `asserts_rationale_or_value(content) and asserts_extra_rationale(content)` のとき
  over-claim を検出。`EXTRA_RATIONALE_SIGNALS` は少数・bounded の追加理由集合
  （tax/audit/reconciliation/... と日本語対応）で、unbounded な禁止語辞書ではない。
- テスト `test_K1_...`（credit のみ → PASS）、`test_K2_...[credit risk and tax
  reporting]` / `[credit risk and audit reconciliation]`（extra あり → FAIL）。

## 10. epistemic status 最終定義

- `FACT`: source（stakeholder）が確定的情報として述べた claim
- `BELIEF`: source が推測・意見として述べた claim
- `UNKNOWN`: 未確認・不明（uncertainty として記録）
- `EXCEPTION`: 例外プロセスそのもの（epistemic claim ではない）
- objective correctness: evaluator が benchmark ground truth と比較して判断

`data_model.py` の `EpistemicStatus` / `TopicEvaluation` docstring と policy /
README をこの定義に統一した。

## 11. split 構成

- `base`（不変）: 全 6 task（EN3 + JA3）
- `base_en`: EN 3 task
- `base_ja`: JA 3 task

`--task-split base_en` / `--task-split base_ja` で EN-only / JA-only を容易に実行
でき、EN+JA comparison も可能。テスト `test_en_ja_only_splits` で保証。

## 12. falsification / regression / lint 結果

- business_interview 全テスト: **56 passed**（既存 41 + 新規 hardening 15）
- `make test`（core）: **286 passed, 1 xfailed**
- `make check-all`（ruff lint + format）: **clean**

## 13. reward / diagnostics 分離の維持

- 既存 scalar reward（`ENV_ASSERTION` 積）は不変。
- `protocol_pass`（finish_interview を呼んだか）と `interview_quality_pass`
  （discovery + epistemic 品質）の分離を維持。
- `test_multi_exception_F_correct_but_protocol_failure` が quality PASS /
  protocol FAIL を確認。
- 「reward が high_value_quote missed で FAIL なのに interview_quality_pass=True」
  の不整合は、JA canonicalization により解消（見逃し topic が required に残り
  quality も False）。
- generic core の既存 hook（`Environment.get_eval_diagnostics(task)` →
  `RewardInfo.info["diagnostics"]`）はそのまま。

## 14. 残る evaluator 弱点

- **content-signal fallback の限界**: explicit `topic` が無い finding は
  `content_signals`（少数・固定）で判定するため、agent が同一の業務要素をまったく
  別の表現で記録すると fallback で結び付けられない。ただし explicit topic を使えば
  回避可能で、これは許容される（同義語辞書を無制限にしない方針）。
- **over-claim 検出の語彙依存**: known-rationale の extra rationale は
  `EXTRA_RATIONALE_SIGNALS` の有限集合で検出するため、その集合に無い理由語
  （例: "for liquidity purposes"）は検出されない。bounded を意図した最小設計。
- **rationale が record_exception 内に埋め込まれた場合**: 正しいケースとして
  credit value を捕捉するが、UNKNOWN/NONE 判定と over-claim は content に依存。
- **言語カバレッジ**: バイリンガル（EN/JA）の固定信号のみ。他言語は追加が必要。

## 15. multi-stakeholder 前に追加修正が必要か

必要と考えるもの（今回はスコープ外）:
- `subject`（誰が何を言ったか）を使った評価（現状はデータモデル上のみ）。
- stakeholder 間の belief 差・矛盾検出（`conflicting_beliefs`）。
- 現状の topic と rationale は **single 真実**を仮定。multi-stakeholder では
  subject 単位の belief aggregation が必要。
- scenario を宣言的に増やす場合、`SCENARIO_TOPICS` / `TopicSpec` の追加が必要。

今回の hardening は単一 stakeholder benchmark を leak-free にし、evaluator の
false positive を修正する範囲で完了。multi-stakeholder 実装は今回の対象外。
