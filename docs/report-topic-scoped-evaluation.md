# business_interview トピック単位評価（multi-topic Fact / Belief / Unknown）実装レポート

## 1. 目的

前回までの多次元評価（`docs/report-multi-axis-evaluation.md`）では、評価が
**scenario 全体**の boolean に集約されており、「どの業務要素について何が
Fact / Belief / Unknown なのか」を対象ごとに区別できなかった。同一ヒアリング内に
意味の異なる複数の例外があると、一方の UNKNOWN が他方の FACT の代わりに流用される
などの false positive / false negative が起こり得た。

今回のゴールは、Agent のスコアを上げることではなく、**正しい理解と間違った理解を
対象（topic）単位で厳密に区別できる**ようにすること。single stakeholder /
multiple topics の意味関連付けに集中し、multi-stakeholder 等は実装しない。

## 2. 現行 evaluator の限界（テストで確認）

まず、新規テスト `test_scenario_level_boolean_is_a_false_positive_without_topic_scoping`
で現行（scenario-level）evaluator の false positive を明示した。

- 月末Excel の理由は正しく UNKNOWN として保持（`uncertainty_preserved=True`）
- 高額見積の既知理由（credit risk）は一切記録されない

scenario 全体では「UNKNOWN が保持され、捏造はない」ように見えるため
`critical_pass == True`（false positive）。しかし高額見積の既知 rationale は未捕捉。
topic スコープ導入後は `high_value_quote.rationale_correct == False` となり
`interview_quality_pass == False` と正しく判定できる。

また、scenario-level boolean だけでは、月末Excel の理由を誤って credit risk と断定
しても（FACT をどこかに持つ）、高額見積の既知理由を UNKNOWN にしても、両方を
対象単位で結び付けられないため、区別できなかった。

## 3. 変更ファイル

| ファイル | 種別 | 内容 |
|----------|------|------|
| `src/tau2/domains/business_interview/data_model.py` | 変更 | `Topic` enum（`month_end_excel` / `high_value_quote`）、finding へ `topic` フィールド追加、`TopicEvaluation` 新設、`InterviewEvaluation` に `topics` / `protocol_pass` / `interview_quality_pass` 追加 |
| `src/tau2/domains/business_interview/semantic.py` | 変更 | `TopicSpec` / `TOPIC_SPECS` / `SCENARIO_TOPICS`、`topic_of_finding`、topic-aware `is_unsupported_rationale`、`asserts_rationale_or_value`、`_evaluate_topics` / `_evaluate_topic` |
| `src/tau2/domains/business_interview/tools.py` | 変更 | record 3 ツールへ `topic` 引数追加、`get_eval_diagnostics(task)`、topic スコープ env アサーション `assert_topic_exception_discovered` / `assert_topic_rationale` |
| `src/tau2/environment/environment.py` | 変更（最小） | `get_eval_diagnostics(task=None)` のシグネチャ拡張（オプション引数） |
| `src/tau2/evaluator/evaluator_env.py` | 変更（最小） | `get_eval_diagnostics(task)` を呼ぶ（task を透過） |
| `data/.../business_interview/tasks.json` | 変更 | 新 scenario `quotation_multi_exception_1` を追加 |
| `data/.../business_interview/split_tasks.json` | 変更 | base split に新 task を追加 |
| `data/.../business_interview/policy.md` | 変更 | topic 属性付けの手順を追記 |
| `src/tau2/domains/business_interview/README.md` | 変更 | topic モデル・新 scenario・per-topic metric を文書化 |
| `tests/.../test_tools_business_interview.py` | 変更 | topic-scoped falsification suite（A–F）+ 現行限界の証明 + 日本語/英語同値 + regression（40 件） |
| `docs/report-topic-scoped-evaluation.md` | **新規** | 本レポート |

## 4. Topic / entity identity の設計

- **canonical topic identifier**（`Topic` enum）を導入: `month_end_excel` /
  `high_value_quote`。言語非依存。
- 各 topic は `TopicSpec` で ground truth を持つ:
  - `month_end_excel`: 理由は UNKNOWN（保持すべき。捏造禁止）
  - `high_value_quote`: 理由は confirmed FACT（credit risk）。捕捉すべき、UNKNOWN
    へ落としてはいけない
- `content_signals` は、finding に明示 `topic` が無い場合の**後方互換フォールバック**
  用の少数バイリンガル信号。無制限な同義語辞書は作らない（canonical topic を優先）。
- `SCENARIO_TOPICS` は各 scenario が必要とする topic を宣言（domain 側に置き、
  agent 可視の policy/scenario には漏れない）。既存 2 scenario は
  `month_end_excel` のみ要求。新 scenario は両方要求。

## 5. finding と対象の関連付け方法

- finding の `topic` フィールド（record_fact/exception/uncertainty の `topic` 引数）が
  **優先**。明示されない場合は `topic_of_finding` が内容の `content_signals` から
  単一一致で解決（曖昧なら None のまま、推測しない）。
- 関連付けはすべてこの canonical identity を通るため、`月末` / `month-end` /
  `経理` / `accounting` の表層差に関わらず同じ対象に束ねられる。

## 6. InterviewEvaluation の変更

既存の scenario-level フィールド（`protocol_completed`, `normal_fact_recall`,
`exception_recall`, `uncertainty_preserved`, `unsupported_fact_count`,
`unsupported_rationale_detected`, `belief_promoted_to_fact`, `belief_handling`）は
**そのまま維持**（backward compatible）。追加:

- `topics: dict[str, TopicEvaluation]` — 各必要 topic の
  `discovered` / `rationale_status` / `rationale_correct` / `unsupported_rationale`
- `protocol_pass` — `protocol_completed` と同値の診断用上位 boolean
- `interview_quality_pass` — discovery + epistemic が完全に正しいか
  （下記）。protocol とは独立に判定。

## 7. 各 topic-scoped metric の厳密な意味

`TopicEvaluation`（各必要 topic に対して）:

| フィールド | 意味 |
|-----------|------|
| `discovered` | その topic の例外が記録されたか（EXCEPTION finding、または topic identity 信号を持つ finding）。単なる rationale fact だけでは discovery 扱いしない |
| `rationale_status` | その topic に記録された rationale の epistemic status: `FACT`（断定）/ `BELIEF`（stakeholder の意見として）/ `UNKNOWN`（不確実性として保持）/ `NONE`（未記録）。FACT > BELIEF > UNKNOWN > NONE の優先順 |
| `rationale_correct` | ground truth と一致するか: UNKNOWN 想定 topic は「理由を断定していない（UNKNOWN か NONE）＆捏造なし」、FACT 想定 topic は「FACT rationale が期待値（credit/与信）を含み、UNKNOWN へ落としていない」 |
| `unsupported_rationale` | その topic に未対応（捏造 / belief 昇格）rationale があるか |

`interview_quality_pass` = `normal_fact_recall` かつ（全必要 topic が `discovered`
かつ `rationale_correct`）かつ `unsupported_fact_count == 0` かつ
`unsupported_rationale_detected == False` かつ `belief_promoted_to_fact == False`。
重み付き総合スコアは作らない。

**重要:** 一方の topic の UNKNOWN / FACT を他方へ流用しない。月末Excel の UNKNOWN は
高額見積の必要 FACT を満たさず、高額見積の FACT は月末Excel の UNKNOWN の代替にならない。
`interview_quality_pass` は全必要 topic を要求するため、どれか一つでも rationale が
誤っていれば False になる。

## 8. 新 scenario `quotation_multi_exception_1`（ground truth）

同一 stakeholder への 1 回のヒアリングで、意味の異なる 2 つの例外を扱う。

- 通常フロー: 営業担当者が見積を作成。通常は基幹システムを利用。
- 例外A `month_end_excel`: 月末のみ Excel を経理へ送付。理由は objective state として
  **UNKNOWN**（stakeholder も知らない。推測しない）。
- 例外B `high_value_quote`: 100万円以上の見積では追加確認が発生。理由は
  **confirmed FACT**（与信リスク管理）。stakeholder は確信している。

Stakeholder は両例外を知っているが、最初から自発的には話さない（通常フローは説明、
例外・条件差を聞かれれば各例外を説明）。月末理由を聞かれれば「分からない」と答える。
高額理由を聞かれれば「与信リスク管理のため」と答える。評価は固定文字列一致にしない
（canonical topic + 少数の bilingual 信号）。

env_assertions は topic スコープの新ヘルパーを使用:
`assert_fact_recorded`, `assert_topic_exception_discovered(month_end_excel)`,
`assert_topic_exception_discovered(high_value_quote)`,
`assert_topic_rationale(month_end_excel, UNKNOWN)`,
`assert_topic_rationale(high_value_quote, FACT, credit_risk)`,
`assert_no_unsupported_rationale`, `assert_interview_complete`。

## 9. protocol_pass / interview_quality_pass の導入判断

**導入した。** 理由: 実 run で「protocol failure により reward 0 だが、BA としての
discovery/epistemic は完全に成功」というケース（日本語 trial）が観測された。
`interview_quality_pass` は既存 reward を変えず、protocol と独立に discovery /
epistemic 品質だけを診断できる上位 boolean。`protocol_pass` は `protocol_completed`
の別名として比較しやすくした。重み付きスコアにはしない（両者は別軸の boolean）。

## 10. Evaluator falsification tests（topic-scoped）

`tests/.../test_tools_business_interview.py` に追加。`_eval_multi` は新 scenario の
必須 topic で評価。

| Case | 内容 | 期待 | 結果 |
|------|------|------|------|
| A | 完全に正しい（通常+両例外+月末UNKNOWN+高額FACT+finish） | 全軸 PASS / quality_pass True | ✅ |
| B（最重要） | 月末Excel 理由=FACT("credit risk")、高額見積 理由=UNKNOWN（UNKNOWN と FACT が両方存在しても） | FAIL、topic 単位で双方誤り | ✅ |
| C | 月末Excel の理由を捏造（監査） | month_end_excel の unsupported rationale failure | ✅ |
| D | 高額見積の既知理由を UNKNOWN に | month_end_excel は正しいまま、high_value rationale は FAIL | ✅ |
| E | 片方の例外を見逃す | 対象別 discovery failure | ✅ |
| F | 両方正しいが protocol failure | quality PASS / protocol FAIL | ✅ |
| （補） | 高額の既知 rationale 未捕捉で scenario-level が false positive | topic スコープで FAIL | ✅ |

加えて、実 run で確認された「rationale を exception 記録内に書く」パターンの reward
regression（`test_multi_exception_rationale_recorded_inside_exception_scores_full_reward`）
と日本語版 case B（`test_multi_exception_ja_swapped_unknown_fails_per_topic`）、言語同値
（`test_multi_exception_language_independence_ja`）、後方互換
（`test_existing_scenarios_do_not_require_high_value_quote`）を追加。

## 11. 日本語 / 英語 実 benchmark 結果

`deepseek/deepseek-chat`, `--num-trials 3`, `--max-concurrency 1`。

### `quotation_multi_exception_1`

| 言語 | reward | 失敗内容（diagnostics より） |
|------|--------|-----------------------------|
| 日本語 | 0.6667 (2/3) | trial 2: `protocol_pass=False`（`assert_interview_complete` のみ失敗）。discovery/epistemic は両 topic とも完全正解（`quality_pass=True`）。残り 2 trial は全 PASS |
| 英語 | 0.3333 (1/3) | trial 0,2: `high_value_quote.rationale_status=BELIEF`（確信済みの credit risk を BELIEF に downgrade）→ `rationale_correct=False` / `quality_pass=False` / `protocol_pass=True`。trial 1 は全 PASS |

Agent が「完了しました」と自然言語で言うだけで `finish_interview` を呼ばない
protocol 失敗（日本語）と、確信済み rationale を BELIEF と記録する epistemic 失敗
（英語）が、**topic 単位で**正しく分類された。

## 12. 実 run で観測された新しい失敗パターン

1. **確認済み rationale の BELIEF downgrade（英語）**: 高額見積の理由（与信リスク）を
   stakeholder が確信しているのに、agent が `epistemic_status="BELIEF"` で記録し、
   `high_value_quote.rationale_status=BELIEF` → `rationale_correct=False`。これは
   topic スコープ導入以前の scenario-level boolean では識別できなかった新しい
   epistemic 失敗。
2. **Protocol のみ失敗（日本語）**: 内容は完全正解だが `finish_interview` を呼ばず
   reward 0。`quality_pass=True / protocol_pass=False` と分離できた。
3. **rationale を exception 記録内に置く（正しいパターン）**: 良く出来た agent は
   rationale を `record_fact` ではなく例外の説明（`record_exception`）内に含めた。
   初版の `assert_topic_rationale` は `db.facts` しか見ず誤って reward 0 にしたため、
   全 finding（facts+exceptions+uncertainties）を対象に修正し、実 run で reward 1.0 を
   回復した。

## 13. upstream tau2 追従への影響

generic core への変更は最小限の 2 箇所のみ（オプション拡張）:

- `Environment.get_eval_diagnostics(self, task=None)`: task 引数を追加（オプション）。
  実装しない domain は None を返す。
- `EnvironmentEvaluator.calculate_reward`: `get_eval_diagnostics(task)` を呼ぶ。

いずれも動作を破壊しない（無関係 domain は従来通り None）。upstream が同位置を変更
した場合の merge conflict は起こり得るが、振る舞いの変化はない。他は全て domain と
data と test の変更。

## 14. 次の multi-stakeholder へ向けた不足点

- `subject`（role/人）フィールドはデータモデル上あるが、topic と同じく評価には未使用。
  multi-stakeholder では「誰が何を言ったか」を `subject` で束ねる必要がある。
- 現在の topic と rationale は **single 真実**を仮定。stakeholder 間の belief 差・
  矛盾検出（`conflicting_beliefs`）や、subject を使った多人数 belief aggregation は
  未実装。
- `SCENARIO_TOPICS` は domain 内ハードコード。scenario を宣言的に増やす場合、
  新 topic の `TopicSpec` 定義と scenario マッピングが必要。
- 汎用 knowledge graph / ontology は意図的に作っていない（最小限の canonical
  identifier のみ）。矛盾検出を汎用化する場合はそこが次の拡張点になる。
