# business_interview 多次元評価（Fact / Belief / Unknown）実装レポート

## 1. 目的

`business_interview` の評価を**単一の reward ではなく独立した診断軸**で計測できるように
する。具体的には、実ベンチマークで観測されている以下の失敗を、それぞれ別々に識別できる
ようにする。

1. `finish_interview` を呼ばず、自然言語で終了宣言するだけになる（protocol 違反）
2. 月末 Excel などの重要例外を発見できない（discovery 失敗）
3. ヒアリングで確認していない理由・承認フロー等を捏造する（epistemic 失敗）
4. stakeholder の belief / uncertainty を fact と誤認する（epistemic 失敗）

**重要:** スコアを上げることを目的としない。失敗した Agent を正しく「どこで失敗したか」
分類して計測できることを優先した。

## 2. 変更ファイル

| ファイル | 種別 | 内容 |
|----------|------|------|
| `src/tau2/domains/business_interview/semantic.py` | **新規** | 言語非依存の semantic evaluator。canonical 概念分類（例外/根拠/不確実性）と多軸 `InterviewEvaluation` |
| `src/tau2/domains/business_interview/data_model.py` | 変更 | `EpistemicStatus` enum、finding への `epistemic_status`/`subject` フィールド、`InterviewEvaluation` |
| `src/tau2/domains/business_interview/tools.py` | 変更 | `record_fact(epistemic_status=...)` 追加、`assert_no_unsupported_rationale` の belief 対応、`get_eval_diagnostics()` 追加 |
| `src/tau2/domains/business_interview/environment.py` | 変更なし | （変更不要） |
| `src/tau2/domains/business_interview/README.md` | 変更 | 多軸評価・新 scenario のドキュメント化 |
| `data/.../business_interview/tasks.json` | 変更 | 新 scenario `quotation_belief_uncertainty_1` を追加 |
| `data/.../business_interview/split_tasks.json` | 変更 | base split に新 task を追加 |
| `data/.../business_interview/policy.md` | 変更 | belief を `epistemic_status="BELIEF"` で記録する旨を追記 |
| `src/tau2/environment/environment.py` | 変更 | 汎用フック `get_eval_diagnostics()`（デフォルト None） |
| `src/tau2/evaluator/evaluator_env.py` | 変更 | diagnostics を `reward_info.info["diagnostics"]` へ露出（半二重のみ） |
| `tests/.../test_tools_business_interview.py` | 変更 | 新 task / falsification / 診断のテストを追加（28 件） |
| `docs/report-multi-axis-evaluation.md` | **新規** | 本レポート |

## 3. 新しい評価データモデル

```python
class EpistemicStatus(str, Enum):
    FACT = "FACT"       # 確定した事実（stakeholder が確定的に述べた）
    BELIEF = "BELIEF"   # stakeholder の意見・推測（確定ではない）
    UNKNOWN = "UNKNOWN" # 不明・未確認
    EXCEPTION = "EXCEPTION"  # 例外プロセス

class InterviewFact(BaseModel):
    content: str
    epistemic_status: EpistemicStatus = FACT   # record_fact で指定可
    subject: Optional[str] = None              # 将来の multi-stakeholder 用（今回は不使用）

class InterviewEvaluation(BaseModel):
    protocol_completed: bool
    normal_fact_recall: bool
    exception_recall: bool
    uncertainty_preserved: bool
    unsupported_fact_count: int
    unsupported_rationale_detected: bool
    belief_promoted_to_fact: bool
    # @property belief_handling = not belief_promoted_to_fact
```

`subject` フィールドは将来の `conflicting_beliefs` / multi-stakeholder への拡張に備えて
データモデル上だけ追加しており、今回は判定に使用しない（先回り実装はしない）。

## 4. 各 metric の厳密な意味

| Metric | 型 | 意味 | 軸 |
|--------|-----|------|-----|
| `protocol_completed` | bool | `finish_interview` を実際に呼んだか（「終了します」と述べただけでは False） | Protocol |
| `normal_fact_recall` | bool | 通常フローの fact（非例外の fact）が記録されたか | Discovery |
| `exception_recall` | bool | 月末例外（month/excel/accounting）が記録されたか | Discovery |
| `uncertainty_preserved` | bool | 例外の理由が uncertainty として記録され、UNKNOWN のまま残ったか | Epistemic |
| `unsupported_fact_count` | int | FACT として記録された finding のうち、未確認の根拠を断定した数 | Epistemic |
| `unsupported_rationale_detected` | bool | BELIEF として明示されていない finding で根拠を断定したものがあるか（捏造検出） | Epistemic |
| `belief_promoted_to_fact` | bool | stakeholder の belief/推測を FACT に昇格させたか | Epistemic |
| `belief_handling` | bool | `not belief_promoted_to_fact`（belief を正しく扱ったか） | Epistemic |

**`is_unsupported_rationale` の規則（言語非依存・共通）:**
finding が BELIEF として記録されていない かつ 例外に関する かつ 根拠を断定している
（`because`/`due to`/`するため`/`監査のため` 等） かつ 理由を「不明」と表現していない場合に
「未確認根拠の断定」と判定する。BELIEF 明示、または「理由は不明」を伴う表現は免除される
（belief を fact に昇格させない限り正しい処理）。

## 5. reward と追加 metric の関係

- **既存 reward は不変。** `reward_basis: ["ENV_ASSERTION"]` の 5 アサーション
  （`assert_fact_recorded` / `assert_exception_recorded` / `assert_uncertainty_recorded` /
  `assert_no_unsupported_rationale` / `assert_interview_complete`）の積のまま。
- **多次元 metric は追加の診断情報。** `SemanticEvaluator` が同じ DB 状態から計算し、
  `reward_info.info["diagnostics"]` として保存される（`SimulationRun` の JSON に永続化）。
  reward には影響しない。
- 新総合スコアの重みは発明しない。既存 reward を維持し、多次元 metric を追加で公開する。

**core への変更は最小限**（2 箇所のみ）:
- `Environment.get_eval_diagnostics()`：ドメインが実装しなければ None を返す汎用フック
- `EnvironmentEvaluator.calculate_reward`：`info["diagnostics"]` に代入（半二重のみ）

## 6. Falsification tests（Evaluator が悪い回答を落とすことの検証）

`tests/test_domains/test_business_interview/test_tools_business_interview.py` に追加。

| Case | 内容 | 期待される失敗軸 | テスト結果 |
|------|------|------------------|-----------|
| A | 通常フロー + 月末例外 + 理由不明 + finish 呼ぶ | 全軸 PASS / reward 1.0 | ✅ |
| B | 通常フローだけ記録、月末例外を記録しない | `exception_recall=False`（discovery） | ✅ |
| C | 「監査のため」等の未確認根拠を FACT で断定 | `unsupported_rationale_detected=True` / `unsupported_fact_count=1`（epistemic） | ✅ |
| D | stakeholder の「経理都合だと思う」を「経理都合である」と FACT 化 | `belief_promoted_to_fact=True` / `belief_handling=False`（epistemic） | ✅ |
| E | findings は記録したが finish を呼ばない | `protocol_completed=False`（protocol のみ） | ✅ |

各ケースが**別々の軸**で期待通り失敗することを確認。D の正の対照（同じ内容を
`epistemic_status="BELIEF"` で記録すれば全軸 PASS）もテスト。

## 7. 日本語 / 英語の実 benchmark 結果

`deepseek/deepseek-chat`, `--num-trials 3`, `--max-concurrency 1`。

### 新 scenario `quotation_belief_uncertainty_1`

| 言語 | reward | 失敗内容（diagnostics から） |
|------|--------|-------------------------------|
| 日本語 | 0.6667 (2/3) | trial 0: `protocol_completed=False` のみ（discovery/epistemic は全て PASS） |
| 英語 | 1.0000 (3/3) | 全 PASS |

### 既存 scenario `quotation_process_interview_1`

| 言語 | reward | 失敗内容 |
|------|--------|---------|
| 日本語 | 0.0 (0/2) | 両 trial とも `protocol_completed=False` のみ |
| 英語 | 1.0 (2/2) | 全 PASS |

## 8. 実際に観測された Agent の失敗パターン

1. **Protocol 違反（最頻出・日本語）**: deepseek-chat が `finish_interview` を呼ばず
   「それでは、インタビューを終了させていただきます」等と自然言語で終了宣言するだけになる。
   discovery/epistemic は全 PASS なのに reward が 0 になり、診断上は
   `protocol_completed=False` だけが立つ。多次元化により「ヒアリング品質は良いのに終了手順だけ
   失敗している」ことが明確に分離できた。
2. **例外・理由の取りこぼし**: 英語試行で月末例外の理由を uncertainty として記録しない
   （`uncertainty_preserved=False`）ことがある。
3. **捏造の検出**: 例外について根拠（`due to` / `probably`）を断定すると
   `unsupported_rationale_detected=True` で捕捉される。ただし、理由を「不明」として保持
   （belief を fact 化しない）場合は正しく免除されるよう調整した。

## 9. 日本語と英語の評価ロジック共通化

`semantic.py` に canonical 概念分類を集約し、既存の env assertion と新 metric の双方が
同じ関数（`is_exception_content` / `asserts_rationale` / `expresses_uncertainty` /
`is_unsupported_rationale`）を使う。表層言語（英語/日本語）に関わらず同じ意味なら同じ判定。
既存の `JAPANESE_KEYWORD_SYNONYMS` は後方互換のマッチ基本プリミティブとして残し、これを
無制限に拡張する設計にはしていない（`把握していない` 等、ごく少数の高頻度 epistemic 表現のみ追加）。

## 10. 次に追加すべき scenario（1〜3 個）

1. **`quotation_two_stakeholders`**: 2 人の stakeholder（営業・経理）が月末 Excel の目的に
   ついて異なる belief を持つ。`subject` フィールドと矛盾検出の土台を試す。
2. **`quotation_multi_exception`**: 月末 Excel に加え、大口顧客向けの別例外があり、その
   例外の理由だけは stakeholder が把握している（例外ごとに unknown/fact が混在）。
3. **`quotation_agent_invents_approval`**: stakeholder が承認フローに言及しないのに、
   agent が「上司承認が必要」と捏造して記録するケースを特化させ、捏造検出を別軸で強化。

## 11. upstream tau2 追従を難しくする変更

- `src/tau2/environment/environment.py` に `get_eval_diagnostics()` を追加、および
  `src/tau2/evaluator/evaluator_env.py` の `EnvironmentEvaluator.calculate_reward` で
  `info["diagnostics"]` を代入。いずれも**オプションの拡張フック**（ドメインが実装しなければ
  None / 影響なし）なので、upstream が同位置を変更した場合は merge conflict の可能性はあるが、
  振る舞いの破壊はない。これ以外は domain と data の変更のみで generic core を壊さない。

## 12. Verification

- `uv run pytest tests/test_domains/test_business_interview/` → **28 passed**
- `uv run make check-all`（ruff lint + format）→ クリーン
- コアテスト（LLM 依存の既知フレーキーを除く）→ 253 passed
- 日本語 / 英語の実 benchmark → 上記 7 のとおり
