# business_interview source-aware multi-claim epistemic model 実装レポート

## 1. 目的

`TopicEvaluation` はこれまで概念的に `rationale_status = FACT | BELIEF | UNKNOWN | NONE`
という**1 topic = 1 rationale_status** の単一状態だった。しかし
`quotation_belief_uncertainty_1` では同一トピック上に

- stakeholder の belief（Accounting 側の事情かもしれない）
- objective state（本当の理由は UNKNOWN）

が同時に成立する。単一状態に潰すとどちらかが失われる（BELIEF があると objective
UNKNOWN が失敗扱いになる等）。

本変更は評価を **source-aware multi-claim model** に拡張する。multi-stakeholder 自体は
実装せず、既存 single-stakeholder の belief/uncertainty scenario で設計を falsify する。

## 2. 変更ファイル

| ファイル | 種別 | 内容 |
|----------|------|------|
| `src/tau2/domains/business_interview/data_model.py` | 変更 | `Source` / `RationaleValue` enum、`InterviewFact.source`/`.value`、`InterviewException.source`、`InterviewUncertainty.source`、`ObjectiveRationale` / `ClaimEvaluation`、`TopicEvaluation` 再構成（objective_rationale + claims + required_claims_complete + legacy 派生フィールド） |
| `src/tau2/domains/business_interview/semantic.py` | 変更 | `ExpectedClaim` / `SCENARIO_CLAIMS`、`TopicSpec` の構造化 matcher、`VALUE_TO_TOPIC` / `detect_claim_values` / `normalize_source` / `_source_matches`、`_evaluate_claims`、objective/claim 分離評価、JA hedged-belief マーカー |
| `src/tau2/domains/business_interview/tools.py` | 変更 | `record_fact(source, value)`、`record_exception(source)`、`record_uncertainty(source)`、topic_of_finding への value 透過 |
| `data/.../business_interview/policy.md` | 変更 | `source` / `value` を任意の記録フィールドとして説明（hidden ID は教えない） |
| `src/tau2/domains/business_interview/README.md` | 変更 | source-aware multi-claim model を文書化、per-topic metrics 更新 |
| `tests/.../test_tools_business_interview.py` | 変更 | source-aware falsification suite（A–H + DoD 補助）を追加 |
| `docs/report-source-aware-model.md` | **新規** | 本レポート |

generic core（`Environment.get_eval_diagnostics` / `EnvironmentEvaluator`）への変更は**なし**。

## 3. 新しい claim / source data model

```python
class Source(str, Enum):        # 誰が claim を述べたか
    SALES = "sales"; ACCOUNTING = "accounting"; UNKNOWN = "unknown"

class RationaleValue(str, Enum):  # rationale claim の最小 canonical value
    ACCOUNTING_NEED = "accounting_need"; CREDIT_RISK = "credit_risk"; UNKNOWN = "unknown"

class InterviewFact:   # 追加
    source: Optional[Source]   # who said it (source provenance)
    value: Optional[RationaleValue]  # 最小 canonical value
```

- `source` は「誰が言ったか」。既存 `subject`（何についての finding か）と**概念的に分離**。
  将来 `source=sales, about=accounting` のような claim も自然に載る。汎用 `about` は未導入。
- `value` は自由文だけでなく比較可能な最小 canonical value（`accounting_need` /
  `credit_risk`）。汎用 proposition engine は作らない。

## 4. objective state と source claim の分離

不変条件 **source claim ≠ objective truth**。

- `objective_rationale.status` は scenario ground truth（`TopicSpec.expected_rationale_status`）
  から決まり、Agent が設定するものではない。Agent は UNKNOWN を uncertainty として「正しく表現
  したか」だけを `objective_rationale.correct` で判定。
- `claims` は source（stakeholder）の主張。`FACT` は「source が確定的に述べた」の意味のまま。
  FACT だから objective に真、とはしない。

## 5. TopicEvaluation 変更

```
topic:
  discovered
  objective_rationale: { status, correct }
  claims: [ { source, epistemic_status, value, source_correct, status_correct,
              value_correct, correct, promoted_to_fact } ]
  required_claims_complete
  unsupported_rationale
  rationale_status      # LEGACY derived（単一値 collapse）
  rationale_correct     # LEGACY derived
```

- `rationale_status` / `rationale_correct` は**後方互換のためだけ**に残す派生フィールド。
  新しい意味モデルの source of truth にはしない（docstring に明記）。
- `interview_quality_pass` は legacy 単一 status ではなく
  `discovered AND objective_rationale.correct AND required_claims_complete AND not unsupported`
  を全 required topic に要求。

## 6. topic resolution 方式

- 明示 `topic` が最優先。ただし **content が一意に別トピックを示す場合は信頼しない**
  （misattribution 対策）。`value` 単独では明示 topic を override しない。
- 明示 topic なしの場合: 構造化 content rule ＋ value ベース推論。
  - `month_end_excel`: `excel AND (month_end OR accounting)` — accounting 単独では判定しない。
  - `high_value_quote`: `amount_threshold OR explicit high-value expression`。
  - rationale claim が canonical value を名指ししたら `VALUE_TO_TOPIC`
    （`accounting_need→month_end_excel`, `credit_risk→high_value_quote`）で帰属。
  → Agent が hidden canonical topic ID を知る必要性を低減（自然な finding を記録すれば
    evaluator が解決）。

## 7. A–H falsification 結果

belief scenario（`quotation_belief_uncertainty_1`）を主要 regression/falsification に使用。

| Case | 内容 | 期待 | 結果 |
|------|------|------|------|
| A | 正解（sales BELIEF accounting_need + objective UNKNOWN） | quality PASS、belief captured、attribution correct、promotion なし | ✅ |
| B | belief だけ記録、UNKNOWN を記録しない | objective uncertainty FAIL | ✅ |
| C | UNKNOWN だけ、belief を落とす | required claim coverage FAIL | ✅ |
| D | sales belief を FACT に昇格 | belief promotion FAIL（per-claim + scenario） | ✅ |
| E | belief value を誤る（credit_risk） | claim correctness FAIL | ✅ |
| F | source を誤る（accounting） | source attribution FAIL | ✅ |
| G | BELIEF + UNKNOWN 両方存在 | 単一 status で失わず両方正しく評価（objective UNKNOWN も claim も correct） | ✅ |
| H | semantic 正しいが finish_interview なし | quality PASS / protocol FAIL | ✅ |

追加（DoD）:
- Agent が topic/source/value を省略しても content/value 推論で正しく帰属。
- `accounting` 単独では month_end_excel と判定しない（構造化推論）。
- `source` と `subject` が分離。

## 8. existing regression

- business_interview 全テスト **68 passed**（既存 56 + 新規 12）。
- multi_exception falsification A–F、JA canonicalization、hardening（L1/L2/J1/U1/U2/N1/T1/T2/T3/K1/K2）、
  scalar reward、protocol_pass / interview_quality_pass を維持。
- `make test`（core）**298 passed, 1 xpassed**。
- `make check-all`（ruff lint + format）**clean**。

## 9. EN/JA real run（deepseek/deepseek-chat, concurrency=1, 各3 trials）

### belief scenario `quotation_belief_uncertainty_1`

| 言語 | reward | diagnostics |
|------|--------|-------------|
| EN | 1.0 (3/3) | 全 trial: objective UNKNOWN preserved、BELIEF(accounting_need) claim 検出、quality True |
| JA | 0.333 (1/3) | 全 trial: objective UNKNOWN preserved、BELIEF(accounting_need) claim 検出、quality True。reward 0 の 2 trial は `assert_interview_complete` のみ失敗（protocol のみ）— quality/protocol 分離で正しく診断 |

JA で agent が belief を `record_uncertainty` 内に merged 記録する挙動にも対応し
（hedged belief「経理側の都合ではないかと思う」を claim として検出）、BELIEF + objective
UNKNOWN を同時に正しく診断。スコアの高さは DoD ではなく、失敗を diagnostics が正しく説明。

## 10. generic core 差分

**なし**。domain / data / test / docs のみの変更。

## 11. 残る弱点

- claim value の content 推論は `CLAIM_VALUE_SIGNALS`（accounting/credit）の有限集合に依存。
  集合外の理由語（例: "for liquidity"）は明示 `value` がないと検出されない。
- 「hedged belief が uncertainty 内に混ざる」判定は JA hedged マーカーの少数集合に依存。
- `_source_matches` は未指定 source を単一 stakeholder（sales）として正扱い。明示的に正しくない
  source のみ検出。multi-stakeholder では source 必須化が必要。
- 単一 topic の expected claim は最大1つ想定（`expected_claims[0]` を参照）。複数 claim 想定の
  完全汎用化は未対応。

## 12. 次に multi-stakeholder へ進めるか

**進める土台はできた**。次は:
1. `record_fact` の `source` を必須（または default 除去）にして、2人目の stakeholder の claim を
   `source` で区別。
2. `SCENARIO_CLAIMS` に複数 source の expected claim を追加（`expected_claims` リストを完全評価）。
3. `subject`（about）を評価に使う（`source=sales, about=accounting` の形）。
4. `conflicting_beliefs` / contradiction resolution はその上に載せる。

現状は「1人の source から複数の epistemic states を正しく保持する」まで実装済み。
