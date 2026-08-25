# Business Interview Simplification Audit

監査対象: commit `c81601b` の `src/tau2/domains/business_interview/` と、その
runtime caller、evaluator、tests、scripts、artifact、既存文書。

本書の LOC は、特記しない限り空行・コメントを含む `wc -l` 相当の raw LOC
である。分類は README の説明だけでなく、registry、runner、replay evaluator、
imports、call sites、tool execution、scenario construction を追跡して決めた。

## Executive Summary

### 結論

現在の `business_interview` は、benchmark の本質そのものが不必要に複雑なのでは
なく、**本質的な runtime/evaluation core と、過去の研究 experiment、毎回生成する
説明用 diagnostics、artifact compatibility が同じ production dependency graph に
載っているため、過度に複雑に見える状態**である。

- production package は **18 Python files / 14,301 LOC**。
- 専用 tests は **11 files / 9,118 LOC / 234 collected cases**（224 test functions）。
- 関連 scripts は **5 files / 3,442 LOC**。
- checked-in real-LLM artifacts は **158 files / 約 21 MiB**。
- 概念上の benchmark core は **9つの substantive module** で説明できる
  （物理的には 5 LOC の `utils.py` を加えた10 files、8,378 LOC）。
- 現在の標準 run は artifact capture と diagnostic imports を含むため、実際には
  **14 domain modules** に到達する。
- score-independent な diagnostic / experiment 実装は **6 files / 5,358 LOC
  (37.5%)**。そのうち 2,519 LOC は runtime evaluator に誤って結合され、
  2,839 LOC は完全に off-runtime である。

最大の accidental complexity は LOC の大きさ自体ではなく、次の3点である。

1. `evaluate()` が primary scoring、reference scoring、failure explanation、usage
   experiment、joint structural experiment、canonical diagnostics、DTO serialization
   を一度に担当する。
2. task の3つの評価 assertion と `get_eval_diagnostics()` が同じ `_evaluate()` を
   cache なしで呼び、同じ最終 graph に対して diagnostics/reference を通常 **4回**
   再計算する。
3. canonical SOURCE/SINK 導入前の `boundary_diagnostics.py` など、意思決定済みの
   experiment が production domain package に残っている。

最も大きな simplification は、**primary comparison を lean・pure・cacheable にし、
reference/usage/joint/failure attribution を明示的な offline/optional enrichment lane に
移すこと**である。Primary score、tool API、graph schema、forgetting behavior を変えずに
実施できる。

最も安全な最初の変更は、production caller が0の
`boundary_diagnostics.py` を script/test と一緒に archive または削除することである。
canonical contract に必要な utility は既に `graph.py` にあり、同 module から core に
移すべき utility はない。

### Phase 1 の効果見積り

「削除」と「移動」を区別する必要がある。

- **確実に削除可能な repository code**: boundary experiment 本体 1,787 LOC + CLI
  56 LOC + dedicated tests 248 LOC = **2,091 LOC**。小さな dead helper/compatibility を
  含めると約 **2.2K LOC**。
- **production package から隔離可能な量**: boundary/offline/run-metrics/joint/usage/
  grounding 合計 **5,358 LOC**。これらを experiment/scripts lane に移すと、domain
  production package の表面は 14,301 LOC から約 **8,943 LOC** へ縮む
  （repository total LOC は、移動した分については減らない）。
- diagnostic output 自体も不要と意思決定できるなら、Phase 1–2 を通じた実削除量は
  約 **3.7–5.4K LOC** まで広がる。

## Current Architecture

### 実コード上の runtime / evaluation flow

```text
registry.py
  ├─ register_domain(get_environment, "business_interview")
  ├─ register_tasks(get_tasks, ...)
  └─ register_user(StakeholderUserSimulator, "business_interview_user")
          │
runner/build.py
  ├─ get_environment() -> InterviewDB + SemanticLedger + InterviewTools
  └─ build_user() -> task と environment を simulator constructor に注入
          │
scenario.get_scenario(task.id)
  ├─ canonical TruthGraph (BusinessProcessGraph)
  ├─ StakeholderFilter
  └─ project_knowledge(Truth, Filter)
       ├─ forgetting sample
       ├─ safe serial shortcut contraction
       ├─ opaque local-id materialization
       └─ StakeholderKnowledgeGraph (3-state slots)
          │
StakeholderUserSimulator
  ├─ private knowledge rendering
  ├─ semantic response plan
  ├─ natural-language realization
  └─ validated private sidecar
          │
BusinessInterviewEnvironment.on_message()
  ├─ sidecar -> private SemanticLedger
  ├─ public utterance -> immutable Observation
  └─ Observation id を Agent-visible text に付与
          │
InterviewTools (21 Agent-visible tools)
  └─ AgentConcepts + AgentGraph (4-state slots) を構築
          │
finish_interview() -> interview_complete
          │
runner/simulation.py
  ├─ live run の Truth/Knowledge を evaluation_inputs artifact に保存
  └─ generic EnvEvaluator に trajectory を渡す
          │
evaluator_env.py
  ├─ fresh environment に trajectory/tool calls を replay
  ├─ task env assertions を順次実行
  │    ├─ assert_finish_interview
  │    ├─ assert_graph_reconstructed -> evaluate()
  │    ├─ assert_necessity_handled   -> evaluate()
  │    └─ assert_evidence_backed     -> evaluate()
  └─ get_eval_diagnostics()          -> evaluate()  # 4回目
          │
evaluation.evaluate()
  ├─ AgentGraph -> TruthGraph primary comparison
  ├─ evidence/reward fields
  ├─ explanation DTO
  ├─ usage alignment experiment
  ├─ joint structural experiment
  └─ StakeholderKnowledge -> Truth reference comparison
```

主要な call sites:

- Registration: `src/tau2/registry.py:295-300,371-378`
- User injection: `src/tau2/runner/build.py:145-207`
- Live execution/artifact capture: `src/tau2/runner/simulation.py:45-108`
- Replay/assertion/diagnostics: `src/tau2/evaluator/evaluator_env.py:95-169`
- Environment construction/ingestion: `environment.py:94-248`
- Scenario lookup: `scenario.py:566-631`
- Projection: `knowledge.py:629-958`
- Simulator: `user_simulator.py:447-878`
- Tool evaluation adapter: `tools.py:1172-1228`
- Primary evaluator: `evaluation.py:2355-2606`

### 重要な実装上の注意

1. `_SCENARIOS` は import 時に Truth と Knowledge を materialize する
   (`scenario.py:595-610`)。現在の authored scenarios は forgetting probability 0 なので
   deterministic だが、確率を有効化すると simulation seed ではなく
   `random.Random(None)` による import-time sample になる。`set_seed()` は Knowledge を
   再投影しない (`user_simulator.py:488-491`)。
2. `run_simulation()` が保存した `evaluation_inputs` は標準 EnvEvaluator の score input
   には使われない。EnvEvaluator は fresh environment を作り、`get_scenario()` から
   module-global scenario を再取得する。artifact は再現性のため重要だが、現在の
   online reward calculation の source of truth ではない。
3. `scenario.get_scenario()` は通常 locale で同じ mutable global object を返す。
   caller mutation が次 run に伝播し得る。
4. `usage_alignment` と `joint_structural_alignment` は score fields を変えないことが
   tests で確認されているが、import、latency、exception/control-flow の意味では
   production dependency である。

## Current Benchmark Core

この benchmark の core contract は次の6行で表せる。

1. canonical SOURCE/SINK を持つ `TruthGraph` を scenario からロードする。
2. Truth と stakeholder filter から、forgetting/contraction を経た
   `StakeholderKnowledge` を生成する。
3. stakeholder simulator はその Knowledge のみを使って Agent と対話する。
4. Agent は `InterviewTools` で `AgentGraph` を構築し interview を完了する。
5. `AgentGraph` を `TruthGraph` の business projection と比較する。
6. `StakeholderKnowledge -> Truth` は情報量を説明する reference-only metric であり、
   Agent の denominator/pass/ranking を変えない。

この core に直接必要な substantive modules は:

- `graph.py`
- `stakeholder.py`
- `knowledge.py`
- `scenario.py`
- `facts.py`（少なくとも sidecar model/catalog integrity 部分）
- `user_simulator.py`
- `tools.py`
- `evaluation.py`
- `environment.py`

`utils.py` は path constants だけの5 LOC support file である。
`artifact_provenance.py` は reproducible artifact の optional production lane であり、
会話・AgentGraph construction・primary comparison の本質ではない。

直接 core ではないもの:

- boundary A/B experiment
- usage identity experiment
- joint label-independent matching experiment
- offline failure attribution/artifact decoder
- smoke-run tool/refusal metrics
- semantic grounding experiment helpers
- bulk real-LLM artifacts と historical goal reports

## Quantitative Summary

### Package / tests / scripts

| 対象 | Files | Raw LOC / size | 備考 |
| --- | ---: | ---: | --- |
| production domain package | 18 | 14,301 LOC | `__init__.py` を含む |
| conceptual core | 10 | 8,378 LOC | 9 substantive + `utils.py` |
| optional artifact provenance | 1 | 564 LOC | standard runner では現在 always-on capture |
| score-independent diagnostics/experiments | 6 | 5,358 LOC | grounding/joint/usage/boundary/offline/run_metrics |
| strictly off-runtime source | 3 | 2,839 LOC | boundary/offline/run_metrics |
| dedicated tests | 11 | 9,118 LOC / 234 collected cases | 224 test functions、root-level 2 filesを含む |
| related scripts | 5 | 3,442 LOC | launcher、smoke、3 diagnostic/audit scripts |
| checked-in real-LLM artifacts | 158 | 21,464,443 bytes | runtime input ではない |

### Largest modules

| Module | LOC |
| --- | ---: |
| `evaluation.py` | 2,606 |
| `boundary_diagnostics.py` | 1,787 |
| `joint_structural_alignment.py` | 1,442 |
| `tools.py` | 1,228 |
| `graph.py` | 1,206 |
| `knowledge.py` | 958 |
| `usage_alignment.py` | 926 |
| `user_simulator.py` | 878 |
| `offline_diagnostics.py` | 837 |
| `scenario.py` | 631 |
| `artifact_provenance.py` | 564 |
| `facts.py` | 458 |

### Primary evaluator size

`evaluation.py` 2,606 LOC のうち、primary matching/scoring とその最小 DTO は概算
**850–950 LOC** である。残りの大半は diagnostic DTO、reference comparison、
canonical trace、failure explanation、usage/joint invocation、serialization glue である。

- DTOs: lines 112-452（約341 LOC）
- lexical/concept/node/slot matching: lines 576-1092（約517 LOC）
- evidence metrics: lines 1105-1182（約78 LOC）
- knowledge/reference evaluation: lines 1190-1760（約571 LOC）
- canonical/slot/concept diagnostic trace: lines 1763-2347（約585 LOC）
- mixed orchestration/return assembly: lines 2355-2606（約252 LOC）

## Module Inventory

分類:

- **PC**: Production critical
- **PO**: Production optional
- **D**: Diagnostic only（現在 runtime に誤結合されている場合を含む）
- **H**: Historical experiment
- **T**: Test/support/infrastructure only

`Deps` は直接 import する business_interview 内 module 数。

| Module | LOC | Class | Main responsibility | Deps | Production runtime import? | Evaluator import/use? | Diagnostics/scripts/tests only? | Public API? | Candidate | Notes |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `__init__.py` | 1 | T | package marker | 0 | 間接 | no | no logic | no | keep | 削減価値なし |
| `artifact_provenance.py` | 564 | PO | Truth/Knowledge input envelope、fingerprint、offline reload/recompute | 3 | `environment.get_evaluation_inputs()` から lazy import。標準 run は現在呼ぶ | standard evaluator は保存 envelope を読まない | scripts/tests からも利用 | yes (`__all__`) | split/move I/O | schema/fingerprint は残し、file loading/recompute は offline lane へ |
| `boundary_diagnostics.py` | 1,787 | H | pre/post canonical boundary A/B experiment、fixtures、artifact writer | 2 | **no** | **no** | script + dedicated test only | script-facing only | **delete/archive** | canonical SOURCE/SINK 確定後は役割終了 |
| `environment.py` | 271 | PC | registered environment/task factory、Observation ingestion、artifact hook | 6 | yes | replay evaluator が構築 | testsも利用 | yes (`get_environment/get_tasks/get_tasks_split`) | keep | private tool method 呼出しは整理候補 |
| `evaluation.py` | 2,606 | PC | primary score + diagnostics + reference + DTOs | 6 | tools assertion 経由 | primary evaluator本体 | scripts/tests多数 | yes (`__all__`) | **split** | 最大の責務集中箇所 |
| `facts.py` | 458 | PC | semantic plan/sidecar models、catalog validation、private ledger | 1 | environment/user/tools | primary score は ledger args を未使用 | grounding/tests | internal-private API | rename/split | `facts` より `semantic_sidecar` が実態に近い |
| `graph.py` | 1,206 | PC | Truth/Agent models、epistemic markers、canonical validation/projection、DB | 0 | yes、14 source importers | yes | diagnostics/testsにも広く利用 | de facto public | keep + helpers cleanup | generic model 化は非推奨 |
| `grounding.py` | 151 | D | sidecar annotation と EvidenceRef span の diagnostic grounding | 2 | `evaluation.py` が re-export のため import | primary evaluate は**呼ばない** | 実利用は tests/diagnostic | re-export only | move/remove eager import | score path に不要 |
| `joint_structural_alignment.py` | 1,442 | D | label-independent bounded joint matching experiment | 1 | 現在 `evaluate()` が毎回実行 | score mapping には未使用 | audit script + tests | diagnostic builder | move offline | production matcher に丸ごと採用しない |
| `knowledge.py` | 958 | PC | stakeholder models、resolver、projection、forgetting、contraction、retry | 1 | scenario/user | reference/term extras | tests/scripts | yes (`project_knowledge`) | merge helpers / optional split | essential logic が多い |
| `offline_diagnostics.py` | 837 | D | historical artifact decode、failed-slot attribution | 2 | **no** | **no** | one script + one test | script-facing | move to scripts | production package に不要 |
| `run_metrics.py` | 215 | D/T | smoke run tool error/refusal accounting | 0 | **no** | **no** | smoke script + tests | script-facing | move/merge | generic partsは `tau2.utils.llm_call_metrics` 候補 |
| `scenario.py` | 631 | PC | inline Truth/filter fixtures、scenario registry、locale view | 3 | yes | tools が task id から取得 | tests/scripts | domain internal | data extraction | 450 LOC前後が authored fixture data |
| `stakeholder.py` | 137 | PC | filter、forgetting config、concept overrides | 1 | scenario/knowledge | reference metadata | tests | yes | keep | 小さく責務明確。無理に merge しない |
| `tools.py` | 1,228 | PC | 21 Agent-visible tools、mutation validation、assertions | 4 | yes | assertion adapter | tests | **tool schema public** | internal simplify/split adapter | public signatures は維持 |
| `usage_alignment.py` | 926 | D | usage-based concept identity experiment | 1 | 現在 `evaluate()` が毎回実行 | score mapping には未使用 | one test | diagnostic builder | move offline | exception が primary evaluation を止め得る |
| `user_simulator.py` | 878 | PC | prompt/rendering、codec、two-stage LLM execution/retry | 3 | registry から構築 | no direct score | tests | registered user | extract sidecar codec | 1 class に4責務 |
| `utils.py` | 5 | PC support | policy/task path constants | 0 | environment | no | tests | internal | keep/merge optional | file削減目的の merge は無意味 |

### Direct dependency observations

- `graph.py` は domain 内最大の hub（14 source modules が直接 import）。
- `evaluation.py` の domain dependencies は `graph`, `grounding`, `knowledge`,
  `stakeholder`, `usage_alignment`, `joint_structural_alignment` の6つ。
- `environment.py` は6 dependencies を持つが、construction boundary として自然。
- `boundary_diagnostics.py` と `offline_diagnostics.py` の production source importer は0。
- `usage_alignment.py` の production importer は `evaluation.py` のみ。
- `joint_structural_alignment.py` の production importers は `evaluation.py` と
  historical `boundary_diagnostics.py` のみ。

## Essential vs Accidental Complexity

### Essential complexity — 残すべきもの

| Complexity | なぜ必要か | 保持方針 |
| --- | --- | --- |
| Truth 2-state / Stakeholder 3-state / Agent 4-state slots | ground truth、knowledge、belief の epistemic semantics が異なる | concrete model を分けたまま維持 |
| canonical SOURCE/SINK | multi-entry/exit を denominator と分離し、projection を一意にする | `graph.py` の explicit metadata + validator を維持 |
| safe forgetting contraction | 知らない serial node を消す際に branch/condition/self-loop/parallel edge を捏造しない | conservative rejection と provenance を維持 |
| deterministic Agent→Truth alignment | agent-local ids/labels と Truth ids は一致しない | lexical/content matcher と one-to-one concept assignment を維持 |
| DONT_KNOW / ABSENT / UNSET semantics | 未調査、既知不存在、未知を区別する benchmark contract | Agent-visible tools と score matrix を維持 |
| opaque stakeholder-local ids | Truth id leakage を防ぐ | private mapping と renderer separation を維持 |
| two-stage stakeholder response + integrity validation | Knowledge 外の assertion が会話に入るのを防ぐ | plan/catalog validation を維持 |
| multiple stakeholder reference support | 複数 knowledge view の情報品質を比較可能にする | optional/reference lane として維持可 |
| immutable Observations | evidence rewardを維持する現 contract に必要 | environment ownership を維持 |

### Accidental complexity — 削減対象

- score-independent experiment が `EvaluationDiagnostics` の required fields である。
- reward assertion ごとに全 diagnostic/reference を再計算する。
- pre-canonical boundary normalizer と current canonical validator が並存する。
- primary result、diagnostic trace、reference report が1つの Pydantic DTO に結合する。
- Agent と stakeholder の Truth comparison aggregation が別実装である。
- node/edge slot access、span search、contraction checks、assignment utility が重複する。
- `user_simulator.py` に protocol text、JSON codec、rendering、provider call、retry が混在。
- `facts.py` の ledger は ingestion validation と永続 diagnostic storage を兼務する。
- legacy endpoint/schema/single-stakeholder compatibility が core branch を増やす。
- scenario fixtures と factory/registry logic が同じ Python file にある。
- historical reports と 21 MiB artifacts が現 architecture の理解を難しくする。

## Diagnostic / Experimental Code Audit

### `boundary_diagnostics.py`

**判定: production から削除する。残す価値は研究再現用 archive のみ。**

- production runtime/evaluator caller は0。
- caller は `scripts/business_interview_boundary_diagnostics.py` と
  `test_boundary_diagnostics.py` のみ。
- module 自身が冒頭で「outside the production evaluator contract」と明記する。
- virtual `STRUCTURAL_START/END` を追加して A/B 比較するが、current TruthGraph は既に
  explicit `STRUCTURAL_SOURCE/SINK` と typed boundary edges を持つ。
- topology、reachability、normalization、boundary ids、canonical objective を再実装し、
  `graph.py` の current canonical contract と重複する。

`graph.py` へ移すべき utility はない。必要な production primitives は既に:

- `canonical_structure_errors`
- `canonicalize_truth_graph`
- `business_entry_node_ids` / `business_exit_node_ids`
- `business_graph_projection`
- `node_is_structural` / `edge_is_structural`

として存在する。unique な historical A/B reproducibility が必要なら、generated JSON と
短い design note を archive に残し、executable implementation は
`src/experiments/` または git history に任せる。

### `joint_structural_alignment.py`

**判定: production path に置かない。offline experiment としてのみ残すか、意思決定後に削除。**

- production score、mapping、pass/fail はこの結果を読まない。
- tests は builder を差し替えても score fields が同一であることを確認する。
- 現在は `evaluation.py:2229-2244` で毎 evaluation 実行される。
- bounded search の上限は node 250K states、concept 1M states、optimal alternatives
  2,048。run ごと、さらに assertion ごとに走らせる合理性はない。
- exception は catch されるが、eager import と DTO dependency は残る。

今後 production matcher に採用を検討できる最小部分は、parallel edges を扱う
endpoint-grouped **one-to-one edge assignment** の考え方である。ただし現在の primary
edge mapper の semantics を変えるため、本 audit では移植しない。joint search 全体を
production matcher に採用するのは過剰である。

### `usage_alignment.py`

**判定: production path に置かない。offline diagnostic へ移す。**

- module docstring が production scoring 非参加を明記。
- current production node/edge mapping を scaffold とするため、完全に独立した matcher
  ではなく、研究結果の解釈にも circularity がある。
- score/mapping/pass は結果を使わない。
- `evaluation.py:2220-2228` で毎回実行され、こちらは joint と違って exception guard が
  ない。diagnostic bug が reward calculation を停止できる。
- assignment、label rendering、referenced concept extraction が primary evaluator と重複。

採用判断が済んでいるなら module/test ごと削除可能。研究継続なら
`src/experiments/business_interview/` か offline diagnostics package に置く。

### `offline_diagnostics.py`

**判定: production package ではなく script/tooling lane に置く。**

- callers は `scripts/business_interview_evaluation_diagnostics.py` と dedicated test のみ。
- artifact decoder、質問 heuristic、failed-slot attribution は interview runtime ではない。
- `ConceptDiagnostics`, `SlotDiagnostic`, `FailureAttribution` を production
  `evaluation.py` から import し、DTO schema coupling を作る。

推奨 location:

```text
scripts/business_interview_diagnostics/
    artifact_decode.py
    failure_attribution.py
    cli.py
```

外部から reusable library として保証する必要がある場合だけ
`business_interview/diagnostics/` subpackage を選ぶ。どちらの場合も production
`evaluation.py` から reverse dependency を作らない。

### `run_metrics.py`

**判定: smoke script support。domain production package から移す。**

`business_interview_real_llm_smoke.py` と tests だけが caller。generic refusal detection は
既に `tau2.utils.llm_call_metrics` にあるので、共通化できる部分はそこへ、tool-specific
report assembly は smoke script の隣へ移す。

### `grounding.py`

**判定: diagnostic/test support。現在の primary evaluator import は不要。**

`evaluation.py` は2関数を alias re-export するが、`evaluate()` はどちらも呼ばない。
実 caller は tests の `grounded_semantic_ids` だけである。tests は `grounding.py` を直接
import し、production evaluator の eager dependency を外せる。

## Evaluation Architecture Audit

### `evaluation.py` が現在持つ責務

| Responsibility | Lines (approx.) | Primary core? | Target |
| --- | ---: | --- | --- |
| production/result DTO | 389-452 | yes | lean resultに限定 |
| diagnostic DTOs | 112-277 | no | diagnostics modelsへ |
| stakeholder reference DTOs | 289-386 | no | reference moduleへ |
| normalization/tokenization/similarity | 576-729 | yes | matcher core |
| maximum-weight concept assignment | 732-821 | yes | shared assignment utilityは1実装 |
| node/edge mapping | 842-926 | yes | alignment resultを返す純粋関数 |
| scalar/list slot scoring | 932-1013 | yes | shared comparison primitive |
| concept alignment/metrics | 1021-1092 | yes | Agent-specific adapter |
| evidence coverage/authenticity | 1105-1182 | current reward yes | reconstructionとは別component |
| informational knowledge coverage | 1190-1243 | no | reference/artifact enrichment |
| stakeholder input compatibility/coercion | 1246-1364 | no | adapter削除/別module |
| Stakeholder→Truth mappings/comparison | 1367-1760 | reference only | reference evaluator |
| canonical contract diagnostics | 1770-1826 | no | diagnostics |
| concept/slot/node/edge explanation | 1844-2347 | no | diagnostics builder |
| usage/joint experiment invocation | 2220-2244 | no | offline only |
| primary orchestration + return assembly | 2355-2606 | mixed | lean facade |

### Primary evaluator に本当に必要な責務

- explicit Truth business projection
- stakeholder-local terminology extras を入力にした Agent concept alignment
- Agent node/edge alignment
- two/four-state slot comparison
- node/edge/endpoint/concept aggregate metrics
- reconstruction pass
- 現 reward contract を維持する間の evidence metrics
- protocol completion field

不要な責務:

- Usage/Joint experiments
- failure attribution
- canonical A/B explanation
- stakeholder input coercion/history compatibility
- Stakeholder reference report
- artifact serialization DTO
- semantic grounding re-export

### Agent→Truth と Stakeholder→Truth の共通化可能範囲

既に共通利用されているもの:

- `business_graph_projection`
- `_score_scalar_slot`
- `_score_list_slot`
- `_NODE_PROPS`
- business node/edge denominators

重複しているもの:

- node/edge recall・precision
- fabricated counts
- start/end comparison
- six node property loops
- edge condition loop
- concept recall/precision
- completeness/pass predicate
- result assembly

異なるため分けるべきもの:

- Agent alignment: lexical/content-based
- Stakeholder alignment: private local→Truth mapping-based
- known absence: Agent は `AbsentType`、Stakeholder は `None`
- Stakeholder shortcut: exact Truth edge credit を与えない
- reference aggregate/forgetting metadata

従って、alignment adapter は分けたまま、**aligned graph comparison の60–70%程度**は
共有できる。

### 推奨 pure core

```python
alignment = align_agent_to_truth(agent, truth, terminology_terms)
result = compare_aligned_graphs(
    candidate=agent,
    truth=business_graph_projection(truth),
    alignment=alignment,
    known_absent=is_absent,
)
```

Reference は:

```python
alignment = alignment_from_private_knowledge_mappings(knowledge, truth)
result = compare_aligned_graphs(
    candidate=knowledge.graph,
    truth=business_graph_projection(truth),
    alignment=alignment,
    known_absent=lambda value: value is None,
)
```

`compare_aligned_graphs()` は Pydantic diagnostic DTO、InterviewDB、script I/O、LLM、
artifact を知らない純粋関数にする。alignment が one-to-one であることを boundary で
validate し、empty denominator policy を明示する。

### 現在の repeated evaluation

Standard task は3つの score assertion と1つの diagnostics hook を持つため、同じ
AgentGraph に対して `_evaluate()` を4回呼ぶ。各回で:

- primary matcher
- evidence metrics
- detailed trace
- usage experiment
- joint search
- stakeholder reference comparison

が再計算される。

behavior-preserving な修正は、final graph/database revision key ごとに primary result を
cache し、diagnostic enrichment を `get_eval_diagnostics()` で1回だけ行うことである。
assertions は同じ cached primary result の field を読む。

### Evidence の contract mismatch

`evaluation.py` の comments は evidence を diagnostic-only と呼び、`quality_pass` には
含めない。しかし standard `tasks.json` は `assert_evidence_backed` を
`ENV_ASSERTION` reward に含め、scalar reward を乗算する。従って現実には:

- semantic grounding は primary score に不要
- しかし evidence coverage/authenticity は standard benchmark reward の gate

である。単純化時にこれを無断で削除してはいけない。product contract として evidence
を本当に reference-only にするなら、task criteria 変更は scoring change なので Phase 3
の別決定にする。

## Graph Model Audit

### 保持すべき epistemic distinction

| Model | Slot semantics | Meaning |
| --- | --- | --- |
| TruthNode/TruthEdge | `ConceptRef \| None` | complete canonical truth; None=known absent |
| StakeholderNode/Edge | `ConceptRef/list \| None \| DONT_KNOW` | known value / known absent / unknown |
| Agent Node/Edge | `ConceptRef/list \| UNSET \| ABSENT \| DONT_KNOW` | uninvestigated / asserted value / explicit absent / unknown |

これを `Graph[NodeType, EdgeType]` / `Node[SlotType]` Pydantic generic hierarchyにするのは
推奨しない。defaults、union serialization、tool schema、validators の差を type machinery
に隠し、削減できるのは主に field declarations 100 LOC未満だからである。

### 実際の重複

| Surface | 重複 | 推奨 |
| --- | --- | --- |
| node id + six slots | 3 concrete node models | declarationsは維持 |
| edge id/from/to/condition | 3 edge models | declarationsは維持 |
| Truth/Stakeholder structural metadata | ほぼ同一 | small immutable metadata helperは可、inheritance不要 |
| `refs/asserted_refs/slot_value/slot_evidence` | Agent/Truth methods + Stakeholder helper | pure helperへ統一 |
| graph traversal/ref extraction | graph/evaluation/usage/joint/boundary | core helperを1つ、experiment duplicationは移動で消す |
| semantic ID generation | `graph_semantic_ids` にほぼ集約済み | 維持 |
| canonical SOURCE/SINK | `graph.py` は集約済み、boundary experimentだけ重複 | boundary削除 |
| serial contraction safety | `contract_serial_node` と `_contract_forgotten_nodes` | pure contraction plan helperを共有 |
| span search | EvidenceRef、Observation、facts、grounding の4実装 | `resolve_occurrence_span` 1つへ |
| validation | GraphMixin/StakeholderGraph で部分重複 | read-only Protocol + pure validator helper |

`_NodeProto`, `_EdgeProto`, `N`, `E` は現在 algorithm の型付けに使われていない。
actual algorithm boundary に使わないなら削除する。既存 `_GraphMixin[C]` は shallow な
共有として許容できるが、nodes/edges が `Any` のため、さらに generic hierarchy を
深くする価値は低い。

### 削減見積り

軽量 helper と legacy cleanup で `graph.py`/`knowledge.py` 合計から約
**150–300 LOC** を削減可能。generic Pydantic model 化の追加 complexity に比べると
小さいため、ここは最大の削減ポイントではない。

## Stakeholder / Forgetting Audit

### Current responsibility boundaries

- **Knowledge construction**: `scenario.py` が Truth/Filter を作り、
  `knowledge.project_knowledge()` を呼ぶ。
- **Forgetting policy/config**: `stakeholder.py` の `StakeholderForgettingConfig` と
  `StakeholderFilter`。
- **Sampling/contraction/materialization/retry**: `knowledge.py:507-958`。
- **Knowledge rendering**: `user_simulator.py:435-579`。
- **Sidecar codec/validation**: `user_simulator.py:271-432` + `facts.py`。
- **LLM execution/retry**: `user_simulator.py:642-878`。
- **Private integrity ledger**: `facts.py:234-458` + `environment.py:155-207`。

`knowledge.py` は大きいが、projection pipeline と epistemic model が密接で、単純に
`knowledge_projection.py`, `forgetting.py`, `models.py` の3 filesへ分けるだけでは改善に
ならない。

推奨順:

1. duplicate contraction checks を pure helper に統一。
2. Truth に対する StakeholderFilter cross-validation を projection boundary に追加
   （unknown node/edge/property typo を silent forgetting にしない）。これは hardening
   であり behavior change のため本 audit では実装しない。
3. `facts.py` を `semantic_sidecar.py` に置き換え、`parse_plan/parse_sidecar` を同 module
   に移す。simulator は orchestration/rendering だけにする。
4. それでも `knowledge.py` の変更頻度が高い場合のみ models (~300 LOC) と projection
   (~550 LOC) に2分する。`forgetting.py` をさらに独立させない。

### 残すべき複雑さ

- rejection sampling と retry reason
- branch/merge/conditioned path contraction rejection
- protected boundaries
- opaque local IDs
- shortcut provenance
- known absent と DONT_KNOW の区別

これらは benchmark integrity の本質で、LOC削減目的で弱めてはいけない。

## Tooling Audit

`InterviewTools` は約1,228 LOC、**21 Agent-visible tools** を持つ。

| Category | Tools | Assessment |
| --- | ---: | --- |
| glossary | 6 | Agent-local concept identityに必要 |
| session/observation/protocol | 5 | start/list/endpoints/validate/finish は必要 |
| node mutation/epistemic state | 5 | public explicitness はLLM usabilityに有益 |
| edge mutation/epistemic state | 5 | 同上 |

大きい理由の一部は tool docstrings と explicit public schemas であり、本質的である。
Agent-visible API を generic `set_property(target, property, state, value)` へ潰すと LOC は
減るが、tool schema の明瞭さと model behavior が変わるため推奨しない。

### API を変えずに可能な simplification

- unused `_PROPERTY_KIND`
- unused `_require_concepts()`
- unused `_refs()`
- unused `_ref_from_arg()`
- marker builders の unused `where/prop/bound_node/bound_edge` plumbing
- node DONT_KNOW/ABSENT の共通 private helper
- edge DONT_KNOW/ABSENT の共通 private helper
- declarative slot schema:

```python
NODE_SLOTS = {
    "activity": ("activity", False),
    "actor": ("actor", False),
    "system": ("system", False),
    "reads": ("data", True),
    "writes": ("data", True),
    "necessity_rationale": ("rationale", False),
}
```

これで add/update parsing の kind dispatch と marker handling を共有し、public signatures
/docstrings はそのまま保つ。

- mutation 前に patch 全体を parse/validate し、成功後に一括適用する。
- `_capture_user_message()` を environment-owned Observation service へ移す。
- `_evaluate()` と assertion adapter を evaluation service へ移し、tool class は薄い hook
  だけ持つ。

現 API を維持した現実的な削減幅は **200–350 LOC**、`tools.py` を約
850–1,000 LOC にする程度。これ以上の大幅削減は tool schema change を伴う可能性が高い。

## Provenance / Grounding Audit

### Runtime integrity に本当に必要

- `PlannedResponseItem`
- `SemanticAnnotation` と semantic mode
- `StakeholderKnowledgeCatalog.validate_plan/validate_annotations`
- exact public span validation
- Knowledge semantic ID resolver
- private IDs/mappings が Agent-visible state に入らない境界
- current evidence reward contract を維持する `EvidenceRef`/Observation coverage

### Primary reconstruction core ではない

- `ConceptAlignmentAssertion`
- `TerminologyConfirmation` の ledger persistence
- `SemanticLedger.annotations/alignments/terminology` の evaluator forwarding
- `grounding.grounded_ids/grounded_refs`
- offline failure attribution
- annotation-to-Agent-evidence semantic binding

特に `tools._evaluate()` は `annotations`, `alignments`, `terminology` を渡すが、
`evaluation.evaluate()` は現在これらの parameters を一度も読むことなく捨てる。
従って ledger は simulator integrity/diagnostic record としては意味があるが、primary
scoring dependency ではない。

behavior-preserving simplification:

- evaluator の unused ledger parameters を削除。
- environment ingestion 時の catalog validation は残す。
- ledger persistence が offline artifact に保存されないなら、catalog holder と atomic
  `bind_turn()` に縮小できる。
- grounding module は diagnostics lane へ移す。

### 注意すべき integrity findings

本 audit では修正しないが、削除ではなく hardening が必要な箇所:

- occurrence に `ge=0` がなく、negative occurrence が span validation を bypass できる。
- empty semantic plan のとき `check_sidecar_covers_plan()` が sidecar extras を検査しない。
- environment は annotation/alignment/terminology を別々に commit し atomic でない。
- artifact fingerprint は non-empty value を再計算・照合しない。

これらは「provenance が不要」ではなく、integrity lane と score lane を分離すべき理由で
ある。

## Dead / Legacy / Compatibility Code

### 明確な dead/helper candidates

- `evaluation._prop_from_node()`
- `evaluation._node_concept_refs()`
- `graph.spans_correspond()`
- unused `_NodeProto`/`_EdgeProto`/`N`/`E`（実 algorithm の typing に使わない場合）
- `tools._PROPERTY_KIND`
- `tools._require_concepts()`
- `tools._refs()`
- `tools._ref_from_arg()`
- evaluator からの grounding helper re-export

### Obsolete compatibility

| Compatibility | Current reason | Recommendation |
| --- | --- | --- |
| empty `EvaluationSpec` | old evaluator API shape | remove; all callers pass empty object |
| `evaluate(... annotations, alignments, terminology)` | old provenance evaluator | remove unused args |
| `truth=None -> knowledge.graph` fallback | pre-Truth-primary API | make Truth required after caller migration |
| singular `knowledge/stakeholder` fallback | pre multi-stakeholder reference API | explicit reference inputへ |
| Truth `start_node_id/end_node_ids` inherited fields | Agent/shared mixin + old diagnostics | Truth semanticsから段階的に除外 |
| Stakeholder `start_node_id/end_node_ids` | reference convenience | boundary edgesからderive |
| tool `start_node_id` arg | old single-entry tool calls | tool schema changeなのでPhase 3 |
| `TruthGraph = BusinessProcessGraph` | historical class name | external API policyが不要なら整理 |
| `ForgettingConfig` alias | historical filter name | internal callerなし; remove candidate |
| artifact `validate=False` | pre-canonical files | current-only schema policyなら削除 |
| offline smoke decoder | old artifact shape | offline laneと一緒にarchive/delete |
| simulator no-task legacy path | direct construction compatibility | registry build は常に task を渡す; fail-fast候補 |

この experimental branch では backward compatibility が重要でないため、current artifacts
を一度 migration/validationした後は compat branches を残す理由は弱い。ただし
`set_graph_endpoints` の signature は Agent-visible tool schema なので、単なる internal
cleanup と同じ PR に混ぜない。

### Historical comments/docs/artifacts

- `scenario.py` は v11、`graph.py`/`tools.py`/`evaluation.py` は v13 と記載し、current
  architecture の version source がない。
- `doc/` の historical goal reports は既に削除された `dag.py`, `ground_truth.py`,
  `semantic.py` 等を参照する。
- 158 real-LLM JSON artifacts / 21 MiB は runtime import caller がない。

推奨:

- historical reports は `doc/archive/business_interview/README.md` の index からリンクする
  か、git history に任せる。
- bulk artifacts は object/release storage へ移し、current schema の最小 golden pair と
  summary のみ repository に残す。
- current architecture の唯一の入口を本書と domain README にする。

## Test Audit

### Quantitative inventory

| Test file | LOC | Tests | Classification |
| --- | ---: | ---: | --- |
| `test_graph_business_interview.py` | 4,191 | 104 | mixed primary behavior + historical negative + metrics |
| `test_stakeholder_truth_reference.py` | 822 | 21 | optional reference contract |
| `test_evaluation_diagnostics.py` | 661 | 19 | diagnostics/offline/snapshot |
| `test_joint_structural_alignment.py` | 652 | 20 | experiment + score-independence contract |
| `test_usage_alignment.py` | 579 | 14 | experiment + score-independence contract |
| `test_canonical_graph_contract.py` | 348 | 14 | essential behavior contract |
| `test_joint_structural_alignment_audit.py` | 325 | 6 | historical experiment audit |
| `test_boundary_diagnostics.py` | 248 | 14 | historical boundary experiment |
| `test_business_interview_roundtrips.py` | 925 | 16 | runtime integration/sidecar/orchestrator |
| `test_business_interview_artifact_provenance.py` | 367 | 6 | optional artifact contract |

### Keep

- canonical SOURCE/SINK construction/validation/projection
- 2/3/4-state epistemic matrices
- safe contraction and rejection cases
- opaque ID/non-leakage
- sidecar plan/catalog integrity
- Agent-visible tool schema and mutation behavior
- full AgentGraph→Truth score contracts
- replay/Observation creation
- current artifact round-trip/fingerprint contracts
- explicit test that optional diagnostics do not alter primary scores

### Move/delete with experiments

- `test_boundary_diagnostics.py`: module と一緒に削除。canonical unique behavior があれば
  `test_canonical_graph_contract.py` へ移す。
- usage/joint tests: experiment を残すなら experiment tree へ移す。採用判断済みなら
  exhaustive ambiguity/search-limit tests は削除。
- `test_joint_structural_alignment_audit.py`: audit script と一緒に archive/delete。
- `test_evaluation_diagnostics.py`: current artifact decoder/root-cause attribution 部分は
  scripts diagnostics の tests へ。primary reason-code tests は optional diagnostics suite。

これら5 files は合計 **2,465 LOC / 73 collected cases**。全てを削除すべきという意味ではなく、
production behavior suite から分離すべき量である。

### Redundant / implementation-history hotspots

`test_graph_business_interview.py` には以下のような「旧実装が存在しない」ことを固定する
historical tests が混在する。

- `test_truthclaim_and_claim_catalogs_are_gone`
- `test_no_stakeholder_semantic_assertion_type`
- `test_semantic_matcher_machinery_is_gone`
- `test_obsolete_concept_lifecycle_tools_are_removed`

endpoint fixtures は canonical、graph、reference、alignment、boundary の複数 files で重複。
large file は contract ごとに分ける価値があるが、file split だけを simplification と
数えてはいけない。shared graph builders を fixtures にし、同じ invariant の重複 assertion
を減らす。

## Proposed Target Architecture

### Dependency shape

```text
                    ┌──────────────────────┐
                    │ scenario factory/data │
                    └──────────┬───────────┘
                               │
                ┌──────────────▼──────────────┐
                │ TruthGraph + knowledge policy│
                └──────────────┬──────────────┘
                               │ project_knowledge
                ┌──────────────▼──────────────┐
                │ StakeholderKnowledge         │
                └───────┬───────────────┬─────┘
                        │               │ optional artifact
                 simulator/sidecar      ▼
                        │       artifact_provenance
                        ▼
conversation -> environment -> InterviewTools -> AgentGraph
                                      │
                                      ▼
                           pure Agent→Truth comparison
                                      │
                                      ▼
                            PrimaryEvaluationResult
                                      │
                    ┌─────────────────┴────────────────┐
                    │ optional/offline enrichment only │
                    │ reference / traces / usage / joint│
                    │ failure attribution / reports     │
                    └───────────────────────────────────┘
```

### Suggested tree

```text
src/tau2/domains/business_interview/
    __init__.py
    environment.py
    scenario.py                 # factory/loader only; authored graphs can be data
    graph.py                    # concrete Truth/Agent models + canonical helpers
    stakeholder.py              # small filter/forgetting config; keep explicit
    knowledge.py                # Stakeholder models + validated projection
    semantic_sidecar.py         # renamed facts.py + plan/sidecar codec/catalog
    user_simulator.py           # rendering + LLM orchestration
    tools.py                    # stable Agent-visible API
    comparison.py               # pure alignment + aligned comparison
    evaluation.py               # thin domain adapter/result/cache
    artifact_provenance.py      # optional artifact schema only

src/experiments/business_interview/       # if experiments are retained
    usage_alignment.py
    joint_structural_alignment.py

scripts/business_interview_diagnostics/
    artifact_decode.py
    failure_attribution.py
    report.py
```

`boundary_diagnostics.py` は target tree に置かない。

file count をさらに抑えたい場合は `comparison.py` を `evaluation.py` 本体とし、thin adapter
を `tools.py` 側に置いてもよい。重要なのは wrapper 数ではなく、primary comparison が
optional diagnostics を import しないことである。

## Refactoring Plan

### Phase 1: Safe cleanup

Goal: score/tool/schema/forgetting を変えず、production dependency graph を短くする。

1. boundary experiment unit を archive/delete。
2. `offline_diagnostics.py` と `run_metrics.py` を scripts/tooling lane へ移す。
3. evaluator の grounding re-export/eager import を外し、tests は direct import にする。
4. `usage/joint` invocation を primary assertion path から外す。
   - primary result を cache
   - `get_eval_diagnostics()` だけが optional enrichment を1回実行
   - current diagnostics JSON が必要なら compatibility assembler で同 shape を返す
5. dead helpers/constants/unused evaluator arguments を削除。
6. stale version comments と current docs entry point を更新。
7. bulk artifacts/historical reports を archive policy に従って整理。

Validation:

- `make test-business-interview`
- primary `EvaluationResult` fields の before/after golden parity
- scalar `RewardInfo.reward/reward_breakdown/env_assertions` parity
- `InterviewTools` JSON schema snapshot parity
- current diagnostic artifact shape を維持する場合は full JSON parity

### Phase 2: Structural simplification

Goal: behaviorを維持し、同じ概念の複数実装をなくす。

1. `align_agent_to_truth()` と `compare_aligned_graphs()` を pure core として抽出。
2. Stakeholder reference は private-mapping alignment adapter + same comparator を使う。
3. primary result、diagnostic trace、reference report DTO を分離。
4. final graph revision ごとの evaluation cache を environment/evaluator adapter に置く。
5. node slot/ref/span helpers を共通化。
6. serial contraction の validation/planning を1実装にする。
7. `facts.py` + simulator codec を `semantic_sidecar.py` に整理。
8. tools の declarative private slot parser と atomic patch construction。
9. scenario authored data を validated JSON/TOML へ移し、factory は fresh objects を返す。

### Phase 3: Optional redesign

Product/scoring/schema decisionを伴うため別 goal/PR にする。

- evidence を本当に diagnostic-only にして task reward assertion を外すか決定。
- primary edge mapping を one-to-one に修正。
- legacy start/end tool API と model fields を削除。
- artifact schema v2 と historical decoder 廃止。
- stochastic Knowledge を per-simulation seed で再生成。
- multiple stakeholder simulator runtime（referenceだけでなく会話）を正式化。
- provenance subsystem の atomic ingestion/fingerprint hardening。
- diagnostics plugin/enrichment interface。
- graph model genericizationは、測定で明確な利益が出る場合だけ検討。

## Concrete Removal / Merge Candidates

| File / symbol | Current callers | Why removable/movable | Replacement | Risk | Required tests |
| --- | --- | --- | --- | --- | --- |
| `boundary_diagnostics.py` whole file | boundary CLI + dedicated test | pre-canonical experiment、0 runtime callers | `graph.py` canonical helpers | low runtime / medium research history | canonical entries/exits/cycles/projection |
| boundary CLI + test | manual CLI | implementation本体と一体 | archived JSON/design note | low | retained canonical tests |
| `joint_structural_alignment.py` from production path | `evaluation.py`, boundary, audit/tests | score-independent、expensive | offline experiment | low score / medium artifact schema | score/reward parity、diagnostic opt-in |
| `usage_alignment.py` from production path | `evaluation.py`, one test | score-independent、unguarded exception | offline experiment | low score / medium artifact schema | forced-exception primary eval test |
| `offline_diagnostics.py` | one script + one test | no runtime/evaluator caller | scripts diagnostics package | low | current artifact decoder/re-eval |
| `run_metrics.py` | smoke script + tests | smoke-only | smoke package / generic utils | low | smoke report unit tests |
| evaluator grounding re-exports | only tests | primary evaluate never calls | direct diagnostic import | low | grounding unit tests |
| `EvaluationSpec` | tools/scripts/tests pass empty object | empty compatibility shell | remove arg | low | call-site/type tests |
| `annotations/alignments/terminology` evaluate args | tools only; unused in body | dead provenance API | ingestion-only validation | low | reward/metrics parity |
| `_prop_from_node`, `_node_concept_refs` | none | dead | none | low | lint/tests |
| tools dead helpers/constants | none | dead | none | low | tool schema parity |
| `graph.spans_correspond` | none | dead | shared span resolver if needed | low | span unit tests |
| grounding span duplicates | diagnostic tests | 4 implementations diverge | one `resolve_occurrence_span` | medium integrity | negative occurrence、repeated quote |
| two contraction implementations | public contraction + projection | same safety rules duplicated | pure contraction plan | medium | all canonical contraction tests |
| Stakeholder reference scoring loop | `evaluate`, artifact recompute | duplicates primary aggregation | shared aligned comparator | medium | all reference-primary component parity |
| legacy Truth fallback in `evaluate` | tests/scripts | all production calls know Truth | required `truth` | low after migration | all evaluate call sites |
| pre-canonical artifact decoder/`validate=False` | offline old files | experimental branch no compatibility need | current schema fail-closed | medium artifacts | corpus migration/current golden |
| bulk real-LLM artifacts | no Python import caller | 21 MiB output data | external archive + minimal golden | low runtime | retained golden re-eval |

## Risks

### Behavior-preserving refactor risks

1. **Diagnostics are observable artifacts.** Score が同じでも DTO field/order/schema を変えると
   stored artifact parity が壊れる。compatibility serializer か schema bump が必要。
2. **Evidence は実 reward gate。** comments だけを信じて削除すると scalar reward が変わる。
3. **Terminology extras は primary matcher input。** `knowledge.concepts[].terms` を
   evaluatorから外すと JA/local terminology score が変わり得る。
4. **Tool signatures are prompts.** Python APIだけでなく generated JSON schema が Agent
   behavior に影響する。
5. **Pydantic union serialization.** graph inheritance/generics は artifact/tool schema を
   silently変え得る。
6. **Mutable/global scenario behavior.** fresh object/seed fix は正しい方向だが、現在の exact
   behaviorを変えるため cleanup PR に混ぜない。
7. **Historical artifacts.** experimental branch でも、削除前に current supported corpus を
   明示する必要がある。

### Separate correctness findings (今回実装しない)

- Primary `_map_nodes_and_edges()` は Agent edge ごとに最初の endpoint-compatible Truth
  edgeを選び、Truth edgeをreserveしない。duplicate Agent edges が precision 1 になり得る。
- negative occurrence が exact-span validation を bypass できる。
- empty plan が extra sidecar annotations を許す。
- invalid StakeholderFilter ids/property names が silent forgetting になる。
- `merge_concepts()` は target を source に含める/duplicate source の prevalidation がない。
- `_call_llm()` が全 exception を response-format fallback として再試行する。
- sidecar ledger ingestion が atomic でない。
- stored fingerprint/schema version の mismatch を reject しない。

これらは cleanup の根拠にはなるが、修正は behavior/scoring/hardening change として
独立させる。

### 最もリスクの高い refactor

最も危険なのは、Truth/Stakeholder/Agent を1つの generic Pydantic graph hierarchy に統合し、
同時に endpoint/tool schema/matcher を変更すること。epistemic semantics、serialization、
LLM tool schema、historical artifacts、score mapping を同時に変えてしまう。

次に危険なのは joint matcher の一部を primary matcher に採用すること。特に one-to-one
edge assignment は correctness fix の可能性が高いが、現在の stored score を変えるため
「挙動を変えない simplification」ではない。

## Recommended First Change

### 次に1つだけ実装するなら

**Boundary experiment retirement PR** を行う。

1. `boundary_diagnostics.py` の unique contract assertions を canonical tests と照合。
2. 欠けている invariant があれば最小限だけ
   `test_canonical_graph_contract.py` へ移す。
3. module、CLI、dedicated test を削除。
4. historical generated comparison は短い archive note と結果 JSON だけ残すか、git history
   に任せる。
5. `make test-business-interview` を実行。

理由:

- production imports/callers 0
- scoring/tool/schema/forgetting への影響 0
- 1 PR で 2,091 LOC を削減
- canonical SOURCE/SINK が current architecture であることを明確化
- 後続の joint/usage isolation を理解しやすくする

その次の PR は primary evaluation caching と diagnostic enrichment の分離である。こちらが
runtime dependency/latency に対する最大の改善だが、artifact parity を扱うため boundary
retirement より一段リスクが高い。

## Audit Validation

本 audit の作成後、production code を変更せずに次を実行した。

- `uv run python scripts/test_business_interview.py quick`
  - domain deterministic suite: **212 passed**
  - live LLM/API call: なし
- `uv run pytest -q --tb=short tests/test_business_interview_artifact_provenance.py tests/test_business_interview_roundtrips.py`
  - root-level BI suites: **22 passed**
- `pytest --collect-only` による対象全体: **234 collected cases**
- Markdown LSP (`marksman`): clean
- `git diff --check`: clean

pytest では既存の `audioop` deprecation warning と unknown
`asyncio_default_fixture_loop_scope` config warning の2件だけが出た。

## Final Assessment

- コード量の一部は本質的である。特に epistemic states、canonical boundaries、safe
  contraction、opaque IDs、deterministic comparison は削ってはいけない。
- しかし 14.3K LOC 全体を production core とみなす必要はない。約37.5%は
  score-independent diagnostics/experiments である。
- `boundary_diagnostics.py` は役割を終えている。
- `joint_structural_alignment.py` と `usage_alignment.py` は研究価値があっても production
  path には不要。
- `offline_diagnostics.py` と `run_metrics.py` は scripts/tooling lane が自然。
- `evaluation.py` を primary comparison に戻すことが architecture simplification の中心。
- graph model は concrete semantics を維持し、小さな helper 共有に留める。
- `tools.py` は public explicitness を残し、private parser/mutation/evaluator adapter を整理する。
- provenance は simulator integrity と current evidence reward に必要な最小部分だけ core に
  残し、semantic grounding/failure attribution は diagnostics へ分離する。
- Phase 1–2 は score/tool/forgetting を変えず実行可能。ただし observable diagnostics
  artifacts を behavior に含める場合は compatibility serializer と parity tests が必要。
