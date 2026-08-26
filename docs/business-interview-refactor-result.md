# Business Interview Refactor Completion Result

## Overview

`docs/business-interview-simplification-audit.md` に基づく
`business_interview` の behavior-preserving simplification と follow-up cleanup は完了した。
今回の cleanup では scoring architecture を変更せず、前回の移動後に残っていた
regression、旧 import、重複 evaluation、Truth fallback を整理した。

維持した contract:

- Agent-to-Truth primary score と scalar reward / reward breakdown
- `quality_pass` / `structural_pass` と evidence reward semantics
- Stakeholder-to-Truth reference semantics
- canonical SOURCE/SINK、forgetting、contraction
- Truth / Stakeholder / Agent graph schemas
- Agent-visible 21-tool API と生成 JSON schema
- simulator behavior
- `evaluation_inputs`、fingerprints、artifact reference recomputation

## Production Package

| State | Python files | Raw LOC |
| --- | ---: | ---: |
| Simplification 前 | 18 | 14,301 |
| Follow-up 完了後 | 15 | 8,996 |
| Difference | -3 | -5,305 (-37.1%) |

主要ファイル:

| File | LOC | Responsibility |
| --- | ---: | --- |
| `evaluation.py` | 496 | primary orchestration、ordinary diagnostics、reference assembly |
| `comparison.py` | 967 | Agent alignment と shared aligned-graph comparison |
| `evaluation_diagnostics.py` | 706 | production diagnostic DTO/builders |
| `reference_evaluation.py` | 446 | Stakeholder-to-Truth reference evaluation |
| `tools.py` | 1,268 | stable 21-tool API、assertion adapter、instance-local evaluation sharing |

前回 production package から移動・削除したものは維持している。

- deleted: `boundary_diagnostics.py`
- experiments: `src/experiments/business_interview/usage_alignment.py`
- experiments: `src/experiments/business_interview/joint_structural_alignment.py`
- offline tooling: `scripts/business_interview_diagnostics/offline_diagnostics.py`
- offline tooling: `scripts/business_interview_diagnostics/grounding.py`
- offline tooling: `scripts/business_interview_run_metrics.py`

## Follow-up Cleanup

### Offline diagnostics regression

原因は、production `EvaluationDiagnostics v5` から usage/joint fields を削除した後も、
`scripts/business_interview_evaluation_diagnostics.py` が次の旧位置を読んでいたことだった。

```text
result.diagnostics.usage_alignment
result.diagnostics.joint_structural_alignment
trace["evaluation"]["diagnostics"][...]
```

そのため stored artifacts の primary re-evaluation 自体は成功しても、joint audit の
assembly で `AttributeError` になり、3つの artifact tests が失敗していた。

修正後の offline flow:

```text
production evaluate()
    -> EvaluationResult + EvaluationDiagnostics v5
    -> scripts/business_interview_diagnostics/experiment_diagnostics.py
         -> build_usage_alignment_diagnostics(...)
         -> build_joint_structural_alignment_diagnostics(...)
    -> offline trace / report
```

Offline trace は責務を明示して次の sibling sections を持つ。

```text
evaluation:
  diagnostics: ...       # production v5 only
experiments:
  schema_version: business_interview.offline_experiments.v1
  usage_alignment: ...
  joint_structural_alignment: ...
```

Report renderer も `trace["experiments"]` を読むよう更新した。stored artifact integration
では primary、usage、joint、Markdown report generation を実際に実行している。
1 artifact の instrumentation 結果は primary / usage / joint = **1 / 1 / 1**。

### Experiment imports

Moved tests の旧 production imports を削除した。

```python
# removed
from tau2.domains.business_interview.usage_alignment import ...
from tau2.domains.business_interview.joint_structural_alignment import ...

# current
from experiments.business_interview.usage_alignment import ...
from experiments.business_interview.joint_structural_alignment import ...
```

Production package に compatibility shim は追加していない。

### Dependency direction

Current dependency direction:

```text
business_interview graph/comparison core
              ^
              |
production evaluation

business_interview graph/comparison core
              ^
              |
offline adapter -> usage/joint experiments
```

次の search で production package に experiment dependency がないことを確認した。

```bash
rg -n 'experiments\.business_interview|usage_alignment|joint_structural_alignment' \
  src/tau2/domains/business_interview
```

結果は 0 matches。experiment implementations は production `graph.py` のみを import する。
旧 `tau2.domains.business_interview.{usage_alignment,joint_structural_alignment}` import も
repository Python source で 0 matches。

### Diagnostic isolation

Usage/joint builder をそれぞれ強制的に `RuntimeError` にする tests を追加・更新した。
production `evaluate()` の full JSON result は builder failure 前後で一致し、production
`EvaluationDiagnostics` に experiment fields が存在しないことも確認した。

## Explicit Truth Contract

旧 logic:

```python
raw_target = truth if truth is not None else knowledge.graph if knowledge else None
```

は削除した。全 production/test/script call sites が Truth を明示していることを確認し、
`truth` を required keyword argument に変更した。明示的な `truth=None` は
`ValueError("evaluate(): truth must be an explicit Truth graph")` で fail-fast する。
StakeholderKnowledge graph を Truth として扱う compatibility fallback は存在しない。

Unused legacy evaluator kwargs `annotations`, `alignments`, `terminology` も削除した。
Sidecar validation/ledger behavior は evaluator の外側で従来どおり維持している。

## Same-state Evaluation Sharing

### Before / after

Standard replay lifecycle は1つの predicted environment / `InterviewTools` instance 上で
次を順番に実行する。

1. `assert_graph_reconstructed`
2. `assert_necessity_handled`
3. `assert_evidence_backed`
4. `get_eval_diagnostics`

| Invocation | Before audit | After initial split | Follow-up final |
| --- | ---: | ---: | ---: |
| production full `evaluate()` | 4 | 4 | **1** |
| production usage experiment | 4 | 0 | **0** |
| production joint experiment | 4 | 0 | **0** |

### Sharing method and stale-result protection

`InterviewTools` に global cache ではなく **instance-local one-entry cache** を置いた。
同じ evaluator lifecycle でのみ共有される。

Cache key は、evaluation に影響し得る全 input の canonical JSON から計算した SHA-256:

- complete `InterviewDB`（AgentGraph、messages、observations/evidence、
  `interview_complete`、summary）
- scenario Truth graph
- active StakeholderKnowledge（terminology terms を含む）
- Stakeholder filter metadata
- all Stakeholder reference identities、forgetting configuration、knowledge views

Evaluation が成功した場合だけ result を保存する。どれかの content が変わると key が変わり
cache miss になるため、mutation hooks の網羅性に依存しない。DB state mutation後に invocation
count が 1 から 2 へ増え、新しい `protocol_completed` が返る test も追加した。

Captured standard-flow wall time は、original pre-split baseline の約 15.68 秒から約 0.011 秒に
低下した。この差には experiment isolation と same-state sharing の両方が含まれる。

## Diagnostics Schemas

Production schema は引き続き:

```text
business_interview.evaluation_diagnostics.v5
```

Production fields:

- `canonical_contract`
- `score_fields_unchanged`
- `node_diagnostics`
- `unmatched_agent_nodes`
- `edge_diagnostics`
- `unmatched_agent_edges`
- `concepts`

`usage_alignment` と `joint_structural_alignment` は production schema に戻していない。
Offline experiment schema は独立した
`business_interview.offline_experiments.v1`。

## Validation Results

### Individual suites

| Suite | Result |
| --- | ---: |
| `tests/test_domains/test_business_interview/` | 161 passed |
| Root-level BI artifact + roundtrip tests | 22 passed |
| `tests/experiments/business_interview/` | 40 passed |
| Affected generic LLM/run-metrics tests | 19 passed |

### Combined related suite

```text
242 passed
0 failed
0 xfailed / 0 unexpected xfail
0 collection errors
```

Only pre-existing warnings remained: Python `audioop` deprecation and an unknown pytest
`asyncio_default_fixture_loop_scope` config option.

## Score, Reward, Reference, Tool, and Artifact Parity

Original baseline commit: `029f83e20cfe350cc219c10e3c021d7e131d9c2f`.

| Contract | Baseline / current evidence | Result |
| --- | --- | --- |
| Primary score fields SHA-256 | `3a3e1c8c4ebbb6b1c85f33042e24f796007552976d4a47684cc266e421cc5da9` | exact match |
| Scalar reward | `1.0` | exact match |
| Reward breakdown / assertions core SHA-256 | `e71a3afd5a5e941100ee892565961bc556d23782c52665d6dda0c23221184ac2` | exact match |
| Stakeholder reference SHA-256 | `3dfab6a8d1438da9107a4762d494c517ec6fa806fa3cbc20a0d7d839c7fee83f` | exact match |
| Agent-visible tool count | 21 | exact match |
| Tool schema SHA-256 | `f3b8dadcac07dcb42a6777bd850c7d62ea7f062f55048f34d283d96eef64f621` | exact match |
| Evaluation-input artifact SHA-256 | `ff2049a8172b0c10619eb337ff7d1522daf141e6da4f6ce07ce65bae4257fcad` | exact match |

Full historical `EvaluationResult` / `RewardInfo` SHA is intentionally not an exact parity target
because the old experiment-coupled diagnostics moved to the separate offline schema. Scalar reward、
breakdown、assertions、primary metrics、reference、artifact provenance are unchanged。

## Remaining Accidental Complexity

1. `EvaluationSpec` remains an empty compatibility model at all call sites.
2. `EvaluationResult` repeats `PrimaryEvaluationResult` fields during full-result assembly.
3. `evaluation.py` still builds ordinary detailed diagnostics and all Stakeholder references together;
   the one-entry sharing makes this once per state, but primary-only and enrichment-only APIs are not
   yet separate.
4. `scripts/business_interview_evaluation_diagnostics.py` remains a large combined artifact loader,
   trace assembler, and Markdown renderer.
5. `_evaluation_state_key()` serializes complete state on each assertion. This is much cheaper than four
   full evaluations and avoids stale results, but a future immutable DB revision counter could reduce
   key-generation work if measurement justifies it.
6. The mutable global scenario behavior remains intentionally unchanged。

## Deferred Correctness Work

The following correctness/hardening changes were intentionally not mixed into this cleanup:

- duplicate Agent edge / Truth edge reservation
- stochastic forgetting seed
- mutable global scenario
- negative occurrence validation
- empty plan sidecar extras
- invalid `StakeholderFilter` silent forgetting
- `merge_concepts` prevalidation
- broad LLM exception retry
- non-atomic sidecar ingestion
- fingerprint/schema mismatch hardening

旧い deferred note の **duplicate Agent edges が同じ Truth edgeを重複利用できる primary edge
reservation issue** は、`docs/business-interview-edge-matching-correctness.md` に記録した
one-to-one Edge fixで解決済みである。

Node identityのfalse-positive hardeningは、topology-first / WL-style refinementと保守的な
ambiguity policyを持つ独立実装として `docs/business-interview-node-matching-correctness.md`
に記録している。
