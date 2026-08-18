# business_interview — DAG Refinement / Cleanup Report

**Date:** 2026-08-18
**Branch:** `business-interview`
**Scope:** DAG refinement / cleanup (working-hypothesis model + node deletion +
finish validation). Stakeholder simulator, evaluator matching, Observation
capture, necessity semantics, and scenario Ground Truth were **not** changed.

---

## 1. Root cause

In real DeepSeek smoke runs the agent often built a coarse node early (e.g. a
single "prepare the quotation" node) and, after detailed follow-up, correctly
decomposed it into several more specific nodes (record → check → create). But it
**kept the old coarse node** as an orphan. The result was a structurally invalid
final DAG:

```
unreachable node: prepare_quotation
```

so `quality_pass = False` even though the agent's actual understanding was
correct. The DAG was being treated as an append-only record of everything ever
inferred, rather than a *working hypothesis* that should be cleaned up as
understanding sharpens.

## 2. Changes

### 2a. Agent prompt / policy (`data/tau2/domains/business_interview/policy.md`)

Added a "Refine the DAG (working hypothesis)" section that tells the agent:

- The inferred DAG is a **working hypothesis, not an append-only record**.
- New evidence may **refine, split, replace, merge, or invalidate** earlier nodes
  and edges.
- When a coarse placeholder node is decomposed into more specific activities, do
  **not** keep both unless the stakeholder describes them as distinct.
- When replacing a coarse node: preserve relevant evidence, reconnect
  incoming/outgoing edges, remove obsolete edges, and **remove the obsolete coarse
  node**.
- Do not finish with obsolete, unreachable, duplicate, or superseded nodes.
- Before `finish_interview`, call `validate_dag` and make the DAG structurally
  consistent.

No Ground Truth-hinting node identifiers are used in the prompt (keeps the
`test_hidden_truth_not_leaked_to_agent` gate green).

### 2b. New tools (`src/tau2/domains/business_interview/tools.py`)

- **`remove_node(node_id)`** *(WRITE)* — removes a node **and all incident edges**
  (incoming + outgoing), so no dangling edge is left. If the removed node was the
  declared start or an end node, that endpoint reference is cleared (the agent
  re-sets it). Stakeholder Observations are **kept** (evidence provenance on other
  nodes is preserved). A nonexistent node id raises a clear `ValueError`.
- **`validate_dag()`** *(READ)* — returns the DAG's **internal** structural
  validation status (`dag.validate()`): unreachable nodes, dangling edges,
  invalid start/end, cycles, etc. It never references the hidden Ground Truth.

No auto-refactor / `replace_node` tool was added — the LLM is expected to do the
editing itself (simple, safe primitives only).

### 2c. `finish_interview` validation

`finish_interview` now **refuses** a structurally invalid DAG. If
`dag.validate()` returns any error (unreachable node, dangling edge, invalid
endpoint, cycle, ...), it raises a `ValueError` listing the errors and does **not**
set `interview_complete`, so the agent can fix the DAG and call it again. The
error message contains only the DAG's internal consistency problems — never the
Ground Truth. An empty DAG (no `start_inference`) is also refused.

### 2d. Docs

`README.md` tool table updated with `remove_node`, `validate_dag`, and the
finish-validation note.

## 3. Required tests (business_interview suite: 67 passed)

New tests in `tests/test_domains/test_business_interview/test_dag_business_interview.py`:

| test | asserts |
|------|---------|
| `test_remove_node_after_decomposition_leaves_no_dangling_edge` | remove a coarse node removes it + its incident edge; no dangling edge remains |
| `test_remove_node_keeps_observations_and_provenance` | Observations kept; provenance on surviving nodes preserved |
| `test_remove_node_nonexistent_rejected` | invalid id → `ValueError` |
| `test_finish_rejects_unreachable_node` | unreachable node → finish rejected, `interview_complete` stays False |
| `test_finish_accepts_valid_dag` | valid DAG → finish succeeds |
| `test_finish_rejection_then_fix_and_refinish` | reject → remove orphan → re-finish succeeds |
| `test_validate_dag_reports_structural_errors_then_clean` | `validate_dag` reports errors, then "structurally valid" |
| `test_coarse_node_refinement_end_to_end_valid` | reproduce the smoke failure: coarse receive→prepare→send refined into receive→record→check→create→send with the coarse node removed; final DAG valid |

Existing gates preserved: quotation reference trajectory (`reward == 1.0`), lab
scenario full pass, EN/JA equivalence, evidence authenticity, node-matching and
observation-capture regressions.

**Note:** `test_arbitrary_ids_and_reasonable_paraphrase_match` now builds a
structurally valid DAG (edges added) so `finish_interview`'s new validation
succeeds — its node-matching assertions are unchanged.

## 4. Verification

- `pytest tests/test_domains/test_business_interview/` → **67 passed**
- `make check-all` (ruff lint + format) → clean
- `make test` → **297 passed**
- quotation reference trajectory, lab, EN/JA, evidence authenticity → pass

## 5. Real DeepSeek smoke (5 runs, seeds 4000–4004)

Interview Agent + Stakeholder both `deepseek/deepseek-chat` (→ `deepseek-v4-flash`).

| run | dag_valid | validation_errors | obsolete coarse node | remove_node | validate_dag | finish rejects | tool errors | node R/P | edge R/P |
|-----|-----------|-------------------|----------------------|-------------|--------------|----------------|-------------|----------|----------|
| run_00_seed4000 | ✅ | [] | ❌ (gone) | 1 | 2 | 0 | 0 | 0.83/0.71 | 0.67/0.62 |
| run_01_seed4001 | ✅ | [] | ❌ (gone) | 1 | 2 | 0 | 0 | 1.0/0.86 | 1.0/0.86 |
| run_02_seed4002 | ✅ | [] | ❌ (gone) | 1 | 4 | 0 | 0 | 1.0/0.86 | 1.0/0.86 |
| run_03_seed4003 | ✅ | [] | ❌ (gone) | 1 | 2 | 0 | 0 | 0.83/0.83 | 0.83/0.83 |
| run_04_seed4004 | ✅ | [] | ❌ (gone) | 1 | 3 | 0 | 0 | 1.0/0.86 | 1.0/0.88 |

### Key checks

1. **Coarse `prepare` node not left after refinement** — ✅ 0/5 (`remove_node`
   called once in every run).
2. **No unreachable / obsolete nodes in the final DAG** — ✅ `dag_valid` True and
   `validation_errors = []` in all 5 runs.
3. **Self-repair after a finish rejection** — no live run hit a rejection because
   every agent used `validate_dag` proactively and fixed structure before
   finishing. The rejection-then-repair path is covered by the unit test
   `test_finish_rejection_then_fix_and_refinish`.
4. **Stakeholder fidelity preserved** — no unsupported facts (0/5), no false
   denial of known facts (0/5), approval branch and month-end disclosed
   appropriately (run_03's "nothing else after sending" is a truthful statement
   about the per-quotation flow, not a denial of the separate month-end activity).
5. **No new tool-error loop** — `tool_errors = 0` in all 5 runs.

`quality_pass` is still `False` in every run, but only because `structural_pass`
additionally requires exact actor / system / read / write correctness vs the
hidden Ground Truth (e.g. actor "Analyst" vs "sales"), which is a scoring-quality
matter outside this task's scope. The cleanup objective — a **structurally valid
final DAG with no obsolete/unreachable/coarse nodes** — is fully met.

## 6. Remaining limitations

- `finish_interview` validates only **internal structural consistency**; it does
  not check Ground Truth completeness (that is the evaluator's job and is
  unchanged).
- The agent still needs to decide itself when a node is coarse and should be
  decomposed/removed; the prompt guides this but does not auto-detect it.
- The gold-environment replay of the tasks.json reference actions logs a cosmetic
  `Cannot finish: no DAG has been built yet` warning (the gold env is constructed
  from the initial greeting only, so it cannot reconstruct the DAG; DB is not in
  this domain's `reward_basis`, so this has no effect on scoring).
- `remove_node` removes incident edges and clears cleared endpoint references;
  the agent must re-set start/end if it removes an endpoint node.

## 7. Deliverables

- Code: `src/tau2/domains/business_interview/tools.py`
- Policy: `data/tau2/domains/business_interview/policy.md`
- Docs: `src/tau2/domains/business_interview/README.md`
- Tests: `tests/test_domains/test_business_interview/test_dag_business_interview.py`
- Smoke artifacts: `artifacts/business_interview_real_llm/run_*_seed400*.json`
- Cleanup summary: `artifacts/business_interview_real_llm/dag_refinement_cleanup_classification.json`
- This report: `doc/business-interview-dag-refinement-cleanup-report.md`
