# business_interview — Evaluator Node-Matching Fix Report

**Date:** 2026-08-18
**Branch:** `business-interview`
**Scope:** evaluator `node matching` only. Stakeholder simulator, `observe_turn`
UX, necessity semantics, evidence hygiene, the primitive resolver, scenario data,
and the smoke runner were **not** changed.

---

## 1. Root cause

In `evaluation.py`, `_match_nodes` mapped each agent node to a hidden Truth node
using `_match_score`, which counted **raw token overlap over every token** (no
stopword filtering) and added a bonus for substring containment:

```python
ov = len(at & _tokens(expr))
if e in t or t in e:
    ov += 1
best = max(best, ov)
```

`_TOKEN_RE = [a-z0-9]+` means even function words (`the`, `to`, `and`) counted,
so **one weak shared token** (e.g. `quotation`, `send`, `customer`) was enough to
make an agent node a candidate. `_match_nodes` then greedily assigned each agent
node to the first unused Truth node with `score > 0`.

**Observed failure** (real DeepSeek smoke run 00): the agent created a
`manager_approval` node but **never created a month-end node**. Because the
approval action ("Get approval from a manager before sending the quotation to the
customer") shared weak tokens with the month-end expressions ("send quotation
summary to accounting at month-end"), the old matcher assigned `manager_approval
→ me`. That hid the missing month-end node:

- `node_recall` stayed `1.0` even though month-end was absent.
- `record_request → ap` (another mis-map) similarly masked a real gap.
- **Edge recall/precision were inflated** because edges touching `me`/`ap`
  matched purely via the wrong node mapping.

Across the smoke runs, month-end was genuinely missed in 4/5 runs and approval in
2/5, yet **node_recall was `1.0` in all 5 runs**.

## 2. New matching rule (deterministic, no embeddings / LLM judge)

Rewrote the matching helpers in `evaluation.py`:

1. **Stopword filtering** (`_STOPWORDS`) + drop single-char tokens → `_sig_tokens`.
   Function words and connectors no longer contribute to overlap.
2. **`_MIN_NODE_OVERLAP = 2`**: an agent node is only a candidate for a Truth node
   when it shares **>= 2 significant tokens** with some hidden expression **OR**
   when it **contains a whole hidden expression** (`_expression_contained`).
3. **Substring containment is restricted** to meaningful phrases: an expression
   must have >= 2 significant tokens **or** CJK content. This is what keeps JA
   matching working (`_TOKEN_RE` cannot tokenize Japanese, so JA relies on
   substring containment of the full expression) while preventing a single common
   token (e.g. `customer`) from matching by substring alone.
4. **Actor/system are reinforcement, not rescue.** `_attribute_match` adds a
   bonus for actor (canonical `norm_role`) and system (`norm_system`) agreement,
   but **only ranks candidates that already cleared the action gate**. A weak
   action match can never be salvaged by attribute agreement.
5. **Global greedy assignment** by `(overlap, bonus)` descending, so an agent node
   with low confidence stays **unmatched** rather than being forced onto a Truth
   node it barely resembles. Arbitrary agent node ids remain allowed (matching is
   by action text only). `EvaluationSpec.expressions` are preserved.

The `EvaluationSpec` (hidden expressions, `primitive`) is unchanged.

## 3. Falsification tests added

New tests in `tests/test_domains/test_business_interview/test_dag_business_interview.py`:

| Test | Asserts |
|------|---------|
| `test_missing_month_end_node_lowers_node_recall` | month-end absent → `node_recall < 1.0` |
| `test_missing_approval_node_lowers_node_recall` | approval absent → `node_recall < 1.0` |
| `test_fabricated_node_not_mapped_to_missing_truth_node` | unrelated/fabricated node stays unmatched; the missing Truth node is not filled |
| `test_weak_single_token_overlap_does_not_match` | single weak shared token → no match |
| `test_valid_quotation_reconstruction_recall_precision_1` | valid reference reconstruction → `node_recall = node_precision = 1.0` |
| `test_arbitrary_ids_and_reasonable_paraphrase_match` | arbitrary ids + paraphrases (not exact expressions) still map 1:1 |
| `test_approval_node_not_mismatched_to_month_end` | approval action mentioning send/quotation/customer → `ap`, not `me` |
| `test_edge_metrics_not_inflated_by_node_mismatch` | missing month-end edge not counted → `edge_recall <= 5/6` |
| `test_en_ja_matching_still_equivalent_after_conservative_gate` | EN and JA full reconstructions still `quality_pass = True`, recall/precision 1.0 |

Existing suite (35 tests) still passes, including the quotation reference
trajectory (`test_evaluator_rewards_full_reconstruction` → reward 1.0,
`assert_dag_reconstructed = True`) and the lab scenario
(`test_non_quotation_lab_scenario_full_pass` → `quality_pass = True`,
`node_recall = 1.0`).

## 4. Smoke-artifact re-evaluation (before / after)

Re-ran `evaluate()` with the new matcher over the **saved final DAGs** from the
DeepSeek smoke runs (`scripts/business_interview_reeval_smoke.py`):

| run | OLD node_recall | NEW node_recall | OLD edge_recall | NEW edge_recall |
|-----|-----------------|-----------------|-----------------|-----------------|
| run_00_seed1000 | 1.000 | **0.833** | 0.500 | 0.667 |
| run_01_seed1001 | 1.000 | **0.667** | 0.500 | 0.333 |
| run_02_seed1002 | 1.000 | **0.667** | 0.000 | 0.333 |
| run_03_seed1003 | 1.000 | **1.000** | 0.333 | 0.833 |
| run_04_seed1004 | 1.000 | **0.667** | 0.833 | **0.500** |

**Interpretation:**

- **4/5 runs (00, 01, 02, 04) dropped below 1.0**, now correctly reflecting that
  month-end (all four) and approval (runs 02, 04) were actually missed.
- **run_03 stays 1.0** — this run genuinely discovered all 6 nodes (including
  month-end), so it is not falsely penalized. This confirms the gate is
  *conservative* without *over-penalizing* valid reconstructions.
- **run_04's edge_recall was inflated** (0.833 → 0.500): the old matcher's
  mis-mapping had fabricated month-end/approval edges. The new matcher removes
  that inflation.
- Concretely, run_00's mapping corrected from
  `{manager_approval→me, record_request→ap}` to
  `{manager_approval→ap, receive→r, check→cc, create→cq, send→sq}`, leaving
  `me` (month-end) and `record_request` **unmatched** — i.e. the missing node is
  now visible in the score.

## 5. Verification run

```
uv run pytest tests/test_domains/test_business_interview/      # 44 passed (35 original + 9 new)
make check-all                                                # ruff check + format clean
# quotation reference trajectory: test_evaluator_rewards_full_reconstruction (reward 1.0) passes
# lab_sample_flow: test_non_quotation_lab_scenario_full_pass (quality_pass True) passes
# smoke artifact re-evaluation: scripts/business_interview_reeval_smoke.py (see table above)
```

## 6. Remaining limitations

- **Matching is still heuristic and token/substring-based.** A genuinely
  ambiguous node (e.g. one action legitimately overlapping two Truth nodes) is
  resolved greedily by overlap + actor/system bonus; this is deterministic but
  not semantic.
- **Actor/system reinforcement depends on the model filling those fields.** When
  DeepSeek leaves `actor`/`system` blank (common in the smoke runs), the bonus is
  0 and overlap alone decides. This is unchanged scope-wise (attribute
  correctness is a separate metric).
- **JA still relies on substring containment** of full expressions; it works for
  the bundled scenarios but would not handle JA paraphrases that don't contain a
  hidden expression verbatim. No CJK tokenizer was added (out of scope, no
  embeddings / LLM judge).
- **`node_precision` can stay 1.0 when a Truth node is absent** (all present agent
  nodes are valid), so the missing-node signal comes from `node_recall`, not
  precision — by design.
- The re-evaluation reconstructs the DAG from saved JSON; authenticity/provenance
  sub-metrics were not recomputed there (node matching does not depend on
  observations).

## 7. Deliverables

- Code: `src/tau2/domains/business_interview/evaluation.py` (matching only)
- Tests: `tests/test_domains/test_business_interview/test_dag_business_interview.py`
- Diagnostic helper (not in CI): `scripts/business_interview_reeval_smoke.py`
- This report: `doc/business-interview-node-matching-fix-report.md`
