# business_interview — Actor / System / Read / Write Mismatch Analysis (visibility-aware, precision-first)

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** **Analysis + evaluator changes.** The analysis is based on the
already-committed DeepSeek artifacts (seeds 4000–4004); no new LLM runs. Two
follow-up changes are reflected: (1) scenario-local
`EvaluationSpec.data_expressions` recovered the visible `quote ↔ quotation`
target on `cq.writes`; (2) data matching is now **precision-first**
(normalized exact concept identity — token/substring matching removed), with
every accepted label declared explicitly in the scenario spec.

> **Supersedes the pre-visibility analysis.** The earlier version of this report
> (commit `be251e8` era) counted **50 mismatches / 22 evaluator-too-strict** under
> the old "all attributes visible" contract. Since commits `cafb944` and `6fa5d9e`
> the evaluator is **stakeholder-visibility-aware**: hidden attributes must remain
> unset, and asserting a hidden attribute is an **epistemic/fabrication error**,
> not an ordinary Truth-vs-agent semantic mismatch. This report re-classifies the
> same five saved real-LLM runs under the **current** contract and supersedes all
> earlier totals, patterns, and recommendations.

---

## 1. Analysis methodology

For each of the 5 artifacts (`run_00_seed4000` … `run_04_seed4004`):

1. Reconstruct the agent's inferred `BusinessDAG` from `final_dag` and the hidden
   Ground Truth from `quotation_truth()` / `quotation_spec()`.
2. Run the evaluator's **own** `_match_nodes` to get the agent→truth node mapping
   (so the inventory uses exactly the same matching the scorer uses).
3. For every matched node, compare `actor`, `system`, `reads`, `writes` between
   the agent node and the Truth node, and record the **current** evaluator result
   (`norm_role` / `norm_system` / `_data_recall` **with the scenario's
   `data_expressions`**).
4. Apply the **current scenario StakeholderFilter** (`quotation_sales_filter`)
   per node/axis:
   - **visible** attribute → compared against Truth as today;
   - **hidden** attribute → `unset` is correct (not a mismatch); `asserted` is an
     **epistemic/fabrication error** (class C), never a synonym error.
5. Classify every remaining mismatch and write the machine-readable analysis.

Helper: `scripts/attribute_mismatch_inventory.py` (reads the artifacts, emits
`attribute_mismatch_inventory.json` **and** `attribute_mismatch_analysis.json`;
the analysis is generated reproducibly from the inventory, not hand-written).
The inventory passes the scenario spec into `_data_recall`, so visible data
axes are scored exactly as the live evaluator scores them.

## 2. Current evaluator semantics (as of this analysis)

- **Node mapping** (`_match_nodes`): an agent node is a candidate for a Truth node
  iff it shares ≥2 significant (stopword-filtered) tokens with a hidden expression
  or contains a whole expression; actor/system agreement is a **ranking bonus**,
  not a rescue. Assignment is greedy by `(overlap, bonus)`. Node matching is
  unchanged.
- **Stakeholder visibility** (`StakeholderFilter.visible_node_attributes`): the
  sales stakeholder can know only
  - `r`: actor, writes
  - `cc`: actor, system, reads
  - `cq`: actor, system, reads, writes
  - `ap`: actor (system/reads/writes hidden; rationale via necessity)
  - `sq`: actor, system (reads/writes hidden)
  - `me`: actor, system, writes (reads hidden)
- **`norm_role` / `norm_system`** (`aliases.py`): substring matching against small
  alias tables. Roles: `sales` (…, interviewee, you, yourself, 営業, 自分, …),
  `manager` (…, supervisor, …). If no variant matches, the **raw** label is
  returned (e.g. `i (the stakeholder)`), which then fails equality.
- **`_data_recall`** (reads/writes) — **precision-first concept identity**:
  a Truth data item matches an agent value iff the **normalized exact
  canonical value** equals the agent value, OR the agent value equals a
  **normalized exact declared complete label** (`spec.data_expressions`).
  Normalization is case-fold + whitespace collapse only. **Token overlap and
  substring containment are removed** — `"quotation request"` /
  `"quotation information"` / `"price quotation"` do not match `quote` even
  though they contain the word `quotation`. Declared labels are the
  stakeholder-grounded complete labels observed in the saved runs
  (`quote ↔ quotation`, `customer ↔ customer information`,
  `pricing ↔ pricing information`, `request ↔ quotation request`,
  `excel_summary ↔ summary of quotation information`).

## 3. Per-run correctness — stored vs replayed

| run | actor | system | read | write | nodeR | nodeP |
| ----- | ------- | -------- | ------ | ------- | ------- | ------- |
| run_00_seed4000 | **0.00** | 1.00 | 0.60 | 0.40 | 0.83 | 0.71 |
| run_01_seed4001 | 1.00 | **0.50** | 0.50 | **0.33** | 1.00 | 0.86 |
| run_02_seed4002 | **0.17** | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |
| run_03_seed4003 | 1.00 | 0.80 | 0.60 | **0.20** | 0.83 | 0.83 |
| run_04_seed4004 | 1.00 | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |

DAG topology is stable (5/5 valid). **Historical vs replayed metrics.** The
table above is the `evaluator_metrics` **stored inside the artifacts at run
time** — they were produced by the **pre-visibility** scorer that was current
when the runs were saved (commit `2ca2413`), **not** by the stakeholder-aware
scorer (added later, commits `cafb944`/`6fa5d9e`), and **not** by the
precision-first matcher. Any visibility-aware / precision-first numbers in
this report are **replays of the saved DAGs under the current evaluator**
(`scripts/attribute_mismatch_inventory.py`,
`scripts/business_interview_reeval_smoke.py`,
`scripts/business_interview_data_expression_reeval.py`) — they are not what is
stored in the artifacts.

Under the current evaluator (replay, seeds 4000–4004):

| run | read (replay) | write (replay) |
| ----- | ------- | ------- |
| run_00_seed4000 | 0.50 | 0.80 |
| run_01_seed4001 | 0.83 | 0.50 |
| run_02_seed4002 | 0.75 | 0.67 |
| run_03_seed4003 | 0.70 | 0.80 |
| run_04_seed4004 | 0.67 | 0.67 |

`cq.writes` `quote↔quotation` is recovered 5/5. `cc.reads` (`customer
information`) and `cq.reads` `pricing information` are recovered by declared
labels; agent-embellished labels (`pricing information from the quoting
system`, `pricing information (from quoting system)`, `Excel file`) are now
**precision-first mismatches** (see §5–§6).

## 4. Mismatch totals under the current evaluator (28 matched node-pairs)

| attribute | mismatches (was) | now (precision-first) |
| ----------- | ------------------ | ----- |
| actor | 10 | **10** |
| system | 2 | **2** |
| reads | 7 | **10** |
| writes | 8 | **9** |
| **total** | **27** | **31** |

The 5 `cq.writes` `quote↔quotation` cases stay **recovered**. The +4 are the
precision-first casualties: visible labels that only matched through token
overlap and are **not** declared complete labels (`pricing information from
the quoting system` ×2 runs, `pricing information (from quoting system)` ×1,
`Excel file` ×1). These are now reported as mismatches rather than silently
accepted.

## 5. Classification (31 total, current evaluator)

| class | count | share | description |
| ------- | ------- | ------- | ------------- |
| **A genuine agent extraction error** | 0 | 0% | visible attr, agent value genuinely wrong |
| **B evaluator semantic/representation mismatch** | 14 | 45.2% | visible attr, semantically same but different representation |
| **C agent epistemic error (asserted hidden)** | 8 | 25.8% | hidden attr asserted by the agent — fabrication, not synonymy |
| **D visible fact never obtained/recorded** | 9 | 29.0% | visible attr, agent left it unset/empty |
| E GT/modeling ambiguity after visibility | 0 | 0% | — |
| F ambiguous | 0 | 0% | — |

### Per attribute

**actor (10, all B):**

- 10 × `"I (the stakeholder)"` vs `"sales"` (runs 00: 5 nodes, 02: 5 nodes).
  Visible; the stakeholder IS the sales employee and says "I", so this is an
  evaluator representation gap (`norm_role` lacks first-person/self-reference
  aliases). Per the terminology-policy goal, first-person coreference is
  treated as an **interviewing responsibility** (the Agent asks for and
  records the business role), so `norm_role` is intentionally **not** changed.

**system (2, all D):**

- run_01 send step: `system=None`; Truth `email` (stakeholder said "by email").
- run_01 month-end: `system=None`; Truth `excel` (stakeholder said "as an Excel
  file").
- The old 4 × approval-`quoting` cases are **gone**: `ap.system` is hidden, and
  leaving it unset is now **correct**.

**reads (10):**

- 7 × **C**: hidden `reads=[quote]` asserted by the agent as `quotation` /
  `quotation information` (me in runs 00/01/02/04; sq in runs 00/03/04).
  Epistemic errors — the expression layer explicitly does **not** apply to
  hidden attributes, and these remain failures.
- 3 × **B** (new, precision-first): `cq.reads` agent wrote `pricing information
  from the quoting system` / `pricing information (from quoting system)` for
  Truth `pricing` (runs 00/02/03; run 01/04 wrote the declared `pricing
  information` and match). The embellished labels are distinct complete labels
  and are **not** declared, so they now fail instead of matching through the
  shared tokens `pricing`/`information`.

**writes (9):**

- 1 × **C**: run_01 send `writes=["quotation"]` asserted for hidden
  `sent_quote` — epistemic error.
- 7 × **D**: `r.writes=["request"]` never recorded (5/5), month-end
  `excel_summary` never recorded (runs 02/04).
- 1 × **B** (new, precision-first): run_01 month-end `writes=["Excel file"]`
  for Truth `excel_summary` — the format, not the summary object; the declared
  label is `summary of quotation information` (recorded correctly by runs
  00/04, which match).
- The previous 5 × **B** `create-quotation GT quote vs agent quotation`
  (5/5 runs) are **recovered**: the quotation scenario declares
  `data_expressions` with `quote ↔ quotation` (normalized exact), a
  stakeholder-visible write, so the pair now scores as a hit.

## 6. Recurring patterns (current evaluator)

| pattern | frequency | class |
| --------- | ----------- | ------- |
| actor `"I (the stakeholder)"` vs GT `sales` | 2/5 runs (seeds 4000, 4002), 10 nodes | B (representation) |
| hidden send/month-end read `quote` asserted as `quotation`… | me 4/5, sq 3/5 | C (epistemic) |
| hidden `sent_quote` write asserted as `quotation` | 1/5 | C (epistemic) |
| `r.writes=["request"]` never recorded | 5/5 | D |
| month-end `excel_summary` never recorded | 2/5 (runs 02/04) | D |
| run_01 send/month-end `system` left None | 2 | D |
| `cq.reads` pricing label embellished (`from the quoting system` / `(from quoting system)`) | 3/5 | B (new, precision-first) |
| `me.writes` recorded as `Excel file` (format, not the summary) | 1/5 | B (new, precision-first) |

The former pattern `create-quotation write quote (agent quotation) not scored`
(5/5, B) is **gone** — recovered by the scenario-local expression.

## 7. Contribution: Agent vs Evaluator vs visibility vs coverage

- **Genuine agent extraction errors: 0** under the current contract.
- **Evaluator semantic/representation gaps (B, 45.2%)**: first-person actor
  labeling (10 cases, intentionally unchanged) + 4 new precision-first
  mismatches from agent-embellished visible labels that are not declared
  complete labels (pricing ×3, Excel-file ×1). These 4 are the honest price of
  exact concept identity: the fix is agent label discipline (use the
  stakeholder's term), not looser matching.
- **Epistemic errors (C, 25.8%)**: the agent asserted stakeholder-hidden reads/
  writes (8 cases). These must **not** be "fixed" by loosening the matcher —
  the correct agent behavior is to leave them unset. Replay confirms all 8
  remain failures.
- **Not-obtained visible facts (D, 29.0%)**: visible `request`/`excel_summary`
  writes and two systems the stakeholder explicitly stated were simply not
  recorded. Extraction/recording behavior, not matching.

## 8. Remaining evaluator mismatch population

After the precision-first change, the replay of the saved seed-4000..4004 DAGs
shows **31 mismatches**:

1. **`"I (the stakeholder)"` ↔ `sales` on visible actor axes — 10 cases.**
   Explicitly **out of scope**: no first-person / global actor aliases are
   added; the interview policy resolves the role by asking.
2. **Agent-embellished visible labels — 4 cases** (`pricing information from
   the quoting system` / `pricing information (from quoting system)` ×3,
   `Excel file` ×1). Precision-first by design; the declared stakeholder
   labels match, these do not.
3. **Epistemic errors (C, 8 cases)** — hidden reads/writes asserted as
   `quotation`. Not semantic targets; correct fix is agent behavior.
4. **Not-obtained visible facts (D, 9 cases)** — recording behavior, not
   matching.

Do **NOT** fix as semantic matching:

- hidden send/month-end `reads=[quote]` asserted as `quotation` (7 cases) — C;
- hidden `sent_quote` write asserted as `quotation` (1 case) — C;
- unrecorded visible `request` / `excel_summary` writes and `email`/`excel`
  systems (9 cases) — D (recording behavior, not matching);
- embellished pricing/Excel labels (4 cases) — B, resolved by the Agent using
  the stakeholder's term, never by restoring token/substring matching.

## 9. Matching rule (implemented)

Reads/writes matching is **precision-first concept identity**:

```text
normalized exact canonical value
        OR
normalized exact scenario-local accepted expression
```

- Normalization: case-fold + whitespace collapse (harmless formatting only).
- Token overlap and substring containment are **removed** — a value that
  merely contains a declared label never matches.
- `EvaluationSpec.data_expressions` lists **complete labels** identifying the
  same scenario concept; it is scenario-local, hidden, evaluator-only; the
  canonical Truth values are unchanged.
- Quotation declares exactly (all stakeholder-grounded, from the saved runs):

  ```python
  data_expressions={
      "quote": ["quote", "quotation"],
      "customer": ["customer", "customer information"],
      "pricing": ["pricing", "pricing information"],
      "request": ["request", "quotation request"],
      "excel_summary": ["excel_summary", "summary of quotation information"],
  }
  ```

- No `estimate` / `proposal` / `price sheet` / `document` / `offer` variants;
  no aliases for hidden `sent_quote` or hidden reads; hidden axes never use
  the expression layer.

Replay results (seeds 4000-4004 and terminology-policy seeds 5000/5001;
`scripts/business_interview_data_expression_reeval.py` →
`artifacts/business_interview_real_llm/data_expression_reeval.json`):

- `cq.writes` `quote↔quotation`: **7/7 recovered** (all replayed runs).
- Hidden epistemic assertions: **12/12 remain failures** (8 in seeds
  4000–4004, 4 in seeds 5000/5001).
- Collision probes (133 truth×probe pairs): **0 false positives**; every
  near-collision (`quotation request`, `quotation information`, `quotation
  document`, `price quotation`, `invoice`) fails against `quote`.
- 18 axis matches kept by declared labels; **7 matches lost for good** (the
  embellished-label cases above) — reported, not restored.
- No axis changed outside reads/writes; actor behavior unchanged (saved
  `final_dag` replayed verbatim; no LLM calls).

## 10. What should NOT be changed

- Do **not** restore token-overlap / substring matching for reads/writes.
- Do **not** add an embeddings / LLM judge / WordNet / stemming.
- Do **not** add first-person / global actor aliases to `norm_role`.
- Do **not** change the node-matching gate (`_match_nodes`).
- Do **not** change the stakeholder fidelity prompt / Ground Truth / `known_info`.
- Do **not** change the StakeholderFilter visibility (already aligned with
  `known_info`).
- Do **not** implement an interview-terminology glossary tool yet (design
  note: `doc/business-interview-terminology-design.md`).

## 11. Verification

- `scripts/attribute_mismatch_inventory.py` regenerates both JSONs reproducibly
  (identical SHA on rerun).
- `scripts/business_interview_data_expression_reeval.py` replays all saved
  seed-4000..4004 and seed-5000/5001 DAGs under loose vs exact-core vs
  exact-declared contracts and emits `data_expression_reeval.json`.
- `make check-all` (ruff lint + format) clean.
- `pytest tests/test_domains/test_business_interview/` → **110 passed**
  (includes the precision-first regression tests).

## 12. Deliverables

- Helper: `scripts/attribute_mismatch_inventory.py` (now also emits the analysis)
- Replay helper: `scripts/business_interview_data_expression_reeval.py`
- Inventory: `artifacts/business_interview_real_llm/attribute_mismatch_inventory.json`
- Summary: `artifacts/business_interview_real_llm/attribute_mismatch_analysis.json`
  (current-evaluator, precision-first)
- Replay: `artifacts/business_interview_real_llm/data_expression_reeval.json`
- Design note: `doc/business-interview-terminology-design.md`
- This report: `doc/business-interview-attribute-mismatch-analysis.md`
