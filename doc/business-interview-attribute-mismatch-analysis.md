# business_interview — Actor / System / Read / Write Mismatch Analysis (visibility-aware)

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** **Analysis only.** No behavior change — Evaluator, Agent prompt,
Stakeholder prompt, Ground Truth, tool API, and matching logic are unchanged. No
new LLM runs. Uses the already-committed DeepSeek artifacts (seeds 4000–4004).

> **Supersedes the pre-visibility analysis.** The earlier version of this report
> (commit `be251e8` era) counted **50 mismatches / 22 evaluator-too-strict** under
> the old "all attributes visible" contract. Since commits `cafb944` and `6fa5d9e`
> the evaluator is **stakeholder-visibility-aware**: hidden attributes must remain
> unset, and asserting a hidden attribute is an **epistemic/fabrication error**,
> not an ordinary Truth-vs-agent semantic mismatch. This report re-classifies the
> same five saved real-LLM runs under the **current** contract and supersedes all
> earlier totals, patterns, and recommendations. Do not use the old numbers to
> design Goal 3.

---

## 1. Analysis methodology

For each of the 5 artifacts (`run_00_seed4000` … `run_04_seed4004`):

1. Reconstruct the agent's inferred `BusinessDAG` from `final_dag` and the hidden
   Ground Truth from `quotation_truth()` / `quotation_spec()`.
2. Run the evaluator's **own** `_match_nodes` to get the agent→truth node mapping
   (so the inventory uses exactly the same matching the scorer uses).
3. For every matched node, compare `actor`, `system`, `reads`, `writes` between
   the agent node and the Truth node, and record the **current** evaluator result
   (`norm_role` / `norm_system` / `_data_recall`).
4. Apply the **current scenario StakeholderFilter** (`quotation_sales_filter`)
   per node/axis:
   - **visible** attribute → compared against Truth as today;
   - **hidden** attribute → `unset` is correct (not a mismatch); `asserted` is an
     **epistemic/fabrication error** (class C), never a synonym error.
5. Classify every remaining mismatch and write the machine-readable analysis.

Helper: `scripts/attribute_mismatch_inventory.py` (reads the artifacts, emits
`attribute_mismatch_inventory.json` **and** `attribute_mismatch_analysis.json`;
the analysis is generated reproducibly from the inventory, not hand-written). No
scoring behavior changed.

## 2. Current evaluator semantics (as of this analysis)

- **Node mapping** (`_match_nodes`): an agent node is a candidate for a Truth node
  iff it shares ≥2 significant (stopword-filtered) tokens with a hidden expression
  or contains a whole expression; actor/system agreement is a **ranking bonus**,
  not a rescue. Assignment is greedy by `(overlap, bonus)`.
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
- **`_data_recall`** (reads/writes): for each Truth item, a hit if **any** raw
  token overlaps OR the Truth value is a substring of the agent value. Raw
  `[a-z0-9]+` tokenization (no stemming, no synonym map): `quote` vs `quotation`
  → **miss**.

## 3. Per-run correctness (from `evaluator_metrics`)

| run | actor | system | read | write | nodeR | nodeP |
| ----- | ------- | -------- | ------ | ------- | ------- | ------- |
| run_00_seed4000 | **0.00** | 1.00 | 0.60 | 0.40 | 0.83 | 0.71 |
| run_01_seed4001 | 1.00 | **0.50** | 0.50 | **0.33** | 1.00 | 0.86 |
| run_02_seed4002 | **0.17** | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |
| run_03_seed4003 | 1.00 | 0.80 | 0.60 | **0.20** | 0.83 | 0.83 |
| run_04_seed4004 | 1.00 | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |

DAG topology is stable (5/5 valid). Note: these recorded metrics were produced by
the **stakeholder-aware** scorer in the artifacts; the mismatch analysis below
explains the remaining gaps.

## 4. Mismatch totals under the visibility contract (28 matched node-pairs)

| attribute | mismatches (was) | now |
| ----------- | ------------------ | ----- |
| actor | 10 | **10** |
| system | 6 | **2** |
| reads | 13 | **7** |
| writes | 21 | **13** |
| **total** | **50** | **32** |

Hidden + unset axes are **correct** and no longer counted (that is where the old
`system`/`reads`/`writes` totals shrank: approval system/reads and most hidden
writes are now correctly unset).

## 5. Classification (32 total, visibility-aware)

| class | count | share | description |
| ------- | ------- | ------- | ------------- |
| **A genuine agent extraction error** | 0 | 0% | visible attr, agent value genuinely wrong |
| **B evaluator semantic/representation mismatch** | 15 | 46.9% | visible attr, semantically same but different representation |
| **C agent epistemic error (asserted hidden)** | 8 | 25.0% | hidden attr asserted by the agent — fabrication, not synonymy |
| **D visible fact never obtained/recorded** | 9 | 28.1% | visible attr, agent left it unset/empty |
| E GT/modeling ambiguity after visibility | 0 | 0% | — |
| F ambiguous | 0 | 0% | — |

### Per attribute

**actor (10, all B):**

- 10 × `"I (the stakeholder)"` vs `"sales"` (runs 00: 5 nodes, 02: 5 nodes).
  Visible; the stakeholder IS the sales employee and says "I", so this is an
  evaluator representation gap (`norm_role` lacks first-person/self-reference
  aliases — has `interviewee`/`you`/`yourself`, not `i`/`me`/`stakeholder`).
  Runs 01/03/04 wrote the matching `"Interviewee"`.

**system (2, all D):**

- run_01 send step: `system=None`; Truth `email` (stakeholder said "by email").
- run_01 month-end: `system=None`; Truth `excel` (stakeholder said "as an Excel
  file").
- The old 4 × approval-`quoting` cases are **gone**: `ap.system` is hidden, and
  leaving it unset is now **correct**.

**reads (7, all C):**

- 7 × hidden `reads=[quote]` asserted by the agent as `quotation` /
  `quotation information` (me in runs 00/01/02/04; sq in runs 00/03/04).
  These are **epistemic errors**: the stakeholder never states these reads, so
  the agent should have left them unset. They are **NOT** `quote↔quotation`
  semantic targets anymore.

**writes (13):**

- 5 × **B**: create-quotation GT `quote` vs agent `quotation` (5/5 runs).
  Visible; the only remaining `quote↔quotation` **semantic** target.
- 1 × **C**: run_01 send `writes=["quotation"]` asserted for hidden
  `sent_quote` — epistemic error.
- 7 × **D**: `r.writes=["request"]` never recorded (5/5), month-end
  `excel_summary` never recorded (runs 02/04). Visible facts the agent did not
  record.

## 6. Recurring patterns (visibility-aware)

| pattern | frequency | class |
| --------- | ----------- | ------- |
| create-quotation write `quote` (agent `quotation`) not scored | 5/5 | B (semantic) |
| actor `"I (the stakeholder)"` vs GT `sales` | 2/5 runs (seeds 4000, 4002), 10 nodes | B (representation) |
| hidden send/month-end read `quote` asserted as `quotation`… | me 4/5, sq 3/5 | C (epistemic) |
| hidden `sent_quote` write asserted as `quotation` | 1/5 | C (epistemic) |
| `r.writes=["request"]` never recorded | 5/5 | D |
| month-end `excel_summary` never recorded | 2/5 (runs 02/04) | D |
| run_01 send/month-end `system` left None | 2 | D |

## 7. Contribution: Agent vs Evaluator vs visibility vs coverage

- **Genuine agent extraction errors: 0** under the current contract. No visible
  attribute carries a truly wrong non-empty value in these runs.
- **Evaluator semantic/representation gaps (B, 46.9%)**: exactly two
  deterministic, visible patterns remain — `quote↔quotation` on the
  create-quotation write (5 cases) and first-person actor labeling
  (`"I (the stakeholder)"` vs `sales`, 10 cases). These are the only candidates
  for semantic matching.
- **Epistemic errors (C, 25%)**: the agent asserted stakeholder-hidden reads/
  writes (8 cases). These must **not** be "fixed" by loosening the matcher —
  the correct agent behavior is to leave them unset, and the evaluator already
  penalizes assertions.
- **Not-obtained visible facts (D, 28%)**: visible `request`/`excel_summary`
  writes and two systems the stakeholder explicitly stated were simply not
  recorded. Extraction/recording behavior, not matching.

## 8. Goal-3 semantic targets (visibility-aware, recomputed)

Still real semantic-matching targets:

1. **`quote` ↔ `quotation` on the create-quotation write (`cq.writes`) — 5/5
   runs (5 cases, 15.6% of mismatches).** Visible, deterministic, explainable.
2. **`"I (the stakeholder)"` ↔ `sales` on visible actor axes — 10 cases
   (2/5 runs, 31.3% of mismatches).** Representation gap in `norm_role`
   (first-person self-reference), not a data-token issue.

Do **NOT** fix as semantic matching (now epistemic/visibility issues):

- hidden send/month-end `reads=[quote]` asserted as `quotation` (7 cases) — C;
- hidden `sent_quote` write asserted as `quotation` (1 case) — C;
- unrecorded visible `request` / `excel_summary` writes and `email`/`excel`
  systems (9 cases) — D (recording behavior, not matching);
- old "approval system/read not stated" cases (8) — resolved by visibility
  (hidden + unset is correct).

## 9. Recommended next change for Goal 3 (NOT implemented here)

Implement exactly two deterministic, explainable fixes, in order of value:

1. **Tiny scenario-local data-token normalization in `_data_recall`** for
   `quote ↔ quotation` (per the WordNet experiment, a small explicit alias map,
   **not** WordNet / embeddings / LLM judge). Recovers 5/5 create-quotation
   writes. Do **not** extend this to hidden reads — those are epistemic errors.
2. **First-person/self-reference actor aliases in `norm_role`**
   (`i`, `me`, `stakeholder`, `the interviewee` → the stakeholder's role).
   Recovers 10/10 visible actor cases.

These two cover **15 of 32 mismatches (46.9%)** — the entire visible semantic/
representation population. No other matcher change is justified by this data.

## 10. What should NOT be changed

- Do **not** relax `_data_recall` into "anything semantically matches" (would
  reward the 8 epistemic assertions).
- Do **not** add an embeddings / LLM judge.
- Do **not** change the node-matching gate (`_match_nodes`).
- Do **not** change the stakeholder fidelity prompt / Ground Truth / `known_info`.
- Do **not** change the StakeholderFilter visibility (already aligned with
  `known_info`).

## 11. Verification

- Analysis-only: no evaluator/agent/stakeholder/GT/tool/matching changes.
- `scripts/attribute_mismatch_inventory.py` regenerates both JSONs reproducibly
  (identical SHA on rerun).
- `make check-all` (ruff lint + format) clean.
- `pytest tests/test_domains/test_business_interview/` → **81 passed** (regression
  baseline; no behavior change).

## 12. Deliverables

- Helper: `scripts/attribute_mismatch_inventory.py` (now also emits the analysis)
- Inventory: `artifacts/business_interview_real_llm/attribute_mismatch_inventory.json`
- Summary: `artifacts/business_interview_real_llm/attribute_mismatch_analysis.json`
  (visibility-aware, supersedes the old 50/22 numbers)
- This report: `doc/business-interview-attribute-mismatch-analysis.md`
