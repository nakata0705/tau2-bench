# business_interview — Actor / System / Read / Write Mismatch Analysis (visibility-aware)

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** **Analysis + one evaluator change.** The analysis is based on the
already-committed DeepSeek artifacts (seeds 4000–4004); no new LLM runs. A
follow-up change (scenario-local `EvaluationSpec.data_expressions`, commit
"feat: add scenario-local data equivalence to business interviews") recovered
the one justified visible semantic target (`quote ↔ quotation` on
`cq.writes`); this report documents the regenerated, current-evaluator
numbers.

> **Supersedes the pre-visibility analysis.** The earlier version of this report
> (commit `be251e8` era) counted **50 mismatches / 22 evaluator-too-strict** under
> the old "all attributes visible" contract. Since commits `cafb944` and `6fa5d9e`
> the evaluator is **stakeholder-visibility-aware**: hidden attributes must remain
> unset, and asserting a hidden attribute is an **epistemic/fabrication error**,
> not an ordinary Truth-vs-agent semantic mismatch. This report re-classifies the
> same five saved real-LLM runs under the **current** contract and supersedes all
> earlier totals, patterns, and recommendations. Do not use the old numbers to
> design future evaluator work.

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
The inventory now passes the scenario spec into `_data_recall`, so a visible
`quote` / `quotation` pair is scored as a hit exactly as the live evaluator
scores it.

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
  token overlaps OR the Truth value is a substring of the agent value **OR the
  Truth value has a declared scenario-local expression that matches** (`spec.
  data_expressions`, e.g. `quote -> [quote, quotation]`). Raw `[a-z0-9]+`
  tokenization (no stemming, no global synonym map): `quote` vs `quotation`
  matches **only** because the quotation scenario declares it, and **only** on
  stakeholder-visible axes.

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
scorer (added later, commits `cafb944`/`6fa5d9e`). Any visibility-aware or
expression-aware numbers in this report are **replays of the saved DAGs under
the current evaluator** (`scripts/attribute_mismatch_inventory.py`,
`scripts/business_interview_reeval_smoke.py`,
`scripts/business_interview_data_expression_reeval.py`) — they are not what is
stored in the artifacts.

Under the current evaluator (replay), the only write-axis change is
`cq.writes`: `quote` vs agent `quotation` is now a hit via the scenario-local
expression, so replayed write correctness rises to **0.80 / 0.67 / 0.67 /
0.80 / 0.67** (the stored pre-visibility numbers above also conflate hidden
assertions with visible misses; see §4-§6 for the visibility-aware view).

## 4. Mismatch totals under the current evaluator (28 matched node-pairs)

| attribute | mismatches (was, visibility) | now (current evaluator) |
| ----------- | ------------------ | ----- |
| actor | 10 | **10** |
| system | 2 | **2** |
| reads | 7 | **7** |
| writes | 13 | **8** |
| **total** | **32** | **27** |

Hidden + unset axes are **correct** and no longer counted. The five
`cq.writes` `quote↔quotation` cases are **recovered** by the scenario-local
expression layer (they now score as hits, so they are no longer mismatches).

## 5. Classification (27 total, current evaluator)

| class | count | share | description |
| ------- | ------- | ------- | ------------- |
| **A genuine agent extraction error** | 0 | 0% | visible attr, agent value genuinely wrong |
| **B evaluator semantic/representation mismatch** | 10 | 37.0% | visible attr, semantically same but different representation |
| **C agent epistemic error (asserted hidden)** | 8 | 29.6% | hidden attr asserted by the agent — fabrication, not synonymy |
| **D visible fact never obtained/recorded** | 9 | 33.3% | visible attr, agent left it unset/empty |
| E GT/modeling ambiguity after visibility | 0 | 0% | — |
| F ambiguous | 0 | 0% | — |

### Per attribute

**actor (10, all B):**

- 10 × `"I (the stakeholder)"` vs `"sales"` (runs 00: 5 nodes, 02: 5 nodes).
  Visible; the stakeholder IS the sales employee and says "I", so this is an
  evaluator representation gap (`norm_role` lacks first-person/self-reference
  aliases — has `interviewee`/`you`/`yourself`, not `i`/`me`/`stakeholder`).
  Runs 01/03/04 wrote the matching `"Interviewee"`. Per the terminology-policy
  goal, first-person coreference is treated as an **interviewing
  responsibility** (the Agent asks for and records the business role), so
  `norm_role` is intentionally **not** changed.

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
  semantic targets — the expression layer explicitly does **not** apply to
  hidden attributes, and these remain failures.

**writes (8):**

- 1 × **C**: run_01 send `writes=["quotation"]` asserted for hidden
  `sent_quote` — epistemic error (no expression is declared for `sent_quote`,
  and hidden axes never use expressions).
- 7 × **D**: `r.writes=["request"]` never recorded (5/5), month-end
  `excel_summary` never recorded (runs 02/04). Visible facts the agent did not
  record.
- The previous 5 × **B** `create-quotation GT quote vs agent quotation`
  (5/5 runs) are **recovered**: the quotation scenario declares
  `data_expressions={"quote": ["quote", "quotation"]}`, a stakeholder-visible
  write, so the pair now scores as a hit.

## 6. Recurring patterns (current evaluator)

| pattern | frequency | class |
| --------- | ----------- | ------- |
| actor `"I (the stakeholder)"` vs GT `sales` | 2/5 runs (seeds 4000, 4002), 10 nodes | B (representation) |
| hidden send/month-end read `quote` asserted as `quotation`… | me 4/5, sq 3/5 | C (epistemic) |
| hidden `sent_quote` write asserted as `quotation` | 1/5 | C (epistemic) |
| `r.writes=["request"]` never recorded | 5/5 | D |
| month-end `excel_summary` never recorded | 2/5 (runs 02/04) | D |
| run_01 send/month-end `system` left None | 2 | D |

The former pattern `create-quotation write quote (agent quotation) not scored`
(5/5, B) is **gone** — recovered by the scenario-local expression.

## 7. Contribution: Agent vs Evaluator vs visibility vs coverage

- **Genuine agent extraction errors: 0** under the current contract. No visible
  attribute carries a truly wrong non-empty value in these runs.
- **Evaluator semantic/representation gaps (B, 37%)**: one deterministic
  visible pattern remains — first-person actor labeling (`"I (the
  stakeholder)"` vs `sales`, 10 cases). Per the terminology-policy goal this is
  intentionally **not** fixed in `norm_role` (first-person coreference is an
  interviewing responsibility; the Agent now asks the stakeholder's business
  role and records it).
- **Epistemic errors (C, 29.6%)**: the agent asserted stakeholder-hidden reads/
  writes (8 cases). These must **not** be "fixed" by loosening the matcher —
  the correct agent behavior is to leave them unset, and the evaluator already
  penalizes assertions. Replay confirms all 8 remain failures.
- **Not-obtained visible facts (D, 33.3%)**: visible `request`/`excel_summary`
  writes and two systems the stakeholder explicitly stated were simply not
  recorded. Extraction/recording behavior, not matching.

## 8. Remaining evaluator mismatch population

After the scenario-local `data_expressions` change, the replay of the saved
seed-4000..4004 DAGs shows:

1. **`"I (the stakeholder)"` ↔ `sales` on visible actor axes — 10 cases
   (2/5 runs, 37% of mismatches).** Representation gap in `norm_role`
   (first-person self-reference). Explicitly **out of scope** for the
   terminology goals: no first-person / global actor aliases are added.
2. **Epistemic errors (C, 8 cases)** — hidden reads/writes asserted as
   `quotation` (7 reads + 1 `sent_quote` write). These are **not** semantic
   targets; the correct fix is agent behavior (epistemic restraint).
3. **Not-obtained visible facts (D, 9 cases)** — recording behavior, not
   matching.

Do **NOT** fix as semantic matching:

- hidden send/month-end `reads=[quote]` asserted as `quotation` (7 cases) — C;
- hidden `sent_quote` write asserted as `quotation` (1 case) — C;
- unrecorded visible `request` / `excel_summary` writes and `email`/`excel`
  systems (9 cases) — D (recording behavior, not matching);
- old "approval system/read not stated" cases (8) — resolved by visibility
  (hidden + unset is correct).

## 9. Change implemented after this analysis

The only justified visible semantic target (`quote ↔ quotation` on
`cq.writes`) is implemented as a **scenario-local** deterministic equivalence:

- `EvaluationSpec.data_expressions: dict[str, list[str]]` — canonical Truth
  data value → accepted expressions; lives in the hidden, evaluator-only
  scenario spec (never a global alias table); canonical Truth values are
  unchanged.
- Used **only for stakeholder-visible reads/writes**; a hidden assertion
  (`Truth: quote`, agent `quotation`) still fails — semantic equivalence never
  overrides visibility.
- Baseline token/substring matching is preserved; no WordNet / stemming /
  embeddings / LLM judges / unrestricted synonymy.
- Quotation declares exactly `data_expressions={"quote": ["quote",
  "quotation"]}`. No `estimate` / `proposal` / `price sheet` / `document` /
  `offer` variants and no aliases for hidden `sent_quote` or hidden reads.

Replay results (seeds 4000-4004; `scripts/business_interview_data_expression_reeval.py`
→ `artifacts/business_interview_real_llm/data_expression_reeval.json`):

- `cq.writes` `quote↔quotation`: **5/5 recovered** (baseline 0.00 → 1.00).
- Hidden epistemic assertions: **8/8 remain failures** (plus 4/4 in the
  terminology-policy seed-5000/5001 runs).
- No axis changed outside `cq.writes` → **no new false positives**.
- Actor behavior unchanged (replays use the saved `final_dag` verbatim; no LLM
  calls).

## 10. What should NOT be changed

- Do **not** relax `_data_recall` into "anything semantically matches" (would
  reward the 8 epistemic assertions).
- Do **not** add an embeddings / LLM judge / WordNet / stemming.
- Do **not** add first-person / global actor aliases to `norm_role`.
- Do **not** change the node-matching gate (`_match_nodes`).
- Do **not** change the stakeholder fidelity prompt / Ground Truth / `known_info`.
- Do **not** change the StakeholderFilter visibility (already aligned with
  `known_info`).

## 11. Verification

- `scripts/attribute_mismatch_inventory.py` regenerates both JSONs reproducibly
  (identical SHA on rerun).
- `scripts/business_interview_data_expression_reeval.py` replays all saved
  seed-4000..4004 and seed-5000/5001 DAGs under baseline vs expression-aware
  contracts and emits `data_expression_reeval.json`.
- `make check-all` (ruff lint + format) clean.
- `pytest tests/test_domains/test_business_interview/` → **105 passed**
  (includes the scenario-local expression regression tests).

## 12. Deliverables

- Helper: `scripts/attribute_mismatch_inventory.py` (now also emits the analysis)
- Replay helper: `scripts/business_interview_data_expression_reeval.py`
- Inventory: `artifacts/business_interview_real_llm/attribute_mismatch_inventory.json`
- Summary: `artifacts/business_interview_real_llm/attribute_mismatch_analysis.json`
  (current-evaluator, supersedes the 32-mismatch numbers)
- Replay: `artifacts/business_interview_real_llm/data_expression_reeval.json`
- This report: `doc/business-interview-attribute-mismatch-analysis.md`
