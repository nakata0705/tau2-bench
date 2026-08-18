# business_interview — Actor / System / Read / Write Mismatch Analysis

**Date:** 2026-08-18
**Branch:** `business-interview`
**Type:** **Analysis only.** No behavior change — Evaluator, Agent prompt,
Stakeholder prompt, Ground Truth, tool API, and matching logic are unchanged. No
new LLM runs. Uses the already-committed DeepSeek artifacts (seeds 4000–4004).

---

## 1. Analysis methodology

For each of the 5 artifacts (`run_00_seed4000` … `run_04_seed4004`):

1. Reconstruct the agent's inferred `BusinessDAG` from `final_dag` and the hidden
   Ground Truth from `quotation_truth()` / `quotation_spec()`.
2. Run the evaluator's **own** `_match_nodes` to get the agent→truth node mapping
   (so the inventory uses exactly the same matching the scorer uses).
3. For every matched node, compare `actor`, `system`, `reads`, `writes` between
   the agent node and the Truth node, and record whether the **current** evaluator
   (`norm_role` / `norm_system` / `_data_recall`) scores it as a hit.
4. Read the stakeholder Observations attached to the agent node (and the full
   conversation) to determine whether the stakeholder actually stated each
   attribute, then classify every mismatch.

Helper: `scripts/attribute_mismatch_inventory.py` (reads the artifacts, emits
`artifacts/business_interview_real_llm/attribute_mismatch_inventory.json`). No
scoring behavior changed.

## 2. Current evaluator semantics (as of this analysis)

- **Node mapping** (`_match_nodes`): an agent node is a candidate for a Truth node
  iff it shares ≥2 significant (stopword-filtered) tokens with a hidden expression
  or contains a whole expression; actor/system agreement is a **ranking bonus**,
  not a rescue. Assignment is greedy by `(overlap, bonus)`.
- **`norm_role` / `norm_system`** (`aliases.py`): substring matching against small
  alias tables. Roles: `sales` (sales, sales employee, sales rep, salesperson,
  quotation handler, …, interviewee, you, yourself, 営業, 自分, …), `manager`
  (manager, approver, supervisor, approval, authority, 承認者, 上司, …). Systems:
  `crm`, `quoting`, `email`, `excel` (with a few variants). If no variant matches,
  the **raw** label is returned (e.g. `analyst`, `i (the stakeholder)`), which
  then fails equality.
- **`_data_recall`** (reads/writes): for each Truth item, a hit if **any** raw
  token overlaps (`_tokens(a) & _tokens(b)` non-empty) OR the Truth value is a
  substring of the agent value. Tokenization is raw `[a-z0-9]+` (no stemming,
  no synonym map). So `quote` vs `quotation` → **miss** (`quote` is not a token or
  substring of `quotation`), while `sent_quote` vs `sent_quotation` → **hit**
  (shared token `sent`).

## 3. Per-run correctness (from `evaluator_metrics`)

| run | actor | system | read | write | nodeR | nodeP |
|-----|-------|--------|------|-------|-------|-------|
| run_00_seed4000 | **0.00** | 1.00 | 0.60 | 0.40 | 0.83 | 0.71 |
| run_01_seed4001 | 1.00 | **0.50** | 0.50 | **0.33** | 1.00 | 0.86 |
| run_02_seed4002 | **0.17** | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |
| run_03_seed4003 | 1.00 | 0.80 | 0.60 | **0.20** | 0.83 | 0.83 |
| run_04_seed4004 | 1.00 | 0.83 | 0.50 | **0.17** | 1.00 | 0.86 |

DAG topology is stable (5/5 valid). The scores below `1.0` come entirely from the
attribute mismatches analyzed here.

## 4. Mismatch totals (all 5 runs, 28 matched node-pairs)

| attribute | mismatches |
|-----------|-----------|
| actor | 10 |
| system | 6 |
| reads | 13 |
| writes | 21 |
| **total** | **50** |

## 5. Representative concrete examples

- **Actor**: agent node `create_quotation` in run_00/02 has
  `actor="I (the stakeholder)"`; Truth `actor="sales"`. The stakeholder says
  *"I'm the one who prepares the quotations"* / *"I create the quotation …"*. The
  stakeholder IS the sales employee, so `"I (the stakeholder)"` is semantically
  the sales actor, but `norm_role` has no first-person / self-reference aliases
  (`interviewee`, `you`, `yourself` are present; `i`, `me`, `stakeholder` are
  not) → returns `"i (the stakeholder)"` ≠ `"sales"`. (The same model wrote
  `"Interviewee"` in runs 01/03/04, which **does** map to `sales` → actor 1.0.)
- **System (genuine miss)**: run_01 `send_quotation` has `system=None`; Truth
  `system="email"`. The stakeholder explicitly said *"Once it's ready, I send it
  to the customer **by email**."* — the agent had the info and didn't record it.
  Same run: `month_end_summary` `system=None`; stakeholder said *"… as an Excel
  file"*.
- **System (not obtainable)**: the approval node has `system="quoting"` in the
  Truth, but the stakeholder only ever says *"… it needs approval from a manager
  before it can be sent … for credit risk management"* — the approval happening
  *in the quoting system* is never stated. Agent leaves it `None` (honest) in 4/5
  runs → scored 0.
- **Reads (synonym)**: Truth `reads=["quote"]`, agent `reads=["quotation"]` (or
  `["quotation information"]`). Same noun; `_data_recall` misses because
  `quote`/`quotation` share no token and `"quote"` is not a substring of
  `"quotation"`.
- **Writes (synonym)**: Truth `writes=["quote"]` on create-quotation; agent
  `writes=["quotation"]` → miss (5/5).
- **Writes (modelled artifact not verbalized)**: Truth `writes=["sent_quote"]`
  on send (and `["request"]` on record, `["approval"]` on approval). The
  stakeholder only says *"I send it to the customer by email"* / *"I record the
  request"* / *"needs approval"* — the GT treats these as output artifacts the
  stakeholder never names as objects. Agent records no write.
- **Writes (genuine miss)**: month-end `writes=["excel_summary"]` left empty in
  runs 02/04 although the stakeholder said *"I send a summary … as an Excel
  file"*.

## 6. Mismatch classification (50 total)

| class | count | share | description |
|-------|-------|-------|-------------|
| **A genuine_agent_error** | 4 | 8% | info was stated, agent missed it |
| **B evaluator_too_strict** | 22 | 44% | synonymous wording / alias gap the evaluator can't normalize |
| **C truth_or_modeling_issue** | 16 | 32% | GT models data artifacts the stakeholder doesn't verbalize |
| **D information_not_obtained** | 8 | 16% | GT attribute not present in stakeholder known_info |
| E ambiguous | 0 | 0% | — |

### Per attribute

**actor (10, all B):**
- 10 × `"I (the stakeholder)"` vs `"sales"` (runs 00: 5 nodes, 02: 5 nodes).
  B (evaluator alias gap for first-person self-reference), with an agent-labeling
  inconsistency note (3/5 runs used the matching `"Interviewee"`).

**system (6):**
- 2 × **A** (run_01 `email`, run_01 `excel` — stated but not recorded).
- 4 × **D** (approval `system="quoting"` not stated by stakeholder).

**reads (13):**
- 7 × **B** (GT `quote` vs agent `quotation`/`quotation information` — send and/or
  month-end).
- 2 × **C** (send node `reads` empty; stakeholder only says "send it").
- 4 × **D** (approval `reads=["quote"]` — manager-reviewing-the-quote never
  stated).

**writes (21):**
- 5 × **B** (create-quotation GT `quote` vs agent `quotation`).
- 14 × **C** (GT write artifacts `request` / `sent_quote` / `approval` not
  verbalized as outputs by the stakeholder).
- 2 × **A** (month-end `excel_summary` empty in runs 02/04 although the Excel
  file/summary was stated).

## 7. Recurring patterns

| pattern | frequency | class |
|---------|-----------|-------|
| create-quotation write `quote` (agent `quotation`) not scored | 5/5 | B |
| send / month-end read `quote` (agent `quotation`…) not scored | 5/5 | B |
| actor `"I (the stakeholder)"` vs GT `sales` | 2/5 runs (seed 4000, 4002); other 3/5 used matching `Interviewee` | B (alias gap + labeling inconsistency) |
| approval `system="quoting"` left None | 4/5 | D |
| approval read `quote` left empty | 4/5 | D |
| writes `request` / `sent_quote` / `approval` not recorded | request 5/5, sent_quote 5/5, approval 4/5 | C |

## 8. Contribution: Agent vs Evaluator vs Ground Truth vs coverage

- **Interview Agent extraction** is largely **sound**: only **~8%** of mismatches
  are genuine agent errors (a few stated attributes missed: send-email system,
  month-end excel system/write). DAG topology is stable.
- **Evaluator strictness** is the **single largest lever (44%)**: a deterministic
  synonym/stem normalization (`quote↔quotation`) alone would fix the recurring
  read/write misses (~24%), and first-person actor aliases would fix another ~20%.
  These are explainable, deterministic fixes — not embeddings / LLM judge.
- **Ground Truth modeling granularity (32%)**: the GT requires write data
  artifacts (`request`, `sent_quote`, `approval`) that the stakeholder's natural
  language never verbalizes as objects. This is a GT granularity choice.
- **Information coverage / scenario (16%)**: a few GT attributes (approval
  `system="quoting"`, approval read `quote`) are **not present in the stakeholder
  `known_info`**, so the agent cannot obtain them no matter how good it is.

## 9. Recommended next change (highest priority, NOT implemented here)

**Add a small deterministic data-token normalization to `_data_recall`** — a tiny
`quote ↔ quotation` (and related singular/word-form) normalization — plus
**first-person/self-reference actor aliases** (`i`, `me`, `stakeholder`, `the
interviewee` → the stakeholder's role) in `norm_role`. These are the two cheapest,
most explainable, and highest-yield fixes (≈44% of mismatches) and keep scoring
deterministic. Do **not** reach for embeddings / an LLM judge first.

Secondary (evaluator/GT-consistency) recommendations:
1. Reconcile GT attributes that are absent from the stakeholder `known_info`
   (approval `system`, approval read `quote`) — either surface them in the
   scenario or drop them from the GT.
2. Reconsider GT **write** granularity for artifacts the stakeholder never names
   (`request`, `sent_quote`, `approval`) — require only what the scenario states.
3. Optionally encourage cleaner actor labels in the agent prompt (role name vs
   first-person placeholder), but a deterministic alias fix is more robust.

## 10. What should NOT be changed yet

- Do **not** relax `_data_recall` into "anything semantically matches" — the
  objective is to keep deterministic, explainable matching, not fuzzy everything.
- Do **not** add an embeddings / LLM judge as the first fix.
- Do **not** change the node-matching gate (`_match_nodes`) for this — it is not
  the cause of the attribute misses.
- Do **not** change the stakeholder fidelity prompt / Ground Truth facts used to
  test it.

## 11. Verification

- Analysis-only: no evaluator/agent/stakeholder/GT/tool/matching changes.
- `scripts/attribute_mismatch_inventory.py` (new) parses all 5 artifacts and emits
  the inventory JSON.
- `make check-all` (ruff lint + format) clean.
- `pytest tests/test_domains/test_business_interview/` → **67 passed** (no
  behavior change; existing suite unaffected).

## 12. Deliverables

- Helper: `scripts/attribute_mismatch_inventory.py`
- Inventory: `artifacts/business_interview_real_llm/attribute_mismatch_inventory.json`
- Summary: `artifacts/business_interview_real_llm/attribute_mismatch_analysis.json`
- This report: `doc/business-interview-attribute-mismatch-analysis.md`
