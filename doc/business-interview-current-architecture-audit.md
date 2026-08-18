# business_interview — Current Architecture & Benchmark Contract Audit

**Date:** 2026-08-18
**Branch:** `business-interview`
**Type:** **Investigation only.** No production behavior, no Ground Truth, no
policy, no evaluator thresholds, no tests changed. This audit traces what
actually runs, inventories stale/dead code, audits GT ↔ stakeholder ↔ evaluator
consistency, documents reads/writes and matching, re-checks the real-run
mismatch classification, and lists invariants + prioritized findings.

Machine-readable companion:
`artifacts/business_interview_real_llm/current_architecture_audit.json`.

---

## 1. Runtime architecture (what actually runs today)

Verified from real imports/callers, not docstrings.

```
task (tasks.json)
  ├─ environment.get_environment()  -> InterviewDB + InterviewTools + policy.md
  │    BusinessInterviewEnvironment.on_message() -> db.messages (role+content ledger)
  ├─ stakeholder prompt = task.user_scenario.instructions.known_info / unknown_info
  │    (from tasks.json; NOT from scenario.StakeholderFilter — see §3/§5)
  ├─ agent (InterviewTools) captures Observations via observe_message()
  │    and builds an inferred BusinessDAG (nodes/edges/necessity/endpoints)
  ├─ EnvironmentEvaluator.calculate_reward (half-duplex)
  │    set_state + run_env_assertion x4 (all on predicted env):
  │      assert_finish_interview      -> protocol
  │      assert_dag_reconstructed     -> evaluate().structural_pass
  │      assert_necessity_handled     -> evaluate().necessity_pass
  │      assert_evidence_backed       -> evaluate().evidence_pass
  │    reward = product of the 4 ENV_ASSERTION booleans
  │    (reward_basis = ["ENV_ASSERTION"]; DB-hash compared but NOT in reward_basis)
  │    domain diagnostics surfaced via tools.get_eval_diagnostics() -> evaluate() dump
  └─ pass/reward
```

Key integration facts (verified):

- `src/tau2/registry.py` registers `business_interview` with
  `get_environment`, `get_tasks`, `get_tasks_split`.
- `src/tau2/cli.py` lists `business_interview` as a domain.
- The reward is **purely `ENV_ASSERTION`**: the DB end-state hash is computed but
  is not in `reward_basis`, so it is diagnostic only.
- `tools.get_eval_diagnostics()` returns `evaluate(db, truth, spec)` as JSON;
  it is called by `EnvironmentEvaluator.calculate_reward` (half-duplex). The
  full-duplex `FullDuplexEnvironmentEvaluator` does **not** call the domain
  diagnostics hook, so real-run `evaluator_metrics` come from the half-duplex
  path.
- The truth DAG and hidden `EvaluationSpec` come from `scenario.py`
  (`quotation_truth`/`quotation_spec`, `lab_sample_truth`/`lab_sample_spec`).
- The **stakeholder's knowledge comes from `tasks.json` `known_info`**, not from
  `Scenario.stakeholder` (`StakeholderFilter`). The filter is constructed and
  stored but **never applied at runtime**.

### Scenarios / tasks

- `tasks.json`: `quotation_workflow_1`, `quotation_workflow_1_ja`,
  `lab_sample_flow`. Each carries a reference action list (the "golden"
  trajectory) + 4 `env_assertions` + `reward_basis=["ENV_ASSERTION"]`.
- `split_tasks.json`: `base` (all 3), `base_en` (quotation EN),
  `base_ja` (quotation JA).

### Observation provenance

- `InterviewTools.observe_message` captures a real user message by stable id
  (`sm_N`) into `db.observations` as `Observation(id="obs_<turn>", ...)`.
- Authenticity is enforced at eval time: an Observation is authentic iff its
  text equals the `db.messages[turn]` user message (`evaluate`'s
  `invalid_observation_source_count`). Evidence hygiene checks that every
  asserted claim references a real, authentic Observation id — it does **not**
  re-interpret the Observation text semantically.

---

## 2. Active / legacy / dead inventory

Classification: **A** active runtime · **B** active but legacy-shaped ·
**C** test/diagnostic only · **D** apparently dead · **E** uncertain.

### A — active runtime

| item | role |
| ------ | ------ |
| `dag.py` `BusinessDAG`/`Node`/`Edge`/`InferredValue`/`Necessity`/`Observation`/`InterviewDB` | domain model |
| `evaluation.py` `evaluate()` + all metrics/gates | scorer + assertions |
| `scenario.py` truth/spec/filters/scenario map | truth + hidden spec |
| `tools.py` `InterviewTools` + `assert_*` + `get_eval_diagnostics` | agent tools + env hooks |
| `environment.py` `get_environment`/`get_tasks`/`get_tasks_split`/`on_message` | wiring |
| `aliases.py` `norm_role`/`norm_system` | role/system normalization (used by evaluation) |
| `concepts.py` `resolve_primitive` | generic primitive guess (used by `primitive_correctness`) |
| `stakeholder.py` `StakeholderFilter` | data model (constructed; `apply` unused at runtime — see D) |

### B — active but legacy-shaped

- `domain_concept_correctness` is literally `node_precision` in `evaluate()`
  (a legacy name surviving from the earlier concept-based design; it no longer
  represents a distinct concept metric).
- `primitive_correctness` is described in README/tests as **diagnostic**
  (unclassified not penalized) but primitives are also baked into the reference
  `actions` in `tasks.json`, so there are two representations of "expected
  primitive" (spec + golden actions) — a duplicated responsibility.
- `concepts.py` docstring says *"resolution returns None"* but the code returns
  `"unclassified"` (doc drift).

### C — test / diagnostic only

- `scripts/attribute_mismatch_inventory.py`
- `scripts/wordnet_attribute_match_experiment.py` (analysis-only)
- `scripts/business_interview_real_llm_smoke.py` (manual exploratory, live API)
- `scripts/business_interview_reeval_smoke.py` (manual re-eval)
- `scripts/stakeholder_fidelity_classify.py`
- `scripts/update_stakeholder_fidelity_prompts.py` (one-off generator for tasks.json)
- `tests/test_domains/test_business_interview/test_dag_business_interview.py` (67 tests)
- `StakeholderFilter.describe()` — used only by tests.

### D — apparently dead

- `aliases.value_signals()` and `_VALUE_VARIANTS` — **no caller** anywhere in
  `src/`, `tests/`, `scripts/`.
- `scenario.quotation_finance_filter()` — **no caller** (finance filter never
  wired into any scenario or test).
- `dag.InterviewResult` / `InterviewDB.interview_result()` — defined, **no
  runtime caller**.
- `concepts.py` `_PRIMITIVES` entries never referenced by any scenario spec:
  `move`, `notify`, `reconcile`, `record`, `reject`, `review`, `update`
  (scenarios use only approve/check/create/receive/send/transform).
- `__pycache__` compiled leftovers of removed modules — `ground_truth.pyc`,
  `data_model.pyc`, `semantic.pyc` (gitignored, not tracked; harmless).

### E — uncertain

- `aliases.py` module docstring references `semantic.py` and `ground_truth.py`,
  which **no longer exist** (removed in the open-world redesign). Stale doc.
- Version-header drift: `dag.py` says v3, `evaluation.py` v5, `concepts.py` v4,
  `scenario.py` v3, README v5 — files describe different versions.
- `policy.md` mentions a "stakeholder-truth consistency check" and
  `aliases` docstring says it is used by a "stakeholder-truth consistency
  check" — no such live module remains (removed `ground_truth.py`).

---

## 3. Ground Truth ↔ stakeholder ↔ evaluator matrix

For each scored fact: where GT defines it, whether the stakeholder actually
knows/provides it, whether a focused question could obtain it, whether it needs
an unstated modeling assumption, whether the evaluator scores it, and whether a
missing/wrong value fails the pass.

Legend: **obtainable** = stakeholder `known_info` actually states it;
**inferable** = implied but not explicit; **P0** = scored, pass-critical, and
**not** obtainable/inferable.

### quotation_workflow_1 (EN/`_ja`)

| scored fact | GT | stakeholder states it? | obtainable | scored | pass-critical | P0 |
| ------------- | ---- | ------------------------ | ----------- | -------- | --------------- | ---- |
| r action "receive quotation request" | scenario | yes ("record the request") | yes | yes | yes | |
| r actor = sales | scenario | yes ("you"/interviewee) | yes (alias gap "I (the stakeholder)" → **B**) | yes | yes | |
| r writes = [request] | scenario | yes | yes | yes | yes | |
| cc action "check customer in CRM" | scenario | yes | yes | yes | yes | |
| cc actor=sales, system=crm, reads=[customer] | scenario | yes | yes | yes | yes | |
| cq action "create quotation in quoting system" | scenario | yes | yes | yes | yes | |
| cq actor=sales, system=quoting, reads=[customer,pricing] | scenario | yes | yes | yes | yes | |
| cq writes=[quote] | scenario | yes (creates the quotation; wording "quote" vs "quotation" → **B**) | yes | yes | yes | |
| ap action "approve high-value quotation" | scenario | yes | yes | yes | yes | |
| ap actor=manager | scenario | yes ("approval by a manager") | yes | yes | yes | |
| ap necessity rationale = credit risk | scenario | yes ("for credit risk management") | yes | yes | yes | |
| **ap system = quoting** | scenario | **no** (never says where approval happens) | **no** | yes | yes | **P0** |
| **ap reads = [quote]** | scenario | **no** (never says approval reads the quote) | **no** | yes | yes | **P0** |
| **ap writes = [approval]** | scenario | **no** (never names an approval artifact) | **no** | yes | yes | **P0** |
| sq action "send quotation to customer" | scenario | yes | yes | yes | yes | |
| sq actor=sales, system=email | scenario | yes ("by email") | yes | yes | yes | |
| **sq reads = [quote]** | scenario | **no** (never states send reads the quote) | **no** | yes | yes | **P0** |
| **sq writes = [sent_quote]** | scenario | **no** (never names a sent_quote artifact) | **no** | yes | yes | **P0** |
| me action "send summary at month-end" | scenario | yes | yes | yes | yes | |
| me actor=sales, system=excel | scenario | yes ("Excel file") | yes | yes | yes | |
| **me reads = [quote]** | scenario | **no** (never states summary reads the quote) | **no** | yes | yes | **P0** |
| me writes=[excel_summary] | scenario | yes ("summary … as Excel file") | yes | yes | yes | |
| me necessity = all unset (unknown) | scenario | yes ("I don't know why") | yes | yes | yes | |
| edges e1..e6 + predicates (amount over/at-or-below 1,000,000; month-end) | scenario | yes | yes | yes | yes | |

### lab_sample_flow

| scored fact | GT | stakeholder states? | obtainable | scored | pass-critical | P0 |
| ------------- | ---- | --------------------- | ----------- | -------- | --------------- | ---- |
| 4 nodes n1..n4 + 3 edges + start/end | scenario | yes | yes | yes | yes | |
| reads/writes artifacts (sample, accessioned sample, seasoned chamber, conditioned sample, batch approval) | scenario | **partial** — artifacts mostly implied ("accession it (record as received)"), not explicitly named as objects | inferable | yes | yes | **P1** |
| system "environment chamber" for n2/n3 | scenario | yes | yes | yes | yes | |
| necessity | scenario (none) | n/a | yes | yes (trivially 1.0) | no | |

### Headline result (verified experimentally)

A **maximally faithful** agent that records **every** GT value (even ones the
stakeholder never states) **passes** `structural_pass`; but a **realistic
faithful** agent that records only what the stakeholder actually says **fails**
`structural_pass` (`system=0.83`, `read=0.67`, `write=0.67`), purely because of
the unstated approval-system / reads / writes facts. Because the reward is the
product of the 4 `ENV_ASSERTION`s, **any** single such mismatch zeroes the reward.

> **P0**: several scored, pass-critical GT facts (ap.system=quoting,
> ap.reads=[quote], ap.writes=[approval], sq.reads=[quote],
> sq.writes=[sent_quote], me.reads=[quote]) are **not obtainable from the
> stakeholder**. A compliant agent that "records only what the interviewee
> states" (policy rule #1) cannot satisfy them without fabricating.

---

## 4. Reads / writes audit

### Current meaning (from model + evaluator + tools + GT + tests)

- `Node.reads` / `Node.writes` are **free-text lists of data artifacts**
  (each an `InferredValue`). There is no data dictionary / ontology.
- Evaluator `_data_recall` scores a truth data item as hit if **any** raw token
  overlaps an agent token **or** the truth value is a substring of the agent
  value. Note: **any single shared token suffices** — this is more lenient than
  node matching (≥2 significant tokens).
- Tools `add_node`/`update_node` accept `reads`/`writes` as free-text string
  lists; the reference `tasks.json` actions hardcode the GT artifact names
  (`request`, `quote`, `approval`, `sent_quote`, `excel_summary`, `customer`,
  `pricing`).

### Specific artifacts

| artifact | status |
| ---------- | -------- |
| **request** | clearly specified (stakeholder says "record the request"). Agent sometimes omits r.writes → A/C. |
| **quote vs quotation** | **B** evaluator-too-strict. No shared token; `"quote"` is not a substring of `"quotation"`. Documented in the WordNet experiment. |
| **sent_quote** | GT write artifact on sq **never stated** by stakeholder → **P0** unobtainable + C modeling. Agent leaves empty or writes "quotation". |
| **approval** | GT write (`ap.writes=[approval]`) **never stated** → **P0**. |
| **excel_summary** | stakeholder states "summary … as Excel file"; agent "summary of quotation information" matches via `summary` token → largely recoverable; empty in runs 02/04 = A genuine miss. |
| **ap/sq/me reads=[quote]** | **never stated** → **P0**. |

### Classification of ambiguity

- **Clearly specified**: request; excel_summary; customer; pricing.
- **Undocumented but inferable**: lab artifact names (accessioned sample,
  seasoned chamber, conditioned sample, batch approval).
- **Ambiguous**: what exactly counts as a "read" vs a "write" for a given node;
  no formal definition in policy or model docs.
- **Internally inconsistent**: `_data_recall` accepts any single shared token
  (lenient) while `_match_nodes` requires ≥2 significant tokens (conservative) —
  different thresholds for different axes.

---

## 5. Matching / normalization responsibility map

| mechanism | location | callers | scope | main risk |
| ----------- | ---------- | --------- | ------- | ----------- |
| Node matching (`_match_nodes`) | evaluation.py | `evaluate()` | scenario-local (`spec.expressions`) | paraphrase miss (`quote`/`quotation`); weak single-token candidates rejected |
| Role normalization (`norm_role`) | aliases.py | `evaluate()` | global alias table | first-person `"I (the stakeholder)"` not aliased → **B** |
| System normalization (`norm_system`) | aliases.py | `evaluate()` | global alias table | missing variants (e.g. unstated approval system) |
| Reads/writes (`_data_recall`) | evaluation.py | `evaluate()` | raw tokens/substring | quote/quotation miss; lenient single-token overlap can over-match |
| Predicates (`_predicate_ok`) | evaluation.py | `evaluate()` | scenario-local (`spec.predicate_expressions`) | numeric/threshold variants, stopword dropping |
| Necessity (`_necessity_value_ok`) | evaluation.py | `evaluate()` | scenario-local (`spec.necessity_expressions`) | requires asserted (confidence>0); substring/token |
| Primitive (`resolve_primitive`) | concepts.py | `evaluate()` (primitive_correctness) | **global** generic primitives | docstring says None vs code returns unclassified; diagnostic |
| EvaluationSpec expressions | scenario.py | `evaluate()` via spec | scenario-local hidden | alias maintenance; not shown to agent |
| Evidence hygiene (`_evidence_metrics`) | evaluation.py | `evaluate()` | all referenced observation ids | only checks reference authenticity, not semantic support |

Verified `aliases.py` and `concepts.py` from their actual callers (`evaluate()`).
`value_signals` (aliases) has **no** caller. No aliases/WordNet/embeddings/LLM
judge added.

---

## 6. Real-run mismatch review (seeds 4000–4004)

> **Superseded numbers.** The table below is the **pre-visibility** classification
> (50 mismatches) from the original analysis. The current stakeholder-aware
> analysis (commits `cafb944`/`6fa5d9e`; see
> `doc/business-interview-attribute-mismatch-analysis.md`) re-classifies the same
> runs as **32 mismatches**: B=15 (visible semantic/representation: `quote↔
> quotation` on create-quotation write + first-person actor labeling), C=8
> (hidden attributes asserted — epistemic errors), D=9 (visible facts never
> recorded), A/E/F=0. Hidden+unset axes are correct and no longer count. Use the
> visibility-aware numbers for Goal 3, not the table below.

All 5 real runs have **reward = 0.0** because `structural_pass` is `False` in
every run. The original (pre-visibility) `attribute_mismatch_analysis.json`
classified 50 mismatches as:

| class | count | share |
| ------- | ------- | ------- |
| A genuine agent error | 4 | 8% |
| B evaluator too strict | 22 | 44% |
| C truth/modeling issue | 16 | 32% |
| D info not obtained | 8 | 16% |
| E ambiguous | 0 | 0% |

**Confirmation / no meaningful change:**

- `quote` vs `quotation` (B) still the dominant evaluator-too-strict pattern and
  still unscored by the current `_data_recall` (re-confirmed).
- Approval node `system="quoting"` left `None` (D) — the stakeholder never states
  it; confirmed **P0-unobtainable**.
- Approval/send reads `[quote]` left empty (D) — never stated; **P0**.
- Writes `sent_quote`/`approval` not recorded (C modeling) — confirmed.
- Actor `"I (the stakeholder)"` vs `sales` (B, norm_role first-person gap) —
  confirmed; this is **not** a WordNet/lexical issue.
- Month-end `excel_summary` empty in runs 02/04 (A genuine miss) — confirmed.

No reclassification is warranted. The documented A/B/C/D split stands; the new
finding is that the **D and part of C are P0** because they gate a hard-AND
reward.

---

## 7. Invariants (what is intentional and must be preserved)

Real current invariants:

1. **Open-world free-text actions** — the agent expresses actions in its own
   words; no fixed domain ontology is required of it.
2. **Same `BusinessDAG` class for Truth and inferred DAG** — no GT-only
   node/edge types.
3. **Authentic Observation provenance** — Observations are captured from real
   stakeholder messages (`observe_message`); text is immutable; agent cannot
   write arbitrary observation text.
4. **Unknown facts remain unknown** — unset values / confidence 0 are valid;
   `unclassified` primitive is a normal open-world state, not a failure.
5. **Deterministic evaluation** — no embeddings / LLM judge; reproducible.
6. **Hidden scenario-local `EvaluationSpec`** — evaluator-only scoring semantics
   (node/predicate/necessity expressions) never shown to the agent.
7. **Optional generic primitives; unclassified valid** — `primitive` is
   diagnostic, not a hard gate.
8. **Evidence hygiene checks reference authenticity, not semantic support** —
   the Observation body is not re-interpreted (stakeholder wording varies).

Accidental legacy (not invariants):

- `StakeholderFilter.apply()` is never used at runtime (stakeholder knowledge
  actually comes from `tasks.json` `known_info`) — two sources of truth.
- `domain_concept_correctness` name (legacy alias for node_precision).
- `concepts.py` "returns None" docstring vs actual `unclassified`.
- `aliases.py` references removed `semantic.py`/`ground_truth.py`.
- `InterviewResult`/`interview_result()` no runtime caller.
- Version-header drift (v3/v4/v5).

---

## 8. Test baseline

```
uv run pytest tests/test_domains/test_business_interview/
=> 67 passed, 2 warnings
```

No production behavior or tests changed.

---

## 9. Prioritized findings

### P0 — unfair / impossible benchmark contract

1. **Approval node `ap.system='quoting'` is scored pass-critical but the
   stakeholder never states where approval happens** → unobtainable.
2. **`ap.reads=[quote]`** scored pass-critical, never stated.
3. **`ap.writes=[approval]`** scored pass-critical, never stated.
4. **`sq.reads=[quote]` and `sq.writes=[sent_quote]`** scored pass-critical,
   never stated.
5. **`me.reads=[quote]`** scored pass-critical, never stated.
6. **Verified**: a faithful agent recording only `known_info` fails
   `structural_pass`; the hard AND-gate reward then zeroes the score. To pass,
   the agent must **fabricate** facts — directly violating policy rule #1
   ("Record only what the interviewee states").

### P1 — materially distorts scoring

1. `quote` ↔ `quotation` evaluator-too-strict (originally 22 B cases; under the
   current visibility contract only the create-quotation **write** remains a
   visible semantic target — 5/5 runs — while the hidden reads are now epistemic
   errors) masks real agent skill; any "quotation"-wording run is penalized
   (already covered by the WordNet experiment — recommended fix: tiny
   scenario-local alias, not WordNet).
2. Actor `"I (the stakeholder)"` vs `sales` not aliased (norm_role first-person
   gap) → 10 visible B cases.
3. Lab reads/writes artifact names are mostly implied, not stated → agent must
   guess.
4. Hard AND-gate reward: **no partial credit**; any single attribute mismatch
   zeroes reward, amplifying P0/P1.

### P2 — architectural debt

1. `StakeholderFilter` constructed & stored but never applied at runtime
   (stakeholder knowledge lives in `tasks.json` `known_info`) — two sources of
   truth that can drift.
2. Primitives duplicated: `concepts.resolve_primitive` + hardcoded in `tasks.json`
   golden actions + `spec.truth_nodes[].primitive`.
3. `_data_recall` (any 1 token) vs `_match_nodes` (≥2 significant tokens)
   threshold inconsistency.
4. `get_eval_diagnostics` wired only for half-duplex; full-duplex path doesn't
   surface domain metrics.
5. Real-run artifacts serialize `truth_dag`/`evaluation_spec_hidden` — convenient
   for analysis but the spec is supposed to be hidden.

### P3 — cleanup / docs

1. Dead code: `value_signals`/`_VALUE_VARIANTS`, `quotation_finance_filter`,
   `InterviewResult`/`interview_result()`, unused primitives
   (move/notify/reconcile/record/reject/review/update).
2. Stale docs: `aliases.py` references removed `semantic.py`/`ground_truth.py`;
   `concepts.py` docstring "returns None"; version-header drift (v3/v4/v5);
   README says v5.
3. Gitignored `__pycache__` leftovers of removed modules (ground_truth/data_model/
   semantic) — harmless but confusing.

---

## 10. Recommended next goal (do NOT implement here)

Make the benchmark contract fair, in order:

1. **Fix the P0 contract**: either (a) surface the currently-unobtainable GT
   facts (ap.system, ap/sq/me reads, ap/sq writes) in the stakeholder
   `known_info`, or (b) drop them from the GT / mark them unclassified/optional,
   so a faithful agent can pass without fabricating. Add a regression test that a
   faithful-known_info agent passes.
2. **Add partial credit** (relax the hard AND-gate) and/or fix the two cheapest
   evaluator-too-strict causes: `quote↔quotation` (tiny scenario-local alias)
   and first-person actor aliases in `norm_role`.
3. Reconcile the single-source-of-truth for stakeholder knowledge
   (StakeholderFilter vs tasks.json `known_info`).

This should be a dedicated follow-up goal with its own tests; it is **not**
implemented in this audit.

---

## 11. Verification

- Investigation only: no evaluator/agent/stakeholder/GT/policy/tool/test changes.
- `uv run pytest tests/test_domains/test_business_interview/` → **67 passed**.
- Machine-readable audit: `artifacts/business_interview_real_llm/current_architecture_audit.json`.

## 12. Deliverables

- This report: `doc/business-interview-current-architecture-audit.md`
- Audit JSON: `artifacts/business_interview_real_llm/current_architecture_audit.json`
