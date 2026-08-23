# business_interview — Real LLM (DeepSeek) Smoke Experiment Report

**Date:** 2026-08-18
**Branch:** `business-interview`
**Scenario:** `quotation_workflow_1` only
**This is an exploratory, manually-run smoke experiment. It is deliberately NOT
part of the pytest suite, `make test`, `make check-all`, or CI.**

---

## 1. Purpose

The goal was **not** to unit-test the evaluator, but to observe what actually
happens when a real Interview Agent talks to a real Stakeholder LLM against the
same hidden Ground Truth, repeatedly. Specifically:

- Does the stakeholder's natural-language phrasing vary run to run?
- Does the Interview Agent go beyond the happy path (ask about conditions,
  exceptions, periodic/month-end processing, necessity)?
- Does it fabricate unknown information?
- What does the final inferred DAG get right / miss / invent?
- Is evidence hygiene preserved?
- How do reward / correctness metrics move?

## 2. How to run it

The experiment is a small manual script (not imported anywhere):

```bash
uv run python scripts/business_interview_real_llm_smoke.py --runs 5 --seed-base 1000
```

It drives the **real tau2 pipeline** (registry → `build_text_orchestrator` →
`run_simulation`), not a replayed reference trajectory. After each run it reads
the live interview `InterviewDB` from the orchestrator environment, runs the
domain `evaluate()`, and dumps conversation / tool calls / observations / final
DAG / evaluator metrics / reward / errors to
`artifacts/business_interview_real_llm/`.

Nothing in the script is wired into tests, `Makefile` targets, or CI.

## 3. LLM configuration (DeepSeek)

| Role | Config value | Provider | Resolved API model |
|------|--------------|----------|--------------------|
| Interview Agent | `deepseek/deepseek-chat` | `deepseek` | `deepseek-v4-flash` |
| Stakeholder LLM | `deepseek/deepseek-chat` | `deepseek` | `deepseek-v4-flash` |

- `llm_args` = `{"temperature": 0.0}` for both sides (the framework default).
- A live connectivity check confirmed `deepseek/deepseek-chat` and
  `deepseek/deepseek-v4-flash` both resolve to API model **`deepseek-v4-flash`**
  in this environment.
- No API keys or secrets are stored in code, the report, or the artifacts. The
  key is read from the process environment (`DEEPSEEK_API_KEY`).

## 4. Runs executed

5 runs, seeds 1000–1004. Every run produced a full conversation and a final
inferred DAG.

| run | seed | termination | reward | quality_pass | structural | necessity | evidence | node R/P | edge R/P | fab nodes/edges |
|-----|------|-------------|--------|--------------|-----------|-----------|----------|----------|----------|-----------------|
| 00 | 1000 | user_stop | 0.0 | ❌ | ❌ | ❌ | ✅ | 1.0 / 1.0 | 0.50 / 0.50 | 0 / 3 |
| 01 | 1001 | too_many_errors | 0.0 | ❌ | ❌ | ❌ | ✅ | 1.0 / 1.0 | 0.50 / 0.43 | 0 / 4 |
| 02 | 1002 | user_stop | 0.0 | ❌ | ❌ | ❌ | ✅ | 1.0 / 1.0 | 0.00 / 0.00 | 0 / 5 |
| 03 | 1003 | user_stop | 0.0 | ❌ | ❌ | ❌ | ✅ | 1.0 / 0.75 | 0.33 / 0.33 | 2 / 4 |
| 04 | 1004 | user_stop | 0.0 | ❌ | ❌ | ❌ | ✅ | 1.0 / 1.0 | 0.83 / 0.83 | 0 / 1 |

**Averages** (5 runs): node_recall 1.0, node_precision 0.95, edge_recall 0.43,
edge_precision 0.42, predicate_correctness 0.72, actor_correctness 0.44,
system_correctness 0.30, read_correctness 0.30, write_correctness 0.14,
necessity_correctness 0.62, primitive_correctness 0.52.

### Note on the tau2 `reward` metric
All runs report `reward = 0.0`. This is the **standard tau2 reward pipeline**
(`reward_basis = ['ENV_ASSERTION']`, no env assertions for this domain), which
is **not** a meaningful signal for `business_interview`. The meaningful signal
is the domain evaluator (`evaluate()`): `quality_pass` / `structural_pass` /
`necessity_pass` / `evidence_pass` and the fine-grained correctness metrics.
`quality_pass` is the closest thing to a "success" boolean and it was `False`
in all 5 runs, primarily because the inferred DAG never reproduced the full
Ground Truth structure (see §6).

## 5. Raw artifacts

- Report: `doc/business-interview-real-llm-smoke-report.md`
- Raw runs: `artifacts/business_interview_real_llm/run_<NN>_seed<SEED>.json`
- Summary: `artifacts/business_interview_real_llm/summary.json`

Each run JSON contains: run id, seed, model/provider/llm_args, resolved model,
termination reason, standard reward, elapsed seconds, errors, full conversation
(roles + tool calls), captured observations, the final inferred DAG, the full
evaluator metrics dict, the Ground Truth DAG, and the (hidden) evaluation spec.

## 6. Discovery vs Ground Truth

Ground Truth (`quotation_workflow_1`): 6 nodes — receive request (r), check
customer in CRM (cc), create quotation (cq), **approve high-value (>¥1,000,000)**
(ap, rationale = credit risk), send quotation (sq), **month-end summary to
accounting** (me, **rationale unknown**). Edges: r→cc→cq; cq→ap (over 1M), cq→sq
(≤1M), ap→sq; cq→me (month-end). Start r; ends sq, me.

Per-run discovery:

| run | approval branch | approval rationale | month-end node | month-end rationale kept unknown | fabricated node(s) |
|-----|-----------------|--------------------|----------------|-----------------------------------|--------------------|
| 00 | ✅ discovered | ✅ credit-risk | ❌ missed | n/a | none (but see §7 mapping) |
| 01 | ✅ discovered | ✅ credit-risk | ❌ missed | n/a | none |
| 02 | ❌ missed | — | ❌ missed | n/a | `resolve_missing_info` |
| 03 | ✅ discovered | ✅ credit-risk | ✅ **discovered** | ✅ **left None (correct)** | `prepare_quotation` (orphan) + mapping artifact |
| 04 | ❌ missed (stakeholder denied) | — | ❌ missed | n/a | `resolve_customer_info` |

Key results:
- **Approval branch** was discovered in 3/5 runs (00, 01, 03) and the credit-risk
  rationale was correct in all three. It was missed in runs 02 and 04. In run 04
  the agent explicitly asked "any other steps or branches between creating and
  sending?" and the **stakeholder denied it** (see §9) — so the miss was not the
  agent's fault.
- **Month-end step** was discovered in only **1/5 runs (run 03)**, and only
  because the agent specifically asked about how the month-end summary fits the
  DAG and about special/recurring cases. In run 03 the agent correctly left the
  month-end rationale as **unknown (None)**, matching the Ground Truth "don't
  fabricate" expectation. In the other 4 runs the month-end node was never
  elicited.
- **Exception/error-handling nodes were fabricated** in runs 02 and 04: the
  stakeholder offhandedly mentioned "if customer info isn't found, we look into
  it", and DeepSeek promoted that into a real node (`resolve_missing_info` /
  `resolve_customer_info`). Relative to Ground Truth these are fabricated nodes,
  and in run 02 the fabricated branch pushed edge_recall to **0.0**.

## 7. Evaluator observations (found issues)

1. **`node_recall` is inflated by the greedy fuzzy matcher.** `node_recall` was
   1.0 in every run even though the agent never created a month-end node in runs
   00/01/02/04. Root cause: `_match_nodes` greedily assigns each agent node to a
   truth node by token overlap. E.g. run 00 mapped agent `manager_approval` → truth
   `me` (month-end) and `record_request` → truth `ap`, purely from token overlap
   on "send quotation", producing false-positive recall. So `node_recall=1.0`
   **overstates** real discovery. This is worth fixing in the evaluator (e.g.
   require a minimum semantic distance / actor+system agreement, or report
   "matched with low confidence").
2. **`edge_recall` can be boosted by node mis-mapping.** Run 04 reached edge_recall
   0.83 with the approval and month-end nodes absent; the edge hits came from the
   fuzzy node mapping realigning nodes, not from true branch discovery.
3. **`fabricated_necessity` triggered in all 5 runs.** The agent asks "why is this
   step necessary?" for **every** node (receive, check, create, send), and then
   sets `rationale`/`removal_impact` on those nodes. For truth nodes whose
   necessity is `None`/unknown this registers as fabricated. In run 03 the
   **month-end** rationale was correctly left None, but `fabricated_necessity`
   still fired because the mapped month-end node (or the approval node's
   removal_impact) got an asserted value. The current metric conflates "fabricated
   a necessity truth says is unknown" with "added informative detail to a node the
   truth doesn't score" — worth tightening.
4. **`actor/system/read/write` correctness is low (system ~0.30, write ~0.14).**
   DeepSeek often leaves `actor`/`system` blank or writes long natural-language
   strings that don't token-match the truth's short values (e.g. actor
   "the quotation handler (interviewee)" vs truth "sales"). The exact-match
   token comparison is brittle against the model's verbose phrasing.
5. **Evidence hygiene was clean in every run**: `evidence_pass=True`,
   `invalid_observation_reference_count=0`, `invalid_observation_source_count=0`,
   `node/attr/edge evidence coverage 1.0`. So DeepSeek consistently captured real
   stakeholder messages and only referenced authentic observations — the
   "hygiene" layer works as designed and was the one thing that passed everywhere.

## 8. Interview Agent question strategy

Strengths (observed across runs):
- Started with a broad "what triggers the process / intended outcome" question,
  then drilled into each step, actors, systems, and necessity.
- Consistently probed for **conditions / exceptions** ("are there situations where
  the process deviates?"). When asked this way (runs 00, 01, 03) the approval
  branch was surfaced.
- Run 03 was the clear best: it asked about special cases **and** specifically
  chased the month-end summary's role in the DAG and its necessity, discovering
  the month-end node and correctly keeping its rationale unknown.

Weaknesses:
- **Over-asks necessity on every node**, flooding the DAG with necessity
  rationales the Ground Truth doesn't require (drives `fabricated_necessity`).
- **Over-infers exception branches** from offhand stakeholder remarks (runs 02,
  04) → fabricated nodes, and in run 02 this destroyed edge_recall.
- **`observe_turn` turn-index confusion** (see §10): repeatedly calls
  `observe_turn` on non-user (tool) message indices, generating tool errors. In
  run 01 this accumulated to `too_many_errors` (10 tool errors) and the agent
  never called `finish_interview` (`protocol_completed=False`). This is the most
  damaging repeated failure mode in terms of run mechanics.
- In run 03 the agent created a coarse `prepare_quotation` node, later superseded
  it with sub-steps but left it as an orphan, producing an **invalid DAG**
  (`unreachable node: prepare_quotation`, `unreachable node: month_end_summary`).

## 9. Stakeholder simulator observations

- **Natural language varies run to run** even at temperature 0.0. Each run used
  different wording, length, and level of detail for the same facts (compare e.g.
  the necessity explanations in run 00 vs run 01 vs run 02). This confirms the
  design assumption that observation *text* should not be semantically gated.
- **Progressive disclosure works but is fragile.** The stakeholder generally
  withholds conditions until asked. When the agent asked about deviations
  (runs 00/01/03), approval + month-end came out. But when the agent only asked
  about "missing customer info" (runs 02/04), the stakeholder pivoted to that
  error case and never volunteered approval.
- **Stakeholder inconsistency on the approval branch:** In run 04 the stakeholder
  was asked directly whether there were other branches and answered **"No, there
  aren't any other steps or branches between creating and sending"** — a flat
  denial of the approval branch that exists in its `known_info`. This is a
  stakeholder-simulator reliability issue with DeepSeek and is the direct cause
  of the run-04 approval miss.
- The stakeholder does not volunteer the **month-end** step at all unless the
  agent asks about recurring/month-end activity (only run 03 did). This makes
  month-end recall depend entirely on the agent's question quality.

## 10. DeepSeek-specific tendencies (fact-based)

- High verbosity: actions, actors, systems, and necessity values are long
  prose, which hurts the exact token matching in the evaluator.
- **Tool turn-index confusion:** the agent calls `observe_turn` with wrong
  indices (off-by-one / counting tool messages as user turns). `list_stakeholder_messages`
  returns text + turn index, but the agent still mis-addresses turns. This caused
  2–10 tool errors per run and ended run 01 with `too_many_errors`.
- Tendency to **create speculative exception/error-handling nodes** from casual
  stakeholder remarks (2/5 runs), inflating fabricated nodes/edges.
- Tendency to **assign necessity to every node**, over-collecting beyond the
  truth's necessity scope.

## 11. What to fix next (prioritized)

1. **Evaluator node matching:** make `_match_nodes` less forgiving (require
   stronger agreement than token overlap, e.g. actor/system agreement or a
   minimum overlap threshold) so `node_recall`/`edge_recall` reflect true
   discovery and don't report 1.0 for absent nodes.
2. **Tool UX / agent prompt:** fix the agent's `observe_turn` turn-index handling
   (the leading cause of `too_many_errors`), e.g. clearer tool docs or turn-index
   semantics, so runs don't die on a technicality.
3. **Necessity gating:** discourage asking/recording necessity on every node, and
   make `fabricated_necessity` distinguish fabricated-unknown from extra detail.
4. **Actor/system/read/write scoring:** normalize verbose prose before matching
   (currently the model's phrasing rarely equals the truth's short labels).
5. **Stakeholder disclosure reliability:** the simulator should not flatly deny a
   branch that exists in `known_info` (run 04), and should surface the approval
   branch more reliably when the agent asks about deviations.

## 12. Scope / status

- No changes were made to the core framework, the domain evaluator logic, the
  stakeholder architecture, or the scenarios for this experiment (scope: observe,
  don't redesign).
- The experiment is isolated in `scripts/business_interview_real_llm_smoke.py`
  and its artifacts; it is **not** referenced by pytest, `Makefile`, or CI.
- The one script fix made during the run (passing the `EvaluationType.ALL` enum
  instead of a string) is internal to the smoke script, not to the framework.

## Verification

- `pytest tests/test_domains/test_business_interview/` — passed (no framework
  changes; existing unit tests unaffected).
- `make check-all` (ruff lint + format) — clean.
- No secrets committed; the report and artifacts contain only model/provider/
  config metadata and no API keys.

## 13. Follow-up cleanup: graph revision and call-level refusals

The follow-up cleanup adds the Agent-facing `remove_edge(edge_id)` write tool.
It requires an existing edge, deletes only that edge, preserves both endpoint
nodes and unrelated edges, and never consults hidden Truth. The policy now
requires removing obsolete shortcut relations after discovering intermediate
steps; `remove_node` remains the operation for an obsolete node. Deterministic
coverage includes endpoint/unrelated-edge preservation, unknown-edge failure,
coarse-shortcut revision, and policy/schema presence.

Refusal diagnostics now run in the shared `generate()` instrumentation path.
Each attempt records side, call name, global/per-call attempt indexes, model,
provider, status, retry flag, finish reason, bounded refusal/provider-refusal
text, and bounded moderation metadata. `model_refusal_count` and
`model_refusals` in real-run artifacts are call-level only, so private
Stakeholder plan/realization refusals remain visible even if a retry produces a
normal accepted public answer. Public trajectory refusal accounting remains a
separate compatibility diagnostic and is not added to the call-level count.
Uncertainty, malformed/validation/tool errors, and provider exceptions are not
refusals without explicit refusal evidence.

### Fresh quotation run (seed 9003)

Artifact: `artifacts/business_interview_real_llm/run_00_seed9003.json`

| metric | result |
|---|---:|
| termination_reason | `max_steps` |
| provider_error_count / call-level provider errors | `0 / 0` |
| tool_error_count / categories | `0 / []` |
| Agent generation attempts | `100` |
| accepted Observations | `9` |
| Stakeholder generation attempts | `17` (1 retry) |
| model_refusal_count (call-level) | `0` |
| node recall / precision | `1.0 / 1.0` |
| edge recall / precision | `1.0 / 1.0` |
| concept recall / precision | `0.7619 / 0.9412` |
| fabricated nodes / edges | `0 / 0` |
| reconstruction / structural / quality pass | `false / false / false` |
| elapsed | `491.34s` |

No actual model refusal was observed. The Agent made no `remove_edge` call in
this run; the final graph nevertheless had no fabricated edge (the refined
edges were built without an obsolete shortcut). The run exhausted `max_steps`
while repeatedly revisiting a rationale update, so the failed reconstruction
is reported as-is.
