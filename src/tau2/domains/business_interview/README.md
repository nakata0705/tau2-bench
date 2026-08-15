# business_interview domain

A minimal benchmark domain for evaluating agents that must **discover and
accurately represent a current business process by interviewing a stakeholder**,
without inventing facts.

This is intentionally a small domain: it exists to prove the pattern end-to-end
and to demonstrate a **multi-axis, language-agnostic evaluation** that reports
*what* succeeded and *what* failed instead of collapsing everything into a
single scalar reward. Adding a third scenario requires only new task JSON
entries (see "Adding a scenario").

## Scenarios (ground truth)

Three scenarios are bundled, sharing one BA policy. The second is one level
harder than the first (the stakeholder holds a **hedged belief**); the third is
about **multiple coexisting exceptions with different epistemic rationales**.

### 1. `quotation_process_interview_1` — basic quotation interview

A sales employee is interviewed about the current quotation process:

1. A sales employee creates quotations.
2. The normal process uses the core business system.
3. At month-end **only**, the sales employee sends an Excel file to Accounting
   as an exception process.
4. The sales employee does **not** know why the month-end Excel process is
   necessary.

The stakeholder (user simulator) knows the normal process and the exception,
does not volunteer the exception unless asked about exceptions/variations/
month-end processing, and truthfully answers "I don't know" when asked for the
reason. The stakeholder never receives (and therefore never leaks) the hidden
ground truth — there is no rationale in this scenario at all.

### 2. `quotation_belief_uncertainty_1` — belief + uncertainty (Japanese focus)

A sales employee is interviewed about the same quotation process, but this time
the stakeholder holds a **hedged belief** about why the exception exists:

1. A sales employee creates quotations.
2. The normal process uses the core business system.
3. At month-end **only**, the sales employee sends an Excel file to Accounting
   as an exception process.
4. The sales employee **believes** (but is not certain) that the month-end
   Excel file may be due to some need on the Accounting team's side.
5. The real reason is **unconfirmed / unknown** even to the sales employee.

When run in Japanese the stakeholder can answer with nuance like
「たしか経理側の都合だったと思うんですけど、正確なところはちょっと分からないですね」.
The agent must: discover the exception, **keep the belief a belief** (ideally
recording it via `record_fact(..., epistemic_status="BELIEF")`), and **leave the
real reason as an uncertainty** (`record_uncertainty`). The task intentionally
imposes no fixed-string requirement, so any wording that carries the same
meaning is accepted.

### 3. `quotation_multi_exception_1` — two exceptions, different rationales

A sales employee is interviewed once, and a single stakeholder describes **two
distinct exceptions with different epistemic rationales**:

1. A sales employee creates quotations.
2. The normal process uses the core business system.
3. Exception A (`month_end_excel`): at month-end **only**, the sales employee
   sends an Excel file to Accounting. The **reason is UNKNOWN** even to the
   sales employee.
4. Exception B (`high_value_quote`): for quotations over 1,000,000 yen, an
   additional confirmation is performed. The **reason is a confirmed FACT**
   (credit risk management), which the sales employee knows.

The stakeholder volunteers neither exception unless asked (about exceptions /
special cases / variations / month-end / large amounts). When asked for the
month-end reason they say they do not know (never guessing); when asked for the
high-value reason they state it is for credit risk management.

The point of this scenario is **per-topic attribution**: the month-end Excel's
UNKNOWN must not be satisfied by, or conflated with, the high-value quote's FACT
(and vice versa). Findings are associated with a canonical **topic** (see
"Canonical topics" below). Evaluation is fully structural via canonical-topic
env assertions plus the multi-axis diagnostics; no LLM judge.

## Files

```
src/tau2/domains/business_interview/
├── data_model.py   # InterviewDB + canonical EpistemicStatus / finding models
├── environment.py  # get_environment(), get_tasks(), get_tasks_split()
├── semantic.py     # language-agnostic semantic evaluator (multi-axis metrics)
├── tools.py        # InterviewTools: record tools + deterministic assertions
└── utils.py        # data paths
data/tau2/domains/business_interview/
├── policy.md       # minimal BA guidance for the interviewing agent
├── tasks.json      # the two interview tasks
└── split_tasks.json
```

The interviewing agent is a plain `llm_agent` given a minimal BA policy
(`policy.md`) and four record tools:

| Tool | Purpose |
|------|---------|
| `record_fact(content, epistemic_status="FACT", topic?)` | A fact the interviewee stated; use `epistemic_status="BELIEF"` for hedged opinions; `topic` is the optional canonical business element the finding is about |
| `record_exception(content, topic?)` | An exception / variation the interviewee described |
| `record_uncertainty(content, topic?)` | Something the interviewee does not know |
| `finish_interview(summary?)` | Mark the interview complete (and end the conversation) |

## Canonical topics

A **topic** is a canonical, language-independent identifier for a business
element that a finding can be about. It is how the evaluator tells one exception
from another even when they coexist in one interview and have different
epistemic rationales.

- `month_end_excel` — a month-end Excel hand-off to Accounting (reason is
  UNKNOWN in this benchmark).
- `high_value_quote` — an additional confirmation for high-value quotations
  (reason is a confirmed FACT: credit risk).

The `topic` argument of the record tools is the **preferred**, language-
independent identity. As a backward-compatible fallback, when a finding has no
explicit `topic`, the evaluator associates it by matching the content against
each topic's small set of bilingual identity signals. The canonical topic (not
an ever-growing keyword dictionary) is what carries the attribution.

Each scenario declares which topics it *requires* (in `semantic.py`
`SCENARIO_TOPICS`): the two original scenarios require only `month_end_excel`;
`quotation_multi_exception_1` requires both. The required topics are the ground
truth about what the interview must cover, and live domain-side (never in the
agent-visible policy or scenario).

## Evaluation semantics

There are **two layers**: the existing scalar reward (unchanged) and a new set of
multi-axis diagnostics.

### Layer 1 — scalar reward (backward compatible)

`tasks.json` sets `reward_basis: ["ENV_ASSERTION"]`. The final reward is the
product of five deterministic assertions run on the predicted environment after
replaying the agent's trajectory (`EnvironmentEvaluator`):

1. `assert_fact_recorded(all_of=["quotation", "system"])` —
   **normal process discovered**.
2. `assert_exception_recorded(all_of=["excel", "accounting", "month"])` —
   **month-end exception discovered**.
3. `assert_uncertainty_recorded(all_of=["month", "excel"], any_of=[...])` —
   **unknown reason preserved as uncertainty**.
4. `assert_no_unsupported_rationale()` — **no invented facts**: no finding
   about the exception process asserts an *assertive* rationale (the
   interviewee never provides one). Findings explicitly recorded as
   `epistemic_status="BELIEF"` are exempt, because a stakeholder's own opinion
   kept as opinion is not an invented fact — but promoting it to a `FACT` is
   still flagged.
5. `assert_interview_complete()` — the agent called `finish_interview`.

Each check appears per-assertion in the saved results
(`reward_info.env_assertions`), so a failed run shows exactly which criterion
was missed. `actions` in `tasks.json` is only a reference trajectory for
diagnostics (partial action reward); it is **not** part of `reward_basis`.

### Layer 2 — multi-axis diagnostics (new, additive)

Beyond the scalar reward, the domain computes a structured
`InterviewEvaluation` (see `data_model.py`) and surfaces it through the generic
`get_eval_diagnostics` hook into `reward_info.info["diagnostics"]` (persisted in
the saved `SimulationRun`). This lets a human see **which axis** failed, not
only a number. The metrics are language-agnostic: they read the canonical
findings (with `epistemic_status`) and classify them with the bilingual signals
in `semantic.py`, so a Japanese and an English finding with the same meaning are
evaluated the same way.

| Metric | Meaning | Axis |
|--------|---------|------|
| `protocol_completed` | The agent actually called `finish_interview` (not merely said it would end) | Protocol |
| `normal_fact_recall` | A fact about the normal process was recorded | Discovery |
| `exception_recall` | An exception (month-end Excel, high-value quote, ...) was discovered & recorded | Discovery |
| `uncertainty_preserved` | An unknown exception reason was recorded as an uncertainty (UNKNOWN kept UNKNOWN) | Epistemic |
| `unsupported_fact_count` | # of findings recorded as FACTs that assert an unsupported rationale | Epistemic |
| `unsupported_rationale_detected` | Any finding (other than an explicitly-tagged BELIEF) asserts an invented rationale | Epistemic |
| `belief_promoted_to_fact` | A stakeholder belief/guess was recorded as a definitive FACT | Epistemic |
| `belief_handling` | Derived: `not belief_promoted_to_fact` | Epistemic |
| `topics` | Per-topic (per business-element) evaluation for the scenario's required topics: each topic reports `discovered`, `rationale_status` (FACT/BELIEF/UNKNOWN/NONE), `rationale_correct`, `unsupported_rationale` | Epistemic (per topic) |
| `protocol_pass` | Diagnostic top-level boolean = `protocol_completed` (named so it can be compared with `interview_quality_pass`) | Protocol |
| `interview_quality_pass` | Diagnostic top-level boolean: full discovery + epistemic quality (every required topic discovered with a correct rationale, nothing invented, no belief promoted), judged independently of protocol | Epistemic |

The diagnostics never change the scalar reward; they are an additional,
diagnosable view on the same outcome.

### Per-topic metrics (exact meaning)

For each required topic, `TopicEvaluation` reports:

- `discovered` — the exception for this topic was recorded.
- `rationale_status` — the epistemic status of the rationale recorded for this
  topic: `FACT` (asserted as a certainty), `BELIEF` (asserted as the
  stakeholder's opinion), `UNKNOWN` (the reason was preserved as an
  uncertainty), or `NONE` (no rationale recorded).
- `rationale_correct` — whether it matches the topic's ground truth:
  - unknown-rationale topic (`month_end_excel`): correct iff the reason was
    explicitly investigated and preserved as `UNKNOWN`, and nothing invented.
    Leaving it `NONE` (not checked) is **incorrect**: UNKNOWN means
    *investigated and unknown*, NONE means *not checked*;
  - known-rationale topic (`high_value_quote`): correct iff a FACT rationale
    carrying the expected value (credit risk) was captured, it was not
    downgraded to UNKNOWN, **and** no unsupported *additional* rationale was
    invented (e.g. `credit risk and tax reporting` fails).
- `unsupported_rationale` — whether an unsupported (invented / promoted)
  rationale was recorded for this topic.

Crucially, these are attributed **per topic**: the month-end Excel's UNKNOWN is
not allowed to satisfy the high-value quote's required FACT, and vice versa.
`interview_quality_pass` requires every required topic to be discovered with a
correct rationale, so a single misplaced UNKNOWN/FACT makes it False.

## Hardening: leak-free discovery & evaluator robustness

The benchmark is hardened so that the agent cannot be handed the answer and the
evaluator cannot be gamed by misattribution or over-claiming.

### No ground-truth leakage (P0)

The agent-visible policy (`policy.md`) and the agent-visible record/finish tool
descriptions contain **no hidden exception identity**: no canonical topic ids
(`month_end_excel`, `high_value_quote`) and no concrete hints (month-end Excel,
high-value quotation, credit risk). They describe only general BA behaviour
(investigate exceptions, record a canonical topic identifier consistently,
keep UNKNOWN unknown). The canonical topic ids, the per-topic ground truth
(`TopicSpec`), and the required-topics mapping (`SCENARIO_TOPICS`) live
**domain-side** in `semantic.py`/`data_model.py` — never in the policy or tool
descriptions. Guard tests assert the leakage-free policy and tool descriptions.

### JA canonical scenario (P1)

The EN and JA variants of a scenario are the **same canonical scenario** with
the same ground truth. `canonical_scenario_id()` strips the `_ja` suffix before
looking up required topics, so a JA multi-exception run in which `high_value_quote`
is completely missed still keeps that topic **required** — it cannot disappear
from the diagnostics and produce a false `interview_quality_pass`.

### UNKNOWN vs NONE (P1)

For an unknown-rationale topic, `UNKNOWN` (investigated, reason unknown) is
correct while `NONE` (rationale never checked / recorded) is **incorrect**. The
agent must actually ask and record the uncertainty.

### normal_fact_recall (P1)

A finding attributed to a canonical exception topic (e.g. the high-value
credit-risk FACT) is never counted as a normal-process fact, so recording only
exception rationales cannot set `normal_fact_recall`.

### Topic attribution robustness (P2)

The evaluator does **not** unconditionally trust an explicit `topic`. `topic_of_finding`
cross-checks the reported topic against the content: if the content unambiguously
identifies a *different* topic, the reported topic is not trusted and the finding
is left unassociated (so it satisfies neither topic). If the content is ambiguous,
no guess is made. This stops both a fabricated month-end rationale hidden under
`topic=high_value_quote` and a correctly-attributed reason being credited to the
wrong topic.

### Known-rationale precision (P2)

For the known-rationale topic, `rationale_correct` requires **both** the expected
rationale value (credit risk) captured **and** no unsupported *additional*
rationale. `credit risk and tax reporting` / `credit risk and audit
reconciliation` fail via a small bounded extra-rationale signal set (not an
unbounded forbidden-word dictionary).

### Epistemic terminology (P3)

`FACT` = the stakeholder asserted the claim as certain; `BELIEF` = the
stakeholder presented it as an opinion/guess; `UNKNOWN` = unconfirmed / not
known. Objective correctness is decided by the evaluator against the benchmark
ground truth, not by the recorded status alone. The `data_model` docstrings and
policy follow this consistently.

## Running

```bash
tau2 run --domain business_interview --agent-llm <model> --user-llm <model> \
  --num-tasks 1 --num-trials 1 --max-concurrency 1
```

The conversation ends when the interviewee replies with `###STOP###` after the
interviewer explicitly closes the interview (standard τ-bench user-stop flow;
the interviewer closes by calling `finish_interview` and thanking the
interviewee). The stakeholder is instructed not to end the conversation on
their own: if the interviewer only *says* they will finish without actually
closing, the stakeholder waits, so a compliant agent always gets a turn to
call `finish_interview` before the interview ends.

## Conducting the interview in another language

Each scenario is available in an **English** task (the base id) and a
**Japanese** task (id suffixed with `_ja`). The Japanese variant is a distinct,
pre-localized task: its `initial_state` (opening messages), persona and
stakeholder knowledge are already written in Japanese, so no `--language` flag
or core machinery is involved. The shared policy carries a neutral instruction
to conduct the interview in the same language the interviewee uses, and the
agent simply mirrors the (Japanese) stakeholder.

```bash
# English scenario
 tau2 run --domain business_interview --task-ids quotation_process_interview_1 \
  --agent-llm <model> --user-llm <model> --num-tasks 1 --num-trials 1

# Japanese scenario (pre-localized variant)
tau2 run --domain business_interview --task-ids quotation_process_interview_1_ja \
  --agent-llm <model> --user-llm <model> --num-tasks 1 --num-trials 1
```

The evaluation is identical for both: the env assertions and the semantic
evaluator match findings with bilingual (English/Japanese) signals, so findings
recorded in Japanese are scored the same way as English ones. The `base` split
contains all six tasks (three English + three Japanese). For EN-only / JA-only
comparisons, `base_en` and `base_ja` splits select just the English or Japanese
tasks respectively:

```bash
tau2 run --domain business_interview --task-split base_en ...
tau2 run --domain business_interview --task-split base_ja ...
```

Note: whether the agent actually discovers and records the month-end exception,
or keeps a belief a belief, is model behavior — run multiple trials
(`--num-trials`) for a reliable score.

## Verification

```bash
uv run pytest tests/test_domains/test_business_interview/
```

The tests cover the tools, every assertion (positive and negative), task/split
loading, a no-leak check on the scenario/policy data, a full orchestrator-level
offline run (scripted agent + user, no LLM calls) that must score 1.0, the
multi-axis diagnostics, and the **evaluator falsification suite**: the original
five deliberately-bad interviews (A normal, B exception missed, C rationale
invented, D belief promoted to fact, E protocol violation) each failing on the
expected axis, plus a **topic-scoped falsification suite** for the multi-exception
scenario covering:

- A. fully correct -> every axis passes;
- B. UNKNOWN assigned to the wrong exception (month-end reason asserted as credit
  risk, high-value reason left UNKNOWN) -> FAIL per topic (the most important
  falsification);
- C. month-end reason fabricated (audit) -> unsupported-rationale failure;
- D. high-value known reason downgraded to UNKNOWN -> high-value rationale FAIL
  even though month-end uncertainty is correct;
- E. one exception missed -> per-topic discovery failure;
- F. all findings correct but no `finish_interview` -> quality PASS, protocol FAIL.

and a demonstration that the scenario-level boolean is a false positive when the
known high-value rationale is simply never captured (per-topic evaluation catches
it).

## Design notes

- **No user tools.** The stakeholder is a plain conversational user; the domain
  passes `user_tools=None`. Everything checkable lives in the agent-side DB.
- **Stakeholder knowledge is separate from benchmark ground truth.** The user
  scenario contains only what the stakeholder knows. The evaluation assertions,
  reference actions, and the semantic evaluator are the only place the expected
  findings are expressed, and none are visible to the agent or the user
  simulator.
- **Canonical epistemic model.** Findings carry an `epistemic_status`
  (FACT / BELIEF / UNKNOWN / EXCEPTION) and an optional `subject`. The `subject`
  field is reserved for future multi-stakeholder / `conflicting_beliefs`
  scenarios but is not used today; the evaluator reasons only over
  `epistemic_status` and content.
- **Shared semantic path.** `semantic.py` centralizes the canonical concept
  classification (exception, rationale, uncertainty) used by both the existing
  assertions and the new diagnostics, so English and Japanese share one
  evaluator. The pre-existing keyword-synonym fallback is preserved as the
  matching primitive, not grown into an unbounded dictionary.
- **Minimal generic-core changes.** The only generic-core changes are a small
  `Environment.get_eval_diagnostics()` hook (returns `None` for domains that do
  not opt in) and threading that dict into `reward_info.info["diagnostics"]` in
  `EnvironmentEvaluator.calculate_reward`. Nothing else in τ-bench is affected.

## Adding a scenario

1. Add a task object to `tasks.json` (and to the `base` split) with its own
   `id`, `user_scenario` (stakeholder knowledge), and `evaluation_criteria`
   whose `env_assertions` express the new ground truth using the assertion
   helpers above (or new helpers added to `tools.py`).
2. Keep `reward_basis` as `["ENV_ASSERTION"]` unless the scenario needs a
   different gate.
3. Add tests in `tests/test_domains/test_business_interview/` mirroring the
   existing ones, including a falsification case if the new scenario introduces
   a new failure mode.

The generic BA guidance in `policy.md` is intentionally scenario-agnostic and
should not need changes for a new interview scenario.
