# business_interview domain

A minimal benchmark domain for evaluating agents that must **discover and
accurately represent a current business process by interviewing a stakeholder**,
without inventing facts.

This is intentionally a single-scenario domain: it exists to prove the pattern
end-to-end with one quotation-process interview. Adding a second scenario should
require only new task JSON entries (see "Adding a second scenario").

## Scenario (ground truth)

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

The interviewing agent is a plain `llm_agent` given a minimal BA policy
(`policy.md`) and four record tools:

| Tool | Purpose |
|------|---------|
| `record_fact` | A fact the interviewee stated about the current process |
| `record_exception` | An exception / variation the interviewee described |
| `record_uncertainty` | Something the interviewee does not know |
| `finish_interview` | Mark the interview complete (and end the conversation) |

## Files

```
src/tau2/domains/business_interview/
├── data_model.py   # InterviewDB: facts / exceptions / uncertainties / complete / summary
├── environment.py  # get_environment(), get_tasks(), get_tasks_split()
├── tools.py        # InterviewTools: record tools + deterministic assertion helpers
└── utils.py        # data paths
data/tau2/domains/business_interview/
├── policy.md       # minimal BA guidance for the interviewing agent
├── tasks.json      # the quotation-process interview task
└── split_tasks.json
```

## Evaluation semantics (fully structural — no LLM judge)

`tasks.json` sets `reward_basis: ["ENV_ASSERTION"]`. The final reward is the
product of five deterministic assertions run on the predicted environment after
replaying the agent's trajectory (`EnvironmentEvaluator`):

1. `assert_fact_recorded(all_of=["quotation", "system"])` —
   **normal process discovered**: some recorded fact mentions quotation +
   system.
2. `assert_exception_recorded(all_of=["excel", "accounting", "month"])` —
   **month-end exception discovered**: some recorded exception mentions Excel +
   Accounting + month.
3. `assert_uncertainty_recorded(all_of=["month", "excel"], any_of=[...])` —
   **unknown reason preserved as uncertainty**: some recorded uncertainty about
   the month-end Excel file expresses that the reason is unknown ("don't know",
   "unknown", "reason", "why", ...).
4. `assert_no_unsupported_rationale()` — **no invented facts**: no finding
   about the exception process (fact, exception, uncertainty, or summary that
   mentions the exception) contains an *assertive* rationale phrase (see
   `RATIONALE_SIGNALS` in `tools.py`), because the interviewee never provides
   one for the exception. Neutral "reason unknown" phrasing is not flagged,
   and purpose statements about the *normal* process (e.g. "reviews the
   quotation to ensure details are correct") are legitimate and not flagged.
5. `assert_interview_complete()` — the agent called `finish_interview`.

Each check appears per-assertion in the saved results (`reward_info.env_assertions`),
so a failed run shows exactly which criterion was missed. `actions` in
`tasks.json` is only a reference trajectory for diagnostics (partial action
reward); it is **not** part of `reward_basis`.

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

Pass `--language <lang>` (e.g. `ja`, `Japanese`, `es`) to have both the
interviewing agent and the stakeholder converse in that language:

```bash
tau2 run --domain business_interview --language ja \
  --agent-llm <model> --user-llm <model> --num-tasks 1 --num-trials 1
```

The language requirement is injected into both system prompts; the scenario
and policy stay in English internally. The env assertions match findings with
bilingual (English/Japanese) keyword synonyms (see `JAPANESE_KEYWORD_SYNONYMS`
in `tools.py`), so findings recorded in Japanese are scored the same way as
English ones. `assert_no_unsupported_rationale` also recognizes Japanese
assertive-rationale phrases (`RATIONALE_SIGNALS_JP`).

Note: whether the agent actually discovers and records the month-end exception
is model behavior — run multiple trials (`--num-trials`) for a reliable score.

## Verification

```bash
uv run pytest tests/test_domains/test_business_interview/
```

The tests cover the tools, every assertion (positive and negative), task/split
loading, a no-leak check on the scenario/policy data, and a full
orchestrator-level offline run (scripted agent + user, no LLM calls) that must
score 1.0.

## Design notes

- **No user tools.** The stakeholder is a plain conversational user; the domain
  passes `user_tools=None`. Everything checkable lives in the agent-side DB.
- **Stakeholder knowledge is separate from benchmark ground truth.** The user
  scenario contains only what the stakeholder knows (normal process + exception
  exists + reason unknown). The evaluation assertions and reference actions are
  the only place the expected findings are expressed, and neither is visible to
  the agent or the user simulator.
- **No new framework concepts.** The domain uses only standard τ-bench pieces:
  a `DB` subclass, a `ToolKitBase` with `@is_tool` writes, non-tool assertion
  helpers invoked via `env_assertions`, a `policy.md`, `tasks.json`, and a
  split file.

## Adding a second scenario

1. Add a task object to `tasks.json` (and to the `base` split) with its own
   `id`, `user_scenario` (stakeholder knowledge), and `evaluation_criteria`
   whose `env_assertions` express the new ground truth using the assertion
   helpers above (or new helpers added to `tools.py`).
2. Keep `reward_basis` as `["ENV_ASSERTION"]` unless the scenario needs a
   different gate.
3. Add tests in `tests/test_domains/test_business_interview/` mirroring the
   existing ones.

The generic BA guidance in `policy.md` is intentionally scenario-agnostic and
should not need changes for a second interview scenario.
