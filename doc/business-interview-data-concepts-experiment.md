# business_interview — Agent-Local Data Concepts: Real-LLM Experiment

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** Real-LLM experiment (4 new quotation seeds, DeepSeek on both sides,
`temperature=0.0`), run under the **agent-local data concepts + hidden
stakeholder provenance** evaluator (`claims.py`).

## 1. Setup

- Script: `scripts/business_interview_real_llm_smoke.py --runs 2 --seed-base 6000`
  (re-run for 6002).
- Artifacts: `artifacts/business_interview_real_llm/run_00_seed6000.json`,
  `run_01_seed6001.json`, `run_00_seed6002.json`, `run_01_seed6003.json` (+
  `summary.json`).
- The reference `evaluation_criteria.actions` in `tasks.json` were regenerated
  to the concept tool API (`create_concept` + concept-id reads/writes) for all
  three tasks. The task reward basis is `ENV_ASSERTION` only; the generic
  gold-action DB replay remains inoperative for this domain (it never worked —
  historical runs all recorded `reward=0.0` from env assertions) and its
  warnings are cosmetic.

## 2. Results

| seed | nodeR | read | write | concept | unsup | structural |
| ------- | ------ | ----- | ------- | --------- | ------ | ----------- |
| 6000 | 0.83 | 0.40 | 0.80 | 1.0 | 0 | False |
| 6001 | 1.00 | 0.50 | 1.00 | 1.0 | 0 | False |
| 6002 | 1.00 | 0.50 | 1.00 | 1.0 | 0 | False |
| 6003 | 1.00 | 0.50 | 1.00 | 1.0 | 0 | False |

### Observed agent-local concepts (labels are the stakeholder's wording)

- 6000: `quotation_request`, `quotation`, `customer_information`,
  `pricing_information`
- 6001: + `quotation_summary` ("summary of the quotation information")
- 6002: `quotation_summary_excel` ("summary of the quotation information as an
  Excel file")
- 6003: `customer_information` ("customer's information")

## 3. What the runs show

- **The Agent creates and reuses stable local concepts.** `customer_information`
  is one concept id reused across `check_customer_info.reads` and
  `create_quotation.reads` in all 4 runs; `quotation` is reused across
  `create_quotation.writes` and `send_quotation.reads`.
- **Wording variation is resolved by the LLM, not evaluator aliases.** The
  evaluator contains no synonym table and never compares labels; the runs
  still bind correctly (`unsupported_concept_ref_count = 0` — every ref is
  grounded by hidden provenance). The evaluator demonstrably cannot tell
  "quotation" from "price quote" — binding comes from the private claim
  ledger.
- **Distinct objects stay distinct.** `quotation_request` (the request) is
  never merged with `quotation` (the document), and the month-end summary is
  its own concept — the hidden derivation (`claims.py` longest-phrase +
  stop-phrase matching) prevents the request/summary utterances from
  supporting the quote claim.
- **Concept integrity holds** (`concept_correctness = 1.0` in all runs): no
  local concept binds to two Truth concepts, and no Truth concept is split
  across unmerged agent ids.
- **Hidden assertions remain failures.** All runs assert `sq.reads` /
  `me.reads` as the `quotation` concept; those hidden axes still score 0
  (`read_correctness < 1.0`), exactly as required — valid provenance cannot
  rescue a hidden assertion.
- **Unsupported refs: none.** No run produced a ConceptRef the hidden
  provenance could not ground.

## 4. Remaining failures (unchanged classes, now concept-era)

- **Epistemic hidden assertions (C)**: `sq.reads` / `me.reads` asserted as
  `quotation` (4/4 runs) — the Agent should leave hidden reads unset.
- **Structural**: seed 6000 kept both `receive_request` and `record_request`
  nodes (one truth node left unmatched, nodeR 0.83) — node refinement
  behavior.
- write axes: 6000 write 0.80 (an extra `quotation_request` write on
  `record_request` for `r.writes` duplicated) — same refinement issue.

These are agent-behavior gaps, not evaluator gaps; the evaluator reports them
deterministically.

## 5. Files

- Runs: `run_00_seed6000.json` … `run_01_seed6003.json`
- Summary: `summary.json`
- Reference actions regenerated in `data/tau2/domains/business_interview/tasks.json`
