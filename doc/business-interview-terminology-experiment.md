# business_interview — Terminology Alignment Real-LLM Experiment

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** **Focused real-LLM experiment.** Runs the existing smoke runner
(`scripts/business_interview_real_llm_smoke.py`, DeepSeek for both agent and
stakeholder) on `quotation_workflow_1` with the new terminology-alignment
policy/instructions. The purpose is to observe interview **behavior** only; no
evaluator tuning was performed and none should be derived from these runs.

---

## 1. Runs

| run | seed | termination | reward |
|-----|------|-------------|--------|
| run_00 | 5000 | user_stop | 0.0 |
| run_01 | 5001 | user_stop | 0.0 |

Artifacts: `run_00_seed5000.json`, `run_01_seed5001.json` (in
`artifacts/business_interview_real_llm/`). Rewards are 0.0 as before — this is
not a scoring experiment.

## 2. Observed terminology behavior

### Agent (interviewer)

- **Establishes the stakeholder role (both runs).**
  - run_00: *"Let me clarify your role first. You mentioned 'I receive the
    quotation request myself' — what is your role in this process?"* →
    stakeholder: *"I'm a sales employee here."* → DAG actor is consistently
    **"Sales employee"** (maps to GT `sales` via `norm_role`, actor=1.0).
  - run_01: *"Let me clarify who you are in this process. You mentioned 'I' —
    what is your role in this process?"* → same, actor=1.0.
  - This confirms the first-person → business-role behavior works without
    forcing the stakeholder to stop using "I".
- **Uses "quotation" consistently** in actions and writes (the stakeholder's own
  term, also confirmed by the role/term clarification). The DAG records
  `writes=['quotation']` on create-quotation and send in both runs — consistent
  terminology, but it remains a **visible** `quote↔quotation` representation gap
  vs the GT label (Goal-3 semantic target, unchanged).
- **No derived-artifact invention**: the month-end node is recorded with the
  stakeholder's own phrasing ("send a summary ... as an Excel file") rather than
  the GT artifact name `excel_summary`; the agent does not invent a "seasoned
  chamber"-style artifact.
- **One short clarification per concept**: role (once), month-end connection
  (once, answered "I don't know" and left unconnected), pricing origin (once).
  No repeated terminology interrogation was observed for the same concept.
- **Residual repetition (not terminology)**: the agent repeatedly asks the same
  *necessity* question across nodes ("what would happen if...") — the
  stakeholder even noted the repetition once. This is a questioning-economy
  issue, not a terminology-consistency issue.

### Stakeholder

- Honors the agreed "quotation" vocabulary naturally (it is already its own
  word).
- Answers ambiguous identity questions truthfully ("I don't know how it connects
  ... beyond that", "I don't know why or how it's linked") rather than
  fabricating.
- Does not reveal hidden GT vocabulary (`sent_quote`, `excel_summary`,
  `quoting system` never appear in the conversation).

## 3. DAG-quality observations under the visibility contract

| issue | run_00 | run_01 | class |
| ------- | -------- | -------- | ------- |
| `create_quotation.writes` `quotation` vs `quote` | yes | yes | B (visible semantic, Goal-3) |
| actor "Sales employee" vs `sales` | ok (1.0) | ok (1.0) | — |
| `send_quotation.system` left None (email stated) | yes | yes | D (visible fact not recorded) |
| month-end `system` left None (excel stated) | yes | yes | D |
| `record_request.writes`/`receive_request.writes` empty | yes | yes | D (visible `request` not recorded) |
| hidden `sq.writes` asserted as `quotation` | yes | yes | C (epistemic) |
| hidden `ap.reads` asserted as `quotation` | yes | no | C (epistemic) |
| hidden `me.reads` asserted as `quotation information` | yes | no | C (epistemic) |

The terminology discipline improved **visible actor representation** (both runs
score actor=1.0 by resolving "I" to the business role) but did not eliminate the
**hidden-attribute assertions** (epistemic errors C) or the **visible
not-recorded facts** (D). Those are separate behaviors to address later; they are
NOT terminology problems and must not be "fixed" by loosening the matcher.

## 4. What this experiment does and does not show

- Shows: role establishment via first-person clarification; consistent use of an
  agreed business-object term; no GT vocabulary leakage; truthful
  don't-know answers on ambiguous identity.
- Does not show: any scoring improvement (not the goal), and it does **not**
  justify evaluator changes beyond what shipped. The visible `quote↔quotation`
  write mismatch (5/5 runs on seeds 4000–4004; observed again here) was the
  Goal-3 semantic target and is now resolved by the scenario-local
  `EvaluationSpec.data_expressions` exact-equivalence layer (see
  `business-interview-attribute-mismatch-analysis.md` §9).

## 5. Files

- New artifacts: `run_00_seed5000.json`, `run_01_seed5001.json`,
  `summary.json` (updated)
- This analysis: `doc/business-interview-terminology-experiment.md`
