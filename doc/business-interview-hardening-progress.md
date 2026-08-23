# business_interview — Truth-reconstruction hardening final report

This is the final handoff for the Truth-reconstruction hardening work on the
`business-interview` branch. The current contract is
`AgentConcepts + AgentGraph` evaluated directly against
`TruthConcepts + TruthGraph`; conversational provenance is diagnostic only.

## Completed changes

- Agent-facing tool docs, policy and runtime now agree that `EvidenceRef` is
  optional diagnostic metadata. Supplied Observation ids must exist, but exact
  quotes, sidecar annotations and private semantic-slot binding never gate a
  structurally valid Agent belief. This covers node/property/edge references,
  `ABSENT` and `DONT_KNOW` markers, and terminology bookkeeping.
- Evaluator documentation states the asymmetric Truth rule explicitly:
  Truth `ConceptRef` requires a matching Agent `ConceptRef`; Truth `None`
  requires explicit Agent `ABSENT`; `UNSET`, `DONT_KNOW` and `ConceptRef` are
  wrong for Truth absence. Reads/writes known-empty slots and unconditional
  edge conditions use the same rule.
- The deterministic matcher retains Unicode/CJK tokenization, Dice scoring
  and global Hungarian matching, but removes a small set of low-information
  generic words from partial overlap. Exact canonical/local labels still
  preserve `CRM`, `SAP`, `Excel` and exact generic labels. No LLM, embedding,
  web service or hidden provenance is used.
- Refusal diagnostics distinguish normal “I don't know” answers, provider or
  runtime errors, formatting/validation failures and explicit model refusal
  text. A refusal record includes side, call index, model/provider, bounded
  response, preceding public prompt, moderation metadata and retry recovery.
- The policy/runtime tool-name regression test and optional-evidence runtime
  regression coverage are retained and expanded.

## Verification

- Domain deterministic tests: **101 passed**.
- Shared business-interview roundtrip tests: **15 passed**.
- Combined focused total: **116 passed**.
- Relevant shared non-LLM tests (`test_environment.py`,
  `test_evaluate_trajectories.py`, `test_tasks.py`, `test_results_format.py`):
  **68 passed**.
- Full attempted shared slice including `test_run.py`: **75 passed, 1
  xfailed, 8 failed**; all eight failures are existing OpenAI-backed mock-user
  tests blocked by missing `OPENAI_API_KEY`, not business-interview failures.
- Ruff: clean on the complete repository.
- Primary LSP diagnostics: clean on changed implementation/script files;
  auxiliary project-lens warnings are pre-existing protocol ellipses and the
  short variable name `anid`.
- `compileall` passed; no `pyright`, `mypy` or `ty` executable is configured in
  the repository environment.

## Fresh real quotation run

- seed: `9002` (fresh; not the old `9000` artifact);
- artifact: `artifacts/business_interview_real_llm/run_00_seed9002.json`;
- private ledger: `artifacts/business_interview_real_llm/run_00_seed9002.private.json`;
- termination reason: `episode_complete`;
- provider/runtime errors: `0`;
- tool errors: `0` (no categories, by-tool counts or by-category counts);
- Agent calls: `32`;
- accepted Observations: `10`;
- elapsed: `418.88` seconds;
- model refusal count: `0`;
- normal stakeholder “I don't know” answers: not refusals;
- all 50 LLM generation attempts had status `success`.

| metric | value |
| --- | ---: |
| node recall / precision | 1.0 / 1.0 |
| edge recall / precision | 0.833333 / 0.714286 |
| concept recall / precision / correctness | 1.0 / 0.954545 / 0.954545 |
| activity / actor / system correctness | 1.0 / 1.0 / 0.666667 |
| read / write / rationale correctness | 0.25 / 0.166667 / 0.166667 |
| condition correctness | 0.5 |
| fabricated node / edge count | 0 / 2 |
| reconstruction_pass | false |
| structural_pass | false |
| quality_pass | false |

`episode_complete` is termination only, not benchmark success. The fresh
trajectory is a non-success because edge, system, read, write, rationale and
condition reconstruction was incomplete; the artifact records the complete
metrics and diagnostics without hiding that result.

## Known limitations

The matcher is lexical rather than semantic. It can miss paraphrases that
share no canonical/local tokens, and Japanese/local-language equivalence is
supported only through deterministic CJK signatures and scenario-provided
locale terms. The fresh run remains an exploratory quality sample, not a
claim that the LLM reconstructs every Truth graph successfully.
