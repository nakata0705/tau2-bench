# business_interview — Truth-reconstruction hardening report

## Contract

The benchmark evaluates `AgentConcepts + AgentGraph` directly against
`TruthConcepts + TruthGraph`. `StakeholderKnowledge` constrains the
conversation but is not the scored target. Conversational sidecar/provenance
is diagnostic only, and unsupported-but-correct reconstruction counts as
correct.

There is no concept grounding/confirmation/unknown/disputed lifecycle. The
runtime exposes only the simplified glossary and graph tools. The policy
regression test verifies that every tool identifier referenced by
`policy.md` exists in the runtime schema and that removed lifecycle names do
not return.

## Evidence semantics

`EvidenceRef` is optional diagnostic metadata. A supplied reference must have
valid shape and an existing Observation id, but quote exactness and private
semantic-slot binding never gate a structurally valid Agent belief. This
applies to node/property/edge references and to explicit `ABSENT` /
`DONT_KNOW` markers. Stakeholder sidecar validation remains strict because it
prevents the simulator from revealing facts outside its own knowledge.

## Evaluator semantics

For a Truth `ConceptRef` slot, only a matching asserted Agent `ConceptRef` is
correct. For a Truth `None` slot, only explicit Agent `ABSENT` is correct;
`UNSET`, `DONT_KNOW` and any `ConceptRef` are incorrect. Reads/writes
known-empty slots and unconditional edge conditions use the same rule.
Provenance/evidence-hygiene fields are reported but never gate
`reconstruction_pass`, `structural_pass` or `quality_pass`.

Concept identity uses deterministic Unicode/CJK-aware lexical signatures,
Dice overlap and global Hungarian matching. A small low-information-token
set blocks generic one-word false positives, while exact canonical/local
label equality preserves `CRM`, `SAP`, `Excel` and exact generic labels. This
is lexical matching, not semantic equivalence: scenario-provided locale
terms can support local-language matching, but arbitrary paraphrases may be
missed. No LLM, embedding, web service or hidden provenance is used.

## Verification

- Business-interview domain deterministic tests: **101 passed**.
- Shared roundtrip tests: **15 passed**.
- Combined focused deterministic total: **116 passed**.
- Relevant shared non-LLM tests: **68 passed**.
- Full attempted shared slice including `test_run.py`: **75 passed, 1
  xfailed, 8 failed** because the existing OpenAI-backed mock-user tests lack
  `OPENAI_API_KEY`; no business-interview assertion failed there.
- Ruff: clean on the complete repository.
- Primary LSP diagnostics: no findings on changed implementation/script
  files; auxiliary project-lens warnings are pre-existing protocol ellipses
  and the short variable name `anid`.
- `compileall` passed; no configured `pyright`, `mypy` or `ty` executable was
  available.

## Fresh real run

The required fresh quotation episode used seed `9002`, not seed `9000`:

- artifact: `artifacts/business_interview_real_llm/run_00_seed9002.json`;
- private ledger: `artifacts/business_interview_real_llm/run_00_seed9002.private.json`;
- termination: `episode_complete` (not benchmark success);
- provider/runtime errors: 0;
- tool errors: 0;
- Agent calls: 32;
- accepted Observations: 10;
- elapsed: 418.88 seconds;
- model refusal count: 0.

| metric | value |
| --- | ---: |
| node recall / precision | 1.0 / 1.0 |
| edge recall / precision | 0.833333 / 0.714286 |
| concept recall / precision / correctness | 1.0 / 0.954545 / 0.954545 |
| activity correctness | 1.0 |
| actor correctness | 1.0 |
| system correctness | 0.666667 |
| read correctness | 0.25 |
| write correctness | 0.166667 |
| rationale correctness | 0.166667 |
| condition correctness | 0.5 |
| fabricated node / edge count | 0 / 2 |
| reconstruction_pass | false |
| structural_pass | false |
| quality_pass | false |

The run is an honest non-success: it completed the episode but did not fully
reconstruct Truth. All 50 provider generation attempts were successful at the
LLM-call layer. The artifact and summary record provider errors, tool-error
categories/counts, calls, Observations, elapsed time and explicit refusal
findings. Normal stakeholder “I don't know” answers were not refusals; no
explicit safety refusal occurred.
