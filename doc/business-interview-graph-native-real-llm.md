# business_interview — current graph-native Truth reconstruction contract

This document describes the current v13 contract. The benchmark target is
always the Truth graph, not a private conversational-provenance graph.

## Architecture

- **Truth:** `BusinessProcessGraph` + `TruthConcept[]`.
- **Stakeholder:** `StakeholderKnowledgeGraph` + local concepts, used to
  constrain what the stakeholder simulator may reveal.
- **Agent:** `AgentGraph` + `AgentConcept[]`, evaluated directly against Truth.

Stakeholder sidecar annotations and dialogue events remain private and are
validated for simulator integrity. They are evidence/provenance diagnostics;
they do not decide whether an otherwise structurally valid Agent belief is
recorded and do not gate Truth-reconstruction quality.

## Agent evidence and epistemic states

`EvidenceRef` is optional diagnostic metadata. If an Agent supplies one, its
Observation id must exist and its shape must be valid. Missing, ambiguous or
nonmatching quotes and private semantic-slot binding are not recording gates.
This applies to `add_node`, `update_node`, `add_edge`, `update_edge`,
`record_dont_know`, `record_edge_condition_dont_know`, `record_absent` and
`record_edge_condition_absent`.

Agent slots have four states:

- `UNSET`: not investigated / no conclusion;
- `ConceptRef`: an asserted known value;
- `ABSENT`: an explicit Agent belief that the value is absent;
- `DONT_KNOW`: an explicit Agent belief that the value is unknowable.

Truth slots are two-valued: `ConceptRef` or `None` (canonical absence).
Scoring is explicit:

- Truth `ConceptRef` -> only a matching asserted Agent `ConceptRef` is
  correct; every other state is incorrect.
- Truth `None` -> only Agent `ABSENT` is correct; `UNSET`, `DONT_KNOW` and a
  `ConceptRef` are incorrect.

The same rule applies to reads/writes known-empty properties and unconditional
edge conditions. `episode_complete` is only a termination signal; it is not
benchmark success. `reconstruction_pass`, `structural_pass` and
`quality_pass` report Truth reconstruction separately.

The obsolete concept grounding/confirmation/unknown/disputed lifecycle is
not part of the runtime schema or policy.

## Deterministic matcher

Concept identity is matched per kind using a deterministic Unicode-aware
lexical signature, Dice similarity and global maximum-weight bipartite
matching. NFKC normalization and CJK character bigrams preserve Japanese and
other CJK support. A small low-information token set (`system`, `document`,
`information`, `quotation`, `process`, `data`, plus simple plurals) is removed
from partial lexical overlap. Exact canonical/local label equality is checked
first, preserving exact generic labels and short identifiers such as `CRM`,
`SAP` and `Excel` without allowing `system` to match `quoting system` merely
because of one broad word.

This remains lexical reconstruction, not language-independent semantic
understanding. Paraphrases that share no tokens can be missed, and locale
aliases work only when supplied by the scenario's local vocabulary. The
matcher uses no LLM, embeddings, web service or hidden provenance.

## Verification

The current deterministic coverage is:

- business-interview domain tests: **101 passed**;
- shared business-interview roundtrip tests: **15 passed**;
- combined focused deterministic total: **116 passed**;
- ruff: clean on the changed business-interview modules, smoke script and
  tests.

The policy/schema regression test checks that every tool identifier mentioned
in `policy.md` is exposed by the runtime schema and that removed lifecycle
tools are absent.

## Fresh quotation run

A fresh real run was executed after deterministic tests:

- artifact: `artifacts/business_interview_real_llm/run_00_seed9002.json`;
- private ledger: `artifacts/business_interview_real_llm/run_00_seed9002.private.json`;
- seed: `9002` (not the old seed `9000`);
- model: `openrouter/deepseek/deepseek-v4-flash-0731` on both sides;
- termination: `episode_complete`; `episode_complete` is **not** benchmark
  success;
- provider/runtime errors: `0` (`provider_errors: []`);
- Agent tool errors: `0`, with empty categories/by-tool/by-category maps;
- Agent calls: `32`;
- accepted Observations: `10`;
- elapsed time: `418.88` seconds;
- model refusals: `0`.

Evaluator metrics:

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
| fabricated node count | 0 |
| fabricated edge count | 2 |
| reconstruction_pass | false |
| structural_pass | false |
| quality_pass | false |

The artifact also records private-ID leakage (`[]`), evaluator evidence
metrics, raw public trajectory, provider errors, refusal diagnostics and LLM
call status. All 50 provider generation attempts in this run reported
`status=success`; the repeated cost-mapping log warnings were local accounting
warnings, not provider failures.

## Refusal classification

Normal stakeholder answers such as “I don't know” are not classified as model
refusals. Provider/runtime exceptions, malformed tool JSON and sidecar/plan
validation failures are recorded in their own error paths. A refusal finding
is emitted only for explicit model text such as “I can't assist with that
request”, with side, call index, model/provider, bounded response text,
preceding public prompt, provider moderation/content-filter metadata and retry
recovery status.
