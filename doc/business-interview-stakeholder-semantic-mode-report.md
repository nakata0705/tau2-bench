# business_interview — stakeholder semantic mode and Agent/Truth split

## Stakeholder-side semantic fidelity

The stakeholder simulator has a private two-phase response contract:

1. It plans semantic addresses and modes from its own
   `StakeholderKnowledge`.
2. The environment validates the plan and realized sidecar against that
   knowledge, including exact public-text spans and mode compatibility, before
   accepting the public message.

This strictness protects simulator integrity: a stakeholder cannot reveal
Truth-only information or claim a known value as `dont_know`. It is not an
Agent reconstruction gate. The accepted public utterance becomes one
immutable environment-owned Observation.

The canonical resolver maps:

| resolved stakeholder value | sidecar mode |
| --- | --- |
| `ConceptRef` | `value` |
| `None` | `absent` |
| `DONT_KNOW` | `dont_know` |
| node/edge existence | `exists` |
| local knowledge concept | `mention` |

## Agent-side evidence semantics

The Agent builds its own glossary and graph. `EvidenceRef` is optional
metadata for diagnostics. When supplied to Agent-facing tools, its
Observation id must exist and its shape must be valid; exact quote matching,
sidecar semantic-slot binding, and private Agent-node/edge binding do not
determine whether a structurally valid Agent belief is recorded.

This applies equally to concept/property references, edge evidence,
`ABSENT` markers and `DONT_KNOW` markers. The private sidecar remains useful
for simulator-integrity and evidence-hygiene metrics, but unsupported yet
Truth-correct reconstruction is accepted by the evaluator.

## Truth-based scoring

Truth is complete canonical data (`ConceptRef | None`). The Agent has
`UNSET`, `ConceptRef`, `ABSENT` and `DONT_KNOW` slots. Scoring is not based on
stakeholder mode:

- Truth `ConceptRef`: only a content-matching asserted Agent `ConceptRef` is
  correct.
- Truth `None`: only explicit Agent `ABSENT` is correct; `UNSET`,
  `DONT_KNOW` and `ConceptRef` are incorrect.

Reads/writes known-empty slots and unconditional edge conditions follow the
same rule. `knowledge_coverage` and evidence-hygiene fields remain
informational diagnostics.

## Regression coverage

The deterministic suite covers mode compatibility on the stakeholder side,
Truth/Agent state separation, policy/runtime tool-name consistency, optional
unbound Agent evidence, malformed tool-argument recovery, generic lexical
matcher adversaries, Japanese matching, and Agent-local id invariance. The
current focused total is **116 passed** (101 domain + 15 roundtrip tests).

The fresh quotation run (seed 9002) recorded `model_refusal_count: 0`,
`provider_error_count: 0` and `tool_error_count: 0`. Its full metrics are in
`artifacts/business_interview_real_llm/summary.json`; its termination was
`episode_complete`, but `reconstruction_pass` and `quality_pass` were false.
