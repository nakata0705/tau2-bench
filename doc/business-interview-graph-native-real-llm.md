# business_interview — graph-native semantic model (v10/v11, real-LLM validation)

Goal report for "replace TruthClaim semantics with graph-native semantic IDs
and redesign StakeholderKnowledge as an explicit stakeholder-world graph".

All deterministic: **no evaluator semantic NLP, no aliases, no embeddings, no
label matching**. No backward compatibility required; `TruthClaim`,
`build_claims`, claim catalogs, `visible_claim_ids` and
`StakeholderSemanticAssertion` are gone.

## 1. Truth = Graph + Concepts

`TruthClaim`/`build_claims()`/claim catalogs/`visible_claim_ids` deleted
(`claims.py` removed). Truth is only a `BusinessProcessGraph` (nodes, edges,
start/end) plus `TruthConcept[]` (`{id, kind, description, canonical_terms}`;
descriptions describe the concept itself, never workflow-position facts).

**Stable semantic IDs** for every addressable element (never list indexes;
survive reordering):

    node:<id> | node:<id>:activity|actor|system|rationale
    node:<id>:reads|writes          (whole-property slots)
    node:<id>:reads:<kcid>          (per-element; one per reads/writes ref)
    edge:<id> | edge:<id>:condition

`graph_semantic_ids()` derives the ID set for any graph; reordering a reads
list does not change it.

## 2. StakeholderKnowledge = StakeholderKnowledgeGraph + concepts

`project_knowledge(truth, filter)` produces the stakeholder's world model:

- unknown nodes/edges are **removed** — never shortcut edges (test proves
  A->B->C with unknown B yields no A->C);
- known property -> `ConceptRef` (into the stakeholder's own concept ids);
- known absent -> `None`;
- unknown property of a known element -> `DONT_KNOW` (singleton, distinct
  from `None`; `ConceptRef != None != DONT_KNOW`);
- reads/writes are whole-property known / DONT_KNOW for v1.

`StakeholderKnowledgeConcept {id, truth_concept_id (private), kind,
description: str|DONT_KNOW, terms: list[str]|DONT_KNOW}` supports the four
independent states (term known/details unknown; details known/local wrong
term; details known/term unknown; both known) — tested. Only concepts
referenced by the masked graph are included; hidden Truth ids/canonical terms
never reach the knowledge or the simulator prompt (tested).

## 3. Graph-native provenance

No `StakeholderSemanticAssertion`: private annotations are
`{semantic_id, quote, occurrence}` pointing directly at stakeholder semantic
IDs. Semantic meaning resolves from `StakeholderKnowledgeGraph` — no
subject/property/value duplication. DONT_KNOW speech anchors to the relevant
property slot id. The global span rule is kept: a span covering several
distinct semantic IDs is ambiguous and grounds nothing.

## 4. Dialogue events

Concept identity / terminology keep minimal private events addressing
`StakeholderKnowledgeConcept` ids (`{semantic_id, quote, occurrence, act}`
with act = confirm|partial|unknown|dispute; terminology adds
`proposed_term`). Ordinary mention != terminology agreement (tested).

## 5. Agent-side redesign

`AgentGraph` + `AgentConcept[]` (renamed from BusinessProcessGraph/
BusinessConcept usage). `AgentConcept.mentions` means only "the Agent
interprets this Observation span as referring to the concept". Concept
mention / graph-property evidence / validation evidence are strictly
separated:

- every AgentGraph property reference carries its own EvidenceRef
  (`add_node`/`update_node`/`add_edge` accept
  `{"concept_id", "evidence"}` per property, incl. actor/system/reads/
  writes/rationale/condition);
- property scoring uses property evidence ONLY — the evaluator has no helper
  combining ref evidence + mentions + validation evidence (tested:
  mention/validation-only backing is unsupported);
- concept identity may use mentions;
- validation uses explicit validation/dialogue evidence ONLY.

## 6. Concept status

`hypothesized -> grounded -> confirmed`. `ground_concept` (new tool) marks
grounded with annotation-corresponding evidence — no confirmation dialogue
required; `finish_interview` normally requires referenced concepts >=
grounded (tested: a full build finishes with every concept grounded and zero
confirmations). confirmed/unknown/disputed/partially_confirmed stay backed by
their private events.

## 7. Evaluation

Primary target: **AgentGraph vs StakeholderKnowledgeGraph** (Truth mapping
private). `evaluate(db, knowledge, spec, ...)`:

- node/edge correspondence falls out of the semantic IDs (deterministic
  assignment maximizing property-level matches; the same activity at several
  positions is disambiguated by its semantic id);
- property scoring: DONT_KNOW/None slots reject any assertion (epistemic
  restraint); known slots require refs resolving to the slot/element whose
  concepts bind to the slot's knowledge concept (property evidence only);
- concept identity per kind (mentions may participate; bijection over the
  knowledge concepts the graph references);
- glossary validation via explicit dialogue evidence (events address the
  bound knowledge concept);
- `knowledge_coverage` (Truth vs StakeholderKnowledge) is reported separately
  and never mixed into Agent performance (tested).

## 8. Tests

`tests/test_domains/test_business_interview/` — **64 passed**, proving:

- TruthClaim/claim catalogs/StakeholderSemanticAssertion are gone
  (module-level + file-content assertions);
- stable IDs incl. reads/writes elements (reordering-invariant);
- `None != DONT_KNOW`;
- removal creates no shortcut edge;
- description and terminology vary independently (all four states);
- hidden concepts never enter stakeholder knowledge/prompt;
- Observation spans resolve directly to semantic IDs (unknown ids rejected
  deterministically at ingestion and at eval);
- edge and edge-condition are separately addressable;
- property/mention/validation evidence stay separate;
- grounded concepts finish without redundant confirmation;
- hidden Truth guesses remain wrong (DONT_KNOW slots + invented elements);
- full end-to-end EnvironmentEvaluator replay rewards the faithful
  trajectory (all four env assertions true).

## 9. Real-LLM validation (inspection only)

DeepSeek quotation runs (seeds 9600-9603, artifacts under
`artifacts/business_interview_real_llm/`) were run to inspect consequences of
the graph-native model:

- **Termination**: all four runs now reach `episode_complete` (previously
  max_steps/too_many_errors) — the agent resolves its glossary (>= grounded)
  and finishes the protocol.
- **Node correspondence**: node_recall/precision reach 1.0 (seeds 9600,
  9602) — node identity falls out of the semantic IDs without topology
  matching.
- **Edges**: edge_recall up to 0.33; the agent still cites whole clauses as
  edge evidence (e.g. "then I create the quotation" instead of the
  annotated relation phrase "then" -> `edge:e2`), which the global span
  rule (correctly) refuses as covering edge + activity. The stakeholder's
  sidecar annotations are exact and graph-native ("then" -> `edge:e2`,
  "check the customer information" -> the activity slot of the step).
- **Concept identity**: `concept_correctness` 0.0 when a single referenced
  concept lacks an exact annotation-corresponding ref (the per-kind
  bijection is all-or-nothing by design); glossary errors are all
  "grounded requires an authentic provenance binding".
- **Private-id leakage**: `[]` on every run; **no tool errors** (the
  observation-reset issue stays fixed; `start_inference` preserves
  observations).
- No aliases, authored sentences or evaluator NLP were tuned.

## Remaining issues

- Agent citation style: LLM agents still tend to cite whole clauses rather
  than per-element phrases; the global span rule then (correctly) refuses
  cross-credit. Policy guidance covers it; follow-up could add stronger
  in-tool evidence selection hints.
- Sidecar precision: the stakeholder LLM occasionally misquotes spans or
  invents semantic ids; strict ingestion rejects these and aborts the
  episode after one retry (by design — invalid metadata must not enter the
  conversation).


## v11 addendum — opaque stakeholder IDs, canonical resolver, binding-aware grounding, explicit Agent DONT_KNOW

Follow-up goal report ("tighten graph-native semantics"):

- **Opaque stakeholder-local IDs**: knowledge element ids are now
  `skn_001` / `ske_001` / `skc_001` style — assigned deterministically from
  sorted Truth ids (invariant to collection reordering) and never derived
  from Truth ids, labels, terms, node ids or edge ids. The private Truth
  mappings (`node_truth_ids` / `edge_truth_ids` / per-concept
  `truth_concept_id`) live only in the evaluator-side `StakeholderKnowledge`.
  Observation annotations use stakeholder-local semantic IDs only.
- **Concept descriptions contain no graph facts**: descriptions such as
  "Create the quotation document from customer and pricing data." or "Check
  the customer information before preparing a quotation." were removed from
  Truth/stakeholder concepts; tests prove DONT_KNOW descriptions render as
  "unknown" in the stakeholder prompt and cannot leak Truth text.
- **One canonical semantic-ID resolver** (`StakeholderKnowledgeGraph.
  resolve`): `node:<id>` -> node EXISTENCE (never the activity slot);
  `node:<id>:<prop>` -> the exact property slot; `node:<id>:reads:<k>` ->
  the exact element; `edge:<id>` / `edge:<id>:condition` / `<skc>` -> the
  exact objects. Annotation validation, provenance, concept binding,
  DONT_KNOW handling, knowledge coverage and diagnostics all use it; the
  duplicate parsers (`element_value`, `_parse_node_slot`, ad-hoc splits)
  are gone.
- **Binding-aware `ground_concept`**: evidence must resolve (global span
  rule + canonical resolver) to exactly one kind-compatible knowledge
  concept; ambiguous, unrelated or kind-incompatible evidence is rejected,
  private stakeholder ids never appear in tool output, and the Agent-visible
  `grounded` status always agrees with the evaluator binding (both run the
  same deterministic check; conflicting evidence leaves the concept
  unresolved).
- **Explicit Agent DONT_KNOW**: AgentGraph slots are three-valued
  (`ConceptRef` / `None` / `DONT_KNOW`); `record_dont_know` /
  `record_edge_condition_dont_know` (or `{"dont_know": true, "evidence":
  [...]}` property args) record a marker ONLY when the cited spans resolve
  to the corresponding stakeholder DONT_KNOW slot. Scoring: stakeholder
  `ConceptRef` -> correct grounded ref; `None` -> unasserted (DONT_KNOW not
  equivalent); `DONT_KNOW` -> explicit evidenced DONT_KNOW (unasserted not
  equivalent, hidden-Truth guess wrong).
- **knowledge_coverage** now counts known values AND known absence as known,
  DONT_KNOW slots and removed nodes/edges as unknown, and never confuses
  node existence with the activity slot; it stays informational and
  separate from Agent performance.
- Evidence separation preserved: property scoring uses property evidence
  only; concept identity may use mentions + graph provenance (incl.
  grounding evidence); confirmation/terminology use dialogue events only.
- Deterministic suite: 75 tests (incl. new opaque-ID, resolver, grounding,
  DONT_KNOW and coverage tests). Real-LLM quotation runs: 4 seeds under
  `artifacts/business_interview_real_llm/` (see the run summaries).
