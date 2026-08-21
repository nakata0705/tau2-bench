# business_interview: stakeholder semantic fidelity — mode, Truth/Agent split, exact marker binding

Goal: harden stakeholder semantic fidelity and finish the graph-native
epistemic model. No backward compatibility required. Architecture preserved:
Truth = `BusinessProcessGraph` + `TruthConcept[]`; Stakeholder =
`StakeholderKnowledgeGraph` + `StakeholderKnowledgeConcept[]`; Agent =
`AgentGraph` + `AgentConcept[]`; observation provenance stays graph-native.
No `TruthClaim`, no semantic NLP/aliases/embeddings.

## 1. Semantic mode on stakeholder utterance annotations

`SemanticAnnotation` now carries `mode` (`value | absent | dont_know |
exists | mention`), validated deterministically against
`StakeholderKnowledgeGraph.resolve()`:

| resolve() result                       | mode     |
|----------------------------------------|----------|
| `ConceptRef` (known value)             | `value`  |
| `None` (known absent)                  | `absent` |
| `DONT_KNOW` (element known, unknown)   | `dont_know` |
| node/edge existence                    | `exists` |
| `StakeholderKnowledgeConcept`          | `mention`|

`mode_for_resolved()` (facts.py) derives the expected mode from the
resolved semantic; `StakeholderKnowledgeCatalog.validate_annotations`
rejects any annotation whose mode contradicts the stakeholder's own world
model, and `SemanticLedger.bind` runs the same check at ingestion (a
contradictory message never enters the ledger or the conversation). The
parse level requires every annotation to declare `mode` (missing mode =
`ValueError` -> retry). Subject/property/value are NOT duplicated — the
semantic value still lives only in `StakeholderKnowledgeGraph`.
Concept-alignment and terminology dialogue events stay separate.

**Headline case fixed:** a stakeholder whose graph knows the approval
rationale can no longer be accepted saying "I don't know the reason" with a
`dont_know` annotation on the rationale slot — the message is rejected and
retried instead of silently entering the conversation.

## 2. TruthNode/TruthEdge separated from Agent Node/Edge

- `TruthNode` / `TruthEdge` (graph.py) are the complete canonical Truth:
  slots are `ConceptRef | None` / `list[ConceptRef] | None` — **no UNSET /
  ABSENT / DONT_KNOW states**, no evidence fields.
- Agent `Node` / `Edge` keep the four-state model (`UNSET` / `ConceptRef` /
  `ABSENT` / `DONT_KNOW`) with evidence on refs/markers.
- `StakeholderKnowledge` remains three-valued (`ConceptRef | None |
  DONT_KNOW`).
- `_GraphMixin` stays generic over the concept only; subclasses redeclare
  concrete `nodes`/`edges` dicts. Shared utilities (`structure_errors`,
  `referenced_concepts`, `successors`) use the common slot/ref surface.
- Compatibility logic removed: `project_knowledge` no longer interprets
  Truth `UNSET` as `None` (Truth has no UNSET); the knowledge projection is
  driven purely by the visible-property filter and canonical `None`.

## 3. Policy/docstring synchronization

Audited `data/tau2/domains/business_interview/policy.md`, the tools module
docstring, `README.md` and tool docstrings. The visible contract now
consistently states:

    UNSET     = not investigated / no conclusion (the default)
    ConceptRef= known value
    ABSENT    = explicitly established absence (evidenced)
    DONT_KNOW = explicitly established unknown (evidenced)

Stale statements fixed: `condition=None means unconditional` (now: an
unconditional edge needs explicit ABSENT with evidence), `unset means known
absent` (now: omitted property -> UNSET, never a conclusion), `empty
reads/writes means known empty` (now: known-empty needs `{"absent": true,
"evidence": [...]}`; an empty list is ambiguous and rejected).

## 4. Marker tools are exact-node binding aware

Before accepting `record_absent(agent_node, prop)` /
`record_dont_know(agent_node, prop)` (and the dict-path markers through
`add_node` / `update_node` / `add_edge` / `update_edge`), the tool derives
the stakeholder element candidate from the Agent element's **authentic
property provenance** (`_bound_stakeholder_node` / `_bound_stake_edge` /
`_node_candidates_from_evidence`), requires exactly ONE unique binding, then
requires the marker evidence to resolve EXACTLY to
`node:<bound>:<prop>` / `edge:<bound>:condition` (via `_group_marker_evidence`
and the `bound_node`/`bound_edge` checks in `_resolve_absent_slots` /
`_resolve_dont_know_slots`). For edge markers the bound stakeholder edge's
endpoints must also match the agent edge's bound endpoints
(`_require_edge_endpoints_match`), matching the evaluator's edge-mapping
rule. If the element is not uniquely bindable, the tool rejects with a
concise error telling the Agent to add authentic graph/property evidence
first. Tool-visible success and evaluator marker validity can no longer
disagree.

## 5. concept_precision diagnostic fix

`_concept_bindings` now derives diagnostics from the Agent's **attempted
referenced concepts** (`referenced`, including refs on unmapped nodes/edges).
Precision = `|correct one-to-one Agent bindings| / |attempted|`. Attempted
concepts that are extra, unbound (no candidate), ambiguously bound,
conflicting, kind-incompatible, or duplicated/split against one stakeholder
concept all reduce precision (previously they silently vanished from the
denominator). Missing expected concepts reduce recall. An empty AgentGraph:
recall 0 when expected concepts exist, documented neutral precision 1.0,
`concept_correctness` 0. Glossary validation correctness stays separate from
reconstruction completeness.

## 6. Recover valid JSON with invalid tool argument shape

`ToolCall.from_string` and the native-provider path (`generate`) now also
treat JSON that parses but is not an object (`[]`, `null`, `"foo"`, `123`)
as a RECOVERABLE error: the ToolCall carries a concise parse/validation
error (`arguments must be a JSON object`), the environment answers an
ordinary Error tool response, the agent consumes its error budget and may
retry; the run never aborts and semantic content is never repaired.

## 7. Deterministic tests + real-LLM runs

New deterministic tests in `test_graph_business_interview.py`:

- semantic mode / knowledge compatibility; known value + `dont_know`
  annotation rejected; `DONT_KNOW` + `value` annotation rejected; `None` +
  `absent` accepted; `exists`/`mention` modes;
- Truth graph contains no UNSET/ABSENT/DONT_KNOW states; knowledge
  projection independent of Truth UNSET;
- policy/README/tools docs consistently describe the four Agent states;
- marker tool rejects evidence from another node/edge (node, update_node,
  add_edge paths); marker requires unique graph binding;
- successful marker tool call also passes evaluator marker validation
  (tool success == evaluator validity);
- unbound/ambiguous/extra concepts reduce concept_precision; empty
  AgentGraph neutral precision + zero correctness;
- valid non-object JSON tool arguments recoverable; malformed JSON remains
  recoverable.

Real-LLM quotation runs (deepseek-chat-v3 via OpenRouter, temperature 0,
seeds 6200-6202 and 6400-6403):

| seed | termination | node_recall | node_prec | edge_recall | concept_correctness | marker_errors | leaks |
| ------ | ------------- | ------------- | ----------- | ------------- | --------------------- | --------------- | ------- |
| 6200 | context-length API error (agent loop grew past 163k ctx) | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6201 | sidecar rejected twice (quote mismatch) | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6202 | too_many_errors | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6400 | too_many_errors | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6401 | sidecar rejected twice (quote mismatch) | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6402 | too_many_errors | 0.0 | 0.0 | 0.0 | 0.0 | 0 | [] |
| 6403 | too_many_errors | 0.167 | 1.0 | 0.0 | 0.095 | 0 | [] |

Live fidelity evidence (from run logs):

- `annotation 'skc_011': mode 'value' contradicts the knowledge — the
  stakeholder's own semantic model says 'mention'` -> retry once, then the
  corrected message enters;
- `annotation 'node:skn_003:reads:skc_014': mode 'dont_know' contradicts
  the knowledge — the stakeholder's own semantic model says 'value'` ->
  rejected (never accepted);
- quote/occurrence mismatches rejected and retried;
- `marker_evidence_errors == 0` on every run (no unsupported
  ABSENT/DONT_KNOW marker ever reached the evaluator);
- `private_id_leakage == []` on every run.

Observed remaining issues (not regressions):

- gpt-4o-mini frequently violates the exact-quote contract (quote not an
  exact substring) — the strict contract rejects and retries, but the model
  often fails twice, terminating the turn; deepseek-chat-v3 is more
  reliable but the agent still exhausts its 30-error budget before
  completing the full graph (strict evidence tools + a hard interview
  budget). No run reached `finish_interview` with a full reconstruction;
  seed 6403 made partial progress (1 node / 3 concepts, structurally valid,
  zero marker errors, zero leakage).
- context-length termination at 163k tokens for agent loops (repeated tool
  errors inflate history) — agent-side, not a fidelity leak.
