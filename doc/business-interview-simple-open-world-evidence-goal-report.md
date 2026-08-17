# business_interview: Simple Open-World Evidence Benchmark — Completion Report

## Goal summary

Simplify `business_interview` into a **simple open-world DAG benchmark** centered
on free-text actions + generic primitives + authentic Observation provenance,
removing `DiscoveredConcept` and concept-discovery evaluation, and fixing
**claim-level evidence relevance** so each claim is evaluated against its own
value (action / primitive / actor / system / read / write / edge relation /
predicate / necessity property).

## DiscoveredConcept removed

Fully removed:
- `DiscoveredConcept` model
- `BusinessDAG.concepts`
- `Node.concept_id`
- `discover_concept` tool
- concept discovery recall / precision
- fabricated / duplicate concept metrics
- `concept_discovery_pass`
- related policy / task / test code

The benchmark now evaluates the ability to DAG-ify an unknown business as free
actions, not the ability to manage discovered concepts as separate objects.

## unclassified semantics

`concepts.py` keeps only **scenario-independent generic operation primitives**
(create / check / approve / reject / send / receive / record / update / transform
/ reconcile / notify / move / review). `resolve_primitive` returns the best
primitive or **`"unclassified"`** when no known primitive can be safely
determined (previously `None`). `unclassified` is a normal open-world state and
is **not** a failure: it is not penalized in `primitive_correctness`, and an
`unclassified` primitive requires no Observation provenance and no relevance
support. A *wrong known* primitive still lowers `primitive_correctness`.

## Hidden EvaluationSpec maintained

`TruthNodeSpec(expressions, primitive)` and `EvaluationSpec(truth_nodes,
predicate_expressions, necessity_expressions)` remain evaluator-only and
scenario-local. The global resolver does not contain quotation-specific aliases.
Arbitrary agent node IDs, EN/JA paraphrases, and unknown-industry expressions are
matched deterministically against the hidden expressions.

## Relevance fixed (claim-specific)

Replaced the node-wide relevance with `support(observation, claim_kind,
claim_value) -> SUPPORTED / CONTRADICTED / UNKNOWN`. Key fixes:
- **actor / system are now evaluated as claims** (previously omitted).
- **claim_value is actually used**: actor is matched against the actor value,
  system against the system value, each read/write against its data value, the
  predicate against the predicate itself. Node-wide signals are not reused.
- **edge** requires a relation word (after / then / followed by / before / 次に /
  後に / その後 / ...) **and** both endpoint actions — a from-node mention alone
  does not support an edge.
- **predicate** is a separate claim, supported only by its own direction / value.
- **negation** takes priority: a relevant observation that explicitly negates the
  claim is CONTRADICTED, never SUPPORTED.
- **CONTRADICTED > SUPPORTED > UNKNOWN**; unrelated observations are UNKNOWN
  (unsupported), not SUPPORTED; asserted claims require at least one SUPPORTED
  provenance.
- No LLM-as-judge / embeddings / vector DB.

## Quality gate

`quality_pass` requires `structural_pass AND necessity_pass AND evidence_pass AND
provenance_authenticity_pass AND relevance_pass`. `concept_discovery_pass` is
removed. `primitive_correctness` remains a diagnostic; `unclassified` is not a
failure.

## Non-quotation scenario

`lab_sample_flow` is maintained and reconstructs to a **full pass without any
DiscoveredConcept**: specimen accession / chamber seasoning / conditioning cycle /
approve conditioned batch are handled as free-text actions with generic
primitives (and `unclassified` where safe classification is not possible). Its
unknown expressions live only in the hidden scenario-local EvaluationSpec.

## Tests

`tests/test_domains/test_business_interview/test_dag_business_interview.py`
(30 tests) covers:
- DiscoveredConcept / discover_concept / concept_id / `concepts` removed
- valid full DAG passes (quality / relevance / evidence / structural)
- **per-claim poisoning** (action / primitive / actor / system / read / write /
  predicate / edge / necessity) — build a quality_pass DAG, replace only the
  target claim's provenance with an unrelated Observation → relevance fails
- correct-action evidence cannot substitute for actor
- from-node evidence alone cannot support an edge
- node evidence alone cannot support a predicate
- explicit negation → CONTRADICTED (fails)
- unrelated evidence → unsupported
- `resolve_primitive(unknown) == "unclassified"`; `unclassified` primitive valid
- correct action + wrong known primitive → `primitive_correctness` drops
- arbitrary agent node IDs pass
- Observation authenticity invariants (idempotent capture, invalid turn/role)
- endpoint / necessity regressions
- EN/JA equivalent behavior
- non-quotation lab scenario full pass; lab `unclassified` primitive valid
- leakage (hidden terms not in policy / tool docs)
- reference trajectory reward 1.0; missing-node / fabricated-necessity failures

Partial-poisoning tests start from a quality_pass DAG and replace **only the
target claim's provenance** — no test fails because another claim is missing
evidence.

## Verification results

- `uv run pytest tests/test_domains/test_business_interview/` → **30 passed**.
- `make check-all` (ruff lint + format) → **passed**.
- `make test` (core) → **260 passed, 1 xpassed, 0 failed**.
- Reference trajectory (quotation) through `EnvironmentEvaluator` → **reward 1.0**
  (unit test `test_evaluator_rewards_full_reconstruction`).

## Limitations

- Relevance is deterministic token / substring / direction based. It reliably
  rejects clearly-unrelated evidence but is not a semantic entailment check; a
  heavy paraphrase sharing no node/claim token could be treated as unsupported.
- Primitive resolution is a heuristic substring matcher; JA verb coverage was
  extended (e.g. 送る / 受け取る / 作る / 確認する) but remains finite — unknown
  operations resolve to `unclassified` (not a failure).
- `unclassified` is a valid primitive state, but the evaluator does not require
  the agent to classify operations at all.
- The bundled `deepseek-chat` agent rarely follows the full tool protocol, so a
  real-model smoke reward is low (weak-agent behavior, not a benchmark defect).

## Commit information

This change is committed on the `business-interview` branch and pushed to
`origin/business-interview`. The report itself is part of the same commit and
cannot reference its own SHA.
