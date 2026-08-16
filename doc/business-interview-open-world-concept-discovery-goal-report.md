# business_interview: Open-World Concept Discovery — Completion Report

## Goal summary

Rebuild `business_interview` so it applies to **unknown industries**: separate
**generic operation primitives** from **open-world discovered domain concepts**,
move all quotation-specific semantic matching into a **hidden scenario-local
EvaluationSpec** (evaluator-only), and let the runtime agent discover unknown
concepts freely rather than classify into a fixed ontology.

## Benchmark vs production separation

In production there is normally no ground truth; in the benchmark a hidden
ground truth exists and **only the evaluator** uses it, deterministically. So:

```
Concept resolver / hidden EvaluationSpec = benchmark scoring aid
Agent's business understanding = open-world, ontology-free
```

The agent never sees the hidden spec and does not depend on any scenario
ontology.

## Generic primitives

`concepts.py` (global) now holds only **reusable generic operation primitives**
and signal words for guessing an abstract operation from free text:

```
create / check / approve / reject / send / receive / record / update /
transform / reconcile / notify / move / review
```

`resolve_primitive(text)` returns the best primitive or `None` (an unknown
operation is allowed and never fails the agent). The quotation-specific
`NODE_CONCEPTS` / `DATA_CONCEPTS` / `PREDICATE_CONCEPTS` / `NECESSITY_CONCEPTS`
were **removed** from the global module.

## DiscoveredConcept design

A first-class model in `dag.py`:

```python
DiscoveredConcept(id, label, description?, aliases[], observation_ids[], confidence)
```

The agent discovers unknown domain concepts via `discover_concept(...)` (id,
label, description?, aliases?, confidence?, observation_id?). New observations
for the same concept are **merged** (aliases + provenance accumulate) — a new
observation never creates a duplicate. Nodes may reference a concept via
`concept_id`.

## Open-world DAG

`BusinessDAG` (used identically for Truth and inferred DAG) is open-world:
- `Node.action` is free text (e.g. "manager approval of high-value quotation",
  "chamber seasoning", "specimen accession").
- `Node.primitive` is the **optional** generic operation (unset / unknown is fine).
- `Node.concept_id` optionally references a `DiscoveredConcept`.
- `BusinessDAG.concepts` holds the agent's discovered concepts.
- No hidden truth concept IDs are required on the runtime DAG.

## Hidden scenario EvaluationSpec

`evaluation.py` defines:

```python
TruthNodeSpec(expressions[], primitive?)
EvaluationSpec(truth_nodes: {truth_node_id: TruthNodeSpec},
               predicate_expressions: {predicate: [EN/JA aliases]},
               necessity_expressions: {value: [EN/JA aliases]})
```

`scenario.py` holds this metadata per scenario (quotation + lab). It is
evaluator-only and never exposed to the agent. The global resolver keeps only
reusable generic primitives.

## Evaluation metrics

Structure + correctness (unchanged): `node_recall`/`node_precision`,
`domain_concept_correctness`, `edge_recall`/`edge_precision`, `start_correct`,
`end_recall`/`end_precision`, `predicate_correctness`,
`actor/system/read/write_correctness`, `necessity_correctness`,
`fabricated_node/edge_count`, `fabricated_necessity`.

New:
- `primitive_correctness` — separate from domain-concept correctness, so
  "correct domain concept + wrong primitive" is diagnosed independently.
- `discovered_concept_recall` / `discovered_concept_precision`,
  `fabricated_concept_count`, `duplicate_concept_count`, `concept_discovery_pass`.
  A discovered concept requires Observation provenance.

`quality_pass` requires `structural AND necessity AND evidence AND
provenance_authenticity AND relevance AND concept_discovery`.

Node matching uses the hidden expressions (token overlap + substring, including
bilingual EN/JA aliases), so a correct unknown concept found under a different
wording still matches. Agent concept/node IDs are arbitrary.

## Claim-level evidence relevance

`relevance_pass` checks each asserted claim's provenance individually (action,
primitive, actor, system, reads/writes, edge, predicate, necessity, discovered
concept) and classifies it SUPPORTED / CONTRADICTED / UNKNOWN. A claim with only
clearly-unrelated observations fails (poisoning / partial-poisoning exploit);
UNKNOWN-without-observation is handled by the evidence gate (asserted claims
need provenance). Deterministic, no LLM/embedding.

## New non-quotation scenario

`lab_sample_flow` — a lab scenario with unknown domain concepts (specimen
accession, chamber seasoning, conditioning cycle, approve conditioned batch).
Its expressions live only in the hidden scenario-local EvaluationSpec. It is
registered as a scenario and as a runnable task (in the `base` split), and a
unit test reconstructs it to a full pass — proving the design is not
quotation-specific.

## Removed closed-world assumptions

- Removed global quotation `NODE_CONCEPTS` / `DATA_CONCEPTS` /
  `PREDICATE_CONCEPTS` / `NECESSITY_CONCEPTS` from `concepts.py`.
- `EvaluationSpec` no longer a simple `truth_node_concepts` id map; it is a
  hidden per-scenario expression/primitive spec.
- Evaluator no longer imports global scenario concepts; matching is driven by the
  hidden spec (bilingual expressions).
- No adapter / deprecated wrapper / migration layer retained.

## Tests / verification

- `tests/test_domains/test_business_interview/test_dag_business_interview.py`:
  **80 passed** — the existing evidence/endpoint/necessity/authenticity/EN-JA
  regression suite plus new open-world falsification tests:
  - scenario-specific truth ID unknown → full pass
  - arbitrary discovered concept IDs → pass
  - synonymous discovered labels → hidden matcher match
  - unseen domain concept created/merged
  - unknown primitive keeps domain concept
  - correct domain concept + wrong primitive → separate diagnostics
  - duplicate concepts → quality drop
  - fabricated concept → precision drop
  - concept without Observation provenance fails
  - unrelated observation cannot support an individual claim
  - partial provenance poisoning (action ok, actor/system poisoned) fails
  - edge / predicate provenance poisoning fails
  - non-quotation lab scenario → full pass
- `make check-all` (ruff lint + format): **passed**.
- `make test` (core): **308 passed, 1 xfailed, 1 failed** — the failure is the
  known **flaky live-LLM mock-domain test** (`test_run_tasks_nl_assertions`,
  gpt-3.5-turbo) which **passes in isolation**; unrelated to this change.
- Reference trajectory (quotation) through `EnvironmentEvaluator` → **reward 1.0**
  (unit test `test_evaluator_rewards_full_reconstruction`).

## Limitations

- **Relevance is deterministic and token/substring-based.** It reliably rejects
  clearly-unrelated evidence but is not a semantic entailment check; a
  heavy-paraphrase statement sharing no node signal could be treated as
  unsupported. Authenticity remains the hard gate; relevance is best-effort.
- **Open-world matching depends on the hidden expressions** being reasonably
  comprehensive (the scenario author provides EN/JA aliases). Ambiguous
  expressions can cause tie-breaks (resolved deterministically by actor/system).
- **Concept discovery requires the agent to create `DiscoveredConcept`s** for the
  truth domain concepts to achieve `concept_discovery_pass`; a DAG that matches
  structurally but records no concepts will not fully pass.
- **Generic primitive resolution is heuristic** (`resolve_primitive`); an unknown
  operation simply resolves to None and is not penalised.
- The bundled `deepseek-chat` agent rarely follows the full tool protocol, so
  real-run reward is low (weak-agent behavior, not a benchmark defect).

## Commit information

This change is committed on the `business-interview` branch and pushed to
`origin/business-interview`. The report itself is part of the same commit and
cannot reference its own SHA.
