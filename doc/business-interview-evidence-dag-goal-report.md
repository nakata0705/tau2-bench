# business_interview: Evidence-Backed DAG Benchmark — Completion Report

## Goal summary

Turn the `business_interview` domain into a complete **evidence-backed business DAG
inference benchmark**. The core contract is:

```
Stakeholder statements -> Observations -> InferredValues -> Nodes / Edges / Necessity -> Inferred BusinessDAG
```

Every claim the agent records must be traceable to a real recorded Observation.
`quality_pass` now requires structural correctness, necessity correctness, **and**
evidence-backedness. This is a continuation of the v3 DAG redesign (commit
`e044255`); no legacy Workflow / Step / Transition / Branch compatibility code is
retained.

## Final architecture

- `dag.py` — `BusinessDAG` (shared by Truth and inferred DAG), `Node`, `Edge`,
  `InferredValue`, `Necessity`, `Observation` (immutable), `InterviewResult`,
  `InterviewDB`, and `BusinessDAG.validate()`.
- `concepts.py` — evaluator-only bilingual concept resolution (nodes / data /
  predicates / necessity values); never stored on nodes.
- `aliases.py` — role / system aliases.
- `evaluation.py` — `EvaluationSpec` + `evaluate()` producing
  `EvaluationResult` with structural, necessity, and evidence metrics.
- `stakeholder.py` — `StakeholderFilter` (a visibility filter over the single
  Truth DAG).
- `scenario.py` — the quotation Truth DAG, `EvaluationSpec`, stakeholder filters.
- `tools.py` — agent tools + assertions (`assert_finish_interview`,
  `assert_dag_reconstructed`, `assert_necessity_handled`,
  `assert_evidence_backed`) + diagnostics.

## Enforced invariants

- **Declared endpoints.** `BusinessDAG.validate()` requires exactly one declared
  start, one or more declared ends (`end_node_ids=[]` is invalid), declared
  endpoints exist, each declared end has out-degree 0, no dangling edges, no
  cycles, every node reachable from the start. The evaluator compares the agent's
  **declared** `start_node_id` / `end_node_ids` to the Truth's declared endpoints —
  it never substitutes auto-inferred sink nodes, so a correct topology with
  undeclared (or wrongly declared) ends cannot get full structural credit.
- **Evidence is a real gate.** `quality_pass = structural_pass AND necessity_pass
  AND evidence_pass`. A perfect DAG built with zero Observations fails.
- **Provenance is per-claim.** Each asserted `InferredValue` (action / actor /
  system / reads / writes / edge predicate / known necessity property) must carry
  a valid Observation reference. `node_evidence_coverage` requires each node to be
  supported by a real Observation.
- **No dangling observation references.** The tools reject any `observation_id`
  that does not exist; the evaluator also counts and fails on any dangling
  reference (defensive).
- **Unknown stays unknown.** An expected-unknown necessity property must be left
  unset; asserting any value there (even with confidence 0) is a fabrication.
  `"UNKNOWN"` is never stored as a factual string.
- **Confidence semantics.** `confidence` is in [0,1]. A value with `confidence
  == 0` is treated as **unasserted** (unknown): it does not need provenance and
  does not satisfy a known necessity.
- **Conditional flow only on `Edge.predicate`.** No Branch / Transition / Step
  classes; branches are multiple outgoing edges with predicates.

## Bugs found

1. `end_node_ids=[]` was previously considered valid (no error). Fixed in
   `validate()`.
2. The old evaluator inferred end nodes from sinks (out-degree 0) instead of the
   agent's declared endpoints, so a correct topology with undeclared ends could
   pass structural scoring. Fixed: endpoints are compared on the declared values.
3. Observation provenance was not a gate — a perfect DAG with zero Observations
   could reach `quality_pass`. Fixed with the evidence gate.
4. `record_observation` returned a human message rather than the observation id,
   making it awkward to reference observations. Fixed (returns the id).
5. `set_node_necessity` / `update_edge` replaced provenance instead of merging /
   exposing clear semantics; now merge on update and support explicit unset /
   predicate-clear.
6. Tools accepted nonexistent `observation_id` and dangling edge endpoints; they
   now reject them.

## Implementation / evaluator changes

- `dag.py`: empty-`end_node_ids` validation; `Observation` made frozen
  (immutable); `InferredValue.asserted` helper (value set AND confidence > 0).
- `evaluation.py`: declared-endpoint comparison; evidence metric computation
  (`_evidence_metrics`); `evidence_pass`; `quality_pass` now requires
  `structural AND necessity AND evidence`; necessity correctness uses `asserted`
  (confidence > 0) for known values and rejects any value on unknown-expected.
- `tools.py`: `_require_observation`, `_require_node_ref` validation;
  `start_inference` is explicitly destructive (resets DAG, observations, and
  interview); `set_node_necessity(..., unset=[...])` resets properties;
  `update_edge(..., clear_predicate=True)` removes a predicate; `add_edge` /
  `update_edge` / `set_dag_endpoints` validate node references;
  `record_observation` returns the observation id; added `assert_evidence_backed`.

## Evidence metrics / gates

| metric | meaning |
|--------|---------|
| `node_evidence_coverage` | fraction of nodes supported by a valid Observation |
| `attribute_provenance_coverage` | fraction of asserted structural attributes with valid provenance |
| `edge_evidence_coverage` | fraction of edges supported by a valid Observation |
| `predicate_provenance_coverage` | fraction of asserted predicates with valid provenance |
| `necessity_provenance_coverage` | fraction of asserted necessity properties with valid provenance |
| `invalid_observation_reference_count` | number of dangling Observation references |
| `evidence_pass` | all coverages == 1.0 and no invalid references |

`quality_pass = structural_pass AND necessity_pass AND evidence_pass`.

## Tests

`tests/test_domains/test_business_interview/test_dag_business_interview.py`
(55 tests) covers the falsification suite A–AE plus the new evidence / endpoint
regression tests:

- empty `end_node_ids` invalid
- undeclared ends cannot get full structural score
- declared endpoints must match truth (not just sinks)
- perfect DAG + zero Observations fails the evidence gate
- valid Observation-backed DAG passes
- nonexistent observation ref rejected by tools
- node attribute / edge / predicate / known-necessity without provenance fails
- expected-unknown necessity unset passes; fabricated unknown fails
- unrelated Observation alone does not pass evidence
- confidence-0 asserted value treated as unasserted
- wrong actor / system / read / write fails structural
- arbitrary agent node ids pass
- EN/JA equivalent inference passes
- cycle / unreachable / dangling fail

## Verification commands / results

- `uv run pytest tests/test_domains/test_business_interview/` → **55 passed**.
- `make check-all` (ruff lint + format) → **All checks passed**.
- `make test` (core) → **284 passed, 1 xpassed, 1 failed**. The failure is a
  **flaky live-LLM mock-domain test** (`test_run_tasks_nl_assertions` or
  `test_run_tasks_env_assertions`, both gpt-3.5-turbo based) which **passes in
  isolation**; unrelated to this change (only business_interview files modified).
- DeepSeek real run (base split, 3× EN + 3× JA): the evidence-backed evaluator ran
  normally — `node_evidence_coverage=1.0`, `attribute_provenance_coverage=1.0`,
  `invalid_observation_reference_count=0` (the agent recorded observations and the
  tools prevented dangling refs); `evidence_pass=False` correctly surfaced
  incomplete edge/predicate provenance, and `end_recall=0.5` reflected the agent
  failing to declare both ends.

## Removed legacy behavior

- `end_node_ids=[]` accepted as valid.
- Sink-auto-inference substituting for declared endpoints.
- Evidence being optional (not a quality gate).
- Tool acceptance of nonexistent Observation references.
- `set_node_necessity` silently replacing provenance.
- `record_observation` returning a non-id message.
- (Earlier v3 redesign already removed Step / Transition / Branch / GTStep /
  GTTransition / GTBranch / StakeholderStepTruth / GTNecessity / challenge state.)

## Remaining limitations

- Same-concept nodes differing only by graph position are not disambiguated
  (actor/system/data tie-break only); graph-context matching is out of scope.
- No confidence calibration scoring (stored / validated / surfaced only).
- The bundled `deepseek-chat` agent is weak on edges/predicates/endpoint
  declaration, so real-run reward is low; the reference trajectory verifies the
  full evidence-backed path with reward 1.0.
- Simulator instructions are hand-authored (not generated from the filter);
  leakage tests guard consistency.

## Final commit SHA

See the git history for this change (committed and pushed to
`origin/business-interview`).
