#!/usr/bin/env python3
"""Re-evaluate saved business_interview smoke artifacts under the current
evaluator with scenario-local ``data_expressions``.

Isolates the effect of the scenario-local semantic-equivalence change:

- **baseline**: current evaluator with ``data_expressions`` emptied — i.e. the
  deterministic token/substring matching that existed before this change;
- **expressions**: same evaluator with the quotation scenario's declared
  ``quote -> [quote, quotation]`` expressions.

For every saved DAG (seed 4000-4004 and terminology-policy seed 5000/5001) it
reports, per run:

- ``cq_writes_hit`` — whether the visible create-quotation write axis is scored
  as a match (the 5/5 ``quote``/``quotation`` semantic target);
- ``hidden_assertions`` — the hidden reads/writes the agent asserted (these
  MUST stay failures: epistemic errors, never rescued by expressions);
- write_correctness under both contracts, and a per-axis diff to prove **no
  other axis changed** (no new false positives);
- the stored historical ``evaluator_metrics`` (recorded at run time by the
  pre-visibility scorer) so stored metrics are never confused with replays.

No LLM calls are made; the actor behavior is the saved ``final_dag`` verbatim.
Output: artifacts/business_interview_real_llm/data_expression_reeval.json

Not part of the test suite / CI — a manual diagnostic helper.
"""

import copy
import glob
import json
from pathlib import Path

from tau2.domains.business_interview.dag import BusinessDAG, InterviewDB
from tau2.domains.business_interview.evaluation import _data_recall, evaluate
from tau2.domains.business_interview.scenario import (
    quotation_sales_filter,
    quotation_spec,
    quotation_truth,
)

ART_DIR = Path("artifacts/business_interview_real_llm")

TRUTH = quotation_truth()
STAKEHOLDER = quotation_sales_filter()


def _cv(x):
    return x if x is not None else {}


def _fix_node(n):
    n = dict(n)
    n["action"] = _cv(n.get("action"))
    n["actor"] = _cv(n.get("actor"))
    n["system"] = _cv(n.get("system"))
    n["primitive"] = _cv(n.get("primitive"))
    n["reads"] = [_cv(r) for r in n.get("reads", [])]
    n["writes"] = [_cv(w) for w in n.get("writes", [])]
    if n.get("necessity"):
        n["necessity"] = {k: _cv(v) for k, v in n["necessity"].items()}
    return n


def _load_artifact(path):
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    fdag = dict(d["final_dag"])
    fdag["nodes"] = {nid: _fix_node(n) for nid, n in fdag["nodes"].items()}
    fdag["edges"] = {
        eid: {**e, "predicate": _cv(e.get("predicate"))}
        for eid, e in fdag["edges"].items()
    }
    agent = BusinessDAG.model_validate(fdag)
    return d, agent


def _axis_ok(agent, mapping, tnid, anid, axis, spec):
    """Score one matched node reads/writes axis the way the evaluator does, so
    the diff can isolate which axes the data-expression layer changes."""
    an = agent.nodes[anid]
    tn = TRUTH.nodes[tnid]
    visible = STAKEHOLDER.visible_attributes_for(tnid)
    if axis not in visible:
        vals = list(getattr(an, axis))
        return 1.0 if not any(v.asserted for v in vals) else 0.0
    return _data_recall(
        [v.value for v in getattr(an, axis)],
        [v.value for v in getattr(tn, axis)],
        spec,
    )


def _reeval(d, agent, spec):
    db = InterviewDB(
        dag=agent,
        messages=list(d["db_messages_ledger"]),
        observations=[],
        interview_complete=bool(d["interview_complete"]),
    )
    return evaluate(db, TRUTH, spec, STAKEHOLDER)


def main():
    spec_expr = quotation_spec()
    spec_plain = copy.deepcopy(spec_expr)
    spec_plain.data_expressions = {}

    runs = []
    paths = sorted(glob.glob(str(ART_DIR / "run_*_seed4*.json"))) + sorted(
        glob.glob(str(ART_DIR / "run_*_seed5*.json"))
    )
    for p in paths:
        d, agent = _load_artifact(p)
        name = Path(p).name
        r_expr = _reeval(d, agent, spec_expr)
        r_plain = _reeval(d, agent, spec_plain)

        # per-axis diff (visible + hidden semantics, evaluator mechanics)
        from tau2.domains.business_interview.evaluation import _match_nodes

        mapping = _match_nodes(agent, TRUTH, spec_expr)
        axis_diff = {}
        for anid, tnid in mapping.items():
            for axis in ("reads", "writes"):
                base = _axis_ok(agent, mapping, tnid, anid, axis, spec_plain)
                expr = _axis_ok(agent, mapping, tnid, anid, axis, spec_expr)
                if base != expr:
                    axis_diff[f"{tnid}.{axis}"] = {
                        "baseline": base,
                        "expressions": expr,
                    }

        cq_writes = None
        hidden_entries = []
        for anid, tnid in mapping.items():
            if tnid == "cq":
                cq_writes = {
                    "baseline": _axis_ok(
                        agent, mapping, "cq", anid, "writes", spec_plain
                    ),
                    "expressions": _axis_ok(
                        agent, mapping, "cq", anid, "writes", spec_expr
                    ),
                }
            for axis in ("reads", "writes"):
                if axis in STAKEHOLDER.visible_attributes_for(tnid):
                    continue
                vals = [v.value for v in getattr(agent.nodes[anid], axis) if v.asserted]
                if not vals:
                    continue
                hidden_entries.append(
                    {
                        "node": tnid,
                        "axis": axis,
                        "values": vals,
                        "baseline_score": _axis_ok(
                            agent, mapping, tnid, anid, axis, spec_plain
                        ),
                        "expressions_score": _axis_ok(
                            agent, mapping, tnid, anid, axis, spec_expr
                        ),
                    }
                )

        runs.append(
            {
                "run": name,
                "termination": d.get("termination_reason"),
                "stored_evaluator_metrics": d.get("evaluator_metrics") or {},
                "replayed_baseline": {
                    "write_correctness": r_plain.write_correctness,
                    "quality_pass": r_plain.quality_pass,
                    "structural_pass": r_plain.structural_pass,
                },
                "replayed_expressions": {
                    "write_correctness": r_expr.write_correctness,
                    "read_correctness": r_expr.read_correctness,
                    "actor_correctness": r_expr.actor_correctness,
                    "system_correctness": r_expr.system_correctness,
                    "node_recall": r_expr.node_recall,
                    "node_precision": r_expr.node_precision,
                    "edge_recall": r_expr.edge_recall,
                    "edge_precision": r_expr.edge_precision,
                    "quality_pass": r_expr.quality_pass,
                    "structural_pass": r_expr.structural_pass,
                },
                "cq_writes_quote_quotation": cq_writes,
                "axis_diffs": axis_diff,
                "hidden_assertions": hidden_entries,
            }
        )

    recovered = sum(
        1
        for r in runs
        if r["cq_writes_quote_quotation"]
        and r["cq_writes_quote_quotation"]["baseline"] < 1.0
        and r["cq_writes_quote_quotation"]["expressions"] == 1.0
    )
    hidden_total = sum(len(r["hidden_assertions"]) for r in runs)
    hidden_still_fail = all(
        e["expressions_score"] == 0.0 for r in runs for e in r["hidden_assertions"]
    )
    unexpected_diffs = [
        (r["run"], k) for r in runs for k in r["axis_diffs"] if k != "cq.writes"
    ]

    summary = {
        "scenario": "quotation_workflow_1",
        "method": (
            "replay each saved final_dag under the current evaluator with the "
            "scenario StakeholderFilter; baseline = data_expressions emptied; "
            "expressions = quotation data_expressions configured. Stored "
            "evaluator_metrics are historical artifacts from the run-time "
            "(pre-visibility) scorer and are NOT replayed metrics."
        ),
        "runs_replayed": len(runs),
        "cq_writes_recovered": {
            "count": recovered,
            "of": sum(
                1
                for r in runs
                if r["cq_writes_quote_quotation"]
                and r["cq_writes_quote_quotation"]["baseline"] < 1.0
            ),
        },
        "hidden_epistemic_assertions": {
            "count": hidden_total,
            "all_still_fail": hidden_still_fail,
            "note": (
                "hidden asserted axes are scored 0.0 by _attribute_ok before "
                "data matching; expressions never apply to hidden attributes"
            ),
        },
        "axis_changes_outside_cq_writes": unexpected_diffs,
        "actor_behavior": (
            "unchanged: replays use the saved final_dag verbatim; no LLM calls"
        ),
        "runs": runs,
    }
    out = ART_DIR / "data_expression_reeval.json"
    try:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}")
    print("wrote", out)
    print(f"runs replayed: {len(runs)}")
    print(f"cq.writes quote<->quotation recovered: {recovered}")
    print(f"hidden epistemic assertions remaining: {hidden_total}")
    print(f"hidden assertions all still fail: {hidden_still_fail}")
    print("axis changes outside cq.writes:", unexpected_diffs or "none")
    for r in runs:
        cq = r["cq_writes_quote_quotation"]
        cq_s = f"{cq['baseline']:.2f}->{cq['expressions']:.2f}" if cq else "n/a"
        print(
            f"{r['run']:28} cq.writes {cq_s:12} "
            f"write_corr {r['replayed_baseline']['write_correctness']:.2f}"
            f"->{r['replayed_expressions']['write_correctness']:.2f} "
            f"hidden={len(r['hidden_assertions'])}"
        )


if __name__ == "__main__":
    main()
