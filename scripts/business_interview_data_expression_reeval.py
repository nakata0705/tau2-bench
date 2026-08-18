#!/usr/bin/env python3
"""Replay saved business_interview smoke artifacts under the precision-first
data matcher.

Isolates the effect of the precision-first data-equivalence change by
replaying every saved DAG (seeds 4000-4004 and terminology-policy seeds
5000/5001) under three read/write matching contracts:

- ``loose``       — the pre-change matcher (normalized token overlap OR
                    substring containment, applied to canonical values AND
                    declared expressions). This is what previously let
                    "quotation request" / "quotation information" match
                    "quote" via the declared expression "quotation".
- ``exact_core``  — the new matcher with the scenario's ``data_expressions``
                    emptied: normalized EXACT canonical value only.
- ``exact_decl``  — the new matcher with the quotation scenario's declared
                    complete labels (the shipped contract).

For every run it reports the per-axis score under each contract, the
``cq.writes`` recovery, hidden assertions (must stay failures), and the
collision-probe matrix (near-collisions must NOT match "quote" and must
produce zero false positives).

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

# Collision probe set: agent labels probed against each canonical Truth data
# value. Only declared complete labels may match; everything else must fail.
COLLISION_PROBES = [
    "quote",
    "quotation",
    "quotation request",
    "quotation information",
    "quotation document",
    "price quotation",
    "invoice",
    "customer",
    "customer information",
    "customer request",
    "request",
    "summary",
    "quotation summary",
    "excel_summary",
    "summary of quotation information",
    "Excel file",
    "pricing",
    "pricing information",
    "pricing information (from quoting system)",
]

_TRUE_DATA_VALUES = sorted(
    {
        v.value
        for n in TRUTH.nodes.values()
        for axis in ("reads", "writes")
        for v in getattr(n, axis)
    }
)


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


def _reeval(d, agent, spec):
    db = InterviewDB(
        dag=agent,
        messages=list(d["db_messages_ledger"]),
        observations=[],
        interview_complete=bool(d["interview_complete"]),
    )
    return evaluate(db, TRUTH, spec, STAKEHOLDER)


def _axis_score(agent, mapping, tnid, anid, axis, spec):
    """Score one matched node reads/writes axis the way the evaluator does."""
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


def _loose_item_ok(tvalue, avalue, spec):
    """The PRE-CHANGE loose matcher (token overlap OR substring containment
    over canonical values AND declared expressions) — kept here only as the
    measurement baseline; it is no longer part of the evaluator."""
    import re

    tokens = re.compile(r"[a-z0-9]+")
    at = set(tokens.findall((avalue or "").lower()))
    a_low = (avalue or "").lower()
    tt = set(tokens.findall(tvalue.lower()))
    if tt & at or tvalue.lower() in a_low:
        return True
    for expr in spec.data_expressions.get(tvalue, ()):
        et = set(tokens.findall(expr.lower()))
        if et & at or expr.lower() in a_low:
            return True
    return False


def _loose_data_recall(agent_items, truth_items, spec):
    if not truth_items:
        return 1.0 if not agent_items else 0.0
    hits = 0
    for gi in range(len(truth_items)):
        if not truth_items[gi]:
            hits += 1
            continue
        if any(_loose_item_ok(truth_items[gi], a, spec) for a in agent_items):
            hits += 1
    return hits / len(truth_items)


def _axis_score_loose(agent, mapping, tnid, anid, axis, spec):
    an = agent.nodes[anid]
    tn = TRUTH.nodes[tnid]
    visible = STAKEHOLDER.visible_attributes_for(tnid)
    if axis not in visible:
        vals = list(getattr(an, axis))
        return 1.0 if not any(v.asserted for v in vals) else 0.0
    return _loose_data_recall(
        [v.value for v in getattr(an, axis)],
        [v.value for v in getattr(tn, axis)],
        spec,
    )


def main():
    spec_expr = quotation_spec()
    spec_core = copy.deepcopy(spec_expr)
    spec_core.data_expressions = {}

    # collision matrix: probe each agent label against each canonical Truth
    # data value under the shipped contract; report every intended hit and
    # every unexpected hit (false positive).
    matrix = []
    for truth in _TRUE_DATA_VALUES:
        for probe in COLLISION_PROBES:
            hit = _data_recall([probe], [truth], spec_expr)
            intended = (
                probe in ([truth] + spec_expr.data_expressions.get(truth, []))
                and _data_recall([probe], [truth], spec_expr) == 1.0
            )
            matrix.append(
                {
                    "truth": truth,
                    "probe": probe,
                    "hit": hit == 1.0,
                    "intended": intended,
                }
            )
    false_positives = [m for m in matrix if m["hit"] and not m["intended"]]

    runs = []
    paths = sorted(glob.glob(str(ART_DIR / "run_*_seed4*.json"))) + sorted(
        glob.glob(str(ART_DIR / "run_*_seed5*.json"))
    )
    for p in paths:
        d, agent = _load_artifact(p)
        name = Path(p).name
        r_loose = _reeval(d, agent, spec_expr)
        r_core = _reeval(d, agent, spec_core)
        r_expr = _reeval(d, agent, spec_expr)

        from tau2.domains.business_interview.evaluation import _match_nodes

        mapping = _match_nodes(agent, TRUTH, spec_expr)
        axis_changes = {}
        for anid, tnid in mapping.items():
            for axis in ("reads", "writes"):
                base = _axis_score_loose(agent, mapping, tnid, anid, axis, spec_expr)
                core = _axis_score(agent, mapping, tnid, anid, axis, spec_core)
                expr = _axis_score(agent, mapping, tnid, anid, axis, spec_expr)
                if base != expr or core != expr:
                    an = agent.nodes[anid]
                    tn = TRUTH.nodes[tnid]
                    axis_changes[f"{tnid}.{axis}"] = {
                        "loose": base,
                        "exact_core": core,
                        "exact_decl": expr,
                        "truth": [v.value for v in getattr(tn, axis)],
                        "agent": [v.value for v in getattr(an, axis) if v.asserted],
                    }

        cq_writes = None
        hidden = []
        for anid, tnid in mapping.items():
            if tnid == "cq":
                cq_writes = {
                    "loose": _axis_score_loose(
                        agent, mapping, "cq", anid, "writes", spec_expr
                    ),
                    "exact_decl": _axis_score(
                        agent, mapping, "cq", anid, "writes", spec_expr
                    ),
                }
            for axis in ("reads", "writes"):
                if axis in STAKEHOLDER.visible_attributes_for(tnid):
                    continue
                vals = [v.value for v in getattr(agent.nodes[anid], axis) if v.asserted]
                if not vals:
                    continue
                hidden.append(
                    {
                        "node": tnid,
                        "axis": axis,
                        "values": vals,
                        "score": _axis_score(
                            agent, mapping, tnid, anid, axis, spec_expr
                        ),
                    }
                )

        runs.append(
            {
                "run": name,
                "termination": d.get("termination_reason"),
                "stored_evaluator_metrics": d.get("evaluator_metrics") or {},
                "replayed_loose": {
                    "read_correctness": r_loose.read_correctness,
                    "write_correctness": r_loose.write_correctness,
                },
                "replayed_exact_core": {
                    "read_correctness": r_core.read_correctness,
                    "write_correctness": r_core.write_correctness,
                },
                "replayed_exact_decl": {
                    "read_correctness": r_expr.read_correctness,
                    "write_correctness": r_expr.write_correctness,
                    "quality_pass": r_expr.quality_pass,
                    "structural_pass": r_expr.structural_pass,
                },
                "cq_writes": cq_writes,
                "axis_changes": axis_changes,
                "hidden_assertions": hidden,
            }
        )

    cq_recovered = sum(
        1
        for r in runs
        if r["cq_writes"]
        and r["cq_writes"]["loose"] == 1.0
        and r["cq_writes"]["exact_decl"] == 1.0
    )
    hidden_total = sum(len(r["hidden_assertions"]) for r in runs)
    hidden_still_fail = all(
        e["score"] == 0.0 for r in runs for e in r["hidden_assertions"]
    )
    # matches kept by a declared expression: exact_core miss -> exact_decl hit
    recovered_by_decl = [
        (r["run"], k, v)
        for r in runs
        for k, v in r["axis_changes"].items()
        if v["exact_core"] < 1.0 and v["exact_decl"] == 1.0
    ]
    # matches lost for good: hit under loose AND exact_core, lost under exact_decl
    lost = [
        (r["run"], k, v)
        for r in runs
        for k, v in r["axis_changes"].items()
        if v["loose"] == 1.0 and v["exact_decl"] < 1.0
    ]

    summary = {
        "scenario": "quotation_workflow_1",
        "method": (
            "replay each saved final_dag under the current evaluator with the "
            "scenario StakeholderFilter. loose = pre-change matcher (token/"
            "substring incl. expressions, simulated locally), exact_core = "
            "normalized-exact canonical only, exact_decl = shipped contract "
            "(normalized-exact canonical OR declared complete labels). Stored "
            "evaluator_metrics are historical artifacts from the run-time "
            "(pre-visibility) scorer and are NOT replayed metrics."
        ),
        "data_matching_rule": (
            "normalized exact canonical value OR normalized exact "
            "scenario-local accepted expression; no token overlap, no "
            "substring containment"
        ),
        "runs_replayed": len(runs),
        "cq_writes_recovered": {"count": cq_recovered, "of": len(runs)},
        "hidden_epistemic_assertions": {
            "count": hidden_total,
            "all_still_fail": hidden_still_fail,
        },
        "collision_probe_matrix": {
            "probes": len(matrix),
            "false_positives": [m for m in matrix if m["hit"] and not m["intended"]],
            "intended_hits": [m for m in matrix if m["hit"] and m["intended"]],
            "near_collisions_vs_quote": {
                p: _data_recall([p], ["quote"], spec_expr) == 1.0
                for p in (
                    "quotation request",
                    "quotation information",
                    "quotation document",
                    "price quotation",
                    "invoice",
                )
            },
        },
        "matches_recovered_by_declared_expression": [
            {"run": r, "axis": k, "details": v} for r, k, v in recovered_by_decl
        ],
        "matches_lost_for_good": [
            {"run": r, "axis": k, "details": v} for r, k, v in lost
        ],
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
    print(f"cq.writes quote<->quotation recovered: {cq_recovered}/{len(runs)}")
    print(
        f"hidden epistemic assertions: {hidden_total}, all still fail: {hidden_still_fail}"
    )
    print(f"collision probes: {len(matrix)}, false positives: {len(false_positives)}")
    print(
        "near collisions vs quote:",
        summary["collision_probe_matrix"]["near_collisions_vs_quote"],
    )
    print(f"matches recovered by declared expression: {len(recovered_by_decl)}")
    print(f"matches lost for good: {len(lost)}")
    for r in runs:
        cq = r["cq_writes"]
        cq_s = f"loose={cq['loose']:.2f} decl={cq['exact_decl']:.2f}" if cq else "n/a"
        rd, rc, rx = (
            r["replayed_loose"]["read_correctness"],
            r["replayed_exact_core"]["read_correctness"],
            r["replayed_exact_decl"]["read_correctness"],
        )
        wd, wc, wx = (
            r["replayed_loose"]["write_correctness"],
            r["replayed_exact_core"]["write_correctness"],
            r["replayed_exact_decl"]["write_correctness"],
        )
        print(
            f"{r['run']:28} cq.writes {cq_s:26} "
            f"read {rd:.2f}/{rc:.2f}/{rx:.2f} write {wd:.2f}/{wc:.2f}/{wx:.2f} "
            f"hidden={len(r['hidden_assertions'])}"
        )


if __name__ == "__main__":
    main()
