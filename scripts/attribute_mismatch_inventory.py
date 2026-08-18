#!/usr/bin/env python3
"""Diagnose actor/system/read/write mismatches in business_interview smoke runs.

Read-only analysis (no scoring behavior changes). For each saved artifact
(seed 4000-4004) it:
  - reconstructs the agent DAG and the Ground Truth DAG,
  - runs the evaluator's own _match_nodes to get the node mapping,
  - for every matched node compares actor / system / reads / writes between the
    agent node and the Truth node, and records whether the CURRENT evaluator
    (norm_role / norm_system / _data_recall) scores them as a hit,
  - collects the stakeholder evidence (Observations attached to the agent node /
    attribute) so a human can classify each mismatch (genuine agent error vs
    evaluator-too-strict vs truth-modeling vs information-not-obtained vs
    ambiguous).

Not part of the test suite / CI — a manual diagnostic helper.
"""

import glob
import json
from pathlib import Path

from tau2.domains.business_interview.aliases import norm_role, norm_system
from tau2.domains.business_interview.dag import BusinessDAG
from tau2.domains.business_interview.evaluation import _data_recall, _match_nodes
from tau2.domains.business_interview.scenario import quotation_spec, quotation_truth

ART_DIR = Path("artifacts/business_interview_real_llm")


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


def _val(v):
    if v is None:
        return None
    return v.value if hasattr(v, "value") else (v or {}).get("value")


def _obs_map(d):
    return {o["id"]: o.get("text") for o in d["observations"]}


def _evidence(ids, obs_map):
    out = []
    for i in ids or []:
        t = obs_map.get(i)
        if t:
            out.append(t)
    return out


def analyze_artifact(path):
    d = json.load(open(path))
    name = Path(path).name
    obs_map = _obs_map(d)
    truth = quotation_truth()
    spec = quotation_spec()
    fdag = dict(d["final_dag"])
    fdag["nodes"] = {nid: _fix_node(n) for nid, n in fdag["nodes"].items()}
    fdag["edges"] = {
        eid: {**e, "predicate": _cv(e.get("predicate"))}
        for eid, e in fdag["edges"].items()
    }
    agent = BusinessDAG.model_validate(fdag)
    mapping = _match_nodes(agent, truth, spec)
    rev = {t: a for a, t in mapping.items()}

    node_analyses = []
    for anid, tnid in mapping.items():
        an = agent.nodes[anid]
        tn = truth.nodes[tnid]
        entry = {
            "agent_node": anid,
            "truth_node": tnid,
            "action_agent": _val(an.action),
            "action_truth": _val(tn.action),
            "actor": {
                "truth": _val(tn.actor),
                "agent": _val(an.actor),
                "norm_truth": norm_role(tn.actor.value),
                "norm_agent": norm_role(an.actor.value),
                "hit": norm_role(an.actor.value) == norm_role(tn.actor.value),
                "agent_evidence": _evidence(an.actor.observation_ids, obs_map),
                "node_evidence": _evidence(an.observation_ids, obs_map),
            },
            "system": {
                "truth": _val(tn.system),
                "agent": _val(an.system),
                "norm_truth": norm_system(tn.system.value),
                "norm_agent": norm_system(an.system.value),
                "hit": norm_system(an.system.value) == norm_system(tn.system.value),
                "agent_evidence": _evidence(an.system.observation_ids, obs_map),
                "node_evidence": _evidence(an.observation_ids, obs_map),
            },
            "reads": {
                "truth": [_val(r) for r in tn.reads],
                "agent": [_val(r) for r in an.reads],
                "hit": _data_recall(
                    [_val(r) for r in an.reads], [_val(r) for r in tn.reads]
                ),
                "agent_evidence": _evidence(an.observation_ids, obs_map),
            },
            "writes": {
                "truth": [_val(w) for w in tn.writes],
                "agent": [_val(w) for w in an.writes],
                "hit": _data_recall(
                    [_val(w) for w in an.writes], [_val(w) for w in tn.writes]
                ),
                "agent_evidence": _evidence(an.observation_ids, obs_map),
            },
        }
        node_analyses.append(entry)

    return {
        "run": name,
        "termination": d.get("termination_reason"),
        "mapping": mapping,
        "unmatched_truth": [t for t in truth.nodes if t not in rev],
        "node_analyses": node_analyses,
    }


def main():
    all_runs = []
    for p in sorted(glob.glob(str(ART_DIR / "run_*_seed4*.json"))):
        res = analyze_artifact(p)
        all_runs.append(res)
        print("\n" + "=" * 100)
        print("RUN", res["run"], "| term:", res["termination"])
        print("mapping agent->truth:", res["mapping"])
        print("unmatched truth nodes:", res["unmatched_truth"])
        for na in res["node_analyses"]:
            print(f"\n  {na['agent_node']} -> {na['truth_node']}")
            print(f"    action: {na['action_agent']!r}")
            a = na["actor"]
            print(
                f"    actor : T={a['truth']!r} A={a['agent']!r} "
                f"norm(T)={a['norm_truth']!r} norm(A)={a['norm_agent']!r} hit={a['hit']}"
            )
            if not a["hit"]:
                for ev in a["agent_evidence"]:
                    print(f"      actor evidence: {ev[:200]}")
            s = na["system"]
            print(
                f"    system: T={s['truth']!r} A={s['agent']!r} "
                f"norm(T)={s['norm_truth']!r} norm(A)={s['norm_agent']!r} hit={s['hit']}"
            )
            if not s["hit"]:
                for ev in s["agent_evidence"]:
                    print(f"      system evidence: {ev[:200]}")
            r_ = na["reads"]
            print(f"    reads : T={r_['truth']} A={r_['agent']} hit={r_['hit']}")
            w_ = na["writes"]
            print(f"    writes: T={w_['truth']} A={w_['agent']} hit={w_['hit']}")
            if r_["hit"] < 1.0 or w_["hit"] < 1.0:
                for ev in r_["agent_evidence"] or w_["agent_evidence"]:
                    print(f"      evidence: {ev[:200]}")

    out = ART_DIR / "attribute_mismatch_inventory.json"
    with open(out, "w") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
