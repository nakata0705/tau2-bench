#!/usr/bin/env python3
"""Diagnose actor/system/read/write mismatches in business_interview smoke runs.

Read-only analysis (no scoring behavior changes). For each saved artifact
(seed 4000-4004) it:
  - reconstructs the agent DAG and the Ground Truth DAG,
  - runs the evaluator's own _match_nodes to get the node mapping,
  - for every matched node compares actor / system / reads / writes between the
    agent node and the Truth node, and records whether the CURRENT evaluator
    (norm_role / norm_system / _data_recall) scores them as a hit,
  - distinguishes **stakeholder-visible** from **stakeholder-hidden**
    attributes (per the scenario StakeholderFilter):
      * visible attribute: compared against Truth as today;
      * hidden attribute: unset is NOT a mismatch (correct epistemic
        restraint); an asserted value is reported as an epistemic/fabrication
        error even if it equals the hidden Truth,
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
from tau2.domains.business_interview.scenario import (
    quotation_sales_filter,
    quotation_spec,
    quotation_truth,
)

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
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}")
    name = Path(path).name
    obs_map = _obs_map(d)
    truth = quotation_truth()
    spec = quotation_spec()
    stakeholder = quotation_sales_filter()
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
        visible = stakeholder.visible_attributes_for(tnid)
        entry = {
            "agent_node": anid,
            "truth_node": tnid,
            "visible_attributes": sorted(visible),
            "action_agent": _val(an.action),
            "action_truth": _val(tn.action),
            "actor": _axis_entry(
                "actor", tnid, visible, an.actor, tn.actor, obs_map, an
            ),
            "system": _axis_entry(
                "system", tnid, visible, an.system, tn.system, obs_map, an
            ),
            "reads": _data_axis_entry(
                "reads", tnid, visible, an.reads, tn.reads, obs_map, an
            ),
            "writes": _data_axis_entry(
                "writes", tnid, visible, an.writes, tn.writes, obs_map, an
            ),
        }
        node_analyses.append(entry)

    return {
        "run": name,
        "termination": d.get("termination_reason"),
        "mapping": mapping,
        "unmatched_truth": [t for t in truth.nodes if t not in rev],
        "node_analyses": node_analyses,
    }


def _axis_entry(axis, tnid, visible, an_val, tn_val, obs_map, an) -> dict:
    """actor/system entry. ``status`` distinguishes visible vs hidden semantics:

    - ``visible``: compared against Truth (hit = match).
    - ``hidden_unset``: stakeholder-hidden; unset is correct (NOT a mismatch).
    - ``hidden_asserted``: stakeholder-hidden; asserting any value is an
      epistemic/fabrication error, even if it equals the hidden Truth.
    """
    if axis not in visible:
        if not an_val.asserted:
            return {
                "visible": False,
                "status": "hidden_unset",
                "hit": True,
                "truth": _val(tn_val),
                "agent": _val(an_val),
                "norm_truth": norm_role(tn_val.value)
                if axis == "actor"
                else norm_system(tn_val.value),
                "norm_agent": norm_role(an_val.value)
                if axis == "actor"
                else norm_system(an_val.value),
                "agent_evidence": _evidence(an_val.observation_ids, obs_map),
                "node_evidence": _evidence(an.observation_ids, obs_map),
            }
        return {
            "visible": False,
            "status": "hidden_asserted",
            "hit": False,
            "truth": _val(tn_val),
            "agent": _val(an_val),
            "norm_truth": norm_role(tn_val.value)
            if axis == "actor"
            else norm_system(tn_val.value),
            "norm_agent": norm_role(an_val.value)
            if axis == "actor"
            else norm_system(an_val.value),
            "agent_evidence": _evidence(an_val.observation_ids, obs_map),
            "node_evidence": _evidence(an.observation_ids, obs_map),
        }
    hit = (
        norm_role(an_val.value) == norm_role(tn_val.value)
        if axis == "actor"
        else norm_system(an_val.value) == norm_system(tn_val.value)
    )
    return {
        "visible": True,
        "status": "hit" if hit else "mismatch",
        "hit": hit,
        "truth": _val(tn_val),
        "agent": _val(an_val),
        "norm_truth": norm_role(tn_val.value)
        if axis == "actor"
        else norm_system(tn_val.value),
        "norm_agent": norm_role(an_val.value)
        if axis == "actor"
        else norm_system(an_val.value),
        "agent_evidence": _evidence(an_val.observation_ids, obs_map),
        "node_evidence": _evidence(an.observation_ids, obs_map),
    }


def _data_axis_entry(axis, tnid, visible, an_list, tn_list, obs_map, an) -> dict:
    """reads/writes entry. Same visible vs hidden semantics as ``_axis_entry``."""
    truth_items = [_val(r) for r in tn_list]
    agent_items = [_val(r) for r in an_list]
    if axis not in visible:
        asserted = any(r.asserted for r in an_list)
        if not asserted:
            return {
                "visible": False,
                "status": "hidden_unset",
                "hit": True,
                "truth": truth_items,
                "agent": agent_items,
                "agent_evidence": _evidence(an.observation_ids, obs_map),
            }
        return {
            "visible": False,
            "status": "hidden_asserted",
            "hit": False,
            "truth": truth_items,
            "agent": agent_items,
            "agent_evidence": _evidence(an.observation_ids, obs_map),
        }
    hit = _data_recall(agent_items, truth_items)
    return {
        "visible": True,
        "status": "hit" if hit >= 1.0 else "mismatch",
        "hit": hit,
        "truth": truth_items,
        "agent": agent_items,
        "agent_evidence": _evidence(an.observation_ids, obs_map),
    }


def _status_label(status: str) -> str:
    """Human label for an axis status (visible hit vs mismatch vs hidden)."""
    return {
        "hit": "hit",
        "mismatch": "MISMATCH",
        "hidden_unset": "hidden-unset (ok)",
        "hidden_asserted": "hidden-ASSERTED (epistemic error)",
    }[status]


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
            print(f"    visible: {na['visible_attributes']}")
            a = na["actor"]
            print(
                f"    actor : T={a['truth']!r} A={a['agent']!r} "
                f"norm(T)={a['norm_truth']!r} norm(A)={a['norm_agent']!r} "
                f"{_status_label(a['status'])}"
            )
            if a["status"] == "mismatch":
                for ev in a["agent_evidence"]:
                    print(f"      actor evidence: {ev[:200]}")
            elif a["status"] == "hidden_asserted":
                print("      epistemic error: hidden actor asserted")
            s = na["system"]
            print(
                f"    system: T={s['truth']!r} A={s['agent']!r} "
                f"norm(T)={s['norm_truth']!r} norm(A)={s['norm_agent']!r} "
                f"{_status_label(s['status'])}"
            )
            if s["status"] == "mismatch":
                for ev in s["agent_evidence"]:
                    print(f"      system evidence: {ev[:200]}")
            elif s["status"] == "hidden_asserted":
                print("      epistemic error: hidden system asserted")
            r_ = na["reads"]
            print(
                f"    reads : T={r_['truth']} A={r_['agent']} "
                f"{_status_label(r_['status'])}"
            )
            w_ = na["writes"]
            print(
                f"    writes: T={w_['truth']} A={w_['agent']} "
                f"{_status_label(w_['status'])}"
            )
            if r_["status"] == "mismatch" or w_["status"] == "mismatch":
                for ev in r_["agent_evidence"] or w_["agent_evidence"]:
                    print(f"      evidence: {ev[:200]}")
            elif r_["status"] == "hidden_asserted" or w_["status"] == "hidden_asserted":
                print("      epistemic error: hidden reads/writes asserted")

    out = ART_DIR / "attribute_mismatch_inventory.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
