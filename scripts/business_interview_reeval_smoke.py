"""Re-evaluate saved business_interview smoke artifacts with the current matcher.

Used to demonstrate the node-matching fix: compares the node/edge recall recorded
in each artifact (old matcher) against a fresh evaluate() run (new matcher) on the
same saved final DAG.

Not part of the test suite / CI — a manual diagnostic helper.
"""

import glob
import json

from tau2.domains.business_interview.dag import BusinessDAG, InterviewDB
from tau2.domains.business_interview.evaluation import evaluate
from tau2.domains.business_interview.scenario import quotation_spec, quotation_truth

TRUTH = quotation_truth()
SPEC = quotation_spec()


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


def main():
    print(
        f"{'run':26} {'OLD node_R':>9} {'NEW node_R':>9} {'OLD edge_R':>9} {'NEW edge_R':>9}  term"
    )
    for p in sorted(
        glob.glob("artifacts/business_interview_real_llm/run_*_seed*.json")
    ):
        d = json.load(open(p))
        name = p.split("/")[-1]
        old = d["evaluator_metrics"] or {}
        fdag = dict(d["final_dag"])
        fdag["nodes"] = {nid: _fix_node(n) for nid, n in fdag["nodes"].items()}
        fdag["edges"] = {
            eid: {**e, "predicate": _cv(e.get("predicate"))}
            for eid, e in fdag["edges"].items()
        }
        agent = BusinessDAG.model_validate(fdag)
        db = InterviewDB(
            dag=agent,
            messages=list(d["db_messages_ledger"]),
            observations=[],
            interview_complete=bool(d["interview_complete"]),
        )
        new = evaluate(db, TRUTH, SPEC)
        print(
            f"{name:26} {old.get('node_recall', 0):>9.3f} {new.node_recall:>9.3f} "
            f"{old.get('edge_recall', 0):>9.3f} {new.edge_recall:>9.3f}  {d['termination_reason']}"
        )


if __name__ == "__main__":
    main()
