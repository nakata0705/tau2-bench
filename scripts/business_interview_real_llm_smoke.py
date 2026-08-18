#!/usr/bin/env python3
"""Manual exploratory smoke experiment for the business_interview domain.

Runs the REAL tau2 pipeline (Interview Agent <-> Stakeholder LLM) on the
``quotation_workflow_1`` scenario a small number of times using DeepSeek for
BOTH the Interview Agent and the Stakeholder LLM. It captures the natural
language conversation, the tool calls, the final inferred DAG, the domain
evaluator metrics (structural/necessity/evidence/quality_pass), the standard
tau2 reward, and any errors.

This is an EXPLORATORY, MANUAL experiment only:
- It is NOT part of the pytest suite.
- It is NOT part of `make test` or `make check-all`.
- It is NOT a CI gate.
It makes live, non-deterministic API calls and costs money, so it must be
invoked explicitly by a human.

Usage:
    uv run python scripts/business_interview_real_llm_smoke.py --runs 5 --seed-base 1000

Outputs are written to:
    artifacts/business_interview_real_llm/
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import uuid
from pathlib import Path

from loguru import logger

# DeepSeek for both sides of the interview.
# In this local environment `deepseek/deepseek-chat` resolves to `deepseek-v4-flash`
# (confirmed via a live litellm completion test).
AGENT_MODEL = "deepseek/deepseek-chat"
USER_MODEL = "deepseek/deepseek-chat"
LLM_ARGS = {"temperature": 0.0}

TASK_ID = "quotation_workflow_1"

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "business_interview_real_llm"


def _iter_inferred(attr):
    """Yield a human-readable rendering of an InferredValue or list thereof."""
    if attr is None:
        return None
    if isinstance(attr, list):
        return [_iter_inferred(a) for a in attr]
    value = getattr(attr, "value", None)
    if value is None:
        return None
    out = {"value": value}
    conf = getattr(attr, "confidence", None)
    if conf:
        out["confidence"] = round(float(conf), 3)
    obs = getattr(attr, "observation_ids", None)
    if obs:
        out["observation_ids"] = list(obs)
    return out


def dag_to_dict(dag) -> dict:
    if dag is None:
        return {}
    return {
        "id": dag.id,
        "name": dag.name,
        "start_node_id": dag.start_node_id,
        "end_node_ids": list(dag.end_node_ids),
        "nodes": {
            nid: {
                "id": node.id,
                "action": _iter_inferred(node.action),
                "primitive": _iter_inferred(node.primitive),
                "actor": _iter_inferred(node.actor),
                "system": _iter_inferred(node.system),
                "reads": _iter_inferred(node.reads),
                "writes": _iter_inferred(node.writes),
                "necessity": (
                    {
                        p: _iter_inferred(getattr(node.necessity, p))
                        for p in ("rationale", "owner", "evidence", "removal_impact")
                    }
                    if node.necessity is not None
                    else None
                ),
                "observation_ids": list(node.observation_ids),
            }
            for nid, node in dag.nodes.items()
        },
        "edges": {
            eid: {
                "id": edge.id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "predicate": _iter_inferred(edge.predicate),
                "observation_ids": list(edge.observation_ids),
            }
            for eid, edge in dag.edges.items()
        },
        "validation_errors": dag.validate(),
        "is_valid": dag.is_valid,
    }


def run_once(run_index: int, seed: int) -> dict:
    """Run quotation_workflow_1 once with DeepSeek and return a full dump."""
    # Importing inside the function keeps the script import-light and explicit.
    from tau2.data_model.simulation import TextRunConfig
    from tau2.domains.business_interview.evaluation import evaluate
    from tau2.domains.business_interview.scenario import get_scenario
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.build import build_text_orchestrator
    from tau2.runner.simulation import run_simulation

    # --- task ---------------------------------------------------------------
    tasks = registry.get_tasks_loader("business_interview")(task_split_name="base_en")
    task = next(t for t in tasks if t.id == TASK_ID)

    config = TextRunConfig(
        domain="business_interview",
        agent="llm_agent",
        user="user_simulator",
        llm_agent=AGENT_MODEL,
        llm_user=USER_MODEL,
        llm_args_agent=dict(LLM_ARGS),
        llm_args_user=dict(LLM_ARGS),
        num_trials=1,
        seed=seed,
        max_steps=200,
        max_errors=10,
        save_to=None,
    )

    simulation_id = str(uuid.uuid4())
    orchestrator = build_text_orchestrator(
        config, task, seed=seed, simulation_id=simulation_id
    )

    started = time.time()
    errors: list[str] = []
    try:
        result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    except Exception as exc:  # noqa: BLE001 - capture whatever happened
        errors.append("".join(traceback.format_exception_only(type(exc), exc)))
        logger.exception("simulation raised")
        result = None
    elapsed = time.time() - started

    # Capture the interview DB from the live environment (domain state).
    db = None
    try:
        db = orchestrator.environment.tools.db
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not read db: {exc}")

    # --- domain evaluator ---------------------------------------------------
    eval_result = None
    truth_dag = None
    spec_dump = None
    try:
        scenario = get_scenario(TASK_ID)
        truth_dag = scenario.truth
        spec_dump = scenario.spec.model_dump(mode="json")
        eval_result = (
            evaluate(
                db, scenario.truth, scenario.spec, scenario.stakeholder
            ).model_dump(mode="json")
            if db is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"evaluate failed: {exc}")

    # --- conversation / messages ---------------------------------------------
    messages = []
    if result is not None:
        for m in result.messages:
            messages.append(
                {
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", None),
                    "turn_idx": getattr(m, "turn_idx", None),
                    "tool_calls": (
                        [
                            {
                                "name": getattr(tc, "name", None),
                                "arguments": getattr(tc, "arguments", None),
                            }
                            for tc in (getattr(m, "tool_calls", None) or [])
                        ]
                        or None
                    ),
                }
            )

    reward_info = None
    termination_reason = None
    if result is not None:
        rw = result.reward_info
        reward_info = {
            "reward": rw.reward,
            "reward_basis": list(rw.reward_basis or []),
            "reward_breakdown": dict(rw.reward_breakdown or {}),
        }
        termination_reason = result.termination_reason

    dump = {
        "run_index": run_index,
        "run_id": f"run_{run_index:02d}_{simulation_id}",
        "task_id": TASK_ID,
        "seed": seed,
        "agent_model": AGENT_MODEL,
        "user_model": USER_MODEL,
        "llm_args": LLM_ARGS,
        "resolved_agent_model": (
            result.info.agent_info.llm
            if result and getattr(result.info, "agent_info", None)
            else AGENT_MODEL
        ),
        "resolved_user_model": (
            result.info.user_info.llm
            if result and getattr(result.info, "user_info", None)
            else USER_MODEL
        ),
        "termination_reason": termination_reason,
        "reward_info": reward_info,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "conversation": messages,
        "observations": (
            [
                {
                    "id": o.id,
                    "source_id": o.source_id,
                    "text": o.text,
                    "order": o.order,
                    "turn": o.turn,
                }
                for o in db.observations
            ]
            if db is not None
            else []
        ),
        "final_dag": dag_to_dict(db.dag) if db is not None else {},
        "evaluator_metrics": eval_result,
        "truth_dag": dag_to_dict(truth_dag) if truth_dag is not None else {},
        "truth_dag_clean": dag_to_dict(truth_dag) if truth_dag is not None else {},
        "evaluation_spec_hidden": spec_dump,
        "db_messages_ledger": (db.messages if db is not None else []),
        "interview_complete": bool(db.interview_complete) if db is not None else None,
    }
    return dump


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (>=3).")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="Base seed; run i uses seed_base + i.",
    )
    args = parser.parse_args()

    runs = max(args.runs, 1)
    logger.info(
        "Manual smoke experiment: business_interview / quotation_workflow_1 "
        "with DeepSeek (agent+stakeholder). runs={}",
        runs,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i in range(runs):
        seed = args.seed_base + i
        logger.info("Starting run {} / {} (seed={})", i + 1, runs, seed)
        dump = run_once(i, seed)
        out_path = ARTIFACT_DIR / f"run_{i:02d}_seed{seed}.json"
        with open(out_path, "w") as fp:
            json.dump(dump, fp, indent=2, ensure_ascii=False)
        logger.info("Wrote {}", out_path)

        summaries.append(
            {
                "run_index": i,
                "run_id": dump["run_id"],
                "seed": seed,
                "termination_reason": dump["termination_reason"],
                "reward": (dump["reward_info"] or {}).get("reward"),
                "quality_pass": (dump["evaluator_metrics"] or {}).get("quality_pass"),
                "structural_pass": (dump["evaluator_metrics"] or {}).get(
                    "structural_pass"
                ),
                "necessity_pass": (dump["evaluator_metrics"] or {}).get(
                    "necessity_pass"
                ),
                "evidence_pass": (dump["evaluator_metrics"] or {}).get("evidence_pass"),
                "node_recall": (dump["evaluator_metrics"] or {}).get("node_recall"),
                "node_precision": (dump["evaluator_metrics"] or {}).get(
                    "node_precision"
                ),
                "edge_recall": (dump["evaluator_metrics"] or {}).get("edge_recall"),
                "edge_precision": (dump["evaluator_metrics"] or {}).get(
                    "edge_precision"
                ),
                "fabricated_node_count": (dump["evaluator_metrics"] or {}).get(
                    "fabricated_node_count"
                ),
                "fabricated_edge_count": (dump["evaluator_metrics"] or {}).get(
                    "fabricated_edge_count"
                ),
                "errors": dump["errors"],
                "elapsed_seconds": dump["elapsed_seconds"],
                "artifact": str(out_path.relative_to(REPO_ROOT)),
            }
        )

    summary_path = ARTIFACT_DIR / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(
            {
                "task_id": TASK_ID,
                "agent_model": AGENT_MODEL,
                "user_model": USER_MODEL,
                "llm_args": LLM_ARGS,
                "num_runs": len(summaries),
                "runs": summaries,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Summary: {}", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
