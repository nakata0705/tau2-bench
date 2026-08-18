#!/usr/bin/env python3
"""Manual exploratory smoke experiment for the business_interview domain.

Runs the REAL tau2 pipeline (Interview Agent <-> Stakeholder LLM) on the
``quotation_workflow_1`` scenario a small number of times using DeepSeek for
BOTH the Interview Agent and the Stakeholder LLM. The stakeholder runs through
the fact-grounded ``business_interview_user`` simulator: it answers only from
hidden StakeholderFacts and returns a private ``used_fact_ids`` sidecar, which
the environment stores privately per turn (never in Agent-visible state).

The script captures the natural language conversation, the tool calls, the
final inferred DAG, the private sidecar ledger (in a separate
``*.private.json`` artifact), the domain evaluator metrics
(structural/necessity/evidence/quality_pass), the standard tau2 reward, a
private-ID leakage scan, and any errors.

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
USER_IMPLEMENTATION = "business_interview_user"

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "business_interview_real_llm"


def _iter_inferred(attr):
    """Yield a human-readable rendering of an InferredValue, ConceptRef,
    ConceptTerm, or list thereof."""
    if attr is None:
        return None
    if isinstance(attr, list):
        return [_iter_inferred(a) for a in attr]
    if hasattr(attr, "concept_id"):  # ConceptRef
        out = {"concept_id": attr.concept_id}
        conf = getattr(attr, "confidence", None)
        if conf:
            out["confidence"] = round(float(conf), 3)
        obs = getattr(attr, "observation_ids", None)
        if obs:
            out["observation_ids"] = list(obs)
        return out
    if hasattr(attr, "text") and hasattr(attr, "observation_ids"):  # ConceptTerm
        return {"text": attr.text, "observation_ids": list(attr.observation_ids)}
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
        "data_concepts": {
            cid: {
                "id": concept.id,
                "preferred_label": concept.preferred_label,
                "terms": [
                    {"text": t.text, "observation_ids": list(t.observation_ids)}
                    for t in concept.terms
                ],
            }
            for cid, concept in dag.data_concepts.items()
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
        "validation_errors": dag.structure_errors(),
        "is_valid": dag.is_valid,
    }


def _leakage_scan(dump: dict, private_ids: set[str]) -> list[str]:
    """Scan every Agent-visible surface for private fact/claim ids.

    Agent-visible surfaces: conversation contents, tool calls, observations,
    summaries, the final DAG (labels/terms/ids), the DB ledger, evaluator
    metrics. Private ids must never appear on any of them.
    """
    leaks: list[str] = []

    def check(surface: str, blob: str) -> None:
        for pid in private_ids:
            if pid in blob:
                leaks.append(f"{surface} contains private id {pid!r}")

    for i, m in enumerate(dump.get("conversation") or []):
        check(f"conversation[{i}].content", str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            check(f"conversation[{i}].tool_calls", json.dumps(tc))
    for o in dump.get("observations") or []:
        check(f"observation {o.get('id')}", json.dumps(o))
    for nid, node in (dump.get("final_dag") or {}).get("nodes", {}).items():
        check(f"final_dag.nodes[{nid}]", json.dumps(node))
    for cid, concept in (dump.get("final_dag") or {}).get("data_concepts", {}).items():
        check(f"final_dag.data_concepts[{cid}]", json.dumps(concept))
    for eid, edge in (dump.get("final_dag") or {}).get("edges", {}).items():
        check(f"final_dag.edges[{eid}]", json.dumps(edge))
    for i, m in enumerate(dump.get("db_messages_ledger") or []):
        check(f"db_messages_ledger[{i}]", json.dumps(m))
    check("summary", str(dump.get("summary") or ""))
    check("evaluator_metrics", json.dumps(dump.get("evaluator_metrics") or {}))
    check("reward_info", json.dumps(dump.get("reward_info") or {}))
    return leaks


def run_once(run_index: int, seed: int) -> tuple[dict, dict]:
    """Run quotation_workflow_1 once with DeepSeek.

    Returns ``(public_dump, private_payload)``: the Agent-visible dump and the
    evaluator-only private sidecar ledger (kept in a separate artifact).
    """
    # Importing inside the function keeps the script import-light and explicit.
    from tau2.data_model.simulation import TextRunConfig
    from tau2.domains.business_interview.evaluation import evaluate
    from tau2.domains.business_interview.scenario import get_scenario
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.build import build_text_orchestrator
    from tau2.runner.simulation import run_simulation

    # --- task ---------------------------------------------------------------
    tasks = registry.get_tasks_loader("business_interview")("base_en")
    task = next(t for t in tasks if t.id == TASK_ID)

    config = TextRunConfig(  # pyright: ignore[reportCallIssue] - pydantic defaults
        domain="business_interview",
        agent="llm_agent",
        user=USER_IMPLEMENTATION,
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
    result = None
    try:
        result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    except Exception as exc:  # noqa: BLE001 - capture whatever happened
        errors.append("".join(traceback.format_exception_only(type(exc), exc)))
        logger.exception("simulation raised")
    elapsed = time.time() - started

    # Capture the interview DB and the PRIVATE used-fact ledger from the live
    # environment. The ledger is never part of the DB and never Agent-visible.
    db = None
    env_tools = getattr(orchestrator.environment, "tools", None)
    try:
        db = getattr(env_tools, "db", None)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not read db: {exc}")
    fact_ledger = getattr(orchestrator.environment, "fact_ledger", None)
    used_facts = fact_ledger.used_fact_ids() if fact_ledger is not None else {}

    # --- domain evaluator ---------------------------------------------------
    eval_result = None
    truth_dag = None
    spec_dump = None
    scenario = None
    try:
        scenario = get_scenario(TASK_ID)
        if scenario is None:
            raise ValueError(f"unknown scenario: {TASK_ID}")
        truth_dag = scenario.truth
        spec_dump = scenario.spec.model_dump(mode="json")
        eval_result = (
            evaluate(
                db,
                scenario.truth,
                scenario.spec,
                scenario.stakeholder,
                claims=scenario.claims,
                facts=scenario.facts,
                used_facts=used_facts,
            ).model_dump(mode="json")
            if db is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"evaluate failed: {exc}")

    # --- conversation / messages ---------------------------------------------
    messages = []
    if result is not None and result.messages is not None:
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
    if result is not None and result.reward_info is not None:
        rw = result.reward_info
        reward_info = {
            "reward": rw.reward,
            "reward_basis": list(rw.reward_basis or []),
            "reward_breakdown": dict(rw.reward_breakdown or {}),
        }
        termination_reason = result.termination_reason

    resolved_agent_model = AGENT_MODEL
    resolved_user_model = USER_MODEL
    if result is not None and result.info is not None:
        info = result.info
        agent_info = info.get("agent_info") if isinstance(info, dict) else None
        user_info = info.get("user_info") if isinstance(info, dict) else None
        if agent_info is not None and getattr(agent_info, "llm", None):
            resolved_agent_model = agent_info.llm
        if user_info is not None and getattr(user_info, "llm", None):
            resolved_user_model = user_info.llm

    dump = {
        "run_index": run_index,
        "run_id": f"run_{run_index:02d}_{simulation_id}",
        "task_id": TASK_ID,
        "seed": seed,
        "agent_model": AGENT_MODEL,
        "user_model": USER_MODEL,
        "user_implementation": USER_IMPLEMENTATION,
        "llm_args": LLM_ARGS,
        "resolved_agent_model": resolved_agent_model,
        "resolved_user_model": resolved_user_model,
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

    # --- private-ID leakage scan ---------------------------------------------
    private_ids: set[str] = set()
    if scenario is not None:
        private_ids.update(scenario.facts.keys())
        for fact in scenario.facts.values():
            private_ids.update(fact.supported_claim_ids)
    leakage = _leakage_scan(dump, private_ids)
    dump["private_id_leakage"] = leakage

    # The PRIVATE sidecar ledger + fact catalog: kept out of the Agent-visible
    # dump and written to a separate artifact by main().
    private_payload = {
        "task_id": TASK_ID,
        "used_fact_ids_by_turn": used_facts,
        "stakeholder_facts": (
            {fid: fact.model_dump() for fid, fact in scenario.facts.items()}
            if scenario is not None
            else {}
        ),
    }
    dump["private_used_facts_artifact"] = f"run_{run_index:02d}_seed{seed}.private.json"
    return dump, private_payload


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
        "with DeepSeek (agent + fact-grounded stakeholder). runs={}",
        runs,
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i in range(runs):
        seed = args.seed_base + i
        logger.info("Starting run {} / {} (seed={})", i + 1, runs, seed)
        dump, private_payload = run_once(i, seed)
        out_path = ARTIFACT_DIR / f"run_{i:02d}_seed{seed}.json"
        private_path = ARTIFACT_DIR / f"run_{i:02d}_seed{seed}.private.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(dump, fp, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise SystemExit(f"cannot write {out_path}: {exc}")
        # Private sidecar ledger (used_fact_ids per turn) — evaluator-only,
        # kept separate from the Agent-visible dump.
        try:
            with open(private_path, "w", encoding="utf-8") as fp:
                json.dump(private_payload, fp, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise SystemExit(f"cannot write {private_path}: {exc}")
        logger.info("Wrote {} (+ private ledger {})", out_path, private_path)

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
                "private_id_leakage": dump["private_id_leakage"],
                "errors": dump["errors"],
                "elapsed_seconds": dump["elapsed_seconds"],
                "artifact": str(out_path.relative_to(REPO_ROOT)),
            }
        )

    summary_path = ARTIFACT_DIR / "summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "task_id": TASK_ID,
                    "agent_model": AGENT_MODEL,
                    "user_model": USER_MODEL,
                    "user_implementation": USER_IMPLEMENTATION,
                    "llm_args": LLM_ARGS,
                    "num_runs": len(summaries),
                    "runs": summaries,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        raise SystemExit(f"cannot write {summary_path}: {exc}")
    logger.info("Summary: {}", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
