#!/usr/bin/env python3
"""Manual exploratory smoke experiment for the business_interview domain.

Runs the REAL tau2 pipeline (Interview Agent <-> Stakeholder LLM) on the
``quotation_workflow_1`` scenario a small number of times using DeepSeek for
BOTH the Interview Agent and the Stakeholder LLM. The stakeholder runs through
the fact-grounded ``business_interview_user`` simulator: it answers only from
hidden atomic StakeholderFacts and returns a private assertion sidecar
(``[{fact_id, quote, occurrence}]``), which the environment stores privately
per turn (never in Agent-visible state).

The script captures the natural language conversation, the tool calls, the
final inferred graph + glossary, the private assertion ledger (in a separate
``*.private.json`` artifact), the domain evaluator metrics
(structural/glossary/evidence/quality_pass), the standard tau2 reward, a
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


def _render_evidence(evs) -> list:
    return [
        {
            "observation_id": ev.observation_id,
            "quote": ev.quote,
            "occurrence": ev.occurrence,
        }
        for ev in (evs or [])
    ]


def _render_ref(ref) -> dict | None:
    if ref is None:
        return None
    out = {"concept_id": ref.concept_id}
    if ref.confidence:
        out["confidence"] = round(ref.confidence, 3)
    if ref.evidence:
        out["evidence"] = _render_evidence(ref.evidence)
    return out


def graph_to_dict(graph) -> dict:
    if graph is None:
        return {}
    return {
        "id": graph.id,
        "name": graph.name,
        "start_node_id": graph.start_node_id,
        "end_node_ids": list(graph.end_node_ids),
        "concepts": {
            cid: {
                "id": concept.id,
                "kind": concept.kind,
                "display_label": concept.display_label,
                "description": concept.description,
                "validation_status": concept.validation_status,
                "mentions": _render_evidence(concept.mentions),
                "validation_evidence": _render_evidence(concept.validation_evidence),
            }
            for cid, concept in graph.concepts.items()
        },
        "terminology_agreements": [
            {
                "concept_id": a.concept_id,
                "term": a.term,
                "stakeholder_id": a.stakeholder_id,
                "evidence": _render_evidence(a.evidence),
            }
            for a in graph.terminology_agreements
        ],
        "nodes": {
            nid: {
                "id": node.id,
                "activity": _render_ref(node.activity),
                "actor": _render_ref(node.actor),
                "system": _render_ref(node.system),
                "reads": [_render_ref(r) for r in node.reads],
                "writes": [_render_ref(w) for w in node.writes],
                "necessity_rationale": _render_ref(node.necessity_rationale),
            }
            for nid, node in graph.nodes.items()
        },
        "edges": {
            eid: {
                "id": edge.id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "condition": _render_ref(edge.condition),
                "evidence": _render_evidence(edge.evidence),
            }
            for eid, edge in graph.edges.items()
        },
        "validation_errors": graph.structure_errors(),
        "is_valid": graph.is_valid,
    }


def _leakage_scan(dump: dict, private_ids: set[str]) -> list[str]:
    """Scan every Agent-visible surface for private fact/claim ids.

    Agent-visible surfaces: conversation contents, tool calls, observations,
    summaries, the final graph (labels/terms/descriptions), the DB ledger,
    evaluator metrics. Private ids must never appear on any of them.
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
    for nid, node in (dump.get("final_graph") or {}).get("nodes", {}).items():
        check(f"final_graph.nodes[{nid}]", json.dumps(node))
    for cid, concept in (dump.get("final_graph") or {}).get("concepts", {}).items():
        check(f"final_graph.concepts[{cid}]", json.dumps(concept))
    for eid, edge in (dump.get("final_graph") or {}).get("edges", {}).items():
        check(f"final_graph.edges[{eid}]", json.dumps(edge))
    for i, m in enumerate(dump.get("db_messages_ledger") or []):
        check(f"db_messages_ledger[{i}]", json.dumps(m))
    check("summary", str(dump.get("summary") or ""))
    check("evaluator_metrics", json.dumps(dump.get("evaluator_metrics") or {}))
    check("reward_info", json.dumps(dump.get("reward_info") or {}))
    return leaks


def run_once(run_index: int, seed: int) -> tuple[dict, dict]:
    """Run quotation_workflow_1 once with DeepSeek.

    Returns ``(public_dump, private_payload)``: the Agent-visible dump and the
    evaluator-only private assertion ledger (kept in a separate artifact).
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

    # Capture the interview DB and the PRIVATE assertion ledger from the live
    # environment. The ledger is never part of the DB and never Agent-visible.
    db = None
    env_tools = getattr(orchestrator.environment, "tools", None)
    try:
        db = getattr(env_tools, "db", None)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"could not read db: {exc}")
    assertion_ledger = getattr(orchestrator.environment, "assertion_ledger", None)
    assertions = assertion_ledger.assertions() if assertion_ledger is not None else {}
    alignments = assertion_ledger.alignments() if assertion_ledger is not None else {}
    terminology = (
        assertion_ledger.terminology() if assertion_ledger is not None else {}
    )

    # --- domain evaluator ---------------------------------------------------
    eval_result = None
    truth_graph = None
    scenario = None
    try:
        scenario = get_scenario(TASK_ID)
        if scenario is None:
            raise ValueError(f"unknown scenario: {TASK_ID}")
        truth_graph = scenario.truth
        eval_result = (
            evaluate(
                db,
                scenario.truth,
                scenario.spec,
                scenario.stakeholder,
                claims=scenario.claims,
                assertions=assertions,
                alignments=alignments,
                terminology=terminology,
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
        "final_graph": graph_to_dict(db.graph) if db is not None else {},
        "evaluator_metrics": eval_result,
        "truth_graph": graph_to_dict(truth_graph) if truth_graph is not None else {},
        "db_messages_ledger": (db.messages if db is not None else []),
        "interview_complete": bool(db.interview_complete) if db is not None else None,
    }

    # --- private-ID leakage scan ---------------------------------------------
    private_ids: set[str] = set()
    if scenario is not None:
        private_ids.update(scenario.claims.keys())
    leakage = _leakage_scan(dump, private_ids)
    dump["private_id_leakage"] = leakage

    # The PRIVATE assertion ledger + claims: kept out of the Agent-visible dump
    # and written to a separate artifact by main().
    private_payload = {
        "task_id": TASK_ID,
        "assertions_by_turn": {
            str(turn): [a.model_dump() for a in ass] for turn, ass in assertions.items()
        },
        "alignments_by_turn": {
            str(turn): [e.model_dump() for e in evs] for turn, evs in alignments.items()
        },
        "terminology_by_turn": {
            str(turn): [e.model_dump() for e in evs] for turn, evs in terminology.items()
        },
        "visible_claims": (
            {cid: claim.model_dump() for cid, claim in scenario.claims.items()}
            if scenario is not None
            else {}
        ),
        "concept_views": (
            scenario.knowledge.concept_views if scenario is not None else {}
        ),
    }
    dump["private_assertions_artifact"] = f"run_{run_index:02d}_seed{seed}.private.json"
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
        # Private assertion ledger (per turn) — evaluator-only, kept separate
        # from the Agent-visible dump.
        try:
            with open(private_path, "w", encoding="utf-8") as fp:
                json.dump(private_payload, fp, indent=2, ensure_ascii=False)
        except OSError as exc:
            raise SystemExit(f"cannot write {private_path}: {exc}")
        logger.info("Wrote {} (+ private ledger {})", out_path, private_path)

        metrics = dump.get("evaluator_metrics") or {}
        summaries.append(
            {
                "run_index": i,
                "run_id": dump["run_id"],
                "seed": seed,
                "termination_reason": dump["termination_reason"],
                "reward": (dump["reward_info"] or {}).get("reward"),
                "quality_pass": metrics.get("quality_pass"),
                "structural_pass": metrics.get("structural_pass"),
                "glossary_pass": metrics.get("glossary_pass"),
                "evidence_pass": metrics.get("evidence_pass"),
                "node_recall": metrics.get("node_recall"),
                "node_precision": metrics.get("node_precision"),
                "edge_recall": metrics.get("edge_recall"),
                "edge_precision": metrics.get("edge_precision"),
                "concept_correctness": metrics.get("concept_correctness"),
                "start_correct": metrics.get("start_correct"),
                "end_recall": metrics.get("end_recall"),
                "fabricated_node_count": metrics.get("fabricated_node_count"),
                "fabricated_edge_count": metrics.get("fabricated_edge_count"),
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
