#!/usr/bin/env python3
"""Manual exploratory smoke experiment for the business_interview domain.

Runs the REAL tau2 pipeline (Interview Agent <-> Stakeholder LLM) on the
``quotation_workflow_1`` scenario a small number of times using DeepSeek for
BOTH the Interview Agent and the Stakeholder LLM. The stakeholder runs through
the knowledge-grounded ``business_interview_user`` simulator: it answers only
from its world model (StakeholderKnowledge) and returns a private sidecar
(``[{semantic_id, quote, occurrence}]`` + dialogue events), which the
environment stores privately per turn (never in Agent-visible state).

The script captures the natural language conversation, the tool calls, the
final inferred graph + glossary, the private semantic ledger (in a separate
``*.private.json`` artifact), and an ``evaluation_inputs`` envelope containing
canonical TruthGraph plus the exact stakeholder Knowledge used by the simulator.
The domain evaluator metrics
(structural/glossary/evidence/quality_pass), the standard tau2 reward, a
private-ID leakage scan, provider/tool error accounting, and bounded explicit
model-refusal diagnostics for every LLM generation attempt. A secondary public
trajectory refusal count is retained for compatibility but is never added to
that call-level count. ``episode_complete`` is recorded separately from
benchmark reconstruction success.

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

# The interview agent and the stakeholder both run on the same model,
# served via an env-configured provider (defaults: DeepSeek V4 Flash 0731
# via OpenRouter). The script is explicitly manual/exploratory; a working
# provider key must be set in the environment. Do not switch these defaults
# to other providers (qwen, deepseek-v3 chat, etc.) for testing.
AGENT_MODEL = __import__("os").environ.get(
    "BI_AGENT_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731"
)
USER_MODEL = __import__("os").environ.get(
    "BI_USER_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731"
)
LLM_ARGS = {"temperature": 0.0}

TASK_ID = "quotation_workflow_1"
USER_IMPLEMENTATION = "business_interview_user"

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "business_interview_real_llm"

_PROVIDER_EXCEPTION_MODULES = ("litellm", "httpx", "openai", "requests")
_PROVIDER_EXCEPTION_NAMES = {
    "apierror",
    "apiconnectionerror",
    "authenticationerror",
    "badrequesterror",
    "connectionerror",
    "connecterror",
    "ratelimiterror",
    "serviceunavailableerror",
    "timeouterror",
    "timeout",
}


def _is_provider_exception(exc: BaseException) -> bool:
    """Identify provider/runtime failures without labeling domain validation."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        module = type(current).__module__.lower()
        if name in _PROVIDER_EXCEPTION_NAMES or module.startswith(
            _PROVIDER_EXCEPTION_MODULES
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


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
    """Render a ConceptRef or an epistemic marker (UNSET/ABSENT/DONT_KNOW)
    for the dump."""
    from tau2.domains.business_interview.graph import (
        is_absent,
        is_dont_know,
        is_unset,
    )

    if ref is None:
        return None
    if is_unset(ref):
        return {"unset": True}
    if is_absent(ref):
        return {
            "absent": True,
            "evidence": _render_evidence(ref.evidence or []),
        }
    if is_dont_know(ref):
        return {
            "dont_know": True,
            "evidence": _render_evidence(ref.evidence or []),
        }
    out = {"concept_id": ref.concept_id}
    if ref.confidence:
        out["confidence"] = round(ref.confidence, 3)
    if ref.evidence:
        out["evidence"] = _render_evidence(ref.evidence)
    return out


def _render_list_slot(slot) -> list | dict | None:
    """Render a reads/writes slot: list of refs, or a whole-property marker
    (UNSET/ABSENT/DONT_KNOW)."""
    from tau2.domains.business_interview.graph import (
        is_absent,
        is_dont_know,
        is_unset,
    )

    if slot is None or is_unset(slot):
        return None if slot is None else {"unset": True}
    if is_absent(slot):
        return {
            "absent": True,
            "evidence": _render_evidence(slot.evidence or []),
        }
    if is_dont_know(slot):
        return {
            "dont_know": True,
            "evidence": _render_evidence(slot.evidence or []),
        }
    return [_render_ref(r) for r in slot]


def graph_to_dict(graph) -> dict:
    """Render an AgentGraph (or Truth graph) for the dump. Tolerant of the
    different concept shapes (AgentConcept vs TruthConcept)."""
    if graph is None:
        return {}
    return {
        "id": graph.id,
        "name": graph.name,
        "source_node_id": getattr(graph, "source_node_id", None),
        "sink_node_id": getattr(graph, "sink_node_id", None),
        "start_node_id": getattr(graph, "start_node_id", None),
        "start_node_ids": list(getattr(graph, "start_node_ids", []) or []),
        "end_node_ids": list(getattr(graph, "end_node_ids", []) or []),
        "concepts": {
            cid: {
                "id": concept.id,
                "kind": concept.kind,
                "display_label": getattr(concept, "display_label", None),
                "description": concept.description,
                "canonical_terms": getattr(concept, "canonical_terms", None),
                "mentions": _render_evidence(getattr(concept, "mentions", []) or []),
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
            for a in getattr(graph, "terminology_agreements", []) or []
        ],
        "nodes": {
            nid: {
                "id": node.id,
                "activity": _render_ref(node.activity),
                "actor": _render_ref(node.actor),
                "system": _render_ref(node.system),
                "reads": _render_list_slot(node.reads),
                "writes": _render_list_slot(node.writes),
                "necessity_rationale": _render_ref(node.necessity_rationale),
                "structural": bool(getattr(node, "is_structural", False)),
                "structural_role": getattr(node, "structural_role", None),
                "protected": bool(getattr(node, "protected", False)),
            }
            for nid, node in graph.nodes.items()
        },
        "edges": {
            eid: {
                "id": edge.id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "condition": _render_ref(edge.condition),
                "edge_kind": getattr(edge, "edge_kind", "business"),
                "structural_only": bool(getattr(edge, "is_structural", False)),
                "protected": bool(getattr(edge, "protected", False)),
                "is_shortcut": bool(getattr(edge, "is_shortcut", False)),
                "contracted_nodes": list(getattr(edge, "contracted_nodes", []) or []),
                "derived_from_edges": list(
                    getattr(edge, "derived_from_edges", []) or []
                ),
                "evidence": _render_evidence(getattr(edge, "evidence", None) or []),
            }
            for eid, edge in graph.edges.items()
        },
        "validation_errors": graph.structure_errors(),
        "is_valid": graph.is_valid,
    }


def knowledge_concept_ids(scenario) -> set[str]:
    """Private semantic ids of the scenario knowledge (leakage scan)."""
    return set(scenario.knowledge.graph.concepts)


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
    check(
        "evaluation_inputs_summary",
        json.dumps(dump.get("evaluation_inputs_summary") or {}),
    )
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
    from tau2.domains.business_interview.artifact_provenance import (
        build_evaluation_inputs,
        serialize_evaluation_inputs,
        serialize_truth_graph,
    )
    from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
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
        max_errors=30,  # v11 strict tools reject bad evidence; recovery needs budget
        max_repeated_questions=3,
        max_repeated_responses=3,
        max_repeated_interactions=3,
        max_stalled_tool_operations=6,
        save_to=None,
    )

    simulation_id = str(uuid.uuid4())
    orchestrator = build_text_orchestrator(
        config, task, seed=seed, simulation_id=simulation_id
    )

    # Install the LLM call metrics collector BEFORE running so every Agent /
    # Stakeholder generation records metrics and bounded refusal diagnostics.
    from tau2.utils.llm_call_metrics import (
        LLMCallMetricsCollector,
        model_refusal_records,
        record_to_dict,
        set_llm_call_metrics_collector,
        slowest_calls,
        summarize_records,
    )

    metrics_collector = LLMCallMetricsCollector()
    set_llm_call_metrics_collector(metrics_collector)

    started = time.time()
    errors: list[str] = []
    provider_errors: list[str] = []
    result = None
    try:
        result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
    except Exception as exc:  # noqa: BLE001 - capture whatever happened
        error_text = "".join(traceback.format_exception_only(type(exc), exc))
        errors.append(error_text)
        if _is_provider_exception(exc):
            provider_errors.append(error_text)
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
    annotations = assertion_ledger.annotations() if assertion_ledger is not None else {}
    alignments = assertion_ledger.alignments() if assertion_ledger is not None else {}
    terminology = assertion_ledger.terminology() if assertion_ledger is not None else {}

    # --- domain evaluator ---------------------------------------------------
    eval_result = None
    truth_graph = None
    scenario = None
    evaluation_inputs = None
    reference_inputs = None
    try:
        scenario = get_scenario(TASK_ID)
        if scenario is None:
            raise ValueError(f"unknown scenario: {TASK_ID}")
        truth_graph = scenario.truth
        actual_knowledge = getattr(
            getattr(orchestrator.user, "_catalog", None), "knowledge", None
        )
        profiles = []
        for index, profile in enumerate(scenario.stakeholder_references):
            profiles.append(
                {
                    "stakeholder_id": profile.stakeholder_id,
                    "stakeholder_name": profile.name,
                    "stakeholder_role": profile.role,
                    "stakeholder": profile.stakeholder,
                    "knowledge": (
                        actual_knowledge
                        if index == 0 and actual_knowledge is not None
                        else profile.knowledge
                    ),
                }
            )
        evaluation_inputs = build_evaluation_inputs(
            truth_graph,
            profiles,
            simulation_seed=seed,
        )
        reference_inputs = [
            {
                "stakeholder_id": profile.stakeholder_id,
                "stakeholder_name": profile.stakeholder_name,
                "stakeholder_role": profile.stakeholder_role,
                "forgetting_configuration": profile.forgetting_configuration,
                "knowledge": profile.knowledge,
            }
            for profile in evaluation_inputs.stakeholders
        ]
        eval_result = (
            evaluate(
                db,
                evaluation_inputs.stakeholders[0].knowledge,
                EvaluationSpec(),
                scenario.stakeholder,
                truth=truth_graph,
                annotations=annotations,
                alignments=alignments,
                terminology=terminology,
                stakeholder_references=reference_inputs,
            ).model_dump(mode="json")
            if db is not None
            else None
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"evaluate failed: {exc}")

    # --- conversation / messages ---------------------------------------------
    from tau2.domains.business_interview.run_metrics import (
        account_model_refusals,
        account_tool_errors,
        provider_error_count,
    )

    trajectory_messages = (
        result.messages if result is not None and result.messages is not None else []
    )
    tool_error_accounting = account_tool_errors(trajectory_messages)
    messages = []
    agent_calls = 0
    if result is not None and result.messages is not None:
        for m in result.messages:
            if getattr(m, "role", None) == "assistant" and (
                getattr(m, "tool_calls", None) or getattr(m, "content", None)
            ):
                agent_calls += 1
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

    public_trajectory_refusals = account_model_refusals(
        trajectory_messages,
        agent_model=resolved_agent_model,
        stakeholder_model=resolved_user_model,
    )
    accepted_observations = len(db.observations) if db is not None else 0

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
        "loop_guard": (
            orchestrator.loop_guard_diagnostics
            if getattr(orchestrator, "loop_guard_diagnostics", None) is not None
            else None
        ),
        "reward_info": reward_info,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "provider_errors": provider_errors,
        "provider_error_count": provider_error_count(provider_errors),
        "tool_error_count": tool_error_accounting["tool_error_count"],
        "tool_error_categories": tool_error_accounting["tool_error_categories"],
        "tool_error_counts_by_tool": tool_error_accounting["tool_error_counts_by_tool"],
        "tool_error_counts_by_category": tool_error_accounting[
            "tool_error_counts_by_category"
        ],
        "agent_calls": agent_calls,
        "accepted_observations": accepted_observations,
        # Filled from every generation attempt below, not only accepted
        # public Agent/Stakeholder trajectory messages.
        "model_refusal_count": 0,
        "model_refusals": [],
        "public_trajectory_refusal_count": len(public_trajectory_refusals),
        "public_trajectory_refusals": public_trajectory_refusals,
        "episode_complete": termination_reason == "episode_complete",
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
        "truth_graph": (
            serialize_truth_graph(truth_graph) if truth_graph is not None else {}
        ),
        "evaluation_inputs_summary": (
            {
                "schema_version": evaluation_inputs.schema_version,
                "seed": evaluation_inputs.seed,
                "truth_graph_fingerprint": evaluation_inputs.truth_graph_fingerprint,
                "stakeholders": [
                    {
                        "stakeholder_id": profile.stakeholder_id,
                        "stakeholder_name": profile.stakeholder_name,
                        "stakeholder_role": profile.stakeholder_role,
                        "stakeholder_knowledge_fingerprint": profile.stakeholder_knowledge_fingerprint,
                    }
                    for profile in evaluation_inputs.stakeholders
                ],
            }
            if evaluation_inputs is not None
            else None
        ),
        "db_messages_ledger": (db.messages if db is not None else []),
        "interview_complete": bool(db.interview_complete) if db is not None else None,
    }

    # --- LLM call metrics (context size + latency per side) ------------------
    set_llm_call_metrics_collector(None)
    llm_records = metrics_collector.records()
    llm_by_side = metrics_collector.by_side()
    llm_call_provider_error_count = sum(
        1 for record in llm_records if record.status == "error"
    )
    dump_provider_error_count = (
        llm_call_provider_error_count
        if llm_call_provider_error_count
        else len(provider_errors)
    )
    dump["provider_error_count"] = dump_provider_error_count
    dump["llm_call_provider_error_count"] = llm_call_provider_error_count
    dump["agent_calls"] = len(llm_by_side.get("agent", []))
    llm_call_metrics = {
        "records": [record_to_dict(r) for r in llm_records],
        "by_side_summary": {
            side: summarize_records(records)
            for side, records in sorted(llm_by_side.items())
        },
        "slowest_calls": slowest_calls(llm_records, n=5),
    }
    dump["llm_call_metrics"] = llm_call_metrics
    model_refusals = model_refusal_records(llm_records)
    dump["model_refusal_count"] = len(model_refusals)
    dump["model_refusals"] = model_refusals
    dump["llm_generation_attempts_by_side"] = {
        side: len(records) for side, records in sorted(llm_by_side.items())
    }

    # --- private-ID leakage scan ---------------------------------------------
    private_ids: set[str] = set()
    if scenario is not None:
        private_ids.update(scenario.knowledge.graph.semantic_ids())
        private_ids.update(knowledge_concept_ids(scenario))
    leakage = _leakage_scan(dump, private_ids)
    dump["private_id_leakage"] = leakage

    # The PRIVATE semantic ledger: kept out of the Agent-visible dump and
    # written to a separate artifact by main().
    private_payload = {
        "task_id": TASK_ID,
        "annotations_by_turn": {
            str(turn): [a.model_dump() for a in ass]
            for turn, ass in annotations.items()
        },
        "alignments_by_turn": {
            str(turn): [e.model_dump() for e in evs] for turn, evs in alignments.items()
        },
        "terminology_by_turn": {
            str(turn): [e.model_dump() for e in evs]
            for turn, evs in terminology.items()
        },
        "evaluation_inputs": (
            serialize_evaluation_inputs(evaluation_inputs)
            if evaluation_inputs is not None
            else None
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
        evaluation_summary = dump.get("evaluation_inputs_summary") or {}
        reference_rows = metrics.get("stakeholder_truth_reference") or []
        summaries.append(
            {
                "run_index": i,
                "run_id": dump["run_id"],
                "seed": seed,
                "truth_graph_fingerprint": evaluation_summary.get(
                    "truth_graph_fingerprint"
                ),
                "stakeholders": [
                    {
                        "stakeholder_id": row.get("stakeholder_id"),
                        "stakeholder_knowledge_fingerprint": row.get(
                            "stakeholder_knowledge_fingerprint"
                        ),
                    }
                    for row in evaluation_summary.get("stakeholders", [])
                ],
                "stakeholder_truth_reference_scores": {
                    row.get("stakeholder_id"): (
                        row.get("truth_reconstruction") or {}
                    ).get("aggregate_score")
                    for row in reference_rows
                },
                "termination_reason": dump["termination_reason"],
                "loop_guard": dump.get("loop_guard"),
                "episode_complete": dump.get("episode_complete"),
                "reward": (dump["reward_info"] or {}).get("reward"),
                "quality_pass": metrics.get("quality_pass"),
                "structural_pass": metrics.get("structural_pass"),
                "glossary_complete": metrics.get("glossary_complete"),
                "evidence_pass": metrics.get("evidence_pass"),
                "provider_error_count": dump.get("provider_error_count"),
                "llm_call_provider_error_count": dump.get(
                    "llm_call_provider_error_count"
                ),
                "provider_errors": dump.get("provider_errors"),
                "tool_error_count": dump.get("tool_error_count"),
                "tool_error_categories": dump.get("tool_error_categories"),
                "tool_error_counts_by_tool": dump.get("tool_error_counts_by_tool"),
                "tool_error_counts_by_category": dump.get(
                    "tool_error_counts_by_category"
                ),
                "agent_calls": dump.get("agent_calls"),
                "accepted_observations": dump.get("accepted_observations"),
                "llm_generation_attempts_by_side": dump.get(
                    "llm_generation_attempts_by_side"
                ),
                "model_refusal_count": dump.get("model_refusal_count"),
                "model_refusals": dump.get("model_refusals"),
                "public_trajectory_refusal_count": dump.get(
                    "public_trajectory_refusal_count"
                ),
                "node_recall": metrics.get("node_recall"),
                "node_precision": metrics.get("node_precision"),
                "edge_recall": metrics.get("edge_recall"),
                "edge_precision": metrics.get("edge_precision"),
                "concept_recall": metrics.get("concept_recall"),
                "concept_precision": metrics.get("concept_precision"),
                "concept_correctness": metrics.get("concept_correctness"),
                "activity_correctness": metrics.get("activity_correctness"),
                "actor_correctness": metrics.get("actor_correctness"),
                "system_correctness": metrics.get("system_correctness"),
                "read_correctness": metrics.get("read_correctness"),
                "write_correctness": metrics.get("write_correctness"),
                "rationale_correctness": metrics.get("rationale_correctness"),
                "condition_correctness": metrics.get("condition_correctness"),
                "reconstruction_pass": metrics.get("reconstruction_pass"),
                "marker_evidence_errors": metrics.get(
                    "marker_evidence_errors_surrogate"
                ),
                "start_correct": metrics.get("start_correct"),
                "end_recall": metrics.get("end_recall"),
                "end_precision": metrics.get("end_precision"),
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
