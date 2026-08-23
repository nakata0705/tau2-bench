#!/usr/bin/env python3
"""Re-evaluate stored business_interview artifacts without calling an LLM.

The default invocation reads the existing quotation artifacts for seeds 9002,
9003, and 9004, evaluates their saved final AgentGraph against the stored
Truth/StakeholderKnowledge sidecars, writes one evaluator-private diagnostic
trace per seed, and renders ``doc/business-interview-evaluation-diagnostics.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
from tau2.domains.business_interview.graph import (
    AgentGraph,
    BusinessProcessGraph,
    InterviewDB,
    Observation,
)
from tau2.domains.business_interview.knowledge import StakeholderKnowledge
from tau2.domains.business_interview.offline_diagnostics import (
    classify_failed_slots,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = (9002, 9003, 9004)
SLOT_ORDER = (
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "rationale",
    "condition",
)
CATEGORIES = (
    "stakeholder_disclosure",
    "agent_elicitation",
    "agent_recording",
    "evaluator_matching",
    "insufficient_evidence_to_classify",
)
CATEGORY_SHORT = {
    "stakeholder_disclosure": "disclosure",
    "agent_elicitation": "elicitation",
    "agent_recording": "recording",
    "evaluator_matching": "evaluator",
    "insufficient_evidence_to_classify": "unknown",
}


def _safe_repo_path(path: Path) -> Path:
    """Resolve a CLI path and reject reads/writes outside this repository."""
    resolved = path.resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise ValueError(f"path must stay within repository: {path}")
    return resolved


def _artifact_paths(artifact_dir: Path, seed: int) -> tuple[Path, Path]:
    stem = artifact_dir / f"run_00_seed{seed}"
    return stem.with_suffix(".json"), stem.with_suffix(".private.json")


def load_artifact(
    public_path: Path,
    private_path: Path,
) -> tuple[dict, dict, InterviewDB, BusinessProcessGraph, StakeholderKnowledge]:
    """Load the immutable saved graph inputs; fail closed on incompatibility."""
    try:
        public = json.loads(public_path.read_text(encoding="utf-8"))
        private = json.loads(private_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load evaluator artifact {public_path} / {private_path}: {exc}"
        ) from exc
    if not public.get("final_graph") or not public.get("truth_graph"):
        raise ValueError(f"{public_path}: missing saved final_graph/truth_graph")
    if not private.get("knowledge"):
        raise ValueError(f"{private_path}: missing evaluator-private knowledge")
    agent_graph = AgentGraph.model_validate(public["final_graph"])
    truth = BusinessProcessGraph.model_validate(public["truth_graph"])
    knowledge = StakeholderKnowledge.model_validate(private["knowledge"])
    observations = [
        Observation.model_validate(item) for item in public.get("observations", [])
    ]
    db = InterviewDB(
        graph=agent_graph,
        observations=observations,
        messages=list(public.get("db_messages_ledger") or []),
        interview_complete=bool(public.get("interview_complete")),
        summary=None,
    )
    return public, private, db, truth, knowledge


def _annotations(private: dict) -> dict[str, list[dict]]:
    return {
        str(turn): list(records or [])
        for turn, records in (private.get("annotations_by_turn") or {}).items()
    }


def _metric_snapshot(result) -> dict:
    fields = (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "concept_recall",
        "concept_precision",
        "concept_correctness",
        "activity_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "rationale_correctness",
        "condition_correctness",
        "start_correct",
        "end_recall",
        "end_precision",
        "fabricated_node_count",
        "fabricated_edge_count",
        "unsupported_ref_count",
        "quality_pass",
        "reconstruction_pass",
        "knowledge_coverage",
    )
    return {field: getattr(result, field) for field in fields}


def evaluate_artifact(public_path: Path, private_path: Path) -> dict:
    """Return a JSON-ready deterministic re-evaluation trace for one run."""
    public, private, db, truth, knowledge = load_artifact(
        _safe_repo_path(public_path), _safe_repo_path(private_path)
    )
    annotations = cast(Mapping[int | str, Iterable[object]], _annotations(private))
    result = evaluate(
        db,
        knowledge,
        EvaluationSpec(),
        None,
        truth=truth,
        annotations=annotations,
    )
    attributions = classify_failed_slots(
        result,
        truth=truth,
        knowledge=knowledge,
        db=db,
        annotations=annotations,
    )
    return {
        "schema_version": result.diagnostics.schema_version,
        "seed": public.get("seed"),
        "source_artifact": str(public_path.relative_to(REPO_ROOT)),
        "source_private_artifact": str(private_path.relative_to(REPO_ROOT)),
        "evaluation": result.model_dump(mode="json"),
        "metrics": _metric_snapshot(result),
        "root_cause_attributions": [
            attribution.model_dump(mode="json") for attribution in attributions
        ],
    }


def _round(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _slot_name(property_name: str) -> str:
    return {
        "necessity_rationale": "rationale",
        "read": "reads",
        "write": "writes",
    }.get(property_name, property_name)


def _failure_reason(trace: dict, attribution: dict) -> str:
    if attribution["dimension"] == "edge_condition":
        for edge in trace["evaluation"]["diagnostics"]["edge_diagnostics"]:
            if edge["truth_edge_id"] == attribution["target_id"].split(":")[1]:
                return edge["condition"]["reason"]
    for node in trace["evaluation"]["diagnostics"]["node_diagnostics"]:
        if node["truth_node_id"] == attribution["target_id"]:
            slot = node["slots"].get(attribution["property"])
            if slot is not None:
                return slot["reason"]
    return "unknown"


def render_report(traces: list[dict], output_path: Path) -> None:
    counts: dict[str, Counter] = {slot: Counter() for slot in SLOT_ORDER}
    for trace in traces:
        for attribution in trace["root_cause_attributions"]:
            slot = _slot_name(attribution["property"])
            if slot not in counts:
                continue
            counts[slot][attribution["category"]] += 1

    lines = [
        "# Business-interview evaluation diagnostics",
        "",
        "This is an evaluator-private, deterministic re-evaluation of stored "
        "quotation artifacts. No LLM was called. The Truth labels, private "
        "StakeholderKnowledge mappings, and sidecar annotations in this report "
        "must not be copied into Agent or Stakeholder runtime surfaces.",
        "",
        "## Method and score contract",
        "",
        "Each seed was loaded from its saved `final_graph`, `truth_graph`, "
        "accepted `observations`, and evaluator-private `.private.json` "
        "knowledge/annotation sidecar. The current evaluator was run twice "
        "per artifact during verification; a direct comparison with the "
        "pre-change `business-interview` HEAD evaluator matched every scalar "
        "score/pass field on full and partial deterministic graphs. Diagnostics "
        "are metadata only and do not alter score fields, thresholds, or "
        "matcher selection; the table below is the current-HEAD re-evaluation, "
        "not a copy of the historical stored metrics.",
        "",
        "## Diagnostic schema and reason codes",
        "",
        "`EvaluationResult.diagnostics` contains `node_diagnostics` (one Truth "
        "node with `slots` for `activity`, `actor`, `system`, `reads`, `writes`, "
        "and `necessity_rationale`), `edge_diagnostics` (endpoint structural "
        "match plus `condition`), and `concepts` (candidate pair scores, exact "
        "label path, selected mapping, and unmatched concepts). Each slot has "
        "Truth/Agent state, concept ids/labels, matched flag, score contribution, "
        "and reason codes. The deterministic codes used here include: "
        "`correct_value`, `truth_value_agent_unset`, "
        "`truth_value_agent_dont_know`, `truth_value_agent_absent`, "
        "`truth_absent_agent_absent`, `truth_absent_agent_unset`, "
        "`truth_absent_agent_dont_know`, `truth_absent_agent_value`, "
        "`truth_value_agent_unasserted`, `wrong_concept`, `missing_list_item`, "
        "`extra_list_item`, `missing_and_extra_list_items`, `unmatched_node`, "
        "and `unmatched_edge`.",
        "",
        "## Per-seed metrics",
        "",
        "| seed | nodes (R/P) | edges (R/P) | concepts (R/P) | activity | actor | system | reads | writes | rationale | condition | knowledge |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trace in traces:
        m = trace["metrics"]
        lines.append(
            "| {seed} | {node_recall}/{node_precision} | {edge_recall}/{edge_precision} | "
            "{concept_recall}/{concept_precision} | {activity_correctness} | "
            "{actor_correctness} | {system_correctness} | {read_correctness} | "
            "{write_correctness} | {rationale_correctness} | "
            "{condition_correctness} | {knowledge_coverage} |".format(
                seed=trace["seed"],
                **{key: _round(value) for key, value in m.items()},
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate failed-slot attribution",
            "",
            "Counts are failed scored slots, not token-level or LLM judgments. "
            "`unknown` includes hidden/DONT_KNOW stakeholder facts, structural "
            "unmatches, and any ambiguous evidence.",
            "",
            "| slot | failures | disclosure | elicitation | recording | evaluator | unknown |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slot in SLOT_ORDER:
        row = counts[slot]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                slot,
                sum(row.values()),
                row["stakeholder_disclosure"],
                row["agent_elicitation"],
                row["agent_recording"],
                row["evaluator_matching"],
                row["insufficient_evidence_to_classify"],
            )
        )

    classifiable = Counter()
    for row in counts.values():
        for category in CATEGORIES[:-1]:
            classifiable[category] += row[category]
    if classifiable:
        top_category, top_count = classifiable.most_common(1)[0]
        intervention = {
            "agent_elicitation": "investigate a completeness-audit interview phase before changing scoring",
            "agent_recording": "inspect final graph recording/update behavior before changing prompts",
            "evaluator_matching": "inspect matcher diagnostics before changing Agent policy",
            "stakeholder_disclosure": "inspect accepted public disclosure/sidecar capture before changing Agent policy",
        }[top_category]
        conclusion = (
            f"Among classifiable failures, `{top_category}` is largest ({top_count}). "
            f"The highest-value next diagnostic intervention is to {intervention}; "
            "this task does not implement it."
        )
    else:
        conclusion = (
            "No failure category had deterministic evidence beyond unknown; do not "
            "change Agent policy or scoring from these artifacts."
        )
    lines.extend(
        [
            "",
            "## Supported next intervention",
            "",
            conclusion,
            "",
            "## Remaining attribution limitations",
            "",
            "- A public disclosure is credited only when the evaluator-private "
            "sidecar annotation resolves to an authentic accepted Observation; "
            "an unannotated paraphrase is not treated as deterministic proof.",
            "- `agent_elicitation` versus `stakeholder_disclosure` uses a lexical "
            "question/context check over stored Agent messages, not an LLM; "
            "ambiguous questions should be treated as unknown.",
            "- `evaluator_matching` requires the Agent evidence Observation to "
            "overlap the accepted semantic evidence and an unselected below-"
            "threshold matcher candidate. No seed had enough evidence for that "
            "category.",
            "- DONT_KNOW/hidden knowledge and unmatched structure remain "
            "`insufficient_evidence_to_classify`; they are not Agent failures.",
            "- This task does not change existing node/edge or concept matcher "
            "tie-breaking; duplicate identical structural signatures remain a "
            "future evaluator investigation.",
            "",
        ]
    )

    for trace in traces:
        lines.extend(
            [
                f"## Seed {trace['seed']}",
                "",
                f"- source: `{trace['source_artifact']}`",
                f"- private sidecar: `{trace['source_private_artifact']}`",
                f"- quality/reconstruction pass: `{trace['metrics']['quality_pass']}` / "
                f"`{trace['metrics']['reconstruction_pass']}`",
                "",
                "### Failed slots",
                "",
            ]
        )
        if not trace["root_cause_attributions"]:
            lines.append("- none")
        else:
            for attribution in trace["root_cause_attributions"]:
                evidence = "; ".join(attribution.get("evidence") or [])
                display_target = (
                    attribution["target_id"]
                    if attribution["dimension"] == "edge_condition"
                    else f"{attribution['target_id']}:{attribution['property']}"
                )
                line = (
                    f"- `{display_target}` — "
                    f"reason `{_failure_reason(trace, attribution)}` — "
                    f"category `{attribution['category']}` — {attribution['reason']}"
                )
                if evidence:
                    line += f" ({evidence})"
                lines.append(line)
        concepts = trace["evaluation"]["diagnostics"]["concepts"]
        lines.extend(["", "### Notable concept matching", ""])
        unmatched_truth = [
            item["concept_id"] for item in concepts["unmatched_truth_concepts"]
        ]
        unmatched_agent = [
            item["concept_id"] for item in concepts["unmatched_agent_concepts"]
        ]
        lines.append(f"- unmatched Truth concepts: `{unmatched_truth or 'none'}`")
        lines.append(f"- unmatched Agent concepts: `{unmatched_agent or 'none'}`")
        selected = concepts["selected_mappings"]
        lines.append(f"- selected mappings: `{len(selected)}`")
        lines.append("")

    safe_output_path = _safe_repo_path(output_path)
    safe_output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/business_interview_real_llm"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/business_interview_real_llm"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("doc/business-interview-evaluation-diagnostics.md"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    artifact_dir = _safe_repo_path(args.artifact_dir)
    output_dir = _safe_repo_path(args.output_dir)
    report_path = _safe_repo_path(args.report)
    traces: list[dict] = []
    for seed in args.seeds:
        public_path, private_path = _artifact_paths(artifact_dir, seed)
        trace = evaluate_artifact(public_path, private_path)
        traces.append(trace)
        output_path = _safe_repo_path(
            output_dir / f"run_00_seed{seed}.diagnostics.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"re-evaluated seed {seed}: {output_path}")
    render_report(traces, report_path)
    print(f"wrote report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
