#!/usr/bin/env python3
"""Re-evaluate stored business_interview artifacts without calling an LLM.

The default invocation reads the existing quotation artifacts for seeds 9002,
9003, and 9004, evaluates their saved final AgentGraph against the stored
Truth/StakeholderKnowledge sidecars, writes one evaluator-private diagnostic
trace per seed, and renders ``doc/business-interview-evaluation-diagnostics.md``.
The trace also contains per-stakeholder Truth reference scores (reference only),
forgetting/shortcut diagnostics, a diagnostic-only usage-based
concept-alignment experiment conditioned on the evaluator's current node/edge
mapping, and a label-independent joint structural Node/Concept alignment
experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from scripts.business_interview_diagnostics.offline_diagnostics import (
    classify_failed_slots,
    decode_agent_graph,
)
from scripts.business_interview_joint_alignment_audit import (  # pyright: ignore[reportMissingImports]
    build_joint_concept_disagreement_audit,
)
from tau2.domains.business_interview.artifact_provenance import (
    deserialize_evaluation_inputs,
    fingerprint,
    serialize_evaluation_inputs,
)
from tau2.domains.business_interview.evaluation import (
    EvaluationResult,
    EvaluationSpec,
    evaluate,
)
from tau2.domains.business_interview.graph import (
    BusinessProcessGraph,
    InterviewDB,
    Observation,
)
from tau2.domains.business_interview.knowledge import StakeholderKnowledge

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
_FLOAT_TOLERANCE = 1e-12


class MetricParityError(ValueError):
    """Raised when an artifact's stored metrics disagree with re-evaluation."""


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
    if not private.get("knowledge") and not private.get("evaluation_inputs"):
        raise ValueError(f"{private_path}: missing evaluator-private knowledge")
    agent_graph = decode_agent_graph(public["final_graph"])
    saved_inputs = private.get("evaluation_inputs")
    if saved_inputs:
        inputs = deserialize_evaluation_inputs(saved_inputs)
        truth = inputs.truth_graph
        if not inputs.stakeholders:
            raise ValueError(f"{private_path}: evaluation_inputs has no stakeholders")
        knowledge = inputs.stakeholders[0].knowledge
    else:
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


def _metric_snapshot(result: EvaluationResult) -> dict:
    """Return JSON-ready scalar and reference metrics, excluding trace detail."""
    payload = result.model_dump(mode="json")
    payload.pop("diagnostics", None)
    return payload


def _metric_equal(expected, stored) -> bool:
    if isinstance(expected, bool) or isinstance(stored, bool):
        return type(expected) is type(stored) and expected == stored
    if isinstance(expected, (int, float)) and isinstance(stored, (int, float)):
        try:
            expected_float = float(expected)
            stored_float = float(stored)
        except (OverflowError, TypeError, ValueError):
            return False
        return math.isclose(
            expected_float,
            stored_float,
            rel_tol=_FLOAT_TOLERANCE,
            abs_tol=_FLOAT_TOLERANCE,
        )
    return expected == stored


def _check_metric_parity(result: EvaluationResult, stored_metrics) -> dict:
    if not isinstance(stored_metrics, Mapping) or not stored_metrics:
        raise MetricParityError(
            "artifact has no compatible evaluator_metrics; refusing offline attribution"
        )
    expected = _metric_snapshot(result)
    # Historical smoke artifacts predate the additive reference-only fields.
    # Their primary scalar parity is still checked; newly generated artifacts
    # must contain and compare the reference fields as well.
    historical_optional = {
        "stakeholder_truth_reference",
        "stakeholder_truth_reference_aggregate",
    }
    missing = sorted((set(expected) - set(stored_metrics)) - historical_optional)
    if missing:
        raise MetricParityError(
            "artifact evaluator_metrics is missing required fields: "
            + ", ".join(missing)
        )
    checked_fields = sorted(set(expected) & set(stored_metrics))
    differences = [
        {
            "field": field,
            "stored": stored_metrics[field],
            "reevaluated": value,
        }
        for field, value in expected.items()
        if field in checked_fields and not _metric_equal(value, stored_metrics[field])
    ]
    if differences:
        details = "; ".join(
            f"{item['field']}: stored={item['stored']!r}, "
            f"reevaluated={item['reevaluated']!r}"
            for item in differences
        )
        raise MetricParityError("stored metric drift detected: " + details)
    return {
        "status": "matched",
        "compatible": True,
        "checked_fields": checked_fields,
        "differences": [],
    }


def evaluate_artifact(public_path: Path, private_path: Path) -> dict:
    """Return a JSON-ready deterministic re-evaluation trace for one run."""
    public, private, db, truth, knowledge = load_artifact(
        _safe_repo_path(public_path), _safe_repo_path(private_path)
    )
    saved_inputs = private.get("evaluation_inputs")
    evaluation_inputs = (
        deserialize_evaluation_inputs(saved_inputs) if saved_inputs else None
    )
    stakeholder_references = None
    if evaluation_inputs is not None:
        stakeholder_references = [
            {
                "stakeholder_id": profile.stakeholder_id,
                "stakeholder_name": profile.stakeholder_name,
                "stakeholder_role": profile.stakeholder_role,
                "forgetting_configuration": profile.forgetting_configuration,
                "knowledge": profile.knowledge,
            }
            for profile in evaluation_inputs.stakeholders
        ]
    annotations = cast(Mapping[int | str, Iterable[object]], _annotations(private))
    result = evaluate(
        db,
        knowledge,
        EvaluationSpec(),
        None,
        truth=truth,
        annotations=annotations,
        stakeholder_references=stakeholder_references,
    )
    metric_parity = _check_metric_parity(result, public.get("evaluator_metrics"))
    legacy_input_provenance = None
    if evaluation_inputs is None:
        legacy_input_provenance = {
            "capture_mode": "legacy_split_artifact_without_evaluation_inputs",
            "simulation_seed": public.get("seed"),
            "stakeholder_generation_seed": None,
            "forgetting_seed": None,
            "truth_graph_fingerprint": fingerprint(public["truth_graph"]),
            "stakeholder_knowledge_fingerprint": fingerprint(private["knowledge"]),
            "canonical_truth_graph_saved": False,
            "canonical_stakeholder_knowledge_saved": False,
            "score_recomputation_note": (
                "The stored legacy score is re-evaluated from the legacy Truth "
                "and Knowledge payloads. New artifacts use evaluation_inputs "
                "and persist canonical boundary metadata explicitly."
            ),
        }
    attributions = classify_failed_slots(
        result,
        truth=truth,
        knowledge=knowledge,
        db=db,
        annotations=annotations,
    )
    joint_concept_disagreement_audit = build_joint_concept_disagreement_audit(
        db.graph,
        truth,
        result.diagnostics.joint_structural_alignment,
        observations=db.observations,
        seed=public.get("seed"),
    )
    return {
        "schema_version": result.diagnostics.schema_version,
        "seed": public.get("seed"),
        "evaluation_inputs": (
            serialize_evaluation_inputs(evaluation_inputs)
            if evaluation_inputs is not None
            else None
        ),
        "legacy_input_provenance": legacy_input_provenance,
        "source_artifact": str(public_path.relative_to(REPO_ROOT)),
        "source_private_artifact": str(private_path.relative_to(REPO_ROOT)),
        "evaluation": result.model_dump(mode="json"),
        "metrics": _metric_snapshot(result),
        "metric_parity": metric_parity,
        "joint_concept_disagreement_audit": joint_concept_disagreement_audit,
        "root_cause_attributions": [
            attribution.model_dump(mode="json") for attribution in attributions
        ],
    }


def _round(value) -> str:
    if value is None:
        return "—"
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


def _md_value(value) -> str:
    """Render a diagnostic value safely inside a Markdown table cell."""
    if value is None or value == "":
        return "—"
    return str(value).replace("|", r"\|").replace("\n", " ")


def _usage_pair_lookup(usage: dict) -> dict[tuple[str, str], dict]:
    return {
        (pair["agent_concept_id"], pair["truth_concept_id"]): pair
        for pair in usage["candidate_pairs"]
    }


def _usage_detail_lines(trace: dict) -> list[str]:
    """Render per-seed usage evidence and lexical/usage disagreements."""
    usage = trace["evaluation"]["diagnostics"]["usage_alignment"]
    pair_by_ids = _usage_pair_lookup(usage)
    lines = [
        "",
        "### Usage alignment comparison",
        "",
        f"- referenced concepts: Truth `{usage['truth_referenced_concept_count']}`, "
        f"Agent `{usage['agent_referenced_concept_count']}`",
        f"- exact/partial assigned usage matches: `{usage['exact_usage_match_count']}` / "
        f"`{usage['partial_usage_match_count']}`",
        f"- structurally ambiguous concepts: `{usage['structurally_ambiguous_concept_count']}`",
        f"- disagreements with current mapping: `{usage['disagreement_count']}` "
        f"(substantially stronger usage evidence: "
        f"`{usage['usage_substantially_stronger_disagreement_count']}`)",
    ]
    if usage["ambiguity_classes"]:
        lines.extend(["", "#### Structural ambiguity classes", ""])
        for ambiguity in usage["ambiguity_classes"]:
            lines.append(
                f"- `{ambiguity['class_id']}` ({ambiguity['kind']}): Truth "
                f"`{ambiguity['truth_concept_ids']}` <-> Agent "
                f"`{ambiguity['agent_concept_ids']}` at "
                f"`{ambiguity['usage_addresses']}`; identity is unresolved."
            )
    else:
        lines.extend(["", "- structural ambiguity classes: none", ""])

    disagreements = [
        item
        for item in usage["comparisons"]
        if item["classification"] != "same_mapping"
    ]
    lines.extend(
        [
            "",
            "#### Concrete disagreements with the current lexical/content matcher",
            "",
            "| Agent concept | label | current Truth | usage Truth | classification | lexical score | usage F1 | exact usage |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    if not disagreements:
        lines.append("| none | — | — | — | — | — | — | — |")
    else:
        for item in disagreements:
            lines.append(
                "| `{agent_concept_id}` | `{agent_label}` | `{current_truth_concept_id}` | "
                "`{usage_truth_concept_id}` | `{classification}` | "
                "{current_lexical_similarity_score:.3f} | {usage_similarity_score:.3f} | "
                "{usage_exact_usage_match} |".format(
                    agent_concept_id=_md_value(item["agent_concept_id"]),
                    agent_label=_md_value(item["agent_label"]),
                    current_truth_concept_id=_md_value(
                        item["current_truth_concept_id"]
                    ),
                    usage_truth_concept_id=_md_value(item["usage_truth_concept_id"]),
                    classification=_md_value(item["classification"]),
                    current_lexical_similarity_score=item[
                        "current_lexical_similarity_score"
                    ],
                    usage_similarity_score=item["usage_similarity_score"],
                    usage_exact_usage_match=item["usage_exact_usage_match"],
                )
            )

    labels_differ = [
        item
        for item in usage["comparisons"]
        if item["labels_differ"]
        and item["usage_exact_usage_match"]
        and item["current_truth_concept_id"] == item["usage_truth_concept_id"]
    ]
    labels_agree = [
        item for item in usage["comparisons"] if item["labels_agree_but_usage_does_not"]
    ]
    lines.extend(["", "#### Examples where labels differ but usage agrees", ""])
    if not labels_differ:
        lines.append("- none")
    else:
        for item in labels_differ:
            lines.append(
                f"- Agent `{_md_value(item['agent_concept_id'])}` "
                f"({_md_value(item['agent_label'])}) -> Truth "
                f"`{_md_value(item['current_truth_concept_id'])}` with exact usage; "
                "the current lexical path was not an exact label match."
            )
    lines.extend(["", "#### Examples where labels agree but usage does not", ""])
    if not labels_agree:
        lines.append("- none")
    else:
        for item in labels_agree:
            lines.append(
                f"- Agent `{_md_value(item['agent_concept_id'])}` -> Truth "
                f"`{_md_value(item['current_truth_concept_id'])}`: exact label "
                f"match, but usage exact=`{item['current_exact_usage_match']}`, "
                f"usage F1=`{item['current_usage_f1']:.3f}`."
            )

    partial_pairs = []
    for assignment in usage["assignments"]:
        pair = pair_by_ids.get(
            (assignment["agent_concept_id"], assignment["truth_concept_id"])
        )
        if pair is not None and (
            pair["usages_only_in_truth"]
            or pair["usages_only_in_agent"]
            or pair["broader_narrower_relation"] != "none"
        ):
            partial_pairs.append((assignment, pair))
    lines.extend(["", "#### Usage-only differences and broader/narrower evidence", ""])
    if not partial_pairs:
        lines.append("- none among assigned pairs")
    else:
        for assignment, pair in partial_pairs:
            lines.append(
                f"- Agent `{pair['agent_concept_id']}` -> Truth "
                f"`{pair['truth_concept_id']}`: only Truth "
                f"`{pair['usages_only_in_truth'] or 'none'}`, only Agent "
                f"`{pair['usages_only_in_agent'] or 'none'}`; relation "
                f"`{pair['broader_narrower_relation']}`."
            )
    return lines


def _joint_detail_lines(trace: dict) -> list[str]:
    """Render the label-independent joint alignment for one stored seed."""
    joint = trace["evaluation"]["diagnostics"]["joint_structural_alignment"]
    lines = [
        "",
        "### Joint structural alignment",
        "",
        f"- status: `{joint['status']}`; method: `{joint['method']}`; "
        f"assignment uses labels: `{joint['assignment_uses_labels']}`",
        f"- objective: `{joint['objective']['total_score']:.6f}` / "
        f"`{joint['objective']['max_score']:.6f}` "
        f"(normalized `{joint['objective']['normalized_score']:.6f}`)",
        f"- search: exact=`{joint['search']['exact_search']}`, "
        f"bound_hit=`{joint['search']['bound_hit']}`, "
        f"node states=`{joint['search']['node_search_states']}`, "
        f"concept states=`{joint['search']['concept_search_states']}`, "
        f"edge states=`{joint['search']['edge_assignment_states']}`, "
        f"leaves=`{joint['search']['node_leaves_evaluated']}`, "
        f"optimal alternatives observed=`{joint['search']['optimal_solution_count']}`, "
        f"count exact=`{joint['search']['optimal_solution_count_is_exact']}`, "
        f"optimum unique=`{joint['search']['optimum_is_unique']}`",
        f"- objective bounds: lower=`{joint['search']['objective_lower_bound']:.6f}`, "
        f"upper=`{joint['search']['objective_upper_bound']:.6f}`",
        f"- deterministic search-runtime estimate: "
        f"`{joint['search']['runtime_estimate_ms']} ms` "
        "(work-unit estimate, not wall-clock time, so regenerated artifacts "
        "remain deterministic)",
    ]
    if joint.get("error_type"):
        lines.append(f"- diagnostic error type: `{_md_value(joint['error_type'])}`")

    lines.extend(
        [
            "",
            "#### Objective components",
            "",
            "| component | matched | Agent total | Truth total | agreement |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    component_order = (
        "nodes",
        "concepts",
        "activity",
        "actor",
        "system",
        "reads",
        "writes",
        "rationale",
        "condition",
        "process_edges",
        "start_node",
        "end_nodes",
    )
    components = joint["objective"].get("components", {})
    for name in component_order:
        component = components.get(name)
        if component is None:
            continue
        lines.append(
            f"| `{name}` | {component['matched_count']} | "
            f"{component['agent_count']} | {component['truth_count']} | "
            f"{component['agreement']:.6f} |"
        )

    lines.extend(
        [
            "",
            "#### Resolved versus ambiguous entities",
            "",
            f"- invariant Node mappings proven: "
            f"`{joint['invariant_agent_node_to_truth_node'] or 'none'}`",
            f"- invariant Concept mappings proven: "
            f"`{joint['invariant_agent_concept_to_truth_concept'] or 'none'}`",
            f"- representative unmatched Agent Nodes: "
            f"`{joint['unmatched_representative_agent_nodes'] or 'none'}`",
            f"- representative unmatched Truth Nodes: "
            f"`{joint['unmatched_representative_truth_nodes'] or 'none'}`",
            f"- representative unmatched Agent Concepts: "
            f"`{joint['unmatched_representative_agent_concepts'] or 'none'}`",
            f"- representative unmatched Truth Concepts: "
            f"`{joint['unmatched_representative_truth_concepts'] or 'none'}`",
        ]
    )
    for title, classes in (
        ("Node ambiguity classes", joint["node_ambiguity_classes"]),
        ("Concept ambiguity classes", joint["concept_ambiguity_classes"]),
    ):
        lines.extend(["", f"##### {title}", ""])
        if not classes:
            lines.append("- none")
        else:
            for ambiguity in classes:
                lines.append(
                    f"- `{ambiguity['class_id']}`: Agent "
                    f"`{ambiguity['agent_ids']}` <-> Truth "
                    f"`{ambiguity['truth_ids']}`; unmatched Agent "
                    f"`{ambiguity['unmatched_agent_ids'] or 'none'}`, "
                    f"unmatched Truth `{ambiguity['unmatched_truth_ids'] or 'none'}`; "
                    f"`{ambiguity['reason']}`."
                )

    lines.extend(
        [
            "",
            "#### Representative structural mappings",
            "",
            f"- Agent Node -> Truth Node: "
            f"`{joint['representative_agent_node_to_truth_node'] or 'none'}`",
            f"- Agent Concept -> Truth Concept: "
            f"`{joint['representative_agent_concept_to_truth_concept'] or 'none'}`",
            f"- Agent Edge -> Truth Edge: "
            f"`{joint['representative_agent_edge_to_truth_edge'] or 'none'}`",
            "",
            "#### Production versus joint mapping differences",
            "",
            "| Agent entity | production Truth | joint Truth |",
            "| --- | --- | --- |",
        ]
    )
    differences = (
        joint["production_vs_joint_node_differences"]
        + joint["production_vs_joint_concept_differences"]
    )
    if not differences:
        lines.append("| none | — | — |")
    else:
        for item in differences:
            lines.append(
                f"| `{_md_value(item['agent_id'])}` | "
                f"`{_md_value(item['left_truth_id'])}` | "
                f"`{_md_value(item['right_truth_id'])}` |"
            )

    lines.extend(
        [
            "",
            "#### Conditioned usage versus joint differences",
            "",
            "| Agent Concept | usage Truth | joint Truth |",
            "| --- | --- | --- |",
        ]
    )
    usage_differences = joint["usage_vs_joint_concept_differences"]
    if not usage_differences:
        lines.append("| none | — | — |")
    else:
        for item in usage_differences:
            lines.append(
                f"| `{_md_value(item['agent_id'])}` | "
                f"`{_md_value(item['left_truth_id'])}` | "
                f"`{_md_value(item['right_truth_id'])}` |"
            )
    return lines


def _joint_concept_disagreement_audit_lines(traces: list[dict]) -> list[str]:
    audits = [trace["joint_concept_disagreement_audit"] for trace in traces]
    records = [record for audit in audits for record in audit["records"]]
    counts = Counter(record["classification"] for record in records)

    def support_summary(candidate: dict) -> str:
        overlap = sum(item["count"] for item in candidate["overlapping_usage"])
        return f"{candidate['structural_usage_status']} ({overlap} overlap)"

    def component_delta_summary(comparison: dict) -> str:
        changed = []
        for name, delta in comparison["component_deltas"].items():
            value = delta["agreement_delta_joint_minus_production"]
            if abs(value) > _FLOAT_TOLERANCE:
                changed.append(f"{name}:{value:+.3f}")
        return ", ".join(changed) or "none"

    lines = [
        "## Production-vs-joint Concept disagreement audit",
        "",
        "This section audits the seven Concept mapping differences from the saved",
        "real-LLM artifacts. Structural coordinates are computed only after",
        "projecting Agent locations through the representative joint Node/edge",
        "mapping. Labels, canonical terms, descriptions, Observation text,",
        "EvidenceRef text, and semantic sidecars are shown only for human review",
        "and are not inputs to candidate selection, tie-breaking, objective",
        "evaluation, or classification.",
        "",
        "### Aggregate classification",
        "",
        "| classification | count |",
        "| --- | ---: |",
    ]
    for classification in (
        "joint_strongly_supported",
        "production_strongly_supported",
        "structurally_ambiguous",
        "insufficient_structural_evidence",
        "possible_objective_failure",
    ):
        lines.append(f"| `{classification}` | {counts[classification]} |")
    lines.extend(
        [
            "",
            "### Evidence table",
            "",
            "| seed | Agent Concept (display label) | kind | production Truth | joint Truth | production structural evidence | joint structural evidence | production overlap | joint overlap | local objective delta (joint-production) | classification |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        production = record["production_candidate_evidence"]
        joint = record["joint_candidate_evidence"]
        local = record["counterfactual"]["local_candidate_counterfactual"]
        lines.append(
            "| {seed} | `{agent}` ({label}) | `{kind}` | `{production}` | "
            "`{joint_id}` | `{production_status}` | `{joint_status}` | {production_overlap} | "
            "{joint_overlap} | {delta:+.6f} | `{classification}` |".format(
                seed=record["seed"],
                agent=_md_value(record["agent_concept_id"]),
                label=_md_value(record["agent_label"]),
                kind=_md_value(record["concept_kind"]),
                production=_md_value(record["production_truth_concept_id"]),
                joint_id=_md_value(record["joint_truth_concept_id"]),
                production_status=support_summary(production),
                joint_status=support_summary(joint),
                production_overlap=sum(
                    item["count"] for item in production["overlapping_usage"]
                ),
                joint_overlap=sum(item["count"] for item in joint["overlapping_usage"]),
                delta=local["objective_delta_joint_minus_production"],
                classification=record["classification"],
            )
        )

    lines.extend(["", "### Per-disagreement structural evidence", ""])
    for record in records:
        production = record["production_candidate_evidence"]
        joint = record["joint_candidate_evidence"]
        local = record["counterfactual"]["local_candidate_counterfactual"]
        lines.extend(
            [
                f"#### Seed {record['seed']} — `{record['agent_concept_id']}` "
                f"({record['concept_kind']}; display label: "
                f"{_md_value(record['agent_label'])})",
                "",
                f"- production candidate: `{_md_value(record['production_truth_concept_id'])}` "
                f"({support_summary(production)}); joint candidate: "
                f"`{_md_value(record['joint_truth_concept_id'])}` "
                f"({support_summary(joint)})",
                f"- Agent locations: `{_md_value(record['agent_structural_usage']['locations'])}`",
                f"- projected Agent locations: "
                f"`{_md_value(production['projected_agent_locations'])}`",
                f"- production Truth locations: `{_md_value([item['location'] for item in production['truth_locations']])}`; "
                f"joint Truth locations: `{_md_value([item['location'] for item in joint['truth_locations']])}`",
                f"- production overlap: `{_md_value(production['overlapping_usage'])}`; "
                f"Agent-only: `{_md_value(production['agent_only_locations'])}`; "
                f"Truth-only: `{_md_value(production['truth_only_locations'])}`",
                f"- joint overlap: `{_md_value(joint['overlapping_usage'])}`; "
                f"Agent-only: `{_md_value(joint['agent_only_locations'])}`; "
                f"Truth-only: `{_md_value(joint['truth_only_locations'])}`",
                f"- relation consistency: production="
                f"`{_md_value(production['relation_type_consistency'])}`, joint="
                f"`{_md_value(joint['relation_type_consistency'])}`",
                f"- repeated usage support: production="
                f"`{_md_value(production['repeated_usage_support'])}`, joint="
                f"`{_md_value(joint['repeated_usage_support'])}`",
                f"- process topology / mapped endpoint support: production="
                f"`{_md_value(production['process_topology_support'])}`, joint="
                f"`{_md_value(joint['process_topology_support'])}`",
                f"- unsupported candidate: production=`{production['candidate_support']['unsupported_candidate']}`, "
                f"joint=`{joint['candidate_support']['unsupported_candidate']}`; "
                f"alternative optimal mapping: `{record['alternative_optimal_mapping']['status']}`",
                f"- full forced objective delta (joint-production): "
                f"`{record['counterfactual']['full_mapping_comparison']['objective_delta_joint_minus_production']:+.6f}`; "
                f"changed components: `{component_delta_summary(record['counterfactual']['full_mapping_comparison'])}`",
                f"- local candidate counterfactual delta (joint-production): "
                f"`{local['objective_delta_joint_minus_production']:+.6f}`; "
                f"changed components: `{component_delta_summary(local)}`",
                f"- classification: `{record['classification']}` — "
                f"{record['classification_rationale']}",
                "",
            ]
        )

    start_end = next(
        (
            audit["seed_9003_start_end_investigation"]
            for audit in audits
            if audit.get("seed_9003_start_end_investigation") is not None
        ),
        None,
    )
    lines.extend(["### Seed 9003 start/end investigation", ""])
    if start_end is None:
        lines.append("- no seed 9003 audit was generated")
    else:
        lines.extend(
            [
                f"- assessment: `{start_end['assessment']}`",
                f"- Agent declared fields: start=`{start_end['agent_declared_start_node']}`, "
                f"ends=`{start_end['agent_declared_end_nodes']}`",
                f"- Truth declared fields: start=`{start_end['truth_declared_start_node']}`, "
                f"ends=`{start_end['truth_declared_end_nodes']}`",
                f"- directed-topology Agent sources/sinks: "
                f"`{start_end['agent_topology_sources']}` / `{start_end['agent_topology_sinks']}`",
                f"- directed-topology Truth sources/sinks: "
                f"`{start_end['truth_topology_sources']}` / `{start_end['truth_topology_sinks']}`",
                f"- projected Agent sources/sinks: "
                f"`{start_end['projected_agent_topology_sources']}` / "
                f"`{start_end['projected_agent_topology_sinks']}`",
                f"- objective components: start=`{start_end['joint_start_component']}`, "
                f"end=`{start_end['joint_end_component']}`",
                f"- conclusion: {start_end['interpretation']}",
                "",
            ]
        )

    joint_count = counts["joint_strongly_supported"]
    production_count = counts["production_strongly_supported"]
    ambiguous_count = counts["structurally_ambiguous"]
    objective_count = counts["possible_objective_failure"]
    insufficient_count = counts["insufficient_structural_evidence"]
    lines.extend(
        [
            "### Audit conclusion",
            "",
            f"1. Joint is more strongly supported by structural evidence in `{joint_count}` of `{len(records)}` disagreements.",
            f"2. Production is more strongly supported in `{production_count}`.",
            f"3. `{ambiguous_count}` are structurally indistinguishable/ambiguous; `{insufficient_count}` have insufficient positive structural evidence without an objective warning.",
            f"4. `{objective_count}` case(s) raise an objective/admissibility warning: a forced production assignment with zero structural overlap raises the raw objective, primarily through the `concepts` component. This is not treated as structural support for production.",
            "5. **Promotion decision: no.** The joint matcher is useful as an evaluator-private diagnostic, but the present real-LLM evidence does not justify making it the production Concept identity signal. It strongly supports only a minority of disagreements and exposes unsupported-assignment objective behavior.",
            "6. The minimum next evidence is an adversarial evaluation set with independently verified structural correspondences, repeated/parallel/symmetric usages, missing/extra locations, and explicit boundary metadata. Before promotion, the objective/admissible-domain contract must also be tested so unsupported mappings cannot improve the reported objective under a forced counterfactual.",
            "",
        ]
    )
    return lines


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
        "knowledge/annotation sidecar. The smoke artifacts use a custom JSON "
        "marker encoding, so offline restoration explicitly decodes UNSET, "
        "ABSENT, DONT_KNOW, and ConceptRef values instead of passing the graph "
        "through the undiscriminated Pydantic union. Stored scalar evaluator "
        "metrics are compared with a floating-point representation tolerance "
        "before any attribution; a mismatch fails closed and no report is "
        "generated. The current evaluator was run twice per artifact during "
        "verification; a direct comparison with the pre-change `business-interview` "
        "HEAD evaluator matched every scalar score/pass field on full and partial "
        "deterministic graphs. Diagnostics are metadata only and do not alter "
        "score fields, thresholds, or matcher selection; the table below is the "
        "current-HEAD re-evaluation after stored-metric parity, not a copy of "
        "historical metrics. All diagnostic output remains evaluator-private/offline.",
        "",
        "## Diagnostic schema and reason codes",
        "",
        "`EvaluationResult.diagnostics` contains `node_diagnostics` (one Truth "
        "node with `slots` for `activity`, `actor`, `system`, `reads`, `writes`, "
        "and `necessity_rationale`), `edge_diagnostics` (endpoint structural "
        "match plus `condition`), and `concepts` (candidate pair scores, exact "
        "label path, selected mapping, and unmatched concepts). The separate "
        "`usage_alignment` section contains usage candidate sets, per-kind "
        "assignments, ambiguity classes, and current-vs-usage comparisons. The "
        "`joint_structural_alignment` section is a separate label-independent "
        "joint Node/Concept search with typed incidence, process-edge, start, "
        "and end objective components; it is not a production matcher. Each "
        "slot has "
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
            "## Stakeholder knowledge coverage (reference only)",
            "",
            "The Agent Truth reconstruction above remains the only primary score. "
            "The following rows compare each StakeholderKnowledge view directly "
            "with the same Truth business projection; they do not alter Agent "
            "denominators, `quality_pass`, or ranking.",
            "",
            "| seed | stakeholder id | name / role | aggregate Truth score | graph valid | nodes R/P | edges R/P | concepts R/P | activity | actor | system | reads | writes | rationale | condition | forgetting config | contracted nodes | shortcuts |",
            "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for trace in traces:
        references = trace["evaluation"].get("stakeholder_truth_reference", [])
        if not references:
            lines.append(
                f"| {trace['seed']} | none | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        for reference in references:
            m = reference["truth_reconstruction"]
            d = reference["diagnostics"]
            identity = reference["stakeholder_id"]
            label = reference.get("stakeholder_name") or "—"
            role = reference.get("stakeholder_role")
            if role:
                label = f"{label} / {role}"
            forgetting_config = d.get("forgetting_configuration") or {}
            forgetting_label = (
                json.dumps(forgetting_config, sort_keys=True)
                if forgetting_config
                else "—"
            )
            lines.append(
                "| {seed} | `{sid}` | {label} | {aggregate} | {valid} | "
                "{nr}/{np} | {er}/{ep} | {cr}/{cp} | {activity} | {actor} | {system} | "
                "{reads} | {writes} | {rationale} | {condition} | {forgetting} | "
                "{contracted} | {shortcuts} |".format(
                    seed=trace["seed"],
                    sid=_md_value(identity),
                    label=_md_value(label),
                    aggregate=_round(m["aggregate_score"]),
                    valid=m["graph_valid"],
                    nr=_round(m["node_recall"]),
                    np=_round(m["node_precision"]),
                    er=_round(m["edge_recall"]),
                    ep=_round(m["edge_precision"]),
                    cr=_round(m["concept_recall"]),
                    cp=_round(m["concept_precision"]),
                    activity=_round(m["activity_correctness"]),
                    actor=_round(m["actor_correctness"]),
                    system=_round(m["system_correctness"]),
                    reads=_round(m["read_correctness"]),
                    writes=_round(m["write_correctness"]),
                    rationale=_round(m["rationale_correctness"]),
                    condition=_round(m["condition_correctness"]),
                    forgetting=_md_value(forgetting_label),
                    contracted=d.get("contracted_node_count", 0),
                    shortcuts=d.get(
                        "shortcut_edge_count", len(d.get("shortcut_provenance", []))
                    ),
                )
            )
    lines.extend(
        [
            "",
            "### Reference aggregate (reference only)",
            "",
            "| seed | stakeholder count | min | max | mean |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in traces:
        aggregate = trace["evaluation"].get("stakeholder_truth_reference_aggregate", {})
        lines.append(
            "| {seed} | {count} | {min_score} | {max_score} | {mean_score} |".format(
                seed=trace["seed"],
                count=aggregate.get("stakeholder_count", 0),
                min_score=_round(aggregate.get("min_stakeholder_truth_score")),
                max_score=_round(aggregate.get("max_stakeholder_truth_score")),
                mean_score=_round(aggregate.get("mean_stakeholder_truth_score")),
            )
        )
    lines.extend(
        [
            "",
            "Per-stakeholder shortcut provenance is retained in each JSON "
            "`diagnostics.shortcut_provenance` entry, including contracted Truth "
            "nodes and derived Truth edges. A shortcut receives no automatic "
            "direct-edge credit.",
            "",
            "## Usage-based concept alignment (diagnostic only)",
            "",
            "The usage experiment is explicitly named `usage_alignment_conditioned_on_current_node_mapping`. "
            "It translates Agent node/edge addresses through the existing production "
            "node/edge correspondence, then compares deterministic sets of "
            "`node:<id>:<property>` and `edge:<id>:condition` addresses. Empty "
            "mapped signatures are insufficient evidence, not exact matches. "
            "Concept kind is a hard constraint. Per-pair precision, recall, F1, "
            "Jaccard, exact equality, set differences, and strict "
            "broader/narrower relations are retained in the JSON traces.",
            "",
            "Interpretation: given the current node/edge correspondence, this asks "
            "whether graph usage is a better concept-identity signal than labels. "
            "Because that correspondence can itself depend partly on concept "
            "alignment, this does not prove a fully label-independent evaluator; "
            "the scaffold is a stated circularity limitation, not a production "
            "matcher change.",
            "",
            "The one-to-one assignment is per kind and uses only usage F1, with "
            "a deterministic priority bonus for non-empty exact usage equality. "
            "Labels, descriptions, canonical terms, translations, embeddings, "
            "and LLM judgment are not inputs to that assignment. Sorted opaque "
            "ids are used only for reproducible serialization; identical usage "
            "signatures and tied usage-candidate rows are reported as ambiguity "
            "classes rather than resolved by labels. The report-only `substantially "
            "stronger` flag means exact "
            "usage or positive usage F1 where the current mapping has zero usage "
            "support; it is not a production threshold. This is an exploratory "
            "comparison and introduces no score change.",
            "",
            "| seed | Truth concepts | Agent concepts | exact usage | partial usage | ambiguous concepts | mapping disagreements | substantially stronger | labels differ + usage agrees | labels agree + usage differs |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in traces:
        usage = trace["evaluation"]["diagnostics"]["usage_alignment"]
        lines.append(
            "| {seed} | {truth_referenced_concept_count} | "
            "{agent_referenced_concept_count} | {exact_usage_match_count} | "
            "{partial_usage_match_count} | {structurally_ambiguous_concept_count} | "
            "{disagreement_count} | {usage_substantially_stronger_disagreement_count} | "
            "{labels_differ_usage_agrees_count} | "
            "{labels_agree_usage_does_not_count} |".format(seed=trace["seed"], **usage)
        )
    lines.extend(
        [
            "",
            "### Compact per-kind summary",
            "",
            "| seed | kind | Truth | Agent | exact | partial | ambiguous | disagreements | stronger | insufficient |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in traces:
        usage = trace["evaluation"]["diagnostics"]["usage_alignment"]
        for summary in usage["kind_summaries"]:
            lines.append(
                "| {seed} | {kind} | {truth_referenced_concept_count} | "
                "{agent_referenced_concept_count} | {exact_usage_matches} | "
                "{partial_usage_matches} | {structurally_ambiguous_concepts} | "
                "{disagreements_with_current_mapping} | "
                "{usage_substantially_stronger_disagreements} | "
                "{insufficient_usage} |".format(seed=trace["seed"], **summary)
            )

    usage_totals = [
        trace["evaluation"]["diagnostics"]["usage_alignment"] for trace in traces
    ]
    exact_total = sum(item["exact_usage_match_count"] for item in usage_totals)
    partial_total = sum(item["partial_usage_match_count"] for item in usage_totals)
    ambiguity_total = sum(
        item["structurally_ambiguous_concept_count"] for item in usage_totals
    )
    disagreement_total = sum(item["disagreement_count"] for item in usage_totals)
    stronger_total = sum(
        item["usage_substantially_stronger_disagreement_count"] for item in usage_totals
    )
    lines.extend(
        [
            "",
            "## Usage experiment conclusion",
            "",
            f"Across these stored seeds, the usage scaffold produced `{exact_total}` "
            f"exact and `{partial_total}` partial assigned matches, with "
            f"`{ambiguity_total}` structurally ambiguous concepts and "
            f"`{disagreement_total}` comparison disagreements (`{stronger_total}` "
            "where usage had exact or positive-vs-zero support substantially "
            "stronger than the current mapping).",
            "",
            "**Does usage appear strong enough to replace lexical matching? No, not "
            "as a production replacement from these artifacts.** Exact usage is a "
            "strong and useful conditional signal, including cases where labels are "
            "not exact, but partial/missing usage, unmapped Agent addresses, and "
            "ambiguous equivalence classes prevent usage from resolving every "
            "concept. A disagreement is evidence to inspect, not proof that the "
            "usage assignment is correct.",
            "",
            "Usage is insufficient when the current node/edge scaffold leaves a "
            "concept with no mapped addresses, when a concept is absent from some "
            "of its Truth locations, or when multiple concepts share the same "
            "translated address set. The JSON candidate records expose the exact "
            "Truth-only and Agent-only usages and strict broader/narrower relations "
            "for follow-up.",
            "",
            "Usage alone is not a fully label-independent matcher because its "
            "address translation is conditioned on the production node/edge "
            "scaffold. The separate `joint_structural_alignment` section below "
            "tests a bounded joint search instead; it remains diagnostic-only and "
            "does not replace either production or usage mappings.",
            "",
            "## Joint structural alignment (diagnostic only)",
            "",
            "The joint experiment searches Node and asserted Concept mappings "
            "together. It uses only typed incidence, directed process topology, "
            "start/end roles, and one-to-one constraints; Concept kind is hard. "
            "A representative mapping is serialized for audit only; opaque IDs "
            "are used only for deterministic ordering/serialization. Equal "
            "optima are retained as ambiguity classes, and a bounded search never "
            "claims uniqueness. Same-kind pairs with no positive typed support "
            "remain unmatched rather than being assigned arbitrarily. Each "
            "component is F1-style `2*matched/(Agent+Truth)` (empty/empty is "
            "exact), and the total is "
            "the unweighted sum of the auditable components. `runtime estimate` "
            "is a deterministic work-unit estimate rather than wall-clock timing, "
            "allowing this report to be "
            "regenerated byte-for-byte.",
            "",
            "| seed | objective | normalized | Nodes | Concepts | process edges | start | end | exact | unique | optimal alternatives | ambiguous Nodes | ambiguous Concepts | states | runtime estimate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in traces:
        joint = trace["evaluation"]["diagnostics"]["joint_structural_alignment"]
        components = joint["objective"].get("components", {})
        node_component = components.get("nodes", {})
        concept_component = components.get("concepts", {})
        edge_component = components.get("process_edges", {})
        start_component = components.get("start_node", {})
        end_component = components.get("end_nodes", {})
        lines.append(
            "| {seed} | {score:.3f}/{max_score:.3f} | {normalized:.3f} | "
            "{nodes:.3f} | {concepts:.3f} | {edges:.3f} | {start:.3f} | "
            "{end:.3f} | {exact} | {unique} | {alternatives} | "
            "{node_ambiguity} | {concept_ambiguity} | {states} | "
            "{runtime} ms |".format(
                seed=trace["seed"],
                score=joint["objective"]["total_score"],
                max_score=joint["objective"]["max_score"],
                normalized=joint["objective"]["normalized_score"],
                nodes=node_component.get("agreement", 0.0),
                concepts=concept_component.get("agreement", 0.0),
                edges=edge_component.get("agreement", 0.0),
                start=start_component.get("agreement", 0.0),
                end=end_component.get("agreement", 0.0),
                exact=joint["search"]["exact_search"],
                unique=joint["search"]["optimum_is_unique"],
                alternatives=joint["search"]["optimal_solution_count"],
                node_ambiguity=len(joint["node_ambiguity_classes"]),
                concept_ambiguity=len(joint["concept_ambiguity_classes"]),
                states=(
                    joint["search"]["node_search_states"]
                    + joint["search"]["concept_search_states"]
                    + joint["search"]["edge_assignment_states"]
                ),
                runtime=joint["search"]["runtime_estimate_ms"],
            )
        )
    joint_scores = [
        trace["evaluation"]["diagnostics"]["joint_structural_alignment"]
        for trace in traces
    ]
    joint_exact_count = sum(item["search"]["exact_search"] for item in joint_scores)
    joint_node_ambiguity_count = sum(
        bool(item["node_ambiguity_classes"]) for item in joint_scores
    )
    joint_concept_ambiguity_count = sum(
        bool(item["concept_ambiguity_classes"]) for item in joint_scores
    )
    joint_difference_count = sum(
        len(item["production_vs_joint_node_differences"])
        + len(item["production_vs_joint_concept_differences"])
        for item in joint_scores
    )
    lines.extend(
        [
            "",
            "### Joint experiment conclusion",
            "",
            f"The three stored seeds completed an exact bounded-space search in "
            f"`{joint_exact_count}/{len(joint_scores)}` reports; production-vs-joint "
            f"mapping differences numbered `{joint_difference_count}`. Node ambiguity "
            f"classes appeared in `{joint_node_ambiguity_count}` report(s), and "
            f"concept ambiguity classes appeared in `{joint_concept_ambiguity_count}` "
            "report(s).",
            "",
            "**Exact fixtures:** the deterministic synthetic suite shows that "
            "identical graphs align perfectly, arbitrary Agent labels/IDs and "
            "insertion order do not affect structural scores, reads and writes "
            "remain distinct, kind mismatches never map, topology can disambiguate "
            "similar nodes, repeated usage strengthens concept alignment, and "
            "symmetric structures are reported ambiguous. A deliberately "
            "misleading-label fixture is resolved by structure rather than text.",
            "",
            "**Identifiability:** directed topology plus start/end roles makes the "
            "small seed Node skeletons identifiable. Concepts with repeated or "
            "slot-specific usage are usually identifiable; concepts whose usage is "
            "missing, extra, or structurally unsupported remain unmatched rather "
            "than being guessed. Symmetric duplicate subgraphs/concept usages "
            "remain valid ambiguity classes.",
            "",
            "**Viability:** this is viable as an evaluator-private diagnostic and "
            "as a candidate for further experiments, not a production migration. "
            "Before production use, validate objective weighting and edge cases on "
            "larger adversarial graphs, retain explicit optimality bounds, and "
            "measure whether the structural mapping is stable under realistic "
            "missing/extra structure. Existing production scoring and mappings are "
            "unchanged.",
        ]
    )
    lines.extend(_joint_concept_disagreement_audit_lines(traces))
    lines.extend(
        [
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
                f"- stored-metric parity: `{trace['metric_parity']['status']}` "
                f"({len(trace['metric_parity']['checked_fields'])} fields)",
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
        lines.extend(_usage_detail_lines(trace))
        lines.extend(_joint_detail_lines(trace))
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
    output_paths: list[tuple[dict, Path]] = []
    for seed in args.seeds:
        public_path, private_path = _artifact_paths(artifact_dir, seed)
        trace = evaluate_artifact(public_path, private_path)
        traces.append(trace)
        output_paths.append(
            (
                trace,
                _safe_repo_path(output_dir / f"run_00_seed{seed}.diagnostics.json"),
            )
        )
    # Do not leave a partial trace/report set behind if a later seed fails
    # parity: all artifacts must pass before any derived output is written.
    for trace, output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"re-evaluated seed {trace['seed']}: {output_path}")

    audit_records = [
        record
        for trace in traces
        for record in trace["joint_concept_disagreement_audit"]["records"]
    ]
    audit_summary = {
        "schema_version": "business_interview.joint_concept_disagreement_audit.aggregate.v1",
        "seeds": [trace["seed"] for trace in traces],
        "disagreement_count": len(audit_records),
        "classification_counts": dict(
            sorted(
                Counter(record["classification"] for record in audit_records).items()
            )
        ),
        "records": audit_records,
        "seed_9003_start_end_investigation": next(
            (
                trace["joint_concept_disagreement_audit"][
                    "seed_9003_start_end_investigation"
                ]
                for trace in traces
                if trace["joint_concept_disagreement_audit"].get(
                    "seed_9003_start_end_investigation"
                )
                is not None
            ),
            None,
        ),
    }
    aggregate_audit_path = _safe_repo_path(
        output_dir / "joint_concept_disagreement_audit.json"
    )
    aggregate_audit_path.write_text(
        json.dumps(audit_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote audit: {aggregate_audit_path.relative_to(REPO_ROOT)}")
    render_report(traces, report_path)
    print(f"wrote report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
