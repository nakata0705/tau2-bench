"""Evaluator-private audit of production/joint Concept disagreements.

The calculations in this module use only graph structure and the Node/edge
mapping produced by the diagnostic joint search.  Labels, descriptions,
canonical terms, EvidenceRef text, and observations are copied into the
result only as explicitly marked human-display context; they are never used
for candidate selection, objective evaluation, tie-breaking, or
classification.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from experiments.business_interview.joint_structural_alignment import (
    evaluate_joint_structural_mapping_objective,
)
from tau2.domains.business_interview.graph import (
    ConceptRef,
    business_entry_node_ids,
    business_exit_node_ids,
    node_is_structural,
)

_NODE_RELATIONS = (
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "rationale",
)
_TOLERANCE = 1e-12


def _as_dict(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"expected mapping or Pydantic model, got {type(value)!r}")


def _truth_label(concept: Any) -> str | None:
    """Return display-only Truth text; never call this during matching."""
    terms = getattr(concept, "canonical_terms", None) or []
    if terms:
        return str(terms[0])
    description = getattr(concept, "description", "")
    return str(description) if description else None


def _agent_label(concept: Any) -> str | None:
    """Return display-only Agent text; never call this during matching."""
    label = getattr(concept, "display_label", None)
    return str(label) if label is not None else None


def _node_degrees(graph, node_id: str) -> tuple[int, int]:
    incoming = sum(1 for edge in graph.edges.values() if edge.to_node == node_id)
    outgoing = sum(1 for edge in graph.edges.values() if edge.from_node == node_id)
    return incoming, outgoing


def _node_topology(graph, node_id: str) -> dict[str, Any]:
    incoming, outgoing = _node_degrees(graph, node_id)
    canonical = bool(
        getattr(graph, "source_node_id", None) in graph.nodes
        and getattr(graph, "sink_node_id", None) in graph.nodes
        and any(node_is_structural(node) for node in graph.nodes.values())
    )
    declared_start = (
        graph.source_node_id if canonical else getattr(graph, "start_node_id", None)
    )
    declared_ends = (
        {graph.sink_node_id} if canonical else set(getattr(graph, "end_node_ids", []))
    )
    return {
        "in_degree": incoming,
        "out_degree": outgoing,
        "declared_start": declared_start == node_id,
        "declared_end": node_id in declared_ends,
        "structural": node_is_structural(graph.nodes[node_id]),
    }


def _edge_topology(graph, edge_id: str) -> dict[str, Any]:
    edge = graph.edges[edge_id]
    return {
        "from": _node_topology(graph, edge.from_node),
        "to": _node_topology(graph, edge.to_node),
    }


def _location_records(graph, concept_id: str) -> list[dict[str, Any]]:
    """Collect every asserted graph incidence of one Concept, deterministically."""
    records: list[dict[str, Any]] = []
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        for relation in _NODE_RELATIONS:
            for ref in node.asserted_refs(relation):
                if ref.concept_id != concept_id:
                    continue
                records.append(
                    {
                        "location": f"node:{node_id}:{relation}",
                        "element_type": "node_slot",
                        "relation": relation,
                        "source_id": node_id,
                        "topology": _node_topology(graph, node_id),
                    }
                )
    for edge_id in sorted(graph.edges):
        condition = graph.edges[edge_id].condition
        if (
            isinstance(condition, ConceptRef)
            and condition.asserted
            and condition.concept_id == concept_id
        ):
            records.append(
                {
                    "location": f"edge:{edge_id}:condition",
                    "element_type": "edge_condition",
                    "relation": "condition",
                    "source_id": edge_id,
                    "topology": _edge_topology(graph, edge_id),
                }
            )
    records.sort(key=lambda item: (item["location"], item["element_type"]))
    return records


def _project_locations(
    agent_locations: list[dict[str, Any]],
    node_mapping: Mapping[str, str],
    edge_mapping: Mapping[str, str],
    truth_graph,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for location in agent_locations:
        item = dict(location)
        if location["element_type"] == "node_slot":
            target_id = node_mapping.get(location["source_id"])
            item["mapped_target_id"] = target_id
            item["projected_location"] = (
                f"node:{target_id}:{location['relation']}"
                if target_id is not None
                else None
            )
            item["projected_topology"] = (
                _node_topology(truth_graph, target_id)
                if target_id is not None and target_id in truth_graph.nodes
                else None
            )
        else:
            target_id = edge_mapping.get(location["source_id"])
            item["mapped_target_id"] = target_id
            item["projected_location"] = (
                f"edge:{target_id}:condition" if target_id is not None else None
            )
            item["projected_topology"] = (
                _edge_topology(truth_graph, target_id)
                if target_id is not None and target_id in truth_graph.edges
                else None
            )
        projected.append(item)
    return projected


def _counter_items(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"location": location, "count": counter[location]}
        for location in sorted(counter)
    ]


def _relation_counts(records: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(str(item["relation"]) for item in records)


def _topology_shape(record: Mapping[str, Any], *, projected: bool = False) -> tuple:
    topology = record.get("projected_topology" if projected else "topology")
    if not topology:
        return (record["element_type"], record["relation"], None)
    if record["element_type"] == "node_slot":
        return (
            "node_slot",
            record["relation"],
            topology["in_degree"],
            topology["out_degree"],
            topology["declared_start"],
            topology["declared_end"],
        )
    return (
        "edge_condition",
        "condition",
        tuple(
            (
                topology[side]["in_degree"],
                topology[side]["out_degree"],
                topology[side]["declared_start"],
                topology[side]["declared_end"],
            )
            for side in ("from", "to")
        ),
    )


def _json_shape(shape: tuple) -> list[Any]:
    return [
        _json_shape(value) if isinstance(value, tuple) else value for value in shape
    ]


def _location_details(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "location": item["location"],
            "relation": item["relation"],
            "topology": item["topology"],
            "topology_shape": _json_shape(_topology_shape(item)),
        }
        for item in sorted(records, key=lambda value: value["location"])
    ]


def _observation_evidence(
    agent_graph,
    concept_id: str,
    observations: Iterable[Any],
) -> dict[str, Any]:
    """Collect display-only observation context for the human audit."""
    observation_by_id = {str(item.id): item for item in observations}
    references: list[tuple[str, Any]] = []
    concept = agent_graph.concepts.get(concept_id)
    if concept is not None:
        references.extend(("concept_mention", ref) for ref in concept.mentions)
    for node_id in sorted(agent_graph.nodes):
        node = agent_graph.nodes[node_id]
        for relation in _NODE_RELATIONS:
            for ref in node.refs(relation):
                if ref.concept_id == concept_id and ref.asserted:
                    references.extend(
                        (f"node:{node_id}:{relation}", item) for item in ref.evidence
                    )
    for edge_id in sorted(agent_graph.edges):
        condition = agent_graph.edges[edge_id].condition
        if (
            isinstance(condition, ConceptRef)
            and condition.concept_id == concept_id
            and condition.asserted
        ):
            references.extend(
                (f"edge:{edge_id}:condition", item) for item in condition.evidence
            )

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for source, ref in references:
        observation_id = str(ref.observation_id)
        observation = observation_by_id.get(observation_id)
        key = (
            source,
            observation_id,
            ref.occurrence,
            str(ref.quote or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "source": source,
                "observation_id": observation_id,
                "quote": ref.quote,
                "occurrence": ref.occurrence,
                "turn": getattr(observation, "turn", None),
                "order": getattr(observation, "order", None),
                "text": getattr(observation, "text", None),
            }
        )
    result.sort(
        key=lambda item: (
            item["observation_id"],
            item["source"],
            item["occurrence"],
            item["quote"] or "",
        )
    )
    return {
        "records": result,
        "used_for_structural_calculation": False,
        "display_only": True,
    }


def _candidate_evidence(
    agent_locations: list[dict[str, Any]],
    projected_locations: list[dict[str, Any]],
    truth_locations: list[dict[str, Any]],
    truth_concept_id: str | None,
    truth_concept_kind: str | None,
    agent_kind: str,
) -> dict[str, Any]:
    projected_strings = [
        item["projected_location"]
        for item in projected_locations
        if item.get("projected_location") is not None
    ]
    truth_strings = [item["location"] for item in truth_locations]
    projected_counter = Counter(projected_strings)
    truth_counter = Counter(truth_strings)
    overlap_counter = projected_counter & truth_counter
    agent_only_counter = projected_counter - truth_counter
    truth_only_counter = truth_counter - projected_counter
    overlap_relations = Counter()
    relation_by_location = {
        item["location"]: item["relation"] for item in truth_locations
    }
    for location, count in overlap_counter.items():
        relation = relation_by_location.get(location)
        if relation is not None:
            overlap_relations[relation] += count
    agent_relations = _relation_counts(projected_locations)
    truth_relations = _relation_counts(truth_locations)
    relation_intersection = Counter(
        {
            relation: min(agent_relations[relation], truth_relations[relation])
            for relation in set(agent_relations) | set(truth_relations)
            if min(agent_relations[relation], truth_relations[relation])
        }
    )
    overlap_count = sum(overlap_counter.values())
    has_any_coordinate = bool(projected_counter or truth_counter)
    exact = has_any_coordinate and not (agent_only_counter or truth_only_counter)
    if truth_concept_id is None:
        usage_status = "no_candidate"
    elif exact:
        usage_status = "exact"
    elif overlap_count:
        usage_status = "partial_overlap"
    else:
        usage_status = "no_structural_overlap"

    mapped_node_count = sum(
        item.get("mapped_target_id") is not None and item["element_type"] == "node_slot"
        for item in projected_locations
    )
    mapped_edge_count = sum(
        item.get("mapped_target_id") is not None
        and item["element_type"] == "edge_condition"
        for item in projected_locations
    )
    unmapped_node_count = sum(
        item.get("mapped_target_id") is None and item["element_type"] == "node_slot"
        for item in projected_locations
    )
    unmapped_edge_count = sum(
        item.get("mapped_target_id") is None
        and item["element_type"] == "edge_condition"
        for item in projected_locations
    )
    overlap_edge_count = sum(
        count
        for location, count in overlap_counter.items()
        if location.startswith("edge:")
    )
    projected_shapes = sorted(
        _json_shape(_topology_shape(item, projected=True))
        for item in projected_locations
        if item.get("projected_location") is not None
    )
    truth_shapes = sorted(
        _json_shape(_topology_shape(item)) for item in truth_locations
    )
    structural_fingerprint = {
        "agent_kind": agent_kind,
        "truth_kind": truth_concept_kind,
        "agent_relation_counts": dict(sorted(agent_relations.items())),
        "truth_relation_counts": dict(sorted(truth_relations.items())),
        "relation_intersection_counts": dict(sorted(relation_intersection.items())),
        "overlap_relation_counts": dict(sorted(overlap_relations.items())),
        "projected_location_count": len(projected_strings),
        "truth_location_count": len(truth_strings),
        "overlap_location_count": overlap_count,
        "agent_only_location_count": sum(agent_only_counter.values()),
        "truth_only_location_count": sum(truth_only_counter.values()),
        "mapped_node_count": mapped_node_count,
        "mapped_edge_count": mapped_edge_count,
        "unmapped_node_count": unmapped_node_count,
        "unmapped_edge_count": unmapped_edge_count,
        "overlap_edge_condition_count": overlap_edge_count,
        "projected_topology_shapes": projected_shapes,
        "truth_topology_shapes": truth_shapes,
        "exact_structural_usage_agreement": exact,
        "positive_structural_support": bool(overlap_count),
    }
    return {
        "truth_concept_id": truth_concept_id,
        "truth_concept_kind": truth_concept_kind,
        "kind_consistent": (
            None if truth_concept_id is None else agent_kind == truth_concept_kind
        ),
        "truth_locations": _location_details(truth_locations),
        "projected_agent_locations": sorted(projected_strings),
        "projected_agent_location_details": [
            {
                "source_location": item["location"],
                "projected_location": item.get("projected_location"),
                "relation": item["relation"],
                "mapped_target_id": item.get("mapped_target_id"),
                "projected_topology": item.get("projected_topology"),
            }
            for item in projected_locations
        ],
        "overlapping_usage": _counter_items(overlap_counter),
        "agent_only_locations": _counter_items(agent_only_counter),
        "truth_only_locations": _counter_items(truth_only_counter),
        "exact_structural_usage_agreement": exact,
        "structural_usage_status": usage_status,
        "relation_type_consistency": {
            "agent_relation_counts": dict(sorted(agent_relations.items())),
            "truth_relation_counts": dict(sorted(truth_relations.items())),
            "relation_intersection_counts": dict(sorted(relation_intersection.items())),
            "overlap_relation_counts": dict(sorted(overlap_relations.items())),
            "same_relation_support_count": sum(relation_intersection.values()),
            "has_same_relation_support": bool(relation_intersection),
        },
        "repeated_usage_support": {
            "agent_occurrence_count": len(projected_strings),
            "truth_occurrence_count": len(truth_strings),
            "overlap_occurrence_count": overlap_count,
            "agent_repeated": len(projected_strings) > 1,
            "truth_repeated": len(truth_strings) > 1,
            "repeated_overlap": overlap_count > 1,
        },
        "process_topology_support": {
            "mapped_node_location_count": mapped_node_count,
            "mapped_edge_condition_count": mapped_edge_count,
            "unmapped_node_location_count": unmapped_node_count,
            "unmapped_edge_condition_count": unmapped_edge_count,
            "overlapping_edge_condition_count": overlap_edge_count,
            "projected_topology_shapes": projected_shapes,
            "truth_topology_shapes": truth_shapes,
        },
        "candidate_support": {
            "positive_structural_support": bool(overlap_count),
            "unsupported_candidate": (
                None if truth_concept_id is None else not bool(overlap_count)
            ),
            "no_candidate": truth_concept_id is None,
        },
        "structural_fingerprint": structural_fingerprint,
    }


def _objective_dict(objective: Any) -> dict[str, Any]:
    if isinstance(objective, Mapping):
        return dict(objective)
    return objective.model_dump(mode="json")


def _objective_comparison(
    joint_objective: Mapping[str, Any],
    production_objective: Mapping[str, Any],
) -> dict[str, Any]:
    component_names = sorted(
        set(joint_objective.get("components", {}))
        | set(production_objective.get("components", {}))
    )
    component_deltas: dict[str, Any] = {}
    for name in component_names:
        joint_component = joint_objective.get("components", {}).get(name, {})
        production_component = production_objective.get("components", {}).get(name, {})
        component_deltas[name] = {
            "agreement_delta_joint_minus_production": (
                joint_component.get("agreement", 0.0)
                - production_component.get("agreement", 0.0)
            ),
            "matched_count_delta_joint_minus_production": (
                joint_component.get("matched_count", 0)
                - production_component.get("matched_count", 0)
            ),
            "contribution_delta_joint_minus_production": (
                joint_component.get("contribution", 0.0)
                - production_component.get("contribution", 0.0)
            ),
        }
    return {
        "joint_objective": dict(joint_objective),
        "production_forced_objective": dict(production_objective),
        "objective_delta_joint_minus_production": (
            joint_objective.get("total_score", 0.0)
            - production_objective.get("total_score", 0.0)
        ),
        "component_deltas": component_deltas,
    }


def _force_assignment(
    base_mapping: Mapping[str, str], agent_id: str, truth_id: str | None
) -> dict[str, str]:
    mapping = dict(base_mapping)
    mapping.pop(agent_id, None)
    if truth_id is None:
        return dict(sorted(mapping.items()))
    for other_agent_id, other_truth_id in list(mapping.items()):
        if other_truth_id == truth_id:
            del mapping[other_agent_id]
    mapping[agent_id] = truth_id
    return dict(sorted(mapping.items()))


def _ambiguity_for_agent(joint: Mapping[str, Any], agent_id: str) -> dict[str, Any]:
    search = joint.get("search", {})
    exact = bool(search.get("exact_search"))
    invariant = agent_id in joint.get("invariant_agent_concept_to_truth_concept", {})
    classes = [
        item
        for item in joint.get("concept_ambiguity_classes", [])
        if agent_id in item.get("agent_ids", [])
    ]
    if not exact:
        status = "unknown_under_search_bound"
    elif classes:
        status = "alternative_optimal_mapping_exists"
    elif invariant:
        status = "mapping_is_invariant"
    else:
        status = "unmatched_in_all_exact_optima"
    return {
        "exact_search": exact,
        "joint_mapping_is_invariant": invariant,
        "alternative_optimal_mapping_exists": bool(classes),
        "ambiguity_classes": classes,
        "status": status,
    }


def _classify(
    production: Mapping[str, Any],
    joint: Mapping[str, Any],
    local_counterfactual: Mapping[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    production_positive = bool(
        production["candidate_support"]["positive_structural_support"]
    )
    joint_positive = bool(joint["candidate_support"]["positive_structural_support"])
    delta = local_counterfactual["objective_delta_joint_minus_production"]
    production_unsupported_gain = bool(
        bool(production["candidate_support"]["unsupported_candidate"])
        and delta < -_TOLERANCE
    )
    signals = {
        "production_positive_structural_support": production_positive,
        "joint_positive_structural_support": joint_positive,
        "local_objective_delta_joint_minus_production": delta,
        "unsupported_production_candidate_gains_objective": production_unsupported_gain,
    }
    if joint_positive and not production_positive:
        classification = "joint_strongly_supported"
        rationale = (
            "The joint candidate has positive mapped-coordinate overlap while the "
            "production candidate has no positive structural support."
        )
    elif production_positive and not joint_positive:
        classification = "production_strongly_supported"
        rationale = (
            "The production candidate has positive mapped-coordinate overlap while "
            "the joint candidate has no positive structural support."
        )
    elif production_positive and joint_positive:
        if abs(delta) <= _TOLERANCE:
            classification = "structurally_ambiguous"
            rationale = (
                "Both candidates have positive structural support and forcing the "
                "production assignment is objective-equivalent to the joint choice."
            )
        elif delta > _TOLERANCE:
            classification = "joint_strongly_supported"
            rationale = (
                "Both candidates have support, but the joint candidate produces the "
                "higher fixed-mapping structural objective."
            )
        else:
            classification = "production_strongly_supported"
            rationale = (
                "Both candidates have support, but the forced production candidate "
                "produces the higher fixed-mapping structural objective."
            )
    elif production_unsupported_gain:
        classification = "possible_objective_failure"
        rationale = (
            "Neither candidate has positive structural overlap, yet forcing the "
            "unsupported production assignment raises the raw objective. This is "
            "an objective/admissibility warning, not evidence that production is "
            "structurally correct."
        )
    else:
        classification = "insufficient_structural_evidence"
        rationale = (
            "Neither candidate has positive structural overlap after projecting "
            "Agent locations through the joint Node/edge mapping."
        )
    return classification, rationale, signals


def _boundary_nodes(graph) -> tuple[list[str], list[str]]:
    incoming = {node_id: 0 for node_id in graph.nodes}
    outgoing = {node_id: 0 for node_id in graph.nodes}
    for edge in graph.edges.values():
        if edge.to_node in incoming:
            incoming[edge.to_node] += 1
        if edge.from_node in outgoing:
            outgoing[edge.from_node] += 1
    return (
        sorted(node_id for node_id, count in incoming.items() if count == 0),
        sorted(node_id for node_id, count in outgoing.items() if count == 0),
    )


def build_start_end_investigation(
    agent_graph, truth_graph, joint: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit explicit and topology-derived boundary roles without inferring them."""
    agent_sources, agent_sinks = _boundary_nodes(agent_graph)
    truth_sources, truth_sinks = _boundary_nodes(truth_graph)
    node_mapping = joint.get("representative_agent_node_to_truth_node", {})
    projected_sources = sorted(
        node_mapping[node_id] for node_id in agent_sources if node_id in node_mapping
    )
    projected_sinks = sorted(
        node_mapping[node_id] for node_id in agent_sinks if node_id in node_mapping
    )
    truth_is_canonical = bool(
        getattr(truth_graph, "source_node_id", None) in truth_graph.nodes
        and getattr(truth_graph, "sink_node_id", None) in truth_graph.nodes
        and any(node_is_structural(node) for node in truth_graph.nodes.values())
    )
    truth_declared_sources = (
        [truth_graph.source_node_id]
        if truth_is_canonical
        else (
            [truth_graph.start_node_id] if truth_graph.start_node_id is not None else []
        )
    )
    truth_declared_sinks = (
        [truth_graph.sink_node_id]
        if truth_is_canonical
        else sorted(set(truth_graph.end_node_ids))
    )
    metadata_missing = (
        agent_graph.start_node_id is None
        and not agent_graph.end_node_ids
        and projected_sources == truth_declared_sources
        and projected_sinks == truth_declared_sinks
    )
    if metadata_missing:
        assessment = "agent_graph_boundary_metadata_missing_or_omitted"
    elif (
        projected_sources != truth_declared_sources
        or projected_sinks != truth_declared_sinks
    ):
        assessment = "boundary_roles_not_supported_by_projected_topology"
    else:
        assessment = "boundary_metadata_and_projected_topology_consistent"
    joint_components = joint.get("objective", {}).get("components", {})
    return {
        "assessment": assessment,
        "evidence_basis": "saved_graph_fields_and_directed_edge_topology_only",
        "agent_declared_start_node": agent_graph.start_node_id,
        "agent_declared_end_nodes": sorted(set(agent_graph.end_node_ids)),
        "truth_declared_start_node": truth_declared_sources[0]
        if len(truth_declared_sources) == 1
        else None,
        "truth_declared_end_nodes": truth_declared_sinks,
        "truth_business_entry_nodes": list(business_entry_node_ids(truth_graph))
        if truth_is_canonical
        else [],
        "truth_business_exit_nodes": list(business_exit_node_ids(truth_graph))
        if truth_is_canonical
        else [],
        "truth_graph_is_canonical": truth_is_canonical,
        "topology_derived_end_inference_used": False,
        "agent_topology_sources": agent_sources,
        "agent_topology_sinks": agent_sinks,
        "truth_topology_sources": truth_sources,
        "truth_topology_sinks": truth_sinks,
        "projected_agent_topology_sources": projected_sources,
        "projected_agent_topology_sinks": projected_sinks,
        "projected_sources_equal_truth_declared_start": projected_sources
        == truth_declared_sources,
        "projected_sinks_equal_truth_declared_ends": projected_sinks
        == truth_declared_sinks,
        "joint_start_component": joint_components.get("start_node", {}),
        "joint_end_component": joint_components.get("end_nodes", {}),
        "joint_node_mapping_used": dict(sorted(node_mapping.items())),
        "interpretation": (
            "The saved Agent graph omits explicit start/end metadata even though "
            "its directed topology projects to the Truth boundaries; this is "
            "closer to extraction/graph-recording omission than to a Node mapping "
            "failure. The audit does not infer or rewrite those fields."
            if metadata_missing
            else "The saved fields and topology require further investigation."
        ),
        "used_for_joint_mapping_or_objective": False,
    }


def _record_objectives(
    agent_graph,
    truth_graph,
    node_mapping: Mapping[str, str],
    joint_concept_mapping: Mapping[str, str],
    agent_id: str,
    joint_candidate: str | None,
    production_candidate: str | None,
    full_joint_objective: Mapping[str, Any],
    full_production_objective: Mapping[str, Any],
) -> dict[str, Any]:
    local_joint_mapping = _force_assignment(
        joint_concept_mapping, agent_id, joint_candidate
    )
    local_production_mapping = _force_assignment(
        joint_concept_mapping, agent_id, production_candidate
    )
    local_joint_objective = _objective_dict(
        evaluate_joint_structural_mapping_objective(
            agent_graph,
            truth_graph,
            node_mapping=node_mapping,
            concept_mapping=local_joint_mapping,
        )
    )
    local_production_objective = _objective_dict(
        evaluate_joint_structural_mapping_objective(
            agent_graph,
            truth_graph,
            node_mapping=node_mapping,
            concept_mapping=local_production_mapping,
        )
    )
    local_comparison = _objective_comparison(
        local_joint_objective, local_production_objective
    )
    return {
        "scope": "same_joint_node_mapping_and_all_other_joint_concept_assignments",
        "full_mapping_comparison": _objective_comparison(
            full_joint_objective, full_production_objective
        ),
        "local_candidate_counterfactual": {
            "joint_candidate_concept_mapping": local_joint_mapping,
            "production_candidate_concept_mapping": local_production_mapping,
            **local_comparison,
        },
    }


def build_joint_concept_disagreement_audit(
    agent_graph,
    truth_graph,
    joint_alignment: Any,
    *,
    observations: Iterable[Any] = (),
    seed: int | None = None,
) -> dict[str, Any]:
    """Build a reproducible, human-auditable disagreement trace.

    The returned structure is evaluator-private.  All identity-bearing text is
    marked display-only.  The structural evidence and counterfactual objective
    are computed before those display fields are assembled.
    """
    joint = _as_dict(joint_alignment)
    production_nodes = dict(
        sorted(joint.get("production_agent_node_to_truth_node", {}).items())
    )
    joint_nodes = dict(
        sorted(joint.get("representative_agent_node_to_truth_node", {}).items())
    )
    production_concepts = dict(
        sorted(joint.get("production_agent_concept_to_truth_concept", {}).items())
    )
    joint_concepts = dict(
        sorted(joint.get("representative_agent_concept_to_truth_concept", {}).items())
    )
    joint_edges = dict(
        sorted(joint.get("representative_agent_edge_to_truth_edge", {}).items())
    )
    differences = [
        item
        for item in joint.get("production_vs_joint_concept_differences", [])
        if item.get("left_truth_id") != item.get("right_truth_id")
    ]
    differences.sort(key=lambda item: str(item.get("agent_id", "")))

    full_joint_objective = _objective_dict(
        evaluate_joint_structural_mapping_objective(
            agent_graph,
            truth_graph,
            node_mapping=joint_nodes,
            concept_mapping=joint_concepts,
        )
    )
    full_production_objective = _objective_dict(
        evaluate_joint_structural_mapping_objective(
            agent_graph,
            truth_graph,
            node_mapping=joint_nodes,
            concept_mapping=production_concepts,
        )
    )
    observation_context = list(observations)
    records: list[dict[str, Any]] = []
    for difference in differences:
        agent_id = str(difference["agent_id"])
        agent_concept = agent_graph.concepts.get(agent_id)
        if agent_concept is None:
            continue
        production_candidate = production_concepts.get(agent_id)
        joint_candidate = joint_concepts.get(agent_id)
        agent_kind = str(agent_concept.kind)
        agent_locations = _location_records(agent_graph, agent_id)
        projected_locations = _project_locations(
            agent_locations, joint_nodes, joint_edges, truth_graph
        )

        def candidate(candidate_id: str | None) -> dict[str, Any]:
            truth_concept = (
                truth_graph.concepts.get(candidate_id)
                if candidate_id is not None
                else None
            )
            truth_locations = (
                _location_records(truth_graph, candidate_id)
                if candidate_id is not None and truth_concept is not None
                else []
            )
            evidence = _candidate_evidence(
                agent_locations,
                projected_locations,
                truth_locations,
                candidate_id,
                str(truth_concept.kind) if truth_concept is not None else None,
                agent_kind,
            )
            evidence["truth_label"] = (
                _truth_label(truth_concept) if truth_concept is not None else None
            )
            evidence["truth_label_display_only"] = True
            return evidence

        production_evidence = candidate(production_candidate)
        joint_evidence = candidate(joint_candidate)
        counterfactual = _record_objectives(
            agent_graph,
            truth_graph,
            joint_nodes,
            joint_concepts,
            agent_id,
            joint_candidate,
            production_candidate,
            full_joint_objective,
            full_production_objective,
        )
        local_comparison = counterfactual["local_candidate_counterfactual"]
        alternative_optimal_mapping = _ambiguity_for_agent(joint, agent_id)
        full_comparison = counterfactual["full_mapping_comparison"]
        if (
            production_evidence["candidate_support"]["positive_structural_support"]
            and joint_evidence["candidate_support"]["positive_structural_support"]
            and alternative_optimal_mapping["alternative_optimal_mapping_exists"]
            and abs(full_comparison["objective_delta_joint_minus_production"])
            <= _TOLERANCE
        ):
            classification = "structurally_ambiguous"
            rationale = (
                "Both candidates have positive structural support, and the forced "
                "production mapping has the same full objective as the joint mapping "
                "within an exact alternative-optimum class."
            )
            signals = {
                "production_positive_structural_support": True,
                "joint_positive_structural_support": True,
                "local_objective_delta_joint_minus_production": local_comparison[
                    "objective_delta_joint_minus_production"
                ],
                "full_objective_delta_joint_minus_production": full_comparison[
                    "objective_delta_joint_minus_production"
                ],
                "unsupported_production_candidate_gains_objective": False,
            }
        else:
            classification, rationale, signals = _classify(
                production_evidence, joint_evidence, local_comparison
            )
        signals["full_objective_delta_joint_minus_production"] = full_comparison[
            "objective_delta_joint_minus_production"
        ]
        records.append(
            {
                "seed": seed,
                "concept_kind": agent_kind,
                "agent_concept_id": agent_id,
                "agent_label": _agent_label(agent_concept),
                "agent_label_display_only": True,
                "production_truth_concept_id": production_candidate,
                "production_truth_label": production_evidence["truth_label"],
                "production_truth_label_display_only": True,
                "joint_truth_concept_id": joint_candidate,
                "joint_truth_label": joint_evidence["truth_label"],
                "joint_truth_label_display_only": True,
                "agent_structural_usage": {
                    "locations": [item["location"] for item in agent_locations],
                    "location_details": _location_details(agent_locations),
                    "projected_locations": projected_locations,
                    "mapped_location_count": sum(
                        item.get("projected_location") is not None
                        for item in projected_locations
                    ),
                    "unmapped_location_count": sum(
                        item.get("projected_location") is None
                        for item in projected_locations
                    ),
                },
                "production_candidate_evidence": production_evidence,
                "joint_candidate_evidence": joint_evidence,
                "counterfactual": counterfactual,
                "alternative_optimal_mapping": alternative_optimal_mapping,
                "observation_context": _observation_evidence(
                    agent_graph, agent_id, observation_context
                ),
                "classification": classification,
                "classification_rationale": rationale,
                "classification_signals": signals,
            }
        )

    records.sort(key=lambda item: item["agent_concept_id"])
    counts = Counter(item["classification"] for item in records)
    return {
        "schema_version": "business_interview.joint_concept_disagreement_audit.v1",
        "status": "ok",
        "seed": seed,
        "disagreement_count": len(records),
        "label_independent_structural_calculation": True,
        "display_text_is_not_a_matching_signal": True,
        "joint_mapping_basis": {
            "node_mapping": joint_nodes,
            "edge_mapping": joint_edges,
            "production_node_mapping": production_nodes,
            "node_mapping_differences": joint.get(
                "production_vs_joint_node_differences", []
            ),
        },
        "objective_basis": {
            "joint_objective": full_joint_objective,
            "production_mapping_forced_objective": full_production_objective,
            "full_mapping_comparison": _objective_comparison(
                full_joint_objective, full_production_objective
            ),
            "objective_implementation": "existing_joint_structural_objective",
            "objective_or_search_was_not_modified": True,
        },
        "classification_counts": dict(sorted(counts.items())),
        "records": records,
        "seed_9003_start_end_investigation": (
            build_start_end_investigation(agent_graph, truth_graph, joint)
            if seed == 9003
            else None
        ),
        "limitations": [
            "Structural coordinates are projected through the representative joint Node/edge mapping; unmapped locations remain explicit.",
            "A representative mapping is not identity truth; exact optimal alternatives are reported separately.",
            "Observation text, labels, descriptions, canonical terms, and evidence quotes are display-only and never enter calculation.",
            "A forced unsupported production assignment can expose an objective/admissibility mismatch; that does not make the assignment structurally supported.",
        ],
    }


__all__ = [
    "build_joint_concept_disagreement_audit",
    "build_start_end_investigation",
]
