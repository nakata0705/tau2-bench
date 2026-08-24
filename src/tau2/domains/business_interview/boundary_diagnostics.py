"""Diagnostic-only structural START/END boundary experiments.

This module deliberately sits outside the production evaluator contract.  It
adds a virtual structural boundary to an immutable copy of a business graph,
then projects that copy back to business nodes/edges for the existing
label-independent joint alignment.  Virtual nodes and synthetic edges are
therefore visible in the diagnostic representation, but they cannot become
Concepts or inflate business process-edge metrics.

The only identity-bearing values used by the alignment experiment are typed
incidence, directed topology, and the reserved structural marker ids defined
below.  Labels, descriptions, canonical terms, embeddings, sidecar ids, and
LLM output are never read by this module.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentConcept,
    AgentGraph,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    Node,
    TruthConcept,
    TruthEdge,
    TruthNode,
)
from tau2.domains.business_interview.joint_structural_alignment import (
    JointStructuralAlignmentDiagnostics,
    build_joint_structural_alignment_diagnostics,
    evaluate_joint_structural_mapping_objective,
)

SCHEMA_VERSION = "business_interview.boundary_diagnostics.v1"
STRUCTURAL_START_NODE_ID = "__tau2_structural_start__"
STRUCTURAL_END_NODE_ID = "__tau2_structural_end__"
STRUCTURAL_EDGE_PREFIX = "__tau2_structural_boundary__"
STRUCTURAL_EDGE_MARKER = "structural_boundary_only"


@dataclass(frozen=True)
class _Topology:
    node_ids: tuple[str, ...]
    incoming: dict[str, int]
    outgoing: dict[str, int]
    adjacency: dict[str, tuple[str, ...]]
    reverse_adjacency: dict[str, tuple[str, ...]]
    sources: tuple[str, ...]
    sinks: tuple[str, ...]
    dangling_edge_ids: tuple[str, ...]
    weak_components: tuple[tuple[str, ...], ...]
    cycle_components: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DiagnosticCanonicalGraph:
    """A canonicalized copy plus the business projection contract.

    ``graph`` contains virtual nodes and synthetic edges.  The original graph
    is never mutated.  ``business_node_ids`` and ``business_edge_ids`` are the
    only ids permitted in business scoring; structural ids are explicitly
    tracked so a renderer or a metric implementation cannot confuse them with
    business relations.
    """

    graph: Any
    business_node_ids: tuple[str, ...]
    business_edge_ids: tuple[str, ...]
    structural_node_ids: frozenset[str]
    synthetic_edge_ids: frozenset[str]
    source_nodes: tuple[str, ...]
    sink_nodes: tuple[str, ...]
    original_start_node_id: str | None
    original_end_node_ids: tuple[str, ...]
    synthetic_edges: tuple[dict[str, Any], ...]
    invariant: dict[str, Any]

    def business_projection(self) -> Any:
        """Return a copy with virtual boundary material removed."""
        projected = self.graph.model_copy(deep=True)
        for node_id in self.structural_node_ids:
            projected.nodes.pop(node_id, None)
        for edge_id in self.synthetic_edge_ids:
            projected.edges.pop(edge_id, None)
        projected.start_node_id = self.original_start_node_id
        projected.end_node_ids = list(self.original_end_node_ids)
        return projected

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, text-free representation for JSON diagnostics."""
        return {
            "business_node_ids": list(self.business_node_ids),
            "business_edge_ids": list(self.business_edge_ids),
            "structural_nodes": [
                {
                    "id": STRUCTURAL_START_NODE_ID,
                    "kind": "structural_start",
                }
                for _ in [0]
                if STRUCTURAL_START_NODE_ID in self.structural_node_ids
            ]
            + [
                {
                    "id": STRUCTURAL_END_NODE_ID,
                    "kind": "structural_end",
                }
                for _ in [0]
                if STRUCTURAL_END_NODE_ID in self.structural_node_ids
            ],
            "synthetic_edges": [dict(edge) for edge in self.synthetic_edges],
            "source_nodes": list(self.source_nodes),
            "sink_nodes": list(self.sink_nodes),
            "original_declared_start_node": self.original_start_node_id,
            "original_declared_end_nodes": list(self.original_end_node_ids),
            "invariant": self.invariant,
        }


@dataclass(frozen=True)
class NormalizationResult:
    """Outcome of a diagnostic normalization attempt."""

    possible: bool
    classification: str
    reason: str
    audit: dict[str, Any]
    canonical: DiagnosticCanonicalGraph | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "possible": self.possible,
            "classification": self.classification,
            "reason": self.reason,
            "audit": self.audit,
        }
        if self.canonical is not None:
            result["canonical_copy"] = self.canonical.to_dict()
        return result


def _topology(graph: Any) -> _Topology:
    node_ids = tuple(sorted(graph.nodes))
    node_set = set(node_ids)
    incoming = {node_id: 0 for node_id in node_ids}
    outgoing = {node_id: 0 for node_id in node_ids}
    adjacency_sets = {node_id: set() for node_id in node_ids}
    reverse_sets = {node_id: set() for node_id in node_ids}
    dangling: list[str] = []
    undirected_sets = {node_id: set() for node_id in node_ids}

    for edge_id in sorted(graph.edges):
        edge = graph.edges[edge_id]
        valid_from = edge.from_node in node_set
        valid_to = edge.to_node in node_set
        if not valid_from or not valid_to:
            dangling.append(edge_id)
            continue
        outgoing[edge.from_node] += 1
        incoming[edge.to_node] += 1
        adjacency_sets[edge.from_node].add(edge.to_node)
        reverse_sets[edge.to_node].add(edge.from_node)
        undirected_sets[edge.from_node].add(edge.to_node)
        undirected_sets[edge.to_node].add(edge.from_node)

    adjacency = {
        node_id: tuple(sorted(targets)) for node_id, targets in adjacency_sets.items()
    }
    reverse_adjacency = {
        node_id: tuple(sorted(sources)) for node_id, sources in reverse_sets.items()
    }
    sources = tuple(node_id for node_id in node_ids if incoming[node_id] == 0)
    sinks = tuple(node_id for node_id in node_ids if outgoing[node_id] == 0)
    weak_components = _components(node_ids, undirected_sets)
    cycle_components = _cycle_components(node_ids, adjacency)
    return _Topology(
        node_ids=node_ids,
        incoming=incoming,
        outgoing=outgoing,
        adjacency=adjacency,
        reverse_adjacency=reverse_adjacency,
        sources=sources,
        sinks=sinks,
        dangling_edge_ids=tuple(sorted(dangling)),
        weak_components=weak_components,
        cycle_components=cycle_components,
    )


def _components(
    node_ids: Sequence[str], adjacency: Mapping[str, Iterable[str]]
) -> tuple[tuple[str, ...], ...]:
    remaining = set(node_ids)
    components: list[tuple[str, ...]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        queue = [root]
        component = [root]
        while queue:
            node_id = queue.pop()
            for neighbor in sorted(adjacency.get(node_id, ())):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _cycle_components(
    node_ids: Sequence[str], adjacency: Mapping[str, Iterable[str]]
) -> tuple[tuple[str, ...], ...]:
    """Return strongly connected components that contain a directed cycle."""
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for neighbor in adjacency.get(node_id, ()):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[neighbor])
        if lowlinks[node_id] != indices[node_id]:
            return
        component: list[str] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node_id:
                break
        component_set = set(component)
        self_loop = any(
            member in set(adjacency.get(member, ())) for member in component
        )
        if len(component) > 1 or self_loop:
            components.append(tuple(sorted(component_set)))

    for node_id in sorted(node_ids):
        if node_id not in indices:
            visit(node_id)
    return tuple(sorted(components))


def _reachable(
    starts: Iterable[str], adjacency: Mapping[str, Iterable[str]]
) -> set[str]:
    reached = set(starts)
    queue = deque(sorted(reached))
    while queue:
        node_id = queue.popleft()
        for neighbor in adjacency.get(node_id, ()):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def _condition_shape(edge: Any) -> dict[str, Any]:
    condition = edge.condition
    return {
        "present": isinstance(condition, ConceptRef),
        "asserted": bool(isinstance(condition, ConceptRef) and condition.asserted),
        "concept_id": (
            condition.concept_id if isinstance(condition, ConceptRef) else None
        ),
    }


def _reserved_collisions(graph: Any) -> dict[str, list[str]]:
    node_collisions = sorted(
        node_id
        for node_id in graph.nodes
        if node_id in {STRUCTURAL_START_NODE_ID, STRUCTURAL_END_NODE_ID}
    )
    edge_collisions = sorted(
        edge_id for edge_id in graph.edges if edge_id.startswith(STRUCTURAL_EDGE_PREFIX)
    )
    return {"node_ids": node_collisions, "edge_ids": edge_collisions}


def audit_boundary(graph: Any, *, graph_name: str | None = None) -> dict[str, Any]:
    """Produce a machine-readable boundary/topology audit without inference."""
    topology = _topology(graph)
    node_set = set(topology.node_ids)
    reachable = _reachable(topology.sources, topology.adjacency)
    can_reach_sink = _reachable(topology.sinks, topology.reverse_adjacency)
    declared_start = graph.start_node_id
    declared_ends = tuple(sorted(set(graph.end_node_ids)))
    declared_start_exists = declared_start is not None and declared_start in node_set
    declared_ends_exist = all(node_id in node_set for node_id in declared_ends)
    declared_start_is_source = (
        declared_start_exists and declared_start in topology.sources
    )
    declared_ends_are_sinks = declared_ends_exist and all(
        node_id in topology.sinks for node_id in declared_ends
    )
    metadata_complete = declared_start is not None and bool(declared_ends)
    metadata_consistent = (
        metadata_complete
        and declared_start_is_source
        and declared_ends_are_sinks
        and set(declared_ends) == set(topology.sinks)
        and len(topology.sources) == 1
        and declared_start == topology.sources[0]
    )
    source_edges = [
        {
            "edge_id": edge_id,
            "from_node": graph.edges[edge_id].from_node,
            "to_node": graph.edges[edge_id].to_node,
            "condition": _condition_shape(graph.edges[edge_id]),
        }
        for edge_id in sorted(graph.edges)
        if graph.edges[edge_id].from_node in topology.sources
    ]
    sink_edges = [
        {
            "edge_id": edge_id,
            "from_node": graph.edges[edge_id].from_node,
            "to_node": graph.edges[edge_id].to_node,
            "condition": _condition_shape(graph.edges[edge_id]),
        }
        for edge_id in sorted(graph.edges)
        if graph.edges[edge_id].to_node in topology.sinks
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_name": graph_name,
        "node_count": len(topology.node_ids),
        "edge_count": len(graph.edges),
        "node_ids": list(topology.node_ids),
        "declared_start_node": declared_start,
        "declared_end_nodes": list(declared_ends),
        "topology_sources": list(topology.sources),
        "topology_sinks": list(topology.sinks),
        "source_count": len(topology.sources),
        "sink_count": len(topology.sinks),
        "unreachable_from_source": sorted(node_set - reachable),
        "cannot_reach_sink": sorted(node_set - can_reach_sink),
        "dangling_edge_ids": list(topology.dangling_edge_ids),
        "weak_components": [list(component) for component in topology.weak_components],
        "weak_component_count": len(topology.weak_components),
        "cycles": {
            "has_cycles": bool(topology.cycle_components),
            "components": [list(component) for component in topology.cycle_components],
            "self_loop_nodes": sorted(
                node_id
                for component in topology.cycle_components
                for node_id in component
                if node_id in topology.adjacency.get(node_id, ())
            ),
        },
        "boundary_conditions": {
            "source_outgoing_edges": source_edges,
            "sink_incoming_edges": sink_edges,
            "source_condition_edge_count": sum(
                item["condition"]["present"] for item in source_edges
            ),
            "sink_condition_edge_count": sum(
                item["condition"]["present"] for item in sink_edges
            ),
        },
        "reserved_boundary_id_collisions": _reserved_collisions(graph),
        "metadata_consistency": {
            "metadata_complete": metadata_complete,
            "declared_start_exists": declared_start_exists,
            "declared_start_is_topology_source": declared_start_is_source,
            "declared_end_nodes_exist": declared_ends_exist,
            "declared_end_nodes_are_topology_sinks": declared_ends_are_sinks,
            "declared_end_nodes_equal_topology_sinks": set(declared_ends)
            == set(topology.sinks),
            "declared_start_covers_all_topology_sources": (
                len(topology.sources) == 1
                and declared_start
                == (topology.sources[0] if topology.sources else None)
            ),
            "consistent": metadata_consistent,
            "assessment": (
                "consistent"
                if metadata_consistent
                else "missing_or_inconsistent_boundary_metadata"
            ),
        },
    }


def _normalization_classification(audit: Mapping[str, Any]) -> tuple[bool, str, str]:
    nodes = list(audit["node_ids"])
    sources = list(audit["topology_sources"])
    sinks = list(audit["topology_sinks"])
    if not nodes:
        return False, "invalid_empty_graph", "a graph needs at least one business node"
    collisions = audit["reserved_boundary_id_collisions"]
    if collisions["node_ids"] or collisions["edge_ids"]:
        return (
            False,
            "invalid_reserved_boundary_id_collision",
            "reserved structural ids already exist in the business graph",
        )
    if audit["dangling_edge_ids"]:
        return (
            False,
            "invalid_dangling_edge",
            "an edge endpoint is not a business node",
        )
    if not sources:
        return (
            False,
            "invalid_source_missing",
            "a finite graph without a topology source has no canonical entry",
        )
    if not sinks:
        return (
            False,
            "invalid_sink_missing",
            "a finite graph without a topology sink has no canonical exit",
        )
    components = audit["weak_components"]
    if len(components) > 1:
        if any(len(component) == 1 for component in components):
            return (
                False,
                "invalid_isolated_node",
                "an isolated business node cannot be part of this single process",
            )
        complete_components = 0
        for component in components:
            component_set = set(component)
            component_sources = set(sources) & component_set
            component_sinks = set(sinks) & component_set
            if component_sources and component_sinks:
                complete_components += 1
        if complete_components == len(components):
            return (
                False,
                "multiple_independent_processes",
                "independent weak components require separate process graphs",
            )
        return (
            False,
            "invalid_disconnected_component",
            "not every business node belongs to the one source-to-sink process",
        )
    if audit["unreachable_from_source"] or audit["cannot_reach_sink"]:
        return (
            False,
            "invalid_unreachable_or_dead_end",
            "business nodes are not both source-reachable and sink-reachable",
        )
    if audit["cycles"]["has_cycles"]:
        if audit["cycles"]["self_loop_nodes"]:
            return (
                True,
                "normalizable_with_self_loop",
                "the self-loop is preserved; all nodes still reach the canonical END",
            )
        return (
            True,
            "normalizable_with_cycle",
            "the cycle is preserved; all nodes still reach the canonical END",
        )
    if len(sources) > 1 and len(sinks) > 1:
        return (
            True,
            "normalizable_multiple_entries_and_exits",
            "virtual boundaries can fan out and merge without changing business edges",
        )
    if len(sources) > 1:
        return (
            True,
            "normalizable_multiple_entries",
            "virtual START can fan out to each topology source",
        )
    if len(sinks) > 1:
        return (
            True,
            "normalizable_multiple_exits",
            "each topology sink can merge into virtual END",
        )
    return True, "normalizable", "one source and one sink already exist topologically"


def _virtual_node(graph: Any, node_id: str) -> Any:
    if isinstance(graph, AgentGraph):
        # Every slot is explicitly absent so the diagnostic copy remains
        # structurally well-formed without introducing a ConceptRef.
        return Node(
            id=node_id,
            activity=AbsentType(),
            actor=AbsentType(),
            system=AbsentType(),
            reads=AbsentType(),
            writes=AbsentType(),
            necessity_rationale=AbsentType(),
        )
    return TruthNode(id=node_id)


def _virtual_edge(graph: Any, edge_id: str, from_node: str, to_node: str) -> Any:
    if isinstance(graph, AgentGraph):
        return Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            condition=AbsentType(),
        )
    return TruthEdge(id=edge_id, from_node=from_node, to_node=to_node)


def _boundary_edge_id(side: str, node_id: str) -> str:
    return f"{STRUCTURAL_EDGE_PREFIX}{side}__{node_id}"


def _canonical_invariant(
    graph: Any,
    business_node_ids: Sequence[str],
    structural_node_ids: set[str],
    add_start: bool,
    add_end: bool,
) -> dict[str, Any]:
    audit = audit_boundary(graph)
    business_nodes = set(business_node_ids)
    reachable = set(graph.nodes)
    if add_start:
        reachable = _reachable(
            [STRUCTURAL_START_NODE_ID],
            _topology(graph).adjacency,
        )
    reverse_reachable = set(graph.nodes)
    if add_end:
        reverse_reachable = _reachable(
            [STRUCTURAL_END_NODE_ID],
            _topology(graph).reverse_adjacency,
        )
    return {
        "exactly_one_structural_start": (
            not add_start or audit["topology_sources"] == [STRUCTURAL_START_NODE_ID]
        ),
        "exactly_one_structural_end": (
            not add_end or audit["topology_sinks"] == [STRUCTURAL_END_NODE_ID]
        ),
        "structural_start_has_no_incoming": (
            not add_start
            or audit["declared_start_node"] == STRUCTURAL_START_NODE_ID
            and _topology(graph).incoming[STRUCTURAL_START_NODE_ID] == 0
        ),
        "structural_end_has_no_outgoing": (
            not add_end
            or STRUCTURAL_END_NODE_ID in audit["topology_sinks"]
            and _topology(graph).outgoing[STRUCTURAL_END_NODE_ID] == 0
        ),
        "all_business_nodes_start_reachable": business_nodes.issubset(reachable),
        "all_business_nodes_end_reachable": business_nodes.issubset(reverse_reachable),
        "structural_nodes_are_not_concepts": all(
            node_id not in graph.concepts for node_id in structural_node_ids
        ),
        "synthetic_edges_are_structural_only": all(
            edge_id.startswith(STRUCTURAL_EDGE_PREFIX)
            for edge_id in graph.edges
            if edge_id.startswith(STRUCTURAL_EDGE_PREFIX)
        ),
    }


def normalize_graph(
    graph: Any,
    *,
    add_start: bool = True,
    add_end: bool = True,
    graph_name: str | None = None,
) -> NormalizationResult:
    """Create a diagnostic canonical copy, or classify why it is unsafe.

    The input graph is never changed.  The canonical copy keeps every original
    business node/edge and only adds reserved structural markers plus
    structural-only synthetic edges.
    """
    audit = audit_boundary(graph, graph_name=graph_name)
    possible, classification, reason = _normalization_classification(audit)
    if not possible:
        return NormalizationResult(False, classification, reason, audit)

    copy = graph.model_copy(deep=True)
    business_node_ids = tuple(sorted(graph.nodes))
    business_edge_ids = tuple(sorted(graph.edges))
    structural_nodes: set[str] = set()
    synthetic_edge_ids: set[str] = set()
    synthetic_edges: list[dict[str, Any]] = []
    original_start = graph.start_node_id
    original_ends = tuple(sorted(set(graph.end_node_ids)))

    if add_start:
        copy.nodes[STRUCTURAL_START_NODE_ID] = _virtual_node(
            graph, STRUCTURAL_START_NODE_ID
        )
        structural_nodes.add(STRUCTURAL_START_NODE_ID)
        for source in audit["topology_sources"]:
            edge_id = _boundary_edge_id("start", source)
            copy.edges[edge_id] = _virtual_edge(
                graph, edge_id, STRUCTURAL_START_NODE_ID, source
            )
            synthetic_edge_ids.add(edge_id)
            synthetic_edges.append(
                {
                    "id": edge_id,
                    "from_node": STRUCTURAL_START_NODE_ID,
                    "to_node": source,
                    "marker": STRUCTURAL_EDGE_MARKER,
                    "structural_only": True,
                    "business_relation": False,
                }
            )
        copy.start_node_id = STRUCTURAL_START_NODE_ID

    if add_end:
        copy.nodes[STRUCTURAL_END_NODE_ID] = _virtual_node(
            graph, STRUCTURAL_END_NODE_ID
        )
        structural_nodes.add(STRUCTURAL_END_NODE_ID)
        for sink in audit["topology_sinks"]:
            edge_id = _boundary_edge_id("end", sink)
            copy.edges[edge_id] = _virtual_edge(
                graph, edge_id, sink, STRUCTURAL_END_NODE_ID
            )
            synthetic_edge_ids.add(edge_id)
            synthetic_edges.append(
                {
                    "id": edge_id,
                    "from_node": sink,
                    "to_node": STRUCTURAL_END_NODE_ID,
                    "marker": STRUCTURAL_EDGE_MARKER,
                    "structural_only": True,
                    "business_relation": False,
                }
            )
        copy.end_node_ids = [STRUCTURAL_END_NODE_ID]

    synthetic_edges.sort(key=lambda item: item["id"])
    invariant = _canonical_invariant(
        copy,
        business_node_ids,
        structural_nodes,
        add_start,
        add_end,
    )
    canonical = DiagnosticCanonicalGraph(
        graph=copy,
        business_node_ids=business_node_ids,
        business_edge_ids=business_edge_ids,
        structural_node_ids=frozenset(structural_nodes),
        synthetic_edge_ids=frozenset(synthetic_edge_ids),
        source_nodes=tuple(audit["topology_sources"]),
        sink_nodes=tuple(audit["topology_sinks"]),
        original_start_node_id=original_start,
        original_end_node_ids=original_ends,
        synthetic_edges=tuple(synthetic_edges),
        invariant=invariant,
    )
    return NormalizationResult(True, classification, reason, audit, canonical)


def _boundary_allowlist(
    agent_graph: Any,
    truth_graph: Any,
    *,
    use_start: bool,
    use_end: bool,
) -> dict[str, set[str]] | None:
    """Project fixed virtual anchors to source/sink role constraints.

    The virtual nodes themselves are fixed START->START and END->END in the
    canonical representation.  They are not sent to the production matcher;
    their synthetic incidence is projected to this label-free candidate
    restriction over business nodes.
    """
    if not use_start and not use_end:
        return None
    agent_audit = audit_boundary(agent_graph)
    truth_audit = audit_boundary(truth_graph)
    agent_sources = set(agent_audit["topology_sources"])
    truth_sources = set(truth_audit["topology_sources"])
    agent_sinks = set(agent_audit["topology_sinks"])
    truth_sinks = set(truth_audit["topology_sinks"])
    allowlist: dict[str, set[str]] = {}
    for agent_node_id in sorted(agent_graph.nodes):
        allowed: set[str] = set()
        for truth_node_id in sorted(truth_graph.nodes):
            if use_start and (agent_node_id in agent_sources) != (
                truth_node_id in truth_sources
            ):
                continue
            if use_end and (agent_node_id in agent_sinks) != (
                truth_node_id in truth_sinks
            ):
                continue
            allowed.add(truth_node_id)
        allowlist[agent_node_id] = allowed
    return allowlist


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def _mapping_differences(
    left: Mapping[str, str], right: Mapping[str, str]
) -> list[dict[str, str | None]]:
    return [
        {
            "agent_id": agent_id,
            "left_truth_id": left.get(agent_id),
            "right_truth_id": right.get(agent_id),
        }
        for agent_id in sorted(set(left) | set(right))
        if left.get(agent_id) != right.get(agent_id)
    ]


def _search_snapshot(alignment: Mapping[str, Any]) -> dict[str, Any]:
    search = alignment["search"]
    keys = (
        "node_candidate_pair_count",
        "node_search_states",
        "node_leaves_evaluated",
        "node_states_pruned",
        "exact_search",
        "bound_hit",
        "bound_reason",
        "optimum_is_unique",
        "optimal_solution_count",
        "optimal_solution_count_is_exact",
        "concept_search_states",
        "edge_assignment_states",
    )
    return {key: search.get(key) for key in keys}


def _canonical_component(matched: int = 1) -> dict[str, Any]:
    agreement = 1.0 if matched else 0.0
    return {
        "matched_count": matched,
        "agent_count": 1,
        "truth_count": 1,
        "agreement": agreement,
        "weight": 1.0,
        "contribution": agreement,
        "boundary_source": "fixed_virtual_structural_anchor",
    }


def _canonical_objective(
    raw_business_objective: Mapping[str, Any],
    *,
    use_start: bool,
    use_end: bool,
) -> dict[str, Any]:
    """Replace only boundary components; business metrics remain original-only."""
    objective = {
        key: value
        for key, value in raw_business_objective.items()
        if key != "components"
    }
    components = {
        name: dict(component)
        for name, component in raw_business_objective.get("components", {}).items()
    }
    if use_start:
        components["start_node"] = _canonical_component()
    if use_end:
        components["end_nodes"] = _canonical_component()
    total = sum(
        _numeric(component.get("contribution"), 0.0)
        for component in components.values()
    )
    max_score = sum(
        _numeric(component.get("weight"), 1.0) for component in components.values()
    )
    objective["components"] = components
    objective["total_score"] = total
    objective["max_score"] = max_score
    objective["normalized_score"] = total / max_score if max_score else 1.0
    objective["boundary_semantics"] = (
        "topology_derived_virtual_anchors" if use_start or use_end else "saved_metadata"
    )
    objective["synthetic_boundary_edges_included"] = False
    return objective


def _classification_summary(
    alignment: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_records = {
        item["agent_concept_id"]: item["classification"]
        for item in baseline.get("records", [])
    }
    current_records = {
        item["agent_concept_id"]: item["classification"]
        for item in alignment.get("records", [])
    }
    changes = [
        {
            "agent_concept_id": concept_id,
            "current": baseline_records.get(concept_id),
            "variant": current_records.get(concept_id),
        }
        for concept_id in sorted(set(baseline_records) | set(current_records))
        if baseline_records.get(concept_id) != current_records.get(concept_id)
    ]
    return {
        "baseline_counts": dict(sorted(_counts(baseline_records.values()).items())),
        "variant_counts": dict(sorted(_counts(current_records.values()).items())),
        "changed": bool(changes),
        "changed_records": changes,
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _alignment_variant(
    name: str,
    agent_graph: Any,
    truth_graph: Any,
    *,
    use_start: bool,
    use_end: bool,
    baseline_alignment: Mapping[str, Any] | None,
    production_node_mapping: Mapping[str, str],
    production_concept_mapping: Mapping[str, str],
    usage_concept_mapping: Mapping[str, str],
    baseline_audit: Mapping[str, Any] | None,
    observations: Iterable[Any] = (),
    seed: int | None = None,
) -> dict[str, Any]:
    allowlist = _boundary_allowlist(
        agent_graph,
        truth_graph,
        use_start=use_start,
        use_end=use_end,
    )
    alignment_model: JointStructuralAlignmentDiagnostics = (
        build_joint_structural_alignment_diagnostics(
            agent_graph,
            truth_graph,
            production_node_to_truth=production_node_mapping,
            production_concept_to_truth=production_concept_mapping,
            usage_concept_to_truth=usage_concept_mapping,
            node_candidate_allowlist=allowlist,
        )
    )
    alignment = _as_dict(alignment_model)
    mapping = alignment["representative_agent_node_to_truth_node"]
    concept_mapping = alignment["representative_agent_concept_to_truth_concept"]
    raw_business_objective = _as_dict(
        evaluate_joint_structural_mapping_objective(
            agent_graph,
            truth_graph,
            node_mapping=mapping,
            concept_mapping=concept_mapping,
        )
    )
    canonical_objective = _canonical_objective(
        raw_business_objective,
        use_start=use_start,
        use_end=use_end,
    )
    variant_audit = None
    if baseline_audit is not None:
        from scripts.business_interview_joint_alignment_audit import (
            build_joint_concept_disagreement_audit,
        )

        audit = build_joint_concept_disagreement_audit(
            agent_graph,
            truth_graph,
            alignment_model,
            observations=observations,
            seed=seed,
        )
        variant_audit = _classification_summary(audit, baseline_audit)
    baseline_node_mapping = (
        baseline_alignment.get("representative_agent_node_to_truth_node", {})
        if baseline_alignment is not None
        else {}
    )
    baseline_concept_mapping = (
        baseline_alignment.get("representative_agent_concept_to_truth_concept", {})
        if baseline_alignment is not None
        else {}
    )
    search = _search_snapshot(alignment)
    base_search = (
        _search_snapshot(baseline_alignment) if baseline_alignment is not None else {}
    )
    return {
        "name": name,
        "virtual_start_enabled": use_start,
        "virtual_end_enabled": use_end,
        "fixed_structural_anchors": {
            "start": (
                {
                    "agent": STRUCTURAL_START_NODE_ID,
                    "truth": STRUCTURAL_START_NODE_ID,
                    "identity": "structural_marker_exact",
                }
                if use_start
                else None
            ),
            "end": (
                {
                    "agent": STRUCTURAL_END_NODE_ID,
                    "truth": STRUCTURAL_END_NODE_ID,
                    "identity": "structural_marker_exact",
                }
                if use_end
                else None
            ),
        },
        "candidate_policy": (
            "topology source/sink role equality projected from fixed virtual anchors"
            if use_start or use_end
            else "current joint diagnostic candidate policy"
        ),
        "alignment_input": {
            "kind": (
                "business_projection_of_canonical_copy"
                if use_start or use_end
                else "saved_business_graph"
            ),
            "structural_nodes_visible_to_alignment": False,
            "synthetic_edges_visible_to_alignment": False,
        },
        "search": search,
        "search_space_delta_vs_current": {
            "candidate_pair_count_delta": (
                search["node_candidate_pair_count"]
                - base_search.get(
                    "node_candidate_pair_count", search["node_candidate_pair_count"]
                )
                if base_search
                else 0
            ),
            "candidate_pair_count_reduction": (
                base_search.get(
                    "node_candidate_pair_count", search["node_candidate_pair_count"]
                )
                - search["node_candidate_pair_count"]
                if base_search
                else 0
            ),
            "candidate_pair_count_reduction_fraction": _reduction_fraction(
                base_search.get("node_candidate_pair_count"),
                search["node_candidate_pair_count"],
            ),
            "node_search_states_reduction": (
                base_search.get("node_search_states", search["node_search_states"])
                - search["node_search_states"]
                if base_search
                else 0
            ),
            "node_search_states_reduction_fraction": _reduction_fraction(
                base_search.get("node_search_states"),
                search["node_search_states"],
            ),
        },
        "representative_node_mapping": dict(sorted(mapping.items())),
        "representative_concept_mapping": dict(sorted(concept_mapping.items())),
        "representative_edge_mapping": dict(
            sorted(alignment["representative_agent_edge_to_truth_edge"].items())
        ),
        "node_mapping_differences_vs_current": _mapping_differences(
            baseline_node_mapping, mapping
        ),
        "concept_mapping_differences_vs_current": _mapping_differences(
            baseline_concept_mapping, concept_mapping
        ),
        "node_ambiguity_classes": alignment["node_ambiguity_classes"],
        "concept_ambiguity_classes": alignment["concept_ambiguity_classes"],
        "ambiguity_summary": {
            "node_class_count": len(alignment["node_ambiguity_classes"]),
            "concept_class_count": len(alignment["concept_ambiguity_classes"]),
            "optimum_is_unique": search["optimum_is_unique"],
        },
        "raw_joint_objective": alignment["objective"],
        "business_objective": raw_business_objective,
        "canonical_boundary_objective": canonical_objective,
        "objective_component_policy": {
            "business_nodes_and_concepts": "original_business_graph_only",
            "process_edges": "original_business_edges_only",
            "synthetic_boundary_edges_included": False,
            "structural_start_end": (
                "fixed virtual marker components; not Concepts"
                if use_start or use_end
                else "saved metadata components"
            ),
        },
        "process_topology_agreement": {
            "agreement": canonical_objective["components"]["process_edges"][
                "agreement"
            ],
            "matched_count": canonical_objective["components"]["process_edges"][
                "matched_count"
            ],
            "agent_business_edge_count": canonical_objective["components"][
                "process_edges"
            ]["agent_count"],
            "truth_business_edge_count": canonical_objective["components"][
                "process_edges"
            ]["truth_count"],
            "synthetic_edges_excluded": True,
        },
        "disagreement_audit_classification": variant_audit,
    }


def _numeric(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return value
    return default


def _reduction_fraction(base: Any, current: Any) -> float | None:
    if not isinstance(base, (int, float)) or base == 0:
        return None
    if not isinstance(current, (int, float)):
        return None
    return (base - current) / base


def run_alignment_experiment(
    agent_graph: Any,
    truth_graph: Any,
    *,
    name: str,
    seed: int | None = None,
    stored_joint_alignment: Mapping[str, Any] | None = None,
    observations: Iterable[Any] = (),
) -> dict[str, Any]:
    """Run current vs START-only/END-only/both diagnostic variants."""
    agent_audit = audit_boundary(agent_graph, graph_name="agent")
    truth_audit = audit_boundary(truth_graph, graph_name="truth")
    agent_normalized = normalize_graph(agent_graph, graph_name="agent")
    truth_normalized = normalize_graph(truth_graph, graph_name="truth")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "seed": seed,
        "boundary_audit": {"agent": agent_audit, "truth": truth_audit},
        "normalization": {
            "agent": agent_normalized.to_dict(),
            "truth": truth_normalized.to_dict(),
        },
        "alignment_variants": [],
    }
    if not agent_normalized.possible or not truth_normalized.possible:
        result["status"] = "normalization_not_attempted"
        result["invalid_reason"] = {
            "agent": agent_normalized.classification,
            "truth": truth_normalized.classification,
        }
        return result

    if agent_normalized.canonical is None or truth_normalized.canonical is None:
        raise RuntimeError("normalization reported possible without a canonical copy")
    agent_business_projection = agent_normalized.canonical.business_projection()
    truth_business_projection = truth_normalized.canonical.business_projection()

    stored_joint = dict(stored_joint_alignment or {})
    production_nodes = stored_joint.get("production_agent_node_to_truth_node", {})
    production_concepts = stored_joint.get(
        "production_agent_concept_to_truth_concept", {}
    )
    usage_concepts = stored_joint.get(
        "conditioned_usage_agent_concept_to_truth_concept", {}
    )
    from scripts.business_interview_joint_alignment_audit import (
        build_joint_concept_disagreement_audit,
    )

    current_model = build_joint_structural_alignment_diagnostics(
        agent_graph,
        truth_graph,
        production_node_to_truth=production_nodes,
        production_concept_to_truth=production_concepts,
        usage_concept_to_truth=usage_concepts,
    )
    current_alignment = _as_dict(current_model)
    current_audit = build_joint_concept_disagreement_audit(
        agent_graph,
        truth_graph,
        current_model,
        observations=observations,
        seed=seed,
    )
    variants = [
        _alignment_variant(
            "A_current_graph",
            agent_graph,
            truth_graph,
            use_start=False,
            use_end=False,
            baseline_alignment=current_alignment,
            production_node_mapping=production_nodes,
            production_concept_mapping=production_concepts,
            usage_concept_mapping=usage_concepts,
            baseline_audit=current_audit,
            observations=observations,
            seed=seed,
        ),
        _alignment_variant(
            "B_virtual_START_only",
            agent_business_projection,
            truth_business_projection,
            use_start=True,
            use_end=False,
            baseline_alignment=current_alignment,
            production_node_mapping=production_nodes,
            production_concept_mapping=production_concepts,
            usage_concept_mapping=usage_concepts,
            baseline_audit=current_audit,
            observations=observations,
            seed=seed,
        ),
        _alignment_variant(
            "C_virtual_END_only",
            agent_business_projection,
            truth_business_projection,
            use_start=False,
            use_end=True,
            baseline_alignment=current_alignment,
            production_node_mapping=production_nodes,
            production_concept_mapping=production_concepts,
            usage_concept_mapping=usage_concepts,
            baseline_audit=current_audit,
            observations=observations,
            seed=seed,
        ),
        _alignment_variant(
            "D_virtual_START_and_END",
            agent_business_projection,
            truth_business_projection,
            use_start=True,
            use_end=True,
            baseline_alignment=current_alignment,
            production_node_mapping=production_nodes,
            production_concept_mapping=production_concepts,
            usage_concept_mapping=usage_concepts,
            baseline_audit=current_audit,
            observations=observations,
            seed=seed,
        ),
    ]
    result["status"] = "ok"
    result["current_alignment_recomputed"] = {
        "matches_saved_snapshot": _alignment_snapshot_equal(
            current_alignment, stored_joint
        )
        if stored_joint
        else None,
        "search": _search_snapshot(current_alignment),
        "objective": current_alignment["objective"],
    }
    result["alignment_variants"] = variants
    result["anchor_effect_summary"] = _anchor_effect_summary(variants)
    result["canonical_contract"] = {
        "virtual_start_id": STRUCTURAL_START_NODE_ID,
        "virtual_end_id": STRUCTURAL_END_NODE_ID,
        "synthetic_edge_marker": STRUCTURAL_EDGE_MARKER,
        "concept_matching_excludes_structural_nodes": True,
        "business_edge_metrics_exclude_synthetic_edges": True,
        "human_renderer_may_hide_structural_nodes_and_edges": True,
    }
    return result


def _alignment_snapshot_equal(
    current: Mapping[str, Any], stored: Mapping[str, Any]
) -> bool:
    if not stored:
        return False
    fields = (
        "representative_agent_node_to_truth_node",
        "representative_agent_concept_to_truth_concept",
        "objective",
        "search",
    )
    return all(current.get(field) == stored.get(field) for field in fields)


def _anchor_effect_summary(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current = variants[0]
    summary: dict[str, Any] = {}
    for variant in variants[1:]:
        summary[variant["name"]] = {
            "candidate_pair_count_reduction": variant["search_space_delta_vs_current"][
                "candidate_pair_count_reduction"
            ],
            "node_search_states_reduction": variant["search_space_delta_vs_current"][
                "node_search_states_reduction"
            ],
            "node_mapping_changed": bool(
                variant["node_mapping_differences_vs_current"]
            ),
            "concept_mapping_changed": bool(
                variant["concept_mapping_differences_vs_current"]
            ),
            "node_ambiguity_class_delta": variant["ambiguity_summary"][
                "node_class_count"
            ]
            - current["ambiguity_summary"]["node_class_count"],
            "concept_ambiguity_class_delta": variant["ambiguity_summary"][
                "concept_class_count"
            ]
            - current["ambiguity_summary"]["concept_class_count"],
        }
    return summary


def trace_boundary_generation(public: Mapping[str, Any]) -> dict[str, Any]:
    """Trace endpoint declaration calls for a saved artifact.

    This is intentionally a protocol trace, not an inference: it identifies
    whether ``set_graph_endpoints`` was ever called and whether completion was
    reached before serialization.
    """
    endpoint_calls: list[dict[str, Any]] = []
    start_calls: list[dict[str, Any]] = []
    finish_calls: list[dict[str, Any]] = []
    for turn in public.get("conversation", []):
        for call in turn.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            record = {
                "turn_idx": turn.get("turn_idx"),
                "name": call.get("name"),
                "arguments": call.get("arguments") or {},
            }
            if call.get("name") == "set_graph_endpoints":
                endpoint_calls.append(record)
            elif call.get("name") == "start_inference":
                start_calls.append(record)
            elif call.get("name") == "finish_interview":
                finish_calls.append(record)
    final_graph = public.get("final_graph") or {}
    return {
        "termination_reason": public.get("termination_reason"),
        "episode_complete": public.get("episode_complete"),
        "start_inference_calls": start_calls,
        "set_graph_endpoints_calls": endpoint_calls,
        "finish_interview_calls": finish_calls,
        "endpoint_declaration_call_count": len(endpoint_calls),
        "endpoint_metadata_was_omitted_by_tool_protocol": not endpoint_calls,
        "serialized_start_node": final_graph.get("start_node_id"),
        "serialized_end_nodes": final_graph.get("end_node_ids", []),
        "diagnosis": (
            "graph construction started, but endpoint declaration was never invoked; "
            "the graph serializer retained the model defaults"
            if not endpoint_calls
            else "endpoint declaration was invoked; inspect its arguments and later updates"
        ),
    }


def _activity_pair_graph(
    node_ids: Sequence[str],
    edge_pairs: Sequence[tuple[str, str]],
    *,
    agent_prefix: str,
    start_node_id: str | None = None,
    end_node_ids: Sequence[str] = (),
    truth_prefix: str = "truth",
) -> tuple[BusinessProcessGraph, AgentGraph]:
    truth_concepts: dict[str, TruthConcept] = {}
    truth_nodes: dict[str, TruthNode] = {}
    truth_edges: dict[str, TruthEdge] = {}
    agent_concepts: dict[str, AgentConcept] = {}
    agent_nodes: dict[str, Node] = {}
    agent_edges: dict[str, Edge] = {}
    agent_node_by_truth = {
        node_id: (f"{agent_prefix}_{node_id}" if agent_prefix else node_id)
        for node_id in node_ids
    }
    for node_id in node_ids:
        truth_concept_id = f"{truth_prefix}_activity_{node_id}"
        agent_concept_id = f"{agent_prefix}_activity_{node_id}"
        truth_concepts[truth_concept_id] = TruthConcept(
            id=truth_concept_id,
            kind="activity",
            canonical_terms=[f"truth-only-{node_id}"],
        )
        agent_concepts[agent_concept_id] = AgentConcept(
            id=agent_concept_id,
            kind="activity",
            display_label=f"agent-only-{node_id}",
        )
        truth_nodes[node_id] = TruthNode(
            id=node_id,
            activity=ConceptRef(concept_id=truth_concept_id),
        )
        agent_node_id = agent_node_by_truth[node_id]
        agent_nodes[agent_node_id] = Node(
            id=agent_node_id,
            activity=ConceptRef(concept_id=agent_concept_id),
        )
    for index, (from_node, to_node) in enumerate(edge_pairs, start=1):
        truth_edges[f"truth_edge_{index}"] = TruthEdge(
            id=f"truth_edge_{index}",
            from_node=from_node,
            to_node=to_node,
        )
        agent_edges[f"agent_edge_{index}"] = Edge(
            id=f"agent_edge_{index}",
            from_node=agent_node_by_truth[from_node],
            to_node=agent_node_by_truth[to_node],
            condition=AbsentType(),
        )
    mapped_start = (
        agent_node_by_truth[start_node_id] if start_node_id is not None else None
    )
    mapped_ends = [agent_node_by_truth[node_id] for node_id in end_node_ids]
    truth = BusinessProcessGraph(
        id="fixture_truth",
        concepts=truth_concepts,
        nodes=truth_nodes,
        edges=truth_edges,
        start_node_id=start_node_id,
        end_node_ids=list(end_node_ids),
    )
    agent = AgentGraph(
        id="fixture_agent",
        concepts=agent_concepts,
        nodes=agent_nodes,
        edges=agent_edges,
        start_node_id=mapped_start,
        end_node_ids=mapped_ends,
    )
    return truth, agent


def _add_condition_concepts(
    truth: BusinessProcessGraph,
    agent: AgentGraph,
    edge_indexes: Iterable[int],
    *,
    prefix: str,
) -> None:
    for index in edge_indexes:
        truth_concept_id = f"{prefix}_truth_condition_{index}"
        agent_concept_id = f"{prefix}_agent_condition_{index}"
        truth.concepts[truth_concept_id] = TruthConcept(
            id=truth_concept_id,
            kind="condition",
        )
        agent.concepts[agent_concept_id] = AgentConcept(
            id=agent_concept_id,
            kind="condition",
            display_label=f"condition-{index}",
        )
        truth.edges[f"truth_edge_{index}"].condition = ConceptRef(
            concept_id=truth_concept_id
        )
        agent.edges[f"agent_edge_{index}"].condition = ConceptRef(
            concept_id=agent_concept_id
        )


def deterministic_fixture_pairs() -> list[tuple[str, BusinessProcessGraph, AgentGraph]]:
    """Return compact, text-independent boundary fixtures for the report."""
    fixtures: list[tuple[str, BusinessProcessGraph, AgentGraph]] = []
    truth, agent = _activity_pair_graph(
        ("entry_a", "entry_b", "join", "exit_x", "exit_y"),
        (
            ("entry_a", "join"),
            ("entry_b", "join"),
            ("join", "exit_x"),
            ("join", "exit_y"),
        ),
        agent_prefix="renamed_agent",
    )
    fixtures.append(("multiple_entry_and_exit", truth, agent))
    truth, agent = _activity_pair_graph(
        ("entry", "cycle_a", "cycle_b", "exit"),
        (
            ("entry", "cycle_a"),
            ("cycle_a", "cycle_b"),
            ("cycle_b", "cycle_a"),
            ("cycle_b", "exit"),
        ),
        agent_prefix="cycle_agent",
    )
    fixtures.append(("cycle_with_exit", truth, agent))
    truth, agent = _activity_pair_graph(
        ("sym_a", "sym_b", "sym_join", "sym_x", "sym_y"),
        (
            ("sym_a", "sym_join"),
            ("sym_b", "sym_join"),
            ("sym_join", "sym_x"),
            ("sym_join", "sym_y"),
        ),
        agent_prefix="renamed_symmetric",
    )
    fixtures.append(("symmetric_entry_exit_roles", truth, agent))
    return fixtures


def deterministic_invalid_fixture_pairs() -> list[
    tuple[str, BusinessProcessGraph, AgentGraph]
]:
    """Return invalid/edge-case fixtures with normalization classification."""
    fixtures: list[tuple[str, BusinessProcessGraph, AgentGraph]] = []
    cases = {
        "disconnected_component": (
            ("a", "b", "cycle_a", "cycle_b"),
            (("a", "b"), ("cycle_a", "cycle_b"), ("cycle_b", "cycle_a")),
        ),
        "multiple_independent_processes": (
            ("p1_a", "p1_b", "p2_a", "p2_b"),
            (("p1_a", "p1_b"), ("p2_a", "p2_b")),
        ),
        "cycle": (
            ("entry", "a", "b", "end"),
            (
                ("entry", "a"),
                ("a", "b"),
                ("b", "a"),
                ("b", "end"),
            ),
        ),
        "self_loop": (
            ("entry", "a", "end"),
            (("entry", "a"), ("a", "a"), ("a", "end")),
        ),
        "source_missing": (("a", "b"), (("a", "b"), ("b", "a"))),
        "sink_missing": (("a", "b"), (("a", "b"), ("b", "b"))),
        "isolated_node": (("a", "b", "isolated"), (("a", "b"),)),
        "multiple_apparent_entry_conditions": (
            ("entry_a", "entry_b", "join", "exit"),
            (("entry_a", "join"), ("entry_b", "join"), ("join", "exit")),
        ),
        "multiple_apparent_exit_conditions": (
            ("entry", "exit_a", "exit_b"),
            (("entry", "exit_a"), ("entry", "exit_b")),
        ),
    }
    for name, (node_ids, edges) in cases.items():
        truth, agent = _activity_pair_graph(
            node_ids,
            edges,
            agent_prefix=f"invalid_{name}",
        )
        if name == "multiple_apparent_entry_conditions":
            _add_condition_concepts(truth, agent, (1, 2), prefix=name)
        elif name == "multiple_apparent_exit_conditions":
            _add_condition_concepts(truth, agent, (1, 2), prefix=name)
        fixtures.append((name, truth, agent))

    truth, agent = _activity_pair_graph(
        (STRUCTURAL_START_NODE_ID, "business", STRUCTURAL_END_NODE_ID),
        (
            (STRUCTURAL_START_NODE_ID, "business"),
            ("business", STRUCTURAL_END_NODE_ID),
        ),
        agent_prefix="",
    )
    fixtures.append(("existing_explicit_structural_boundary_ids", truth, agent))
    return fixtures


def run_deterministic_fixture_audits() -> dict[str, Any]:
    """Return JSON-ready audits/experiments for deterministic fixture graphs."""
    fixture_audits: list[dict[str, Any]] = []
    for name, truth, agent in deterministic_fixture_pairs():
        fixture_audits.append(run_alignment_experiment(agent, truth, name=name))
    invalid_audits: list[dict[str, Any]] = []
    for name, truth, agent in deterministic_invalid_fixture_pairs():
        agent_result = normalize_graph(agent, graph_name="agent")
        truth_result = normalize_graph(truth, graph_name="truth")
        invalid_audits.append(
            {
                "name": name,
                "agent": agent_result.to_dict(),
                "truth": truth_result.to_dict(),
                "classification_agrees": agent_result.classification
                == truth_result.classification,
                "expected_handling": _expected_handling(agent_result.classification),
            }
        )
    return {
        "fixtures": fixture_audits,
        "invalid_graph_cases": invalid_audits,
    }


def _expected_handling(classification: str) -> str:
    if classification.startswith("normalizable"):
        return "normalization_possible"
    if classification == "multiple_independent_processes":
        return "separate_process_representation_required"
    return "invalid_single_process_graph"


def load_saved_seed_pair(
    artifact_dir: Path, seed: int
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Load one saved public artifact and its decoded Agent/Truth graphs."""
    from scripts.business_interview_evaluation_diagnostics import load_artifact

    public_path = artifact_dir / f"run_00_seed{seed}.json"
    private_path = artifact_dir / f"run_00_seed{seed}.private.json"
    public, _private, db, truth, _knowledge = load_artifact(public_path, private_path)
    return public, db.graph, truth, db.observations


def run_saved_seed_experiments(
    artifact_dir: Path,
    seeds: Iterable[int] = (9002, 9003, 9004),
) -> dict[str, Any]:
    """Run boundary audits for the requested immutable saved artifacts."""
    seed_list = list(seeds)
    experiments: list[dict[str, Any]] = []
    for seed in seed_list:
        public, agent, truth, observations = load_saved_seed_pair(artifact_dir, seed)
        stored_joint: dict[str, Any] = {}
        diagnostics_path = artifact_dir / f"run_00_seed{seed}.diagnostics.json"
        if diagnostics_path.exists():
            import json

            try:
                diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                diagnostics = {}
            stored_joint = (
                diagnostics.get("evaluation", {})
                .get("diagnostics", {})
                .get("joint_structural_alignment", {})
            )
        experiment = run_alignment_experiment(
            agent,
            truth,
            name=f"seed_{seed}",
            seed=seed,
            stored_joint_alignment=stored_joint,
            observations=observations,
        )
        experiment["generation_trace"] = trace_boundary_generation(public)
        experiments.append(experiment)
    return {"seeds": seed_list, "experiments": experiments}


def build_boundary_audit_json(
    saved_experiments: Mapping[str, Any], deterministic: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the compact boundary audit deliverable."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "boundary_audit",
        "saved_real_llm": [
            {
                "name": experiment["name"],
                "seed": experiment.get("seed"),
                "boundary_audit": experiment["boundary_audit"],
                "generation_trace": experiment.get("generation_trace"),
                "normalization": {
                    side: {
                        "possible": value["possible"],
                        "classification": value["classification"],
                        "reason": value["reason"],
                        "canonical_invariant": (
                            value.get("canonical_copy", {}).get("invariant")
                            if value["possible"]
                            else None
                        ),
                    }
                    for side, value in experiment["normalization"].items()
                },
            }
            for experiment in saved_experiments["experiments"]
        ],
        "deterministic_fixtures": {
            "boundary_cases": [
                {
                    "name": item["name"],
                    "status": item["status"],
                    "boundary_audit": item["boundary_audit"],
                    "normalization": item["normalization"],
                }
                for item in deterministic["fixtures"]
            ],
            "invalid_graph_cases": deterministic["invalid_graph_cases"],
        },
    }


def build_comparison_json(
    saved_experiments: Mapping[str, Any], deterministic: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the current-vs-A/B/C/D comparison deliverable."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "current_vs_normalized_alignment_comparison",
        "method": {
            "label_independent": True,
            "production_scoring_changed": False,
            "production_matcher_changed": False,
            "synthetic_edges_in_business_metrics": False,
            "anchor_projection": "fixed virtual marker to topology source/sink role equality",
        },
        "saved_real_llm": [
            {
                "name": experiment["name"],
                "seed": experiment.get("seed"),
                "current_alignment_recomputed": experiment.get(
                    "current_alignment_recomputed"
                ),
                "alignment_variants": experiment.get("alignment_variants", []),
                "anchor_effect_summary": experiment.get("anchor_effect_summary", {}),
            }
            for experiment in saved_experiments["experiments"]
        ],
        "deterministic_fixtures": [
            {
                "name": item["name"],
                "alignment_variants": item.get("alignment_variants", []),
                "anchor_effect_summary": item.get("anchor_effect_summary", {}),
            }
            for item in deterministic["fixtures"]
        ],
    }


def write_deliverables(
    artifact_dir: Path,
    audit_path: Path,
    comparison_path: Path,
    *,
    seeds: Iterable[int] = (9002, 9003, 9004),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate both JSON deliverables without touching saved artifacts."""
    import json

    saved = run_saved_seed_experiments(artifact_dir, seeds)
    deterministic = run_deterministic_fixture_audits()
    audit = build_boundary_audit_json(saved, deterministic)
    comparison = build_comparison_json(saved, deterministic)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit, comparison
