"""Stakeholder as a filter over the single Truth process graph.

The stakeholder does not get a separate model — there is exactly one Truth
``BusinessProcessGraph``. A stakeholder is represented by a ``StakeholderFilter``
describing which parts of the Truth graph (nodes / edges / node properties /
edge properties) they can observe and talk about. Multiple filters can be
applied to the same Truth graph.

Visibility is per **property**:

- node properties: activity / actor / system / reads / writes / rationale
- edge properties: condition (edge existence is visible for every visible edge)

``apply`` returns a filtered graph with non-visible nodes / edges / properties
dropped, so hidden information can never leak to the simulator. The Truth graph
itself is never serialized wholesale into a simulator prompt.
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import (
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    Node,
)

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")
_EDGE_PROPS = ("condition",)


class StakeholderFilter(BaseModel):
    """What a stakeholder can observe of the Truth graph.

    - ``visible_node_ids`` / ``visible_edge_ids``: which nodes / edges are visible.
    - ``visible_attributes``: global default node properties (activity / actor /
      system / reads / writes / rationale).
    - ``visible_node_attributes``: **per-node** property visibility; a node
      listed here is limited to exactly these properties; a node not listed
      uses the global ``visible_attributes``.
    - ``visible_edge_attributes``: per-edge property visibility (``condition``);
      an edge not listed has no visible condition.

    The full Truth may contain information not available to this stakeholder;
    ``visible_*`` define what the stakeholder can actually assert. Hidden
    (non-visible) properties are expected to remain unknown, not invented.
    """

    name: str
    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_ids: list[str] = Field(default_factory=list)
    visible_attributes: list[str] = Field(default_factory=list)
    visible_node_attributes: dict[str, list[str]] = Field(default_factory=dict)
    visible_edge_attributes: dict[str, list[str]] = Field(default_factory=dict)

    def node_properties_for(self, node_id: str) -> set[str]:
        """The set of visible node properties for ``node_id``.

        Per-node visibility wins when present; otherwise the global
        ``visible_attributes`` applies.
        """
        if node_id in self.visible_node_attributes:
            return set(self.visible_node_attributes[node_id])
        return set(self.visible_attributes)

    def edge_properties_for(self, edge_id: str) -> set[str]:
        """The set of visible edge properties (condition) for ``edge_id``."""
        return set(self.visible_edge_attributes.get(edge_id, []))

    def apply(self, truth: BusinessProcessGraph) -> BusinessProcessGraph:
        """Return a filtered graph containing only visible information."""
        visible_nodes = set(self.visible_node_ids)
        visible_edges = set(self.visible_edge_ids)

        new_nodes: dict[str, Node] = {}
        for nid, node in truth.nodes.items():
            if nid not in visible_nodes:
                continue
            props = self.node_properties_for(nid)
            new_nodes[nid] = _filtered_node(node, props)

        new_edges: dict[str, Edge] = {}
        for eid, edge in truth.edges.items():
            if eid not in visible_edges:
                continue
            if edge.from_node not in new_nodes or edge.to_node not in new_nodes:
                continue
            eprops = self.edge_properties_for(eid)
            new_edges[eid] = Edge(
                id=edge.id,
                from_node=edge.from_node,
                to_node=edge.to_node,
                condition=_clone_ref(edge.condition)
                if edge.condition is not None and "condition" in eprops
                else None,
                evidence=list(edge.evidence),
            )

        return BusinessProcessGraph(
            id=truth.id,
            name=truth.name,
            nodes=new_nodes,
            edges=new_edges,
            concepts={},
        )

    def describe(self, truth: BusinessProcessGraph) -> str:
        """A short human-readable description of the visible portion (for tests /
        verification). The full Truth graph is never serialized here."""
        filtered = self.apply(truth)
        parts = [f"Stakeholder '{self.name}' sees:"]
        for nid in filtered.nodes:
            n = filtered.nodes[nid]
            parts.append(
                f"node {nid}: {n.activity.concept_id}"
                + (f" (actor {n.actor.concept_id})" if n.actor is not None else "")
            )
        for e in filtered.edges.values():
            cond = e.condition.concept_id if e.condition is not None else None
            parts.append(
                f"edge {e.from_node}->{e.to_node}" + (f" [{cond}]" if cond else "")
            )
        return "\n".join(parts)


def _clone_ref(ref: Optional[ConceptRef]) -> Optional[ConceptRef]:
    if ref is None:
        return None
    return ConceptRef(
        concept_id=ref.concept_id,
        confidence=ref.confidence,
        evidence=list(ref.evidence),
    )


def _filtered_node(node: Node, props: set[str]) -> Node:
    def keep(prop: str) -> bool:
        return prop in props

    return Node(
        id=node.id,
        activity=_clone_ref(node.activity) if keep("activity") else None,  # type: ignore[arg-type]
        actor=_clone_ref(node.actor) if keep("actor") else None,
        system=_clone_ref(node.system) if keep("system") else None,
        reads=[_clone_ref(r) for r in node.reads] if keep("reads") else [],  # type: ignore[arg-type]
        writes=[_clone_ref(w) for w in node.writes] if keep("writes") else [],  # type: ignore[arg-type]
        necessity_rationale=(
            _clone_ref(node.necessity_rationale) if keep("rationale") else None
        ),
    )
