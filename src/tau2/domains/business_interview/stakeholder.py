"""Stakeholder as a filter over the single Truth DAG.

The stakeholder does not get a separate Workflow/DAG model — there is exactly one
Truth DAG. A stakeholder is represented by a ``StakeholderFilter`` describing the
part of the Truth DAG (nodes / edges / attributes / necessity) they can observe
and talk about. Multiple filters can be applied to the same Truth DAG.

``apply`` returns a filtered ``BusinessDAG`` with non-visible nodes / edges /
attributes dropped, so hidden information can never leak to the simulator. The
Truth DAG itself is never serialized wholesale into a simulator prompt.
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.dag import (
    BusinessDAG,
    Edge,
    InferredValue,
    Necessity,
    Node,
)

_ATTRIBUTES = ("actor", "system", "reads", "writes")
_NECESSITY_PROPS = ("rationale", "owner", "evidence", "removal_impact")


class StakeholderFilter(BaseModel):
    """What a stakeholder can observe of the Truth DAG.

    - ``visible_node_ids`` / ``visible_edge_ids``: which nodes / edges are visible.
    - ``visible_attributes``: which of actor / system / reads / writes are visible.
    - ``visible_necessity``: per node id, which necessity props are visible
      (an absent node id => no necessity is visible for it).
    """

    name: str
    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_ids: list[str] = Field(default_factory=list)
    visible_attributes: list[str] = Field(default_factory=list)
    visible_necessity: dict[str, list[str]] = Field(default_factory=dict)

    def apply(self, truth: BusinessDAG) -> BusinessDAG:
        """Return a filtered DAG containing only visible information."""
        visible_nodes = set(self.visible_node_ids)
        visible_edges = set(self.visible_edge_ids)
        attr_set = set(self.visible_attributes)

        new_nodes: dict[str, Node] = {}
        for nid, node in truth.nodes.items():
            if nid not in visible_nodes:
                continue
            new_nodes[nid] = _filtered_node(
                node, attr_set, self.visible_necessity.get(nid, [])
            )

        new_edges: dict[str, Edge] = {}
        for eid, edge in truth.edges.items():
            if eid not in visible_edges:
                continue
            if edge.from_node not in new_nodes or edge.to_node not in new_nodes:
                continue
            new_edges[eid] = Edge(
                id=edge.id,
                from_node=edge.from_node,
                to_node=edge.to_node,
                predicate=_filter_value(edge.predicate),
                observation_ids=list(edge.observation_ids),
            )

        start = truth.start_node_id if truth.start_node_id in new_nodes else None
        ends = [e for e in truth.end_node_ids if e in new_nodes]
        return BusinessDAG(
            id=truth.id,
            name=truth.name,
            nodes=new_nodes,
            edges=new_edges,
            start_node_id=start,
            end_node_ids=ends,
        )

    def describe(self, truth: BusinessDAG) -> str:
        """A short human-readable description of the visible portion (for tests /
        verification). The full Truth DAG is never serialized here."""
        filtered = self.apply(truth)
        parts = [f"Stakeholder '{self.name}' sees:"]
        for nid in filtered.nodes:
            n = filtered.nodes[nid]
            parts.append(f"node {nid}: {n.action.value or '?'}")
        for e in filtered.edges.values():
            pred = e.predicate.value if e.predicate else None
            parts.append(
                f"edge {e.from_node}->{e.to_node}" + (f" [{pred}]" if pred else "")
            )
        return "\n".join(parts)


def _filter_value(v: Optional[InferredValue]) -> Optional[InferredValue]:
    if v is None:
        return None
    return InferredValue(
        value=v.value, confidence=v.confidence, observation_ids=list(v.observation_ids)
    )


def _filtered_node(node: Node, attr_set: set[str], visible_nec: list[str]) -> Node:
    def keep(attr: str) -> bool:
        return attr in attr_set

    necessity = None
    if node.necessity is not None:
        nec_props: dict[str, InferredValue] = {}
        for prop in _NECESSITY_PROPS:
            if prop in visible_nec:
                nec_props[prop] = _filter_value(getattr(node.necessity, prop))
        necessity = Necessity(**nec_props) if nec_props else None

    return Node(
        id=node.id,
        action=_filter_value(node.action),
        actor=_filter_value(node.actor) if keep("actor") else InferredValue(),
        system=_filter_value(node.system) if keep("system") else InferredValue(),
        reads=[_filter_value(r) for r in node.reads] if keep("reads") else [],
        writes=[_filter_value(w) for w in node.writes] if keep("writes") else [],
        necessity=necessity,
        observation_ids=list(node.observation_ids),
    )
