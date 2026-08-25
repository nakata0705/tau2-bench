"""The stakeholder's world model: an explicit stakeholder-knowledge graph.

``StakeholderKnowledge`` is a **physical projection of the Truth graph** —
the stakeholder's own world model:

    StakeholderKnowledge
      ├─ StakeholderKnowledgeGraph     (masked nodes/edges, three-valued slots)
      └─ StakeholderKnowledgeConcept[] (local understanding of Truth concepts)

Masking rules (``project_knowledge``):

- does not know a node exists -> node is removed only through a validated
  serial-path shortcut contraction; branch/merge/conditioned Truth paths are
  rejected;
- does not know an edge exists -> edge is removed, and the post-forgetting
  graph is rejected if that removal breaks the canonical topology;
- knows an element exists but not a property -> that property slot is
  ``DONT_KNOW``;
- knows a value is absent -> the slot is ``None``;
- knows a value -> ``ConceptRef`` (referencing a StakeholderKnowledgeConcept).

So ``ConceptRef != None != DONT_KNOW``. For v1, reads/writes use
whole-property known / DONT_KNOW (partial-list knowledge is future work).

**Opaque stakeholder-local IDs.** Every element of the knowledge graph has a
stable, semantically opaque local id assigned deterministically from sorted
Truth ids (``skn_001`` / ``ske_001`` / ``skc_001`` ...): ids never derive
from Truth ids, labels, terms, node ids or edge ids, and are invariant to
collection reordering. The private mappings to Truth
(``node_truth_ids`` / ``edge_truth_ids`` / ``StakeholderKnowledgeConcept.
truth_concept_id``) are evaluator-only and never exposed to the simulator
prompt or the Agent.

**One canonical semantic-ID resolver.** ``StakeholderKnowledgeGraph.resolve``
is the single deterministic resolver for the knowledge graph. Every semantic
ID resolves to exactly one address/object:

    node:<skn>                node existence/object (never the activity slot)
    node:<skn>:<prop>         exact property slot (three-valued value)
    node:<skn>:reads:<skc>    exact reads element (one ConceptRef)
    node:<skn>:writes:<skc>   exact writes element (one ConceptRef)
    edge:<ske>                edge existence/object
    edge:<ske>:condition      exact condition slot
    <skc>                     exact StakeholderKnowledgeConcept

``resolve`` returns ``None`` for anything unresolvable; annotation
validation, provenance, concept binding, DONT_KNOW handling, knowledge
coverage and diagnostics all go through it (no other semantic-id parsers
exist).

``StakeholderKnowledgeConcept`` is the stakeholder's local understanding of
one Truth concept; ``truth_concept_id`` is the private evaluator mapping and
is never exposed to the simulator or the Agent. Description and terminology
vary independently (term known/details unknown, details known/wrong local
term, details known/term unknown, both known). Descriptions describe the
concept itself, never workflow-position facts.
"""

import random
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import (
    _NODE_PROPS,
    DONT_KNOW,
    STRUCTURAL_SINK_ID,
    STRUCTURAL_SOURCE_ID,
    ConceptKind,
    ConceptRef,
    DontKnowType,
    TruthEdge,
    business_edge_ids,
    business_node_ids,
    canonical_structure_errors,
    edge_is_structural,
    graph_semantic_ids,
    is_dont_know,
)

# A property slot: ConceptRef (value known) | None (known absent) |
# DONT_KNOW (element known, value unknown).
Slot = Optional[Union[ConceptRef, DontKnowType]]
ListSlot = Optional[Union[list[ConceptRef], DontKnowType]]


def _node_slot_value(node, prop: str):
    """The three-valued knowledge slot value of a stakeholder node."""
    if prop in ("reads", "writes"):
        return getattr(node, prop)
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


def slot_concepts(value) -> set[str]:
    """Knowledge concept ids represented by a three-valued slot value
    (ConceptRef -> itself; list -> its elements; None / DONT_KNOW -> none)."""
    if isinstance(value, ConceptRef):
        return {value.concept_id}
    if isinstance(value, list):
        return {r.concept_id for r in value}
    return set()


def _slot_refs(value) -> list[ConceptRef]:
    if isinstance(value, ConceptRef):
        return [value]
    if isinstance(value, list):
        return [ref for ref in value if isinstance(ref, ConceptRef)]
    return []


class StakeholderKnowledgeConcept(BaseModel):
    """The stakeholder's local understanding of one TruthConcept.

    ``id`` is stakeholder-local and semantically opaque (never the hidden
    Truth id); the private ``truth_concept_id`` mapping is evaluator-only.
    ``description`` and ``terms`` are each ``str | list[str] | DONT_KNOW``
    and vary independently. Descriptions describe the concept itself, never
    workflow-position facts.
    """

    id: str
    truth_concept_id: str = Field(description="Private evaluator mapping.")
    kind: ConceptKind
    description: Union[str, DontKnowType] = Field(default_factory=lambda: DONT_KNOW)
    terms: Union[list[str], DontKnowType] = Field(default_factory=lambda: DONT_KNOW)

    def has_description(self) -> bool:
        return not is_dont_know(self.description)

    def has_terms(self) -> bool:
        return not is_dont_know(self.terms)


class StakeholderNode(BaseModel):
    """One stakeholder node with explicit structural/protection metadata.

    Structural SOURCE/SINK nodes are retained independently of semantic
    knowledge.  Their business slots are intentionally empty and are not
    forgettable.
    """

    id: str
    activity: Slot = None
    actor: Slot = None
    system: Slot = None
    reads: ListSlot = None
    writes: ListSlot = None
    necessity_rationale: Slot = None
    structural: bool = False
    structural_role: Optional[Literal["source", "sink"]] = None
    protected: bool = False

    @property
    def is_structural(self) -> bool:
        return self.structural or self.structural_role is not None


class StakeholderEdge(BaseModel):
    """One stakeholder edge with explicit structural/shortcut provenance."""

    id: str
    from_node: str
    to_node: str
    condition: Slot = None
    edge_kind: Literal["business", "structural_boundary", "shortcut"] = "business"
    structural_only: bool = False
    protected: bool = False
    is_shortcut: bool = False
    contracted_nodes: list[str] = Field(default_factory=list)
    derived_from_edges: list[str] = Field(default_factory=list)

    @property
    def is_structural(self) -> bool:
        return self.structural_only or self.edge_kind == "structural_boundary"


ResolvedKind = Literal[
    "node", "node_slot", "node_element", "edge", "edge_slot", "concept"
]


class ResolvedSemantic(BaseModel):
    """The canonical resolution of ONE semantic id in the stakeholder
    knowledge graph. Exactly one address/object per id:

    - ``kind="node"``        -> ``node`` (the element exists)
    - ``kind="node_slot"``   -> ``node`` + ``prop`` + three-valued ``value``
    - ``kind="node_element"``-> ``node`` + ``prop`` + the exact ``ref``
    - ``kind="edge"``        -> ``edge`` (the element exists)
    - ``kind="edge_slot"``   -> ``edge`` + ``prop="condition"`` + ``value``
    - ``kind="concept"``     -> ``concept``

    ``node_id`` / ``edge_id`` carry the local (opaque) ids.
    """

    kind: ResolvedKind
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    prop: Optional[str] = None
    value: object = None
    node: Optional[StakeholderNode] = None
    edge: Optional[StakeholderEdge] = None
    ref: Optional[ConceptRef] = None
    concept: Optional[StakeholderKnowledgeConcept] = None


class StakeholderKnowledgeGraph(BaseModel):
    """The stakeholder's own canonical world model: masked nodes/edges with
    three-valued property slots, explicit protected SOURCE/SINK boundaries,
    and the stakeholder's local concepts.

    Element ids are stakeholder-local and opaque (``skn_001`` / ``ske_001`` /
    ``skc_001`` style); the private ``node_truth_ids`` / ``edge_truth_ids``
    mappings (local -> Truth) and per-concept ``truth_concept_id`` are
    evaluator-only and never exposed to the simulator prompt or the Agent.
    """

    id: str = "knowledge"
    name: str = ""
    nodes: dict[str, StakeholderNode] = Field(default_factory=dict)
    edges: dict[str, StakeholderEdge] = Field(default_factory=dict)
    concepts: dict[str, StakeholderKnowledgeConcept] = Field(default_factory=dict)
    source_node_id: str = STRUCTURAL_SOURCE_ID
    sink_node_id: str = STRUCTURAL_SINK_ID
    # Legacy Agent endpoint fields are retained only as a business projection
    # convenience; canonical stakeholder topology is source_node_id/sink_node_id.
    start_node_id: Optional[str] = None
    end_node_ids: list[str] = Field(default_factory=list)
    node_truth_ids: dict[str, str] = Field(
        default_factory=dict, description="Private: local node id -> Truth node id."
    )
    edge_truth_ids: dict[str, str] = Field(
        default_factory=dict, description="Private: local edge id -> Truth edge id."
    )
    shortcut_provenance: dict[str, dict] = Field(
        default_factory=dict,
        description="Private evaluator provenance for derived shortcut edges.",
    )

    def semantic_ids(self) -> set[str]:
        """All addressable semantic IDs, including protected boundaries."""
        ids = graph_semantic_ids(self.nodes, self.edges)
        ids.update(self.concepts)
        return ids

    def structure_errors(self) -> list[str]:
        """Validate local references and the post-forgetting topology."""
        errors: list[str] = []
        for edge_id, edge in self.edges.items():
            if edge.from_node not in self.nodes:
                errors.append(f"edge {edge_id}: from_node not found: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"edge {edge_id}: to_node not found: {edge.to_node}")
            if (
                isinstance(edge.condition, ConceptRef)
                and edge.condition.concept_id not in self.concepts
            ):
                errors.append(
                    f"edge {edge_id}: condition references unknown concept "
                    f"{edge.condition.concept_id!r}"
                )
        for node_id, node in self.nodes.items():
            for prop in _NODE_PROPS:
                for ref in _slot_refs(_node_slot_value(node, prop)):
                    if ref.concept_id not in self.concepts:
                        errors.append(
                            f"node {node_id}: {prop} references unknown concept "
                            f"{ref.concept_id!r}"
                        )
        return errors + canonical_structure_errors(self)

    @property
    def is_valid(self) -> bool:
        return not self.structure_errors()

    def resolve(self, semantic_id: str) -> Optional[ResolvedSemantic]:
        """The canonical resolver: each semantic id resolves to exactly one
        address/object, or ``None`` when unresolvable.

        ``node:<id>`` resolves to the node's EXISTENCE — never to the
        activity slot; ``node:<id>:reads:<k>`` resolves to the exact reads
        element, never the whole reads list. This is the ONLY semantic-id
        parser for the stakeholder knowledge graph.
        """
        if semantic_id in self.concepts:
            concept = self.concepts[semantic_id]
            return ResolvedSemantic(kind="concept", concept=concept, value=concept)
        if semantic_id.startswith("node:"):
            parts = semantic_id.split(":")
            if len(parts) == 2:
                node = self.nodes.get(parts[1])
                if node is not None:
                    return ResolvedSemantic(
                        kind="node", node_id=parts[1], node=node, value=node
                    )
            elif len(parts) == 3:
                prop = parts[2]
                if prop in _NODE_PROPS:
                    node = self.nodes.get(parts[1])
                    if node is not None:
                        return ResolvedSemantic(
                            kind="node_slot",
                            node_id=parts[1],
                            prop=prop,
                            node=node,
                            value=_node_slot_value(node, prop),
                        )
            elif len(parts) == 4 and parts[2] in ("reads", "writes"):
                node = self.nodes.get(parts[1])
                if node is not None:
                    axis = getattr(node, parts[2])
                    if isinstance(axis, list):
                        for ref in axis:
                            if ref.concept_id == parts[3]:
                                return ResolvedSemantic(
                                    kind="node_element",
                                    node_id=parts[1],
                                    prop=parts[2],
                                    node=node,
                                    ref=ref,
                                    value=ref,
                                )
            return None
        if semantic_id.startswith("edge:"):
            parts = semantic_id.split(":")
            if len(parts) == 2:
                edge = self.edges.get(parts[1])
                if edge is not None:
                    return ResolvedSemantic(
                        kind="edge", edge_id=parts[1], edge=edge, value=edge
                    )
            elif len(parts) == 3 and parts[2] == "condition":
                edge = self.edges.get(parts[1])
                if edge is not None:
                    return ResolvedSemantic(
                        kind="edge_slot",
                        edge_id=parts[1],
                        prop="condition",
                        edge=edge,
                        value=edge.condition,
                    )
            return None
        return None

    def referenced_concept_ids(self) -> set[str]:
        """Knowledge-concept ids referenced by any known property value."""
        ids: set[str] = set()
        for node in self.nodes.values():
            for prop in _NODE_PROPS:
                ids.update(slot_concepts(_node_slot_value(node, prop)))
        for edge in self.edges.values():
            ids.update(slot_concepts(edge.condition))
        return ids


class StakeholderKnowledge(BaseModel):
    """The complete semantic knowledge of one stakeholder.

    Contains ONLY what the stakeholder knows: the masked world graph and the
    local concepts it references. Hidden Truth concepts/relations never
    enter this structure (and therefore never reach the simulator prompt).
    """

    graph: StakeholderKnowledgeGraph = Field(default_factory=StakeholderKnowledgeGraph)

    @property
    def concepts(self) -> dict[str, StakeholderKnowledgeConcept]:
        return self.graph.concepts

    def semantic_ids(self) -> set[str]:
        return self.graph.semantic_ids()


def validate_stakeholder_knowledge_graph(graph: StakeholderKnowledgeGraph) -> None:
    """Raise when a post-forgetting stakeholder graph is not canonical."""
    errors = graph.structure_errors()
    if errors:
        raise ValueError(
            "Invalid stakeholder knowledge graph:\n- " + "\n- ".join(errors)
        )


def contract_serial_node(
    graph: StakeholderKnowledgeGraph,
    node_id: str,
) -> StakeholderKnowledgeGraph:
    """Safely contract one local serial node and return a validated copy.

    This public primitive is intentionally stricter than generic graph
    simplification: both incident conditions must be known unconditional
    (``None``), the node must be non-structural with one predecessor and one
    successor, and no parallel endpoint relation may already exist.
    """
    validate_stakeholder_knowledge_graph(graph)
    if node_id not in graph.nodes:
        raise ValueError(f"node not found: {node_id}")
    node = graph.nodes[node_id]
    if node.is_structural or node_id in {graph.source_node_id, graph.sink_node_id}:
        raise ValueError("structural SOURCE/SINK nodes are protected")
    incoming = [edge for edge in graph.edges.values() if edge.to_node == node_id]
    outgoing = [edge for edge in graph.edges.values() if edge.from_node == node_id]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise ValueError("serial contraction requires indegree=1 and outdegree=1")
    left, right = incoming[0], outgoing[0]
    if left.condition is not None or right.condition is not None:
        raise ValueError(
            "serial contraction requires known unconditional incident edges"
        )
    if left.from_node == right.to_node:
        raise ValueError("serial contraction would create a self-loop")
    if any(
        edge.id not in {left.id, right.id}
        and edge.from_node == left.from_node
        and edge.to_node == right.to_node
        for edge in graph.edges.values()
    ):
        raise ValueError("serial contraction would create ambiguous parallel edges")
    copy = graph.model_copy(deep=True)
    new_id = f"__tau2_shortcut__{node_id}"
    if new_id in copy.edges:
        raise ValueError(f"shortcut edge id collision: {new_id}")
    structural_boundary = (
        left.from_node == copy.source_node_id or right.to_node == copy.sink_node_id
    )
    edge_kind = "structural_boundary" if structural_boundary else "shortcut"
    derived = (list(left.derived_from_edges) or [left.id]) + (
        list(right.derived_from_edges) or [right.id]
    )
    contracted = list(left.contracted_nodes) + [node_id] + list(right.contracted_nodes)
    copy.edges.pop(left.id)
    copy.edges.pop(right.id)
    copy.nodes.pop(node_id)
    if copy.start_node_id == node_id:
        copy.start_node_id = (
            right.to_node
            if right.to_node not in {copy.source_node_id, copy.sink_node_id}
            else None
        )
    copy.end_node_ids = [
        left.from_node if end_id == node_id else end_id
        for end_id in copy.end_node_ids
        if end_id != copy.source_node_id
    ]
    copy.edges[new_id] = StakeholderEdge(
        id=new_id,
        from_node=left.from_node,
        to_node=right.to_node,
        condition=None,
        edge_kind=edge_kind,
        structural_only=structural_boundary,
        protected=structural_boundary,
        is_shortcut=True,
        contracted_nodes=contracted,
        derived_from_edges=derived,
    )
    copy.node_truth_ids.pop(node_id, None)
    copy.edge_truth_ids.pop(left.id, None)
    copy.edge_truth_ids.pop(right.id, None)
    copy.edge_truth_ids[new_id] = "__shortcut__"
    copy.shortcut_provenance[new_id] = {
        "is_shortcut": True,
        "contracted_nodes": contracted,
        "derived_from_edges": derived,
    }
    validate_stakeholder_knowledge_graph(copy)
    return copy


def _local_ids(truth_ids: list[str], prefix: str) -> dict[str, str]:
    """Deterministic opaque local ids: sorted by Truth id, zero-padded
    (``skn_001`` ...). Invariant to collection reordering and never derived
    from Truth id meaning."""
    return {tid: f"{prefix}{i:03d}" for i, tid in enumerate(sorted(truth_ids), 1)}


class KnowledgeProjectionError(ValueError):
    """Raised when bounded forgetting rejection sampling cannot produce a valid graph."""

    def __init__(
        self, message: str, *, attempts: int, reasons: list[str], config: dict
    ):
        self.attempts = attempts
        self.reasons = reasons
        self.config = config
        super().__init__(message)


class _ProjectionRejected(Exception):
    """Internal reason for one invalid forgetting sample."""


def _truth_edge_condition_is_safe(edge, known_condition_edges: set[str]) -> bool:
    """Only condition-free Truth paths may be contracted.

    ``known_condition_edges`` controls the epistemic state of the derived
    condition slot; it does not turn a Truth condition into an unconditional
    path.  Thus a stakeholder may retain a shortcut with DONT_KNOW condition
    knowledge, but never receives an invented composed condition.
    """
    return edge.condition is None


def _contract_forgotten_nodes(
    truth,
    nodes: set[str],
    edges: dict[str, TruthEdge],
    forgotten_nodes: set[str],
    known_condition_edges: set[str],
    *,
    allow_shortcut: bool,
) -> tuple[set[str], dict[str, TruthEdge], set[str]]:
    """Remove forgotten nodes only through safe serial-path contraction.

    This operation is intentionally conservative: it counts actual incident
    edges, rejects branches/merges, rejects any conditioned or unknown
    incident condition, rejects self-loops/parallel endpoint pairs, and never
    leaves a dangling reference.  The returned edge objects retain private
    provenance in ``contracted_nodes`` / ``derived_from_edges``.
    """
    if forgotten_nodes and not allow_shortcut:
        raise _ProjectionRejected("shortcut contraction is disabled")
    source = truth.source_node_id
    sink = truth.sink_node_id
    shortcut_index = 1
    remaining_nodes = set(nodes)
    remaining_edges = dict(edges)
    safe_edges = set(known_condition_edges)

    for node_id in sorted(forgotten_nodes):
        if node_id not in remaining_nodes:
            continue
        if node_id in {source, sink}:
            raise _ProjectionRejected(
                "structural SOURCE/SINK was selected for forgetting"
            )
        incoming = [
            edge for edge in remaining_edges.values() if edge.to_node == node_id
        ]
        outgoing = [
            edge for edge in remaining_edges.values() if edge.from_node == node_id
        ]
        if len(incoming) != 1 or len(outgoing) != 1:
            raise _ProjectionRejected(
                f"node {node_id!r} is not an eligible serial node "
                f"(indegree={len(incoming)}, outdegree={len(outgoing)})"
            )
        incoming_edge = incoming[0]
        outgoing_edge = outgoing[0]
        predecessor = incoming_edge.from_node
        successor = outgoing_edge.to_node
        if predecessor == successor:
            raise _ProjectionRejected(
                f"contraction of {node_id!r} would create a self-loop"
            )
        if not _truth_edge_condition_is_safe(incoming_edge, safe_edges):
            raise _ProjectionRejected(
                f"contraction of {node_id!r} has unsafe incoming condition semantics"
            )
        if not _truth_edge_condition_is_safe(outgoing_edge, safe_edges):
            raise _ProjectionRejected(
                f"contraction of {node_id!r} has unsafe outgoing condition semantics"
            )
        if any(
            edge.id not in {incoming_edge.id, outgoing_edge.id}
            and edge.from_node == predecessor
            and edge.to_node == successor
            for edge in remaining_edges.values()
        ):
            raise _ProjectionRejected(
                f"contraction of {node_id!r} would create ambiguous parallel edges"
            )

        incoming_provenance = list(incoming_edge.derived_from_edges) or [
            incoming_edge.id
        ]
        outgoing_provenance = list(outgoing_edge.derived_from_edges) or [
            outgoing_edge.id
        ]
        contracted = (
            list(incoming_edge.contracted_nodes)
            + [node_id]
            + list(outgoing_edge.contracted_nodes)
        )
        new_id = f"__tau2_shortcut__{shortcut_index:03d}"
        while new_id in remaining_edges:
            shortcut_index += 1
            new_id = f"__tau2_shortcut__{shortcut_index:03d}"
        shortcut_index += 1
        structural_boundary = predecessor == source or successor == sink
        edge_type = "structural_boundary" if structural_boundary else "shortcut"
        new_edge = incoming_edge.__class__(
            id=new_id,
            from_node=predecessor,
            to_node=successor,
            condition=None,
            edge_kind=edge_type,
            structural_only=structural_boundary,
            protected=structural_boundary,
            is_shortcut=True,
            contracted_nodes=contracted,
            derived_from_edges=incoming_provenance + outgoing_provenance,
        )
        remaining_edges.pop(incoming_edge.id, None)
        remaining_edges.pop(outgoing_edge.id, None)
        remaining_edges[new_id] = new_edge
        safe_edges.discard(incoming_edge.id)
        safe_edges.discard(outgoing_edge.id)
        safe_edges.add(new_id)
        remaining_nodes.remove(node_id)

    return remaining_nodes, remaining_edges, safe_edges


def _project_knowledge_once(
    truth, stakeholder, rng: random.Random
) -> StakeholderKnowledge:
    config = stakeholder.forgetting
    truth_errors = canonical_structure_errors(truth)
    if truth_errors:
        raise ValueError(
            "Truth graph is not canonical:\n- " + "\n- ".join(truth_errors)
        )

    source = truth.source_node_id
    sink = truth.sink_node_id
    all_business_nodes = set(business_node_ids(truth))
    all_business_edges = set(business_edge_ids(truth))
    explicitly_visible_nodes = all_business_nodes & set(stakeholder.visible_node_ids)
    explicitly_visible_edges = all_business_edges & set(stakeholder.visible_edge_ids)

    forgotten_nodes = all_business_nodes - explicitly_visible_nodes
    node_probability = config.effective_node_probability
    for node_id in sorted(explicitly_visible_nodes):
        if rng.random() < node_probability:
            forgotten_nodes.add(node_id)
    forgotten_edges = set(all_business_edges - explicitly_visible_edges)
    edge_probability = config.effective_edge_probability
    for edge_id in sorted(explicitly_visible_edges):
        if rng.random() < edge_probability:
            forgotten_edges.add(edge_id)

    # The working graph includes forgotten business nodes until contraction so
    # their full known incident path can be checked.  Unknown edges are not
    # invented: they are simply absent and can make the sample reject.
    working_nodes = set(all_business_nodes) | {source, sink}
    working_edges = {
        edge_id: truth.edges[edge_id].model_copy(deep=True)
        for edge_id in set(truth.edges)
        - set(
            edge_id for edge_id, edge in truth.edges.items() if edge_is_structural(edge)
        )
        if edge_id in explicitly_visible_edges and edge_id not in forgotten_edges
    }
    for edge_id, edge in truth.edges.items():
        if edge_is_structural(edge):
            working_edges[edge_id] = edge.model_copy(deep=True)

    property_probability = config.property_forget_probability
    known_condition_edges = {
        edge_id
        for edge_id in explicitly_visible_edges - forgotten_edges
        if "condition" in stakeholder.edge_properties_for(edge_id)
    }
    known_condition_edges.update(
        edge_id for edge_id, edge in working_edges.items() if edge_is_structural(edge)
    )
    remaining_nodes, remaining_edges, known_condition_edges = _contract_forgotten_nodes(
        truth,
        working_nodes,
        working_edges,
        forgotten_nodes,
        known_condition_edges,
        allow_shortcut=config.allow_shortcut_contraction,
    )
    # Semantic property forgetting also applies to edge-condition slots.  It
    # happens after topology contraction so epistemic forgetting cannot make a
    # condition-free path unsafe to contract; the derived slot simply becomes
    # DONT_KNOW when the sample forgets it.
    for edge_id in sorted(known_condition_edges):
        edge = remaining_edges.get(edge_id)
        if edge is not None and not edge_is_structural(edge):
            if rng.random() < property_probability:
                known_condition_edges.discard(edge_id)
    final_business_nodes = remaining_nodes - {source, sink}
    if not final_business_nodes:
        raise _ProjectionRejected("forgetting removed every business node")

    # Semantic forgetting never removes topology.  It only converts known
    # property slots to DONT_KNOW; structural nodes bypass this sampling.
    known_node_props: dict[str, set[str]] = {}
    for node_id in sorted(final_business_nodes):
        props = stakeholder.node_properties_for(node_id)
        known_node_props[node_id] = {
            prop for prop in props if rng.random() >= property_probability
        }

    final_business_node_ids = sorted(final_business_nodes)
    final_edge_ids = sorted(remaining_edges)
    node_to_local = _local_ids(final_business_node_ids, "skn_")
    node_to_local[source] = STRUCTURAL_SOURCE_ID
    node_to_local[sink] = STRUCTURAL_SINK_ID
    edge_to_local = _local_ids(final_edge_ids, "ske_")

    referenced_tids: set[str] = set()

    def add_ref(ref) -> None:
        if isinstance(ref, ConceptRef):
            referenced_tids.add(ref.concept_id)

    for node_id in final_business_node_ids:
        truth_node = truth.nodes[node_id]
        props = known_node_props[node_id]
        for prop, attr in (
            ("activity", "activity"),
            ("actor", "actor"),
            ("system", "system"),
            ("rationale", "necessity_rationale"),
        ):
            if prop in props:
                add_ref(getattr(truth_node, attr))
        for axis in ("reads", "writes"):
            if axis in props:
                for ref in getattr(truth_node, axis) or []:
                    add_ref(ref)
    for edge_id, edge in remaining_edges.items():
        if edge_is_structural(edge) or edge.is_shortcut:
            continue
        if edge_id in known_condition_edges:
            add_ref(edge.condition)

    truth_to_local = _local_ids(sorted(referenced_tids), "skc_")

    def local_slot(ref, *, known: bool):
        if not known:
            return DONT_KNOW
        if ref is None:
            return None
        return ConceptRef(concept_id=truth_to_local[ref.concept_id])

    k_nodes: dict[str, StakeholderNode] = {
        STRUCTURAL_SOURCE_ID: StakeholderNode(
            id=STRUCTURAL_SOURCE_ID,
            structural=True,
            structural_role="source",
            protected=True,
        ),
        STRUCTURAL_SINK_ID: StakeholderNode(
            id=STRUCTURAL_SINK_ID,
            structural=True,
            structural_role="sink",
            protected=True,
        ),
    }
    for node_id in final_business_node_ids:
        truth_node = truth.nodes[node_id]
        props = known_node_props[node_id]
        k_nodes[node_to_local[node_id]] = StakeholderNode(
            id=node_to_local[node_id],
            activity=local_slot(truth_node.activity, known="activity" in props),
            actor=local_slot(truth_node.actor, known="actor" in props),
            system=local_slot(truth_node.system, known="system" in props),
            reads=(
                DONT_KNOW
                if "reads" not in props
                else (
                    None
                    if not isinstance(truth_node.reads, list) or not truth_node.reads
                    else [
                        ConceptRef(concept_id=truth_to_local[ref.concept_id])
                        for ref in truth_node.reads
                    ]
                )
            ),
            writes=(
                DONT_KNOW
                if "writes" not in props
                else (
                    None
                    if not isinstance(truth_node.writes, list) or not truth_node.writes
                    else [
                        ConceptRef(concept_id=truth_to_local[ref.concept_id])
                        for ref in truth_node.writes
                    ]
                )
            ),
            necessity_rationale=local_slot(
                truth_node.necessity_rationale,
                known="rationale" in props,
            ),
        )

    k_edges: dict[str, StakeholderEdge] = {}
    shortcut_provenance: dict[str, dict] = {}
    for edge_id in final_edge_ids:
        edge = remaining_edges[edge_id]
        local_id = edge_to_local[edge_id]
        if edge_is_structural(edge):
            # Boundary structure is unconditional by contract, even when the
            # incident business condition slot was not known.
            condition = None
        elif edge.is_shortcut:
            condition = None if edge_id in known_condition_edges else DONT_KNOW
        elif edge_id not in known_condition_edges:
            condition = DONT_KNOW
        elif edge.condition is None:
            condition = None
        else:
            condition = ConceptRef(concept_id=truth_to_local[edge.condition.concept_id])
        k_edges[local_id] = StakeholderEdge(
            id=local_id,
            from_node=node_to_local[edge.from_node],
            to_node=node_to_local[edge.to_node],
            condition=condition,
            edge_kind=edge.edge_kind,
            structural_only=edge.structural_only,
            protected=edge.protected,
            is_shortcut=edge.is_shortcut,
            contracted_nodes=list(edge.contracted_nodes),
            derived_from_edges=list(edge.derived_from_edges),
        )
        if edge.is_shortcut:
            shortcut_provenance[local_id] = {
                "is_shortcut": True,
                "contracted_nodes": list(edge.contracted_nodes),
                "derived_from_edges": list(edge.derived_from_edges),
            }

    concepts: dict[str, StakeholderKnowledgeConcept] = {}
    for truth_id, local_id in sorted(truth_to_local.items()):
        truth_concept = truth.concepts[truth_id]
        concepts[local_id] = StakeholderKnowledgeConcept(
            id=local_id,
            truth_concept_id=truth_id,
            kind=truth_concept.kind,
            description=stakeholder.concept_description_for(truth_id, truth_concept),
            terms=stakeholder.concept_terms_for(truth_id, truth_concept),
        )

    graph = StakeholderKnowledgeGraph(
        id=truth.id,
        name=truth.name,
        nodes=k_nodes,
        edges=k_edges,
        concepts=concepts,
        source_node_id=STRUCTURAL_SOURCE_ID,
        sink_node_id=STRUCTURAL_SINK_ID,
        start_node_id=(
            node_to_local[
                next(
                    iter(
                        sorted(
                            edge.to_node
                            for edge in remaining_edges.values()
                            if edge.from_node == source
                            and edge.to_node in final_business_nodes
                        )
                    )
                )
            ]
            if sum(
                edge.from_node == source and edge.to_node in final_business_nodes
                for edge in remaining_edges.values()
            )
            == 1
            else None
        ),
        end_node_ids=[
            node_to_local[node_id]
            for node_id in sorted(
                {
                    edge.from_node
                    for edge in remaining_edges.values()
                    if edge.to_node == sink and edge.from_node in final_business_nodes
                }
            )
        ],
        node_truth_ids={local: truth_id for truth_id, local in node_to_local.items()},
        edge_truth_ids={edge_to_local[edge_id]: edge_id for edge_id in final_edge_ids},
        shortcut_provenance=shortcut_provenance,
    )
    errors = graph.structure_errors()
    if errors:
        raise _ProjectionRejected(
            "post-forgetting validation failed: " + "; ".join(errors)
        )
    return StakeholderKnowledge(graph=graph)


def project_knowledge(
    truth,
    stakeholder,
    *,
    rng: random.Random | None = None,
    seed: int | None = None,
    max_retries: int | None = None,
) -> StakeholderKnowledge:
    """Project Truth into a valid stakeholder graph using bounded rejection sampling.

    Structural SOURCE/SINK nodes and every boundary edge are always installed
    as protected elements.  Semantic forgetting masks slots in place.  Whole
    node forgetting is allowed only through condition-free serial contraction;
    branch/merge nodes, conditioned paths, parallel-edge cases, and all invalid
    results are rejected rather than repaired.
    """
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    # Production projection accepts only the canonical Truth contract.  Raw
    # fixture migration belongs at the construction boundary
    # (``canonicalize_truth_graph``), never inside forgetting where it could
    # look like automatic graph repair.
    truth_errors = canonical_structure_errors(truth)
    if truth_errors:
        raise ValueError(
            "Truth graph is not canonical:\n- " + "\n- ".join(truth_errors)
        )
    generator = rng if rng is not None else random.Random(seed)
    config = stakeholder.forgetting
    attempts_limit = max_retries if max_retries is not None else config.max_retries
    if attempts_limit < 1:
        raise ValueError("max_retries must be positive")
    reasons: list[str] = []
    for _attempt in range(1, attempts_limit + 1):
        try:
            return _project_knowledge_once(truth, stakeholder, generator)
        except _ProjectionRejected as exc:
            reasons.append(str(exc))
    reason_summary = "; ".join(dict.fromkeys(reasons[-8:]))
    raise KnowledgeProjectionError(
        "unable to generate a valid stakeholder graph after "
        f"{attempts_limit} forgetting attempts; configuration="
        f"{config.model_dump()}; validation_failures={reason_summary}",
        attempts=attempts_limit,
        reasons=reasons,
        config=config.model_dump(),
    )
