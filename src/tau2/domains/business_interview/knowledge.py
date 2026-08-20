"""The stakeholder's world model: an explicit stakeholder-knowledge graph.

``StakeholderKnowledge`` is a **physical projection of the Truth graph** —
the stakeholder's own world model:

    StakeholderKnowledge
      ├─ StakeholderKnowledgeGraph     (masked nodes/edges, three-valued slots)
      └─ StakeholderKnowledgeConcept[] (local understanding of Truth concepts)

Masking rules (``project_knowledge``):

- does not know a node exists -> node removed (never a shortcut edge: if A->B
  and B->C but B is unknown, the knowledge graph has neither edge);
- does not know an edge exists -> edge removed;
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

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import (
    _NODE_PROPS,
    DONT_KNOW,
    ConceptKind,
    ConceptRef,
    DontKnowType,
    graph_semantic_ids,
    is_dont_know,
    is_unset,
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
    """One node of the stakeholder's world model: the node exists, and each
    property slot is three-valued (ConceptRef | None | DONT_KNOW)."""

    id: str
    activity: Slot = None
    actor: Slot = None
    system: Slot = None
    reads: ListSlot = None
    writes: ListSlot = None
    necessity_rationale: Slot = None


class StakeholderEdge(BaseModel):
    """One edge of the stakeholder's world model. No shortcut edges are ever
    created when nodes are removed."""

    id: str
    from_node: str
    to_node: str
    condition: Slot = None


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
    """The stakeholder's own world model: masked nodes/edges with
    three-valued property slots and the stakeholder's local concepts.

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
    start_node_id: Optional[str] = None
    end_node_ids: list[str] = Field(default_factory=list)
    node_truth_ids: dict[str, str] = Field(
        default_factory=dict, description="Private: local node id -> Truth node id."
    )
    edge_truth_ids: dict[str, str] = Field(
        default_factory=dict, description="Private: local edge id -> Truth edge id."
    )

    def semantic_ids(self) -> set[str]:
        """All addressable semantic IDs: graph elements (incl. DONT_KNOW
        slots) plus the stakeholder's knowledge-concept ids. Exactly the set
        of ids ``resolve`` accepts."""
        ids = graph_semantic_ids(self.nodes, self.edges)
        ids.update(self.concepts)
        return ids

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


def _local_ids(truth_ids: list[str], prefix: str) -> dict[str, str]:
    """Deterministic opaque local ids: sorted by Truth id, zero-padded
    (``skn_001`` ...). Invariant to collection reordering and never derived
    from Truth id meaning."""
    return {tid: f"{prefix}{i:03d}" for i, tid in enumerate(sorted(truth_ids), 1)}


def project_knowledge(
    truth,
    stakeholder,
) -> StakeholderKnowledge:
    """Build the stakeholder's world model from the Truth graph + filter.

    Deterministic masking:
    - nodes/edges the stakeholder does not know exist are removed (no
      shortcut edges);
    - known properties keep their values (mapped to knowledge concepts);
    - known-absent properties are None;
    - unknown properties of known elements are DONT_KNOW.

    Every element gets a stable, semantically opaque stakeholder-local id
    (``skn_001`` / ``ske_001`` / ``skc_001`` style, assigned in sorted Truth
    id order); the private Truth mappings are stored evaluator-only. Only the
    concepts referenced by the resulting graph become knowledge concepts.

    Local node/edge ids are allocated ONLY after visibility filtering (the
    visible sets, sorted), so the ids are contiguous (``skn_001``,
    ``skn_002``, ...) and never reveal hidden elements through gaps such as
    ``skn_001, skn_003``.
    """
    visible_nodes = set(stakeholder.visible_node_ids)
    # edges whose endpoints are both known survive (never shortcut edges);
    # they get their ids from the post-filter set
    surviving_edges = [
        eid
        for eid in stakeholder.visible_edge_ids
        if eid in truth.edges
        and truth.edges[eid].from_node in visible_nodes
        and truth.edges[eid].to_node in visible_nodes
    ]

    node_to_local = _local_ids(sorted(visible_nodes), "skn_")
    edge_to_local = _local_ids(sorted(surviving_edges), "ske_")

    # Truth concept ids referenced by any known property value of a visible
    # element — these are exactly the concepts that become knowledge
    # concepts (numbered deterministically, sorted by Truth id).
    referenced_tids: set[str] = set()

    def add_node_refs(nid: str) -> None:
        node = truth.nodes[nid]
        props = stakeholder.node_properties_for(nid)
        for prop, attr in (
            ("activity", "activity"),
            ("actor", "actor"),
            ("system", "system"),
            ("rationale", "necessity_rationale"),
        ):
            if prop in props:
                ref = getattr(node, attr)
                if isinstance(ref, ConceptRef):
                    referenced_tids.add(ref.concept_id)
        for axis in ("reads", "writes"):
            if axis in props:
                for ref in getattr(node, axis) or []:
                    referenced_tids.add(ref.concept_id)

    for nid in visible_nodes:
        add_node_refs(nid)
    for eid in surviving_edges:
        edge = truth.edges[eid]
        if "condition" in stakeholder.edge_properties_for(eid):
            if isinstance(edge.condition, ConceptRef):
                referenced_tids.add(edge.condition.concept_id)

    truth_to_local = _local_ids(sorted(referenced_tids), "skc_")

    k_nodes: dict[str, StakeholderNode] = {}
    for nid in visible_nodes:
        node = truth.nodes[nid]
        props = stakeholder.node_properties_for(nid)

        def slot(prop: str, refs, known_prop: bool):
            if not known_prop:
                return DONT_KNOW
            if refs is None or is_unset(refs):
                return None
            return ConceptRef(concept_id=truth_to_local[refs.concept_id])

        k_nodes[node_to_local[nid]] = StakeholderNode(
            id=node_to_local[nid],
            activity=slot("activity", node.activity, "activity" in props),
            actor=slot("actor", node.actor, "actor" in props),
            system=slot("system", node.system, "system" in props),
            reads=(
                DONT_KNOW
                if "reads" not in props
                else (
                    None
                    if not isinstance(node.reads, list) or not node.reads
                    else [
                        ConceptRef(concept_id=truth_to_local[r.concept_id])
                        for r in node.reads
                    ]
                )
            ),
            writes=(
                DONT_KNOW
                if "writes" not in props
                else (
                    None
                    if not isinstance(node.writes, list) or not node.writes
                    else [
                        ConceptRef(concept_id=truth_to_local[r.concept_id])
                        for r in node.writes
                    ]
                )
            ),
            necessity_rationale=slot(
                "rationale", node.necessity_rationale, "rationale" in props
            ),
        )

    k_edges: dict[str, StakeholderEdge] = {}
    for eid in surviving_edges:
        edge = truth.edges[eid]
        eprops = stakeholder.edge_properties_for(eid)
        k_edges[edge_to_local[eid]] = StakeholderEdge(
            id=edge_to_local[eid],
            from_node=node_to_local[edge.from_node],
            to_node=node_to_local[edge.to_node],
            condition=(
                DONT_KNOW
                if "condition" not in eprops
                else (
                    None
                    if is_unset(edge.condition) or edge.condition is None
                    else ConceptRef(
                        concept_id=truth_to_local[edge.condition.concept_id]
                    )
                )
            ),
        )

    # knowledge concepts: exactly the referenced ones, opaque ids
    concepts: dict[str, StakeholderKnowledgeConcept] = {}
    for tid, local in sorted(truth_to_local.items()):
        tconcept = truth.concepts[tid]
        concepts[local] = StakeholderKnowledgeConcept(
            id=local,
            truth_concept_id=tid,
            kind=tconcept.kind,
            description=stakeholder.concept_description_for(tid, tconcept),
            terms=stakeholder.concept_terms_for(tid, tconcept),
        )

    return StakeholderKnowledge(
        graph=StakeholderKnowledgeGraph(
            id=truth.id,
            name=truth.name,
            nodes=k_nodes,
            edges=k_edges,
            concepts=concepts,
            start_node_id=(
                node_to_local[truth.start_node_id]
                if truth.start_node_id is not None
                and truth.start_node_id in visible_nodes
                else None
            ),
            end_node_ids=[
                node_to_local[e] for e in truth.end_node_ids if e in visible_nodes
            ],
            node_truth_ids={v: k for k, v in node_to_local.items()},
            edge_truth_ids={v: k for k, v in edge_to_local.items()},
        )
    )
