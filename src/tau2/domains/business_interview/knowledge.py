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

Every addressable element keeps its **stable semantic ID** (the same scheme
as the Truth graph: ``node:<id>``, ``node:<id>:<prop>``,
``node:<id>:reads:<concept_id>``, ``edge:<id>``, ``edge:<id>:condition``);
``DONT_KNOW`` slots are addressable too (that is how "I don't know" speech is
anchored).

``StakeholderKnowledgeConcept`` is the stakeholder's local understanding of
one Truth concept; ``truth_concept_id`` is the private evaluator mapping and
is never exposed to the simulator or the Agent. Description and terminology
vary independently (term known/details unknown, details known/wrong local
term, details known/term unknown, both known).
"""

from typing import Optional, Union

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import (
    ConceptKind,
    ConceptRef,
    DONT_KNOW,
    DontKnowType,
    graph_semantic_ids,
    is_dont_know,
)

# A property slot: ConceptRef (value known) | None (known absent) |
# DONT_KNOW (element known, value unknown).
Slot = Optional[Union[ConceptRef, DontKnowType]]
ListSlot = Optional[Union[list[ConceptRef], DontKnowType]]


class StakeholderKnowledgeConcept(BaseModel):
    """The stakeholder's local understanding of one TruthConcept.

    ``id`` is stakeholder-local (never the hidden Truth id); the private
    ``truth_concept_id`` mapping is evaluator-only. ``description`` and
    ``terms`` are each ``str | list[str] | DONT_KNOW`` and vary
    independently. Descriptions describe the concept itself, never
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


class StakeholderKnowledgeGraph(BaseModel):
    """The stakeholder's own world model: masked nodes/edges with
    three-valued property slots and the stakeholder's local concepts.

    Element ids are the Truth node/edge ids (masking keeps ids); the
    semantic-ID scheme is identical to the Truth graph's, so spans resolve
    to the same element names in both worlds.
    """

    id: str = "knowledge"
    name: str = ""
    nodes: dict[str, StakeholderNode] = Field(default_factory=dict)
    edges: dict[str, StakeholderEdge] = Field(default_factory=dict)
    concepts: dict[str, StakeholderKnowledgeConcept] = Field(default_factory=dict)
    start_node_id: Optional[str] = None
    end_node_ids: list[str] = Field(default_factory=list)

    def semantic_ids(self) -> set[str]:
        """All addressable semantic IDs: graph elements (incl. DONT_KNOW
        slots) plus the stakeholder's knowledge-concept ids."""
        ids = graph_semantic_ids(self.nodes, self.edges)
        ids.update(self.concepts)
        return ids

    def referenced_concept_ids(self) -> set[str]:
        """Knowledge-concept ids referenced by any known property value."""
        ids: set[str] = set()
        for node in self.nodes.values():
            for slot in (
                node.activity,
                node.actor,
                node.system,
                node.necessity_rationale,
            ):
                if isinstance(slot, ConceptRef):
                    ids.add(slot.concept_id)
            for slot in (node.reads, node.writes):
                if isinstance(slot, list):
                    for ref in slot:
                        ids.add(ref.concept_id)
        for edge in self.edges.values():
            if isinstance(edge.condition, ConceptRef):
                ids.add(edge.condition.concept_id)
        return ids

    def element_value(self, semantic_id: str):
        """The three-valued value of a graph-element semantic id, or None when
        the id is not an element of this graph (unknown id)."""
        node = self._parse_node_slot(semantic_id)
        if node is not None:
            nid, prop = node
            n = self.nodes.get(nid)
            if n is None:
                return None
            if prop in ("reads", "writes"):
                value = getattr(n, prop)
                if isinstance(value, list):
                    return list(value)
                return value
            return getattr(n, prop, None)
        if semantic_id.startswith("edge:"):
            parts = semantic_id.split(":")
            if len(parts) == 2:
                e = self.edges.get(parts[1])
                return e if e is not None else None
            if len(parts) == 3 and parts[2] == "condition":
                e = self.edges.get(parts[1])
                return e.condition if e is not None else None
        return None

    def _parse_node_slot(self, semantic_id: str):
        if not semantic_id.startswith("node:"):
            return None
        parts = semantic_id.split(":")
        if len(parts) == 2:
            return (parts[1], "activity")  # node id itself -> activity slot
        if len(parts) == 3:
            return (parts[1], parts[2])
        if len(parts) == 4 and parts[2] in ("reads", "writes"):
            # element: node:<nid>:<axis>:<kcid>
            return (parts[1], parts[2])
        return None


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


def _mask_slot(known: bool, refs, concept_map: dict[str, str]) -> ListSlot:
    """Map one Truth property to its three-valued knowledge slot.

    ``known`` = the stakeholder knows this property; ``refs`` = the Truth
    value(s); ``concept_map`` = Truth concept id -> knowledge concept id.
    """
    if not known:
        return DONT_KNOW
    if not refs:
        return None
    return [ConceptRef(concept_id=concept_map[r.concept_id]) for r in refs]


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

    Only the concepts referenced by the resulting graph become knowledge
    concepts, with locally-mapped ids (``truth_concept_id`` stays private).
    """
    visible_nodes = set(stakeholder.visible_node_ids)
    visible_edges = set(stakeholder.visible_edge_ids)

    # local concept ids: one per referenced Truth concept (stable, local)
    truth_to_local: dict[str, str] = {}

    def local_concept_id(tid: str) -> str:
        if tid not in truth_to_local:
            truth_to_local[tid] = f"skc_{tid.removeprefix('tc_')}"
        return truth_to_local[tid]

    k_nodes: dict[str, StakeholderNode] = {}
    for nid, node in truth.nodes.items():
        if nid not in visible_nodes:
            continue
        props = stakeholder.node_properties_for(nid)

        def slot(prop: str, refs, known_prop: bool):
            if not known_prop:
                return DONT_KNOW
            if not refs:
                return None
            return ConceptRef(concept_id=local_concept_id(refs.concept_id))

        k_nodes[nid] = StakeholderNode(
            id=nid,
            activity=slot("activity", node.activity, "activity" in props),
            actor=slot("actor", node.actor, "actor" in props),
            system=slot("system", node.system, "system" in props),
            reads=(
                DONT_KNOW
                if "reads" not in props
                else (
                    None
                    if not node.reads
                    else [
                        ConceptRef(concept_id=local_concept_id(r.concept_id))
                        for r in node.reads
                    ]
                )
            ),
            writes=(
                DONT_KNOW
                if "writes" not in props
                else (
                    None
                    if not node.writes
                    else [
                        ConceptRef(concept_id=local_concept_id(r.concept_id))
                        for r in node.writes
                    ]
                )
            ),
            necessity_rationale=slot(
                "rationale", node.necessity_rationale, "rationale" in props
            ),
        )

    k_edges: dict[str, StakeholderEdge] = {}
    for eid, edge in truth.edges.items():
        if eid not in visible_edges:
            continue
        if edge.from_node not in k_nodes or edge.to_node not in k_nodes:
            continue  # endpoints unknown -> edge removed; never a shortcut
        eprops = stakeholder.edge_properties_for(eid)
        k_edges[eid] = StakeholderEdge(
            id=eid,
            from_node=edge.from_node,
            to_node=edge.to_node,
            condition=(
                DONT_KNOW
                if "condition" not in eprops
                else (
                    None
                    if edge.condition is None
                    else ConceptRef(
                        concept_id=local_concept_id(edge.condition.concept_id)
                    )
                )
            ),
        )

    # knowledge concepts: exactly the ones the masked graph references
    referenced: set[str] = set()
    for n in k_nodes.values():
        for slot_v in (n.activity, n.actor, n.system, n.necessity_rationale):
            if isinstance(slot_v, ConceptRef):
                referenced.add(slot_v.concept_id)
        for slot_v in (n.reads, n.writes):
            if isinstance(slot_v, list):
                for ref in slot_v:
                    referenced.add(ref.concept_id)
    for e in k_edges.values():
        if isinstance(e.condition, ConceptRef):
            referenced.add(e.condition.concept_id)
    concepts: dict[str, StakeholderKnowledgeConcept] = {}
    for tid, local in sorted(truth_to_local.items()):
        if local not in referenced:
            continue
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
            start_node_id=truth.start_node_id if truth.start_node_id in k_nodes else None,
            end_node_ids=[e for e in truth.end_node_ids if e in k_nodes],
        )
    )
