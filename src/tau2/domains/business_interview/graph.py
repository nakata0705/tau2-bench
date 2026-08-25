"""Graph-native semantic model for business_interview (v13).

The semantic model is the **graph itself**. Truth is a
``BusinessProcessGraph`` plus ``TruthConcept``\\ s; the agent builds its own
``AgentGraph`` plus ``AgentConcept``\\ s; the stakeholder's world model is a
``StakeholderKnowledgeGraph`` (knowledge.py) derived from Truth by masking.

Every addressable graph element carries a **stable semantic ID** (never a
list index — IDs survive reordering):

    node:<node_id>                     the node itself (existence, NOT activity)
    node:<node_id>:activity            scalar property slot (value/absent/unknown)
    node:<node_id>:actor
    node:<node_id>:system
    node:<node_id>:rationale
    node:<node_id>:reads               whole-property slot
    node:<node_id>:reads:<concept_id>  one reads element
    node:<node_id>:writes              whole-property slot
    node:<node_id>:writes:<concept_id> one writes element
    edge:<edge_id>                     the edge itself
    edge:<edge_id>:condition           the edge-condition slot

The stakeholder's element ids are **opaque and stakeholder-local**
(``skn_001`` / ``ske_001`` / ``skc_001`` style, assigned deterministically
from sorted Truth ids and never derived from Truth ids, labels, terms, or
positions); the private Truth mapping lives only in the evaluator-side
``StakeholderKnowledge`` structure.

**Truth and the Agent have DIFFERENT slot semantics.**
- Truth (``BusinessProcessGraph`` / ``TruthNode`` / ``TruthEdge``) is
  complete canonical data: every slot is **two-valued** ``ConceptRef | None``
  (``None`` = canonical absence). Truth has no UNSET / ABSENT / DONT_KNOW
  states.
- The stakeholder's world model (``StakeholderKnowledgeGraph``) is
  **three-valued**: ``ConceptRef`` (value known), ``None`` (value known
  absent), or ``DONT_KNOW`` (element known, value unknown).
- The AgentGraph is **four-state**: ``UNSET`` (not investigated / no
  conclusion — the default), ``ConceptRef`` (known value), ``ABSENT``
  (explicitly established absent) or ``DONT_KNOW`` (explicitly established
  unknowable). Each state is an Agent belief; optional ``EvidenceRef`` lists
  are diagnostic metadata only. If supplied, an EvidenceRef points to an
  existing Observation, but quote spans and private semantic-slot binding
  never decide whether the belief can be recorded or whether a Truth
  reconstruction is correct.

**Mention != evidence != validation.** ``AgentConcept.mentions`` are spans
the Agent interprets as referring to the concept; property references and
markers may carry optional ``EvidenceRef`` metadata. There is no validation
/ grounding lifecycle and no status gate: concept identity is scored by
content against Truth.

The evaluator scores AgentConcepts + AgentGraph directly against
TruthConcepts + TruthGraph. Private annotations, alignments and terminology
remain simulator/evidence diagnostics; none of the text fields are
interpreted as hidden provenance for reconstruction.
"""

from collections import deque
from typing import Any, Generic, Literal, Optional, Protocol, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

from tau2.environment.db import DB

C = TypeVar("C")

# Canonical Truth/Stakeholder boundary identifiers.  The explicit node and
# edge metadata below is authoritative; these constants only provide stable
# serialization anchors and collision detection.
STRUCTURAL_SOURCE_ID = "__tau2_structural_source__"
STRUCTURAL_SINK_ID = "__tau2_structural_sink__"
# Descriptive aliases for callers that prefer the node-oriented names.
STRUCTURAL_SOURCE_NODE_ID = STRUCTURAL_SOURCE_ID
STRUCTURAL_SINK_NODE_ID = STRUCTURAL_SINK_ID
SOURCE_NODE_ID = STRUCTURAL_SOURCE_ID
SINK_NODE_ID = STRUCTURAL_SINK_ID
STRUCTURAL_BOUNDARY_EDGE_PREFIX = "__tau2_structural_boundary__"

StructuralRole = Literal["source", "sink"]
TruthEdgeKind = Literal["business", "structural_boundary", "shortcut"]


class _NodeProto(Protocol):
    """The shared node surface of Truth (two-valued slots) and the Agent
    (four-state slots): both expose the same slot/ref helpers."""

    id: str
    activity: Any
    actor: Any
    system: Any
    reads: Any
    writes: Any
    necessity_rationale: Any

    def refs(self, property_name: str) -> list["ConceptRef"]: ...

    def asserted_refs(self, property_name: str) -> list["ConceptRef"]: ...

    def slot_value(self, property_name: str) -> Any: ...

    def slot_evidence(self, property_name: str) -> list["EvidenceRef"]: ...


class _EdgeProto(Protocol):
    """The shared edge surface of Truth and the Agent."""

    id: str
    from_node: str
    to_node: str
    condition: Any

    def condition_evidence(self) -> list["EvidenceRef"]: ...


N = TypeVar("N", bound=_NodeProto)
E = TypeVar("E", bound=_EdgeProto)

ConceptKind = Literal["activity", "actor", "system", "data", "condition", "rationale"]

# Property names scored on nodes / edges.
NodeProperty = Literal["activity", "actor", "system", "reads", "writes", "rationale"]
EdgeProperty = Literal["condition"]

_NODE_PROPS: tuple[str, ...] = (
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "rationale",
)


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    """A lightweight, optional citation of one immutable Observation.

    ``observation_id`` is the primary reference (a debuggable conversation
    pointer). ``quote`` and ``occurrence`` are **optional diagnostic hints**
    only --- exact quote matching is never required for graph/concept
    reconstruction, so the tools never fail merely because a quote is missing
    or ambiguous. When a quote is provided it should be an exact substring
    occurrence of the Observation text, but this is advisory.
    """

    observation_id: str
    quote: Optional[str] = Field(
        default=None,
        description="Optional diagnostic hint: a substring of the Observation text.",
    )
    occurrence: int = Field(
        default=0,
        description="0-based occurrence index of ``quote`` (diagnostic only).",
    )

    def resolve_span(self, text: str) -> Optional[tuple[int, int]]:
        """Resolve to (start, end) character offsets in ``text``, or None when
        no quote is present or the quote/occurrence does not match ``text``."""
        if not self.quote:
            return None
        start = -1
        for _ in range(self.occurrence + 1):
            start = text.find(self.quote, start + 1)
            if start == -1:
                return None
        return (start, start + len(self.quote))


def spans_correspond(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when two spans correspond: one contains the other (or they are
    equal). Deterministic character-span relation — no semantic matching."""
    return (a[0] <= b[0] and b[1] <= a[1]) or (b[0] <= a[0] and a[1] <= b[1])


class UnsetType(BaseModel):
    """The UNSET marker: this Agent slot has NOT been investigated yet —
    no conclusion exists.

    ``UNSET`` (module singleton) is the default for every new Agent
    property. It is distinct from ``ABSENT`` (explicitly established
    absent), from ``DONT_KNOW`` (explicitly established unknowable) and
    from ``ConceptRef`` (known value). Reset/unset means UNSET, never
    ABSENT. It carries no evidence and never scores as a known absence.
    """

    @property
    def asserted(self) -> bool:
        """UNSET carries no conclusion — never an active claim."""
        return False

    pass


UNSET = UnsetType()


def is_unset(value) -> bool:
    """True when an Agent slot carries the UNSET marker."""
    return isinstance(value, UnsetType)


class AbsentType(BaseModel):
    """The ABSENT marker: the Agent explicitly established that the value
    is ABSENT at this slot.

    ``ABSENT`` is distinct from ``UNSET`` (not investigated), from
    ``DONT_KNOW`` (unknowable) and from ``ConceptRef`` (known value). An
    optional Observation evidence list may explain the Agent's belief, but
    it is diagnostic only and is not validated against a private stakeholder
    known-absent slot for reconstruction.
    """

    evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def asserted(self) -> bool:
        """An ABSENT marker IS an active epistemic claim."""
        return True


def is_absent(value) -> bool:
    """True when an Agent slot carries the ABSENT marker."""
    return isinstance(value, AbsentType)


class DontKnowType(BaseModel):
    """The DONT_KNOW marker: the Agent explicitly established that the
    value is unknowable from this stakeholder.

    ``DONT_KNOW`` (module singleton) is distinct from ``UNSET`` (not
    investigated), from ``ABSENT`` (explicitly established absent) and
    from ``ConceptRef`` (known value). An optional Observation evidence list
    may explain the Agent's belief, but it is diagnostic only and is not
    validated against a private stakeholder DONT_KNOW slot for
    reconstruction.
    """

    evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def asserted(self) -> bool:
        """A DONT_KNOW marker IS an active epistemic claim ("I cannot
        determine this") — distinct from an unset slot."""
        return True


DONT_KNOW = DontKnowType()


def is_dont_know(value) -> bool:
    """True when a property slot carries the DONT_KNOW marker."""
    return isinstance(value, DontKnowType)


# ---------------------------------------------------------------------------
# Semantic IDs
# ---------------------------------------------------------------------------


def node_id(nid: str) -> str:
    return f"node:{nid}"


def slot_id(nid: str, prop: str) -> str:
    return f"node:{nid}:{prop}"


def element_id(nid: str, axis: str, concept_id: str) -> str:
    return f"node:{nid}:{axis}:{concept_id}"


def edge_id(eid: str) -> str:
    return f"edge:{eid}"


def condition_id(eid: str) -> str:
    return f"edge:{eid}:condition"


def graph_semantic_ids(nodes, edges) -> set[str]:
    """All addressable semantic IDs of a graph (Truth, Agent or stakeholder
    knowledge): node ids, every property slot, every reads/writes element,
    edge ids and the condition slot. Stable — never derived from list
    positions."""
    ids: set[str] = set()
    for nid in nodes:
        ids.add(node_id(nid))
        for prop in _NODE_PROPS:
            ids.add(slot_id(nid, prop))
        node = nodes[nid]
        for axis in ("reads", "writes"):
            refs = getattr(node, axis, None)
            if not isinstance(refs, list):
                continue  # None (known absent) / DONT_KNOW (unknown)
            for ref in refs:
                if ref.concept_id:
                    ids.add(element_id(nid, axis, ref.concept_id))
    for eid in edges:
        ids.add(edge_id(eid))
        ids.add(condition_id(eid))
    return ids


class ConceptRef(BaseModel):
    """A reference from a node/edge to a concept (Truth, Agent or stakeholder
    knowledge — the id namespace depends on the graph).

    ``evidence`` optionally cites Observation spans that explain using this
    concept at this slot; it is diagnostic metadata. ``confidence`` in [0, 1];
    0 = unasserted.
    """

    concept_id: str
    confidence: float = Field(default=1.0, description="Confidence in [0, 1].")
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    @property
    def asserted(self) -> bool:
        """True if this reference is an active claim (confidence > 0)."""
        return self.confidence > 0


# The Agent's four epistemic states per slot.
AgentSlot = Union[ConceptRef, UnsetType, AbsentType, DontKnowType]
AgentListSlot = Union[list[ConceptRef], UnsetType, AbsentType, DontKnowType]


class Node(BaseModel):
    """A vertex in the Agent's inferred business process graph.

    Every property slot is one of the FOUR epistemic states:
    ``UNSET`` (not investigated — the default), ``ConceptRef`` (known
    value), ``ABSENT`` (explicitly established absent) or ``DONT_KNOW``
    (explicitly established unknowable). Optional evidence on refs/markers is
    diagnostic metadata only.
    ``reads`` / ``writes`` are lists of data-concept references, or the
    whole-property markers UNSET / ABSENT / DONT_KNOW. Every ref and marker
    may carry optional diagnostic evidence.
    ``from``/``to`` structural identity is expressed only through edges;
    node ids are local to the graph.
    """

    id: str
    activity: AgentSlot = Field(default_factory=lambda: UNSET)
    actor: AgentSlot = Field(default_factory=lambda: UNSET)
    system: AgentSlot = Field(default_factory=lambda: UNSET)
    reads: AgentListSlot = Field(default_factory=lambda: UNSET)
    writes: AgentListSlot = Field(default_factory=lambda: UNSET)
    necessity_rationale: AgentSlot = Field(default_factory=lambda: UNSET)

    def refs(self, property_name: str) -> list[ConceptRef]:
        """The concept refs of a node property (single or list).

        ``property_name`` accepts the claim-style names (activity/actor/system/
        reads/writes/rationale) as well as the attribute name
        ``necessity_rationale``. UNSET / ABSENT / DONT_KNOW slots contribute
        no refs.
        """
        if property_name == "reads":
            return list(self.reads) if isinstance(self.reads, list) else []
        if property_name == "writes":
            return list(self.writes) if isinstance(self.writes, list) else []
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        ref = getattr(self, attr)
        return [ref] if isinstance(ref, ConceptRef) else []

    def asserted_refs(self, property_name: str) -> list[ConceptRef]:
        return [r for r in self.refs(property_name) if r.asserted]

    def slot_value(self, property_name: str):
        """The four-state property slot: ConceptRef / UNSET / ABSENT /
        DONT_KNOW for scalars; list[ConceptRef] / UNSET / ABSENT /
        DONT_KNOW for reads/writes."""
        if property_name in ("reads", "writes"):
            return getattr(self, property_name)
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        return getattr(self, attr)

    def slot_evidence(self, property_name: str) -> list[EvidenceRef]:
        """The evidence of an ABSENT/DONT_KNOW marker on this slot (empty
        for ConceptRef / UNSET / list slots)."""
        slot = self.slot_value(property_name)
        if isinstance(slot, (AbsentType, DontKnowType)):
            return list(slot.evidence)
        return []


class Edge(BaseModel):
    """A directed edge between two nodes.

    ``from_node`` / ``to_node`` are structural identities (node ids), not
    concepts. ``condition`` is one of the four epistemic states: ``UNSET``
    (default), ``ConceptRef`` (a condition concept, kind=condition),
    ``ABSENT`` (explicitly established unconditional) or ``DONT_KNOW``
    (unknowable). ``evidence`` optionally cites Observation spans for
    diagnostics; it is not required to support the relation.
    """

    id: str
    from_node: str
    to_node: str
    condition: AgentSlot = Field(default_factory=lambda: UNSET)
    evidence: list[EvidenceRef] = Field(default_factory=list)

    def condition_evidence(self) -> list[EvidenceRef]:
        """The evidence of an ABSENT/DONT_KNOW condition marker."""
        if isinstance(self.condition, (AbsentType, DontKnowType)):
            return list(self.condition.evidence)
        return []


def _node_property_refs(node) -> dict[str, list[ConceptRef]]:
    """All refs of a node keyed by property name (single refs as one-element
    lists; reads/writes as their lists). DONT_KNOW slots contribute no
    refs."""
    return {
        "activity": node.refs("activity"),
        "actor": node.refs("actor"),
        "system": node.refs("system"),
        "reads": node.refs("reads"),
        "writes": node.refs("writes"),
        "necessity_rationale": node.refs("rationale"),
    }


class _GraphMixin(BaseModel, Generic[C]):
    """Shared graph structure plus utilities (cycles are valid; no acyclicity
    requirement). ``nodes``/``edges`` hold the graph's concrete node/edge
    types, fixed by the subclass: Truth uses ``TruthNode``/``TruthEdge``
    (two-valued slots: ``ConceptRef | None``), the Agent uses ``Node``/
    ``Edge`` (four-state slots)."""

    id: str = "graph"
    name: str = ""
    nodes: dict[str, Any] = Field(default_factory=dict)
    edges: dict[str, Any] = Field(default_factory=dict)
    concepts: dict[str, C] = Field(default_factory=dict)
    start_node_id: Optional[str] = None
    end_node_ids: list[str] = Field(default_factory=list)

    def successors(self, node_id: str) -> list[str]:
        return [e.to_node for e in self.edges.values() if e.from_node == node_id]

    def incoming_edges(self, node_id: str) -> list[str]:
        return [e.id for e in self.edges.values() if e.to_node == node_id]

    def structure_errors(self) -> list[str]:
        """Internal self-consistency errors (empty = well-formed).

        Checks references, required fields and declared endpoints — cycles are
        valid and never reported as errors. Nothing here references hidden
        Ground Truth.
        """
        errors: list[str] = []
        for nid, node in self.nodes.items():
            if is_unset(node.activity):
                errors.append(f"node {nid}: activity is unset (no conclusion)")
            elif not isinstance(node.activity, ConceptRef):
                pass  # ABSENT / DONT_KNOW — an explicit epistemic state
            elif not node.activity.concept_id:
                errors.append(f"node {nid}: activity reference is required")
            for prop, refs in _node_property_refs(node).items():
                for ref in refs:
                    if ref.concept_id not in self.concepts:
                        errors.append(
                            f"node {nid}: {prop} references unknown concept "
                            f"{ref.concept_id!r}"
                        )
        for eid, edge in self.edges.items():
            if edge.from_node not in self.nodes:
                errors.append(f"edge {eid}: from_node not found: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"edge {eid}: to_node not found: {edge.to_node}")
            if (
                isinstance(edge.condition, ConceptRef)
                and edge.condition.concept_id not in self.concepts
            ):
                errors.append(
                    f"edge {eid}: condition references unknown concept "
                    f"{edge.condition.concept_id!r}"
                )
        if self.start_node_id is not None and self.start_node_id not in self.nodes:
            errors.append(f"start node not found: {self.start_node_id}")
        for sid in getattr(self, "start_node_ids", []):
            if sid not in self.nodes:
                errors.append(f"start node not found: {sid}")
        for eid in self.end_node_ids:
            if eid not in self.nodes:
                errors.append(f"end node not found: {eid}")
        return errors

    @property
    def is_valid(self) -> bool:
        return not self.structure_errors()

    def referenced_concepts(self) -> set[str]:
        """Ids of every concept referenced by any asserted node/edge ref."""
        ids: set[str] = set()
        for node in self.nodes.values():
            for refs in _node_property_refs(node).values():
                for ref in refs:
                    if ref.asserted:
                        ids.add(ref.concept_id)
        for edge in self.edges.values():
            if isinstance(edge.condition, ConceptRef) and edge.condition.asserted:
                ids.add(edge.condition.concept_id)
        return ids


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------


class TruthConcept(BaseModel):
    """One Truth concept: a business thing of one kind.

    ``description`` describes the concept itself (never a workflow-position
    fact); ``canonical_terms`` are the Truth's own wordings (private — the
    stakeholder sees only its own knowledge concepts).
    """

    id: str
    kind: ConceptKind
    description: str = Field(default="")
    canonical_terms: list[str] = Field(default_factory=list)


class TruthNode(BaseModel):
    """A vertex in the COMPLETE canonical Truth graph.

    Truth is complete canonical data and has no ``UNSET``/``ABSENT``/
    ``DONT_KNOW`` states: every scalar property slot holds either a
    ``ConceptRef`` (a known value) or ``None`` (canonical absence).
    ``reads`` / ``writes`` hold ``list[ConceptRef] | None`` where ``None``
    means canonical absence. ``from``/``to`` structural identity is
    expressed only through edges; node ids are Truth-local.
    """

    id: str
    activity: Optional[ConceptRef] = None
    actor: Optional[ConceptRef] = None
    system: Optional[ConceptRef] = None
    reads: Optional[list[ConceptRef]] = None
    writes: Optional[list[ConceptRef]] = None
    necessity_rationale: Optional[ConceptRef] = None
    # Explicit structural typing.  A structural node has no business
    # semantics; its role is part of the graph contract, not inferred from an
    # id or from topology alone.
    structural: bool = False
    structural_role: Optional[StructuralRole] = None
    protected: bool = False

    @property
    def is_structural(self) -> bool:
        """Whether this is the explicit canonical SOURCE or SINK node."""
        return self.structural or self.structural_role is not None

    def refs(self, property_name: str) -> list[ConceptRef]:
        """The concept refs of a Truth property slot (single or list)."""
        if property_name == "reads":
            return list(self.reads) if isinstance(self.reads, list) else []
        if property_name == "writes":
            return list(self.writes) if isinstance(self.writes, list) else []
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        ref = getattr(self, attr)
        return [ref] if isinstance(ref, ConceptRef) else []

    def asserted_refs(self, property_name: str) -> list[ConceptRef]:
        return [r for r in self.refs(property_name) if r.asserted]

    def slot_value(self, property_name: str):
        """The canonical two-valued slot: ``ConceptRef | None`` (None =
        canonical absence)."""
        if property_name in ("reads", "writes"):
            return getattr(self, property_name)
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        return getattr(self, attr)

    def slot_evidence(self, property_name: str) -> list[EvidenceRef]:
        """Truth carries no evidence — always empty."""
        return []


class TruthEdge(BaseModel):
    """A directed edge in the complete canonical Truth graph.

    ``condition`` is ``ConceptRef | None`` (``None`` = canonically
    unconditional). Truth carries no ``UNSET``/``ABSENT``/``DONT_KNOW``
    states; ``from_node``/``to_node`` are structural identities.
    """

    id: str
    from_node: str
    to_node: str
    condition: Optional[ConceptRef] = None
    # ``structural_boundary`` edges are SOURCE/entry and exit/SINK edges.
    # ``shortcut`` edges are evaluator-private derived relations produced by
    # safe stakeholder contraction; neither is a business semantic edge.
    edge_kind: TruthEdgeKind = "business"
    structural_only: bool = False
    protected: bool = False
    is_shortcut: bool = False
    contracted_nodes: list[str] = Field(default_factory=list)
    derived_from_edges: list[str] = Field(default_factory=list)

    @property
    def is_structural(self) -> bool:
        """Whether the edge is excluded from ordinary business scoring."""
        return self.structural_only or self.edge_kind == "structural_boundary"

    def condition_evidence(self) -> list[EvidenceRef]:
        """Truth edges carry no evidence; always empty."""
        return []


class BusinessProcessGraph(_GraphMixin[TruthConcept]):
    """The canonical Truth graph.

    A production Truth graph contains exactly one explicitly typed SOURCE and
    exactly one explicitly typed SINK.  Business entry/exit nodes are ordinary
    nodes connected to those boundary nodes by ``structural_boundary`` edges.
    The boundary nodes/edges are structural-only and must not enter business
    node/edge scoring denominators.

    ``start_node_id`` / ``end_node_ids`` remain on the shared graph surface for
    the Agent reconstruction model and old diagnostic readers, but they are
    not Truth semantics.  ``source_node_id`` / ``sink_node_id`` and the typed
    node/edge metadata below are authoritative for Truth.
    """

    nodes: dict[str, TruthNode] = Field(default_factory=dict)
    edges: dict[str, TruthEdge] = Field(default_factory=dict)
    source_node_id: str = STRUCTURAL_SOURCE_ID
    sink_node_id: str = STRUCTURAL_SINK_ID

    def structure_errors(self) -> list[str]:
        """Return internal and canonical Truth invariant errors."""
        return super().structure_errors() + canonical_structure_errors(self)

    @property
    def business_entry_node_ids(self) -> tuple[str, ...]:
        return business_entry_node_ids(self)

    @property
    def business_exit_node_ids(self) -> tuple[str, ...]:
        return business_exit_node_ids(self)

    @property
    def business_node_ids(self) -> tuple[str, ...]:
        return tuple(business_node_ids(self))

    @property
    def business_edge_ids(self) -> tuple[str, ...]:
        return tuple(business_edge_ids(self))


# Public name used by the canonical contract documentation; the historical
# implementation name remains available for domain callers.
TruthGraph = BusinessProcessGraph


# ---------------------------------------------------------------------------
# Canonical graph contract
# ---------------------------------------------------------------------------


def node_is_structural(node: Any) -> bool:
    """Return whether a node carries explicit structural metadata."""
    return bool(
        getattr(node, "structural", False)
        or getattr(node, "structural_role", None) is not None
        or getattr(node, "is_structural", False)
    )


def edge_is_structural(edge: Any) -> bool:
    """Return whether an edge is structural-only, never a business relation."""
    return bool(
        getattr(edge, "structural_only", False)
        or getattr(edge, "edge_kind", None) == "structural_boundary"
        or getattr(edge, "is_structural", False)
    )


def _graph_node_ids(graph: Any) -> set[str]:
    return set(getattr(graph, "nodes", {}))


def _valid_adjacency(
    graph: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[str]]:
    """Build directed adjacency while reporting dangling edge ids."""
    nodes = _graph_node_ids(graph)
    adjacency = {node_id: set() for node_id in nodes}
    reverse = {node_id: set() for node_id in nodes}
    dangling: list[str] = []
    for edge_id, edge in getattr(graph, "edges", {}).items():
        if edge.from_node not in nodes or edge.to_node not in nodes:
            dangling.append(edge_id)
            continue
        adjacency[edge.from_node].add(edge.to_node)
        reverse[edge.to_node].add(edge.from_node)
    return adjacency, reverse, sorted(dangling)


def _reachable(starts: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    reached = set(starts)
    queue = deque(sorted(reached))
    while queue:
        node_id = queue.popleft()
        for successor in sorted(adjacency.get(node_id, ())):
            if successor not in reached:
                reached.add(successor)
                queue.append(successor)
    return reached


def canonical_structure_errors(graph: Any) -> list[str]:
    """Validate the explicit single-SOURCE/single-SINK graph contract.

    The validator is deliberately independent of insertion order and of
    business node/edge ids.  It rejects dangling/disconnected topology,
    topology-derived extra sources/sinks, unprotected boundary elements, and
    structural elements carrying business semantics.  It is usable for both
    ``BusinessProcessGraph`` and ``StakeholderKnowledgeGraph``.
    """
    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", {})
    source_id = getattr(graph, "source_node_id", None)
    sink_id = getattr(graph, "sink_node_id", None)
    errors: list[str] = []
    if not nodes:
        return ["canonical graph must contain at least one business node"]
    if source_id is None or sink_id is None:
        return ["canonical graph must declare source_node_id and sink_node_id"]
    if source_id == sink_id:
        errors.append("SOURCE and SINK must be distinct")

    source_roles = sorted(
        node_id
        for node_id, node in nodes.items()
        if getattr(node, "structural_role", None) == "source"
    )
    sink_roles = sorted(
        node_id
        for node_id, node in nodes.items()
        if getattr(node, "structural_role", None) == "sink"
    )
    if source_roles != [source_id]:
        errors.append(
            f"expected exactly one structural SOURCE {source_id!r}; "
            f"found role nodes {source_roles!r}"
        )
    if sink_roles != [sink_id]:
        errors.append(
            f"expected exactly one structural SINK {sink_id!r}; "
            f"found role nodes {sink_roles!r}"
        )
    if source_id not in nodes:
        errors.append(f"structural SOURCE node not found: {source_id}")
    if sink_id not in nodes:
        errors.append(f"structural SINK node not found: {sink_id}")

    for node_id, node in nodes.items():
        if node_is_structural(node):
            if node_id not in {source_id, sink_id}:
                errors.append(f"unexpected structural node: {node_id}")
            if not getattr(node, "protected", False):
                errors.append(f"structural node is not protected: {node_id}")
            # Structural nodes cannot smuggle business facts into the Truth.
            for prop in (
                "activity",
                "actor",
                "system",
                "reads",
                "writes",
                "necessity_rationale",
            ):
                value = getattr(node, prop, None)
                if value not in (None, [], ()):
                    errors.append(
                        f"structural node {node_id}: semantic property {prop} is set"
                    )
        elif getattr(node, "structural_role", None) is not None:
            errors.append(f"business node {node_id} has a structural role")

    adjacency, reverse, dangling = _valid_adjacency(graph)
    if dangling:
        errors.append(f"dangling edges: {dangling!r}")
    incoming = {node_id: len(reverse[node_id]) for node_id in nodes}
    outgoing = {node_id: len(adjacency[node_id]) for node_id in nodes}
    topology_sources = sorted(
        node_id for node_id, degree in incoming.items() if degree == 0
    )
    topology_sinks = sorted(
        node_id for node_id, degree in outgoing.items() if degree == 0
    )
    if topology_sources != [source_id]:
        errors.append(
            f"topology sources must be exactly [SOURCE]; found {topology_sources!r}"
        )
    if topology_sinks != [sink_id]:
        errors.append(
            f"topology sinks must be exactly [SINK]; found {topology_sinks!r}"
        )
    if source_id in incoming and incoming[source_id] != 0:
        errors.append("SOURCE must have indegree 0")
    if sink_id in outgoing and outgoing[sink_id] != 0:
        errors.append("SINK must have outdegree 0")

    business_nodes = set(nodes) - {source_id, sink_id}
    if not business_nodes:
        errors.append("canonical graph must contain at least one business node")
    reachable = _reachable({source_id} if source_id in nodes else set(), adjacency)
    can_reach_sink = _reachable({sink_id} if sink_id in nodes else set(), reverse)
    missing_from_source = sorted(business_nodes - reachable)
    missing_sink_path = sorted(business_nodes - can_reach_sink)
    if missing_from_source:
        errors.append(f"business nodes not SOURCE-reachable: {missing_from_source!r}")
    if missing_sink_path:
        errors.append(f"business nodes cannot reach SINK: {missing_sink_path!r}")
    isolated = sorted(
        node_id
        for node_id in business_nodes
        if incoming.get(node_id, 0) == 0 and outgoing.get(node_id, 0) == 0
    )
    if isolated:
        errors.append(f"isolated business nodes: {isolated!r}")

    structural_edge_ids: set[str] = set()
    for edge_id, edge in edges.items():
        structural = edge_is_structural(edge)
        if structural:
            structural_edge_ids.add(edge_id)
            if not getattr(edge, "protected", False):
                errors.append(f"structural edge is not protected: {edge_id}")
            if getattr(edge, "condition", None) is not None:
                errors.append(f"structural edge must be unconditional: {edge_id}")
            valid_boundary = (
                edge.from_node == source_id and edge.to_node in business_nodes
            ) or (edge.to_node == sink_id and edge.from_node in business_nodes)
            if not valid_boundary:
                errors.append(
                    f"structural edge is not a SOURCE/entry or exit/SINK boundary: {edge_id}"
                )
        elif edge.from_node in {source_id, sink_id} or edge.to_node in {
            source_id,
            sink_id,
        }:
            errors.append(f"non-structural edge touches SOURCE/SINK: {edge_id}")
    if source_id in nodes and not any(
        edge_id in structural_edge_ids and edges[edge_id].from_node == source_id
        for edge_id in edges
    ):
        errors.append("SOURCE must have at least one protected boundary edge")
    if sink_id in nodes and not any(
        edge_id in structural_edge_ids and edges[edge_id].to_node == sink_id
        for edge_id in edges
    ):
        errors.append("SINK must have at least one protected boundary edge")
    return errors


def validate_canonical_graph(graph: Any) -> None:
    """Raise ``ValueError`` unless ``graph`` satisfies the canonical contract."""
    errors = canonical_structure_errors(graph)
    if errors:
        raise ValueError("Invalid canonical graph:\n- " + "\n- ".join(errors))


def business_node_ids(graph: Any) -> list[str]:
    """Return business node ids, excluding explicit structural nodes."""
    return sorted(
        node_id
        for node_id, node in getattr(graph, "nodes", {}).items()
        if not node_is_structural(node)
    )


def business_edge_ids(graph: Any) -> list[str]:
    """Return business edge ids, excluding structural-only boundary edges."""
    return sorted(
        edge_id
        for edge_id, edge in getattr(graph, "edges", {}).items()
        if not edge_is_structural(edge)
    )


def business_entry_node_ids(graph: Any) -> tuple[str, ...]:
    """Return canonical business entries (SOURCE's direct successors)."""
    source_id = getattr(graph, "source_node_id", None)
    if source_id in getattr(graph, "nodes", {}):
        return tuple(
            sorted(
                {
                    edge.to_node
                    for edge in getattr(graph, "edges", {}).values()
                    if edge.from_node == source_id
                    and edge.to_node in business_node_ids(graph)
                }
            )
        )
    legacy = getattr(graph, "start_node_id", None)
    return (legacy,) if legacy is not None else tuple()


def business_exit_node_ids(graph: Any) -> tuple[str, ...]:
    """Return canonical business exits (SINK's direct predecessors)."""
    sink_id = getattr(graph, "sink_node_id", None)
    if sink_id in getattr(graph, "nodes", {}):
        return tuple(
            sorted(
                {
                    edge.from_node
                    for edge in getattr(graph, "edges", {}).values()
                    if edge.to_node == sink_id
                    and edge.from_node in business_node_ids(graph)
                }
            )
        )
    return tuple(sorted(set(getattr(graph, "end_node_ids", []))))


def business_graph_projection(graph: Any) -> Any:
    """Copy a canonical graph with structural elements removed for scoring.

    The projection intentionally carries no structural node/edge denominator;
    its legacy endpoint fields are populated only as a diagnostic business
    entry/exit view.  The input graph is never mutated.
    """
    if not getattr(graph, "source_node_id", None):
        return graph
    projected = graph.model_copy(deep=True)
    projected.nodes = {
        node_id: node
        for node_id, node in graph.nodes.items()
        if not node_is_structural(node)
    }
    projected.edges = {
        edge_id: edge
        for edge_id, edge in graph.edges.items()
        if not edge_is_structural(edge)
    }
    entries = business_entry_node_ids(graph)
    exits = business_exit_node_ids(graph)
    projected.start_node_id = entries[0] if len(entries) == 1 else None
    projected.end_node_ids = list(exits)
    return projected


def _boundary_edge_id(side: str, ordinal: int) -> str:
    return f"{STRUCTURAL_BOUNDARY_EDGE_PREFIX}{side}_{ordinal:03d}"


def canonicalize_truth_graph(
    graph: BusinessProcessGraph,
    *,
    entry_node_ids: Optional[list[str]] = None,
    exit_node_ids: Optional[list[str]] = None,
) -> BusinessProcessGraph:
    """Return a canonical copy with explicit SOURCE/SINK boundaries.

    This helper is used for fixture migration and deterministic scenario
    construction.  It never invents business relations: it only adds typed,
    protected, unconditional boundary edges.  Entry/exit ids are sorted before
    boundary edge allocation, so insertion order cannot affect the result.
    """
    has_structural_metadata = any(
        node_is_structural(node) for node in graph.nodes.values()
    ) or any(edge_is_structural(edge) for edge in graph.edges.values())
    if has_structural_metadata:
        # A graph that already advertises structural elements is either
        # canonical or invalid; never silently repair it while canonicalizing.
        validate_canonical_graph(graph)
    business_nodes = {
        node_id: node
        for node_id, node in graph.nodes.items()
        if not node_is_structural(node)
    }
    business_edges = {
        edge_id: edge
        for edge_id, edge in graph.edges.items()
        if not edge_is_structural(edge)
    }
    node_ids = set(business_nodes)
    if entry_node_ids is None:
        if has_structural_metadata:
            entry_node_ids = list(business_entry_node_ids(graph))
        else:
            legacy_start = getattr(graph, "start_node_id", None)
            entry_node_ids = [legacy_start] if legacy_start in node_ids else None
    if exit_node_ids is None:
        if has_structural_metadata:
            exit_node_ids = list(business_exit_node_ids(graph))
        else:
            legacy_ends = [
                node_id
                for node_id in getattr(graph, "end_node_ids", [])
                if node_id in node_ids
            ]
            exit_node_ids = legacy_ends or None
    adjacency = {node_id: set() for node_id in node_ids}
    reverse = {node_id: set() for node_id in node_ids}
    for edge in business_edges.values():
        if edge.from_node in node_ids and edge.to_node in node_ids:
            adjacency[edge.from_node].add(edge.to_node)
            reverse[edge.to_node].add(edge.from_node)
    if entry_node_ids is None:
        entry_node_ids = sorted(node_id for node_id in node_ids if not reverse[node_id])
    if exit_node_ids is None:
        exit_node_ids = sorted(
            node_id for node_id in node_ids if not adjacency[node_id]
        )
    entries = sorted(set(entry_node_ids))
    exits = sorted(set(exit_node_ids))
    if not entries or not exits:
        raise ValueError(
            "cannot canonicalize a graph without explicit entries and exits"
        )
    if any(node_id not in node_ids for node_id in entries + exits):
        raise ValueError("canonical boundary references an unknown business node")
    if STRUCTURAL_SOURCE_ID in node_ids or STRUCTURAL_SINK_ID in node_ids:
        raise ValueError("business graph uses a reserved structural node id")
    if any(
        edge_id.startswith(STRUCTURAL_BOUNDARY_EDGE_PREFIX)
        for edge_id in business_edges
    ):
        raise ValueError("business graph uses a reserved structural edge id")

    canonical = graph.model_copy(deep=True)
    canonical.nodes = dict(business_nodes)
    canonical.edges = dict(business_edges)
    canonical.nodes[STRUCTURAL_SOURCE_ID] = TruthNode(
        id=STRUCTURAL_SOURCE_ID,
        structural=True,
        structural_role="source",
        protected=True,
    )
    canonical.nodes[STRUCTURAL_SINK_ID] = TruthNode(
        id=STRUCTURAL_SINK_ID,
        structural=True,
        structural_role="sink",
        protected=True,
    )
    for ordinal, node_id in enumerate(entries, 1):
        edge_id = _boundary_edge_id("source", ordinal)
        canonical.edges[edge_id] = TruthEdge(
            id=edge_id,
            from_node=STRUCTURAL_SOURCE_ID,
            to_node=node_id,
            edge_kind="structural_boundary",
            structural_only=True,
            protected=True,
        )
    for ordinal, node_id in enumerate(exits, 1):
        edge_id = _boundary_edge_id("sink", ordinal)
        canonical.edges[edge_id] = TruthEdge(
            id=edge_id,
            from_node=node_id,
            to_node=STRUCTURAL_SINK_ID,
            edge_kind="structural_boundary",
            structural_only=True,
            protected=True,
        )
    canonical.source_node_id = STRUCTURAL_SOURCE_ID
    canonical.sink_node_id = STRUCTURAL_SINK_ID
    canonical.start_node_id = None
    canonical.end_node_ids = []
    validate_canonical_graph(canonical)
    return canonical


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentConcept(BaseModel):
    """An Agent-local glossary concept (a business thing of one ConceptKind).

    Agent ids are local and arbitrary; ``display_label`` / ``description`` are
    the Agent's own working text. ``mentions`` are Observation spans the Agent
    interprets as referring to this concept (a diagnostic hint, not a
    correctness gate). There is no validation/grounding lifecycle: concept
    identity is judged by content against Truth, and no private provenance
    status gates recording or scoring.
    """

    id: str
    kind: ConceptKind
    display_label: str
    description: str = Field(default="")
    mentions: list[EvidenceRef] = Field(default_factory=list)


class TerminologyAgreement(BaseModel):
    """A recorded explicit terminology agreement chosen by the Agent.

    Optional Observation evidence is diagnostic metadata; this record is
    separate from ordinary mentions and is never a Truth-reconstruction gate.
    """

    concept_id: str
    term: str
    stakeholder_id: str = Field(default="stakeholder")
    evidence: list[EvidenceRef] = Field(default_factory=list)


class AgentGraph(_GraphMixin[AgentConcept]):
    """The Agent's inferred business process graph + AgentConcept glossary.

    Every property reference may carry optional diagnostic EvidenceRefs;
    cycles are valid; ``start_node_id`` / ``end_node_ids`` restore explicit
    start/end semantics. Slots are four-state (``UNSET`` / ``ConceptRef`` / ``ABSENT``
    / ``DONT_KNOW``).
    """

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
    # Multiple business entries are representable in the Agent graph too;
    # ``start_node_id`` remains a single-entry convenience for old tool calls.
    start_node_ids: list[str] = Field(default_factory=list)

    terminology_agreements: list[TerminologyAgreement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Observations / interview state
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """A single immutable thing the stakeholder said during the interview.

    Observations are independent evidence, not nodes. Immutable once recorded;
    ``turn`` is the conversation message index it derives from.
    """

    model_config = {"frozen": True}

    id: str
    source_id: str = Field(description="Who said it (e.g. the stakeholder).")
    text: str
    order: int = Field(description="Sequence / turn number.")
    locale: Optional[str] = Field(default=None, description="Optional language tag.")
    turn: int = Field(
        description="The conversation message index this observation derives from."
    )

    def has_span(self, quote: str, occurrence: int = 0) -> bool:
        """Deterministic check: ``quote`` occurs at index ``occurrence``."""
        if not quote:
            return False
        start = -1
        for _ in range(occurrence + 1):
            start = self.text.find(quote, start + 1)
            if start == -1:
                return False
        return True


class InterviewResult(BaseModel):
    """The product of an interview: the graph and its evidence."""

    graph: AgentGraph = Field(default_factory=AgentGraph)
    observations: list[Observation] = Field(default_factory=list)


class InterviewDB(DB):
    """State of an interview: the inferred AgentGraph, the conversation
    ledger, and the authentic Observations captured from stakeholder
    messages.

    ``messages`` is an environment-controlled ledger of the conversation
    (role + content per message index); ``observations`` are the ACCEPTED
    stakeholder utterances, auto-created by the environment — one per accepted
    Stakeholder message, BEFORE the Agent sees it. The private semantic
    annotation ledger lives outside this DB (never Agent-visible).
    """

    graph: Optional[AgentGraph] = Field(default=None)
    messages: list[dict] = Field(
        default_factory=list,
        description="Conversation ledger: {role, content} by message index.",
    )
    observations: list[Observation] = Field(default_factory=list)
    interview_complete: bool = Field(default=False)
    summary: Optional[str] = Field(default=None)

    def interview_result(self) -> InterviewResult:
        return InterviewResult(
            graph=self.graph if self.graph is not None else AgentGraph(),
            observations=list(self.observations),
        )
