"""Graph-native semantic model for business_interview (v11 — opaque stakeholder IDs).

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

A property slot is **three-valued**: ``ConceptRef`` (value known), ``None``
(value known absent), or ``DONT_KNOW`` (element known, value unknown). This
is the stakeholder's epistemic state AND the AgentGraph's epistemic state:
``DONT_KNOW`` markers carry the Observation evidence that the stakeholder
really said "I don't know" for that exact slot.

**Mention != evidence != validation.** ``AgentConcept.mentions`` are spans
the Agent interprets as referring to the concept; property references carry
their own ``EvidenceRef``\\ s (property scoring uses property evidence ONLY);
validation statuses are backed by explicit validation/dialogue evidence.

The evaluator derives correctness **only** through private provenance:
Agent EvidenceRef -> Observation span -> private annotation (stakeholder
semantic ID) -> StakeholderKnowledgeGraph / StakeholderKnowledgeConcept.
None of the text fields are ever interpreted semantically.
"""

from typing import Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator

from tau2.environment.db import DB

C = TypeVar("C")

ConceptKind = Literal["activity", "actor", "system", "data", "condition", "rationale"]

ValidationStatus = Literal[
    "hypothesized",
    "grounded",
    "confirmed",
    "partially_confirmed",
    "disputed",
    "unknown",
]

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
    """One span of an immutable Observation cited as evidence.

    ``quote`` must be an exact substring occurrence of the immutable
    Observation's text; ``occurrence`` selects which occurrence (0-based) when
    the quote appears multiple times. Validity is checked deterministically
    (the evaluator never infers what the quote *means*). The span resolves to
    concrete character offsets in the Observation text.
    """

    observation_id: str
    quote: str = Field(description="Exact substring of the Observation text.")
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )

    def resolve_span(self, text: str) -> Optional[tuple[int, int]]:
        """Resolve to (start, end) character offsets in ``text``, or None when
        the quote/occurrence does not exactly match ``text``."""
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


class DontKnowType(BaseModel):
    """The DONT_KNOW marker: the element exists but its value is unknown.

    ``DONT_KNOW`` (module singleton) is distinct from ``None`` (known
    absent) and from ``ConceptRef`` (known value). On the AgentGraph side a
    marker carries the Observation evidence that the stakeholder really said
    "I don't know" for that exact slot (validated against the private
    stakeholder DONT_KNOW slots by the tools/evaluator).
    """

    evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def asserted(self) -> bool:
        """A DONT_KNOW marker IS an active epistemic claim ("I cannot
        determine this") — distinct from an unasserted slot."""
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

    ``evidence`` cites the Observation spans that support using this concept
    at this slot. ``confidence`` in [0, 1]; 0 = unasserted.
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


class Node(BaseModel):
    """A vertex in a business process graph.

    ``activity`` is required and is either an activity reference or
    ``DONT_KNOW``; ``actor`` / ``system`` / ``necessity_rationale`` are
    optional single references (``None`` = known absent, ``DONT_KNOW`` =
    unknown); ``reads`` / ``writes`` are ``None`` (known absent),
    ``DONT_KNOW`` (unknown), or lists of data-concept references. Every ref
    and every DONT_KNOW marker carries its own evidence. ``from``/``to``
    structural identity is expressed only through edges; node ids are local
    to the graph.
    """

    id: str
    activity: Union[ConceptRef, DontKnowType]
    actor: Optional[Union[ConceptRef, DontKnowType]] = None
    system: Optional[Union[ConceptRef, DontKnowType]] = None
    reads: Optional[Union[list[ConceptRef], DontKnowType]] = None
    writes: Optional[Union[list[ConceptRef], DontKnowType]] = None
    necessity_rationale: Optional[Union[ConceptRef, DontKnowType]] = None

    def refs(self, property_name: str) -> list[ConceptRef]:
        """The concept refs of a node property (single or list).

        ``property_name`` accepts the claim-style names (activity/actor/system/
        reads/writes/rationale) as well as the attribute name
        ``necessity_rationale``. DONT_KNOW slots contribute no refs.
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
        """The three-valued property slot: ConceptRef / DONT_KNOW / None for
        scalars; list[ConceptRef] / DONT_KNOW / None for reads/writes."""
        if property_name in ("reads", "writes"):
            return getattr(self, property_name)
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        return getattr(self, attr)


class Edge(BaseModel):
    """A directed edge between two nodes.

    ``from_node`` / ``to_node`` are structural identities (node ids), not
    concepts. ``condition`` is ``None`` (known absent), ``DONT_KNOW``
    (unknown) or a reference to a condition concept (kind=condition).
    ``evidence`` cites the Observation spans supporting the relation.
    """

    id: str
    from_node: str
    to_node: str
    condition: Optional[Union[ConceptRef, DontKnowType]] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


def _node_property_refs(node: Node) -> dict[str, list[ConceptRef]]:
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
    """Shared graph structure + utilities (cycles are valid; no acyclicity
    requirement). The concrete concept type is fixed by the subclass."""

    id: str = "graph"
    name: str = ""
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
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
            if not isinstance(node.activity, ConceptRef):
                pass  # DONT_KNOW — the agent cannot determine it; valid
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


class BusinessProcessGraph(_GraphMixin[TruthConcept]):
    """The Truth: nodes, edges and the TruthConcept glossary.

    The graph itself is the semantic model — there are no generated claims.
    Node/edge ids are Truth-local and stable; every addressable element has a
    semantic ID (see module docstring).
    """


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentConcept(BaseModel):
    """An Agent-local glossary concept (a business thing of one ConceptKind).

    Agent ids are local and arbitrary; ``display_label`` / ``description`` are
    the Agent's own working text and are never compared to Truth.
    ``mentions`` are Observation spans the Agent interprets as referring to
    this concept — a mention is NOT evidence for a graph property and NOT a
    terminology agreement. ``validation_status`` is one of hypothesized /
    grounded / confirmed / partially_confirmed / disputed / unknown;
    ``validation_evidence`` records the explicit dialogue/validation evidence
    behind the status.
    """

    id: str
    kind: ConceptKind
    display_label: str
    description: str = Field(default="")
    mentions: list[EvidenceRef] = Field(default_factory=list)
    validation_status: ValidationStatus = Field(default="hypothesized")
    validation_evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """True when the concept is no longer merely hypothesized."""
        return self.validation_status != "hypothesized"


class TerminologyAgreement(BaseModel):
    """A recorded explicit terminology agreement: the Agent proposed ``term``
    for ``concept_id`` and the stakeholder confirmed it (evidenced by an
    Observation span). Recorded separately from mere mentions."""

    concept_id: str
    term: str
    stakeholder_id: str = Field(default="stakeholder")
    evidence: list[EvidenceRef] = Field(default_factory=list)


class AgentGraph(_GraphMixin[AgentConcept]):
    """The Agent's inferred business process graph + AgentConcept glossary.

    Every property reference carries its own EvidenceRef; cycles are valid;
    ``start_node_id`` / ``end_node_ids`` restore explicit start/end
    semantics.
    """

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
    (role + content per message index); ``observations`` are derived only from
    user (stakeholder) messages via ``observe_message``. The private semantic
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
