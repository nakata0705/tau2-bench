"""Unified Agent-local glossary + business process graph (v8 — provenance-only).

The interview's product is a **BusinessProcessGraph**: nodes (what happens, by
whom, on what data) and directed edges (with optional conditions), both
referencing a **glossary** of Agent-local ``BusinessConcept``\\ s. Concepts are
typed (``ConceptKind``) and carry a validation status; every claim in the graph
cites **EvidenceRef**\\ s — exact spans of immutable stakeholder Observations.

There is no DAG terminology and no acyclicity requirement: **cycles are valid**
(a process may revisit a step). There is no ``LoopNode``.

The evaluator derives correctness **only** through private provenance (see
``evaluation.py``): ConceptRef -> EvidenceRef -> exact Observation span ->
private stakeholder assertion -> StakeholderFact -> TruthClaim -> Truth
concept. None of the text fields here (labels, descriptions, terms, quotes)
are ever interpreted semantically by the evaluator.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.environment.db import DB

ConceptKind = Literal["activity", "actor", "system", "data", "condition", "rationale"]

ValidationStatus = Literal[
    "hypothesized", "confirmed", "partially_confirmed", "disputed", "unknown"
]

# Property names scored on nodes / edges (TruthClaim.property).
NodeProperty = Literal["activity", "actor", "system", "reads", "writes", "rationale"]
EdgeProperty = Literal["condition", "edge_exists"]


class EvidenceRef(BaseModel):
    """One exact span of an immutable Observation cited as evidence.

    ``quote`` must be an exact substring occurrence of the immutable
    Observation's text; ``occurrence`` selects which occurrence (0-based) when
    the quote appears multiple times. Validity is checked deterministically
    (the evaluator never infers what the quote *means*).
    """

    observation_id: str
    quote: str = Field(description="Exact substring of the Observation text.")
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )


class ConceptTerm(BaseModel):
    """One observed wording of a concept, with its supporting evidence."""

    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class BusinessConcept(BaseModel):
    """An Agent-local glossary concept (a business object / role / activity).

    Agent ids are local and arbitrary; labels/descriptions are the Agent's own
    wording and are never compared to Truth. ``validation_status`` tracks the
    Agent's confidence in the concept's identity; ``validation_evidence``
    records the authentic stakeholder evidence behind confirmations.
    """

    id: str
    kind: ConceptKind
    preferred_label: str
    description: str = Field(default="")
    terms: list[ConceptTerm] = Field(default_factory=list)
    validation_status: ValidationStatus = Field(default="hypothesized")
    validation_evidence: list[EvidenceRef] = Field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """True when the concept is no longer merely hypothesized."""
        return self.validation_status != "hypothesized"


class ConceptRef(BaseModel):
    """A reference from a node/edge to an Agent-local glossary concept.

    ``evidence`` cites the exact Observation spans that support using this
    concept at this slot. ``confidence`` in [0, 1]; 0 = unasserted.
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
    """A vertex in the business process graph.

    ``activity`` is required and references an activity concept; ``actor`` /
    ``system`` / ``necessity_rationale`` are optional single references;
    ``reads`` / ``writes`` are lists of data-concept references. Every ref
    carries its own evidence. ``from``/``to`` structural identity is expressed
    only through edges; node ids are Agent-local.
    """

    id: str
    activity: ConceptRef
    actor: Optional[ConceptRef] = None
    system: Optional[ConceptRef] = None
    reads: list[ConceptRef] = Field(default_factory=list)
    writes: list[ConceptRef] = Field(default_factory=list)
    necessity_rationale: Optional[ConceptRef] = None

    def refs(self, property_name: str) -> list[ConceptRef]:
        """The refs for a node property (single or list).

        ``property_name`` accepts the claim-style names (activity/actor/system/
        reads/writes/rationale) as well as the attribute name
        ``necessity_rationale``.
        """
        if property_name == "reads":
            return list(self.reads)
        if property_name == "writes":
            return list(self.writes)
        attr = "necessity_rationale" if property_name == "rationale" else property_name
        ref = getattr(self, attr)
        return [ref] if ref is not None else []

    def asserted_refs(self, property_name: str) -> list[ConceptRef]:
        return [r for r in self.refs(property_name) if r.asserted]


class Edge(BaseModel):
    """A directed edge between two nodes.

    ``from_node`` / ``to_node`` are structural identities (node ids), not
    concepts. ``condition`` optionally references a condition concept.
    ``evidence`` cites the Observation spans supporting the relation.
    """

    id: str
    from_node: str
    to_node: str
    condition: Optional[ConceptRef] = None
    evidence: list[EvidenceRef] = Field(default_factory=list)


def _node_property_refs(node: "Node") -> dict[str, list[ConceptRef]]:
    """All refs of a node keyed by property name (single refs as one-element
    lists; reads/writes as their lists)."""
    return {
        "activity": [node.activity],
        "actor": [node.actor] if node.actor is not None else [],
        "system": [node.system] if node.system is not None else [],
        "reads": list(node.reads),
        "writes": list(node.writes),
        "necessity_rationale": (
            [node.necessity_rationale] if node.necessity_rationale is not None else []
        ),
    }


class BusinessProcessGraph(BaseModel):
    """A business process graph: nodes, edges, and the Agent-local glossary.

    Cycles are valid — there is deliberately no acyclicity validation.
    """

    id: str = Field(default="graph")
    name: str = Field(default="")
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
    concepts: dict[str, BusinessConcept] = Field(default_factory=dict)

    # ---------------------------------------------------------------- graph utils

    def successors(self, node_id: str) -> list[str]:
        return [e.to_node for e in self.edges.values() if e.from_node == node_id]

    def structure_errors(self) -> list[str]:
        """Internal self-consistency errors (empty = well-formed).

        Checks only references and required fields — cycles are valid and
        never reported as errors. Nothing here references hidden Ground Truth.
        """
        errors: list[str] = []
        for nid, node in self.nodes.items():
            if not node.activity.concept_id:
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
                edge.condition is not None
                and edge.condition.concept_id not in self.concepts
            ):
                errors.append(
                    f"edge {eid}: condition references unknown concept "
                    f"{edge.condition.concept_id!r}"
                )
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
            if edge.condition is not None and edge.condition.asserted:
                ids.add(edge.condition.concept_id)
        return ids


class InterviewResult(BaseModel):
    """The product of an interview: the graph and its evidence."""

    graph: BusinessProcessGraph = Field(default_factory=BusinessProcessGraph)
    observations: list["Observation"] = Field(default_factory=list)


class Observation(BaseModel):
    """A single immutable thing the stakeholder said during the interview.

    Observations are independent evidence, not nodes. Immutable once recorded;
    ``turn`` is the conversation message index it derives from.
    """

    model_config = ConfigDict(frozen=True)

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


class InterviewDB(DB):
    """State of an interview: the inferred graph, the conversation ledger, and
    the authentic Observations captured from stakeholder messages.

    ``messages`` is an environment-controlled ledger of the conversation
    (role + content per message index); ``observations`` are derived only from
    user (stakeholder) messages via ``observe_message``. The private
    assertion ledger lives outside this DB (never Agent-visible).
    """

    graph: Optional[BusinessProcessGraph] = Field(default=None)
    messages: list[dict] = Field(
        default_factory=list,
        description="Conversation ledger: {role, content} by message index.",
    )
    observations: list[Observation] = Field(default_factory=list)
    interview_complete: bool = Field(default=False)
    summary: Optional[str] = Field(default=None)

    def interview_result(self) -> InterviewResult:
        return InterviewResult(
            graph=self.graph if self.graph is not None else BusinessProcessGraph(),
            observations=list(self.observations),
        )
