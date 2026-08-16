"""DAG-based domain model for business_interview (v3 — evidence-backed DAG inference).

This replaces the old Step / Transition / Branch workflow model. The core idea:

- The real business process is a **Truth DAG** (1 start, 1+ ends, acyclic, all
  nodes reachable, end nodes have out-degree 0).
- The stakeholder is a **filter** over that Truth DAG: they know only a part of
  it and observe it in natural language.
- Each thing the stakeholder says during the interview is an **Observation**
  (independent, immutable evidence).
- The agent incrementally integrates observations into an **Inferred DAG**,
  attaching multiple observations to a node and updating node attributes /
  confidence, creating a node only when no existing node corresponds.

The **same ``BusinessDAG`` class** is used for the Truth DAG (scenario) and the
agent's Inferred DAG. There are no Truth-only node/edge classes.

Every inferred attribute carries its own confidence and observation provenance
via the small ``InferredValue`` value object — confidence is not a single number
per node.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau2.environment.db import DB


class InferredValue(BaseModel):
    """A value plus its confidence and the observations it is inferred from.

    ``confidence`` is in [0, 1]. ``observation_ids`` record provenance (which
    observed statements support this value). On the Truth DAG these are simply
    unset (confidence 0, no provenance) — confidence/provenance describe an
    *inference*, not the objective workflow.
    """

    value: Optional[str] = Field(
        default=None, description="The value (canonical label)."
    )
    confidence: float = Field(default=0.0, description="Confidence in [0, 1].")
    observation_ids: list[str] = Field(
        default_factory=list, description="Observation ids that support this value."
    )

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {v}")
        return v

    @property
    def is_set(self) -> bool:
        return self.value is not None

    @property
    def asserted(self) -> bool:
        """True if this is an active claim (a value asserted with confidence > 0).

        A value with ``confidence == 0`` is treated as unknown/unasserted: it is
        not evidence-backed and does not satisfy a known expectation.
        """
        return self.value is not None and self.confidence > 0


class Necessity(BaseModel):
    """Why a node is needed.

    Necessity is a node property (0 or 1 per node). Each property is an
    ``InferredValue``: an integrated estimate across multiple observations, with
    its own confidence and provenance — not a single answer.
    """

    rationale: InferredValue = Field(
        default_factory=InferredValue, description="Why this step is needed."
    )
    owner: InferredValue = Field(
        default_factory=InferredValue, description="Who requires / owns it."
    )
    evidence: InferredValue = Field(
        default_factory=InferredValue,
        description="Evidence supporting the requirement.",
    )
    removal_impact: InferredValue = Field(
        default_factory=InferredValue, description="What happens if removed."
    )


class Node(BaseModel):
    """A vertex in the business DAG.

    ``id`` is the (possibly agent-assigned) node identifier. ``action`` is the
    natural-language action; ``actor`` / ``system`` / ``reads`` / ``writes`` and
    ``necessity`` describe the node. Each attribute is an ``InferredValue`` so
    confidence / provenance can differ per attribute.
    """

    id: str
    action: InferredValue = Field(default_factory=InferredValue)
    actor: InferredValue = Field(default_factory=InferredValue)
    system: InferredValue = Field(default_factory=InferredValue)
    reads: list[InferredValue] = Field(default_factory=list)
    writes: list[InferredValue] = Field(default_factory=list)
    necessity: Optional[Necessity] = Field(
        default=None,
        description="Why the node is needed (0 or 1 per node); None = no requirement.",
    )
    observation_ids: list[str] = Field(
        default_factory=list,
        description="Observations attached to this node (multiple allowed).",
    )


class Edge(BaseModel):
    """A directed edge between two nodes.

    ``from_node`` / ``to_node`` reference node ids. ``predicate`` is the optional
    control-flow condition; ``None`` means unconditional flow. Conditional
    branching is expressed purely as multiple outgoing edges with different
    predicates — there is no Branch class and no Node-level condition.
    """

    id: str
    from_node: str
    to_node: str
    predicate: Optional[InferredValue] = Field(
        default=None, description="Control-flow condition (None = unconditional)."
    )
    observation_ids: list[str] = Field(default_factory=list)


class Observation(BaseModel):
    """A single immutable thing the stakeholder said during the interview.

    An Observation is independent evidence, not a Node. Multiple observations may
    be attached to the same Node. Observations are immutable once recorded.
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


class BusinessDAG(BaseModel):
    """A business process DAG. Used identically for the Truth DAG and the
    agent's inferred DAG."""

    id: str = Field(default="dag")
    name: str = Field(default="")
    nodes: dict[str, Node] = Field(default_factory=dict, description="Nodes by id.")
    edges: dict[str, Edge] = Field(default_factory=dict, description="Edges by id.")
    start_node_id: Optional[str] = Field(default=None)
    end_node_ids: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- graph utils

    def successors(self, node_id: str) -> list[str]:
        return [e.to_node for e in self.edges.values() if e.from_node == node_id]

    def out_degree(self, node_id: str) -> int:
        return len(self.successors(node_id))

    def reachable_from(self, start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(self.successors(n))
        return seen

    def find_cycle(self) -> Optional[list[str]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self.nodes}

        def dfs(nid: str, path: list[str]) -> Optional[list[str]]:
            color[nid] = GRAY
            path.append(nid)
            for s in self.successors(nid):
                if color.get(s) == GRAY:
                    return path[path.index(s) :] + [s]
                if color.get(s) == WHITE:
                    r = dfs(s, path)
                    if r:
                        return r
            color[nid] = BLACK
            path.pop()
            return None

        for nid in self.nodes:
            if color[nid] == WHITE:
                r = dfs(nid, [])
                if r:
                    return r
        return None

    # ---------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Return a list of structural validation errors (empty = well-formed).

        A well-formed DAG has: exactly one start node, one or more end nodes,
        no cycles, every node reachable from the start, no dangling edges, and
        end nodes with out-degree 0.
        """
        errors: list[str] = []
        if self.start_node_id is None:
            errors.append("start_node_id must be set")
        elif self.start_node_id not in self.nodes:
            errors.append(f"start node not found: {self.start_node_id}")
        if not self.end_node_ids:
            errors.append("end_node_ids must declare at least one end node")
        for eid in self.end_node_ids:
            if eid not in self.nodes:
                errors.append(f"end node not found: {eid}")
            elif self.out_degree(eid) != 0:
                errors.append(f"end node has outgoing edge: {eid}")
        for e in self.edges.values():
            if e.from_node not in self.nodes:
                errors.append(f"edge {e.id}: from_node not found: {e.from_node}")
            if e.to_node not in self.nodes:
                errors.append(f"edge {e.id}: to_node not found: {e.to_node}")
        cycle = self.find_cycle()
        if cycle:
            errors.append("cycle detected: " + " -> ".join(cycle))
        if self.start_node_id in self.nodes:
            reachable = self.reachable_from(self.start_node_id)
            for nid in self.nodes:
                if nid not in reachable:
                    errors.append(f"unreachable node: {nid}")
        return errors

    @property
    def is_valid(self) -> bool:
        return not self.validate()


class InterviewResult(BaseModel):
    """The product of an interview.

    ``dag`` is the agent's current business understanding; ``observations`` are
    the evidence it is based on.
    """

    dag: BusinessDAG = Field(default_factory=BusinessDAG)
    observations: list[Observation] = Field(default_factory=list)


class InterviewDB(DB):
    """State of an interview: the inferred DAG, the conversation ledger, and the
    authentic Observations captured from stakeholder messages.

    ``messages`` is an environment-controlled ledger of the conversation
    (role + content per message index); ``observations`` are derived only from
    user (stakeholder) messages via ``observe_turn``.
    """

    dag: Optional[BusinessDAG] = Field(default=None)
    messages: list[dict] = Field(
        default_factory=list,
        description="Conversation ledger: {role, content} by message index.",
    )
    observations: list[Observation] = Field(default_factory=list)
    interview_complete: bool = Field(default=False)
    summary: Optional[str] = Field(default=None)

    def interview_result(self) -> InterviewResult:
        return InterviewResult(
            dag=self.dag if self.dag is not None else BusinessDAG(),
            observations=list(self.observations),
        )
