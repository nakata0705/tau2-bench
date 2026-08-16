"""Agent tools for the evidence-backed DAG business_interview benchmark (v3).

The agent records **Observations** (immutable evidence) and builds / updates an
inferred **BusinessDAG**: create node, update node, attach observation, add /
update edge, set necessity (node property with per-property confidence and
provenance), and set DAG endpoints. Asking a question is recorded as an
Observation; there is no Step / Transition / Branch and no asked/challenged
state in the final DAG.
"""

from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.dag import (
    BusinessDAG,
    Edge,
    InferredValue,
    InterviewDB,
    Necessity,
    Node,
    Observation,
)
from tau2.domains.business_interview.evaluation import evaluate
from tau2.domains.business_interview.scenario import get_scenario
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class InterviewTools(ToolKitBase):
    """Tools to infer the business DAG from stakeholder observations."""

    db: InterviewDB

    def __init__(self, db: InterviewDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------- helpers

    def _dag(self) -> BusinessDAG:
        if self.db.dag is None:
            self.db.dag = BusinessDAG(id="dag", name="")
        return self.db.dag

    def _node(self, node_id: str) -> Node:
        dag = self._dag()
        if node_id not in dag.nodes:
            raise ValueError(f"node not found: {node_id}")
        return dag.nodes[node_id]

    def _edge(self, edge_id: str) -> Edge:
        dag = self._dag()
        if edge_id not in dag.edges:
            raise ValueError(f"edge not found: {edge_id}")
        return dag.edges[edge_id]

    def _require_node_ref(self, node_id: str) -> None:
        if node_id not in self._dag().nodes:
            raise ValueError(f"node not found: {node_id}")

    def _require_observation(self, observation_id: Optional[str]) -> None:
        """Reject a reference to an observation that does not exist.

        ``None`` is allowed (provenance may be added later); a non-None id must
        refer to a recorded observation.
        """
        if observation_id is None:
            return
        if not any(o.id == observation_id for o in self.db.observations):
            raise ValueError(f"observation not found: {observation_id}")

    @staticmethod
    def _iv(
        value: Optional[str],
        confidence: float = 1.0,
        observation_id: Optional[str] = None,
    ) -> InferredValue:
        return InferredValue(
            value=value,
            confidence=confidence,
            observation_ids=[observation_id] if observation_id else [],
        )

    @classmethod
    def _set_value(
        cls,
        current: InferredValue,
        value: str,
        confidence: float,
        observation_id: Optional[str],
    ) -> InferredValue:
        """Set a value while accumulating observation provenance.

        If ``current`` already carried a value, its observation_ids are kept and
        the new observation is merged in (multiple observations can support one
        attribute).
        """
        ids = list(current.observation_ids) if current.value is not None else []
        if observation_id:
            ids.append(observation_id)
        return InferredValue(
            value=value,
            confidence=confidence,
            observation_ids=list(dict.fromkeys(ids)),
        )

    # ------------------------------------------------------------- tools

    @is_tool(ToolType.WRITE)
    def start_inference(self, name: str = "") -> str:
        """Start building the inferred business DAG.

        This is destructive: it discards any previous inferred DAG and any
        recorded observations, and resets the interview. Call it once at the
        beginning of the inference.

        Args:
            name: Optional name for the DAG.

        Returns:
            A confirmation message.
        """
        self.db.dag = BusinessDAG(id="dag", name=name)
        self.db.observations = []
        self.db.interview_complete = False
        self.db.summary = None
        return "Inference started (previous DAG and observations discarded)."

    @is_tool(ToolType.WRITE)
    def record_observation(
        self,
        text: str,
        source_id: str = "stakeholder",
        locale: Optional[str] = None,
    ) -> str:
        """Record an observation (something the stakeholder said).

        Observations are immutable evidence. Record each statement the
        stakeholder makes, then attach it to the node it supports.

        Args:
            text: What the stakeholder said.
            source_id: Who said it (default 'stakeholder').
            locale: Optional language tag.

        Returns:
            The new observation id (to reference when attaching/recording).
        """
        obs = Observation(
            id=f"o{len(self.db.observations) + 1}",
            source_id=source_id,
            text=text,
            order=len(self.db.observations),
            locale=locale,
        )
        self.db.observations.append(obs)
        return obs.id

    @is_tool(ToolType.WRITE)
    def add_node(
        self,
        node_id: str,
        action: str,
        actor: Optional[str] = None,
        system: Optional[str] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        confidence: float = 1.0,
        observation_id: Optional[str] = None,
    ) -> str:
        """Add a node to the inferred DAG.

        Only create a node when no existing node corresponds to the
        observation; otherwise update the existing node.

        Args:
            node_id: Your own identifier for this node.
            action: What is done in this node (natural language).
            actor: Who performs it (optional).
            system: Which system / tool (optional).
            reads: Data this node reads (optional).
            writes: Data this node writes (optional).
            confidence: Confidence in the recorded attributes [0, 1].
            observation_id: Observation supporting this node (optional).

        Returns:
            A confirmation message.
        """
        dag = self._dag()
        if node_id in dag.nodes:
            raise ValueError(f"node already exists: {node_id}")
        self._require_observation(observation_id)
        dag.nodes[node_id] = Node(
            id=node_id,
            action=self._iv(action, confidence, observation_id),
            actor=self._iv(actor, confidence, observation_id)
            if actor is not None
            else InferredValue(),
            system=self._iv(system, confidence, observation_id)
            if system is not None
            else InferredValue(),
            reads=[self._iv(r, confidence, observation_id) for r in (reads or [])],
            writes=[self._iv(w, confidence, observation_id) for w in (writes or [])],
            observation_ids=[observation_id] if observation_id else [],
        )
        return f"Added node {node_id}."

    @is_tool(ToolType.WRITE)
    def update_node(
        self,
        node_id: str,
        action: Optional[str] = None,
        actor: Optional[str] = None,
        system: Optional[str] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        confidence: Optional[float] = None,
        observation_id: Optional[str] = None,
    ) -> str:
        """Update an existing node with new attribute values / evidence.

        Args:
            node_id: The node to update.
            action: New action text (optional).
            actor: New actor (optional).
            system: New system (optional).
            reads: New read data list (optional).
            writes: New write data list (optional).
            confidence: Confidence for the updated attributes [0, 1] (default 1.0).
            observation_id: Observation supporting this update (optional).

        Returns:
            A confirmation message.
        """
        node = self._node(node_id)
        self._require_observation(observation_id)
        conf = confidence if confidence is not None else 1.0
        if action is not None:
            node.action = self._set_value(node.action, action, conf, observation_id)
        if actor is not None:
            node.actor = self._set_value(node.actor, actor, conf, observation_id)
        if system is not None:
            node.system = self._set_value(node.system, system, conf, observation_id)
        if reads is not None:
            node.reads = [self._iv(r, conf, observation_id) for r in reads]
        if writes is not None:
            node.writes = [self._iv(w, conf, observation_id) for w in writes]
        if observation_id:
            node.observation_ids = list(
                dict.fromkeys(node.observation_ids + [observation_id])
            )
        return f"Updated node {node_id}."

    @is_tool(ToolType.WRITE)
    def set_node_necessity(
        self,
        node_id: str,
        rationale: Optional[str] = None,
        owner: Optional[str] = None,
        evidence: Optional[str] = None,
        removal_impact: Optional[str] = None,
        rationale_confidence: Optional[float] = None,
        owner_confidence: Optional[float] = None,
        evidence_confidence: Optional[float] = None,
        removal_confidence: Optional[float] = None,
        observation_id: Optional[str] = None,
        unset: Optional[list[str]] = None,
    ) -> str:
        """Record why a node is needed (a node property).

        Each necessity property is an integrated estimate with its own
        confidence and observation provenance. Leave a property unset (None) to
        record that the necessity is unknown / not asserted. Pass ``unset`` to
        reset a previously recorded property back to unset.

        Args:
            node_id: The node this necessity concerns.
            rationale: Why the node is needed (optional).
            owner: Who requires it (optional).
            evidence: Evidence supporting it (optional).
            removal_impact: What happens if removed (optional).
            *_confidence: Per-property confidence in [0, 1] (default 1.0).
            observation_id: Observation supporting this necessity (optional).
            unset: List of property names (rationale/owner/evidence/removal_impact)
                to reset to unset (optional).

        Returns:
            A confirmation message.
        """
        node = self._node(node_id)
        self._require_observation(observation_id)
        nec = node.necessity if node.necessity is not None else Necessity()
        if unset:
            for p in unset:
                if p in ("rationale", "owner", "evidence", "removal_impact"):
                    setattr(nec, p, InferredValue())
        if rationale is not None:
            nec.rationale = self._set_value(
                nec.rationale,
                rationale,
                rationale_confidence if rationale_confidence is not None else 1.0,
                observation_id,
            )
        if owner is not None:
            nec.owner = self._set_value(
                nec.owner,
                owner,
                owner_confidence if owner_confidence is not None else 1.0,
                observation_id,
            )
        if evidence is not None:
            nec.evidence = self._set_value(
                nec.evidence,
                evidence,
                evidence_confidence if evidence_confidence is not None else 1.0,
                observation_id,
            )
        if removal_impact is not None:
            nec.removal_impact = self._set_value(
                nec.removal_impact,
                removal_impact,
                removal_confidence if removal_confidence is not None else 1.0,
                observation_id,
            )
        node.necessity = nec
        if observation_id:
            node.observation_ids = list(
                dict.fromkeys(node.observation_ids + [observation_id])
            )
        return f"Set necessity for node {node_id}."

    @is_tool(ToolType.WRITE)
    def add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        predicate: Optional[str] = None,
        confidence: float = 1.0,
        observation_id: Optional[str] = None,
    ) -> str:
        """Add a directed edge between two nodes.

        Use ``predicate`` for a control-flow condition (e.g. 'amount over
        1,000,000'). Leave it None for unconditional flow. A conditional branch
        is expressed as multiple outgoing edges with different predicates.

        Args:
            edge_id: Your own identifier for this edge.
            from_node: Source node id.
            to_node: Destination node id.
            predicate: Optional control-flow condition.
            confidence: Confidence in the edge [0, 1].
            observation_id: Observation supporting this edge (optional).

        Returns:
            A confirmation message.
        """
        dag = self._dag()
        if edge_id in dag.edges:
            raise ValueError(f"edge already exists: {edge_id}")
        self._require_observation(observation_id)
        self._require_node_ref(from_node)
        self._require_node_ref(to_node)
        dag.edges[edge_id] = Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            predicate=self._iv(predicate, confidence, observation_id)
            if predicate is not None
            else None,
            observation_ids=[observation_id] if observation_id else [],
        )
        return f"Added edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def update_edge(
        self,
        edge_id: str,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        predicate: Optional[str] = None,
        clear_predicate: bool = False,
        confidence: Optional[float] = None,
        observation_id: Optional[str] = None,
    ) -> str:
        """Update an edge's endpoints or predicate.

        Args:
            edge_id: The edge to update.
            from_node: New source (optional).
            to_node: New destination (optional).
            predicate: New predicate (optional).
            clear_predicate: If True, remove the predicate (unconditional).
            confidence: Confidence for updated values [0, 1].
            observation_id: Observation supporting this edge (optional).

        Returns:
            A confirmation message.
        """
        edge = self._edge(edge_id)
        self._require_observation(observation_id)
        conf = confidence if confidence is not None else 1.0
        if from_node is not None:
            self._require_node_ref(from_node)
            edge.from_node = from_node
        if to_node is not None:
            self._require_node_ref(to_node)
            edge.to_node = to_node
        if clear_predicate:
            edge.predicate = None
        elif predicate is not None:
            cur = edge.predicate if edge.predicate is not None else InferredValue()
            edge.predicate = self._set_value(cur, predicate, conf, observation_id)
        if observation_id:
            edge.observation_ids = list(
                dict.fromkeys(edge.observation_ids + [observation_id])
            )
        return f"Updated edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def attach_observation(self, node_id: str, observation_id: str) -> str:
        """Attach an observation to a node (multiple observations per node ok)."""
        node = self._node(node_id)
        self._require_observation(observation_id)
        node.observation_ids = list(
            dict.fromkeys(node.observation_ids + [observation_id])
        )
        return f"Attached {observation_id} to node {node_id}."

    @is_tool(ToolType.WRITE)
    def set_dag_endpoints(
        self,
        start_node_id: Optional[str] = None,
        end_node_ids: Optional[list[str]] = None,
    ) -> str:
        """Set the DAG's start and end node ids."""
        dag = self._dag()
        if start_node_id is not None:
            self._require_node_ref(start_node_id)
            dag.start_node_id = start_node_id
        if end_node_ids is not None:
            for eid in end_node_ids:
                self._require_node_ref(eid)
            dag.end_node_ids = list(end_node_ids)
        return "Set DAG endpoints."

    @is_tool(ToolType.WRITE)
    def finish_interview(self, summary: Optional[str] = None) -> str:
        """Mark the interview complete."""
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked complete."

    # ------------------------------------------------------------- assertions

    def assert_finish_interview(self) -> bool:
        return self.db.interview_complete

    def assert_dag_reconstructed(self, scenario_id: str) -> bool:
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        return evaluate(self.db, sc.truth, sc.spec).structural_pass

    def assert_necessity_handled(self, scenario_id: str) -> bool:
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        return evaluate(self.db, sc.truth, sc.spec).necessity_pass

    def assert_evidence_backed(self, scenario_id: str) -> bool:
        """True if every asserted claim is traceable to a recorded Observation."""
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        return evaluate(self.db, sc.truth, sc.spec).evidence_pass

    # ------------------------------------------------------------- diagnostics

    def get_eval_diagnostics(self, task: Optional[Task] = None) -> Optional[dict]:
        sc = get_scenario(task.id if task is not None else None)
        if sc is None:
            return None
        return evaluate(self.db, sc.truth, sc.spec).model_dump(mode="json")
