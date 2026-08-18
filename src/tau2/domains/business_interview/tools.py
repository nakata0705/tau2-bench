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

        This is destructive: it discards the previous inferred DAG and any
        captured Observations, and resets the interview. The conversation ledger
        (the stakeholder's actual messages) is kept — observations are re-captured
        from it with ``observe_message`` / ``observe_latest_stakeholder_message``.
        Call it once at the beginning.

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

    def _stakeholder_entries(self) -> list[tuple[str, int, str]]:
        """Deterministic list of stakeholder messages as (stable id, ledger turn,
        content). The id is ``sm_N`` where N is the ordinal of the user message in
        the (append-only) conversation ledger, so ids are stable and unique even
        across state replay / ``set_state``.
        """
        entries = []
        counter = 0
        for i, msg in enumerate(self.db.messages):
            if msg.get("role") == "user":
                counter += 1
                entries.append((f"sm_{counter}", i, msg.get("content") or ""))
        return entries

    def _capture_user_message(self, turn: int, content: str) -> str:
        """Capture a user message already validated as a stakeholder message.

        Idempotent: re-capturing the same ledger turn returns the same
        observation id. The observation's text/turn come from the actual ledger,
        so evaluator provenance-authenticity is preserved.
        """
        for obs in self.db.observations:
            if obs.turn == turn:
                return obs.id
        obs = Observation(
            id=f"obs_{turn}",
            source_id="stakeholder",
            text=content,
            order=len(self.db.observations),
            turn=turn,
        )
        self.db.observations.append(obs)
        return obs.id

    @is_tool(ToolType.WRITE)
    def observe_message(self, message_id: str) -> str:
        """Capture the stakeholder message with ``message_id`` as an Observation.

        Message ids look like ``sm_1``, ``sm_2``, ... and are listed by
        ``list_stakeholder_messages``. Prefer
        ``observe_latest_stakeholder_message`` right after the stakeholder
        speaks; use this to re-reference an earlier message by its stable id.

        The Observation's text, source and turn are taken from the actual
        conversation message — you cannot write arbitrary text. The capture is
        idempotent: re-capturing the same message returns the same observation id.

        Only user (stakeholder) messages can be observed; a fabricated or
        non-stakeholder message id is rejected.

        Args:
            message_id: The stable stakeholder message id (e.g. ``"sm_3"``).

        Returns:
            The observation id (use it as provenance on nodes / edges /
            necessity).
        """
        for sm_id, turn, content in self._stakeholder_entries():
            if sm_id == message_id:
                return self._capture_user_message(turn, content)
        raise ValueError(f"no stakeholder message with id {message_id!r}")

    @is_tool(ToolType.READ)
    def observe_latest_stakeholder_message(self) -> str:
        """Return the id of the most recent stakeholder (user) message.

        Then call ``observe_message(message_id=...)`` to capture that message as
        an Observation. This lets you act on the newest statement without
        tracking conversation turn indices. (It is a read — it does not itself
        create an Observation — so it is deterministic under state replay.)

        Returns:
            The stable id of the latest stakeholder message (e.g. ``"sm_3"``).
        """
        entries = self._stakeholder_entries()
        if not entries:
            raise ValueError("no stakeholder (user) message has been recorded yet")
        return entries[-1][0]

    @is_tool(ToolType.READ)
    def list_stakeholder_messages(self) -> str:
        """List the stakeholder (user) messages by their stable message ids.

        Use ``observe_message(message_id)`` to capture a specific one, or
        ``observe_latest_stakeholder_message()`` for the most recent.

        Returns:
            A list of stakeholder messages keyed by stable id (``sm_1``, ...).
        """
        lines = [
            f"{sm_id}: {content}"
            for sm_id, turn, content in self._stakeholder_entries()
        ]
        return "\n".join(lines) if lines else "(no stakeholder messages yet)"

    @is_tool(ToolType.WRITE)
    def add_node(
        self,
        node_id: str,
        action: str,
        primitive: Optional[str] = None,
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
            action: What is done in this node (open-world natural language).
            primitive: Optional generic operation (create/check/approve/... or
                'unclassified').
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
            primitive=self._iv(primitive, confidence, observation_id)
            if primitive is not None
            else None,
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
        primitive: Optional[str] = None,
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
            primitive: New generic primitive (optional).
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
        if primitive is not None:
            node.primitive = self._set_value(
                node.primitive if node.primitive is not None else InferredValue(),
                primitive,
                conf,
                observation_id,
            )
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
    def remove_node(self, node_id: str) -> str:
        """Remove a node and its incident edges from the inferred DAG.

        The DAG is a working hypothesis, not an append-only record: you may
        remove a coarse placeholder node once you have decomposed it into more
        specific activities (e.g. a single ``coarse_step`` replaced by
        ``step_a`` -> ``step_b`` -> ``step_c``).

        This also removes every edge that connects to the node (incoming or
        outgoing), so no dangling edge is left behind. If the removed node was
        the declared start or an end node, that endpoint reference is cleared
        (re-set it afterwards with ``set_dag_endpoints``). The stakeholder
        Observations themselves are kept — the node's evidence provenance on
        other nodes is unaffected.

        Args:
            node_id: The node to remove.

        Returns:
            A confirmation message listing what was removed.

        Raises:
            ValueError: If the node does not exist.
        """
        dag = self._dag()
        if node_id not in dag.nodes:
            raise ValueError(f"node not found: {node_id}")
        removed_edges = sorted(
            eid
            for eid, e in dag.edges.items()
            if e.from_node == node_id or e.to_node == node_id
        )
        for eid in removed_edges:
            del dag.edges[eid]
        touched_endpoint = False
        if dag.start_node_id == node_id:
            dag.start_node_id = None
            touched_endpoint = True
        if node_id in dag.end_node_ids:
            dag.end_node_ids = [e for e in dag.end_node_ids if e != node_id]
            touched_endpoint = True
        del dag.nodes[node_id]
        parts = [f"Removed node {node_id}."]
        if removed_edges:
            parts.append(f"Removed incident edges: {', '.join(removed_edges)}.")
        if touched_endpoint:
            parts.append("The removed node was a declared endpoint; re-set it.")
        return " ".join(parts)

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

    @is_tool(ToolType.READ)
    def validate_dag(self) -> str:
        """Validate the inferred DAG's **internal** structural consistency.

        Reports only self-consistency problems (unreachable nodes, dangling
        edges, invalid start/end, cycles, ...), never the hidden Ground Truth.
        Call this to review the final structure before ``finish_interview``.

        Returns:
            The validation result (or the list of structural errors).
        """
        dag = self._dag()
        if not dag.nodes:
            return "No DAG nodes yet. Call start_inference to begin."
        errors = dag.validate()
        if not errors:
            return "DAG is structurally valid."
        return "DAG validation errors:\n- " + "\n- ".join(errors)

    @is_tool(ToolType.WRITE)
    def finish_interview(self, summary: Optional[str] = None) -> str:
        """Mark the interview complete.

        Refuses to finish a structurally invalid DAG (unreachable nodes, dangling
        edges, invalid start/end, cycles, ...). The reported errors are only about
        the DAG's own internal consistency — never about the hidden Ground Truth.
        Fix the issues and call this again to finish.

        Args:
            summary: Optional summary of what was captured.

        Returns:
            A confirmation message.

        Raises:
            ValueError: If no DAG was built, or the DAG is structurally invalid.
        """
        dag = self._dag()
        if not dag.nodes:
            raise ValueError(
                "Cannot finish: no DAG has been built yet. Call start_inference first."
            )
        errors = dag.validate()
        if errors:
            raise ValueError(
                "Cannot finish: DAG is structurally invalid.\n- " + "\n- ".join(errors)
            )
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
