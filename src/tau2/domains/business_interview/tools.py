"""Agent tools for the unified-glossary business_interview benchmark (v5).

The agent records **Observations** (immutable evidence), builds a private
**glossary** of typed ``BusinessConcept``\\ s (activity / actor / system / data /
condition / rationale), and constructs an inferred **BusinessProcessGraph**
whose node/edge properties reference glossary concepts. Every claim cites
**EvidenceRef**\\ s — exact spans (quote + occurrence) of authentic
Observations.

Glossary lifecycle: concepts start ``hypothesized``; the agent confirms them
with authentic stakeholder evidence (``confirm_concept``), marks them unknown
or disputed, merges mistakenly split concepts, and must resolve every
*referenced* concept before ``finish_interview``.
"""

from typing import Optional

from pydantic import ValidationError

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.evaluation import EvaluationResult, evaluate
from tau2.domains.business_interview.facts import StakeholderFactLedger
from tau2.domains.business_interview.graph import (
    BusinessConcept,
    BusinessProcessGraph,
    ConceptRef,
    ConceptTerm,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    Observation,
)
from tau2.domains.business_interview.scenario import get_scenario
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

_SINGLE_PROPS = ("actor", "system", "necessity_rationale")
_SINGLE_PROP_NAMES = {
    "actor": "actor",
    "system": "system",
    "necessity_rationale": "rationale",
}


def _ev(value: dict) -> EvidenceRef:
    """Coerce one EvidenceRef dict, raising a clear error on bad shape."""
    try:
        return EvidenceRef(**value)
    except ValidationError as exc:
        raise ValueError(f"invalid evidence ref {value!r}: {exc}") from exc


class InterviewTools(ToolKitBase):
    """Tools to infer the business process graph from stakeholder observations.

    ``fact_ledger`` is the private assertion sidecar ledger, shared with the
    environment and the stakeholder simulator adapter. It is evaluator-only:
    no tool exposes it, and it is never serialized into the Agent-visible DB.
    """

    db: InterviewDB

    def __init__(
        self, db: InterviewDB, fact_ledger: Optional[StakeholderFactLedger] = None
    ) -> None:
        super().__init__(db)
        self.fact_ledger = (
            fact_ledger if fact_ledger is not None else StakeholderFactLedger()
        )

    # ------------------------------------------------------------- helpers

    def _graph(self) -> BusinessProcessGraph:
        if self.db.graph is None:
            self.db.graph = BusinessProcessGraph(id="graph", name="")
        return self.db.graph

    def _node(self, node_id: str) -> Node:
        graph = self._graph()
        if node_id not in graph.nodes:
            raise ValueError(f"node not found: {node_id}")
        return graph.nodes[node_id]

    def _edge(self, edge_id: str) -> Edge:
        graph = self._graph()
        if edge_id not in graph.edges:
            raise ValueError(f"edge not found: {edge_id}")
        return graph.edges[edge_id]

    def _concept(self, concept_id: str) -> BusinessConcept:
        graph = self._graph()
        if concept_id not in graph.concepts:
            raise ValueError(f"concept not found: {concept_id}")
        return graph.concepts[concept_id]

    def _require_observation(self, observation_id: Optional[str]) -> None:
        """Reject a reference to an observation that does not exist."""
        if observation_id is None:
            return
        if not any(o.id == observation_id for o in self.db.observations):
            raise ValueError(f"observation not found: {observation_id}")

    def _require_evidence(self, evidence: Optional[list]) -> list[EvidenceRef]:
        """Validate a list of EvidenceRef dicts: shape, observation existence,
        and exact quote/occurrence spans in the immutable Observation."""
        refs: list[EvidenceRef] = []
        for raw in evidence or []:
            ref = _ev(raw)
            self._require_observation(ref.observation_id)
            obs = next(o for o in self.db.observations if o.id == ref.observation_id)
            if not obs.has_span(ref.quote, ref.occurrence):
                raise ValueError(
                    f"evidence quote {ref.quote!r} occurrence {ref.occurrence} "
                    f"is not an exact span of observation {ref.observation_id}"
                )
            refs.append(ref)
        return refs

    def _require_concepts(self, concept_ids: Optional[list[str]]) -> None:
        for cid in concept_ids or []:
            self._concept(cid)

    @staticmethod
    def _ref(
        concept_id: str, confidence: float = 1.0, evidence: Optional[list] = None
    ) -> ConceptRef:
        return ConceptRef(
            concept_id=concept_id,
            confidence=confidence,
            evidence=list(evidence or []),
        )

    # ------------------------------------------------------------- glossary

    @is_tool(ToolType.WRITE)
    def create_concept(
        self,
        concept_id: str,
        kind: str,
        label: str,
        description: Optional[str] = None,
        evidence: Optional[list] = None,
    ) -> str:
        """Create an Agent-local glossary concept (a business object / role /
        activity / system / condition / rationale).

        Create one concept when you first discover something the stakeholder
        names, choosing the kind that matches what it is:
        - activity: what is done (a step)
        - actor: who does it
        - system: which system/tool is used
        - data: a business object / artifact that flows in or out
        - condition: a branch condition / threshold
        - rationale: why a step is needed

        Concepts start as ``hypothesized``. Use the stakeholder's own wording
        as the label and cite the Observation span that introduced it.

        Args:
            concept_id: Your own identifier for this concept (reuse it
                consistently in node/edge references).
            kind: activity | actor | system | data | condition | rationale.
            label: The preferred label (stakeholder wording).
            description: Optional free-text description (never evaluated).
            evidence: Optional list of evidence refs
                [{"observation_id", "quote", "occurrence"}] supporting this
                concept (quote must be an exact span of that Observation).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if concept_id in graph.concepts:
            raise ValueError(f"concept already exists: {concept_id}")
        if kind not in (
            "activity",
            "actor",
            "system",
            "data",
            "condition",
            "rationale",
        ):
            raise ValueError(
                f"kind must be one of activity/actor/system/data/condition/rationale, "
                f"got {kind!r}"
            )
        evs = self._require_evidence(evidence)
        graph.concepts[concept_id] = BusinessConcept(
            id=concept_id,
            kind=kind,  # type: ignore[arg-type]
            preferred_label=label,
            description=description or "",
            terms=[ConceptTerm(text=label, evidence=list(evs))] if evs else [],
            validation_status="hypothesized",
            validation_evidence=list(evs),
        )
        return f"Created {kind} concept {concept_id} (label: {label!r})."

    @is_tool(ToolType.WRITE)
    def add_concept_term(
        self,
        concept_id: str,
        term: str,
        evidence: Optional[list] = None,
    ) -> str:
        """Add an observed term to an existing concept.

        When the stakeholder later uses a different wording that you decide
        refers to the same concept, record that wording as another term of the
        same concept (with the Observation span where it was said).

        Args:
            concept_id: The concept to extend.
            term: The observed wording (as the stakeholder said it).
            evidence: Optional evidence refs for this term.

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if any(t.text == term for t in concept.terms):
            existing = next(t for t in concept.terms if t.text == term)
            for ev in evs:
                if ev not in existing.evidence:
                    existing.evidence.append(ev)
            return f"Term {term!r} already recorded on {concept_id}."
        concept.terms.append(ConceptTerm(text=term, evidence=evs))
        return f"Added term {term!r} to concept {concept_id}."

    @is_tool(ToolType.WRITE)
    def update_concept_description(
        self,
        concept_id: str,
        description: str,
    ) -> str:
        """Update the free-text description of a concept.

        Descriptions are working notes for yourself — the evaluator never
        compares them to anything.

        Args:
            concept_id: The concept to update.
            description: The new description.

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        concept.description = description
        return f"Updated description of {concept_id}."

    @is_tool(ToolType.WRITE)
    def merge_concepts(
        self,
        target_concept_id: str,
        source_concept_ids: list[str],
    ) -> str:
        """Merge concepts you previously split by mistake.

        Only concepts of the SAME kind can be merged (merging different kinds
        would conflate distinct business concepts and is rejected). Every node
        / edge reference is re-pointed to ``target_concept_id`` and the source
        concepts' terms and validation evidence are folded into the target.
        The source concepts are removed.

        Args:
            target_concept_id: The concept to keep.
            source_concept_ids: The concepts to merge into it (removed).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        target = self._concept(target_concept_id)
        sources = [self._concept(cid) for cid in source_concept_ids]
        for src in sources:
            if src.kind != target.kind:
                raise ValueError(
                    f"cannot merge {src.id} (kind {src.kind}) into "
                    f"{target.id} (kind {target.kind}): kinds differ"
                )
            target.terms.extend(src.terms)
            for ev in src.validation_evidence:
                if ev not in target.validation_evidence:
                    target.validation_evidence.append(ev)
        for node in graph.nodes.values():
            for prop in (
                "activity",
                "actor",
                "system",
                "reads",
                "writes",
                "necessity_rationale",
            ):
                for ref in (
                    node.refs(prop)
                    if prop in ("reads", "writes")
                    else (
                        [node.activity]
                        if prop == "activity"
                        else ([getattr(node, prop)] if getattr(node, prop) else [])
                    )
                ):
                    if ref.concept_id in source_concept_ids:
                        ref.concept_id = target_concept_id
        for edge in graph.edges.values():
            if (
                edge.condition is not None
                and edge.condition.concept_id in source_concept_ids
            ):
                edge.condition.concept_id = target_concept_id
        for cid in source_concept_ids:
            del graph.concepts[cid]
        return (
            f"Merged {', '.join(source_concept_ids)} into {target_concept_id}; "
            "all references re-pointed."
        )

    @is_tool(ToolType.WRITE)
    def confirm_concept(
        self,
        concept_id: str,
        evidence: Optional[list] = None,
        partial: bool = False,
    ) -> str:
        """Confirm a concept with authentic stakeholder evidence.

        Confirmation must cite at least one authentic stakeholder Observation
        span (an EvidenceRef whose quote is an exact span of a recorded
        Observation). ``partial=True`` records ``partially_confirmed`` (some
        aspects still open); the default records ``confirmed``.

        Args:
            concept_id: The concept to confirm.
            evidence: Evidence refs confirming this concept (required).
            partial: If True, mark as partially_confirmed instead.

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if not evs:
            raise ValueError(
                f"confirm_concept requires at least one authentic evidence ref "
                f"for {concept_id}"
            )
        for ev in evs:
            if ev not in concept.validation_evidence:
                concept.validation_evidence.append(ev)
        concept.validation_status = "partially_confirmed" if partial else "confirmed"  # type: ignore[assignment]
        return f"Marked {concept_id} as {'partially_confirmed' if partial else 'confirmed'}."

    @is_tool(ToolType.WRITE)
    def mark_concept_unknown(self, concept_id: str) -> str:
        """Mark a concept as unknown (the stakeholder cannot clarify it)."""
        concept = self._concept(concept_id)
        concept.validation_status = "unknown"  # type: ignore[assignment]
        return f"Marked {concept_id} as unknown."

    @is_tool(ToolType.WRITE)
    def mark_concept_disputed(self, concept_id: str) -> str:
        """Mark a concept as disputed (conflicting stakeholder statements)."""
        concept = self._concept(concept_id)
        concept.validation_status = "disputed"  # type: ignore[assignment]
        return f"Marked {concept_id} as disputed."

    @is_tool(ToolType.READ)
    def list_concepts(self) -> str:
        """List your glossary concepts (ids, kinds, labels, terms, status).

        This is your working vocabulary — nothing here comes from any hidden
        ground truth.

        Returns:
            One line per concept.
        """
        graph = self._graph()
        if not graph.concepts:
            return "(no concepts yet — create one with create_concept)"
        lines = []
        for cid, concept in graph.concepts.items():
            terms = ", ".join(t.text for t in concept.terms) or "(no terms)"
            lines.append(
                f"{cid}: [{concept.kind}] {concept.preferred_label!r} "
                f"[{concept.validation_status}] [terms: {terms}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------- graph

    @is_tool(ToolType.WRITE)
    def start_inference(self, name: str = "") -> str:
        """Start building the inferred business process graph.

        This is destructive: it discards the previous inferred graph, glossary
        and any captured Observations, and resets the interview. The
        conversation ledger (the stakeholder's actual messages) is kept —
        observations are re-captured from it with ``observe_message`` /
        ``observe_latest_stakeholder_message``. Call it once at the beginning.

        Args:
            name: Optional name for the graph.

        Returns:
            A confirmation message.
        """
        self.db.graph = BusinessProcessGraph(id="graph", name=name)
        self.db.observations = []
        self.db.interview_complete = False
        self.db.summary = None
        return (
            "Inference started (previous graph, glossary and observations discarded)."
        )

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
            The observation id (use it in evidence refs on concepts, nodes and
            edges).
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
        activity: str,
        actor: Optional[str] = None,
        system: Optional[str] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        necessity_rationale: Optional[str] = None,
        evidence: Optional[list] = None,
    ) -> str:
        """Add a node to the inferred process graph.

        Only create a node when no existing node corresponds to the
        observation; otherwise update the existing node. Every property is a
        concept id from your glossary.

        Args:
            node_id: Your own identifier for this node.
            activity: Concept id (kind=activity) for what is done (required).
            actor: Concept id (kind=actor) for who does it (optional).
            system: Concept id (kind=system) for the system/tool (optional).
            reads: Concept ids (kind=data) this node reads (optional).
            writes: Concept ids (kind=data) this node writes (optional).
            necessity_rationale: Concept id (kind=rationale) for why the node
                is needed (optional; leave unset when unknown).
            evidence: Evidence refs supporting this node's activity (optional;
                add more on individual refs with update_node).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if node_id in graph.nodes:
            raise ValueError(f"node already exists: {node_id}")
        self._require_concepts([activity])
        self._require_concepts(reads)
        self._require_concepts(writes)
        if actor is not None:
            self._concept(actor)
        if system is not None:
            self._concept(system)
        if necessity_rationale is not None:
            self._concept(necessity_rationale)
        evs = self._require_evidence(evidence)
        graph.nodes[node_id] = Node(
            id=node_id,
            activity=self._ref(activity, evidence=evs),
            actor=self._ref(actor) if actor is not None else None,
            system=self._ref(system) if system is not None else None,
            reads=self._refs(reads),
            writes=self._refs(writes),
            necessity_rationale=(
                self._ref(necessity_rationale)
                if necessity_rationale is not None
                else None
            ),
        )
        return f"Added node {node_id}."

    @staticmethod
    def _refs(concept_ids: Optional[list[str]]) -> list[ConceptRef]:
        return [
            ConceptRef(concept_id=cid, confidence=1.0) for cid in (concept_ids or [])
        ]

    @is_tool(ToolType.WRITE)
    def update_node(
        self,
        node_id: str,
        activity: Optional[str] = None,
        actor: Optional[str] = None,
        system: Optional[str] = None,
        reads: Optional[list[str]] = None,
        writes: Optional[list[str]] = None,
        necessity_rationale: Optional[str] = None,
        evidence: Optional[list] = None,
        unset: Optional[list[str]] = None,
    ) -> str:
        """Update an existing node's property references.

        Args:
            node_id: The node to update.
            activity: New activity concept id (optional).
            actor: New actor concept id (optional).
            system: New system concept id (optional).
            reads: New read concept-id list (optional; replaces the list).
            writes: New write concept-id list (optional; replaces the list).
            necessity_rationale: New rationale concept id (optional).
            evidence: Evidence refs to append to the activity reference (optional).
            unset: List of property names to clear (actor/system/
                necessity_rationale/reads/writes) (optional).

        Returns:
            A confirmation message.
        """
        node = self._node(node_id)
        if unset:
            for prop in unset:
                if prop == "reads":
                    node.reads = []
                elif prop == "writes":
                    node.writes = []
                elif prop in ("actor", "system", "necessity_rationale"):
                    setattr(node, prop, None)
                else:
                    raise ValueError(f"cannot unset property {prop!r}")
        if activity is not None:
            self._concept(activity)
            node.activity = self._ref(activity)
        if actor is not None:
            self._concept(actor)
            node.actor = self._ref(actor)
        if system is not None:
            self._concept(system)
            node.system = self._ref(system)
        if necessity_rationale is not None:
            self._concept(necessity_rationale)
            node.necessity_rationale = self._ref(necessity_rationale)
        if reads is not None:
            self._require_concepts(reads)
            node.reads = self._refs(reads)
        if writes is not None:
            self._require_concepts(writes)
            node.writes = self._refs(writes)
        if evidence:
            evs = self._require_evidence(evidence)
            node.activity.evidence.extend(evs)
        return f"Updated node {node_id}."

    @is_tool(ToolType.WRITE)
    def remove_node(self, node_id: str) -> str:
        """Remove a node and its incident edges from the inferred graph.

        The graph is a working hypothesis, not an append-only record: you may
        remove a coarse placeholder node once you have decomposed it into more
        specific activities.

        This also removes every edge that connects to the node (incoming or
        outgoing), so no dangling edge is left behind. The stakeholder
        Observations themselves are kept.

        Args:
            node_id: The node to remove.

        Returns:
            A confirmation message listing what was removed.
        """
        graph = self._graph()
        if node_id not in graph.nodes:
            raise ValueError(f"node not found: {node_id}")
        removed_edges = sorted(
            eid
            for eid, e in graph.edges.items()
            if e.from_node == node_id or e.to_node == node_id
        )
        for eid in removed_edges:
            del graph.edges[eid]
        del graph.nodes[node_id]
        parts = [f"Removed node {node_id}."]
        if removed_edges:
            parts.append(f"Removed incident edges: {', '.join(removed_edges)}.")
        return " ".join(parts)

    @is_tool(ToolType.WRITE)
    def add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        condition: Optional[str] = None,
        evidence: Optional[list] = None,
    ) -> str:
        """Add a directed edge between two nodes.

        The edge's existence must be supported by stakeholder evidence
        (``evidence`` refs citing the Observation where the relation was
        stated). ``condition`` optionally references a condition concept
        (kind=condition) for a branch threshold.

        Args:
            edge_id: Your own identifier for this edge.
            from_node: Source node id.
            to_node: Destination node id.
            condition: Concept id (kind=condition) for the branch condition
                (optional).
            evidence: Evidence refs supporting this relation (required).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if edge_id in graph.edges:
            raise ValueError(f"edge already exists: {edge_id}")
        if from_node not in graph.nodes:
            raise ValueError(f"node not found: {from_node}")
        if to_node not in graph.nodes:
            raise ValueError(f"node not found: {to_node}")
        if condition is not None:
            self._concept(condition)
        evs = self._require_evidence(evidence)
        graph.edges[edge_id] = Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            condition=self._ref(condition) if condition is not None else None,
            evidence=evs,
        )
        return f"Added edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def update_edge(
        self,
        edge_id: str,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        condition: Optional[str] = None,
        unset_condition: bool = False,
        evidence: Optional[list] = None,
    ) -> str:
        """Update an edge's endpoints, condition or evidence.

        Args:
            edge_id: The edge to update.
            from_node: New source (optional).
            to_node: New destination (optional).
            condition: New condition concept id (optional).
            unset_condition: If True, remove the condition (optional).
            evidence: Evidence refs to append (optional).

        Returns:
            A confirmation message.
        """
        edge = self._edge(edge_id)
        graph = self._graph()
        if from_node is not None:
            if from_node not in graph.nodes:
                raise ValueError(f"node not found: {from_node}")
            edge.from_node = from_node
        if to_node is not None:
            if to_node not in graph.nodes:
                raise ValueError(f"node not found: {to_node}")
            edge.to_node = to_node
        if unset_condition:
            edge.condition = None
        elif condition is not None:
            self._concept(condition)
            edge.condition = self._ref(condition)
        if evidence:
            evs = self._require_evidence(evidence)
            edge.evidence.extend(evs)
        return f"Updated edge {edge_id}."

    @is_tool(ToolType.READ)
    def validate_graph(self) -> str:
        """Validate the inferred graph's **internal** structural consistency.

        Reports only self-consistency problems (dangling edges, unknown
        concept references, missing activity refs, ...). Cycles are VALID and
        are never reported as errors. This never references the hidden Ground
        Truth. Call this to review the final structure before
        ``finish_interview``.

        Returns:
            The validation result (or the list of structural errors).
        """
        graph = self._graph()
        if not graph.nodes:
            return "No graph nodes yet. Call start_inference to begin."
        errors = graph.structure_errors()
        if not errors:
            return "Graph is structurally valid (cycles are allowed)."
        return "Graph validation errors:\n- " + "\n- ".join(errors)

    @is_tool(ToolType.WRITE)
    def finish_interview(self, summary: Optional[str] = None) -> str:
        """Mark the interview complete.

        Refuses to finish a structurally invalid graph (dangling edges,
        unknown concept references, ...) and refuses while any **referenced**
        concept is still ``hypothesized``: every concept used by a node/edge
        must be resolved (confirmed / partially_confirmed / disputed /
        unknown) with ``confirm_concept``, ``mark_concept_unknown`` or
        ``mark_concept_disputed``. The reported errors are only about the
        graph's own internal consistency and glossary state — never about the
        hidden Ground Truth.

        Args:
            summary: Optional summary of what was captured.

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if not graph.nodes:
            raise ValueError(
                "Cannot finish: no graph has been built yet. Call start_inference first."
            )
        errors = graph.structure_errors()
        if errors:
            raise ValueError(
                "Cannot finish: graph is structurally invalid.\n- "
                + "\n- ".join(errors)
            )
        hypothesized = sorted(
            cid
            for cid in graph.referenced_concepts()
            if cid in graph.concepts
            and graph.concepts[cid].validation_status == "hypothesized"
        )
        if hypothesized:
            raise ValueError(
                "Cannot finish: referenced concepts are still hypothesized "
                "(confirm, mark unknown, or mark disputed first): "
                + ", ".join(hypothesized)
            )
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked complete."

    # ------------------------------------------------------------- assertions

    def _evaluate(self, sc) -> EvaluationResult:
        """Evaluate against the scenario truth+spec under the scenario's
        stakeholder visibility, using the hidden TruthClaim catalog, the
        private StakeholderFact catalog, and the private assertion sidecar
        ledger (all evaluator-only; never exposed to the Agent)."""

        return evaluate(
            self.db,
            sc.truth,
            sc.spec,
            sc.stakeholder,
            claims=sc.claims,
            facts=sc.facts,
            assertions=self.fact_ledger.assertions(),
        )

    def assert_finish_interview(self) -> bool:
        return self.db.interview_complete

    def assert_graph_reconstructed(self, scenario_id: str) -> bool:
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        return self._evaluate(sc).structural_pass

    def assert_necessity_handled(self, scenario_id: str) -> bool:
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        res = self._evaluate(sc)
        return res.rationale_correctness == 1.0

    def assert_evidence_backed(self, scenario_id: str) -> bool:
        """True if every asserted claim is traceable to a recorded Observation."""
        sc = get_scenario(scenario_id)
        if sc is None:
            return False
        return self._evaluate(sc).evidence_pass

    # ------------------------------------------------------------- diagnostics

    def get_eval_diagnostics(self, task: Optional[Task] = None) -> Optional[dict]:
        sc = get_scenario(task.id if task is not None else None)
        if sc is None:
            return None
        return self._evaluate(sc).model_dump(mode="json")
