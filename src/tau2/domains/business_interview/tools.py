"""Agent tools for the graph-context business_interview benchmark (v8).

Dialogue events: confirmations/unknown/disputed/terminology require private
semantic dialogue events, not ordinary workflow mentions.

The agent records **Observations** (immutable evidence), builds a private
**glossary** of typed ``BusinessConcept``\\ s (activity / actor / system / data /
condition / rationale), and constructs an inferred **BusinessProcessGraph**
whose node/edge properties reference glossary concepts. Every claim cites
**EvidenceRef**\\ s — spans of authentic Observations.

**Mention != terminology**: a concept's ``mentions`` are Observation spans the
Agent believes refer to it; an explicit terminology agreement with the
stakeholder is recorded separately (``record_terminology_agreement``).
Confirmation (``confirm_concept``) must represent actual stakeholder
confirmation of concept identity — evidence must correspond to a private
assertion of the concept's claims, and one evidence span may back at most one
concept (no bulk self-confirmation). ``unknown``/``disputed`` also require
stakeholder evidence. Structural ConceptKind rules are enforced: activity ->
activity, actor -> actor, system -> system, reads/writes -> data,
rationale -> rationale, edge condition -> condition.
"""

from typing import Optional

from pydantic import ValidationError

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.evaluation import EvaluationResult, evaluate
from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    StakeholderAssertionLedger,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import (
    BusinessConcept,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    Observation,
    TerminologyAgreement,
    spans_correspond,
)
from tau2.domains.business_interview.scenario import get_scenario
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

_KINDS = ("activity", "actor", "system", "data", "condition", "rationale")

# Structural ConceptKind rules per node property / edge property.
_PROPERTY_KIND: dict[str, str] = {
    "activity": "activity",
    "actor": "actor",
    "system": "system",
    "reads": "data",
    "writes": "data",
    "necessity_rationale": "rationale",
}


def _ev(value: dict) -> EvidenceRef:
    """Coerce one EvidenceRef dict, raising a clear error on bad shape."""
    try:
        return EvidenceRef(**value)
    except ValidationError as exc:
        raise ValueError(f"invalid evidence ref {value!r}: {exc}") from exc


def _resolve_span_text(
    text: str, quote: str, occurrence: int
) -> Optional[tuple[int, int]]:
    """Resolve (quote, occurrence) to a character span in ``text``, or None."""
    if not quote:
        return None
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start == -1:
            return None
    return (start, start + len(quote))


class InterviewTools(ToolKitBase):
    """Tools to infer the business process graph from stakeholder observations.

    ``assertion_ledger`` is the private assertion sidecar ledger, shared with
    the environment and the stakeholder simulator adapter. It is
    evaluator-only: no tool exposes it, and it is never serialized into the
    Agent-visible DB.
    """

    db: InterviewDB

    def __init__(
        self,
        db: InterviewDB,
        assertion_ledger: Optional[StakeholderAssertionLedger] = None,
    ) -> None:
        super().__init__(db)
        self.assertion_ledger = (
            assertion_ledger
            if assertion_ledger is not None
            else StakeholderAssertionLedger()
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

    def _require_kind(self, concept_id: str, expected_kind: str, where: str) -> None:
        concept = self._concept(concept_id)
        if concept.kind != expected_kind:
            raise ValueError(
                f"{where}: concept {concept_id!r} has kind {concept.kind!r} but "
                f"requires kind {expected_kind!r}"
            )

    @staticmethod
    def _ref(
        concept_id: str, confidence: float = 1.0, evidence: Optional[list] = None
    ) -> ConceptRef:
        return ConceptRef(
            concept_id=concept_id,
            confidence=confidence,
            evidence=list(evidence or []),
        )

    @staticmethod
    def _refs(concept_ids: Optional[list[str]]) -> list[ConceptRef]:
        return [
            ConceptRef(concept_id=cid, confidence=1.0) for cid in (concept_ids or [])
        ]

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
        """Create an Agent-local glossary concept (a business thing of one kind).

        Choose the kind that matches what the thing is:
        - activity: what is done (a step)
        - actor: who does it
        - system: which system/tool is used
        - data: a business object / artifact that flows in or out
        - condition: a branch condition / threshold
        - rationale: why a step is needed

        Concepts start as ``hypothesized``. ``label`` is your working display
        label (never evaluated). ``evidence`` spans are recorded as
        **mentions** — Observation spans you believe refer to this concept; a
        mention is not a terminology agreement.

        Args:
            concept_id: Your own identifier for this concept (reuse it
                consistently in node/edge references).
            kind: activity | actor | system | data | condition | rationale.
            label: The display label (your working text).
            description: Optional free-text description (never evaluated).
            evidence: Optional mention spans
                [{"observation_id", "quote", "occurrence"}].

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if concept_id in graph.concepts:
            raise ValueError(f"concept already exists: {concept_id}")
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {'/'.join(_KINDS)}, got {kind!r}")
        evs = self._require_evidence(evidence)
        graph.concepts[concept_id] = BusinessConcept(
            id=concept_id,
            kind=kind,  # type: ignore[arg-type]
            display_label=label,
            description=description or "",
            mentions=list(evs),
            validation_status="hypothesized",
            validation_evidence=[],
        )
        return f"Created {kind} concept {concept_id} (label: {label!r})."

    @is_tool(ToolType.WRITE)
    def add_concept_mention(
        self,
        concept_id: str,
        evidence: list,
    ) -> str:
        """Record an Observation span as a mention of a concept.

        A mention means only: you believe this span refers to this local
        concept. It does NOT establish terminology — record an explicit
        terminology agreement separately when the stakeholder confirms a term.

        Args:
            concept_id: The concept the span refers to.
            evidence: Mention spans [{"observation_id", "quote", "occurrence"}].

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        for ev in evs:
            if ev not in concept.mentions:
                concept.mentions.append(ev)
        return f"Recorded {len(evs)} mention(s) on {concept_id}."

    @is_tool(ToolType.WRITE)
    def record_terminology_agreement(
        self,
        concept_id: str,
        term: str,
        evidence: list,
    ) -> str:
        """Record an explicit terminology agreement: you proposed ``term`` for
        this concept and the stakeholder explicitly confirmed it.

        Record this ONLY when the stakeholder explicitly agreed to the term in
        the interview. The evidence must cite the Observation span where the
        stakeholder performed that agreement (a private terminology-
        confirmation event). A mere authentic mention of the term in ordinary
        workflow speech is NOT an agreement and cannot authorize this call.

        Args:
            concept_id: The concept the term refers to.
            term: The agreed term (must match the proposed term the
                stakeholder confirmed).
            evidence: Evidence spans of the confirmation (required).

        Returns:
            A confirmation message.
        """
        self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if not evs:
            raise ValueError(
                f"record_terminology_agreement requires evidence spans for {concept_id}"
            )
        matches = self._terminology_matches(evs, term)
        if not matches:
            raise ValueError(
                f"record_terminology_agreement for {concept_id}: evidence does "
                f"not correspond to a private terminology-confirmation event "
                f"for proposed term {term!r} — an ordinary mention is not an "
                f"agreement"
            )
        graph = self._graph()
        graph.terminology_agreements.append(
            TerminologyAgreement(
                concept_id=concept_id,
                term=term,
                stakeholder_id="stakeholder",
                evidence=evs,
            )
        )
        return f"Recorded terminology agreement: {term!r} for {concept_id}."

    @is_tool(ToolType.WRITE)
    def update_concept_description(
        self,
        concept_id: str,
        description: str,
    ) -> str:
        """Update the free-text description of a concept (working notes only;
        never evaluated)."""
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

        Only concepts of the SAME kind can be merged. Every node/edge
        reference is re-pointed to ``target_concept_id`` and the source
        concepts' mentions, validation evidence and terminology agreements are
        folded into the target. The source concepts are removed.

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
            for ev in src.mentions:
                if ev not in target.mentions:
                    target.mentions.append(ev)
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
        for agreement in graph.terminology_agreements:
            if agreement.concept_id in source_concept_ids:
                agreement.concept_id = target_concept_id
        for cid in source_concept_ids:
            del graph.concepts[cid]
        return (
            f"Merged {', '.join(source_concept_ids)} into {target_concept_id}; "
            "all references re-pointed."
        )

    def _alignment_matches(
        self, evidence: list[EvidenceRef], acts: set[str]
    ) -> list[tuple[str, ConceptAlignmentAssertion]]:
        """Concept-alignment events (act in ``acts``) whose spans correspond
        (containment) to the given evidence spans, as
        ``[(observation_id, event)]``. Deterministic."""
        matches: list[tuple[str, ConceptAlignmentAssertion]] = []
        events_by_turn = self.assertion_ledger.alignments()
        for ev in evidence:
            obs = next(
                (o for o in self.db.observations if o.id == ev.observation_id), None
            )
            if obs is None:
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                continue
            for event in events_by_turn.get(obs.turn, []):
                if event.act not in acts:
                    continue
                event_span = _resolve_span_text(obs.text, event.quote, event.occurrence)
                if event_span is not None and spans_correspond(ev_span, event_span):
                    matches.append((ev.observation_id, event))
        return matches

    def _terminology_matches(
        self, evidence: list[EvidenceRef], term: str
    ) -> list[tuple[str, TerminologyConfirmation]]:
        """Terminology-confirmation events whose ``proposed_term`` equals
        ``term`` and whose spans correspond to the given evidence spans, as
        ``[(observation_id, event)]``. Deterministic."""
        matches: list[tuple[str, TerminologyConfirmation]] = []
        events_by_turn = self.assertion_ledger.terminology()
        for ev in evidence:
            obs = next(
                (o for o in self.db.observations if o.id == ev.observation_id), None
            )
            if obs is None:
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                continue
            for event in events_by_turn.get(obs.turn, []):
                if event.proposed_term != term:
                    continue
                event_span = _resolve_span_text(obs.text, event.quote, event.occurrence)
                if event_span is not None and spans_correspond(ev_span, event_span):
                    matches.append((ev.observation_id, event))
        return matches

    @is_tool(ToolType.WRITE)
    def confirm_concept(
        self,
        concept_id: str,
        evidence: list,
        partial: bool = False,
    ) -> str:
        """Confirm a concept with genuine stakeholder evidence.

        Confirmation must represent actual stakeholder confirmation of the
        concept's identity: the evidence must correspond to a private
        concept-alignment event (act ``confirm``, or ``partial`` when
        ``partial=True``) — the stakeholder actually performed that dialogue
        act in that message. A mere mention in ordinary workflow speech is
        NOT confirmation. A span may back at most one concept (no bulk
        self-confirmation).

        Args:
            concept_id: The concept to confirm.
            evidence: Evidence spans of the confirmation (required).
            partial: If True, record partially_confirmed instead.

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
        # anti-bulk: a span may back at most one concept's validation evidence
        graph = self._graph()
        for ev in evs:
            for other in graph.concepts.values():
                if other.id == concept_id:
                    continue
                if ev in other.validation_evidence:
                    raise ValueError(
                        f"evidence span {ev.observation_id}:{ev.quote!r} "
                        f"already backs {other.id}; a span cannot confirm "
                        f"several concepts"
                    )
        acts = {"partial"} if partial else {"confirm"}
        if not self._alignment_matches(evs, acts):
            raise ValueError(
                f"confirm_concept for {concept_id}: evidence does not correspond "
                f"to a private concept-alignment event (act="
                f"{'partial' if partial else 'confirm'}) — mention-only speech "
                f"is not confirmation"
            )
        for ev in evs:
            if ev not in concept.validation_evidence:
                concept.validation_evidence.append(ev)
        concept.validation_status = "partially_confirmed" if partial else "confirmed"  # type: ignore[assignment]
        return (
            f"Marked {concept_id} as "
            f"{'partially_confirmed' if partial else 'confirmed'}."
        )

    @is_tool(ToolType.WRITE)
    def mark_concept_unknown(self, concept_id: str, evidence: list) -> str:
        """Mark a concept as unknown with stakeholder evidence.

        The evidence must correspond to a private concept-alignment event with
        act ``unknown`` (the stakeholder explicitly said they do not know /
        could not assert the concept's identity).

        Args:
            concept_id: The concept to mark unknown.
            evidence: Evidence spans (required).

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if not evs:
            raise ValueError(f"mark_concept_unknown requires evidence for {concept_id}")
        if not self._alignment_matches(evs, {"unknown"}):
            raise ValueError(
                f"mark_concept_unknown for {concept_id}: evidence does not "
                f"correspond to a private concept-alignment event (act=unknown)"
            )
        # a status change replaces the concept's validation evidence
        concept.validation_evidence = list(evs)
        concept.validation_status = "unknown"  # type: ignore[assignment]
        return f"Marked {concept_id} as unknown."

    @is_tool(ToolType.WRITE)
    def mark_concept_disputed(self, concept_id: str, evidence: list) -> str:
        """Mark a concept as disputed with stakeholder evidence.

        The evidence must correspond to private concept-alignment events with
        act ``dispute`` (the stakeholder contradicted the proposed identity),
        from at least two distinct Observations.

        Args:
            concept_id: The concept to mark disputed.
            evidence: Evidence spans from >= 2 distinct Observations
                (required).

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if not evs:
            raise ValueError(
                f"mark_concept_disputed requires evidence for {concept_id}"
            )
        matches = self._alignment_matches(evs, {"dispute"})
        if len({oid for oid, _ in matches}) < 2:
            raise ValueError(
                f"mark_concept_disputed for {concept_id}: evidence must "
                f"correspond to private concept-alignment events (act=dispute) "
                f"from at least two distinct Observations"
            )
        # a status change replaces the concept's validation evidence
        concept.validation_evidence = list(evs)
        concept.validation_status = "disputed"  # type: ignore[assignment]
        return f"Marked {concept_id} as disputed."

    @is_tool(ToolType.READ)
    def list_concepts(self) -> str:
        """List your glossary concepts (ids, kinds, labels, mentions, status)."""
        graph = self._graph()
        if not graph.concepts:
            return "(no concepts yet — create one with create_concept)"
        lines = []
        for cid, concept in graph.concepts.items():
            mentions = len(concept.mentions)
            agreements = [
                a.term for a in graph.terminology_agreements if a.concept_id == cid
            ]
            extra = ""
            if agreements:
                extra = f" [agreed terms: {', '.join(agreements)}]"
            lines.append(
                f"{cid}: [{concept.kind}] {concept.display_label!r} "
                f"[{concept.validation_status}] "
                f"[{mentions} mention(s)]{extra}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------- graph

    @is_tool(ToolType.WRITE)
    def start_inference(self, name: str = "") -> str:
        """Start (or restart) building the inferred business process graph.

        Resets the inferred graph, the glossary and the completion state.
        Already captured Observations and the conversation ledger are
        IMMUTABLE primary evidence and are preserved — you can keep citing
        them as evidence after a restart.
        """
        self.db.graph = BusinessProcessGraph(id="graph", name=name)
        self.db.interview_complete = False
        self.db.summary = None
        return (
            "Inference (re)started; previous graph and glossary discarded, "
            "captured Observations and the conversation ledger preserved."
        )

    def _stakeholder_entries(self) -> list[tuple[str, int, str]]:
        entries = []
        counter = 0
        for i, msg in enumerate(self.db.messages):
            if msg.get("role") == "user":
                counter += 1
                entries.append((f"sm_{counter}", i, msg.get("content") or ""))
        return entries

    def _capture_user_message(self, turn: int, content: str) -> str:
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

        Returns the observation id (use it in evidence refs on concepts, nodes
        and edges). Idempotent; only user messages can be observed.
        """
        for sm_id, turn, content in self._stakeholder_entries():
            if sm_id == message_id:
                return self._capture_user_message(turn, content)
        raise ValueError(f"no stakeholder message with id {message_id!r}")

    @is_tool(ToolType.READ)
    def observe_latest_stakeholder_message(self) -> str:
        entries = self._stakeholder_entries()
        if not entries:
            raise ValueError("no stakeholder (user) message has been recorded yet")
        return entries[-1][0]

    @is_tool(ToolType.READ)
    def list_stakeholder_messages(self) -> str:
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

        Every property is a concept id from your glossary and its kind is
        enforced: activity -> activity, actor -> actor, system -> system,
        reads/writes -> data, necessity_rationale -> rationale.

        Args:
            node_id: Your own identifier for this node.
            activity: Concept id (kind=activity) for what is done (required).
            actor: Concept id (kind=actor) for who does it (optional).
            system: Concept id (kind=system) for the system/tool (optional).
            reads: Concept ids (kind=data) this node reads (optional).
            writes: Concept ids (kind=data) this node writes (optional).
            necessity_rationale: Concept id (kind=rationale) (optional).
            evidence: Evidence refs supporting this node's activity (optional).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if node_id in graph.nodes:
            raise ValueError(f"node already exists: {node_id}")
        self._require_kind(activity, "activity", "add_node activity")
        for cid in reads or []:
            self._require_kind(cid, "data", f"add_node reads[{cid}]")
        for cid in writes or []:
            self._require_kind(cid, "data", f"add_node writes[{cid}]")
        if actor is not None:
            self._require_kind(actor, "actor", "add_node actor")
        if system is not None:
            self._require_kind(system, "system", "add_node system")
        if necessity_rationale is not None:
            self._require_kind(
                necessity_rationale, "rationale", "add_node necessity_rationale"
            )
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
        """Update an existing node's property references (kinds enforced)."""
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
            self._require_kind(activity, "activity", "update_node activity")
            node.activity = self._ref(activity)
        if actor is not None:
            self._require_kind(actor, "actor", "update_node actor")
            node.actor = self._ref(actor)
        if system is not None:
            self._require_kind(system, "system", "update_node system")
            node.system = self._ref(system)
        if necessity_rationale is not None:
            self._require_kind(
                necessity_rationale, "rationale", "update_node necessity_rationale"
            )
            node.necessity_rationale = self._ref(necessity_rationale)
        if reads is not None:
            for cid in reads:
                self._require_kind(cid, "data", "update_node reads")
            node.reads = self._refs(reads)
        if writes is not None:
            for cid in writes:
                self._require_kind(cid, "data", "update_node writes")
            node.writes = self._refs(writes)
        if evidence:
            evs = self._require_evidence(evidence)
            node.activity.evidence.extend(evs)
        return f"Updated node {node_id}."

    @is_tool(ToolType.WRITE)
    def remove_node(self, node_id: str) -> str:
        """Remove a node and its incident edges from the inferred graph."""
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

        The edge's existence must be supported by stakeholder evidence.
        ``condition`` must reference a condition concept (kind=condition).

        Args:
            edge_id: Your own identifier for this edge.
            from_node: Source node id.
            to_node: Destination node id.
            condition: Concept id (kind=condition) (optional).
            evidence: Evidence refs supporting this relation.

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
            self._require_kind(condition, "condition", "add_edge condition")
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
        """Update an edge's endpoints, condition or evidence."""
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
            self._require_kind(condition, "condition", "update_edge condition")
            edge.condition = self._ref(condition)
        if evidence:
            evs = self._require_evidence(evidence)
            edge.evidence.extend(evs)
        return f"Updated edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def set_graph_endpoints(
        self,
        start_node_id: Optional[str] = None,
        end_node_ids: Optional[list[str]] = None,
    ) -> str:
        """Declare the graph's start and end node(s).

        The start node is the position where the process begins (it may still
        receive incoming edges when the process loops). End nodes are terminal
        positions. Declare both before finishing the interview.

        Args:
            start_node_id: The start node id.
            end_node_ids: The end node ids.

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if start_node_id is not None:
            if start_node_id not in graph.nodes:
                raise ValueError(f"node not found: {start_node_id}")
            graph.start_node_id = start_node_id
        if end_node_ids is not None:
            for eid in end_node_ids:
                if eid not in graph.nodes:
                    raise ValueError(f"node not found: {eid}")
            graph.end_node_ids = list(end_node_ids)
        return "Set graph endpoints."

    @is_tool(ToolType.READ)
    def validate_graph(self) -> str:
        """Validate the inferred graph's **internal** structural consistency.

        Cycles are VALID and never reported as errors. This never references
        the hidden Ground Truth.
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

        Refuses a structurally invalid graph, missing declared endpoints, or
        any **referenced** concept still ``hypothesized``. Completing the
        interview successfully terminates the episode immediately.

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
        if graph.start_node_id is None:
            raise ValueError(
                "Cannot finish: declare the start node with set_graph_endpoints."
            )
        if not graph.end_node_ids:
            raise ValueError(
                "Cannot finish: declare at least one end node with set_graph_endpoints."
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
        stakeholder visibility, using the hidden TruthClaim catalog and the
        private assertion/dialogue-event sidecar ledger (evaluator-only)."""

        return evaluate(
            self.db,
            sc.truth,
            sc.spec,
            sc.stakeholder,
            claims=sc.claims,
            assertions=self.assertion_ledger.assertions(),
            alignments=self.assertion_ledger.alignments(),
            terminology=self.assertion_ledger.terminology(),
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
        return self._evaluate(sc).rationale_correctness == 1.0

    def assert_evidence_backed(self, scenario_id: str) -> bool:
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

