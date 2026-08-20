"""Agent tools for the graph-native business_interview benchmark (v10).

The agent records **Observations** (immutable evidence), builds a private
glossary of typed ``AgentConcept``\\ s, and constructs an inferred
``AgentGraph`` whose node/edge property references each carry their own
``EvidenceRef``\\ s.

**Mention != evidence != validation.**
- ``AgentConcept.mentions`` are Observation spans the Agent interprets as
  referring to the concept (identity may use them; property scoring never
  does).
- Every graph property reference carries its own property evidence; the
  evaluator scores properties from property evidence ONLY.
- Validation statuses (grounded / confirmed / partially_confirmed / unknown /
  disputed) are backed by explicit validation/dialogue evidence:
  ``ground_concept`` requires evidence corresponding to private semantic
  annotations; ``confirm_concept`` / ``mark_concept_unknown`` /
  ``mark_concept_disputed`` require private dialogue events (act confirm /
  partial / unknown / dispute); ``record_terminology_agreement`` requires a
  private terminology-confirmation event with the same proposed term. An
  ordinary workflow mention is never enough.

``start_inference`` may reset the AgentGraph, the glossary and the
completion state, but preserves already captured Observations and the
conversation ledger.
"""

from typing import Optional

from pydantic import ValidationError

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.evaluation import (
    EvaluationResult,
    EvaluationSpec,
    evaluate,
)
from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    SemanticAnnotation,
    SemanticLedger,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import (
    AgentConcept,
    AgentGraph,
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

    ``assertion_ledger`` is the private semantic sidecar ledger, shared with
    the environment and the stakeholder simulator adapter. It is
    evaluator-only: no tool exposes it, and it is never serialized into the
    Agent-visible DB.
    """

    db: InterviewDB

    def __init__(
        self,
        db: InterviewDB,
        assertion_ledger: Optional[SemanticLedger] = None,
    ) -> None:
        super().__init__(db)
        self.assertion_ledger = (
            assertion_ledger if assertion_ledger is not None else SemanticLedger()
        )

    # ------------------------------------------------------------- helpers

    def _graph(self) -> AgentGraph:
        if self.db.graph is None:
            self.db.graph = AgentGraph(id="graph", name="")
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

    def _concept(self, concept_id: str) -> AgentConcept:
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
            ConceptRef(concept_id=cid, confidence=1.0) for cid in concept_ids or []
        ]

    def _ref_from_arg(
        self, arg, where: str, expected_kind: str, default_evidence: Optional[list]
    ) -> ConceptRef:
        """Build a ConceptRef from ``str | {"concept_id", "evidence"} | None``.

        The reference carries its OWN evidence: either the dict's ``evidence``
        or the call-level ``default_evidence`` (the node ``evidence``
        shorthand applies to the activity only).
        """
        if arg is None:
            raise ValueError(f"{where}: missing concept reference")
        if isinstance(arg, dict):
            cid = str(arg.get("concept_id") or "")
            evidence = self._require_evidence(arg.get("evidence"))
        else:
            cid = str(arg)
            evidence = (
                self._require_evidence(default_evidence)
                if where == "add_node activity" or where == "update_node activity"
                else []
            )
        self._require_kind(cid, expected_kind, where)
        return self._ref(cid, evidence=evidence)

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
        """Create a glossary concept (a business thing of one kind).

        ``evidence`` (optional) seeds the concept's mentions: Observation
        spans you interpret as referring to this concept. A mention is not
        graph-property evidence and not a terminology agreement.

        Args:
            concept_id: Your own identifier for this concept.
            kind: activity | actor | system | data | condition | rationale.
            label: Your working label for the concept (never evaluated).
            description: Optional working notes (never evaluated).
            evidence: Optional mention spans.

        Returns:
            A confirmation message.
        """
        if kind not in _KINDS:
            raise ValueError(f"invalid concept kind {kind!r}; must be one of {_KINDS}")
        graph = self._graph()
        if concept_id in graph.concepts:
            raise ValueError(f"concept already exists: {concept_id}")
        evs = self._require_evidence(evidence)
        graph.concepts[concept_id] = AgentConcept(
            id=concept_id,
            kind=kind,  # type: ignore[arg-type]
            display_label=label,
            description=description or "",
            mentions=evs,
        )
        return f"Created {kind} concept {concept_id} (label: {label!r})."

    @is_tool(ToolType.WRITE)
    def add_concept_mention(
        self,
        concept_id: str,
        evidence: list,
    ) -> str:
        """Record Observation spans you interpret as mentions of a concept.

        A mention means only: you believe this span refers to this local
        concept. It is NOT graph-property evidence and NOT terminology.

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

    def _annotation_matches(self, evidence: list[EvidenceRef]) -> bool:
        """True when any evidence span corresponds to a private semantic
        annotation (deterministic)."""
        annotations_by_turn = self.assertion_ledger.annotations()
        for ev in evidence:
            obs = next(
                (o for o in self.db.observations if o.id == ev.observation_id), None
            )
            if obs is None:
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                continue
            for annotation in annotations_by_turn.get(obs.turn, []):
                ann_span = _resolve_span_text(
                    obs.text, annotation.quote, annotation.occurrence
                )
                if ann_span is not None and spans_correspond(ev_span, ann_span):
                    return True
        return False

    def _alignment_matches(
        self, evidence: list[EvidenceRef], acts: set[str]
    ) -> list[tuple[str, ConceptAlignmentAssertion]]:
        """Concept-alignment events (act in ``acts``) whose spans correspond
        to the given evidence spans, as [(observation_id, event)]."""
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
        ``term`` and whose spans correspond to the given evidence spans."""
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

    def _check_no_bulk_validation(self, concept_id: str, evs: list[EvidenceRef]) -> None:
        """A span may back at most one concept's validation evidence."""
        graph = self._graph()
        for ev in evs:
            for other in graph.concepts.values():
                if other.id == concept_id:
                    continue
                if ev in other.validation_evidence:
                    raise ValueError(
                        f"evidence span {ev.observation_id}:{ev.quote!r} "
                        f"already backs {other.id}; a span cannot validate "
                        f"several concepts"
                    )

    @is_tool(ToolType.WRITE)
    def ground_concept(self, concept_id: str, evidence: list) -> str:
        """Mark a concept as grounded with authentic provenance.

        Grounding means the stakeholder's own speech (its private semantic
        annotations) supports this concept's identity — no confirmation
        dialogue is needed. The evidence must correspond to private semantic
        annotations (an ordinary invented span is not enough).

        Args:
            concept_id: The concept to ground.
            evidence: Evidence spans (required).

        Returns:
            A confirmation message.
        """
        concept = self._concept(concept_id)
        evs = self._require_evidence(evidence)
        if not evs:
            raise ValueError(f"ground_concept requires evidence for {concept_id}")
        self._check_no_bulk_validation(concept_id, evs)
        if not self._annotation_matches(evs):
            raise ValueError(
                f"ground_concept for {concept_id}: evidence does not "
                f"correspond to any private semantic annotation — grounding "
                f"needs the stakeholder's own speech"
            )
        for ev in evs:
            if ev not in concept.validation_evidence:
                concept.validation_evidence.append(ev)
        concept.validation_status = "grounded"  # type: ignore[assignment]
        return f"Marked {concept_id} as grounded."

    @is_tool(ToolType.WRITE)
    def confirm_concept(
        self,
        concept_id: str,
        evidence: list,
        partial: bool = False,
    ) -> str:
        """Confirm a concept's identity with genuine stakeholder evidence.

        Confirmation requires a private concept-alignment dialogue event
        (act ``confirm``, or ``partial`` when ``partial=True``) at a
        corresponding span — the stakeholder actually answered an identity
        question. A mere mention in ordinary workflow speech is NOT
        confirmation.

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
        self._check_no_bulk_validation(concept_id, evs)
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

        The evidence must correspond to a private concept-alignment event
        with act ``unknown`` (the stakeholder explicitly said they do not
        know / could not assert the concept's identity).

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
        self._check_no_bulk_validation(concept_id, evs)
        if not self._alignment_matches(evs, {"unknown"}):
            raise ValueError(
                f"mark_concept_unknown for {concept_id}: evidence does not "
                f"correspond to a private concept-alignment event (act=unknown)"
            )
        concept.validation_evidence = list(evs)
        concept.validation_status = "unknown"  # type: ignore[assignment]
        return f"Marked {concept_id} as unknown."

    @is_tool(ToolType.WRITE)
    def mark_concept_disputed(self, concept_id: str, evidence: list) -> str:
        """Mark a concept as disputed with stakeholder evidence.

        The evidence must correspond to private concept-alignment events with
        act ``dispute``, from at least two distinct Observations.

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
        concept.validation_evidence = list(evs)
        concept.validation_status = "disputed"  # type: ignore[assignment]
        return f"Marked {concept_id} as disputed."

    @is_tool(ToolType.WRITE)
    def record_terminology_agreement(
        self,
        concept_id: str,
        term: str,
        evidence: list,
    ) -> str:
        """Record an explicit terminology agreement: you proposed ``term`` for
        this concept and the stakeholder explicitly confirmed it.

        The evidence must correspond to a private terminology-confirmation
        event with the same proposed term. A mere authentic mention of the
        term in ordinary workflow speech is NOT an agreement.

        Args:
            concept_id: The concept the term refers to.
            term: The agreed term.
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
        if not self._terminology_matches(evs, term):
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
        concepts' mentions, validation evidence and terminology agreements
        are folded into the target. The source concepts are removed.

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
        self.db.graph = AgentGraph(id="graph", name=name)
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

    def _ref_from_prop_arg(self, arg, where: str, expected_kind: str) -> ConceptRef:
        """Property arg: str | {"concept_id", "evidence"} | None."""
        if arg is None:
            raise ValueError(f"{where}: missing concept reference")
        if isinstance(arg, dict):
            cid = str(arg.get("concept_id") or "")
            evidence = self._require_evidence(arg.get("evidence"))
        else:
            cid = str(arg)
            evidence = []
        self._require_kind(cid, expected_kind, where)
        return self._ref(cid, evidence=evidence)

    def _refs_from_prop_list(
        self, args: Optional[list], where: str, expected_kind: str
    ) -> list[ConceptRef]:
        """reads/writes arg: list of str | {"concept_id", "evidence"}."""
        refs: list[ConceptRef] = []
        for arg in args or []:
            refs.append(self._ref_from_prop_arg(arg, where, expected_kind))
        return refs

    @is_tool(ToolType.WRITE)
    def add_node(
        self,
        node_id: str,
        activity,
        actor=None,
        system=None,
        reads=None,
        writes=None,
        necessity_rationale=None,
        evidence: Optional[list] = None,
    ) -> str:
        """Add a node to the inferred process graph.

        Every property reference carries its OWN evidence: pass a property
        as ``concept_id`` or as ``{"concept_id": ..., "evidence": [...]}``.
        The call-level ``evidence`` list is the shorthand for the activity's
        evidence (when ``activity`` is a plain concept id). Concept kinds are
        enforced: activity->activity, actor->actor, system->system,
        reads/writes->data, necessity_rationale->rationale.

        Args:
            node_id: Your own identifier for this node.
            activity: Concept id or {concept_id, evidence} (kind=activity).
            actor: Concept id or {concept_id, evidence} (kind=actor).
            system: Concept id or {concept_id, evidence} (kind=system).
            reads: List of data concept ids / {concept_id, evidence}.
            writes: List of data concept ids / {concept_id, evidence}.
            necessity_rationale: Concept id or {concept_id, evidence}
                (kind=rationale).
            evidence: Activity evidence shorthand (optional).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if node_id in graph.nodes:
            raise ValueError(f"node already exists: {node_id}")
        activity_ref = self._ref_from_prop_arg(activity, "add_node activity", "activity")
        if evidence and not activity_ref.evidence:
            activity_ref.evidence = self._require_evidence(evidence)
        graph.nodes[node_id] = Node(
            id=node_id,
            activity=activity_ref,
            actor=(
                self._ref_from_prop_arg(actor, "add_node actor", "actor")
                if actor is not None
                else None
            ),
            system=(
                self._ref_from_prop_arg(system, "add_node system", "system")
                if system is not None
                else None
            ),
            reads=self._refs_from_prop_list(reads, "add_node reads", "data"),
            writes=self._refs_from_prop_list(writes, "add_node writes", "data"),
            necessity_rationale=(
                self._ref_from_prop_arg(
                    necessity_rationale,
                    "add_node necessity_rationale",
                    "rationale",
                )
                if necessity_rationale is not None
                else None
            ),
        )
        return f"Added node {node_id}."

    @is_tool(ToolType.WRITE)
    def update_node(
        self,
        node_id: str,
        activity=None,
        actor=None,
        system=None,
        reads=None,
        writes=None,
        necessity_rationale=None,
        evidence: Optional[list] = None,
        unset: Optional[list[str]] = None,
    ) -> str:
        """Update an existing node's property references (kinds enforced).

        Each property accepts a concept id or ``{"concept_id", "evidence"}``
        so the reference carries its own evidence.
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
            node.activity = self._ref_from_prop_arg(activity, "update_node activity", "activity")
            if evidence and not node.activity.evidence:
                node.activity.evidence = self._require_evidence(evidence)
        if actor is not None:
            node.actor = self._ref_from_prop_arg(actor, "update_node actor", "actor")
        if system is not None:
            node.system = self._ref_from_prop_arg(system, "update_node system", "system")
        if necessity_rationale is not None:
            node.necessity_rationale = self._ref_from_prop_arg(
                necessity_rationale, "update_node necessity_rationale", "rationale"
            )
        if reads is not None:
            node.reads = self._refs_from_prop_list(reads, "update_node reads", "data")
        if writes is not None:
            node.writes = self._refs_from_prop_list(writes, "update_node writes", "data")
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
        condition=None,
        evidence: Optional[list] = None,
    ) -> str:
        """Add a directed edge between two nodes.

        The edge's existence must be supported by stakeholder evidence.
        ``condition`` accepts a condition concept id or
        ``{"concept_id", "evidence"}`` so it carries its own evidence.

        Args:
            edge_id: Your own identifier for this edge.
            from_node: Source node id.
            to_node: Destination node id.
            condition: Condition concept id / {concept_id, evidence}.
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
        cond_ref = None
        if condition is not None:
            cond_ref = self._ref_from_prop_arg(condition, "add_edge condition", "condition")
        evs = self._require_evidence(evidence)
        graph.edges[edge_id] = Edge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            condition=cond_ref,
            evidence=evs,
        )
        return f"Added edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def update_edge(
        self,
        edge_id: str,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        condition=None,
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
            edge.condition = self._ref_from_prop_arg(condition, "update_edge condition", "condition")
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
        any **referenced** concept still ``hypothesized`` (concepts should
        normally be at least ``grounded`` — explicit confirmation is not
        required for every concept). Completing the interview successfully
        terminates the episode immediately.

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
                "(ground them with authentic evidence, or confirm / mark "
                "unknown / mark disputed): "
                + ", ".join(hypothesized)
            )
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked complete."

    # ------------------------------------------------------------- assertions

    def _evaluate(self, sc) -> EvaluationResult:
        """Evaluate the AgentGraph against the scenario's StakeholderKnowledge
        using the private semantic sidecar ledger (evaluator-only)."""
        return evaluate(
            self.db,
            sc.knowledge,
            EvaluationSpec(),
            sc.stakeholder,
            truth=sc.truth,
            annotations=self.assertion_ledger.annotations(),
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
