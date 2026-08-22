"""Agent tools for the graph-native business_interview benchmark (v11).

The agent records **Observations** (immutable evidence), builds a private
glossary of typed ``AgentConcept``\\ s, and constructs an inferred
``AgentGraph`` whose node/edge property references each carry their own
``EvidenceRef``\\ s. Every Agent slot is one of the FOUR epistemic states:
``UNSET`` (not investigated / no conclusion) / ``ConceptRef`` (known value)
/ ``ABSENT`` (explicitly established absent, with evidence) /
``DONT_KNOW`` (explicitly established unknown, with evidence).

**Mention != evidence != validation.**
- ``AgentConcept.mentions`` are Observation spans the Agent interprets as
  referring to the concept (identity may use them; property scoring never
  does).
- Every graph property reference carries its own property evidence; the
  evaluator scores properties from property evidence ONLY.
- Validation statuses (grounded / confirmed / partially_confirmed / unknown /
  disputed) are backed by explicit validation/dialogue evidence:
  ``ground_concept`` is BINDING-AWARE — it resolves its evidence (global
  span rule + the canonical stakeholder resolver) to exactly one
  kind-compatible knowledge concept, rejecting ambiguous, unrelated or
  kind-incompatible evidence (private stakeholder ids never appear in tool
  output; the Agent-visible ``grounded`` status always agrees with the
  evaluator binding); ``confirm_concept`` / ``mark_concept_unknown`` /
  ``mark_concept_disputed`` require private dialogue events (act confirm /
  partial / unknown / dispute); ``record_terminology_agreement`` requires a
  private terminology-confirmation event with the same proposed term. An
  ordinary workflow mention is never enough.

**DONT_KNOW is explicit and evidenced.** ``record_dont_know`` /
``record_edge_condition_dont_know`` (or ``{"dont_know": true, "evidence":
[...]}`` property args) record a DONT_KNOW marker ONLY when the cited
Observation spans resolve to the corresponding stakeholder DONT_KNOW
semantic slot — a known value or a known-absent slot rejects the recording.

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
    SemanticLedger,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import (
    UNSET,
    AbsentType,
    AgentConcept,
    AgentGraph,
    ConceptRef,
    DontKnowType,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    Observation,
    TerminologyAgreement,
    is_dont_know,
    spans_correspond,
)
from tau2.domains.business_interview.grounding import (
    grounded_refs,
)
from tau2.domains.business_interview.grounding import (
    resolve_span_text as _resolve_span_text,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    slot_concepts,
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

    def _knowledge(self) -> Optional[StakeholderKnowledge]:
        """The scenario's StakeholderKnowledge (via the private catalog), or
        None when no scenario is wired. Evaluator-only: never exposed to the
        Agent, never rendered in tool output."""
        catalog = self.assertion_ledger.catalog
        return catalog.knowledge if catalog is not None else None

    @staticmethod
    def _normalize_prop(prop: str) -> str:
        """Tool-facing property name -> semantic slot property name
        (``necessity_rationale`` -> ``rationale``)."""
        return "rationale" if prop == "necessity_rationale" else prop

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
        return [ConceptRef(concept_id=cid, confidence=1.0) for cid in concept_ids or []]

    def _resolve_dont_know_slots(
        self,
        evs: list[EvidenceRef],
        prop: str,
        where: str,
        allowed_props: Optional[set[str]] = None,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> list[EvidenceRef]:
        """Validate DONT_KNOW evidence for one property slot.

        Every evidence span must resolve (global span rule + the canonical
        stakeholder resolver) to exactly one semantic id that IS the
        stakeholder's DONT_KNOW slot of a property in ``allowed_props`` — a
        DONT_KNOW may only be recorded when the evidence resolves to the
        corresponding stakeholder DONT_KNOW semantic slot. When ``bound_node``
        / ``bound_edge`` is given (the Agent element's unique binding), the
        evidence must resolve EXACTLY to that element's slot
        (``node:<bound_node>:<prop>`` / ``edge:<bound_edge>:condition``) —
        evidence about another node's or edge's slot is rejected. Returns the
        refs that resolve to ``prop`` itself (private stakeholder ids never
        appear in error messages)."""
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"{where}: no knowledge catalog is wired for this interview "
                f"(cannot verify DONT_KNOW slots)"
            )
        if not evs:
            raise ValueError(f"{where}: DONT_KNOW requires evidence")
        allowed = allowed_props if allowed_props is not None else {prop}
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evs
        )
        if invalid or ambiguous or len(results) != len(evs):
            raise ValueError(
                f"{where}: every evidence span must resolve to exactly one semantic id"
            )
        resolver = knowledge.graph.resolve
        for _ref, sid in results:
            resolved = resolver(sid)
            if (
                resolved is None
                or resolved.kind not in ("node_slot", "edge_slot")
                or resolved.prop not in allowed
                or not is_dont_know(resolved.value)
            ):
                raise ValueError(
                    f"{where}: evidence must resolve to the stakeholder's "
                    f"DONT_KNOW slot for property {sorted(allowed)!r} (the "
                    f"stakeholder knows this value, knows it is absent, or "
                    f"the span is unrelated)"
                )
        if bound_node is not None:
            expected = f"node:{bound_node}:{prop}"
            for _ref, sid in results:
                if sid != expected:
                    raise ValueError(
                        f"{where}: evidence must resolve to the EXACT "
                        f"stakeholder DONT_KNOW slot of the bound node's "
                        f"property {prop!r} — it currently resolves to "
                        f"another element's slot, which cannot support this "
                        f"marker"
                    )
        if bound_edge is not None:
            expected = f"edge:{bound_edge}:condition"
            for _ref, sid in results:
                if sid != expected:
                    raise ValueError(
                        f"{where}: evidence must resolve to the EXACT "
                        f"stakeholder DONT_KNOW condition slot of the bound "
                        f"edge — it currently resolves to another edge's "
                        f"slot, which cannot support this marker"
                    )
        return [
            ref
            for ref, sid in results
            if (r := resolver(sid)) is not None and r.prop == prop
        ]

    def _dont_know_marker(
        self,
        arg: dict,
        where: str,
        prop: str,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> DontKnowType:
        """Build a DONT_KNOW marker from ``{"dont_know": true, "evidence":
        [...]}`` after validating the evidence against the corresponding
        stakeholder DONT_KNOW slot (when a unique binding is available,
        EXACTLY that bound element's slot)."""
        evs = self._require_evidence(arg.get("evidence"))
        matched = self._resolve_dont_know_slots(
            evs, prop, where, bound_node=bound_node, bound_edge=bound_edge
        )
        return DontKnowType(evidence=matched)

    def _resolve_absent_slots(
        self,
        evs: list[EvidenceRef],
        prop: str,
        where: str,
        allowed_props: Optional[set[str]] = None,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> list[EvidenceRef]:
        """Validate ABSENT evidence for one property slot.

        Every evidence span must resolve (global span rule + the canonical
        stakeholder resolver) to exactly one semantic id that IS the
        stakeholder's KNOWN-ABSENT slot (value None) of a property in
        ``allowed_props`` — ABSENT may only be recorded when the evidence
        resolves to the corresponding stakeholder known-absent semantic
        slot. When ``bound_node`` / ``bound_edge`` is given (the Agent
        element's unique binding), the evidence must resolve EXACTLY to that
        element's slot (``node:<bound_node>:<prop>`` /
        ``edge:<bound_edge>:condition``) — evidence about another node's or
        edge's slot is rejected. Returns the refs that resolve to ``prop``
        itself (private stakeholder ids never appear in error messages)."""
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"{where}: no knowledge catalog is wired for this interview "
                f"(cannot verify ABSENT slots)"
            )
        if not evs:
            raise ValueError(f"{where}: ABSENT requires evidence")
        allowed = allowed_props if allowed_props is not None else {prop}
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evs
        )
        if invalid or ambiguous or len(results) != len(evs):
            raise ValueError(
                f"{where}: every evidence span must resolve to exactly one semantic id"
            )
        resolver = knowledge.graph.resolve
        for _ref, sid in results:
            resolved = resolver(sid)
            if (
                resolved is None
                or resolved.kind not in ("node_slot", "edge_slot")
                or resolved.prop not in allowed
                or resolved.value is not None
            ):
                raise ValueError(
                    f"{where}: evidence must resolve to the stakeholder's "
                    f"KNOWN-ABSENT slot for property {sorted(allowed)!r} "
                    f"(the stakeholder knows a value here, does not know, "
                    f"or the span is unrelated)"
                )
        if bound_node is not None:
            expected = f"node:{bound_node}:{prop}"
            for _ref, sid in results:
                if sid != expected:
                    raise ValueError(
                        f"{where}: evidence must resolve to the EXACT "
                        f"stakeholder known-absent slot of the bound node's "
                        f"property {prop!r} — it currently resolves to "
                        f"another element's slot, which cannot support this "
                        f"marker"
                    )
        if bound_edge is not None:
            expected = f"edge:{bound_edge}:condition"
            for _ref, sid in results:
                if sid != expected:
                    raise ValueError(
                        f"{where}: evidence must resolve to the EXACT "
                        f"stakeholder known-absent condition slot of the "
                        f"bound edge — it currently resolves to another "
                        f"edge's slot, which cannot support this marker"
                    )
        return [
            ref
            for ref, sid in results
            if (r := resolver(sid)) is not None and r.prop == prop
        ]

    def _absent_marker(
        self,
        arg: dict,
        where: str,
        prop: str,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> AbsentType:
        """Build an ABSENT marker from ``{"absent": true, "evidence":
        [...]}`` after validating the evidence against the corresponding
        stakeholder known-absent slot (when a unique binding is available,
        EXACTLY that bound element's slot)."""
        evs = self._require_evidence(arg.get("evidence"))
        matched = self._resolve_absent_slots(
            evs, prop, where, bound_node=bound_node, bound_edge=bound_edge
        )
        return AbsentType(evidence=matched)

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
            if arg.get("dont_know"):
                raise ValueError(
                    f"{where}: DONT_KNOW is not supported for this argument"
                )
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

    def _grounding_concept_ids(
        self, evidence: list[EvidenceRef], where: str
    ) -> set[str]:
        """The knowledge concept(s) represented by the evidence's grounded
        semantic ids (global span rule + canonical resolver)."""
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"{where}: no knowledge catalog is wired for this interview "
                f"(cannot resolve grounding evidence)"
            )
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evidence
        )
        if invalid or ambiguous or len(results) != len(evidence):
            raise ValueError(
                f"{where}: every evidence span must resolve to exactly one semantic id"
            )
        concepts: set[str] = set()
        resolver = knowledge.graph.resolve
        for _ref, sid in results:
            resolved = resolver(sid)
            if resolved is None:
                continue
            if resolved.kind == "node_element":
                if resolved.ref is None:
                    continue
                concepts.add(resolved.ref.concept_id)
            elif resolved.kind in ("node_slot", "edge_slot"):
                concepts.update(slot_concepts(resolved.value))
            elif resolved.kind == "concept":
                if resolved.concept is None:
                    continue
                concepts.add(resolved.concept.id)
            # node / edge existence: represents NO concept (unrelated)
        return concepts

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

    def _check_no_bulk_validation(
        self, concept_id: str, evs: list[EvidenceRef]
    ) -> None:
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
        """Mark a concept as grounded with authentic, binding-aware
        provenance.

        Grounding means the stakeholder's own speech resolves this concept's
        identity to exactly ONE knowledge concept of a compatible kind. The
        cited evidence spans must resolve (via the private semantic
        annotations) to that single knowledge concept: ambiguous evidence,
        unrelated elements (e.g. a node or edge position) and
        kind-incompatible concepts are REJECTED. No confirmation dialogue is
        needed; an ordinary invented span is not enough.

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
        represented = self._grounding_concept_ids(
            evs, f"ground_concept for {concept_id}"
        )
        if len(represented) != 1:
            raise ValueError(
                f"ground_concept for {concept_id}: evidence must represent "
                f"exactly one knowledge concept (ambiguous or unrelated "
                f"evidence)"
            )
        kid = next(iter(represented))
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"ground_concept for {concept_id}: no knowledge catalog is "
                f"wired for this interview"
            )
        if knowledge.graph.concepts[kid].kind != concept.kind:
            raise ValueError(
                f"ground_concept for {concept_id}: evidence represents a "
                f"knowledge concept of kind "
                f"{knowledge.graph.concepts[kid].kind!r}, incompatible with "
                f"your concept's kind {concept.kind!r}"
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
                for ref in node.refs(prop):
                    if ref.concept_id in source_concept_ids:
                        ref.concept_id = target_concept_id
        for edge in graph.edges.values():
            if (
                isinstance(edge.condition, ConceptRef)
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

    @is_tool(ToolType.WRITE)
    def observe_latest_stakeholder_message(self) -> str:
        """Capture the newest stakeholder (user) message as an Observation and
        return its Observation id directly — the single-step equivalent of
        ``observe_latest_stakeholder_message`` + ``observe_message(message_id)``,
        so one Agent decision suffices for this deterministic bookkeeping step.

        Idempotent: re-calling returns the same Observation id, and the
        captured Observation is byte-for-byte identical to ``observe_message``
        on the same message (exact ``obs_<turn>`` id, text, source, order).
        To capture an EARLIER message use ``observe_message(message_id)``;
        ``list_stakeholder_messages`` still shows the stable ``sm_<n>`` ids.
        """
        entries = self._stakeholder_entries()
        if not entries:
            raise ValueError("no stakeholder (user) message has been recorded yet")
        _, turn, content = entries[-1]
        return self._capture_user_message(turn, content)

    @is_tool(ToolType.READ)
    def list_stakeholder_messages(self) -> str:
        lines = [
            f"{sm_id}: {content}"
            for sm_id, turn, content in self._stakeholder_entries()
        ]
        return "\n".join(lines) if lines else "(no stakeholder messages yet)"

    def _ref_from_prop_arg(
        self,
        arg,
        where: str,
        expected_kind: str,
        prop: Optional[str] = None,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ):
        """Property arg: concept id | {"concept_id", "evidence"} |
        {"dont_know": true, "evidence": [...]} | {"absent": true,
        "evidence": [...]} | None.

        DONT_KNOW / ABSENT markers are validated against the corresponding
        stakeholder slot; when ``bound_node`` / ``bound_edge`` is given
        (the Agent element's unique binding) they must resolve EXACTLY to
        that element's slot.
        """
        prop = self._normalize_prop(prop or where.rsplit(" ", 1)[-1])
        if arg is None:
            raise ValueError(f"{where}: missing concept reference")
        if isinstance(arg, dict):
            if arg.get("dont_know"):
                return self._dont_know_marker(
                    arg, where, prop, bound_node=bound_node, bound_edge=bound_edge
                )
            if arg.get("absent"):
                return self._absent_marker(
                    arg, where, prop, bound_node=bound_node, bound_edge=bound_edge
                )
            cid = str(arg.get("concept_id") or "")
            evidence = self._require_evidence(arg.get("evidence"))
        elif arg == "DONT_KNOW":
            raise ValueError(
                f"{where}: a bare DONT_KNOW needs evidence — pass "
                f'{{"dont_know": true, "evidence": [...]}}'
            )
        elif arg == "ABSENT":
            raise ValueError(
                f"{where}: a bare ABSENT needs evidence — pass "
                f'{{"absent": true, "evidence": [...]}}'
            )
        else:
            cid = str(arg)
            evidence = []
        self._require_kind(cid, expected_kind, where)
        return self._ref(cid, evidence=evidence)

    def _list_slot_arg(
        self,
        arg,
        where: str,
        expected_kind: str,
        prop: str,
        bound_node: Optional[str] = None,
    ):
        """reads/writes arg: list of concept ids / {concept_id, evidence},
        {"dont_know": true, "evidence": [...]} for a whole-property
        DONT_KNOW, or {"absent": true, "evidence": [...]} for a
        whole-property ABSENT. None means UNSET (not investigated)."""
        if arg is None:
            return UNSET
        if isinstance(arg, dict) and arg.get("dont_know"):
            return self._dont_know_marker(
                arg, where, self._normalize_prop(prop), bound_node=bound_node
            )
        if isinstance(arg, dict) and arg.get("absent"):
            return self._absent_marker(
                arg, where, self._normalize_prop(prop), bound_node=bound_node
            )
        if not isinstance(arg, list):
            raise ValueError(
                f"{where}: expected a list of concept refs, "
                f'{{"dont_know": true, "evidence": [...]}} or '
                f'{{"absent": true, "evidence": [...]}}'
            )
        if not arg:
            raise ValueError(
                f"{where}: an empty reads/writes list is ambiguous — record "
                f'{{"absent": true, "evidence": [...]}} when the stakeholder '
                f"established that nothing is read/written"
            )
        return self._refs_from_prop_list(arg, where, expected_kind)

    def _refs_from_prop_list(
        self, args: Optional[list], where: str, expected_kind: str
    ) -> list[ConceptRef]:
        """reads/writes arg: list of str | {"concept_id", "evidence"}."""
        refs: list[ConceptRef] = []
        for arg in args or []:
            ref = self._ref_from_prop_arg(arg, where, expected_kind)
            if isinstance(ref, (DontKnowType, AbsentType)):
                raise ValueError(
                    f"{where}: {type(ref).__name__} applies to the whole "
                    f"reads/writes property, not to one element — pass "
                    f'{{"dont_know": true, "evidence": [...]}} or '
                    f'{{"absent": true, "evidence": [...]}} as the '
                    f"property value"
                )
            refs.append(ref)
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

        To record that you CANNOT determine a property (the stakeholder said
        so), pass ``{"dont_know": true, "evidence": [...]}`` — the evidence
        must resolve to the stakeholder's DONT_KNOW slot for that property.

        Args:
            node_id: Your own identifier for this node.
            activity: Concept id or {concept_id, evidence} (kind=activity).
            actor: Concept id or {concept_id, evidence} (kind=actor).
            system: Concept id or {concept_id, evidence} (kind=system).
            reads: List of data concept ids / {concept_id, evidence}, or
                {dont_know, evidence}.
            writes: List of data concept ids / {concept_id, evidence}, or
                {dont_know, evidence}.
            necessity_rationale: Concept id or {concept_id, evidence}
                (kind=rationale).
            evidence: Activity evidence shorthand (optional).

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if node_id in graph.nodes:
            raise ValueError(f"node already exists: {node_id}")
        # When any property carries an ABSENT/DONT_KNOW marker, the node must
        # first be UNIQUELY bound to a stakeholder node from the call's
        # authentic property evidence — a marker is only valid on the bound
        # element's own slot (never another node's).
        has_marker = any(
            isinstance(v, dict) and (v.get("dont_know") or v.get("absent"))
            for v in (actor, system, reads, writes, necessity_rationale)
        )
        bound_node: Optional[str] = None
        if has_marker:
            call_evs: list[EvidenceRef] = []
            if evidence:
                call_evs.extend(self._require_evidence(evidence))
            for prop_arg in (
                activity,
                actor,
                system,
                reads,
                writes,
                necessity_rationale,
            ):
                if isinstance(prop_arg, dict):
                    call_evs.extend(self._require_evidence(prop_arg.get("evidence")))
                elif isinstance(prop_arg, list):
                    for item in prop_arg:
                        if isinstance(item, dict):
                            call_evs.extend(
                                self._require_evidence(item.get("evidence"))
                            )
            bound_node = self._node_candidates_from_evidence(
                call_evs, f"add_node for {node_id}", "node"
            )
        activity_ref = self._ref_from_prop_arg(
            activity,
            "add_node activity",
            "activity",
            bound_node=bound_node,
        )
        if evidence and isinstance(activity_ref, ConceptRef):
            if not activity_ref.evidence:
                activity_ref.evidence = self._require_evidence(evidence)
        graph.nodes[node_id] = Node(
            id=node_id,
            activity=activity_ref,
            actor=(
                self._ref_from_prop_arg(
                    actor, "add_node actor", "actor", bound_node=bound_node
                )
                if actor is not None
                else UNSET
            ),
            system=(
                self._ref_from_prop_arg(
                    system, "add_node system", "system", bound_node=bound_node
                )
                if system is not None
                else UNSET
            ),
            reads=(
                self._list_slot_arg(
                    reads,
                    "add_node reads",
                    "data",
                    "reads",
                    bound_node=bound_node,
                )
                if reads is not None
                else UNSET
            ),
            writes=(
                self._list_slot_arg(
                    writes,
                    "add_node writes",
                    "data",
                    "writes",
                    bound_node=bound_node,
                )
                if writes is not None
                else UNSET
            ),
            necessity_rationale=(
                self._ref_from_prop_arg(
                    necessity_rationale,
                    "add_node necessity_rationale",
                    "rationale",
                    bound_node=bound_node,
                )
                if necessity_rationale is not None
                else UNSET
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
        so the reference carries its own evidence, or
        ``{"dont_know": true, "evidence": [...]}`` to record that you
        cannot determine it (evidence must resolve to the stakeholder's
        DONT_KNOW slot for that property), or ``{"absent": true,
        "evidence": [...]}`` to record that the stakeholder established the
        property is ABSENT (known absent / known-empty — never an empty
        list, which is ambiguous). ``unset`` returns a property to UNSET
        (not investigated — never a conclusion).
        """
        node = self._node(node_id)
        if unset:
            for prop in unset:
                if prop in ("reads", "writes"):
                    setattr(node, prop, UNSET)
                elif prop in ("actor", "system", "necessity_rationale"):
                    setattr(node, prop, UNSET)
                else:
                    raise ValueError(f"cannot unset property {prop!r}")
        # When any property carries an ABSENT/DONT_KNOW marker, the node must
        # be UNIQUELY bound to its stakeholder node first — a marker is only
        # valid on the bound element's own slot (never another node's).
        has_marker = any(
            isinstance(v, dict) and (v.get("dont_know") or v.get("absent"))
            for v in (activity, actor, system, reads, writes, necessity_rationale)
        )
        bound_node: Optional[str] = None
        if has_marker:
            bound_node = self._bound_stakeholder_node(
                node_id, f"update_node for {node_id}"
            )
        if activity is not None:
            node.activity = self._ref_from_prop_arg(
                activity,
                "update_node activity",
                "activity",
                bound_node=bound_node,
            )
            if evidence and isinstance(node.activity, ConceptRef):
                if not node.activity.evidence:
                    node.activity.evidence = self._require_evidence(evidence)
        if actor is not None:
            node.actor = self._ref_from_prop_arg(
                actor, "update_node actor", "actor", bound_node=bound_node
            )
        if system is not None:
            node.system = self._ref_from_prop_arg(
                system, "update_node system", "system", bound_node=bound_node
            )
        if necessity_rationale is not None:
            node.necessity_rationale = self._ref_from_prop_arg(
                necessity_rationale,
                "update_node necessity_rationale",
                "rationale",
                bound_node=bound_node,
            )
        if reads is not None:
            node.reads = self._list_slot_arg(
                reads,
                "update_node reads",
                "data",
                "reads",
                bound_node=bound_node,
            )
        if writes is not None:
            node.writes = self._list_slot_arg(
                writes,
                "update_node writes",
                "data",
                "writes",
                bound_node=bound_node,
            )
        return f"Updated node {node_id}."

    def _node_candidates_from_evidence(
        self,
        evs: list[EvidenceRef],
        where: str,
        kind: str,
    ) -> str:
        """The UNIQUE stakeholder node/edge an Agent element is
        authentically bound to, derived from its evidence provenance
        (global span rule + the canonical resolver). Raises a concise
        error when not yet uniquely-bound — the Agent must first
        add authentic graph/property evidence identifying this element."""
        knowledge = self._knowledge()
        assert knowledge is not None
        resolver = knowledge.graph.resolve
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evs
        )
        if invalid or ambiguous or len(results) != len(evs):
            raise ValueError(
                f"{where}: every evidence span must resolve to exactly one semantic id"
            )
        candidates: set[str] = set()
        for _ref, sid in results:
            resolved = resolver(sid)
            if resolved is None:
                continue
            if kind == "edge":
                if resolved.kind in ("edge", "edge_slot") and resolved.edge_id:
                    candidates.add(resolved.edge_id)
            else:
                if (
                    resolved.kind
                    in (
                        "node",
                        "node_slot",
                        "node_element",
                    )
                    and resolved.node_id
                ):
                    candidates.add(resolved.node_id)
        if len(candidates) != 1:
            raise ValueError(
                f"{where}: the agent {kind} is not uniquely bindable to a "
                f"stakeholder element (candidates: "
                f"{sorted(candidates) or 'none'}); add authentic graph/"
                f"property evidence that identifies it first"
            )
        return next(iter(candidates))

    def _require_edge_endpoints_match(
        self,
        bound_edge: str,
        from_node: str,
        to_node: str,
        where: str,
    ) -> None:
        """Require that the bound stakeholder edge's endpoints equal the
        stakeholder nodes bound by the agent edge's endpoints — the same
        condition the evaluator uses to map the edge. A mismatch means the
        agent edge (and any marker on it) would be unmapped/unsupported."""
        knowledge = self._knowledge()
        assert knowledge is not None
        resolved_edge = knowledge.graph.resolve(f"edge:{bound_edge}")
        if resolved_edge is None or resolved_edge.edge is None:
            raise ValueError(
                f"{where}: bound stakeholder edge {bound_edge!r} does not resolve"
            )
        sf = self._bound_stakeholder_node(from_node, f"{where} (edge endpoint)")
        st = self._bound_stakeholder_node(to_node, f"{where} (edge endpoint)")
        if resolved_edge.edge.from_node != sf or resolved_edge.edge.to_node != st:
            raise ValueError(
                f"{where}: the bound stakeholder edge's endpoints do not "
                f"match this agent edge's bound endpoints — the marker "
                f"would be unsupported; add authentic evidence that "
                f"identifies this edge first"
            )

    def _bound_stakeholder_node(self, node_id: str, where: str) -> str:
        """The UNIQUE stakeholder node an Agent node is authentically bound
        to, derived from its property and marker provenance (global span
        rule + the canonical resolver).

        Raises a concise error when the Agent graph element is not yet
        uniquely bindable — the Agent must add authentic graph/property
        evidence first."""
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"{where}: no knowledge catalog is wired for this interview "
                f"(cannot bind the agent node)"
            )
        node = self._node(node_id)
        resolver = knowledge.graph.resolve
        candidates: set[str] = set()
        for prop in (
            "activity",
            "actor",
            "system",
            "reads",
            "writes",
            "rationale",
        ):
            slot = node.slot_value(prop)
            evs: list[EvidenceRef] = []
            if isinstance(slot, (AbsentType, DontKnowType)):
                evs.extend(slot.evidence)
            else:
                for r in node.asserted_refs(prop):
                    evs.extend(r.evidence)
            if not evs:
                continue
            results, invalid, ambiguous = grounded_refs(
                self.db, self.assertion_ledger.annotations(), evs
            )
            if invalid or ambiguous or len(results) != len(evs):
                continue
            for _ref, sid in results:
                resolved = resolver(sid)
                if resolved is not None and resolved.kind in (
                    "node",
                    "node_slot",
                    "node_element",
                ):
                    if resolved.node_id is not None:
                        candidates.add(resolved.node_id)
        if len(candidates) != 1:
            raise ValueError(
                f"{where}: agent node {node_id!r} is not uniquely bindable "
                f"to a stakeholder node (candidates: "
                f"{sorted(candidates) or 'none'}); add authentic "
                f"graph/property evidence that identifies this node first"
            )
        return next(iter(candidates))

    def _bound_stake_edge(self, edge_id: str, where: str) -> str:
        """The UNIQUE stakeholder edge an Agent edge is authentically bound
        to, derived from its evidence provenance (global span rule + the
        canonical resolver). Raises when not uniquely bindable."""
        knowledge = self._knowledge()
        if knowledge is None:
            raise ValueError(
                f"{where}: no knowledge catalog is wired for this interview "
                f"(cannot bind the agent edge)"
            )
        edge = self._edge(edge_id)
        resolver = knowledge.graph.resolve
        candidates: set[str] = set()
        evs: list[EvidenceRef] = list(edge.evidence)
        if isinstance(edge.condition, ConceptRef) and edge.condition.asserted:
            evs.extend(edge.condition.evidence)
        elif isinstance(edge.condition, (AbsentType, DontKnowType)):
            evs.extend(edge.condition.evidence)
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evs
        )
        if not (invalid or ambiguous or len(results) != len(evs)):
            for _ref, sid in results:
                resolved = resolver(sid)
                if resolved is not None and resolved.kind in (
                    "edge",
                    "edge_slot",
                ):
                    if resolved.edge_id is not None:
                        candidates.add(resolved.edge_id)
        if len(candidates) != 1:
            raise ValueError(
                f"{where}: agent edge {edge_id!r} is not uniquely bindable "
                f"to a stakeholder edge (candidates: "
                f"{sorted(candidates) or 'none'}); add authentic evidence "
                f"that identifies this edge first"
            )
        return next(iter(candidates))

    def _group_marker_evidence(
        self,
        evs: list[EvidenceRef],
        props: set[str],
        where: str,
        kind: str,
        bound_node: str,
    ) -> dict[str, list[EvidenceRef]]:
        """Partition marker evidence (ABSENT/DONT_KNOW) per property slot of
        the bound stakeholder node.

        Every ref must resolve (global span rule + the canonical resolver) to
        EXACTLY the bound node's slot ``node:<bound>:<prop>`` for one of
        ``props``, whose value matches the marker ``kind`` (``absent`` =
        value None, ``dont_know`` = DONT_KNOW). Each listed property must be
        covered by at least one ref. Returns ``{prop: [refs...]}``."""
        knowledge = self._knowledge()
        assert knowledge is not None
        resolver = knowledge.graph.resolve
        results, invalid, ambiguous = grounded_refs(
            self.db, self.assertion_ledger.annotations(), evs
        )
        if invalid or ambiguous or len(results) != len(evs):
            raise ValueError(
                f"{where}: every evidence span must resolve to exactly one semantic id"
            )
        grouped: dict[str, list[EvidenceRef]] = {p: [] for p in props}
        for ref, sid in results:
            prop = None
            for p in sorted(props):
                if sid == f"node:{bound_node}:{p}":
                    prop = p
                    break
            if prop is None:
                raise ValueError(
                    f"{where}: evidence must resolve to the EXACT "
                    f"stakeholder {kind} slot of the bound node — it "
                    f"currently resolves to another element's slot, which "
                    f"cannot support this marker"
                )
            resolved = resolver(f"node:{bound_node}:{prop}")
            if resolved is None:
                raise ValueError(f"{where}: evidence resolves to an unresolvable slot")
            if kind == "absent" and resolved.value is not None:
                raise ValueError(
                    f"{where}: evidence must resolve to the stakeholder's "
                    f"KNOWN-ABSENT slot for property {prop!r}"
                )
            if kind == "dont_know" and not is_dont_know(resolved.value):
                raise ValueError(
                    f"{where}: evidence must resolve to the stakeholder's "
                    f"DONT_KNOW slot for property {prop!r}"
                )
            grouped[prop].append(ref)
        uncovered = sorted(p for p, refs in grouped.items() if not refs)
        if uncovered:
            raise ValueError(
                f"{where}: every listed property must be covered by at least "
                f"one evidence span; no span covers {uncovered}"
            )
        return grouped

    @is_tool(ToolType.WRITE)
    def record_dont_know(
        self,
        node_id: str,
        properties: list[str],
        evidence: list,
    ) -> str:
        """Record that you cannot determine property values of an existing
        node.

        ``DONT_KNOW`` is an explicit, evidenced epistemic state — distinct
        from UNSET (omitted / not investigated, never a conclusion) and from
        ABSENT (explicitly established absent). Every cited span
        must resolve to the stakeholder's DONT_KNOW slot of one of the given
        properties (a slot the stakeholder knows or knows to be absent
        rejects the recording), and every listed property must be covered by
        at least one span.

        Args:
            node_id: The node whose properties are unknown.
            properties: Properties you cannot determine
                (activity/actor/system/reads/writes/rationale).
            evidence: Evidence spans resolving to the stakeholder's DONT_KNOW
                slots for these properties (required).

        Returns:
            A confirmation message.
        """
        node = self._node(node_id)
        props = {self._normalize_prop(p) for p in properties}
        unknown = props - set(
            ("activity", "actor", "system", "reads", "writes", "rationale")
        )
        if unknown:
            raise ValueError(
                f"record_dont_know: unknown property {sorted(unknown)}; must "
                f"be one of activity/actor/system/reads/writes/rationale"
            )
        if not props:
            raise ValueError("record_dont_know: properties must not be empty")
        bound = self._bound_stakeholder_node(node_id, f"record_dont_know for {node_id}")
        evs = self._require_evidence(evidence)
        grouped = self._group_marker_evidence(
            evs, props, f"record_dont_know for {node_id}", "dont_know", bound
        )
        by_prop: dict[str, list[EvidenceRef]] = {}
        for prop in props:
            where = f"record_dont_know for {node_id} {prop}"
            by_prop[prop] = self._resolve_dont_know_slots(
                grouped[prop],
                prop,
                where,
                allowed_props={prop},
                bound_node=bound,
            )
        for prop in sorted(props):
            attr = "necessity_rationale" if prop == "rationale" else prop
            setattr(node, attr, DontKnowType(evidence=by_prop[prop]))
        return f"Recorded DONT_KNOW on {node_id} for: {', '.join(sorted(props))}."

    @is_tool(ToolType.WRITE)
    def record_edge_condition_dont_know(self, edge_id: str, evidence: list) -> str:
        """Record that you cannot determine an edge's condition.

        ``DONT_KNOW`` is an explicit, evidenced epistemic state — distinct
        from a missing condition (known absent / unconditional). Every cited
        span must resolve to the stakeholder's DONT_KNOW condition slot of
        this kind of edge.

        Args:
            edge_id: The edge whose condition is unknown.
            evidence: Evidence spans resolving to a stakeholder DONT_KNOW
                condition slot (required).

        Returns:
            A confirmation message.
        """
        self._edge(edge_id)
        evs = self._require_evidence(evidence)
        bound = self._bound_stake_edge(
            edge_id, f"record_edge_condition_dont_know for {edge_id}"
        )
        self._resolve_dont_know_slots(
            evs,
            "condition",
            f"record_edge_condition_dont_know for {edge_id}",
            bound_edge=bound,
        )
        self._edge(edge_id).condition = DontKnowType(evidence=evs)
        return f"Recorded DONT_KNOW on edge {edge_id} condition."

    @is_tool(ToolType.WRITE)
    def record_absent(
        self,
        node_id: str,
        properties: list[str],
        evidence: list,
    ) -> str:
        """Record that you explicitly established a node property to be
        ABSENT (e.g. the stakeholder said the step reads nothing).

        ``ABSENT`` is an explicit, evidenced epistemic state — distinct from
        UNSET (not investigated; ``update_node(unset=...)`` returns a slot to
        UNSET) and from DONT_KNOW. Every cited span must resolve to the
        stakeholder's KNOWN-ABSENT slot of one of the given properties (a
        slot with a known value or a DONT_KNOW slot rejects the recording),
        and every listed property must be covered by at least one span.

        Args:
            node_id: The node whose properties are absent.
            properties: Properties established absent
                (activity/actor/system/reads/writes/rationale).
            evidence: Evidence spans resolving to the stakeholder's
                known-absent slots for these properties (required).

        Returns:
            A confirmation message.
        """
        node = self._node(node_id)
        props = {self._normalize_prop(p) for p in properties}
        unknown = props - set(
            ("activity", "actor", "system", "reads", "writes", "rationale")
        )
        if unknown:
            raise ValueError(
                f"record_absent: unknown property {sorted(unknown)}; must "
                f"be one of activity/actor/system/reads/writes/rationale"
            )
        if not props:
            raise ValueError("record_absent: properties must not be empty")
        bound = self._bound_stakeholder_node(node_id, f"record_absent for {node_id}")
        evs = self._require_evidence(evidence)
        grouped = self._group_marker_evidence(
            evs, props, f"record_absent for {node_id}", "absent", bound
        )
        by_prop: dict[str, list[EvidenceRef]] = {}
        for prop in props:
            where = f"record_absent for {node_id} {prop}"
            by_prop[prop] = self._resolve_absent_slots(
                grouped[prop],
                prop,
                where,
                allowed_props={prop},
                bound_node=bound,
            )
        for prop in sorted(props):
            attr = "necessity_rationale" if prop == "rationale" else prop
            setattr(node, attr, AbsentType(evidence=by_prop[prop]))
        return f"Recorded ABSENT on {node_id} for: {', '.join(sorted(props))}."

    @is_tool(ToolType.WRITE)
    def record_edge_condition_absent(self, edge_id: str, evidence: list) -> str:
        """Record that you explicitly established an edge to be
        UNCONDITIONAL (its condition is ABSENT).

        ``ABSENT`` is an explicit, evidenced epistemic state — distinct from
        UNSET (not investigated) and from DONT_KNOW. Every cited span must
        resolve to the stakeholder's KNOWN-ABSENT condition slot (value
        None) of this kind of edge.

        Args:
            edge_id: The edge whose condition is absent.
            evidence: Evidence spans resolving to a stakeholder known-absent
                condition slot (required).

        Returns:
            A confirmation message.
        """
        self._edge(edge_id)
        evs = self._require_evidence(evidence)
        bound = self._bound_stake_edge(
            edge_id, f"record_edge_condition_absent for {edge_id}"
        )
        self._resolve_absent_slots(
            evs,
            "condition",
            f"record_edge_condition_absent for {edge_id}",
            bound_edge=bound,
        )
        self._edge(edge_id).condition = AbsentType(evidence=evs)
        return f"Recorded ABSENT on edge {edge_id} condition."

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
        # When the condition carries an ABSENT/DONT_KNOW marker, the edge
        # must be UNIQUELY bound to a stakeholder edge from the call's
        # authentic evidence first — the marker is only valid on the bound
        # edge's own condition slot (never another edge's).
        has_marker = isinstance(condition, dict) and (
            condition.get("dont_know") or condition.get("absent")
        )
        bound_edge: Optional[str] = None
        if has_marker:
            call_evs: list[EvidenceRef] = list(self._require_evidence(evidence))
            if isinstance(condition, dict):
                call_evs.extend(self._require_evidence(condition.get("evidence")))
            bound_edge = self._node_candidates_from_evidence(
                call_evs, f"add_edge for {edge_id}", "edge"
            )
            # the bound stakeholder edge must have the same endpoints as the
            # agent edge's bound endpoints (else the edge stays unmapped and
            # any marker on it would be unsupported)
            self._require_edge_endpoints_match(
                bound_edge, from_node, to_node, f"add_edge for {edge_id}"
            )
        cond_ref = (
            self._ref_from_prop_arg(
                condition,
                "add_edge condition",
                "condition",
                bound_edge=bound_edge,
            )
            if condition is not None
            else UNSET
        )
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
            edge.condition = UNSET
        elif condition is not None:
            has_marker = isinstance(condition, dict) and (
                condition.get("dont_know") or condition.get("absent")
            )
            bound_edge: Optional[str] = None
            if has_marker:
                bound_edge = self._bound_stake_edge(
                    edge_id, f"update_edge for {edge_id}"
                )
            edge.condition = self._ref_from_prop_arg(
                condition,
                "update_edge condition",
                "condition",
                bound_edge=bound_edge,
            )
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
                "unknown / mark disputed): " + ", ".join(hypothesized)
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
