"""Agent tools for the graph-native business_interview benchmark (v13 —
env-owned Observations and Truth reconstruction).

The environment creates one immutable Observation per ACCEPTED Stakeholder
message BEFORE the Agent sees it and delivers the Observation id inline with
the public text (``[Observation obs_N] <text>``). There is therefore NO
observation-capture tool: the Agent never calls one and never creates or
mutates an Observation. It builds a private glossary of typed
``AgentConcept``\\ s and constructs an inferred ``AgentGraph`` whose
node/edge property references may carry optional ``EvidenceRef``\\ s.
Every Agent slot is one of the FOUR epistemic states: ``UNSET``
(not investigated / no conclusion) / ``ConceptRef`` (known value) /
``ABSENT`` (explicitly established absent, with evidence) /
``DONT_KNOW`` (explicitly established unknowable, with evidence).

**Mention != evidence != validation.**
- ``AgentConcept.mentions`` are Observation spans the Agent interprets as
  referring to the concept (diagnostic only).
- Graph/property references may carry optional evidence spans, but evidence
  is **diagnostic only**: evaluation scores reconstruction against Truth and
  never requires exact quote provenance.

**DONT_KNOW / ABSENT are beliefs, not provenance-gated.** ``record_dont_know`` /
``record_absent`` (and edge-condition variants) record explicit epistemic
markers; cited evidence spans are optional diagnostic hints and are never
required to resolve to a private stakeholder slot.

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
    SemanticLedger,
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
        """Validate an Observation id when diagnostic evidence supplies one.

        Evidence is optional and never a reconstruction gate. A supplied id
        must nevertheless be non-empty and refer to an existing Observation;
        quote/occurrence and private semantic-slot binding are not checked
        here as acceptance criteria.
        """
        if observation_id is None:
            return
        if not observation_id.strip():
            raise ValueError("observation id must not be empty")
        if not any(o.id == observation_id for o in self.db.observations):
            raise ValueError(f"observation not found: {observation_id}")

    def _require_evidence(self, evidence: Optional[list]) -> list[EvidenceRef]:
        """Coerce optional EvidenceRef dicts using shape-only validation.

        Provenance is no longer a prerequisite for recording Agent beliefs:
        a supplied reference must point to an existing Observation, but a
        missing/ambiguous/nonmatching quote and private semantic-slot binding
        never reject the tool call. Evidence is retained as a diagnostic hint.
        """
        refs: list[EvidenceRef] = []
        for raw in evidence or []:
            ref = _ev(raw)
            self._require_observation(ref.observation_id)
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

    def _dont_know_marker(
        self,
        arg: dict,
        where: str,
        prop: str,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> DontKnowType:
        """Build a DONT_KNOW marker from ``{"dont_know": true, ...}``.

        Evidence is optional diagnostic metadata. When supplied, only its
        shape and Observation ids are validated; private sidecar content,
        quote spans and semantic-slot binding never gate recording the belief.
        """
        evs = self._require_evidence(arg.get("evidence"))
        return DontKnowType(evidence=evs)

    def _absent_marker(
        self,
        arg: dict,
        where: str,
        prop: str,
        bound_node: Optional[str] = None,
        bound_edge: Optional[str] = None,
    ) -> AbsentType:
        """Build an ABSENT marker from ``{"absent": true, ...}``.

        Evidence is optional diagnostic metadata. When supplied, only its
        shape and Observation ids are validated; private sidecar content,
        quote spans and semantic-slot binding never gate recording the belief.
        """
        evs = self._require_evidence(arg.get("evidence"))
        return AbsentType(evidence=evs)

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
        evidence: Optional[list] = None,
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

    @is_tool(ToolType.WRITE)
    def record_terminology_agreement(
        self,
        concept_id: str,
        term: str,
        evidence: Optional[list] = None,
    ) -> str:
        """Record an explicit terminology agreement: you proposed ``term`` for
        this concept and the stakeholder explicitly confirmed it.

        Evidence is optional diagnostic metadata. It may point to the
        Observation where the agreement was discussed, but no private
        terminology event or exact quote/semantic binding is required for the
        record to be accepted. A mere mention is still not automatically
        recorded as an agreement; this tool records the Agent's explicit
        bookkeeping choice.

        Args:
            concept_id: The concept the term refers to.
            term: The agreed term.
            evidence: Optional diagnostic Observation spans.

        Returns:
            A confirmation message.
        """
        self._concept(concept_id)
        evs = self._require_evidence(evidence)
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
        concepts' mentions and terminology agreements are folded into the
        target. The source concepts are removed.

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
        """List your glossary concepts (ids, kinds, labels, mentions)."""
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

    def _capture_user_message(self, turn: int, content: str) -> str:
        """PRIVATE (environment-owned) Observation creation for one accepted
        stakeholder utterance at ledger turn ``turn``.

        This is NOT an Agent-facing tool: no Agent can call it, and the Agent
        never creates or mutates Observations. The environment calls it after
        its private sidecar validation succeeds, so every accepted Stakeholder
        utterance becomes exactly one Observation / one Observation id. A
        rejection before this point never consumes an id.
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

    @is_tool(ToolType.READ)
    def list_stakeholder_messages(self) -> str:
        """List the accepted Stakeholder statements and their Observation ids.

        Every accepted Stakeholder response was automatically ingested as an
        Observation by the environment BEFORE the Agent saw it, so the entry
        ids here are the same Observation ids delivered inline with each
        message (``[Observation obs_N] <text>``). Use those ids directly in
        evidence refs — never guess or reuse an older Observation id.
        """
        lines = [
            f"{o.id}: {o.text}"
            for o in sorted(self.db.observations, key=lambda o: o.turn)
        ]
        return "\n".join(lines) if lines else "(no observations yet)"

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

        Evidence on a reference or marker is optional diagnostic metadata.
        When supplied, only its shape and Observation ids are validated;
        private sidecar/provenance never determines whether a structurally
        valid Agent belief is recorded.
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

        Every property reference may carry its OWN diagnostic evidence: pass
        a property as ``concept_id`` or as
        ``{"concept_id": ..., "evidence": [...]}``.
        The call-level ``evidence`` list is the shorthand for the activity's
        evidence (when ``activity`` is a plain concept id). Concept kinds are
        enforced: activity->activity, actor->actor, system->system,
        reads/writes->data, necessity_rationale->rationale.

        To record that you CANNOT determine a property, pass
        ``{"dont_know": true, "evidence": [...]}``. Evidence is optional
        diagnostic metadata; if supplied, its Observation id must exist, but
        its quote and private semantic-slot binding are not a recording gate.

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
            evidence: Optional diagnostic Activity evidence shorthand.

        Returns:
            A confirmation message.
        """
        graph = self._graph()
        if node_id in graph.nodes:
            raise ValueError(f"node already exists: {node_id}")
        # Markers/evidence are diagnostic only: the Agent records beliefs
        # without private-provenance binding requirements.
        bound_node: Optional[str] = None
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
        so the reference may carry diagnostic evidence, or
        ``{"dont_know": true, "evidence": [...]}`` / ``{"absent": true,
        "evidence": [...]}`` for explicit epistemic markers. Evidence is
        optional; if supplied, its Observation id must exist, but quote and
        private semantic-slot binding never gate recording. ``unset`` returns
        a property to UNSET (not investigated — never a conclusion). A known
        empty reads/writes property still uses the ABSENT marker, not an empty
        list.
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
        # Markers are diagnostic only; no private-provenance binding needed.
        bound_node: Optional[str] = None
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

    @is_tool(ToolType.WRITE)
    def record_dont_know(
        self,
        node_id: str,
        properties: list[str],
        evidence: Optional[list] = None,
    ) -> str:
        """Record explicit DONT_KNOW beliefs for an existing node.

        ``DONT_KNOW`` is distinct from UNSET (omitted / not investigated) and
        ABSENT (explicitly established absence). Evidence is optional
        diagnostic metadata; when supplied, only its shape and Observation
        ids are validated. It need not resolve to a stakeholder DONT_KNOW
        slot, and every listed property is marked regardless of private
        sidecar content.

        Args:
            node_id: The node whose properties are unknown.
            properties: Properties you cannot determine
                (activity/actor/system/reads/writes/rationale).
            evidence: Optional diagnostic Observation spans. If supplied,
                their Observation ids must exist; exact quote and semantic-slot
                binding are not required.

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
        evs = self._require_evidence(evidence)
        # Provenance is diagnostic only; each listed property gets the marker.
        by_prop: dict[str, list[EvidenceRef]] = {p: list(evs) for p in props}
        for prop in sorted(props):
            attr = "necessity_rationale" if prop == "rationale" else prop
            setattr(node, attr, DontKnowType(evidence=by_prop[prop]))
        return f"Recorded DONT_KNOW on {node_id} for: {', '.join(sorted(props))}."

    @is_tool(ToolType.WRITE)
    def record_edge_condition_dont_know(
        self, edge_id: str, evidence: Optional[list] = None
    ) -> str:
        """Record an explicit DONT_KNOW belief for an edge condition.

        ``DONT_KNOW`` is distinct from UNSET (not investigated) and ABSENT
        (explicitly unconditional). Evidence is optional diagnostic metadata;
        when supplied, only its shape and Observation ids are validated and
        it need not resolve to a private stakeholder condition slot.

        Args:
            edge_id: The edge whose condition is unknown.
            evidence: Optional diagnostic Observation spans. Exact quote and
                private semantic-slot binding are not required.

        Returns:
            A confirmation message.
        """
        self._edge(edge_id)
        evs = self._require_evidence(evidence)
        self._edge(edge_id).condition = DontKnowType(evidence=evs)
        return f"Recorded DONT_KNOW on edge {edge_id} condition."

    @is_tool(ToolType.WRITE)
    def record_absent(
        self,
        node_id: str,
        properties: list[str],
        evidence: Optional[list] = None,
    ) -> str:
        """Record explicit ABSENT beliefs for an existing node property.

        ``ABSENT`` is distinct from UNSET (not investigated) and DONT_KNOW.
        Evidence is optional diagnostic metadata; when supplied, only its
        shape and Observation ids are validated. It need not resolve to a
        stakeholder known-absent slot, and every listed property is marked
        regardless of private sidecar content.

        Args:
            node_id: The node whose properties are absent.
            properties: Properties established absent
                (activity/actor/system/reads/writes/rationale).
            evidence: Optional diagnostic Observation spans. Exact quote and
                semantic-slot binding are not required.

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
        evs = self._require_evidence(evidence)
        # Provenance is diagnostic only; each listed property gets the marker.
        by_prop: dict[str, list[EvidenceRef]] = {p: list(evs) for p in props}
        for prop in sorted(props):
            attr = "necessity_rationale" if prop == "rationale" else prop
            setattr(node, attr, AbsentType(evidence=by_prop[prop]))
        return f"Recorded ABSENT on {node_id} for: {', '.join(sorted(props))}."

    @is_tool(ToolType.WRITE)
    def record_edge_condition_absent(
        self, edge_id: str, evidence: Optional[list] = None
    ) -> str:
        """Record an explicit ABSENT belief for an edge condition.

        This represents an UNCONDITIONAL edge. ``ABSENT`` is distinct from
        UNSET (not investigated) and DONT_KNOW. Evidence is optional
        diagnostic metadata; when supplied, only its shape and Observation
        ids are validated and it need not resolve to a private stakeholder
        condition slot.

        Args:
            edge_id: The edge whose condition is absent.
            evidence: Optional diagnostic Observation spans. Exact quote and
                semantic-slot binding are not required.

        Returns:
            A confirmation message.
        """
        self._edge(edge_id)
        evs = self._require_evidence(evidence)
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
    def remove_edge(self, edge_id: str) -> str:
        """Remove one edge while preserving both endpoint nodes.

        This is an explicit hypothesis-revision operation: only the requested
        edge is deleted; endpoint nodes and every unrelated edge remain
        unchanged.
        """
        graph = self._graph()
        self._edge(edge_id)  # require the edge to exist before mutating state
        del graph.edges[edge_id]
        return f"Removed edge {edge_id}."

    @is_tool(ToolType.WRITE)
    def add_edge(
        self,
        edge_id: str,
        from_node: str,
        to_node: str,
        condition=None,
        evidence: Optional[list] = None,
    ) -> str:
        """Add a directed edge between two existing nodes.

        Structural endpoint validity is required; stakeholder evidence is
        not. ``condition`` accepts a condition concept id or
        ``{"concept_id", "evidence"}``; all evidence is optional diagnostic
        metadata and never a private-provenance recording gate.

        Args:
            edge_id: Your own identifier for this edge.
            from_node: Source node id.
            to_node: Destination node id.
            condition: Condition concept id / {concept_id, evidence}.
            evidence: Optional diagnostic EvidenceRefs for this relation.

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
        # Markers are diagnostic only; no private-provenance binding needed.
        bound_edge: Optional[str] = None
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
        """Update an edge's endpoints, condition or diagnostic evidence.

        Endpoints and referenced concept kinds are validated structurally.
        Optional evidence is shape-checked (including supplied Observation
        ids) but is never required to support the edge or its condition.
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
            edge.condition = UNSET
        elif condition is not None:
            edge.condition = self._ref_from_prop_arg(
                condition,
                "update_edge condition",
                "condition",
                bound_edge=None,
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

        Refuses a structurally invalid graph or missing declared endpoints.
        It does NOT inspect any concept validation status: the concept
        lifecycle is gone. Completing the interview successfully terminates
        the episode immediately.

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
        # No provenance gate: referenced concepts do NOT need grounding to
        # finish. The Agent's reconstruction is judged against Truth later.
        self.db.interview_complete = True
        if summary:
            self.db.summary = summary
        return "Interview marked complete."

    # ------------------------------------------------------------- assertions

    def _evaluate(self, sc) -> EvaluationResult:
        """Evaluate AgentGraph reconstruction against Truth.

        The private sidecar ledger is passed only for diagnostic metrics and
        simulator-integrity reporting; it never gates Truth reconstruction.
        """
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
