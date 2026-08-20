"""Stakeholder knowledge + private sidecar ledger (simulator/evaluator-only).

The stakeholder's knowledge is a **physical projection of the Truth graph** —
pure semantic structure, never authored workflow sentences:

    StakeholderKnowledge:
        visible_claim_ids      # claims this stakeholder can assert
        contextual_knowledge   # positions (incoming edges, is_start) of visible nodes
        visible_node_ids       # nodes the stakeholder can talk about
        visible_edge_ids       # relations whose existence/condition it can assert
        concept_views          # wordings ONLY for visible claim concepts

``concept_views`` are single wordings (not sentences) the stakeholder
naturally uses for concepts; they carry no workflow facts. Hidden concepts
have NO lexical view and hidden relations never appear in the knowledge.

When the stakeholder LLM answers, it returns a **private sidecar** alongside
its natural-language message:

    {
      "message": "I check the customer in CRM, then prepare the quotation.",
      "assertions": [
        {"claim_id": "cc.system", "quote": "CRM", "occurrence": 0},
        {"claim_id": "cq.activity",
         "quote": "prepare the quotation", "occurrence": 0}
      ],
      "alignments": [
        {"truth_concept_id": "tc_quote", "quote": "Yes.",
         "occurrence": 0, "act": "confirm"}
      ],
      "terminology": [
        {"truth_concept_id": "tc_customer",
         "proposed_term": "customer master", "quote": "Yes.",
         "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation. ``assertions`` anchor the claims
used to exact message spans; ``alignments`` / ``terminology`` are **private
semantic dialogue events** (concept-identity and terminology agreements) that
must NOT be confused with ordinary business claims — the stakeholder emits
them only when the reply genuinely performs that dialogue act (e.g. answering
"Yes." to "do you mean X?" or "can we call this Y?"). The environment
stores all three privately against that exact message/turn
(``StakeholderAssertionLedger``). Provenance is **never reconstructed from
message text** — the evaluator links an Agent ``EvidenceRef`` to an assertion
or event by character-span correspondence (containment), never by meaning.

- ``StakeholderKnowledgeCatalog`` deterministically validates sidecar
  metadata: every claim id exists and is stakeholder-visible; every dialogue
  event references a visible concept and an exact message span.
- ``StakeholderAssertion`` / dialogue-event validation additionally checks
  that ``quote``/``occurrence`` exactly match the message text.
- ``StakeholderAssertionLedger`` stores assertions and events keyed by
  conversation turn. It is private: it is never part of ``InterviewDB``,
  never serialized into Agent-visible state, and never exposed through tools.

Claim ids, truth concept ids and event metadata never appear in Agent-visible
messages, tools, Observations, summaries, or serialized Agent state.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.graph import TruthNodeContext
from tau2.domains.business_interview.stakeholder import StakeholderFilter

AlignmentAct = Literal["confirm", "partial", "unknown", "dispute"]


class StakeholderAssertion(BaseModel):
    """One private annotation: a claim used, anchored to an exact message span.

    ``quote`` must be an exact substring of the stakeholder's message and
    ``occurrence`` must select an existing occurrence of that substring.
    """

    claim_id: str
    quote: str = Field(description="Exact substring of the stakeholder message.")
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("assertion quote must not be empty")
        return v


class ConceptAlignmentAssertion(BaseModel):
    """A private semantic dialogue event: the stakeholder performed a
    concept-identity dialogue act (confirm / partial / unknown / dispute)
    about ONE Truth concept in this message.

    This is NOT a business claim: it does not express a workflow fact and
    must not be confused with a ``StakeholderAssertion``. The stakeholder
    emits it only when the reply genuinely performs that act (e.g. answering
    "Yes." to "do you mean the quotation?"). ``quote``/``occurrence`` anchor
    the act to an exact span of the message.
    """

    truth_concept_id: str
    quote: str = Field(
        description="Exact span of the message that performs the act."
    )
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )
    act: AlignmentAct

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("alignment quote must not be empty")
        return v


class TerminologyConfirmation(BaseModel):
    """A private semantic dialogue event: the stakeholder explicitly agreed to
    a term the interviewer proposed for ONE Truth concept.

    ``proposed_term`` is the exact term the interviewer proposed;
    ``quote``/``occurrence`` anchor the agreement to an exact span of the
    message. Ordinary workflow speech that merely uses a word creates NO such
    event.
    """

    truth_concept_id: str
    proposed_term: str = Field(
        description="The exact term the interviewer proposed."
    )
    quote: str = Field(
        description="Exact span of the message that performs the agreement."
    )
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )

    @field_validator("proposed_term")
    @classmethod
    def _term_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("terminology proposed_term must not be empty")
        return v

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("terminology quote must not be empty")
        return v


class StakeholderKnowledge(BaseModel):
    """The semantic knowledge of one stakeholder: a physical projection of the
    Truth.

    Contains ONLY: visible claim ids, graph contexts the stakeholder is
    allowed to know (visible nodes with their visible incoming edges and the
    start flag), and concept views for concepts referenced by those visible
    claims/contexts. Hidden concepts have no lexical view and hidden
    relations never appear. Built by ``project_knowledge`` — never assembled
    by hand from the full Truth.
    """

    visible_claim_ids: list[str] = Field(default_factory=list)
    contextual_knowledge: dict[str, TruthNodeContext] = Field(default_factory=dict)
    concept_views: dict[str, str] = Field(default_factory=dict)
    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_ids: list[str] = Field(default_factory=list)


def project_knowledge(
    truth,
    stakeholder: StakeholderFilter,
    claims: dict[str, TruthClaim],
    views: dict[str, str],
) -> StakeholderKnowledge:
    """Build one stakeholder's knowledge as a **physical projection** of the
    Truth graph — never the Truth with a filter beside it.

    The knowledge contains ONLY:
    - the visible claims (``build_claims`` output — hidden contexts/properties
      produce no claims);
    - graph contexts for nodes the stakeholder can actually talk about, with
      their incoming edges restricted to visible relations and the start flag;
    - concept views for exactly the concepts those visible claims reference.

    Hidden concepts get no lexical view and hidden relations never appear in
    the knowledge, so nothing can leak into a stakeholder prompt through
    ``concept_views`` or ``contextual_knowledge``.
    """
    visible_concepts: set[str] = {
        c.concept_id for c in claims.values() if c.concept_id is not None
    }
    visible_nodes = sorted(
        nid
        for nid in stakeholder.visible_node_ids
        if nid in truth.nodes and any(c.context_id == nid for c in claims.values())
    )
    visible_edges = sorted(
        eid
        for eid in stakeholder.visible_edge_ids
        if eid in truth.edges
        and truth.edges[eid].from_node in visible_nodes
        and truth.edges[eid].to_node in visible_nodes
    )
    contexts: dict[str, TruthNodeContext] = {}
    for nid in visible_nodes:
        contexts[nid] = TruthNodeContext(
            node_id=nid,
            incoming_edge_ids=sorted(
                e for e in truth.incoming_edges(nid) if e in visible_edges
            ),
            is_start=(nid == truth.start_node_id),
        )
    return StakeholderKnowledge(
        visible_claim_ids=sorted(claims),
        contextual_knowledge=contexts,
        concept_views={
            cid: word
            for cid, word in views.items()
            if cid in visible_concepts and cid in truth_concepts(truth)
        },
        visible_node_ids=visible_nodes,
        visible_edge_ids=visible_edges,
    )


def truth_concepts(truth) -> set[str]:
    """All Truth concept ids of a graph (for view filtering)."""
    return set(truth.concepts)


def message_contains_span(message: Optional[str], quote: str, occurrence: int) -> bool:
    """Deterministic check that ``quote`` occurs at index ``occurrence`` in
    the message text."""
    text = message or ""
    if not quote:
        return False
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start == -1:
            return False
    return True


class StakeholderKnowledgeCatalog:
    """Deterministic validation of private sidecar metadata.

    The catalog is per-scenario / per-stakeholder: every claim in it is
    visible to exactly that stakeholder, and every dialogue event references
    a concept that the visible claims reference. Validation raises
    ``ValueError`` on any unknown claim id, unknown concept id, or a
    quote/occurrence that does not exactly match the message.
    """

    def __init__(
        self,
        stakeholder_name: str,
        claims: dict[str, TruthClaim],
        stakeholder: StakeholderFilter,
        knowledge: StakeholderKnowledge,
    ) -> None:
        self.stakeholder_name = stakeholder_name
        self.claims = dict(claims)
        self.stakeholder = stakeholder
        self.knowledge = knowledge

    @classmethod
    def from_scenario(cls, scenario) -> "StakeholderKnowledgeCatalog":
        """Build a catalog from a Scenario (claims + filter + knowledge)."""
        return cls(
            stakeholder_name=scenario.stakeholder.name,
            claims=scenario.claims,
            stakeholder=scenario.stakeholder,
            knowledge=scenario.knowledge,
        )

    def visible_concept_ids(self) -> set[str]:
        """Truth concept ids the stakeholder can talk about (evaluator-only)."""
        return {
            c.concept_id
            for c in self.claims.values()
            if c.concept_id is not None
        }

    def validate_assertions(
        self, assertions: list[StakeholderAssertion], message: Optional[str]
    ) -> None:
        """Reject invalid sidecar metadata deterministically.

        Raises ``ValueError`` if any claim id is unknown (equivalently: not
        visible to this stakeholder), or the assertion's quote/occurrence do
        not exactly match the message.
        """
        for assertion in assertions:
            if assertion.claim_id not in self.claims:
                raise ValueError(
                    f"assertion claim_id {assertion.claim_id!r} is not in "
                    f"stakeholder {self.stakeholder_name!r}'s visible claims"
                )
            if not message_contains_span(
                message, assertion.quote, assertion.occurrence
            ):
                raise ValueError(
                    f"assertion {assertion.claim_id!r}: quote "
                    f"{assertion.quote!r} occurrence {assertion.occurrence} "
                    f"does not exactly match the stakeholder message"
                )

    def validate_events(
        self,
        alignments: Optional[list[ConceptAlignmentAssertion]],
        terminology: Optional[list[TerminologyConfirmation]],
        message: Optional[str],
    ) -> None:
        """Reject invalid private dialogue-event metadata deterministically.

        Raises ``ValueError`` if an event references a concept the stakeholder
        cannot talk about, or its quote/occurrence do not exactly match the
        message.
        """
        visible = self.visible_concept_ids()
        for event in alignments or []:
            if event.truth_concept_id not in visible:
                raise ValueError(
                    f"alignment event for concept {event.truth_concept_id!r} is "
                    f"not visible to stakeholder {self.stakeholder_name!r}"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"alignment event {event.act!r} for "
                    f"{event.truth_concept_id!r}: quote {event.quote!r} "
                    f"occurrence {event.occurrence} does not exactly match "
                    f"the stakeholder message"
                )
        for event in terminology or []:
            if event.truth_concept_id not in visible:
                raise ValueError(
                    f"terminology event for concept {event.truth_concept_id!r} is "
                    f"not visible to stakeholder {self.stakeholder_name!r}"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"terminology event for {event.truth_concept_id!r}: quote "
                    f"{event.quote!r} occurrence {event.occurrence} does not "
                    f"exactly match the stakeholder message"
                )


class StakeholderAssertionLedger:
    """Private storage of stakeholder assertions and dialogue events keyed by
    conversation turn (the index of the message in ``InterviewDB.messages``).

    The ledger is the **only** source of provenance: nothing is ever derived
    from natural-language text. It is not part of ``InterviewDB`` and is never
    exposed to the Agent.
    """

    def __init__(self) -> None:
        self._catalog: Optional[StakeholderKnowledgeCatalog] = None
        self._by_turn: dict[int, list[StakeholderAssertion]] = {}
        self._alignment_by_turn: dict[int, list[ConceptAlignmentAssertion]] = {}
        self._terminology_by_turn: dict[int, list[TerminologyConfirmation]] = {}

    def install_catalog(self, catalog: StakeholderKnowledgeCatalog) -> None:
        """Attach the stakeholder's knowledge catalog (enables ingestion-time
        validation). Installed by the stakeholder simulator adapter."""
        self._catalog = catalog

    @property
    def catalog(self) -> Optional[StakeholderKnowledgeCatalog]:
        return self._catalog

    def bind(
        self,
        turn: int,
        assertions: list[StakeholderAssertion],
        message: Optional[str] = None,
    ) -> None:
        """Record the assertion sidecar of the stakeholder message at ``turn``.

        When a catalog is installed, the assertions are validated
        deterministically first (unknown claim id or quote/occurrence mismatch
        => raise).
        """
        if self._catalog is not None:
            self._catalog.validate_assertions(assertions, message)
        self._by_turn[turn] = list(assertions)

    def bind_alignment(
        self,
        turn: int,
        events: list[ConceptAlignmentAssertion],
        message: Optional[str] = None,
    ) -> None:
        """Record the private concept-alignment events of the message at
        ``turn`` (validated deterministically when a catalog is installed)."""
        if self._catalog is not None:
            self._catalog.validate_events(events, None, message)
        self._alignment_by_turn[turn] = list(events)

    def bind_terminology(
        self,
        turn: int,
        events: list[TerminologyConfirmation],
        message: Optional[str] = None,
    ) -> None:
        """Record the private terminology-confirmation events of the message
        at ``turn`` (validated deterministically when a catalog is installed)."""
        if self._catalog is not None:
            self._catalog.validate_events(None, events, message)
        self._terminology_by_turn[turn] = list(events)

    def assertions(self) -> dict[int, list[StakeholderAssertion]]:
        """Copy of ``{turn: [assertions]}`` (evaluator-only)."""
        return {turn: list(ass) for turn, ass in self._by_turn.items()}

    def alignments(self) -> dict[int, list[ConceptAlignmentAssertion]]:
        """Copy of ``{turn: [concept-alignment events]}`` (evaluator-only)."""
        return {
            turn: list(ev) for turn, ev in self._alignment_by_turn.items()
        }

    def terminology(self) -> dict[int, list[TerminologyConfirmation]]:
        """Copy of ``{turn: [terminology-confirmation events]}``
        (evaluator-only)."""
        return {
            turn: list(ev) for turn, ev in self._terminology_by_turn.items()
        }
