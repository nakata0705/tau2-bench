"""Private provenance ledger (simulator/evaluator-only).

The stakeholder's replies carry a **private sidecar** alongside the public
message:

    {
      "message": "I check customer information in CRM.",
      "annotations": [
        {"semantic_id": "node:cc:activity",
         "quote": "check customer information", "occurrence": 0},
        {"semantic_id": "node:cc:reads:skc_customer",
         "quote": "customer information", "occurrence": 0},
        {"semantic_id": "node:cc:system",
         "quote": "CRM", "occurrence": 0}
      ],
      "alignments": [
        {"semantic_id": "skc_quote", "quote": "Yes.",
         "occurrence": 0, "act": "confirm"}
      ],
      "terminology": [
        {"semantic_id": "skc_customer",
         "proposed_term": "customer master", "quote": "Yes.",
         "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation. ``annotations`` anchor business
facts directly to **stakeholder semantic IDs** (graph elements of the
StakeholderKnowledgeGraph, including DONT_KNOW slots); ``alignments`` /
``terminology`` are private semantic dialogue events addressing
StakeholderKnowledgeConcept ids (a bare "Yes." does not encode the act).

The environment stores all three privately against the exact message/turn
(``SemanticLedger``). Provenance is never reconstructed from message text:
the evaluator links an Agent ``EvidenceRef`` to an annotation by
character-span correspondence (containment), never by meaning.

- ``StakeholderKnowledgeCatalog`` deterministically validates sidecar
  metadata: every semantic id exists in the stakeholder knowledge (graph
  element or knowledge concept) and every quote/occurrence exactly matches
  the message.
- The ledger is private: never part of ``InterviewDB``, never serialized
  into Agent-visible state, never exposed through tools.

Semantic ids, knowledge concept ids and event metadata never appear in
Agent-visible messages, tools, Observations, summaries, or serialized Agent
state.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

AlignmentAct = Literal["confirm", "partial", "unknown", "dispute"]


class SemanticAnnotation(BaseModel):
    """One private annotation: an Observation span resolves to ONE
    stakeholder semantic ID (a graph element of StakeholderKnowledgeGraph,
    including DONT_KNOW slots).

    ``quote`` must be an exact substring of the stakeholder's message and
    ``occurrence`` must select an existing occurrence of that substring.
    """

    semantic_id: str
    quote: str = Field(description="Exact substring of the stakeholder message.")
    occurrence: int = Field(
        default=0, description="0-based occurrence index of ``quote``."
    )

    @field_validator("quote")
    @classmethod
    def _quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("annotation quote must not be empty")
        return v


class ConceptAlignmentAssertion(BaseModel):
    """A private semantic dialogue event: the stakeholder performed a
    concept-identity dialogue act (confirm / partial / unknown / dispute)
    about ONE StakeholderKnowledgeConcept in this message.

    This is NOT a business fact. The stakeholder emits it only when the reply
    genuinely performs that act (e.g. answering "Yes." to "do you mean the
    quotation?"). ``quote``/``occurrence`` anchor the act to an exact span of
    the message.
    """

    semantic_id: str = Field(
        description="The StakeholderKnowledgeConcept id the act is about."
    )
    quote: str = Field(description="Exact span of the message that performs the act.")
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
    a term the interviewer proposed for ONE StakeholderKnowledgeConcept.

    ``proposed_term`` is the exact term the interviewer proposed;
    ``quote``/``occurrence`` anchor the agreement to an exact span of the
    message. Ordinary workflow speech that merely uses a word creates NO such
    event.
    """

    semantic_id: str = Field(
        description="The StakeholderKnowledgeConcept id the term is about."
    )
    proposed_term: str = Field(description="The exact term the interviewer proposed.")
    quote: str = Field(description="Exact span of the message that performs the agreement.")
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

    The catalog is per-scenario / per-stakeholder: every semantic id in it
    exists in the stakeholder knowledge (graph element ids or knowledge
    concept ids). Validation raises ``ValueError`` on any unknown semantic id
    or a quote/occurrence that does not exactly match the message.
    """

    def __init__(self, stakeholder_name: str, knowledge) -> None:
        self.stakeholder_name = stakeholder_name
        self.knowledge = knowledge
        self._graph_ids = knowledge.graph.semantic_ids()
        self._concept_ids = set(knowledge.graph.concepts)

    @classmethod
    def from_scenario(cls, scenario) -> "StakeholderKnowledgeCatalog":
        return cls(stakeholder_name=scenario.stakeholder.name, knowledge=scenario.knowledge)

    def validate_annotations(
        self,
        annotations: list[SemanticAnnotation],
        message: Optional[str],
    ) -> None:
        for annotation in annotations:
            if annotation.semantic_id not in self._graph_ids:
                raise ValueError(
                    f"annotation semantic_id {annotation.semantic_id!r} is not "
                    f"an element of stakeholder {self.stakeholder_name!r}'s "
                    f"knowledge graph"
                )
            if not message_contains_span(
                message, annotation.quote, annotation.occurrence
            ):
                raise ValueError(
                    f"annotation {annotation.semantic_id!r}: quote "
                    f"{annotation.quote!r} occurrence {annotation.occurrence} "
                    f"does not exactly match the stakeholder message"
                )

    def validate_events(
        self,
        alignments: Optional[list[ConceptAlignmentAssertion]],
        terminology: Optional[list[TerminologyConfirmation]],
        message: Optional[str],
    ) -> None:
        for event in alignments or []:
            if event.semantic_id not in self._concept_ids:
                raise ValueError(
                    f"alignment event for concept {event.semantic_id!r} is not "
                    f"a knowledge concept of stakeholder {self.stakeholder_name!r}"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"alignment event {event.act!r} for {event.semantic_id!r}: "
                    f"quote {event.quote!r} occurrence {event.occurrence} does "
                    f"not exactly match the stakeholder message"
                )
        for event in terminology or []:
            if event.semantic_id not in self._concept_ids:
                raise ValueError(
                    f"terminology event for concept {event.semantic_id!r} is not "
                    f"a knowledge concept of stakeholder {self.stakeholder_name!r}"
                )
            if not message_contains_span(message, event.quote, event.occurrence):
                raise ValueError(
                    f"terminology event for {event.semantic_id!r}: quote "
                    f"{event.quote!r} occurrence {event.occurrence} does not "
                    f"exactly match the stakeholder message"
                )


class SemanticLedger:
    """Private storage of the stakeholder sidecar keyed by conversation turn
    (the index of the message in ``InterviewDB.messages``).

    The ledger is the **only** source of provenance: nothing is ever derived
    from natural-language text. It is not part of ``InterviewDB`` and is
    never exposed to the Agent.
    """

    def __init__(self) -> None:
        self._catalog: Optional[StakeholderKnowledgeCatalog] = None
        self._by_turn: dict[int, list[SemanticAnnotation]] = {}
        self._alignment_by_turn: dict[int, list[ConceptAlignmentAssertion]] = {}
        self._terminology_by_turn: dict[int, list[TerminologyConfirmation]] = {}

    def install_catalog(self, catalog: StakeholderKnowledgeCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> Optional[StakeholderKnowledgeCatalog]:
        return self._catalog

    def bind(
        self,
        turn: int,
        annotations: list[SemanticAnnotation],
        message: Optional[str] = None,
    ) -> None:
        """Record the annotation sidecar of the stakeholder message at
        ``turn`` (validated deterministically when a catalog is installed)."""
        if self._catalog is not None:
            self._catalog.validate_annotations(annotations, message)
        self._by_turn[turn] = list(annotations)

    def bind_alignment(
        self,
        turn: int,
        events: list[ConceptAlignmentAssertion],
        message: Optional[str] = None,
    ) -> None:
        if self._catalog is not None:
            self._catalog.validate_events(events, None, message)
        self._alignment_by_turn[turn] = list(events)

    def bind_terminology(
        self,
        turn: int,
        events: list[TerminologyConfirmation],
        message: Optional[str] = None,
    ) -> None:
        if self._catalog is not None:
            self._catalog.validate_events(None, events, message)
        self._terminology_by_turn[turn] = list(events)

    def annotations(self) -> dict[int, list[SemanticAnnotation]]:
        """Copy of ``{turn: [annotations]}`` (evaluator-only)."""
        return {turn: list(a) for turn, a in self._by_turn.items()}

    def alignments(self) -> dict[int, list[ConceptAlignmentAssertion]]:
        return {turn: list(e) for turn, e in self._alignment_by_turn.items()}

    def terminology(self) -> dict[int, list[TerminologyConfirmation]]:
        return {turn: list(e) for turn, e in self._terminology_by_turn.items()}
