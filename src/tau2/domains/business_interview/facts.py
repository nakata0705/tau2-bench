"""Stakeholder knowledge + assertion sidecar ledger (simulator/evaluator-only).

The stakeholder's knowledge is a **projection of the Truth graph** — pure
semantic structure, never authored workflow sentences:

    StakeholderKnowledge:
        visible_claim_ids      # claims this stakeholder can assert
        contextual_knowledge   # graph positions (incoming edges, is_start) + claims
        concept_views          # local lexical preferences: tc_quote -> "quotation"

``concept_views`` are single wordings (not sentences) the stakeholder
naturally uses for concepts; they carry no workflow facts.

When the stakeholder LLM answers, it returns a **private assertion sidecar**
alongside its natural-language message, anchoring each claim it used to an
exact span of the message:

    {
      "message": "I check the customer in CRM, then prepare the quotation.",
      "assertions": [
        {"claim_id": "cc.system", "quote": "CRM", "occurrence": 0},
        {"claim_id": "cq.activity",
         "quote": "prepare the quotation", "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation. Each assertion's ``quote`` /
``occurrence`` must exactly match a span of that message; the environment
stores the assertions privately against that exact message/turn
(``StakeholderAssertionLedger``). Provenance is **never reconstructed from
message text** — the evaluator links an Agent ``EvidenceRef`` to an assertion
by character-span correspondence (containment), never by meaning.

- ``StakeholderKnowledgeCatalog`` deterministically validates sidecar
  metadata: every claim id exists and is stakeholder-visible.
- ``StakeholderAssertion`` validation additionally checks that
  ``quote``/``occurrence`` exactly match the message text.
- ``StakeholderAssertionLedger`` stores assertions keyed by conversation turn.
  It is private: it is never part of ``InterviewDB``, never serialized into
  Agent-visible state, and never exposed through tools.

Claim ids never appear in Agent-visible messages, tools, Observations,
summaries, or serialized Agent state.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.graph import TruthNodeContext
from tau2.domains.business_interview.stakeholder import StakeholderFilter


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


class StakeholderKnowledge(BaseModel):
    """The semantic knowledge of one stakeholder: a projection of the Truth.

    Pure structure: visible claim ids, the contextual graph knowledge they
    belong to (node contexts with ALL incoming edges + start flag), and
    concept views (local lexical preferences, e.g. ``tc_quote ->
    "quotation"`` — never workflow sentences).
    """

    visible_claim_ids: list[str] = Field(default_factory=list)
    contextual_knowledge: dict[str, TruthNodeContext] = Field(default_factory=dict)
    concept_views: dict[str, str] = Field(default_factory=dict)


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
    """Deterministic validation of private assertion metadata.

    The catalog is per-scenario / per-stakeholder: every claim in it is
    visible to exactly that stakeholder. Validation raises ``ValueError`` on
    any unknown claim id, or a quote/occurrence that does not exactly match
    the message.
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


class StakeholderAssertionLedger:
    """Private storage of stakeholder assertions keyed by conversation turn
    (the index of the message in ``InterviewDB.messages``).

    The ledger is the **only** source of provenance: nothing is ever derived
    from natural-language text. It is not part of ``InterviewDB`` and is never
    exposed to the Agent.
    """

    def __init__(self) -> None:
        self._catalog: Optional[StakeholderKnowledgeCatalog] = None
        self._by_turn: dict[int, list[StakeholderAssertion]] = {}

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

    def assertions(self) -> dict[int, list[StakeholderAssertion]]:
        """Copy of ``{turn: [assertions]}`` (evaluator-only)."""
        return {turn: list(ass) for turn, ass in self._by_turn.items()}
