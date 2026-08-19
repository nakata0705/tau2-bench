"""Private stakeholder facts + assertion sidecar ledger (simulator/evaluator-only).

The stakeholder simulator's knowledge is a set of hidden, structured
``StakeholderFact``\\ s:

    StakeholderFact:
        id: str
        text: str                      # natural wording the simulator may use
        supported_claim_ids: list[str] # hidden TruthClaims this fact supports

Facts are **atomic**: prefer exactly one supported TruthClaim per fact so a
fact never cross-credits independent claims.

When the stakeholder LLM answers, it returns a **private assertion sidecar**
alongside its natural-language message:

    {
      "message": "I check the customer in CRM, then prepare the quotation.",
      "assertions": [
        {"fact_id": "quotation.check.system", "quote": "CRM", "occurrence": 0},
        {"fact_id": "quotation.create.activity",
         "quote": "prepare the quotation", "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation. Each assertion's ``quote`` /
``occurrence`` must exactly match a span of that message; the environment
stores the assertions privately against that exact message/turn
(``StakeholderFactLedger``). Provenance is **never reconstructed from message
text** — the evaluator links an Agent ``EvidenceRef`` to an assertion by exact
span only.

- ``StakeholderFactCatalog`` deterministically validates sidecar metadata:
  every fact id exists and belongs to this stakeholder, every supported
  TruthClaim exists and is stakeholder-visible.
- ``StakeholderAssertion`` validation additionally checks that
  ``quote``/``occurrence`` exactly match the message text.
- ``StakeholderFactLedger`` stores assertions keyed by conversation turn. It
  is private: it is never part of ``InterviewDB``, never serialized into
  Agent-visible state, and never exposed through tools.

Fact ids and claim ids never appear in Agent-visible messages, tools,
Observations, summaries, or serialized Agent state.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.stakeholder import StakeholderFilter


class StakeholderFact(BaseModel):
    """One hidden structured business fact of a stakeholder.

    ``id`` is simulator/evaluator-private; ``text`` is the natural wording the
    stakeholder may use (the simulator naturalizes it — it need not copy it);
    ``supported_claim_ids`` are the hidden TruthClaims this fact supports
    (prefer exactly one — atomic facts).
    """

    id: str
    text: str
    supported_claim_ids: list[str] = Field(default_factory=list)


class StakeholderAssertion(BaseModel):
    """One private annotation: a fact used, anchored to an exact message span.

    ``quote`` must be an exact substring of the stakeholder's message and
    ``occurrence`` must select an existing occurrence of that substring.
    """

    fact_id: str
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


class StakeholderFactCatalog:
    """Deterministic validation of private fact/assertion metadata.

    The catalog is per-scenario / per-stakeholder: every fact in it belongs to
    exactly that stakeholder. Validation raises ``ValueError`` on any unknown
    fact id, unknown supported claim, or claim outside stakeholder visibility.
    """

    def __init__(
        self,
        stakeholder_name: str,
        facts: dict[str, StakeholderFact],
        claims: dict[str, TruthClaim],
        stakeholder: StakeholderFilter,
    ) -> None:
        self.stakeholder_name = stakeholder_name
        self.facts = dict(facts)
        self.claims = dict(claims)
        self.stakeholder = stakeholder

    @classmethod
    def from_scenario(cls, scenario) -> "StakeholderFactCatalog":
        """Build a catalog from a Scenario (facts + claims + filter)."""
        return cls(
            stakeholder_name=scenario.stakeholder.name,
            facts=scenario.facts,
            claims=scenario.claims,
            stakeholder=scenario.stakeholder,
        )

    def validate_assertions(
        self, assertions: list[StakeholderAssertion], message: Optional[str]
    ) -> None:
        """Reject invalid sidecar metadata deterministically.

        Raises ``ValueError`` if any used fact id is unknown (equivalently:
        belongs to a different stakeholder), any supported claim does not
        exist / is outside stakeholder visibility, or the assertion's
        quote/occurrence do not exactly match the message.
        """
        for assertion in assertions:
            fact = self.facts.get(assertion.fact_id)
            if fact is None:
                raise ValueError(
                    f"assertion fact_id {assertion.fact_id!r} is not in "
                    f"stakeholder {self.stakeholder_name!r}'s fact catalog"
                )
            for cid in fact.supported_claim_ids:
                claim = self.claims.get(cid)
                if claim is None:
                    raise ValueError(
                        f"fact {assertion.fact_id!r} supports unknown "
                        f"TruthClaim id {cid!r}"
                    )
                if not self._claim_visible(claim):
                    raise ValueError(
                        f"fact {assertion.fact_id!r} supports claim {cid!r} "
                        f"which is outside stakeholder visibility"
                    )
            if not message_contains_span(
                message, assertion.quote, assertion.occurrence
            ):
                raise ValueError(
                    f"assertion {assertion.fact_id!r}: quote "
                    f"{assertion.quote!r} occurrence {assertion.occurrence} "
                    f"does not exactly match the stakeholder message"
                )

    def _claim_visible(self, claim: TruthClaim) -> bool:
        if claim.subject_kind == "node":
            return claim.property in self.stakeholder.node_properties_for(
                claim.subject_id
            )
        if claim.subject_id not in self.stakeholder.visible_edge_ids:
            return False
        if claim.property == "condition":
            return "condition" in self.stakeholder.edge_properties_for(claim.subject_id)
        return claim.property == "edge_exists"

    def claims_for_assertions(
        self, assertions: list[StakeholderAssertion]
    ) -> dict[str, TruthClaim]:
        """Resolve assertions to their supported TruthClaims (by id).

        Deterministic; callers must validate first (or rely on the
        evaluator's own validation).
        """
        out: dict[str, TruthClaim] = {}
        for assertion in assertions:
            fact = self.facts.get(assertion.fact_id)
            if fact is None:
                raise ValueError(
                    f"assertion fact_id {assertion.fact_id!r} is not in the fact catalog"
                )
            for cid in fact.supported_claim_ids:
                claim = self.claims.get(cid)
                if claim is None:
                    raise ValueError(
                        f"fact {assertion.fact_id!r} supports unknown "
                        f"TruthClaim id {cid!r}"
                    )
                out[cid] = claim
        return out


class StakeholderFactLedger:
    """Private storage of stakeholder assertions keyed by conversation turn
    (the index of the message in ``InterviewDB.messages``).

    The ledger is the **only** source of fact provenance: nothing is ever
    derived from natural-language text. It is not part of ``InterviewDB`` and
    is never exposed to the Agent.
    """

    def __init__(self) -> None:
        self._catalog: Optional[StakeholderFactCatalog] = None
        self._by_turn: dict[int, list[StakeholderAssertion]] = {}

    def install_catalog(self, catalog: StakeholderFactCatalog) -> None:
        """Attach the stakeholder's fact catalog (enables ingestion-time
        validation). Installed by the stakeholder simulator adapter."""
        self._catalog = catalog

    @property
    def catalog(self) -> Optional[StakeholderFactCatalog]:
        return self._catalog

    def bind(
        self,
        turn: int,
        assertions: list[StakeholderAssertion],
        message: Optional[str] = None,
    ) -> None:
        """Record the assertion sidecar of the stakeholder message at ``turn``.

        When a catalog is installed, the assertions are validated
        deterministically first (unknown/foreign fact, unknown or hidden
        claim, or quote/occurrence mismatch => raise).
        """
        if self._catalog is not None:
            self._catalog.validate_assertions(assertions, message)
        self._by_turn[turn] = list(assertions)

    def assertions(self) -> dict[int, list[StakeholderAssertion]]:
        """Copy of ``{turn: [assertions]}`` (evaluator-only)."""
        return {turn: list(ass) for turn, ass in self._by_turn.items()}
