"""Private stakeholder facts + used-fact sidecar ledger (simulator/evaluator-only).

The stakeholder simulator's knowledge is a set of hidden, structured
``StakeholderFact``s:

    StakeholderFact:
        id: str
        text: str                      # natural wording the simulator may use
        supported_claim_ids: list[str] # hidden TruthClaims this fact supports

When the stakeholder LLM answers, it returns a **private sidecar** alongside
its natural-language message: the ids of the facts it actually used. Only the
message enters the conversation; the sidecar is stored privately against that
exact message/turn (``StakeholderFactLedger``). Provenance is **never
reconstructed from message text** — the evaluator consumes the stored sidecar.

- ``StakeholderFactCatalog`` deterministically validates sidecar metadata:
  every used fact id exists, belongs to this stakeholder, every supported
  TruthClaim exists, and every claim is stakeholder-visible.
- ``StakeholderFactLedger`` stores ``used_fact_ids`` keyed by conversation
  turn. It is private: it is never part of ``InterviewDB``, never serialized
  into Agent-visible state, and never exposed through tools.

Fact ids and claim ids never appear in Agent-visible messages, tools,
Observations, summaries, or serialized Agent state.
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.stakeholder import StakeholderFilter


class StakeholderFact(BaseModel):
    """One hidden structured business fact of a stakeholder.

    ``id`` is simulator/evaluator-private; ``text`` is the natural wording the
    stakeholder may use (the simulator naturalizes it — it need not copy it);
    ``supported_claim_ids`` are the hidden TruthClaims this fact supports.
    """

    id: str
    text: str
    supported_claim_ids: list[str] = Field(default_factory=list)


class StakeholderFactCatalog:
    """Deterministic validation of private fact metadata for one stakeholder.

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

    def validate_used_fact_ids(self, fact_ids: list[str]) -> None:
        """Reject invalid sidecar metadata deterministically.

        Raises ``ValueError`` if any used id is unknown, belongs to a
        different stakeholder (equivalently: is not in this catalog), or
        supports a claim that does not exist / is outside stakeholder
        visibility.
        """
        for fid in fact_ids:
            fact = self.facts.get(fid)
            if fact is None:
                raise ValueError(
                    f"used_fact_id {fid!r} is not in stakeholder "
                    f"{self.stakeholder_name!r}'s fact catalog"
                )
            for cid in fact.supported_claim_ids:
                claim = self.claims.get(cid)
                if claim is None:
                    raise ValueError(
                        f"fact {fid!r} supports unknown TruthClaim id {cid!r}"
                    )
                if claim.axis not in self.stakeholder.visible_attributes_for(
                    claim.node_id
                ):
                    raise ValueError(
                        f"fact {fid!r} supports claim {cid!r} which is outside "
                        f"stakeholder visibility"
                    )

    def claims_supported_by(self, fact_ids: list[str]) -> list[TruthClaim]:
        """The hidden TruthClaims supported by the given used fact ids.

        Deterministic resolution; callers must validate first (or rely on
        ``evaluate``'s own validation).
        """
        seen: dict[str, TruthClaim] = {}
        for fid in fact_ids:
            fact = self.facts.get(fid)
            if fact is None:
                raise ValueError(f"used_fact_id {fid!r} is not in the fact catalog")
            for cid in fact.supported_claim_ids:
                claim = self.claims.get(cid)
                if claim is None:
                    raise ValueError(
                        f"fact {fid!r} supports unknown TruthClaim id {cid!r}"
                    )
                seen[cid] = claim
        return list(seen.values())


class StakeholderFactLedger:
    """Private storage of stakeholder ``used_fact_ids`` keyed by conversation
    turn (the index of the message in ``InterviewDB.messages``).

    The ledger is the **only** source of fact provenance: nothing is ever
    derived from natural-language text. It is not part of ``InterviewDB`` and
    is never exposed to the Agent.
    """

    def __init__(self) -> None:
        self._catalog: Optional[StakeholderFactCatalog] = None
        self._by_turn: dict[int, list[str]] = {}

    def install_catalog(self, catalog: StakeholderFactCatalog) -> None:
        """Attach the stakeholder's fact catalog (enables ingestion-time
        validation). Installed by the stakeholder simulator adapter."""
        self._catalog = catalog

    @property
    def catalog(self) -> Optional[StakeholderFactCatalog]:
        return self._catalog

    def bind(self, turn: int, fact_ids: list[str]) -> None:
        """Record the sidecar of the stakeholder message at ``turn``.

        When a catalog is installed, the ids are validated deterministically
        first (unknown id / foreign fact / unknown or hidden claim => raise).
        """
        if self._catalog is not None:
            self._catalog.validate_used_fact_ids(fact_ids)
        self._by_turn[turn] = list(dict.fromkeys(fact_ids))

    def used_fact_ids(self) -> dict[int, list[str]]:
        """Copy of ``{turn: [used fact ids]}`` (evaluator-only)."""
        return {turn: list(ids) for turn, ids in self._by_turn.items()}
