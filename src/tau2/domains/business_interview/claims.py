"""Hidden Truth claims (evaluator/simulator-private) — no surface terms.

A ``TruthClaim`` is one hidden read/write fact of the benchmark: a Truth node +
axis + Truth data concept (e.g. ``cq.writes.tc_quote``). Claims exist **only**
for stakeholder-visible axes; hidden axes produce no claims and can never enter
any catalog.

Claims are referenced by private ``StakeholderFact``s (``facts.py``): a fact
declares which Truth claims it supports, and the stakeholder simulator returns
the ids of the facts it actually used. Nothing here is derived from
natural-language text — the evaluator never parses stakeholder wording.

The Agent never sees claim ids, the claim catalog, or the fact ledger: they
live only in evaluator/simulator-private state.
"""

from typing import Literal

from pydantic import BaseModel

from tau2.domains.business_interview.dag import BusinessDAG
from tau2.domains.business_interview.stakeholder import StakeholderFilter

DataAxis = Literal["reads", "writes"]


class TruthClaim(BaseModel):
    """One hidden Truth claim: a stakeholder-visible read/write fact.

    ``id`` binds a Truth node + axis + Truth data concept (e.g.
    ``cq.writes.tc_quote``). Claim ids are evaluator/simulator-private and
    never visible to the Agent.
    """

    id: str
    node_id: str
    axis: DataAxis
    concept_id: str


def build_claims(
    truth: BusinessDAG,
    stakeholder: StakeholderFilter,
) -> dict[str, TruthClaim]:
    """Build the private claim catalog from the Truth DAG + visibility.

    One claim per Truth read/write concept ref on a **stakeholder-visible**
    axis. Claim ids look like ``cq.writes.tc_quote``. Hidden axes (sq.reads,
    sq.writes, ...) produce no claims and can never enter the catalog.
    """
    claims: dict[str, TruthClaim] = {}
    for nid, node in truth.nodes.items():
        visible = stakeholder.visible_attributes_for(nid)
        for axis in ("reads", "writes"):
            if axis not in visible:
                continue
            for ref in getattr(node, axis):
                cid = f"{nid}.{axis}.{ref.concept_id}"
                claims[cid] = TruthClaim(
                    id=cid,
                    node_id=nid,
                    axis=axis,  # type: ignore[arg-type]
                    concept_id=ref.concept_id,
                )
    return claims
