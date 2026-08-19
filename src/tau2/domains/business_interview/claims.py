"""Hidden Truth claims (evaluator/simulator-private) — graph-contextual.

A ``TruthClaim`` is one hidden scored semantic property at a **workflow graph
position**:

    TruthClaim = graph position (context_id) + property + value (concept_id)

    cq.activity     -> tc_activity_create_quotation
    cq.actor        -> tc_actor_sales
    cq.reads        -> tc_customer
    e3.edge_exists  (concept_id None — structural fact)
    e3.condition    -> tc_cond_over_1m

The context (node incoming edges, start flag) is resolved from the Truth
``BusinessProcessGraph`` via ``TruthNodeContext`` — never duplicated in
claims. Claims exist **only** for stakeholder-visible contexts/properties;
hidden contexts/properties produce no claims and can never enter any catalog.

Claims are referenced directly by the stakeholder's private assertion sidecar
(``StakeholderAssertion.claim_id``). Nothing here is derived from
natural-language text.

The Agent never sees claim ids, the claim catalog, or the assertion ledger:
they live only in evaluator/simulator-private state.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from tau2.domains.business_interview.graph import BusinessProcessGraph
from tau2.domains.business_interview.stakeholder import StakeholderFilter

ClaimProperty = Literal[
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "condition",
    "rationale",
    "edge_exists",
]


class TruthClaim(BaseModel):
    """One hidden Truth claim: a scored semantic property at a graph position.

    ``context_id`` is the Truth node id (node claims) or Truth edge id (edge
    claims); ``property`` is the scored property; ``concept_id`` is the Truth
    BusinessConcept the property must bind to (``None`` for ``edge_exists``,
    which is purely structural).
    """

    id: str
    context_id: str
    property: ClaimProperty
    concept_id: Optional[str] = None


def _node_claim(nid: str, prop: str, concept_id: Optional[str]) -> TruthClaim:
    suffix = f".{concept_id}" if prop in ("reads", "writes") else ""
    return TruthClaim(
        id=f"{nid}.{prop}{suffix}",
        context_id=nid,
        property=prop,  # type: ignore[arg-type]
        concept_id=concept_id,
    )


def build_claims(
    truth: BusinessProcessGraph,
    stakeholder: StakeholderFilter,
) -> dict[str, TruthClaim]:
    """Build the private claim catalog from the Truth graph + visibility.

    One claim per visible node property (activity/actor/system/reads/writes/
    rationale) and per visible edge property (edge_exists always; condition
    only when visible). Hidden contexts/properties produce no claims.
    """
    claims: dict[str, TruthClaim] = {}
    for nid, node in truth.nodes.items():
        props = stakeholder.node_properties_for(nid)
        if "activity" in props and node.activity is not None:
            claims[f"{nid}.activity"] = _node_claim(
                nid, "activity", node.activity.concept_id
            )
        for prop in ("actor", "system", "necessity_rationale"):
            ref = getattr(node, prop)
            if ref is None:
                continue
            pname = "rationale" if prop == "necessity_rationale" else prop
            if pname in props:
                claims[f"{nid}.{pname}"] = _node_claim(nid, pname, ref.concept_id)
        for axis in ("reads", "writes"):
            if axis not in props:
                continue
            for ref in getattr(node, axis):
                claims[f"{nid}.{axis}.{ref.concept_id}"] = _node_claim(
                    nid, axis, ref.concept_id
                )
    for eid, edge in truth.edges.items():
        if eid not in stakeholder.visible_edge_ids:
            continue
        claims[f"{eid}.edge_exists"] = TruthClaim(
            id=f"{eid}.edge_exists",
            context_id=eid,
            property="edge_exists",  # type: ignore[arg-type]
            concept_id=None,
        )
        if (
            edge.condition is not None
            and "condition" in stakeholder.edge_properties_for(eid)
        ):
            claims[f"{eid}.condition"] = TruthClaim(
                id=f"{eid}.condition",
                context_id=eid,
                property="condition",  # type: ignore[arg-type]
                concept_id=edge.condition.concept_id,
            )
    return claims
