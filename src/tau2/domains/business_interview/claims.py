"""Hidden Truth claims (evaluator/simulator-private) — general properties.

A ``TruthClaim`` is one hidden scored semantic property of the Truth process
graph: a subject (node or edge) + property + optional Truth concept. Examples:

    cq.activity      -> tc_activity_create_quotation
    cq.actor         -> tc_actor_sales
    cq.reads         -> tc_customer
    e3.edge_exists   (concept_id None — structural fact)
    e3.condition     -> tc_cond_over_1m

Claims exist **only** for stakeholder-visible subjects/properties; hidden
subjects/properties produce no claims and can never enter any catalog.

Claims are referenced by private ``StakeholderFact``\\ s (``facts.py``): a fact
declares which Truth claims it supports, and the stakeholder simulator returns
private assertions (fact id + exact message span) for the facts it actually
used. Nothing here is derived from natural-language text.

The Agent never sees claim ids, the claim catalog, or the fact/assertion
ledger: they live only in evaluator/simulator-private state.
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
    """One hidden Truth claim: a scored semantic property of a Truth subject.

    ``subject_kind`` is ``node`` or ``edge``; ``subject_id`` is the Truth
    node/edge id; ``property`` is the scored property; ``concept_id`` is the
    Truth BusinessConcept the property must bind to (``None`` for
    ``edge_exists``, which is purely structural).
    """

    id: str
    subject_kind: Literal["node", "edge"]
    subject_id: str
    property: ClaimProperty
    concept_id: Optional[str] = None


def build_claims(
    truth: BusinessProcessGraph,
    stakeholder: StakeholderFilter,
) -> dict[str, TruthClaim]:
    """Build the private claim catalog from the Truth graph + visibility.

    One claim per visible node property (activity/actor/system/reads/writes/
    rationale) and per visible edge property (edge_exists always; condition
    only when visible). Hidden subjects/properties produce no claims.
    """
    claims: dict[str, TruthClaim] = {}
    # node property claims
    for nid, node in truth.nodes.items():
        props = stakeholder.node_properties_for(nid)
        if "activity" in props and node.activity is not None:
            claims[f"{nid}.activity"] = TruthClaim(
                id=f"{nid}.activity",
                subject_kind="node",
                subject_id=nid,
                property="activity",  # type: ignore[arg-type]
                concept_id=node.activity.concept_id,
            )
        for prop in ("actor", "system", "necessity_rationale"):
            ref = getattr(node, prop)
            if ref is None:
                continue
            pname = "rationale" if prop == "necessity_rationale" else prop
            if pname in props:
                claims[f"{nid}.{pname}"] = TruthClaim(
                    id=f"{nid}.{pname}",
                    subject_kind="node",
                    subject_id=nid,
                    property=pname,  # type: ignore[arg-type]
                    concept_id=ref.concept_id,
                )
        for axis in ("reads", "writes"):
            if axis not in props:
                continue
            for ref in getattr(node, axis):
                cid = f"{nid}.{axis}.{ref.concept_id}"
                claims[cid] = TruthClaim(
                    id=cid,
                    subject_kind="node",
                    subject_id=nid,
                    property=axis,  # type: ignore[arg-type]
                    concept_id=ref.concept_id,
                )
    for eid, edge in truth.edges.items():
        if eid not in stakeholder.visible_edge_ids:
            continue
        claims[f"{eid}.edge_exists"] = TruthClaim(
            id=f"{eid}.edge_exists",
            subject_kind="edge",
            subject_id=eid,
            property="edge_exists",  # type: ignore[arg-type]
            concept_id=None,
        )
        if (
            edge.condition is not None
            and "condition" in stakeholder.edge_properties_for(eid)
        ):
            claims[f"{eid}.condition"] = TruthClaim(
                id=f"{eid}.condition",
                subject_kind="edge",
                subject_id=eid,
                property="condition",  # type: ignore[arg-type]
                concept_id=edge.condition.concept_id,
            )
    return claims
