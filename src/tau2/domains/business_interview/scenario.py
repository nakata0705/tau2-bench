"""Scenarios for the graph-context business_interview benchmark (v7).

A scenario is a **Truth process graph** (a ``BusinessProcessGraph`` whose
nodes/edges reference evaluator-only Truth concepts of every ``ConceptKind``,
with explicit start/end), an evaluator-only ``EvaluationSpec`` (empty — no
semantic matchers), the stakeholder filter(s), the **hidden claim catalog**
(``claims.py``: claims keyed by graph position) and the **StakeholderKnowledge**
(``facts.py``): the stakeholder's projection of the Truth — visible claim ids,
graph contexts (incoming edges / start) and concept views (lexical
preferences). There are **no authored business sentences anywhere**: facts are
semantic structure; the stakeholder LLM realizes them into natural language.

The Truth graph and the agent's inferred graph use the same class; Truth
concept ids are evaluator-only and need not equal Agent concept ids.
"""

from dataclasses import dataclass, replace
from typing import Optional

from tau2.domains.business_interview.claims import TruthClaim, build_claims
from tau2.domains.business_interview.evaluation import EvaluationSpec
from tau2.domains.business_interview.facts import (
    StakeholderKnowledge,
    project_knowledge,
)
from tau2.domains.business_interview.graph import (
    BusinessConcept,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    Node,
)
from tau2.domains.business_interview.stakeholder import StakeholderFilter

JA_SCENARIO_SUFFIX = "_ja"


def _ref(concept_id: str) -> ConceptRef:
    return ConceptRef(concept_id=concept_id, confidence=1.0)


def _concept(cid: str, kind: str, label: str) -> BusinessConcept:
    return BusinessConcept(id=cid, kind=kind, display_label=label)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# quotation_workflow_1
# ---------------------------------------------------------------------------


def quotation_truth() -> BusinessProcessGraph:
    """The quotation workflow Truth graph (1 start / 2 ends; a directed
    process graph — cycles remain valid in general)."""
    return BusinessProcessGraph(
        id="quotation",
        name="Quotation creation",
        concepts={
            # activities
            "tc_activity_receive_request": _concept(
                "tc_activity_receive_request", "activity", "receive quotation request"
            ),
            "tc_activity_check_customer": _concept(
                "tc_activity_check_customer", "activity", "check customer information"
            ),
            "tc_activity_create_quotation": _concept(
                "tc_activity_create_quotation", "activity", "create quotation"
            ),
            "tc_activity_approve_quotation": _concept(
                "tc_activity_approve_quotation",
                "activity",
                "approve high-value quotation",
            ),
            "tc_activity_send_quotation": _concept(
                "tc_activity_send_quotation", "activity", "send quotation to customer"
            ),
            "tc_activity_send_month_end_summary": _concept(
                "tc_activity_send_month_end_summary",
                "activity",
                "send month-end summary to accounting",
            ),
            # actors
            "tc_actor_sales": _concept("tc_actor_sales", "actor", "sales"),
            "tc_actor_manager": _concept("tc_actor_manager", "actor", "manager"),
            # systems
            "tc_system_crm": _concept("tc_system_crm", "system", "crm"),
            "tc_system_quoting": _concept(
                "tc_system_quoting", "system", "quoting system"
            ),
            "tc_system_email": _concept("tc_system_email", "system", "email"),
            "tc_system_excel": _concept("tc_system_excel", "system", "excel"),
            # data
            "tc_request": _concept("tc_request", "data", "request"),
            "tc_customer": _concept("tc_customer", "data", "customer"),
            "tc_pricing": _concept("tc_pricing", "data", "pricing"),
            "tc_quote": _concept("tc_quote", "data", "quote"),
            "tc_approval": _concept("tc_approval", "data", "approval"),
            "tc_sent_quote": _concept("tc_sent_quote", "data", "sent_quote"),
            "tc_excel_summary": _concept("tc_excel_summary", "data", "excel_summary"),
            # conditions
            "tc_cond_over_1m": _concept(
                "tc_cond_over_1m", "condition", "amount over 1,000,000"
            ),
            "tc_cond_at_or_below_1m": _concept(
                "tc_cond_at_or_below_1m", "condition", "amount at or below 1,000,000"
            ),
            "tc_cond_month_end": _concept(
                "tc_cond_month_end", "condition", "month-end"
            ),
            # rationale
            "tc_rationale_credit_risk": _concept(
                "tc_rationale_credit_risk", "rationale", "credit risk management"
            ),
        },
        nodes={
            "r": Node(
                id="r",
                activity=_ref("tc_activity_receive_request"),
                actor=_ref("tc_actor_sales"),
                writes=[_ref("tc_request")],
            ),
            "cc": Node(
                id="cc",
                activity=_ref("tc_activity_check_customer"),
                actor=_ref("tc_actor_sales"),
                system=_ref("tc_system_crm"),
                reads=[_ref("tc_customer")],
            ),
            "cq": Node(
                id="cq",
                activity=_ref("tc_activity_create_quotation"),
                actor=_ref("tc_actor_sales"),
                system=_ref("tc_system_quoting"),
                reads=[_ref("tc_customer"), _ref("tc_pricing")],
                writes=[_ref("tc_quote")],
            ),
            "ap": Node(
                id="ap",
                activity=_ref("tc_activity_approve_quotation"),
                actor=_ref("tc_actor_manager"),
                necessity_rationale=_ref("tc_rationale_credit_risk"),
            ),
            "sq": Node(
                id="sq",
                activity=_ref("tc_activity_send_quotation"),
                actor=_ref("tc_actor_sales"),
                system=_ref("tc_system_email"),
            ),
            "me": Node(
                id="me",
                activity=_ref("tc_activity_send_month_end_summary"),
                actor=_ref("tc_actor_sales"),
                system=_ref("tc_system_excel"),
                writes=[_ref("tc_excel_summary")],
            ),
        },
        edges={
            "e1": Edge(id="e1", from_node="r", to_node="cc"),
            "e2": Edge(id="e2", from_node="cc", to_node="cq"),
            "e3": Edge(
                id="e3",
                from_node="cq",
                to_node="ap",
                condition=_ref("tc_cond_over_1m"),
            ),
            "e4": Edge(
                id="e4",
                from_node="cq",
                to_node="sq",
                condition=_ref("tc_cond_at_or_below_1m"),
            ),
            "e5": Edge(id="e5", from_node="ap", to_node="sq"),
            "e6": Edge(
                id="e6",
                from_node="cq",
                to_node="me",
                condition=_ref("tc_cond_month_end"),
            ),
        },
        start_node_id="r",
        end_node_ids=["sq", "me"],
    )


# The stakeholder's local lexical preferences (single wordings, never
# workflow sentences). EN and JA share concept ids; wording differs.
def quotation_concept_views(locale: str = "en") -> dict[str, str]:
    if locale == "ja":
        return {
            "tc_activity_receive_request": "見積依頼を受け付ける",
            "tc_activity_check_customer": "顧客情報を確認する",
            "tc_activity_create_quotation": "見積書を作成する",
            "tc_activity_approve_quotation": "高額見積書を承認する",
            "tc_activity_send_quotation": "見積書を顧客に送付する",
            "tc_activity_send_month_end_summary": "月末の集計を経理に送る",
            "tc_actor_sales": "営業担当者",
            "tc_actor_manager": "管理者",
            "tc_system_crm": "CRM",
            "tc_system_quoting": "見積システム",
            "tc_system_email": "メール",
            "tc_system_excel": "Excel",
            "tc_request": "見積依頼",
            "tc_customer": "顧客情報",
            "tc_pricing": "価格情報",
            "tc_quote": "見積書",
            "tc_excel_summary": "見積情報の集計",
            "tc_cond_over_1m": "100万円超",
            "tc_cond_at_or_below_1m": "100万円以下",
            "tc_cond_month_end": "月末",
            "tc_rationale_credit_risk": "与信リスク管理",
        }
    return {
        "tc_activity_receive_request": "receive the quotation request",
        "tc_activity_check_customer": "check the customer information",
        "tc_activity_create_quotation": "create the quotation",
        "tc_activity_approve_quotation": "approve the high-value quotation",
        "tc_activity_send_quotation": "send the quotation to the customer",
        "tc_activity_send_month_end_summary": "send the month-end summary to Accounting",
        "tc_actor_sales": "sales employee",
        "tc_actor_manager": "manager",
        "tc_system_crm": "CRM",
        "tc_system_quoting": "quoting system",
        "tc_system_email": "email",
        "tc_system_excel": "Excel",
        "tc_request": "quotation request",
        "tc_customer": "customer information",
        "tc_pricing": "pricing information",
        "tc_quote": "quotation",
        "tc_excel_summary": "summary of the quotation information",
        "tc_cond_over_1m": "over 1,000,000 yen",
        "tc_cond_at_or_below_1m": "at or below 1,000,000 yen",
        "tc_cond_month_end": "month-end",
        "tc_rationale_credit_risk": "credit risk management",
    }


def quotation_knowledge(locale: str = "en") -> StakeholderKnowledge:
    """The sales stakeholder's semantic knowledge: a **physical projection**
    of the Truth — only the visible claims, their visible contexts (incoming
    relations + start), and views for the concepts those claims reference.
    Hidden concepts/relations never enter the knowledge."""
    truth = quotation_truth()
    filter_ = quotation_sales_filter()
    claims = build_claims(truth, filter_)
    return project_knowledge(truth, filter_, claims, quotation_concept_views(locale))


def quotation_claims() -> dict[str, TruthClaim]:
    """The hidden TruthClaim catalog for the quotation scenario."""
    return build_claims(quotation_truth(), quotation_sales_filter())


def quotation_sales_filter() -> StakeholderFilter:
    """The sales employee. Knows the whole workflow and the approval rationale,
    but cannot explain the month-end step's rationale, and does not know where
    the approval happens, nor the approval/send/month-end read+write data
    artifacts.

    Per-node property visibility (activity/actor/system/reads/writes/rationale):
    - r: activity, actor, writes
    - cc: activity, actor, system, reads
    - cq: activity, actor, system, reads, writes
    - ap: activity, actor, rationale
    - sq: activity, actor, system
    - me: activity, actor, system, writes
    Per-edge condition visibility: e3/e4/e6 conditions are known; e1/e2/e5 are
    unconditional.
    """
    return StakeholderFilter(
        name="sales",
        visible_node_ids=["r", "cc", "cq", "ap", "sq", "me"],
        visible_edge_ids=["e1", "e2", "e3", "e4", "e5", "e6"],
        visible_node_attributes={
            "r": ["activity", "actor", "writes"],
            "cc": ["activity", "actor", "system", "reads"],
            "cq": ["activity", "actor", "system", "reads", "writes"],
            "ap": ["activity", "actor", "rationale"],
            "sq": ["activity", "actor", "system"],
            "me": ["activity", "actor", "system", "writes"],
        },
        visible_edge_attributes={
            "e3": ["condition"],
            "e4": ["condition"],
            "e6": ["condition"],
        },
    )


def quotation_finance_filter() -> StakeholderFilter:
    """A finance/audit stakeholder: sees the month-end tail with all its
    properties but not the approval-branch credit-risk rationale."""
    return StakeholderFilter(
        name="finance",
        visible_node_ids=["cq", "sq", "me"],
        visible_edge_ids=["e4", "e6"],
        visible_attributes=["activity", "actor", "system", "reads", "writes"],
        visible_node_attributes={
            "cq": ["activity", "actor", "system", "reads", "writes"],
            "sq": ["activity", "actor", "system", "reads", "writes"],
            "me": ["activity", "actor", "system", "reads", "writes", "rationale"],
        },
        visible_edge_attributes={"e6": ["condition"]},
    )


def quotation_spec() -> EvaluationSpec:
    """Evaluator-only spec for the quotation scenario.

    The spec carries NO semantic matchers: node/edge/property correctness is
    expressed entirely through hidden TruthClaims + private assertions.
    """
    return EvaluationSpec()


# ---------------------------------------------------------------------------
# lab_sample_flow
# ---------------------------------------------------------------------------


def lab_sample_truth() -> BusinessProcessGraph:
    """A non-quotation Truth graph (lab sample conditioning)."""
    return BusinessProcessGraph(
        id="lab",
        name="Lab sample conditioning",
        concepts={
            "tc_activity_accession": _concept(
                "tc_activity_accession", "activity", "specimen accession"
            ),
            "tc_activity_seasoning": _concept(
                "tc_activity_seasoning", "activity", "chamber seasoning"
            ),
            "tc_activity_conditioning": _concept(
                "tc_activity_conditioning", "activity", "conditioning cycle"
            ),
            "tc_activity_batch_approval": _concept(
                "tc_activity_batch_approval", "activity", "approve conditioned batch"
            ),
            "tc_actor_lab_tech": _concept("tc_actor_lab_tech", "actor", "lab tech"),
            "tc_actor_lab_supervisor": _concept(
                "tc_actor_lab_supervisor", "actor", "lab supervisor"
            ),
            "tc_system_chamber": _concept(
                "tc_system_chamber", "system", "environment chamber"
            ),
            "tc_sample": _concept("tc_sample", "data", "sample"),
            "tc_accessioned_sample": _concept(
                "tc_accessioned_sample", "data", "accessioned sample"
            ),
            "tc_seasoned_chamber": _concept(
                "tc_seasoned_chamber", "data", "seasoned chamber"
            ),
            "tc_conditioned_sample": _concept(
                "tc_conditioned_sample", "data", "conditioned sample"
            ),
            "tc_batch_approval": _concept(
                "tc_batch_approval", "data", "batch approval"
            ),
        },
        nodes={
            "n1": Node(
                id="n1",
                activity=_ref("tc_activity_accession"),
                actor=_ref("tc_actor_lab_tech"),
                reads=[_ref("tc_sample")],
                writes=[_ref("tc_accessioned_sample")],
            ),
            "n2": Node(
                id="n2",
                activity=_ref("tc_activity_seasoning"),
                actor=_ref("tc_actor_lab_tech"),
                system=_ref("tc_system_chamber"),
                writes=[_ref("tc_seasoned_chamber")],
            ),
            "n3": Node(
                id="n3",
                activity=_ref("tc_activity_conditioning"),
                actor=_ref("tc_actor_lab_tech"),
                system=_ref("tc_system_chamber"),
                reads=[_ref("tc_accessioned_sample")],
                writes=[_ref("tc_conditioned_sample")],
            ),
            "n4": Node(
                id="n4",
                activity=_ref("tc_activity_batch_approval"),
                actor=_ref("tc_actor_lab_supervisor"),
                reads=[_ref("tc_conditioned_sample")],
                writes=[_ref("tc_batch_approval")],
            ),
        },
        edges={
            "l1": Edge(id="l1", from_node="n1", to_node="n2"),
            "l2": Edge(id="l2", from_node="n2", to_node="n3"),
            "l3": Edge(id="l3", from_node="n3", to_node="n4"),
        },
        start_node_id="n1",
        end_node_ids=["n4"],
    )


def lab_sample_views() -> dict[str, str]:
    return {
        "tc_activity_accession": "accession the specimen",
        "tc_activity_seasoning": "season the chamber",
        "tc_activity_conditioning": "run the conditioning cycle",
        "tc_activity_batch_approval": "approve the conditioned batch",
        "tc_actor_lab_tech": "lab technician",
        "tc_actor_lab_supervisor": "lab supervisor",
        "tc_system_chamber": "environment chamber",
        "tc_sample": "sample",
        "tc_accessioned_sample": "accessioned sample",
        "tc_seasoned_chamber": "seasoned chamber",
        "tc_conditioned_sample": "conditioned sample",
        "tc_batch_approval": "batch approval",
    }


def lab_sample_knowledge() -> StakeholderKnowledge:
    """The lab tech's semantic knowledge: a physical projection of the Truth
    (visible claims + visible contexts + views for visible claim concepts
    only)."""
    truth = lab_sample_truth()
    filter_ = lab_sample_filter()
    claims = build_claims(truth, filter_)
    return project_knowledge(truth, filter_, claims, lab_sample_views())


def lab_sample_claims() -> dict[str, TruthClaim]:
    return build_claims(lab_sample_truth(), lab_sample_filter())


def lab_sample_spec() -> EvaluationSpec:
    return EvaluationSpec()


def lab_sample_filter() -> StakeholderFilter:
    """The lab technician. Can state the activities, actors, the
    environment-chamber system, and the raw specimen/sample input, but the GT
    read/write artifacts are hidden: the correct agent behavior is to leave
    them unset (epistemic restraint), not to invent them.
    """
    return StakeholderFilter(
        name="lab tech",
        visible_node_ids=["n1", "n2", "n3", "n4"],
        visible_edge_ids=["l1", "l2", "l3"],
        visible_node_attributes={
            "n1": ["activity", "actor", "reads"],
            "n2": ["activity", "actor", "system"],
            "n3": ["activity", "actor", "system"],
            "n4": ["activity", "actor"],
        },
        visible_edge_attributes={},
    )


@dataclass
class Scenario:
    scenario_id: str
    truth: BusinessProcessGraph
    spec: EvaluationSpec
    stakeholder: StakeholderFilter
    claims: dict[str, TruthClaim]
    knowledge: StakeholderKnowledge


_SCENARIOS: dict[str, Scenario] = {
    "quotation_workflow_1": Scenario(
        scenario_id="quotation_workflow_1",
        truth=quotation_truth(),
        spec=quotation_spec(),
        stakeholder=quotation_sales_filter(),
        claims=quotation_claims(),
        knowledge=quotation_knowledge("en"),
    ),
    "lab_sample_flow": Scenario(
        scenario_id="lab_sample_flow",
        truth=lab_sample_truth(),
        spec=lab_sample_spec(),
        stakeholder=lab_sample_filter(),
        claims=lab_sample_claims(),
        knowledge=lab_sample_knowledge(),
    ),
}


def canonical_scenario_id(scenario_id: Optional[str]) -> Optional[str]:
    if scenario_id is None:
        return None
    if scenario_id.endswith(JA_SCENARIO_SUFFIX):
        return scenario_id[: -len(JA_SCENARIO_SUFFIX)]
    return scenario_id


def get_scenario(scenario_id: Optional[str]) -> Optional[Scenario]:
    canonical = canonical_scenario_id(scenario_id)
    if canonical is None:
        return None
    sc = _SCENARIOS.get(canonical)
    if sc is None:
        return None
    if scenario_id is not None and scenario_id.endswith(JA_SCENARIO_SUFFIX):
        # JA locale: same truth/spec/filter/claims; JA concept views.
        return replace(sc, knowledge=quotation_knowledge("ja"))
    return sc
