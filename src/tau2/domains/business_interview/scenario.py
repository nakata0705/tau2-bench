"""Scenarios for the unified-glossary business_interview benchmark (v6).

A scenario is a **Truth process graph** (a ``BusinessProcessGraph`` whose
nodes/edges reference evaluator-only Truth concepts of every ``ConceptKind``:
activity / actor / system / data / condition / rationale), an evaluator-only
``EvaluationSpec`` (kept for API stability; contains no semantic matchers —
everything semantic is expressed as TruthClaims), the stakeholder filter(s),
the **hidden claim catalog** (``claims.py``) and the **private atomic
StakeholderFact catalog** (``facts.py``).

There are **no surface-term tables, no stop phrases, no semantic expression
lists and no duplicated business facts in ``known_info``**: the simulator
answers from structured facts and returns private assertions (fact id + exact
message span); the evaluator grounds everything through that private
provenance, never through wording.

The Truth graph and the agent's inferred graph use the same class; Truth
concept ids are evaluator-only and need not equal Agent concept ids.
"""

from dataclasses import dataclass, replace
from typing import Optional

from tau2.domains.business_interview.claims import TruthClaim, build_claims
from tau2.domains.business_interview.evaluation import EvaluationSpec
from tau2.domains.business_interview.facts import StakeholderFact
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
    return BusinessConcept(id=cid, kind=kind, preferred_label=label)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# quotation_workflow_1
# ---------------------------------------------------------------------------


def quotation_truth() -> BusinessProcessGraph:
    """The quotation workflow Truth graph (cycles allowed; here it is a DAG).

    Reads/writes are ConceptRefs into evaluator-only Truth data concepts;
    ``tc_quote`` flows through cq.writes / ap.reads / sq.reads / me.reads,
    but only ``cq.writes`` is stakeholder-visible (see the sales filter).
    """
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
    )


# Hidden structured StakeholderFacts for the quotation sales stakeholder.
# Facts are ATOMIC: exactly one supported TruthClaim per fact, so a fact can
# never cross-credit independent claims. Fact ids are the SAME across EN and
# JA (semantic fact identities are shared); only the wording differs.
def quotation_facts(locale: str = "en") -> dict[str, StakeholderFact]:
    """The private atomic StakeholderFact catalog for quotation (EN/JA)."""
    if locale == "ja":
        text = {
            "receive": "お客様から見積依頼を受け付けます。",
            "record": "見積依頼を記録します。",
            "check": "CRMで顧客情報を確認します。",
            "create": "顧客情報と価格情報を使って見積書を作成します。",
            "create_system": "見積システムで見積書を作成します。",
            "approve": "100万円を超える見積書は、送付前に管理者の承認が必要です。",
            "approve_rationale": "この承認は与信リスク管理のためのものです。",
            "send": "見積書をメールで顧客に送付します。",
            "month_end": "月末には見積情報の集計を経理チームに送ります。",
            "month_end_excel": "月末の集計はExcelファイルとして送ります。",
            "flow_e1": "依頼を受け付けたら、顧客情報を確認します。",
            "flow_e2": "顧客情報を確認したら、見積書を作成します。",
            "flow_e3": "100万円を超える見積書は管理者の承認に回ります。",
            "flow_e4": "100万円以下の見積書はそのまま顧客に送付します。",
            "flow_e5": "承認された見積書は顧客に送付します。",
            "flow_e6": "月末には経理チームへの集計も行います。",
            "cond_over_1m": "100万円を超える場合です。",
            "cond_at_or_below_1m": "100万円以下の場合です。",
            "cond_month_end": "月末の場合です。",
        }
    else:
        text = {
            "receive": "You receive quotation requests from customers.",
            "record": "You record the quotation request.",
            "check": "You check the customer's information in the CRM.",
            "create": "You create the quotation using the customer and pricing information.",
            "create_system": "You create the quotation in the quoting system.",
            "approve": "Quotations over 1,000,000 yen must be approved by a manager before they are sent.",
            "approve_rationale": "The approval is for credit risk management.",
            "send": "You send the quotation to the customer by email.",
            "month_end": "At month-end you send a summary of the quotation information to the Accounting team.",
            "month_end_excel": "At month-end the summary is sent as an Excel file.",
            "flow_e1": "After receiving the request, you check the customer information.",
            "flow_e2": "After checking the customer information, you create the quotation.",
            "flow_e3": "Quotations over 1,000,000 yen go to the manager for approval.",
            "flow_e4": "Quotations at or below 1,000,000 yen are sent directly to the customer.",
            "flow_e5": "Once approved, the quotation is sent to the customer.",
            "flow_e6": "At month-end you also send the summary to Accounting.",
            "cond_over_1m": "over 1,000,000 yen",
            "cond_at_or_below_1m": "at or below 1,000,000 yen",
            "cond_month_end": "month-end",
        }
    # (fact id, text key, supported claim id)
    spec: list[tuple[str, str, str]] = [
        ("quotation.receive.activity", "receive", "r.activity"),
        ("quotation.receive.actor", "receive", "r.actor"),
        ("quotation.receive.writes", "record", "r.writes.tc_request"),
        ("quotation.check.activity", "check", "cc.activity"),
        ("quotation.check.actor", "check", "cc.actor"),
        ("quotation.check.system", "check", "cc.system"),
        ("quotation.check.reads", "check", "cc.reads.tc_customer"),
        ("quotation.create.activity", "create", "cq.activity"),
        ("quotation.create.actor", "create", "cq.actor"),
        ("quotation.create.system", "create_system", "cq.system"),
        ("quotation.create.reads.customer", "create", "cq.reads.tc_customer"),
        ("quotation.create.reads.pricing", "create", "cq.reads.tc_pricing"),
        ("quotation.create.writes", "create", "cq.writes.tc_quote"),
        ("quotation.approve.activity", "approve", "ap.activity"),
        ("quotation.approve.actor", "approve", "ap.actor"),
        ("quotation.approve.rationale", "approve_rationale", "ap.rationale"),
        ("quotation.send.activity", "send", "sq.activity"),
        ("quotation.send.actor", "send", "sq.actor"),
        ("quotation.send.system", "send", "sq.system"),
        ("quotation.month_end.activity", "month_end", "me.activity"),
        ("quotation.month_end.actor", "month_end", "me.actor"),
        ("quotation.month_end.system", "month_end_excel", "me.system"),
        ("quotation.month_end.writes", "month_end", "me.writes.tc_excel_summary"),
        ("flow.e1.edge_exists", "flow_e1", "e1.edge_exists"),
        ("flow.e2.edge_exists", "flow_e2", "e2.edge_exists"),
        ("flow.e3.edge_exists", "flow_e3", "e3.edge_exists"),
        ("flow.e4.edge_exists", "flow_e4", "e4.edge_exists"),
        ("flow.e5.edge_exists", "flow_e5", "e5.edge_exists"),
        ("flow.e6.edge_exists", "flow_e6", "e6.edge_exists"),
        ("condition.e3.over_1m", "cond_over_1m", "e3.condition"),
        ("condition.e4.at_or_below_1m", "cond_at_or_below_1m", "e4.condition"),
        ("condition.e6.month_end", "cond_month_end", "e6.condition"),
    ]
    return {
        fid: StakeholderFact(
            id=fid,
            text=text[key],
            supported_claim_ids=[claim_id],
        )
        for fid, key, claim_id in spec
    }


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
    properties but not the approval-branch credit-risk rationale. Used to show
    multiple filters."""
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
    expressed entirely through hidden TruthClaims + private StakeholderFacts.
    This object is kept for API stability and is empty.
    """
    return EvaluationSpec()


# ---------------------------------------------------------------------------
# lab_sample_flow
# ---------------------------------------------------------------------------


def lab_sample_truth() -> BusinessProcessGraph:
    """A non-quotation Truth graph (lab sample conditioning).

    Demonstrates open-world domain concepts ("specimen accession",
    "chamber seasoning", "conditioning cycle") that are not part of any global
    ontology. Derived read/write artifacts are hidden from the stakeholder.
    """
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
    )


def lab_sample_facts() -> dict[str, StakeholderFact]:
    """The private atomic StakeholderFact catalog for the lab technician.

    Only n1 reads (sample) plus the visible activity/actor/system properties
    are stakeholder-visible; hidden derived read/write artifacts are
    deliberately NOT supported by any fact (hidden properties stay unset —
    epistemic restraint).
    """
    spec: list[tuple[str, str, str]] = [
        (
            "lab.accession.activity",
            "When a specimen arrives, you accession it — you record it as received.",
            "n1.activity",
        ),
        (
            "lab.accession.actor",
            "I am the lab technician who accessions the specimens.",
            "n1.actor",
        ),
        (
            "lab.accession.reads",
            "When a specimen arrives, you accession it — you record it as received.",
            "n1.reads.tc_sample",
        ),
        (
            "lab.seasoning.activity",
            "You season the environment chamber to prepare it.",
            "n2.activity",
        ),
        ("lab.seasoning.actor", "I season the chamber myself.", "n2.actor"),
        (
            "lab.seasoning.system",
            "The seasoning is done in the environment chamber.",
            "n2.system",
        ),
        (
            "lab.conditioning.activity",
            "You run a conditioning cycle that processes the samples inside the chamber.",
            "n3.activity",
        ),
        ("lab.conditioning.actor", "I run the conditioning cycle.", "n3.actor"),
        (
            "lab.conditioning.system",
            "The conditioning cycle runs in the environment chamber.",
            "n3.system",
        ),
        (
            "lab.approval.activity",
            "Finally, the lab supervisor approves the conditioned batch before it is released.",
            "n4.activity",
        ),
        ("lab.approval.actor", "The lab supervisor approves the batch.", "n4.actor"),
        (
            "lab.flow.l1",
            "After specimen accession, you prepare the chamber for seasoning.",
            "l1.edge_exists",
        ),
        (
            "lab.flow.l2",
            "After chamber seasoning, you run the conditioning cycle.",
            "l2.edge_exists",
        ),
        (
            "lab.flow.l3",
            "After the conditioning cycle, the lab supervisor approves the batch.",
            "l3.edge_exists",
        ),
    ]
    return {
        fid: StakeholderFact(id=fid, text=text, supported_claim_ids=[claim_id])
        for fid, text, claim_id in spec
    }


def lab_sample_claims() -> dict[str, TruthClaim]:
    return build_claims(lab_sample_truth(), lab_sample_filter())


def lab_sample_spec() -> EvaluationSpec:
    """Evaluator-only spec for the lab scenario (no semantic matchers)."""
    return EvaluationSpec()


def lab_sample_filter() -> StakeholderFilter:
    """The lab technician. Can state the activities, actors, the
    environment-chamber system, and the raw specimen/sample input, but the GT
    read/write artifacts (``accessioned sample``, ``seasoned chamber``,
    ``conditioned sample``, ``batch approval``) are benchmark-derived
    state/artifact conventions the stakeholder never names. They are therefore
    hidden: the correct agent behavior is to leave them unset (epistemic
    restraint), not to invent them.
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
    facts: dict[str, StakeholderFact]


_SCENARIOS: dict[str, Scenario] = {
    "quotation_workflow_1": Scenario(
        scenario_id="quotation_workflow_1",
        truth=quotation_truth(),
        spec=quotation_spec(),
        stakeholder=quotation_sales_filter(),
        claims=quotation_claims(),
        facts=quotation_facts("en"),
    ),
    "lab_sample_flow": Scenario(
        scenario_id="lab_sample_flow",
        truth=lab_sample_truth(),
        spec=lab_sample_spec(),
        stakeholder=lab_sample_filter(),
        claims=lab_sample_claims(),
        facts=lab_sample_facts(),
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
        # JA locale: same truth/spec/filter/claims, JA fact wording.
        # Fact ids and claim ids are shared across locales.
        return replace(sc, facts=quotation_facts("ja"))
    return sc
