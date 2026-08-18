"""Scenarios for the agent-local-concept business_interview benchmark (v4).

A scenario is a **Truth DAG** (a plain ``BusinessDAG`` whose reads/writes are
``ConceptRef``\\ s into evaluator-only Truth data concepts), an evaluator-only
``EvaluationSpec`` (concepts for node/edge/necessity matching), the
stakeholder filter(s), and the **hidden claim catalog** (``claims.py``): the
private stakeholder fact catalog binding visible Truth read/write concepts to
the natural surface terms the stakeholder uses.

The Truth DAG and the agent's inferred DAG use the same class; Truth concept
ids are evaluator-only and need not equal Agent concept ids.
"""

from dataclasses import dataclass, replace
from typing import Optional

from tau2.domains.business_interview.claims import Claim, build_claims
from tau2.domains.business_interview.dag import (
    BusinessDAG,
    ConceptRef,
    DataConcept,
    Edge,
    InferredValue,
    Necessity,
    Node,
)
from tau2.domains.business_interview.evaluation import EvaluationSpec, TruthNodeSpec
from tau2.domains.business_interview.stakeholder import StakeholderFilter

JA_SCENARIO_SUFFIX = "_ja"


def _iv(value: Optional[str]) -> InferredValue:
    return InferredValue(value=value)


def _ref(concept_id: str) -> ConceptRef:
    return ConceptRef(concept_id=concept_id, confidence=1.0)


def _concept(cid: str, label: str) -> DataConcept:
    return DataConcept(id=cid, preferred_label=label)


# ---------------------------------------------------------------------------
# quotation_workflow_1
# ---------------------------------------------------------------------------


def quotation_truth() -> BusinessDAG:
    """The quotation workflow Truth DAG (1 start / 2 ends).

    Reads/writes are ConceptRefs into evaluator-only Truth data concepts;
    ``tc_quote`` flows through cq.writes / ap.reads / sq.reads / me.reads,
    but only ``cq.writes`` is stakeholder-visible (see the sales filter).
    """
    return BusinessDAG(
        id="quotation",
        name="Quotation creation",
        data_concepts={
            "tc_request": _concept("tc_request", "request"),
            "tc_customer": _concept("tc_customer", "customer"),
            "tc_pricing": _concept("tc_pricing", "pricing"),
            "tc_quote": _concept("tc_quote", "quote"),
            "tc_approval": _concept("tc_approval", "approval"),
            "tc_sent_quote": _concept("tc_sent_quote", "sent_quote"),
            "tc_excel_summary": _concept("tc_excel_summary", "excel_summary"),
        },
        nodes={
            "r": Node(
                id="r",
                action=_iv("receive quotation request"),
                actor=_iv("sales"),
                writes=[_ref("tc_request")],
            ),
            "cc": Node(
                id="cc",
                action=_iv("check customer information in the CRM"),
                actor=_iv("sales"),
                system=_iv("crm"),
                reads=[_ref("tc_customer")],
            ),
            "cq": Node(
                id="cq",
                action=_iv("create quotation in the quoting system"),
                actor=_iv("sales"),
                system=_iv("quoting"),
                reads=[_ref("tc_customer"), _ref("tc_pricing")],
                writes=[_ref("tc_quote")],
            ),
            "ap": Node(
                id="ap",
                action=_iv("approve high-value quotation"),
                actor=_iv("manager"),
                system=_iv("quoting"),
                reads=[_ref("tc_quote")],
                writes=[_ref("tc_approval")],
                necessity=Necessity(rationale=_iv("for credit risk management")),
            ),
            "sq": Node(
                id="sq",
                action=_iv("send quotation to customer"),
                actor=_iv("sales"),
                system=_iv("email"),
                reads=[_ref("tc_quote")],
                writes=[_ref("tc_sent_quote")],
            ),
            "me": Node(
                id="me",
                action=_iv("send quotation summary to accounting at month-end"),
                actor=_iv("sales"),
                system=_iv("excel"),
                reads=[_ref("tc_quote")],
                writes=[_ref("tc_excel_summary")],
                # rationale / owner / evidence / removal all unknown (no value)
                necessity=Necessity(),
            ),
        },
        edges={
            "e1": Edge(id="e1", from_node="r", to_node="cc"),
            "e2": Edge(id="e2", from_node="cc", to_node="cq"),
            "e3": Edge(
                id="e3",
                from_node="cq",
                to_node="ap",
                predicate=_iv("amount over 1,000,000"),
            ),
            "e4": Edge(
                id="e4",
                from_node="cq",
                to_node="sq",
                predicate=_iv("amount at or below 1,000,000"),
            ),
            "e5": Edge(id="e5", from_node="ap", to_node="sq"),
            "e6": Edge(
                id="e6",
                from_node="cq",
                to_node="me",
                predicate=_iv("month-end"),
            ),
        },
        start_node_id="r",
        end_node_ids=["sq", "me"],
    )


# Natural surface terms per visible Truth (node, axis, concept) — the
# stakeholder's private known-info wording (EN). Keys must match the claims
# builder's expectations; only visible axes appear.
QUOTATION_SURFACE_TERMS_EN: dict[tuple[str, str, str], list[str]] = {
    ("r", "writes", "tc_request"): ["request", "quotation request"],
    ("cc", "reads", "tc_customer"): [
        "customer",
        "customer information",
        "customer's information",
    ],
    ("cq", "reads", "tc_customer"): [
        "customer",
        "customer information",
        "customer's information",
    ],
    ("cq", "reads", "tc_pricing"): ["pricing", "pricing information"],
    ("cq", "writes", "tc_quote"): ["quotation", "quotations"],
    ("me", "writes", "tc_excel_summary"): [
        "summary of quotation information",
        "summary of the quotation information",
    ],
}

QUOTATION_SURFACE_TERMS_JA: dict[tuple[str, str, str], list[str]] = {
    ("r", "writes", "tc_request"): ["見積依頼", "依頼"],
    ("cc", "reads", "tc_customer"): ["顧客情報", "顧客"],
    ("cq", "reads", "tc_customer"): ["顧客情報", "顧客"],
    ("cq", "reads", "tc_pricing"): ["価格情報", "価格"],
    ("cq", "writes", "tc_quote"): ["見積書", "見積"],
    ("me", "writes", "tc_excel_summary"): ["見積情報の集計", "月次の集計"],
}

# Phrases that consume a span without supporting any claim (the simulator
# knows "quotation information" is not the quote object).
QUOTATION_STOP_PHRASES: tuple[str, ...] = (
    "quotation information",
    "quotation summary",
    "見積情報",
)


def quotation_spec() -> EvaluationSpec:
    """Hidden, evaluator-only, scenario-local semantic spec.

    The expressions are aliases/paraphrases the evaluator uses to match the
    agent's free-text **actions** to the Truth nodes; ``primitive`` is the
    expected generic operation. There is NO read/write equivalence here —
    reads/writes bind through hidden claim provenance (``claims.py``).
    This metadata is never shown to the agent.
    """
    return EvaluationSpec(
        truth_nodes={
            "r": TruthNodeSpec(
                expressions=[
                    "receive quotation request",
                    "receive the request",
                    "record the quotation request",
                    "intake request",
                    "見積依頼を受け付ける",
                    "見積依頼の受付",
                    "依頼を受ける",
                ],
                primitive="receive",
            ),
            "cc": TruthNodeSpec(
                expressions=[
                    "check customer information in the CRM",
                    "check the customer",
                    "verify customer information",
                    "customer check in the CRM",
                    "CRMで顧客情報を確認する",
                    "顧客情報を確認",
                    "顧客を確認",
                ],
                primitive="check",
            ),
            "cq": TruthNodeSpec(
                expressions=[
                    "create quotation in the quoting system",
                    "create the quotation",
                    "prepare quotation",
                    "generate a quote",
                    "見積システムで見積を作成する",
                    "見積を作成する",
                    "見積書を作成",
                ],
                primitive="create",
            ),
            "ap": TruthNodeSpec(
                expressions=[
                    "approve high-value quotation",
                    "manager approval",
                    "approve the high-value quote",
                    "high-value quotation approval",
                    "高額見積を承認する",
                    "見積を承認する",
                    "高額承認",
                ],
                primitive="approve",
            ),
            "sq": TruthNodeSpec(
                expressions=[
                    "send quotation to customer",
                    "send the quotation",
                    "deliver the quote",
                    "send quote by email",
                    "見積を顧客に送付する",
                    "見積を送付",
                    "見積書を送る",
                ],
                primitive="send",
            ),
            "me": TruthNodeSpec(
                expressions=[
                    "send quotation summary to accounting at month-end",
                    "month-end accounting summary",
                    "send summary to accounting",
                    "月末に経理へ見積集計を送る",
                    "月末の見積集計",
                    "経理へ集計を送る",
                ],
                primitive="send",
            ),
        },
        predicate_expressions={
            "amount over 1,000,000": [
                "amount over 1,000,000",
                "over 1,000,000",
                "100万円超",
                "100万円を超える",
            ],
            "amount at or below 1,000,000": [
                "amount at or below 1,000,000",
                "at or below 1,000,000",
                "100万円以下",
                "100万円未満",
            ],
            "month-end": ["month-end", "monthly", "月末"],
        },
        necessity_expressions={
            "for credit risk management": [
                "for credit risk management",
                "credit risk management",
                "与信リスク管理",
                "与信リスク管理のため",
                "credit risk",
                "与信",
            ],
        },
    )


def quotation_claims(locale: str = "en") -> dict[str, Claim]:
    """The hidden claim catalog for the quotation scenario (EN or JA)."""
    terms = QUOTATION_SURFACE_TERMS_EN if locale == "en" else QUOTATION_SURFACE_TERMS_JA
    return build_claims(quotation_truth(), quotation_sales_filter(), terms)


def quotation_sales_filter() -> StakeholderFilter:
    """The sales employee. Knows the whole workflow and the approval rationale,
    but cannot explain the month-end step's necessity, and does not know where the
    approval happens, nor the approval/send/month-end read+write data artifacts.

    Per-node attribute visibility matches tasks.json ``known_info``:
    - r: actor, writes
    - cc: actor, system, reads
    - cq: actor, system, reads, writes
    - ap: actor (system/reads/writes hidden; rationale via necessity)
    - sq: actor, system (reads/writes hidden)
    - me: actor, system, writes (reads hidden; necessity unknown)
    """
    return StakeholderFilter(
        name="sales",
        visible_node_ids=["r", "cc", "cq", "ap", "sq", "me"],
        visible_edge_ids=["e1", "e2", "e3", "e4", "e5", "e6"],
        visible_node_attributes={
            "r": ["actor", "writes"],
            "cc": ["actor", "system", "reads"],
            "cq": ["actor", "system", "reads", "writes"],
            "ap": ["actor"],
            "sq": ["actor", "system"],
            "me": ["actor", "system", "writes"],
        },
        visible_necessity={"ap": ["rationale"], "me": []},
    )


def quotation_finance_filter() -> StakeholderFilter:
    """A finance/audit stakeholder: sees the month-end step's rationale but not
    the approval-branch credit-risk rationale. Used to show multiple filters.

    The finance stakeholder only sees the month-end tail; per-node attribute
    visibility for the nodes it can see is kept at the global default (all four
    axes visible) since this filter is a demonstrative/secondary filter."""
    return StakeholderFilter(
        name="finance",
        visible_node_ids=["cq", "sq", "me"],
        visible_edge_ids=["e4", "e6"],
        visible_attributes=["actor", "system", "reads", "writes"],
        visible_necessity={
            "me": ["rationale", "owner", "evidence", "removal_impact"],
            "ap": [],
        },
    )


# ---------------------------------------------------------------------------
# lab_sample_flow
# ---------------------------------------------------------------------------


def lab_sample_truth() -> BusinessDAG:
    """A non-quotation Truth DAG (lab sample conditioning).

    Demonstrates open-world domain concepts ("specimen accession",
    "chamber seasoning", "conditioning cycle") that are not part of any global
    ontology. Derived read/write artifacts are hidden from the stakeholder.
    """
    return BusinessDAG(
        id="lab",
        name="Lab sample conditioning",
        data_concepts={
            "tc_sample": _concept("tc_sample", "sample"),
            "tc_accessioned_sample": _concept(
                "tc_accessioned_sample", "accessioned sample"
            ),
            "tc_seasoned_chamber": _concept("tc_seasoned_chamber", "seasoned chamber"),
            "tc_conditioned_sample": _concept(
                "tc_conditioned_sample", "conditioned sample"
            ),
            "tc_batch_approval": _concept("tc_batch_approval", "batch approval"),
        },
        nodes={
            "n1": Node(
                id="n1",
                action=_iv("specimen accession"),
                actor=_iv("lab tech"),
                reads=[_ref("tc_sample")],
                writes=[_ref("tc_accessioned_sample")],
            ),
            "n2": Node(
                id="n2",
                action=_iv("chamber seasoning"),
                actor=_iv("lab tech"),
                system=_iv("environment chamber"),
                writes=[_ref("tc_seasoned_chamber")],
            ),
            "n3": Node(
                id="n3",
                action=_iv("conditioning cycle"),
                actor=_iv("lab tech"),
                system=_iv("environment chamber"),
                reads=[_ref("tc_accessioned_sample")],
                writes=[_ref("tc_conditioned_sample")],
            ),
            "n4": Node(
                id="n4",
                action=_iv("approve conditioned batch"),
                actor=_iv("lab supervisor"),
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


LAB_SURFACE_TERMS: dict[tuple[str, str, str], list[str]] = {
    ("n1", "reads", "tc_sample"): ["sample", "specimen"],
}


def lab_sample_claims() -> dict[str, Claim]:
    return build_claims(lab_sample_truth(), lab_sample_filter(), LAB_SURFACE_TERMS)


def lab_sample_spec() -> EvaluationSpec:
    return EvaluationSpec(
        truth_nodes={
            "n1": TruthNodeSpec(
                expressions=[
                    "specimen accession",
                    "accession the sample",
                    "receive the specimen",
                ],
                primitive="receive",
            ),
            "n2": TruthNodeSpec(
                expressions=[
                    "chamber seasoning",
                    "season the chamber",
                    "prepare the conditioning chamber",
                ],
                primitive="create",
            ),
            "n3": TruthNodeSpec(
                expressions=[
                    "conditioning cycle",
                    "run the conditioning cycle",
                    "process samples in the chamber",
                ],
                primitive="transform",
            ),
            "n4": TruthNodeSpec(
                expressions=[
                    "approve conditioned batch",
                    "approve the batch",
                    "approve conditioned samples",
                ],
                primitive="approve",
            ),
        }
    )


def lab_sample_filter() -> StakeholderFilter:
    """The lab technician. Can state the actors, the environment-chamber system,
    and the raw specimen/sample input, but the GT read/write artifacts
    (``accessioned sample``, ``seasoned chamber``, ``conditioned sample``,
    ``batch approval``) are benchmark-derived state/artifact conventions that the
    stakeholder never names. They are therefore hidden: the correct agent
    behavior is to leave them unset (epistemic restraint), not to invent them.
    """
    return StakeholderFilter(
        name="lab tech",
        visible_node_ids=["n1", "n2", "n3", "n4"],
        visible_edge_ids=["l1", "l2", "l3"],
        visible_node_attributes={
            "n1": ["actor", "reads"],
            "n2": ["actor", "system"],
            "n3": ["actor", "system"],
            "n4": ["actor"],
        },
        visible_necessity={},
    )


@dataclass
class Scenario:
    scenario_id: str
    truth: BusinessDAG
    spec: EvaluationSpec
    stakeholder: StakeholderFilter
    claims: dict[str, Claim]
    stop_phrases: tuple[str, ...] = ()


_SCENARIOS: dict[str, Scenario] = {
    "quotation_workflow_1": Scenario(
        scenario_id="quotation_workflow_1",
        truth=quotation_truth(),
        spec=quotation_spec(),
        stakeholder=quotation_sales_filter(),
        claims=quotation_claims("en"),
        stop_phrases=QUOTATION_STOP_PHRASES,
    ),
    "lab_sample_flow": Scenario(
        scenario_id="lab_sample_flow",
        truth=lab_sample_truth(),
        spec=lab_sample_spec(),
        stakeholder=lab_sample_filter(),
        claims=lab_sample_claims(),
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
        # JA locale: same truth/spec/filter, JA claim catalog
        return replace(sc, claims=quotation_claims("ja"))
    return sc
