"""Scenarios for the evidence-backed DAG business_interview benchmark (v3).

A scenario is a **Truth DAG** (a plain ``BusinessDAG``), an evaluator-only
``EvaluationSpec`` (concepts for semantic matching), and the stakeholder
filter(s). The Truth DAG and the agent's inferred DAG use the same class.
"""

from dataclasses import dataclass
from typing import Optional

from tau2.domains.business_interview.dag import (
    BusinessDAG,
    Edge,
    InferredValue,
    Necessity,
    Node,
)
from tau2.domains.business_interview.evaluation import EvaluationSpec
from tau2.domains.business_interview.stakeholder import StakeholderFilter

JA_SCENARIO_SUFFIX = "_ja"


def _iv(value: Optional[str]) -> InferredValue:
    return InferredValue(value=value)


def quotation_truth() -> BusinessDAG:
    """The quotation workflow Truth DAG (1 start / 2 ends)."""
    return BusinessDAG(
        id="quotation",
        name="Quotation creation",
        nodes={
            "r": Node(
                id="r",
                action=_iv("receive quotation request"),
                actor=_iv("sales"),
                writes=[_iv("request")],
            ),
            "cc": Node(
                id="cc",
                action=_iv("check customer information in the CRM"),
                actor=_iv("sales"),
                system=_iv("crm"),
                reads=[_iv("customer")],
            ),
            "cq": Node(
                id="cq",
                action=_iv("create quotation in the quoting system"),
                actor=_iv("sales"),
                system=_iv("quoting"),
                reads=[_iv("customer"), _iv("pricing")],
                writes=[_iv("quote")],
            ),
            "ap": Node(
                id="ap",
                action=_iv("approve high-value quotation"),
                actor=_iv("manager"),
                system=_iv("quoting"),
                reads=[_iv("quote")],
                writes=[_iv("approval")],
                necessity=Necessity(rationale=_iv("for credit risk management")),
            ),
            "sq": Node(
                id="sq",
                action=_iv("send quotation to customer"),
                actor=_iv("sales"),
                system=_iv("email"),
                reads=[_iv("quote")],
                writes=[_iv("sent_quote")],
            ),
            "me": Node(
                id="me",
                action=_iv("send quotation summary to accounting at month-end"),
                actor=_iv("sales"),
                system=_iv("excel"),
                reads=[_iv("quote")],
                writes=[_iv("excel_summary")],
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


def quotation_spec() -> EvaluationSpec:
    return EvaluationSpec(
        truth_node_concepts={
            "r": "receive_request",
            "cc": "check_customer",
            "cq": "create_quote",
            "ap": "approve_quote",
            "sq": "send_quote",
            "me": "month_end_summary",
        }
    )


def quotation_sales_filter() -> StakeholderFilter:
    """The sales employee. Knows the whole workflow and the approval rationale,
    but cannot explain the month-end step's necessity."""
    return StakeholderFilter(
        name="sales",
        visible_node_ids=["r", "cc", "cq", "ap", "sq", "me"],
        visible_edge_ids=["e1", "e2", "e3", "e4", "e5", "e6"],
        visible_attributes=["actor", "system", "reads", "writes"],
        visible_necessity={"ap": ["rationale"], "me": []},
    )


def quotation_finance_filter() -> StakeholderFilter:
    """A finance/audit stakeholder: sees the month-end step's rationale but not
    the approval-branch credit-risk rationale. Used to show multiple filters."""
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


@dataclass
class Scenario:
    scenario_id: str
    truth: BusinessDAG
    spec: EvaluationSpec
    stakeholder: StakeholderFilter


_SCENARIOS: dict[str, Scenario] = {
    "quotation_workflow_1": Scenario(
        scenario_id="quotation_workflow_1",
        truth=quotation_truth(),
        spec=quotation_spec(),
        stakeholder=quotation_sales_filter(),
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
    return _SCENARIOS.get(canonical)
