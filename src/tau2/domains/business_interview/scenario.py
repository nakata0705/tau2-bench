"""Scenarios for the graph-native business_interview benchmark (v11).

A scenario is a canonical **Truth graph** (a ``BusinessProcessGraph`` whose
nodes/edges reference ``TruthConcept``\\ s and whose explicit structural SOURCE
and SINK own the boundary), the stakeholder filter(s), and the
**StakeholderKnowledge** (``knowledge.py``): the
stakeholder's world model — a masked graph with three-valued property slots
(``ConceptRef | None | DONT_KNOW``) plus its local ``StakeholderKnowledgeConcept``\\ s.

There are **no generated claims and no authored business sentences**:
the graph is the semantic model; the stakeholder LLM realizes its knowledge
into natural language.
"""

from dataclasses import dataclass, replace
from typing import Optional

from tau2.domains.business_interview.graph import (
    BusinessProcessGraph,
    ConceptKind,
    ConceptRef,
    TruthConcept,
    TruthEdge,
    TruthNode,
    canonicalize_truth_graph,
)
from tau2.domains.business_interview.knowledge import (
    StakeholderKnowledge,
    project_knowledge,
)  # noqa: E402 - see module docstring
from tau2.domains.business_interview.stakeholder import (
    ConceptKnowledgeOverride,
    StakeholderFilter,
)

JA_SCENARIO_SUFFIX = "_ja"


@dataclass
class ScenarioStakeholder:
    """One named stakeholder simulator and its private knowledge view.

    The current runtime uses ``Scenario.stakeholder``/``knowledge`` as the
    active legacy simulator.  ``Scenario.stakeholders`` can additionally hold
    multiple named views for evaluator reference diagnostics; each view keeps
    its own stable identifier, role, filter, and StakeholderKnowledge.
    """

    stakeholder_id: str
    stakeholder: StakeholderFilter
    knowledge: StakeholderKnowledge
    stakeholder_name: Optional[str] = None
    stakeholder_role: Optional[str] = None

    @property
    def name(self) -> str:
        return self.stakeholder_name or self.stakeholder.name

    @property
    def role(self) -> Optional[str]:
        return self.stakeholder_role or self.stakeholder.role


def _default_stakeholder_id(filter_: StakeholderFilter) -> str:
    """Use explicit metadata first, then a stable name-derived fallback."""
    if filter_.stakeholder_id:
        return filter_.stakeholder_id
    slug = "_".join(filter_.name.lower().split())
    return "stakeholder" if not slug else slug


def _ref(concept_id: str) -> ConceptRef:
    return ConceptRef(concept_id=concept_id, confidence=1.0)


def _tconcept(
    cid: str, kind: ConceptKind, description: str, terms: list[str]
) -> TruthConcept:
    return TruthConcept(
        id=cid, kind=kind, description=description, canonical_terms=terms
    )


# ---------------------------------------------------------------------------
# quotation_workflow_1
# ---------------------------------------------------------------------------


def quotation_truth() -> BusinessProcessGraph:
    """The quotation workflow Truth graph (one entry, two business exits;
    structural SOURCE/SINK boundaries are explicit)."""
    return canonicalize_truth_graph(
        BusinessProcessGraph(
            id="quotation",
            name="Quotation creation",
            concepts={
                # activities
                "tc_activity_receive_request": _tconcept(
                    "tc_activity_receive_request",
                    "activity",
                    "Receive the customer's quotation request.",
                    ["receive the quotation request"],
                ),
                "tc_activity_check_customer": _tconcept(
                    "tc_activity_check_customer",
                    "activity",
                    "Check the customer information.",
                    ["check the customer information"],
                ),
                "tc_activity_create_quotation": _tconcept(
                    "tc_activity_create_quotation",
                    "activity",
                    "Create the quotation document.",
                    ["create the quotation"],
                ),
                "tc_activity_approve_quotation": _tconcept(
                    "tc_activity_approve_quotation",
                    "activity",
                    "Approve the quotation.",
                    ["approve the high-value quotation"],
                ),
                "tc_activity_send_quotation": _tconcept(
                    "tc_activity_send_quotation",
                    "activity",
                    "Send the quotation to the customer.",
                    ["send the quotation to the customer"],
                ),
                "tc_activity_send_month_end_summary": _tconcept(
                    "tc_activity_send_month_end_summary",
                    "activity",
                    "Send the month-end summary of quotation information.",
                    ["send the month-end summary to Accounting"],
                ),
                # actors
                "tc_actor_sales": _tconcept(
                    "tc_actor_sales", "actor", "The sales employee.", ["sales employee"]
                ),
                "tc_actor_manager": _tconcept(
                    "tc_actor_manager",
                    "actor",
                    "The manager who approves.",
                    ["manager"],
                ),
                # systems
                "tc_system_crm": _tconcept(
                    "tc_system_crm",
                    "system",
                    "The customer relationship management system.",
                    ["CRM"],
                ),
                "tc_system_quoting": _tconcept(
                    "tc_system_quoting",
                    "system",
                    "The quoting system.",
                    ["quoting system"],
                ),
                "tc_system_email": _tconcept(
                    "tc_system_email", "system", "The email system.", ["email"]
                ),
                "tc_system_excel": _tconcept(
                    "tc_system_excel",
                    "system",
                    "The spreadsheet application.",
                    ["Excel"],
                ),
                # data
                "tc_request": _tconcept(
                    "tc_request",
                    "data",
                    "The customer's quotation request.",
                    ["quotation request"],
                ),
                "tc_customer": _tconcept(
                    "tc_customer",
                    "data",
                    "The customer's information.",
                    ["customer information"],
                ),
                "tc_pricing": _tconcept(
                    "tc_pricing",
                    "data",
                    "The pricing information.",
                    ["pricing information"],
                ),
                "tc_quote": _tconcept(
                    "tc_quote", "data", "The quotation document.", ["quotation"]
                ),
                "tc_approval": _tconcept(
                    "tc_approval", "data", "The approval record.", ["approval"]
                ),
                "tc_sent_quote": _tconcept(
                    "tc_sent_quote", "data", "The sent quotation.", ["sent quotation"]
                ),
                "tc_excel_summary": _tconcept(
                    "tc_excel_summary",
                    "data",
                    "The quotation information summary.",
                    ["summary of the quotation information"],
                ),
                # conditions
                "tc_cond_over_1m": _tconcept(
                    "tc_cond_over_1m",
                    "condition",
                    "The quotation amount is over 1,000,000 yen.",
                    ["over 1,000,000 yen"],
                ),
                "tc_cond_at_or_below_1m": _tconcept(
                    "tc_cond_at_or_below_1m",
                    "condition",
                    "The quotation amount is at or below 1,000,000 yen.",
                    ["at or below 1,000,000 yen"],
                ),
                "tc_cond_month_end": _tconcept(
                    "tc_cond_month_end",
                    "condition",
                    "It is the end of the month.",
                    ["month-end"],
                ),
                # rationale
                "tc_rationale_credit_risk": _tconcept(
                    "tc_rationale_credit_risk",
                    "rationale",
                    "Credit risk management.",
                    ["credit risk management"],
                ),
            },
            nodes={
                "r": TruthNode(
                    id="r",
                    activity=_ref("tc_activity_receive_request"),
                    actor=_ref("tc_actor_sales"),
                    writes=[_ref("tc_request")],
                ),
                "cc": TruthNode(
                    id="cc",
                    activity=_ref("tc_activity_check_customer"),
                    actor=_ref("tc_actor_sales"),
                    system=_ref("tc_system_crm"),
                    reads=[_ref("tc_customer")],
                ),
                "cq": TruthNode(
                    id="cq",
                    activity=_ref("tc_activity_create_quotation"),
                    actor=_ref("tc_actor_sales"),
                    system=_ref("tc_system_quoting"),
                    reads=[_ref("tc_customer"), _ref("tc_pricing")],
                    writes=[_ref("tc_quote")],
                ),
                "ap": TruthNode(
                    id="ap",
                    activity=_ref("tc_activity_approve_quotation"),
                    actor=_ref("tc_actor_manager"),
                    necessity_rationale=_ref("tc_rationale_credit_risk"),
                ),
                "sq": TruthNode(
                    id="sq",
                    activity=_ref("tc_activity_send_quotation"),
                    actor=_ref("tc_actor_sales"),
                    system=_ref("tc_system_email"),
                ),
                "me": TruthNode(
                    id="me",
                    activity=_ref("tc_activity_send_month_end_summary"),
                    actor=_ref("tc_actor_sales"),
                    system=_ref("tc_system_excel"),
                    writes=[_ref("tc_excel_summary")],
                ),
            },
            edges={
                "e1": TruthEdge(id="e1", from_node="r", to_node="cc"),
                "e2": TruthEdge(id="e2", from_node="cc", to_node="cq"),
                "e3": TruthEdge(
                    id="e3",
                    from_node="cq",
                    to_node="ap",
                    condition=_ref("tc_cond_over_1m"),
                ),
                "e4": TruthEdge(
                    id="e4",
                    from_node="cq",
                    to_node="sq",
                    condition=_ref("tc_cond_at_or_below_1m"),
                ),
                "e5": TruthEdge(id="e5", from_node="ap", to_node="sq"),
                "e6": TruthEdge(
                    id="e6",
                    from_node="cq",
                    to_node="me",
                    condition=_ref("tc_cond_month_end"),
                ),
            },
            start_node_id="r",
            end_node_ids=["sq", "me"],
        ),
        entry_node_ids=["r"],
        exit_node_ids=["sq", "me"],
    )


def quotation_sales_filter() -> StakeholderFilter:
    """The sales employee. Knows the whole workflow and the approval
    rationale; does not know where the approval happens, nor the
    approval/send/month-end read+write data artifacts, nor any rationale
    other than the approval's.

    Per-node property knowledge:
    - r: activity, actor, writes
    - cc: activity, actor, system, reads
    - cq: activity, actor, system, reads, writes
    - ap: activity, actor, rationale
    - sq: activity, actor, system
    - me: activity, actor, system, writes
    Per-edge condition knowledge: e3/e4/e6 conditions known; e1/e2/e5
    conditions are NOT known (DONT_KNOW — asserting any condition there is
    wrong; restraint is correct).
    """
    return StakeholderFilter(
        name="sales",
        stakeholder_id="sales",
        role="sales",
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
            "e1": [],
            "e2": [],
            "e3": ["condition"],
            "e4": ["condition"],
            "e5": [],
            "e6": ["condition"],
        },
    )


def quotation_finance_filter() -> StakeholderFilter:
    """A finance/audit stakeholder: sees the month-end tail and retains the
    approval waypoint needed for safe topology projection, but does not know
    most upstream/approval semantic properties."""
    return StakeholderFilter(
        name="finance",
        stakeholder_id="finance",
        role="finance",
        # The approval node remains known as a structural waypoint; otherwise
        # its conditioned branch could not be safely contracted.  Upstream
        # serial nodes may still contract because all incident edge existence
        # is known, while their semantic properties remain unknown.
        visible_node_ids=["cq", "ap", "sq", "me"],
        visible_edge_ids=["e1", "e2", "e3", "e4", "e5", "e6"],
        visible_node_attributes={
            "cq": ["activity", "actor", "system", "reads", "writes"],
            "sq": ["activity", "actor", "system", "reads", "writes"],
            "me": ["activity", "actor", "system", "reads", "writes", "rationale"],
        },
        visible_edge_attributes={"e6": ["condition"]},
    )


# Japanese local terms for the quotation concepts (used by the JA locale
# knowledge projection so the stakeholder/agent vocabulary can be Japanese).
_JA_CONCEPT_TERMS: dict[str, list[str]] = {
    "tc_activity_receive_request": ["受注依頼の受領"],
    "tc_activity_check_customer": ["顧客情報の確認"],
    "tc_activity_create_quotation": ["見積書の作成"],
    "tc_activity_approve_quotation": ["見積書の承認"],
    "tc_activity_send_quotation": ["見積書の送付"],
    "tc_activity_send_month_end_summary": ["月末サマリーの送付"],
    "tc_actor_sales": ["営業担当"],
    "tc_actor_manager": ["管理者"],
    "tc_system_crm": ["ＣＲＭ"],
    "tc_system_quoting": ["見積システム"],
    "tc_system_email": ["メール"],
    "tc_system_excel": ["エクセル"],
    "tc_request": ["受注依頼"],
    "tc_customer": ["顧客情報"],
    "tc_pricing": ["価格情報"],
    "tc_quote": ["見積書"],
    "tc_excel_summary": ["見積情報サマリー"],
    "tc_cond_over_1m": ["100万円超"],
    "tc_cond_at_or_below_1m": ["100万円以下"],
    "tc_cond_month_end": ["月末"],
    "tc_rationale_credit_risk": ["与信リスク管理"],
}


def _localized_filter() -> StakeholderFilter:
    """The sales filter with per-concept Japanese local terms (used only by
    the JA locale knowledge projection)."""
    base = quotation_sales_filter()
    overrides = {
        cid: ConceptKnowledgeOverride(local_terms=terms)
        for cid, terms in _JA_CONCEPT_TERMS.items()
    }
    return base.model_copy(update={"concept_overrides": overrides})


def quotation_knowledge(locale: str = "en") -> StakeholderKnowledge:
    """The sales stakeholder's world model (projection of the Truth).

    For ``locale="ja"`` the knowledge concepts carry Japanese local terms
    (via per-concept overrides) so the stakeholder's vocabulary — and
    therefore the agent's labels — can be Japanese.
    """
    filt = _localized_filter() if locale == "ja" else quotation_sales_filter()
    return project_knowledge(quotation_truth(), filt)


# ---------------------------------------------------------------------------
# lab_sample_flow
# ---------------------------------------------------------------------------


def lab_sample_truth() -> BusinessProcessGraph:
    """A non-quotation Truth graph (lab sample conditioning)."""
    return canonicalize_truth_graph(
        BusinessProcessGraph(
            id="lab",
            name="Lab sample conditioning",
            concepts={
                "tc_activity_accession": _tconcept(
                    "tc_activity_accession",
                    "activity",
                    "Accession the specimen.",
                    ["specimen accession"],
                ),
                "tc_activity_seasoning": _tconcept(
                    "tc_activity_seasoning",
                    "activity",
                    "Season the environment chamber to prepare it.",
                    ["chamber seasoning"],
                ),
                "tc_activity_conditioning": _tconcept(
                    "tc_activity_conditioning",
                    "activity",
                    "Run the conditioning cycle on the samples.",
                    ["conditioning cycle"],
                ),
                "tc_activity_batch_approval": _tconcept(
                    "tc_activity_batch_approval",
                    "activity",
                    "Approve the conditioned batch.",
                    ["approve the conditioned batch"],
                ),
                "tc_actor_lab_tech": _tconcept(
                    "tc_actor_lab_tech",
                    "actor",
                    "The lab technician.",
                    ["lab technician"],
                ),
                "tc_actor_lab_supervisor": _tconcept(
                    "tc_actor_lab_supervisor",
                    "actor",
                    "The lab supervisor.",
                    ["lab supervisor"],
                ),
                "tc_system_chamber": _tconcept(
                    "tc_system_chamber",
                    "system",
                    "The environment chamber.",
                    ["environment chamber"],
                ),
                "tc_sample": _tconcept(
                    "tc_sample", "data", "The raw sample.", ["sample"]
                ),
                "tc_accessioned_sample": _tconcept(
                    "tc_accessioned_sample",
                    "data",
                    "The accessioned sample.",
                    ["accessioned sample"],
                ),
                "tc_seasoned_chamber": _tconcept(
                    "tc_seasoned_chamber",
                    "data",
                    "The seasoned chamber.",
                    ["seasoned chamber"],
                ),
                "tc_conditioned_sample": _tconcept(
                    "tc_conditioned_sample",
                    "data",
                    "The conditioned sample.",
                    ["conditioned sample"],
                ),
                "tc_batch_approval": _tconcept(
                    "tc_batch_approval",
                    "data",
                    "The batch approval.",
                    ["batch approval"],
                ),
            },
            nodes={
                "n1": TruthNode(
                    id="n1",
                    activity=_ref("tc_activity_accession"),
                    actor=_ref("tc_actor_lab_tech"),
                    reads=[_ref("tc_sample")],
                    writes=[_ref("tc_accessioned_sample")],
                ),
                "n2": TruthNode(
                    id="n2",
                    activity=_ref("tc_activity_seasoning"),
                    actor=_ref("tc_actor_lab_tech"),
                    system=_ref("tc_system_chamber"),
                    writes=[_ref("tc_seasoned_chamber")],
                ),
                "n3": TruthNode(
                    id="n3",
                    activity=_ref("tc_activity_conditioning"),
                    actor=_ref("tc_actor_lab_tech"),
                    system=_ref("tc_system_chamber"),
                    reads=[_ref("tc_accessioned_sample")],
                    writes=[_ref("tc_conditioned_sample")],
                ),
                "n4": TruthNode(
                    id="n4",
                    activity=_ref("tc_activity_batch_approval"),
                    actor=_ref("tc_actor_lab_supervisor"),
                    reads=[_ref("tc_conditioned_sample")],
                    writes=[_ref("tc_batch_approval")],
                ),
            },
            edges={
                "l1": TruthEdge(id="l1", from_node="n1", to_node="n2"),
                "l2": TruthEdge(id="l2", from_node="n2", to_node="n3"),
                "l3": TruthEdge(id="l3", from_node="n3", to_node="n4"),
            },
            start_node_id="n1",
            end_node_ids=["n4"],
        ),
        entry_node_ids=["n1"],
        exit_node_ids=["n4"],
    )


def lab_sample_filter() -> StakeholderFilter:
    """The lab technician. Can state the activities, actors, the
    environment-chamber system, and the raw specimen/sample input; the
    read/write artifacts are DONT_KNOW (correct agent behavior is epistemic
    restraint, not invention)."""
    return StakeholderFilter(
        name="lab tech",
        stakeholder_id="lab_tech",
        role="lab",
        visible_node_ids=["n1", "n2", "n3", "n4"],
        visible_edge_ids=["l1", "l2", "l3"],
        visible_node_attributes={
            "n1": ["activity", "actor", "reads"],
            "n2": ["activity", "actor", "system"],
            "n3": ["activity", "actor", "system"],
            "n4": ["activity", "actor"],
        },
        visible_edge_attributes={"l1": [], "l2": [], "l3": []},
    )


def lab_sample_knowledge() -> StakeholderKnowledge:
    return project_knowledge(lab_sample_truth(), lab_sample_filter())


@dataclass
class Scenario:
    scenario_id: str
    truth: BusinessProcessGraph
    stakeholder: StakeholderFilter
    knowledge: StakeholderKnowledge
    # Optional multi-stakeholder reference views.  The first profile is also
    # exposed through the legacy singular fields so existing simulation code
    # remains unchanged.
    stakeholders: Optional[tuple[ScenarioStakeholder, ...]] = None

    def __post_init__(self) -> None:
        if self.stakeholders:
            first = self.stakeholders[0]
            self.stakeholder = first.stakeholder
            self.knowledge = first.knowledge

    @property
    def stakeholder_references(self) -> tuple[ScenarioStakeholder, ...]:
        """Named views used by the evaluator's reference-only comparison."""
        if self.stakeholders:
            return self.stakeholders
        return (
            ScenarioStakeholder(
                stakeholder_id=_default_stakeholder_id(self.stakeholder),
                stakeholder=self.stakeholder,
                knowledge=self.knowledge,
            ),
        )


_SCENARIOS: dict[str, Scenario] = {
    "quotation_workflow_1": Scenario(
        scenario_id="quotation_workflow_1",
        truth=quotation_truth(),
        stakeholder=quotation_sales_filter(),
        knowledge=quotation_knowledge("en"),
    ),
    "lab_sample_flow": Scenario(
        scenario_id="lab_sample_flow",
        truth=lab_sample_truth(),
        stakeholder=lab_sample_filter(),
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
        # JA locale: same truth/filter; JA knowledge concept terms would be
        # applied via a locale-aware projection (concept overrides).
        return replace(sc, knowledge=quotation_knowledge("ja"))
    return sc
