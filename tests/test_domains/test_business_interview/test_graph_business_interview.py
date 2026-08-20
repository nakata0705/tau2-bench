"""Tests for the graph-context business_interview domain (v7).

The agent builds a typed **glossary** (BusinessConcept with mentions and
validation status) and an inferred **BusinessProcessGraph** with declared
start/end. Truth semantics are **graph-contextual TruthClaims** (position +
property + value); node identity uses provenance plus reconstructed incoming
topology (the same activity may occur at several positions). Correctness is
grounded ONLY through private provenance with **span-correspondence
containment** (no cross-credit):

    ConceptRef -> EvidenceRef -> Observation span
        -> private assertion (claim_id + span) -> TruthClaim
        -> Truth BusinessConcept

Mention != terminology; confirmation requires genuine stakeholder evidence;
ConceptKind rules are enforced; a successful finish_interview terminates the
episode immediately. The same mechanism serves EN, JA and the lab scenario.
"""

from typing import Optional

import pytest  # type: ignore[reportMissingImports]

from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.evaluation import evaluate
from tau2.domains.business_interview.facts import (
    StakeholderAssertion,
)
from tau2.domains.business_interview.graph import (
    BusinessConcept,
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    TruthNodeContext,
)
from tau2.domains.business_interview.scenario import (
    get_scenario,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import BUSINESS_INTERVIEW_POLICY_PATH

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
LAB_SCENARIO = "lab_sample_flow"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO, LAB_SCENARIO]

# (agent node id, truth node id)
_NODE_MAP = {"a": "r", "b": "cc", "c": "cq", "d": "ap", "e": "sq", "f": "me"}

# Per-node observation text + private assertions (claim_id, quote) — EN.
# Quotes must be exact substrings of the observation text.
_NODE_OBS_EN = {
    "r": (
        "I receive the quotation request from the customer and record it.",
        [
            ("r.activity", "receive the quotation request"),
            ("r.actor", "I"),
            ("r.writes.tc_request", "record it"),
        ],
    ),
    "cc": (
        "I check the customer's information in the CRM.",
        [
            ("cc.activity", "check the customer's information"),
            ("cc.actor", "I"),
            ("cc.system", "CRM"),
            ("cc.reads.tc_customer", "customer's information"),
        ],
    ),
    "cq": (
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            ("cq.activity", "create the quotation"),
            ("cq.actor", "I"),
            ("cq.system", "quoting system"),
            ("cq.reads.tc_customer", "customer"),
            ("cq.reads.tc_pricing", "pricing information"),
            ("cq.writes.tc_quote", "quotation"),
        ],
    ),
    "ap": (
        "Quotations over 1,000,000 yen are approved by the manager; the "
        "approval is for credit risk management.",
        [
            ("ap.activity", "approved"),
            ("ap.actor", "manager"),
            ("ap.rationale", "credit risk management"),
        ],
    ),
    "sq": (
        "I send the quotation to the customer by email.",
        [
            ("sq.activity", "send the quotation"),
            ("sq.actor", "I"),
            ("sq.system", "email"),
        ],
    ),
    "me": (
        "At month-end I send the quotation information summary to Accounting "
        "as an Excel file.",
        [
            ("me.activity", "send the quotation information summary"),
            ("me.actor", "I"),
            ("me.system", "Excel"),
            ("me.writes.tc_excel_summary", "summary"),
        ],
    ),
}

_EDGE_OBS_EN = {
    "e1": (
        "After receiving the request, I check the customer information.",
        [("e1.edge_exists", "After receiving the request, I check")],
    ),
    "e2": (
        "After checking the customer information, I create the quotation.",
        [("e2.edge_exists", "After checking the customer information, I create")],
    ),
    "e3": (
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [
            ("e3.edge_exists", "go to the manager for approval"),
            ("e3.condition", "over 1,000,000 yen"),
        ],
    ),
    "e4": (
        "Quotations at or below 1,000,000 yen are sent directly to the customer.",
        [
            ("e4.edge_exists", "sent directly to the customer"),
            ("e4.condition", "at or below 1,000,000 yen"),
        ],
    ),
    "e5": (
        "Once approved, the quotation is sent to the customer.",
        [("e5.edge_exists", "Once approved, the quotation is sent")],
    ),
    "e6": (
        "At month-end I also send the summary to Accounting.",
        [
            ("e6.edge_exists", "send the summary to Accounting"),
            ("e6.condition", "month-end"),
        ],
    ),
}

_NODE_OBS_JA = {
    "r": (
        "私は顧客から見積依頼を受け付けて記録します。",
        [
            ("r.activity", "見積依頼を受け付け"),
            ("r.actor", "私"),
            ("r.writes.tc_request", "記録"),
        ],
    ),
    "cc": (
        "私はCRMで顧客情報を確認します。",
        [
            ("cc.activity", "確認"),
            ("cc.actor", "私"),
            ("cc.system", "CRM"),
            ("cc.reads.tc_customer", "顧客情報"),
        ],
    ),
    "cq": (
        "私は見積システムで顧客情報と価格情報を使って見積書を作成します。",
        [
            ("cq.activity", "見積書を作成"),
            ("cq.actor", "私"),
            ("cq.system", "見積システム"),
            ("cq.reads.tc_customer", "顧客情報"),
            ("cq.reads.tc_pricing", "価格情報"),
            ("cq.writes.tc_quote", "見積書"),
        ],
    ),
    "ap": (
        "100万円を超える見積書は管理者の承認が必要で、承認は与信リスク管理のためのものです。",
        [
            ("ap.activity", "承認が必要"),
            ("ap.actor", "管理者"),
            ("ap.rationale", "与信リスク管理"),
        ],
    ),
    "sq": (
        "私は見積書をメールで顧客に送付します。",
        [
            ("sq.activity", "送付"),
            ("sq.actor", "私"),
            ("sq.system", "メール"),
        ],
    ),
    "me": (
        "私は月末に見積情報の集計をExcelファイルとして経理チームに送ります。",
        [
            ("me.activity", "送り"),
            ("me.actor", "私"),
            ("me.system", "Excel"),
            ("me.writes.tc_excel_summary", "集計"),
        ],
    ),
}

_EDGE_OBS_JA = {
    "e1": (
        "依頼を受け付けたら、顧客情報を確認します。",
        [("e1.edge_exists", "受け付けたら、顧客情報を確認")],
    ),
    "e2": (
        "顧客情報を確認したら、見積書を作成します。",
        [("e2.edge_exists", "確認したら、見積書を作成")],
    ),
    "e3": (
        "100万円を超える見積書は管理者の承認に回ります。",
        [
            ("e3.edge_exists", "管理者の承認に回ります"),
            ("e3.condition", "100万円を超える"),
        ],
    ),
    "e4": (
        "100万円以下の見積書はそのまま顧客に送付します。",
        [
            ("e4.edge_exists", "そのまま顧客に送付します"),
            ("e4.condition", "100万円以下"),
        ],
    ),
    "e5": (
        "承認された見積書は顧客に送付します。",
        [("e5.edge_exists", "承認された見積書は顧客に送付")],
    ),
    "e6": (
        "月末には経理チームへの集計も送ります。",
        [
            ("e6.edge_exists", "経理チームへの集計も送ります"),
            ("e6.condition", "月末"),
        ],
    ),
}

# Agent-local glossary ids (kind, label) per truth concept.
_AGENT_CONCEPTS = {
    "r": ("act_receive", "activity", "receive the quotation request"),
    "cc": ("act_check", "activity", "check the customer information"),
    "cq": ("act_create", "activity", "create the quotation"),
    "ap": ("act_approve", "activity", "approve the high-value quotation"),
    "sq": ("act_send", "activity", "send the quotation"),
    "me": ("act_me", "activity", "send the month-end summary"),
    "tc_actor_sales": ("sales", "actor", "sales"),
    "tc_actor_manager": ("manager", "actor", "manager"),
    "tc_system_crm": ("crm", "system", "CRM"),
    "tc_system_quoting": ("quoting", "system", "quoting system"),
    "tc_system_email": ("email", "system", "email"),
    "tc_system_excel": ("excel", "system", "Excel"),
    "tc_request": ("request", "data", "quotation request"),
    "tc_customer": ("customer", "data", "customer information"),
    "tc_pricing": ("pricing", "data", "pricing information"),
    "tc_quote": ("quote", "data", "quotation"),
    "tc_excel_summary": ("excel_summary", "data", "quotation information summary"),
    "tc_cond_over_1m": ("over_1m", "condition", "over 1,000,000 yen"),
    "tc_cond_at_or_below_1m": (
        "at_or_below_1m",
        "condition",
        "at or below 1,000,000 yen",
    ),
    "tc_cond_month_end": ("month_end", "condition", "month-end"),
    "tc_rationale_credit_risk": ("credit_risk", "rationale", "credit risk management"),
}


def _tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


def _ingest(tools: InterviewTools, role: str = "user", content: str = "") -> int:
    tools.db.messages.append({"role": role, "content": content})
    return len(tools.db.messages) - 1


def _assertion(claim_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"claim_id": claim_id, "quote": quote, "occurrence": occurrence}


def _claim_obs(
    tools: InterviewTools,
    text: str,
    assertions: Optional[list[dict]] = None,
    alignments: Optional[list[dict]] = None,
    terminology: Optional[list[dict]] = None,
) -> str:
    """Ingest a stakeholder message, bind its private sidecar (assertions +
    optional dialogue events) at that exact turn, and capture it as an
    Observation."""
    from tau2.domains.business_interview.facts import (
        ConceptAlignmentAssertion,
        TerminologyConfirmation,
    )

    turn = _ingest(tools, "user", text)
    if assertions:
        tools.assertion_ledger.bind(
            turn, [StakeholderAssertion(**a) for a in assertions], text
        )
    if alignments:
        tools.assertion_ledger.bind_alignment(
            turn, [ConceptAlignmentAssertion(**a) for a in alignments], text
        )
    if terminology:
        tools.assertion_ledger.bind_terminology(
            turn, [TerminologyConfirmation(**a) for a in terminology], text
        )
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


def _ev(obs_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"observation_id": obs_id, "quote": quote, "occurrence": occurrence}


def _evr(obs_id: str, quote: str, occurrence: int = 0) -> EvidenceRef:
    """Direct EvidenceRef (for graph mutation); tools accept dicts via _ev."""
    return EvidenceRef(observation_id=obs_id, quote=quote, occurrence=occurrence)


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    """Evaluate under the scenario's stakeholder visibility + private
    provenance (runtime behavior: the sidecar ledger of the tools)."""
    sc = get_scenario(scenario)
    assert sc is not None
    return evaluate(
        tools.db,
        sc.truth,
        sc.spec,
        sc.stakeholder,
        claims=sc.claims,
        assertions=tools.assertion_ledger.assertions(),
        alignments=tools.assertion_ledger.alignments(),
        terminology=tools.assertion_ledger.terminology(),
    )


def _make_concept(
    tools: InterviewTools,
    truth_cid: str,
    oid: str,
    quote: str,
) -> str:
    """Create (and confirm) the agent concept for a truth concept id using the
    given observation span, returning the agent concept id."""
    agent_cid, kind, _label = _AGENT_CONCEPTS[truth_cid]
    tools.create_concept(agent_cid, kind, _label, evidence=[_ev(oid, quote)])
    tools.confirm_concept(agent_cid, evidence=[_ev(oid, quote)])
    return agent_cid


def _quote_for_concept(node_obs, truth_cid: str) -> tuple[str, str]:
    for sid, (text, assertions) in node_obs.items():
        for cid, q in assertions:
            if _claim_to_concept(cid) == truth_cid:
                return sid, q
    raise KeyError(truth_cid)


def _confirm_alignment_for(
    truth_cid: str, quote: str, occurrence: int = 0
) -> dict:
    """A private concept-alignment event simulating a genuine confirmation of
    ``truth_cid`` performed at ``quote`` (test-side ledger binding)."""
    event: dict = {"truth_concept_id": truth_cid, "quote": quote, "act": "confirm"}
    if occurrence:
        event["occurrence"] = occurrence
    return event


def _unknown_alignment_for(truth_cid: str, quote: str) -> dict:
    """A private concept-alignment event (act=unknown) at ``quote``."""
    return {"truth_concept_id": truth_cid, "quote": quote, "act": "unknown"}


def _dispute_alignment_for(truth_cid: str, quote: str) -> dict:
    """A private concept-alignment event (act=dispute) at ``quote``."""
    return {"truth_concept_id": truth_cid, "quote": quote, "act": "dispute"}


def _claim_to_concept(claim_id: str) -> Optional[str]:
    mapping = {
        "r.activity": "r",
        "r.actor": "tc_actor_sales",
        "r.writes.tc_request": "tc_request",
        "cc.activity": "cc",
        "cc.actor": "tc_actor_sales",
        "cc.system": "tc_system_crm",
        "cc.reads.tc_customer": "tc_customer",
        "cq.activity": "cq",
        "cq.actor": "tc_actor_sales",
        "cq.system": "tc_system_quoting",
        "cq.reads.tc_customer": "tc_customer",
        "cq.reads.tc_pricing": "tc_pricing",
        "cq.writes.tc_quote": "tc_quote",
        "ap.activity": "ap",
        "ap.actor": "tc_actor_manager",
        "ap.rationale": "tc_rationale_credit_risk",
        "sq.activity": "sq",
        "sq.actor": "tc_actor_sales",
        "sq.system": "tc_system_email",
        "me.activity": "me",
        "me.actor": "tc_actor_sales",
        "me.system": "tc_system_excel",
        "me.writes.tc_excel_summary": "tc_excel_summary",
    }
    return mapping.get(claim_id)


_ACTIVITY_TRUTH_CONCEPTS = {
    "r": "tc_activity_receive_request",
    "cc": "tc_activity_check_customer",
    "cq": "tc_activity_create_quotation",
    "ap": "tc_activity_approve_quotation",
    "sq": "tc_activity_send_quotation",
    "me": "tc_activity_send_month_end_summary",
}


def _claim_to_truth_concept(claim_id: str) -> Optional[str]:
    """The REAL Truth concept id of a claim (node-id activity claims are
    translated to their Truth activity concepts) — used for private
    dialogue-event bindings."""
    cid = _claim_to_concept(claim_id)
    if cid is None:
        return None
    return _ACTIVITY_TRUTH_CONCEPTS.get(cid, cid)


def _has_quote(node_obs, sid: str, truth_cid: str) -> bool:
    return any(_claim_to_concept(cid) == truth_cid for cid, q in node_obs[sid][1])


def _visible_for(sid: str, prop: str, ja: bool) -> bool:
    """True when ``prop`` is visible for the TRUTH node id ``sid``."""
    sc = get_scenario(JA_SCENARIO if ja else SCENARIO)
    assert sc is not None
    visible = sc.stakeholder.node_properties_for(sid)
    if prop == "necessity_rationale":
        return "rationale" in visible
    return prop in visible


def _quote_for_sid_activity(assertions) -> str:
    return next(q for cid, q in assertions if cid.endswith(".activity"))


def _agent_to_truth(agent_cid: str) -> Optional[str]:
    for truth_cid, (cid, kind, label) in _AGENT_CONCEPTS.items():
        if cid == agent_cid:
            return truth_cid
    return None


def _attach_ref_evidence(
    tools: InterviewTools, anid: str, oid: str, assertions, ja: bool
) -> None:
    """Attach per-property evidence spans to the node's refs."""
    assert tools.db.graph is not None
    node = tools.db.graph.nodes[anid]
    for prop, truth_cid in (
        ("actor", "tc_actor_sales"),
        ("actor", "tc_actor_manager"),
        ("system", "tc_system_crm"),
        ("system", "tc_system_quoting"),
        ("system", "tc_system_email"),
        ("system", "tc_system_excel"),
        ("necessity_rationale", "tc_rationale_credit_risk"),
    ):
        ref = getattr(node, prop)
        if ref is None:
            continue
        for cid, q in assertions:
            if _claim_to_concept(cid) == truth_cid:
                ref.evidence.append(
                    EvidenceRef(observation_id=oid, quote=q, occurrence=0)
                )
    for axis in ("reads", "writes"):
        for ref in getattr(node, axis):
            for cid, q in assertions:
                if _claim_to_concept(cid) == _agent_to_truth(ref.concept_id):
                    ref.evidence.append(
                        EvidenceRef(observation_id=oid, quote=q, occurrence=0)
                    )


def _build(tools: InterviewTools, ja: bool = False) -> None:
    """Build the correct quotation graph with full provenance, asserting only
    stakeholder-visible properties. Every referenced concept is confirmed with
    genuine evidence (each confirmation bound as a private concept-alignment
    event); endpoints are declared, so completion succeeds and the evaluator
    returns a full pass."""
    tools.start_inference("quotation")
    _ingest(tools, "assistant", "Hello.")
    node_obs = _NODE_OBS_JA if ja else _NODE_OBS_EN
    edge_obs = _EDGE_OBS_JA if ja else _EDGE_OBS_EN
    created: dict[str, str] = {}
    node_oids: dict[str, str] = {}
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, assertions = node_obs[sid]
        alignments = []
        for cid, q in assertions:
            truth_cid = _claim_to_truth_concept(cid)
            if truth_cid is not None:
                alignments.append(_confirm_alignment_for(truth_cid, q))
        oid = _claim_obs(
            tools,
            text,
            [_assertion(cid, q) for cid, q in assertions],
            alignments=alignments,
        )
        node_oids[sid] = oid

    # concepts first (activity per node; actor/system/data per truth concept)
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        oid = node_oids[sid]
        act_quote = _quote_for_sid_activity(node_obs[sid][1])
        created[sid] = _make_concept(tools, sid, oid, act_quote)
    for truth_cid in (
        "tc_actor_sales",
        "tc_actor_manager",
        "tc_system_crm",
        "tc_system_quoting",
        "tc_system_email",
        "tc_system_excel",
        "tc_request",
        "tc_customer",
        "tc_pricing",
        "tc_quote",
        "tc_excel_summary",
        "tc_rationale_credit_risk",
    ):
        sid, quote = _quote_for_concept(node_obs, truth_cid)
        created[truth_cid] = _make_concept(tools, truth_cid, node_oids[sid], quote)

    for anid, sid in _NODE_MAP.items():
        oid = node_oids[sid]
        props: dict[str, str] = {}
        for prop, truth_cid in (
            ("actor", "tc_actor_sales"),
            ("actor", "tc_actor_manager"),
            ("system", "tc_system_crm"),
            ("system", "tc_system_quoting"),
            ("system", "tc_system_email"),
            ("system", "tc_system_excel"),
            ("necessity_rationale", "tc_rationale_credit_risk"),
        ):
            if (
                truth_cid in created
                and _visible_for(sid, prop, ja)
                and _has_quote(node_obs, sid, truth_cid)
            ):
                props[prop] = created[truth_cid]
        reads = []
        writes = []
        for truth_cid, axis in (
            ("tc_customer", "reads"),
            ("tc_pricing", "reads"),
            ("tc_request", "writes"),
            ("tc_quote", "writes"),
            ("tc_excel_summary", "writes"),
        ):
            if (
                truth_cid in created
                and _visible_for(sid, axis, ja)
                and _has_quote(node_obs, sid, truth_cid)
            ):
                (reads if axis == "reads" else writes).append(created[truth_cid])
        tools.add_node(
            anid,
            activity=created[sid],
            actor=props.get("actor"),
            system=props.get("system"),
            reads=reads or None,
            writes=writes or None,
            necessity_rationale=props.get("necessity_rationale"),
            evidence=[_ev(oid, _quote_for_sid_activity(node_obs[sid][1]))],
        )
        _attach_ref_evidence(tools, anid, oid, node_obs[sid][1], ja)

    # edges
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, assertions = edge_obs[eid]
        alignments = [
            _confirm_alignment_for(
                {
                    "e3": "tc_cond_over_1m",
                    "e4": "tc_cond_at_or_below_1m",
                    "e6": "tc_cond_month_end",
                }[eid],
                q,
            )
            for cid, q in assertions
            if cid.endswith(".condition")
        ]
        oid = _claim_obs(
            tools,
            text,
            [_assertion(cid, q) for cid, q in assertions],
            alignments=alignments,
        )
        frm, to = {
            "e1": ("a", "b"),
            "e2": ("b", "c"),
            "e3": ("c", "d"),
            "e4": ("c", "e"),
            "e5": ("d", "e"),
            "e6": ("c", "f"),
        }[eid]
        cond = None
        for claim_id, q in assertions:
            if claim_id.endswith(".condition"):
                truth_cid = {
                    "e3": "tc_cond_over_1m",
                    "e4": "tc_cond_at_or_below_1m",
                    "e6": "tc_cond_month_end",
                }[eid]
                cond = _make_concept(tools, truth_cid, oid, q)
        tools.add_edge(
            eid,
            frm,
            to,
            condition=cond,
            evidence=[
                _ev(oid, q) for cid, q in assertions if cid.endswith(".edge_exists")
            ],
        )
    tools.set_graph_endpoints(start_node_id="a", end_node_ids=["e", "f"])
    tools.finish_interview()


def _lab_claim_to_concept(claim_id: str) -> Optional[str]:
    """The Truth concept referenced by a lab claim id."""
    if claim_id in (
        "n1.reads.tc_sample",
        "n2.writes.tc_seasoned_chamber",
        "n3.reads.tc_accessioned_sample",
        "n3.writes.tc_conditioned_sample",
        "n4.reads.tc_conditioned_sample",
        "n4.writes.tc_batch_approval",
    ):
        return claim_id.rsplit(".", 1)[1]
    return {
        "n1.activity": "tc_activity_accession",
        "n2.activity": "tc_activity_seasoning",
        "n3.activity": "tc_activity_conditioning",
        "n4.activity": "tc_activity_batch_approval",
        "n1.actor": "tc_actor_lab_tech",
        "n2.actor": "tc_actor_lab_tech",
        "n3.actor": "tc_actor_lab_tech",
        "n4.actor": "tc_actor_lab_supervisor",
        "n2.system": "tc_system_chamber",
        "n3.system": "tc_system_chamber",
    }.get(claim_id)


def _build_lab(tools: InterviewTools) -> None:
    """Build a lab graph asserting ONLY stakeholder-visible properties with
    full provenance and confirmed concepts (each confirmation bound as a
    private concept-alignment event)."""
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        (
            "n1",
            "When a specimen arrives, I accession it and record it as received.",
            [
                ("n1.activity", "accession"),
                ("n1.actor", "I"),
                ("n1.reads.tc_sample", "specimen"),
            ],
        ),
        (
            "n2",
            "I season the environment chamber to prepare it.",
            [
                ("n2.activity", "season"),
                ("n2.actor", "I"),
                ("n2.system", "environment chamber"),
            ],
        ),
        (
            "n3",
            "I run a conditioning cycle that processes the samples inside the chamber.",
            [
                ("n3.activity", "conditioning cycle"),
                ("n3.actor", "I"),
                ("n3.system", "chamber"),
            ],
        ),
        (
            "n4",
            "The lab supervisor approves the conditioned batch before it is released.",
            [
                ("n4.activity", "approves"),
                ("n4.actor", "lab supervisor"),
            ],
        ),
    ]
    created: dict[str, str] = {}

    def mk(cid: str, kind: str, label: str, oid: str, quote: str) -> str:
        tools.create_concept(cid, kind, label, evidence=[_ev(oid, quote)])
        tools.confirm_concept(cid, evidence=[_ev(oid, quote)])
        return cid

    for sid, text, assertions in nodes:
        alignments = []
        for cid, q in assertions:
            truth_cid = _lab_claim_to_concept(cid)
            if truth_cid is not None:
                alignments.append(_confirm_alignment_for(truth_cid, q))
        oid = _claim_obs(
            tools,
            text,
            [_assertion(cid, q) for cid, q in assertions],
            alignments=alignments,
        )
        act_cid = {
            "n1": "act_accession",
            "n2": "act_seasoning",
            "n3": "act_conditioning",
            "n4": "act_approval",
        }[sid]
        created[f"act_{sid}"] = mk(
            act_cid,
            "activity",
            {
                "n1": "specimen accession",
                "n2": "chamber seasoning",
                "n3": "conditioning cycle",
                "n4": "approve conditioned batch",
            }[sid],
            oid,
            next(q for cid, q in assertions if cid.endswith(".activity")),
        )
        actor_quote = {"n1": "I", "n2": "I", "n3": "I", "n4": "lab supervisor"}[sid]
        actor_cid = "lab_tech" if sid != "n4" else "lab_supervisor"
        if actor_cid not in created:
            created[actor_cid] = mk(
                actor_cid,
                "actor",
                "lab tech" if actor_cid == "lab_tech" else "lab supervisor",
                oid,
                actor_quote,
            )
        args = {
            "node_id": sid,
            "activity": created[f"act_{sid}"],
            "actor": created[actor_cid],
            "evidence": [
                _ev(oid, next(q for cid, q in assertions if cid.endswith(".activity")))
            ],
        }
        if sid == "n1":
            created["sample"] = mk("sample", "data", "sample", oid, "specimen")
            args["reads"] = [created["sample"]]
        if sid in ("n2", "n3"):
            if "chamber" not in created:
                created["chamber"] = mk(
                    "chamber",
                    "system",
                    "environment chamber",
                    oid,
                    "environment chamber" if sid == "n2" else "chamber",
                )
            args["system"] = "chamber"
        tools.add_node(**args)
        assert tools.db.graph is not None
        node = tools.db.graph.nodes[sid]
        if sid == "n1":
            node.reads[0].evidence.append(_evr(oid, "specimen"))
        if sid in ("n2", "n3"):
            assert node.system is not None
            node.system.evidence.append(
                _evr(oid, "environment chamber" if sid == "n2" else "chamber")
            )
        assert node.actor is not None
        node.actor.evidence.append(_evr(oid, actor_quote))

    edges = [
        (
            "l1",
            "n1",
            "n2",
            "After specimen accession, I prepare the chamber for seasoning.",
            [("l1.edge_exists", "After specimen accession, I prepare")],
        ),
        (
            "l2",
            "n2",
            "n3",
            "After chamber seasoning, I run the conditioning cycle.",
            [("l2.edge_exists", "After chamber seasoning, I run")],
        ),
        (
            "l3",
            "n3",
            "n4",
            "After the conditioning cycle, the lab supervisor approves the batch.",
            [
                (
                    "l3.edge_exists",
                    "After the conditioning cycle, the lab supervisor approves",
                )
            ],
        ),
    ]
    for eid, frm, to, text, assertions in edges:
        oid = _claim_obs(tools, text, [_assertion(cid, q) for cid, q in assertions])
        tools.add_edge(eid, frm, to, evidence=[_ev(oid, q) for cid, q in assertions])
    tools.set_graph_endpoints(start_node_id="n1", end_node_ids=["n4"])
    tools.finish_interview()


def _reference_trajectory() -> list:
    """A full faithful trajectory (messages + tool calls) mirroring ``_build``,
    with the private assertion sidecars carried on the UserMessages.

    The construction tools' ledger is bound the same way the environment binds
    it during replay, so tool-time validation (e.g. confirm_concept evidence)
    sees the assertions.
    """
    from tau2.data_model.message import AssistantMessage as AM
    from tau2.data_model.message import UserMessage as UM

    traj = []
    tools = InterviewTools(InterviewDB())
    cid = 0
    traj.append(AM(role="assistant", content="Hello, I'd like to interview you."))
    tools.db.messages.append(
        {"role": "assistant", "content": "Hello, I'd like to interview you."}
    )

    def mk(name, args):
        nonlocal cid
        cid += 1
        tc = ToolCall(id=f"c{cid}", name=name, arguments=args, requestor="assistant")
        traj.append(AM(role="assistant", tool_calls=[tc]))
        tools.db.messages.append({"role": "assistant", "content": None})
        res = getattr(tools, name)(**args)
        traj.append(
            ToolMessage(role="tool", id=f"c{cid}", content=res, requestor="assistant")
        )
        tools.db.messages.append({"role": "tool", "content": res})
        return res

    def say(text, assertions, alignments=None, terminology=None):
        from tau2.domains.business_interview.facts import (
            ConceptAlignmentAssertion,
            TerminologyConfirmation,
        )

        traj.append(
            UM(
                role="user",
                content=text,
                stakeholder_assertions=[
                    _assertion(claim_id, q) for claim_id, q in assertions
                ],
                stakeholder_alignments=list(alignments or []),
                stakeholder_terminology=list(terminology or []),
            )
        )
        tools.db.messages.append({"role": "user", "content": text})
        turn = len(tools.db.messages) - 1
        tools.assertion_ledger.bind(
            turn,
            [
                StakeholderAssertion(**_assertion(claim_id, q))
                for claim_id, q in assertions
            ],
            text,
        )
        if alignments:
            tools.assertion_ledger.bind_alignment(
                turn, [ConceptAlignmentAssertion(**a) for a in alignments], text
            )
        if terminology:
            tools.assertion_ledger.bind_terminology(
                turn, [TerminologyConfirmation(**a) for a in terminology], text
            )

    mk("start_inference", {"name": "Quotation creation"})
    created: set[str] = set()
    node_oid: dict[str, str] = {}
    sm = 0
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, assertions = _NODE_OBS_EN[sid]
        sm += 1
        alignments = []
        for claim_id, q in assertions:
            truth_cid = _claim_to_truth_concept(claim_id)
            if truth_cid is not None:
                alignments.append(_confirm_alignment_for(truth_cid, q))
        say(text, assertions, alignments=alignments)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        node_oid[sid] = oid
        for claim_id, q in assertions:
            truth_cid = _claim_to_concept(claim_id)
            if truth_cid is None:
                continue
            agent_cid, kind, label = _AGENT_CONCEPTS[truth_cid]
            if agent_cid not in created:
                created.add(agent_cid)
                mk(
                    "create_concept",
                    {
                        "concept_id": agent_cid,
                        "kind": kind,
                        "label": label,
                        "evidence": [_ev(oid, q)],
                    },
                )
                mk(
                    "confirm_concept",
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                )
            else:
                mk(
                    "add_concept_mention",
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                )
    for anid, sid in _NODE_MAP.items():
        props: dict[str, str] = {}
        for prop, truth_cid in (
            ("actor", "tc_actor_sales"),
            ("actor", "tc_actor_manager"),
            ("system", "tc_system_crm"),
            ("system", "tc_system_quoting"),
            ("system", "tc_system_email"),
            ("system", "tc_system_excel"),
            ("necessity_rationale", "tc_rationale_credit_risk"),
        ):
            if _visible_for(sid, prop, False) and _has_quote(
                _NODE_OBS_EN, sid, truth_cid
            ):
                props[prop] = _AGENT_CONCEPTS[truth_cid][0]
        reads = [
            _AGENT_CONCEPTS[t][0]
            for t in ("tc_customer", "tc_pricing")
            if _visible_for(sid, "reads", False) and _has_quote(_NODE_OBS_EN, sid, t)
        ]
        writes = [
            _AGENT_CONCEPTS[t][0]
            for t in ("tc_request", "tc_quote", "tc_excel_summary")
            if _visible_for(sid, "writes", False) and _has_quote(_NODE_OBS_EN, sid, t)
        ]
        act_quote = _quote_for_sid_activity(_NODE_OBS_EN[sid][1])
        args = {
            "node_id": anid,
            "activity": _AGENT_CONCEPTS[sid][0],
            "evidence": [_ev(node_oid[sid], act_quote)],
        }
        if "actor" in props:
            args["actor"] = props["actor"]
        if "system" in props:
            args["system"] = props["system"]
        if reads:
            args["reads"] = reads
        if writes:
            args["writes"] = writes
        if "necessity_rationale" in props:
            args["necessity_rationale"] = props["necessity_rationale"]
        mk("add_node", args)
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, assertions = _EDGE_OBS_EN[eid]
        sm += 1
        alignments = []
        for claim_id, q in assertions:
            if claim_id.endswith(".condition"):
                truth_cid = {
                    "e3": "tc_cond_over_1m",
                    "e4": "tc_cond_at_or_below_1m",
                    "e6": "tc_cond_month_end",
                }[eid]
                alignments.append(_confirm_alignment_for(truth_cid, q))
        say(text, assertions, alignments=alignments)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        cond = None
        for claim_id, q in assertions:
            if claim_id.endswith(".condition"):
                truth_cid = {
                    "e3": "tc_cond_over_1m",
                    "e4": "tc_cond_at_or_below_1m",
                    "e6": "tc_cond_month_end",
                }[eid]
                agent_cid, kind, label = _AGENT_CONCEPTS[truth_cid]
                if agent_cid not in created:
                    created.add(agent_cid)
                    mk(
                        "create_concept",
                        {
                            "concept_id": agent_cid,
                            "kind": kind,
                            "label": label,
                            "evidence": [_ev(oid, q)],
                        },
                    )
                    mk(
                        "confirm_concept",
                        {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                    )
                cond = agent_cid
        frm, to = {
            "e1": ("a", "b"),
            "e2": ("b", "c"),
            "e3": ("c", "d"),
            "e4": ("c", "e"),
            "e5": ("d", "e"),
            "e6": ("c", "f"),
        }[eid]
        eargs = {
            "edge_id": eid,
            "from_node": frm,
            "to_node": to,
            "evidence": [
                _ev(oid, q) for cid, q in assertions if cid.endswith(".edge_exists")
            ],
        }
        if cond:
            eargs["condition"] = cond
        mk("add_edge", eargs)
    mk("set_graph_endpoints", {"start_node_id": "a", "end_node_ids": ["e", "f"]})
    mk("finish_interview", {"summary": "Inferred the quotation graph."})
    return traj


# ---------------------------------------------------------------------------
# Graph contexts / topology
# ---------------------------------------------------------------------------


def test_truth_node_contexts_include_all_incoming_edges():
    """TruthNodeContext lists ALL incoming Truth edges per position, with the
    start flag — including for the quotation scenario's branch joins."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    ctx = sc.truth.node_contexts()
    assert ctx["r"].incoming_edge_ids == []
    assert ctx["r"].is_start is True
    assert ctx["cc"].incoming_edge_ids == ["e1"]
    assert ctx["cq"].incoming_edge_ids == ["e2"]
    # the send position merges the approval and direct branches
    assert ctx["sq"].incoming_edge_ids == ["e4", "e5"]
    assert ctx["sq"].is_start is False
    assert ctx["me"].incoming_edge_ids == ["e6"]
    assert isinstance(ctx["sq"], TruthNodeContext)


def test_truth_node_context_cycle_start_and_rework():
    """A cycle: the rework node's incoming includes the normal edge AND the
    back edge; the start node may receive a back edge yet stays the start."""
    truth = BusinessProcessGraph(
        nodes={
            "start": Node(id="start", activity=ConceptRef(concept_id="a")),
            "work": Node(id="work", activity=ConceptRef(concept_id="b")),
            "rework": Node(id="rework", activity=ConceptRef(concept_id="c")),
        },
        edges={
            "normal_edge": Edge(id="normal_edge", from_node="start", to_node="work"),
            "e2": Edge(id="e2", from_node="work", to_node="rework"),
            "back_edge": Edge(id="back_edge", from_node="rework", to_node="work"),
        },
        concepts={
            "a": BusinessConcept(id="a", kind="activity", display_label="a"),
            "b": BusinessConcept(id="b", kind="activity", display_label="b"),
            "c": BusinessConcept(id="c", kind="activity", display_label="c"),
        },
        start_node_id="start",
        end_node_ids=["rework"],
    )
    ctx = truth.node_contexts()
    assert ctx["start"].incoming_edge_ids == [] and ctx["start"].is_start
    assert ctx["work"].incoming_edge_ids == ["back_edge", "normal_edge"]
    assert ctx["rework"].incoming_edge_ids == ["e2"]


def test_same_activity_at_distinct_positions_disambiguated_by_topology():
    """The same activity concept at two Truth positions: node correspondence
    uses provenance PLUS reconstructed incoming topology."""
    truth = BusinessProcessGraph(
        nodes={
            "s": Node(id="s", activity=ConceptRef(concept_id="act_start")),
            "x1": Node(id="x1", activity=ConceptRef(concept_id="act_x")),
            "x2": Node(id="x2", activity=ConceptRef(concept_id="act_x")),
        },
        edges={
            "e1": Edge(id="e1", from_node="s", to_node="x1"),
            "e2": Edge(id="e2", from_node="s", to_node="x2"),
        },
        concepts={
            "act_start": BusinessConcept(
                id="act_start", kind="activity", display_label="start"
            ),
            "act_x": BusinessConcept(id="act_x", kind="activity", display_label="x"),
        },
        start_node_id="s",
        end_node_ids=["x1", "x2"],
    )
    from tau2.domains.business_interview.claims import build_claims
    from tau2.domains.business_interview.stakeholder import StakeholderFilter

    filter_ = StakeholderFilter(
        name="any",
        visible_node_ids=["s", "x1", "x2"],
        visible_edge_ids=["e1", "e2"],
        visible_attributes=["activity"],
    )
    claims = build_claims(truth, filter_)
    assert "x1.activity" in claims and "x2.activity" in claims

    tools = _tools()
    tools.db.graph = BusinessProcessGraph(
        nodes={
            "s": Node(id="s", activity=ConceptRef(concept_id="s_act")),
            "n1": Node(id="n1", activity=ConceptRef(concept_id="x_act")),
            "n2": Node(id="n2", activity=ConceptRef(concept_id="x_act")),
        },
        edges={
            "z1": Edge(id="z1", from_node="s", to_node="n1"),
            "z2": Edge(id="z2", from_node="s", to_node="n2"),
        },
        concepts={
            "s_act": BusinessConcept(
                id="s_act",
                kind="activity",
                display_label="start",
                validation_status="confirmed",
            ),
            "x_act": BusinessConcept(
                id="x_act",
                kind="activity",
                display_label="x",
                validation_status="confirmed",
            ),
        },
        start_node_id="s",
        end_node_ids=["n1", "n2"],
    )
    # observations: first position and second position, plus the two relations
    obs_start = _claim_obs(
        tools, "We start the process.", [_assertion("s.activity", "start the process")]
    )
    obs_x1 = _claim_obs(
        tools, "First we do X here.", [_assertion("x1.activity", "do X")]
    )
    obs_x2 = _claim_obs(
        tools, "Then we do X again.", [_assertion("x2.activity", "do X")]
    )
    obs_e1 = _claim_obs(
        tools,
        "After the start we do X.",
        [_assertion("e1.edge_exists", "After the start we do X")],
    )
    obs_e2 = _claim_obs(
        tools,
        "Later we do X again.",
        [_assertion("e2.edge_exists", "Later we do X again")],
    )
    tools.db.graph.nodes["s"].activity.evidence.append(
        EvidenceRef(observation_id=obs_start, quote="start the process")
    )
    # the shared activity concept cites BOTH X positions (ambiguous by activity)
    tools.db.graph.concepts["x_act"].mentions.append(
        EvidenceRef(observation_id=obs_x1, quote="do X")
    )
    tools.db.graph.concepts["x_act"].mentions.append(
        EvidenceRef(observation_id=obs_x2, quote="do X")
    )
    tools.db.graph.nodes["n1"].activity.evidence.append(
        EvidenceRef(observation_id=obs_x1, quote="do X")
    )
    tools.db.graph.nodes["n2"].activity.evidence.append(
        EvidenceRef(observation_id=obs_x2, quote="do X")
    )
    tools.db.graph.edges["z1"].evidence.append(
        EvidenceRef(observation_id=obs_e1, quote="After the start we do X")
    )
    tools.db.graph.edges["z2"].evidence.append(
        EvidenceRef(observation_id=obs_e2, quote="Later we do X again")
    )
    from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate

    res = evaluate(
        tools.db,
        truth,
        EvaluationSpec(),
        filter_,
        claims=claims,
        assertions=tools.assertion_ledger.assertions(),
    )
    # both X nodes are mapped through topology; start/ends correct
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0
    assert res.start_correct is True
    assert res.end_recall == 1.0
    assert res.end_precision == 1.0


# ---------------------------------------------------------------------------
# Model shape / cycles / no sentence facts
# ---------------------------------------------------------------------------


def test_cycles_are_valid():
    graph = BusinessProcessGraph(
        nodes={
            "a": Node(id="a", activity=ConceptRef(concept_id="x")),
            "b": Node(id="b", activity=ConceptRef(concept_id="y")),
        },
        edges={
            "e1": Edge(id="e1", from_node="a", to_node="b"),
            "e2": Edge(id="e2", from_node="b", to_node="a"),
        },
        concepts={
            "x": BusinessConcept(id="x", kind="activity", display_label="x"),
            "y": BusinessConcept(id="y", kind="activity", display_label="y"),
        },
    )
    assert graph.structure_errors() == []
    assert graph.is_valid


def test_no_sentence_based_facts():
    """Stakeholder knowledge is semantic structure: no StakeholderFact model,
    no authored business sentences anywhere in the scenario."""

    from tau2.domains.business_interview import scenario

    assert not hasattr(scenario, "StakeholderFact")
    from tau2.domains.business_interview import facts

    assert not hasattr(facts, "StakeholderFact")
    sc = get_scenario(SCENARIO)
    assert sc is not None
    # knowledge has no free-text business sentences
    for word in (
        "You record the quotation request",
        "You create the quotation using",
        "You check the customer",
    ):
        text = str(sc.knowledge.model_dump())
        assert word not in text, word
    # task prose (known_info) carries no business facts
    for task in get_tasks():
        assert not (
            getattr(task.user_scenario.instructions, "known_info", None) or ""
        ).strip()


def test_truth_and_agent_result_use_same_graph_class():
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert isinstance(sc.truth, BusinessProcessGraph)
    assert sc.truth.start_node_id == "r"
    assert set(sc.truth.end_node_ids) == {"sq", "me"}
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    assert isinstance(tools.db.graph, BusinessProcessGraph)


# ---------------------------------------------------------------------------
# Faithful full pass + failure modes
# ---------------------------------------------------------------------------


def test_valid_full_graph_passes():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.graph_valid is True
    assert res.structural_pass is True
    assert res.quality_pass is True
    assert res.evidence_pass is True
    assert res.glossary_pass is True
    assert res.glossary_validation_errors == []
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0
    assert res.edge_recall == 1.0
    assert res.edge_precision == 1.0
    assert res.start_correct is True
    assert res.end_recall == 1.0
    assert res.end_precision == 1.0
    assert res.concept_correctness == 1.0


def test_missing_node_lowers_node_recall():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    del tools.db.graph.nodes["f"]
    tools.db.graph.edges.pop("e6", None)
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.structural_pass is False


def test_fabricated_node_stays_unmapped():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _claim_obs(
        tools,
        "I like pizza on Fridays.",
        alignments=[_unknown_alignment_for("tc_activity_receive_request", "pizza")],
    )
    tools.create_concept(
        "pizza_act", "activity", "eat pizza", evidence=[_ev(pizza, "pizza")]
    )
    tools.add_node("fab", activity="pizza_act", evidence=[_ev(pizza, "pizza")])
    tools.mark_concept_unknown("pizza_act", evidence=[_ev(pizza, "pizza")])
    res = _eval(tools)
    assert res.node_precision < 1.0
    assert res.fabricated_node_count >= 1
    assert res.structural_pass is False


def test_wrong_actor_drops_actor_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["b"].actor = ConceptRef(concept_id="manager", confidence=1.0)
    res = _eval(tools)
    assert res.actor_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_read_drops_read_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["c"].reads = [ConceptRef(concept_id="quote", confidence=1.0)]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.structural_pass is False


def test_missing_visible_property_lowers_score():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["b"].system = None
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_undeclared_or_wrong_endpoints_fail():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.end_node_ids = ["f"]
    res = _eval(tools)
    assert res.end_recall < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Hidden properties stay prior
# ---------------------------------------------------------------------------


def test_hidden_unset_still_correct():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
    assert res.quality_pass is True


def test_hidden_assertions_fail_regardless_of_provenance():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I send the quotation to the customer by email.",
        [_assertion("sq.activity", "send the quotation")],
        alignments=[_confirm_alignment_for("tc_quote", "quotation")],
    )
    tools.create_concept(
        "quote_dup", "data", "quotation", evidence=[_ev(oid, "quotation")]
    )
    tools.confirm_concept("quote_dup", evidence=[_ev(oid, "quotation")])
    tools.db.graph.nodes["e"].reads = [
        ConceptRef(
            concept_id="quote_dup", confidence=1.0, evidence=[_evr(oid, "quotation")]
        )
    ]
    tools.db.graph.nodes["e"].writes = [
        ConceptRef(
            concept_id="quote_dup", confidence=1.0, evidence=[_evr(oid, "quotation")]
        )
    ]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Provenance-only binding (labels/mentions never determine correctness)
# ---------------------------------------------------------------------------


def test_arbitrary_labels_bind_through_provenance():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [_assertion("cq.writes.tc_quote", "quotation")],
        alignments=[_confirm_alignment_for("tc_quote", "quotation")],
    )
    tools.create_concept("whatever", "data", "zzz", evidence=[_ev(oid, "quotation")])
    tools.confirm_concept("whatever", evidence=[_ev(oid, "quotation")])
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="whatever", confidence=1.0, evidence=[_evr(oid, "quotation")]
        )
    ]
    res = _eval(tools)
    assert res.write_correctness == 1.0
    assert res.concept_correctness == 1.0
    assert res.structural_pass is True


def test_paraphrasing_labels_descriptions_does_not_change_score():
    a = _tools()
    _build(a)
    b = _tools()
    _build(b)
    assert b.db.graph is not None
    for concept in b.db.graph.concepts.values():
        concept.display_label = "renamed-" + concept.id
        concept.description = "completely different description"
        concept.mentions = []
    ra = _eval(a)
    rb = _eval(b)
    for field in (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "activity_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "rationale_correctness",
        "condition_correctness",
        "concept_correctness",
        "structural_pass",
        "quality_pass",
    ):
        assert getattr(ra, field) == getattr(rb, field), field


def test_identical_text_different_provenance_grounds_differently():
    text = "I handle the paperwork for the deal."
    ok = _tools()
    _build(ok)
    assert ok.db.graph is not None
    oid = _claim_obs(ok, text, [_assertion("cq.writes.tc_quote", "paperwork")],
                     alignments=[_confirm_alignment_for("tc_quote", "paperwork")])
    ok.create_concept("paper", "data", "paperwork", evidence=[_ev(oid, "paperwork")])
    ok.confirm_concept("paper", evidence=[_ev(oid, "paperwork")])
    ok.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="paper", confidence=1.0, evidence=[_evr(oid, "paperwork")]
        )
    ]
    res_ok = _eval(ok)
    assert res_ok.write_correctness == 1.0

    bad = _tools()
    _build(bad)
    assert bad.db.graph is not None
    oid2 = _claim_obs(bad, text, [_assertion("cq.activity", "handle the paperwork")],
                      alignments=[_confirm_alignment_for("tc_quote", "paperwork")])
    bad.create_concept("paper2", "data", "paperwork", evidence=[_ev(oid2, "paperwork")])
    bad.confirm_concept("paper2", evidence=[_ev(oid2, "paperwork")])
    bad.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="paper2", confidence=1.0, evidence=[_evr(oid2, "paperwork")]
        )
    ]
    res_bad = _eval(bad)
    assert res_bad.write_correctness < 1.0
    assert res_bad.unsupported_ref_count >= 1


# ---------------------------------------------------------------------------
# Span correspondence (containment) without cross-credit
# ---------------------------------------------------------------------------


def test_span_containment_grounds_without_cross_credit():
    """An evidence span that CONTAINS exactly one assertion span grounds that
    claim; a span containing assertions of several claims is ambiguous and
    grounds nothing (no cross-credit)."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # one utterance asserting the check activity only (single claim)
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.activity", "check the customer's information")],
        alignments=[
            _confirm_alignment_for(
                "tc_activity_check_customer", "check the customer's information in the CRM"
            )
        ],
    )
    # evidence span BROADER than the assertion span (contains it exactly once)
    tools.create_concept(
        "broad",
        "activity",
        "check step",
        evidence=[_ev(oid, "check the customer's information in the CRM")],
    )
    tools.confirm_concept(
        "broad", evidence=[_ev(oid, "check the customer's information in the CRM")]
    )
    tools.db.graph.nodes["b"].activity = ConceptRef(
        concept_id="broad",
        confidence=1.0,
        evidence=[_evr(oid, "check the customer's information in the CRM")],
    )
    res = _eval(tools)
    assert res.activity_correctness == 1.0

    # a reads ref whose broad span contains assertions of BOTH expected
    # claims (customer and pricing) is ambiguous and grounds neither
    tools2 = _tools()
    _build(tools2)
    assert tools2.db.graph is not None
    oid2 = _claim_obs(
        tools2,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            _assertion("cq.reads.tc_customer", "customer"),
            _assertion("cq.reads.tc_pricing", "pricing information"),
        ],
        alignments=[
            _confirm_alignment_for(
                "tc_customer", "using the customer and pricing information"
            )
        ],
    )
    tools2.create_concept(
        "broad2",
        "data",
        "customer pricing",
        evidence=[_ev(oid2, "using the customer and pricing information")],
    )
    tools2.confirm_concept(
        "broad2", evidence=[_ev(oid2, "using the customer and pricing information")]
    )
    tools2.db.graph.nodes["c"].reads = [
        ConceptRef(
            concept_id="broad2",
            confidence=1.0,
            evidence=[_evr(oid2, "using the customer and pricing information")],
        )
    ]
    res2 = _eval(tools2)
    assert res2.read_correctness < 1.0
    assert res2.ambiguous_evidence_ref_count >= 1


def test_evidence_contained_in_assertion_span_grounds():
    """A NARROWER evidence span inside a single assertion span grounds it."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.activity", "check the customer's information")],
        alignments=[_confirm_alignment_for("tc_activity_check_customer", "check")],
    )
    tools.create_concept("narrow", "activity", "check", evidence=[_ev(oid, "check")])
    tools.confirm_concept("narrow", evidence=[_ev(oid, "check")])
    tools.db.graph.nodes["b"].activity = ConceptRef(
        concept_id="narrow", confidence=1.0, evidence=[_evr(oid, "check")]
    )
    res = _eval(tools)
    assert res.activity_correctness == 1.0


def test_invalid_evidence_span_rejected():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = next(o.id for o in tools.db.observations if o.turn == 3)  # cq obs
    tools.create_concept("bogus", "data", "quotation")
    with pytest.raises(ValueError):
        tools.confirm_concept("bogus", evidence=[_ev(oid, "not in the text")])
    tools.db.graph.concepts["bogus"].validation_evidence.append(
        EvidenceRef(observation_id=oid, quote="not in the text", occurrence=0)
    )
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="bogus", confidence=1.0, evidence=[_evr(oid, "not in the text")]
        )
    ]
    res = _eval(tools)
    assert res.invalid_evidence_ref_count >= 1
    assert res.evidence_pass is False
    assert res.write_correctness < 1.0


def test_invalid_assertion_quote_rejected_at_ingestion():
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment
    from tau2.domains.business_interview.user_simulator import StakeholderUserSimulator

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    task = next(t for t in get_tasks() if t.id == SCENARIO)
    StakeholderUserSimulator(llm="dummy", task=task, environment=env, instructions="x")
    assert env.assertion_ledger is not None
    assert env.assertion_ledger.catalog is not None
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer.",
                stakeholder_assertions=[_assertion("cc.system", "CRM")],
            )
        )
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_assertions=[_assertion("cc.system", "CRM", occurrence=3)],
            )
        )
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_assertions=[_assertion("nope.claim", "CRM")],
            )
        )


def test_second_occurrence_span_matching():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "the quotation is one thing, and the quotation is another"
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("cq.writes.tc_quote", "quotation", occurrence=0),
            _assertion("cq.activity", "quotation", occurrence=1),
        ],
        alignments=[
            _confirm_alignment_for("tc_activity_create_quotation", "quotation", 1)
        ],
    )
    tools.create_concept(
        "occ_act",
        "activity",
        "create the quotation",
        evidence=[_ev(oid, "quotation", 1)],
    )
    tools.confirm_concept("occ_act", evidence=[_ev(oid, "quotation", 1)])
    tools.db.graph.nodes["c"].activity = ConceptRef(
        concept_id="occ_act", confidence=1.0, evidence=[_evr(oid, "quotation", 1)]
    )
    res = _eval(tools)
    assert res.activity_correctness == 1.0


def test_multi_fact_utterance_cannot_cross_credit_unrelated_concepts():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "I check the customer in the CRM and prepare the quotation."
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("cc.system", "CRM"),
            _assertion("cq.activity", "prepare the quotation"),
        ],
        alignments=[_confirm_alignment_for("tc_system_crm", "CRM")],
    )
    tools.create_concept("crm_data", "data", "CRM data", evidence=[_ev(oid, "CRM")])
    tools.confirm_concept("crm_data", evidence=[_ev(oid, "CRM")])
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(concept_id="crm_data", confidence=1.0, evidence=[_evr(oid, "CRM")])
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_ref_count >= 1
    assert res.concept_correctness == 0.0


def test_unrelated_observation_cannot_support_a_claim():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    tools.create_concept(
        "pizza_quote", "data", "quotation", evidence=[_ev(pizza, "pizza")]
    )
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="pizza_quote", confidence=1.0, evidence=[_evr(pizza, "pizza")]
        )
    ]
    # a hypothesis the stakeholder never asserted cannot be confirmed
    with pytest.raises(ValueError):
        tools.confirm_concept("pizza_quote", evidence=[_ev(pizza, "pizza")])
    # ... nor marked unknown (no private unknown dialogue event exists)
    with pytest.raises(ValueError):
        tools.mark_concept_unknown("pizza_quote", evidence=[_ev(pizza, "pizza")])
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_ref_count >= 1
    assert res.evidence_pass is True
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Concept identity (reuse / split / merge / kinds)
# ---------------------------------------------------------------------------


def test_reuse_passes():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.concept_correctness == 1.0
    assert res.structural_pass is True


def test_split_identity_fails_until_merged():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.actor", "I")],
        alignments=[_confirm_alignment_for("tc_actor_sales", "I")],
    )
    tools.create_concept("sales_dup", "actor", "sales", evidence=[_ev(oid, "I")])
    tools.confirm_concept("sales_dup", evidence=[_ev(oid, "I")])
    tools.db.graph.nodes["b"].actor = ConceptRef(
        concept_id="sales_dup", confidence=1.0, evidence=[_evr(oid, "I")]
    )
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    tools.merge_concepts("sales", ["sales_dup"])
    res2 = _eval(tools)
    assert res2.concept_correctness == 1.0
    assert res2.structural_pass is True


def test_merge_incompatible_kinds_rejected():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "actor", "alpha")
    tools.create_concept("b", "data", "beta")
    with pytest.raises(ValueError):
        tools.merge_concepts("a", ["b"])


def test_merge_distinct_truth_concepts_fails():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            _assertion("cq.reads.tc_customer", "customer"),
            _assertion("cq.reads.tc_pricing", "pricing information"),
        ],
        alignments=[_confirm_alignment_for("tc_customer", "customer")],
    )
    tools.create_concept(
        "blend", "data", "customer pricing", evidence=[_ev(oid, "customer")]
    )
    tools.add_concept_mention("blend", [_ev(oid, "pricing information")])
    tools.confirm_concept("blend", evidence=[_ev(oid, "customer")])
    tools.db.graph.nodes["c"].reads = [
        ConceptRef(concept_id="blend", confidence=1.0, evidence=[_evr(oid, "customer")])
    ]
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


def test_incompatible_kind_binding_fails():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.actor", "I")],
        alignments=[_confirm_alignment_for("tc_actor_sales", "I")],
    )
    tools.create_concept("actor_data", "data", "some data", evidence=[_ev(oid, "I")])
    tools.confirm_concept("actor_data", evidence=[_ev(oid, "I")])
    tools.db.graph.nodes["b"].actor = ConceptRef(
        concept_id="actor_data", confidence=1.0, evidence=[_evr(oid, "I")]
    )
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


def test_condition_participates_in_identity():
    """Conditions join the identity validation: one condition concept bound to
    two distinct Truth conditions fails (merging distinct concepts)."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid3 = _claim_obs(
        tools,
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [_assertion("e3.condition", "over 1,000,000 yen")],
        alignments=[_confirm_alignment_for("tc_cond_over_1m", "over 1,000,000 yen")],
    )
    oid4 = _claim_obs(
        tools,
        "Quotations at or below 1,000,000 yen are sent directly to the customer.",
        [_assertion("e4.condition", "at or below 1,000,000 yen")],
    )
    tools.create_concept(
        "cond_blend", "condition", "amount", evidence=[_ev(oid3, "over 1,000,000 yen")]
    )
    tools.add_concept_mention("cond_blend", [_ev(oid4, "at or below 1,000,000 yen")])
    tools.confirm_concept("cond_blend", evidence=[_ev(oid3, "over 1,000,000 yen")])
    tools.db.graph.edges["e3"].condition = ConceptRef(
        concept_id="cond_blend",
        confidence=1.0,
        evidence=[_evr(oid3, "over 1,000,000 yen")],
    )
    tools.db.graph.edges["e4"].condition = ConceptRef(
        concept_id="cond_blend",
        confidence=1.0,
        evidence=[_evr(oid4, "at or below 1,000,000 yen")],
    )
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# ConceptKind enforcement (structural rules)
# ---------------------------------------------------------------------------


def test_concept_kind_enforcement_in_tools():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("act", "activity", "an activity")
    tools.create_concept("sales", "actor", "sales")
    tools.create_concept("data_c", "data", "some data")
    tools.create_concept("sys", "system", "a system")
    tools.create_concept("cond", "condition", "a condition")
    tools.create_concept("rat", "rationale", "a rationale")
    tools.add_node("n1", activity="act")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="sales")  # actor as activity
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", actor="data_c")  # data as actor
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", system="data_c")  # data as system
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", reads=["act"])  # activity as data
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", writes=["cond"])  # condition as data
    with pytest.raises(ValueError):
        tools.add_node(
            "n2", activity="act", necessity_rationale="sys"
        )  # system as rationale
    tools.add_node(
        "n2",
        activity="act",
        actor="sales",
        system="sys",
        reads=["data_c"],
        writes=["data_c"],
        necessity_rationale="rat",
    )
    with pytest.raises(ValueError):
        tools.add_edge("e1", "n1", "n2", condition="data_c")  # data as condition
    tools.add_edge("e1", "n1", "n2", condition="cond")


# ---------------------------------------------------------------------------
# Edge grounding
# ---------------------------------------------------------------------------


def test_edge_existence_needs_provenance():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(tools, "I like pizza on Fridays.")
    tools.add_edge("x1", "b", "c", evidence=[_ev(oid, "pizza")])
    res = _eval(tools)
    assert res.edge_precision < 1.0
    assert res.fabricated_edge_count >= 1
    assert res.structural_pass is False


def test_missing_edge_lowers_edge_recall():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    del tools.db.graph.edges["e5"]
    res = _eval(tools)
    assert res.edge_recall < 1.0
    assert res.structural_pass is False


def test_condition_grounded_and_wrong_condition_fails():
    good = _tools()
    _build(good)
    assert _eval(good).condition_correctness == 1.0

    bad = _tools()
    _build(bad)
    assert bad.db.graph is not None
    oid = _claim_obs(
        bad,
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [_assertion("e3.edge_exists", "over 1,000,000 yen")],
        alignments=[_confirm_alignment_for("tc_cond_over_1m", "over 1,000,000 yen")],
    )
    bad.create_concept(
        "wrong_cond",
        "condition",
        "month-end",
        evidence=[_ev(oid, "over 1,000,000 yen")],
    )
    bad.confirm_concept("wrong_cond", evidence=[_ev(oid, "over 1,000,000 yen")])
    bad.db.graph.edges["e3"].condition = ConceptRef(
        concept_id="wrong_cond",
        confidence=1.0,
        evidence=[_evr(oid, "over 1,000,000 yen")],
    )
    res2 = _eval(bad)
    assert res2.condition_correctness < 1.0
    assert res2.structural_pass is False


def test_condition_on_unconditional_edge_fails():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "After receiving the request, I check the customer information.",
        [_assertion("e1.edge_exists", "After receiving the request, I check")],
        alignments=[
            _confirm_alignment_for("tc_cond_over_1m", "After receiving the request")
        ],
    )
    tools.create_concept(
        "fake_cond",
        "condition",
        "sometimes",
        evidence=[_ev(oid, "After receiving the request")],
    )
    tools.confirm_concept(
        "fake_cond", evidence=[_ev(oid, "After receiving the request")]
    )
    tools.db.graph.edges["e1"].condition = ConceptRef(
        concept_id="fake_cond",
        confidence=1.0,
        evidence=[_evr(oid, "After receiving the request")],
    )
    res = _eval(tools)
    assert res.condition_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Glossary validation / completion / genuine confirmation
# ---------------------------------------------------------------------------


def test_hypothesized_referenced_concept_blocks_completion():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.concepts["customer"].validation_status = "hypothesized"
    with pytest.raises(ValueError):
        tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is False
    assert "customer" in res.referenced_hypothesized_concepts
    assert res.structural_pass is False


def test_unreferenced_hypothesized_concept_does_not_block():
    tools = _tools()
    _build(tools)
    tools.create_concept("unused", "data", "unused concept")
    tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is True


def test_confirm_concept_requires_stakeholder_assertion_evidence():
    """Confirmation must correspond to a private concept-alignment event — a
    mention from unrelated speech is not confirmation."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("c", "data", "thing")
    # evidence that does not correspond to any private dialogue event -> rejected
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    with pytest.raises(ValueError):
        tools.confirm_concept("c", evidence=[_ev(pizza, "pizza")])
    # unknown observation -> rejected
    with pytest.raises(ValueError):
        tools.confirm_concept("c", evidence=[_ev("obs_missing", "x")])


def test_ordinary_mention_cannot_confirm_concept():
    """Requirement 2/3 regression: an ordinary workflow mention ("quotation"
    in normal speech) creates NO concept-alignment event, so it cannot
    authorize confirm_concept."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("quote_c", "data", "quotation")
    # ordinary workflow speech carries claims but NO alignment events
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer information in the quoting "
        "system.",
        [_assertion("cq.writes.tc_quote", "quotation")],
    )
    with pytest.raises(ValueError):
        tools.confirm_concept("quote_c", evidence=[_ev(oid, "quotation")])
    # the same span DOES confirm once the stakeholder genuinely performs the
    # dialogue act (a private concept-alignment event is bound)
    oid2 = _claim_obs(
        tools,
        "Yes.",
        alignments=[_confirm_alignment_for("tc_quote", "Yes.")],
    )
    tools.confirm_concept("quote_c", evidence=[_ev(oid2, "Yes.")])
    assert tools.db.graph is not None
    assert tools.db.graph.concepts["quote_c"].validation_status == "confirmed"


def test_bulk_self_confirmation_rejected():
    """One evidence span cannot confirm several concepts."""
    tools = _tools()
    tools.start_inference("q")
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.system", "CRM")],
        alignments=[_confirm_alignment_for("tc_system_crm", "CRM")],
    )
    tools.create_concept("a", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.create_concept("b", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.confirm_concept("a", evidence=[_ev(oid, "CRM")])
    with pytest.raises(ValueError):
        tools.confirm_concept("b", evidence=[_ev(oid, "CRM")])


def test_unknown_and_disputed_require_evidence_and_can_complete():
    """unknown/disputed need the appropriate private concept-alignment events
    and resolve completion when referenced."""
    # unknown: a private alignment event with act=unknown
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    dont_know = _claim_obs(
        tools,
        "I do not know the reason for that.",
        alignments=[_unknown_alignment_for("tc_pricing", "do not know")],
    )
    tools.mark_concept_unknown(
        "pricing",
        evidence=[_ev(dont_know, "do not know")],
    )
    tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is True
    assert res.structural_pass is True

    # disputed: private dispute events from >= 2 distinct observations
    tools2 = _tools()
    _build(tools2)
    assert tools2.db.graph is not None
    o1 = _claim_obs(
        tools2,
        "Actually, they are not the same thing.",
        alignments=[_dispute_alignment_for("tc_customer", "not the same thing")],
    )
    o2 = _claim_obs(
        tools2,
        "I keep telling you, those are different.",
        alignments=[_dispute_alignment_for("tc_customer", "different")],
    )
    tools2.mark_concept_disputed(
        "customer",
        evidence=[_ev(o1, "not the same thing"), _ev(o2, "different")],
    )
    tools2.finish_interview()
    res2 = _eval(tools2)
    assert res2.glossary_pass is True
    assert res2.structural_pass is True


def test_confirmation_evidence_must_match_concepts_claims():
    """The evaluator rejects a confirmed concept whose validation evidence does
    not correspond to an assertion of the concept's own claims."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # confirm `sales` with evidence of the manager's claim -> invalid
    tools.db.graph.concepts["sales"].validation_evidence = [
        EvidenceRef(
            observation_id=next(o.id for o in tools.db.observations if o.turn == 4),
            quote="manager",
            occurrence=0,
        )
    ]
    res = _eval(tools)
    assert res.glossary_pass is False
    assert any("sales" in e for e in res.glossary_validation_errors)


def test_partially_confirmed_can_complete():
    """partially_confirmed must be backed by a private concept-alignment
    event with act=partial."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "Partly — only for the pricing information.",
        alignments=[
            {
                "truth_concept_id": "tc_pricing",
                "quote": "Partly — only for the pricing information.",
                "act": "partial",
            }
        ],
    )
    tools.db.graph.concepts["pricing"].validation_evidence = [_evr(oid, "Partly — only for the pricing information.")]
    tools.db.graph.concepts["pricing"].validation_status = "partially_confirmed"
    tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is True
    # a plain confirm event can NOT authorize partially_confirmed
    tools2 = _tools()
    _build(tools2)
    assert tools2.db.graph is not None
    oid2 = _claim_obs(
        tools2,
        "Yes.",
        alignments=[_confirm_alignment_for("tc_pricing", "Yes.")],
    )
    tools2.db.graph.concepts["pricing"].validation_evidence = [_evr(oid2, "Yes.")]
    tools2.db.graph.concepts["pricing"].validation_status = "partially_confirmed"
    res2 = _eval(tools2)
    assert res2.glossary_pass is False


def test_mention_is_not_terminology():
    """Mentions never establish terminology; an ordinary authentic mention
    cannot authorize a terminology agreement — only a private
    terminology-confirmation event (same bound concept + same proposed term +
    cited span) can."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [_assertion("cq.writes.tc_quote", "quotation")],
    )
    # a mention of a NEW term on the quote concept does not change the score
    tools.add_concept_mention("quote", [_ev(oid, "quotation")])
    res_before = _eval(tools)
    assert res_before.structural_pass is True
    # regression: an ordinary mention of "quotation" cannot authorize an
    # agreement for "the offer document" (no terminology event exists)
    with pytest.raises(ValueError):
        tools.record_terminology_agreement(
            "quote", "the offer document", evidence=[_ev(oid, "quotation")]
        )
    # a genuine private terminology-confirmation event does authorize it
    oid_agree = _claim_obs(
        tools,
        "Yes, the offer document is fine.",
        terminology=[
            {
                "truth_concept_id": "tc_quote",
                "proposed_term": "the offer document",
                "quote": "the offer document is fine",
            }
        ],
    )
    tools.record_terminology_agreement(
        "quote", "the offer document", evidence=[_ev(oid_agree, "the offer document is fine")]
    )
    assert len(tools.db.graph.terminology_agreements) == 1
    agreement = tools.db.graph.terminology_agreements[0]
    assert agreement.concept_id == "quote"
    assert agreement.term == "the offer document"
    res_after = _eval(tools)
    assert res_after.structural_pass is True
    assert res_after.concept_correctness == 1.0
    assert res_after.glossary_pass is True


def test_terminology_agreement_requires_evidence():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("c", "data", "thing")
    with pytest.raises(ValueError):
        tools.record_terminology_agreement("c", "term", evidence=[])
    # an agreement whose proposed term does not match any private
    # terminology-confirmation event is rejected
    oid = _claim_obs(
        tools,
        "Yes.",
        terminology=[
            {
                "truth_concept_id": "tc_quote",
                "proposed_term": "the offer document",
                "quote": "Yes.",
            }
        ],
    )
    with pytest.raises(ValueError):
        tools.record_terminology_agreement(
            "c", "something else", evidence=[_ev(oid, "Yes.")]
        )


# ---------------------------------------------------------------------------
# Evidence hygiene / authenticity
# ---------------------------------------------------------------------------


def test_nonexistent_observation_ref_rejected():
    tools = _tools()
    tools.start_inference("Q")
    with pytest.raises(ValueError):
        tools.add_node("n1", activity="x", evidence=[_ev("does_not_exist", "y")])


def test_fabricated_observation_source_fails_authenticity():
    tools = _tools()
    _build(tools)
    from tau2.domains.business_interview.graph import Observation

    fake = Observation(
        id="obs_fake", source_id="stakeholder", text="made up", order=999, turn=999
    )
    tools.db.observations.append(fake)
    assert tools.db.graph is not None
    assert tools.db.graph.nodes["b"].actor is not None
    tools.db.graph.nodes["b"].actor.evidence.append(_evr("obs_fake", "made up"))
    res = _eval(tools)
    assert res.invalid_observation_source_count >= 1
    assert res.provenance_authenticity_pass is False
    assert res.evidence_pass is False
    assert res.quality_pass is False


def test_zero_evidence_refs_fail_evidence_gate():
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    obs = _claim_obs(
        tools,
        "We start things off.",
        alignments=[_unknown_alignment_for("tc_activity_receive_request", "start")],
    )
    for cid, kind, label in [
        ("act", "activity", "receive"),
        ("sales", "actor", "sales"),
        ("req", "data", "request"),
    ]:
        tools.create_concept(cid, kind, label)
        tools.mark_concept_unknown(cid, evidence=[_ev(obs, "start")])
    tools.add_node("a", activity="act", actor="sales", writes=["req"])
    tools.add_edge("e1", "a", "a", evidence=[])
    tools.set_graph_endpoints(start_node_id="a", end_node_ids=["a"])
    res = _eval(tools)
    assert res.evidence_pass is False


# ---------------------------------------------------------------------------
# Private-id leakage
# ---------------------------------------------------------------------------


def test_private_ids_absent_from_agent_visible_state():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    listing = tools.list_concepts()
    for forbidden in ("cq.", "tc_", "claim", "edge_exists"):
        assert forbidden not in listing, forbidden
    dumped = str(tools.db.model_dump(mode="json"))
    sc = get_scenario(SCENARIO)
    assert sc is not None
    for cid in list(sc.claims)[:5]:
        assert cid not in dumped
    assert "assertion_ledger" not in InterviewDB.model_fields


def test_user_message_serialization_excludes_assertions():
    msg = UserMessage(
        role="user",
        content="hello",
        stakeholder_assertions=[_assertion("cq.writes.tc_quote", "quotation")],
        stakeholder_alignments=[
            {"truth_concept_id": "tc_quote", "quote": "yes", "act": "confirm"}
        ],
        stakeholder_terminology=[
            {
                "truth_concept_id": "tc_quote",
                "proposed_term": "offer",
                "quote": "yes",
            }
        ],
    )
    dumped = msg.model_dump(mode="json")
    assert "stakeholder_assertions" not in dumped
    assert "stakeholder_alignments" not in dumped
    assert "stakeholder_terminology" not in dumped
    assert "cq.writes.tc_quote" not in str(dumped)
    assert "cq.writes.tc_quote" not in str(msg)
    assert "cq.writes.tc_quote" not in repr(msg)
    assert "truth_concept_id" not in str(dumped)
    assert "truth_concept_id" not in repr(msg)


def test_ledger_binds_assertions_per_turn():
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    env.on_message(UserMessage(role="user", content="plain statement"))
    assert env.assertion_ledger is not None
    assert env.assertion_ledger.assertions() == {}
    env.on_message(
        UserMessage(
            role="user",
            content="I check the customer in the CRM.",
            stakeholder_assertions=[
                {"claim_id": "cc.system", "quote": "CRM", "occurrence": 0}
            ],
        )
    )
    assert 1 in env.assertion_ledger.assertions()


# ---------------------------------------------------------------------------
# Old semantic matcher machinery is gone
# ---------------------------------------------------------------------------


def test_semantic_matcher_machinery_is_gone():
    import pathlib

    from tau2.domains.business_interview import evaluation

    assert not hasattr(evaluation, "_predicate_ok")
    assert not hasattr(evaluation, "_necessity_value_ok")
    assert not hasattr(evaluation, "_attribute_match")
    assert not hasattr(evaluation, "_node_overlap")
    assert not hasattr(evaluation, "_STOPWORDS")
    assert not hasattr(evaluation, "_term_covers")
    assert not hasattr(evaluation, "_supported_claims")
    assert not hasattr(evaluation, "TruthNodeSpec")
    domain_dir = pathlib.Path(evaluation.__file__).parent
    for src in domain_dir.glob("*.py"):
        text = src.read_text(encoding="utf-8")
        for token in (
            "_predicate_ok",
            "_necessity_value_ok",
            "_attribute_match",
            "norm_role",
            "norm_system",
        ):
            assert token not in text, f"{src.name} still contains {token!r}"
    assert not (domain_dir / "dag.py").exists()
    assert not (domain_dir / "concepts.py").exists()
    assert not (domain_dir / "aliases.py").exists()


def test_evaluation_spec_is_empty():
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert sc.spec.model_dump() == {}


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


def test_finish_terminates_episode():
    """A successful finish_interview() terminates the episode immediately
    (EPISODE_COMPLETE), distinct from max_steps truncation."""
    from tau2.data_model.simulation import TerminationReason
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment
    from tau2.orchestrator.orchestrator import Orchestrator

    class StubUser:
        def get_init_state(self, message_history=None):
            return {}

        def generate_next_message(self, message, state):
            return UserMessage(role="user", content="ok"), state

        def set_seed(self, seed):
            pass

        def stop(self, message, state):
            pass

    class StubAgent:
        def __init__(self):
            self.called = 0

        def get_init_state(self, message_history=None):
            return {}

        def generate_next_message(self, message, state):
            self.called += 1
            return (
                AssistantMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="finish_interview",
                            arguments={"summary": "done"},
                            requestor="assistant",
                        )
                    ],
                ),
                state,
            )

        def is_stop(self, message):
            return False

        def set_seed(self, seed):
            pass

        def stop(self, message, state):
            pass

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    # pre-build a valid finished graph in the environment's tools
    assert env.tools is not None
    assert env.tools is not None
    _build(env.tools)  # type: ignore[arg-type]
    # reset completion so the stub's finish call triggers it
    assert env.tools.db is not None
    env.tools.db.interview_complete = False

    task = next(t for t in get_tasks() if t.id == SCENARIO)
    orch = Orchestrator(
        domain="business_interview",
        agent=StubAgent(),  # type: ignore[arg-type]
        user=StubUser(),  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=100,
        max_errors=10,
        seed=1,
        solo_mode=False,
        simulation_id="termination-test",
    )
    result = orch.run()
    assert env.episode_complete() is True
    assert result.termination_reason == TerminationReason.EPISODE_COMPLETE
    assert result.termination_reason != TerminationReason.MAX_STEPS


def test_episode_complete_reflects_finish():
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    assert env.episode_complete() is False
    assert env.tools is not None
    _build(env.tools)  # type: ignore[arg-type]
    assert env.episode_complete() is True


# ---------------------------------------------------------------------------
# Observation capture UX (stable ids)
# ---------------------------------------------------------------------------


def test_list_stakeholder_messages_returns_stable_ids():
    tools = _tools()
    _ingest(tools, "assistant", "Hello.")
    _ingest(tools, "user", "Sure, that's fine.")
    _ingest(tools, "assistant", "How does it start?")
    _ingest(tools, "user", "The process starts when a customer sends a request.")
    listing = tools.list_stakeholder_messages()
    lines = [ln for ln in listing.splitlines()]
    assert lines[0].startswith("sm_1:")
    assert lines[1].startswith("sm_2:")
    assert "turn" not in listing


def test_observe_message_by_id_creates_correct_observation():
    tools = _tools()
    _ingest(tools, "user", "First statement.")
    _ingest(tools, "user", "Second statement.")
    oid = tools.observe_message("sm_2")
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "Second statement."
    assert obs.source_id == "stakeholder"
    assert tools.db.messages[obs.turn]["content"] == "Second statement."
    assert tools.observe_message("sm_2") == oid


def test_observation_capture_survives_set_state_replay():
    from tau2.data_model.message import (
        AssistantMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.domains.business_interview.environment import get_environment

    env = get_environment()
    traj = []

    def push(msg):
        traj.append(msg)
        env.on_message(msg)
        return msg

    def observe_and_record(name, args):
        cid = len(traj)
        tc = ToolCall(id=f"c{cid}", name=name, arguments=args, requestor="assistant")
        push(AssistantMessage(role="assistant", tool_calls=[tc]))
        assert env.tools is not None
        res = getattr(env.tools, name)(**args)
        push(ToolMessage(role="tool", id=f"c{cid}", content=res, requestor="assistant"))
        return res

    push(AssistantMessage(role="assistant", content="Hello."))
    push(UserMessage(role="user", content="First stakeholder statement."))
    oid1 = observe_and_record("observe_message", {"message_id": "sm_1"})
    push(UserMessage(role="user", content="Second stakeholder statement."))
    oid2 = observe_and_record("observe_message", {"message_id": "sm_2"})
    push(AssistantMessage(role="assistant", content="done"))

    replay = get_environment()
    replay.set_state(
        initialization_data=None,
        initialization_actions=None,
        message_history=list(traj),
        strict=False,
    )
    assert replay.tools is not None
    listing = replay.tools.list_stakeholder_messages()  # type: ignore[attr-defined]
    assert listing.splitlines()[0].startswith("sm_1:")
    assert "sm_2:" in listing
    assert replay.tools.db is not None
    assert len(replay.tools.db.observations) == 2
    assert oid1 in {o.id for o in replay.tools.db.observations}
    assert oid2 in {o.id for o in replay.tools.db.observations}


# ---------------------------------------------------------------------------
# Tool-level validation / refinement
# ---------------------------------------------------------------------------


def test_add_node_rejects_unknown_concept_id():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("known", "activity", "known thing")
    tools.add_node("n1", activity="known")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="unknown_concept")
    with pytest.raises(ValueError):
        tools.add_node("n3", activity="known", reads=["unknown_concept"])


def test_create_concept_rejects_bad_kind():
    tools = _tools()
    tools.start_inference("q")
    with pytest.raises(ValueError):
        tools.create_concept("x", "widget", "x")


def test_merge_concepts_repoints_refs_and_folds_mentions():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "data", "alpha")
    tools.create_concept("b", "data", "beta")
    tools.create_concept("act", "activity", "first step")
    tools.add_node("n1", activity="act", reads=["a"], writes=["b"])
    tools.merge_concepts("a", ["b"])
    assert tools.db.graph is not None
    assert "b" not in tools.db.graph.concepts
    assert tools.db.graph.nodes["n1"].reads[0].concept_id == "a"
    assert tools.db.graph.nodes["n1"].writes[0].concept_id == "a"


def test_remove_node_after_decomposition_leaves_no_dangling_edge():
    tools = _tools()
    tools.start_inference("q")
    for cid, kind, label in (
        ("act1", "activity", "coarse"),
        ("act2", "activity", "fine"),
    ):
        tools.create_concept(cid, kind, label)
    tools.add_node("coarse", activity="act1")
    tools.add_node("fine_a", activity="act2")
    tools.add_edge("e1", "coarse", "fine_a", evidence=[])
    tools.remove_node("coarse")
    assert tools.db.graph is not None
    assert "coarse" not in tools.db.graph.nodes
    assert "e1" not in tools.db.graph.edges
    assert tools.db.graph.is_valid


def test_validate_graph_accepts_cycles():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("act", "activity", "loop step")
    tools.add_node("a", activity="act")
    tools.add_node("b", activity="act")
    tools.add_edge("e1", "a", "b", evidence=[])
    tools.add_edge("e2", "b", "a", evidence=[])
    out = tools.validate_graph()
    assert "validation errors" not in out.lower()
    assert "valid" in out.lower()


def test_finish_requires_endpoints():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.start_node_id = None
    with pytest.raises(ValueError):
        tools.finish_interview()
    tools.db.graph.start_node_id = "a"
    tools.db.graph.end_node_ids = []
    with pytest.raises(ValueError):
        tools.finish_interview()


# ---------------------------------------------------------------------------
# EN / JA / lab equivalence
# ---------------------------------------------------------------------------


def test_en_ja_equivalent():
    en = _tools()
    _build(en, ja=False)
    ja = _tools()
    _build(ja, ja=True)
    ren = _eval(en, SCENARIO)
    rja = _eval(ja, JA_SCENARIO)
    for f in (
        "node_recall",
        "node_precision",
        "edge_recall",
        "edge_precision",
        "start_correct",
        "end_recall",
        "end_precision",
        "activity_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "rationale_correctness",
        "condition_correctness",
        "concept_correctness",
        "structural_pass",
        "quality_pass",
    ):
        assert getattr(ren, f) == getattr(rja, f), f


def test_lab_same_mechanism_full_pass():
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.structural_pass is True
    assert res.quality_pass is True
    assert res.node_recall == 1.0
    assert res.start_correct is True
    assert res.concept_correctness == 1.0


def test_lab_hidden_derived_artifact_assertion_fails():
    tools = _tools()
    _build_lab(tools)
    assert tools.db.graph is not None
    cant_say = _claim_obs(
        tools,
        "I cannot say.",
        alignments=[_unknown_alignment_for("tc_sample", "cannot say")],
    )
    tools.create_concept("seasoned", "data", "seasoned chamber")
    tools.mark_concept_unknown("seasoned", evidence=[_ev(cant_say, "cannot say")])
    tools.db.graph.nodes["n2"].writes = [
        ConceptRef(concept_id="seasoned", confidence=1.0)
    ]
    res = _eval(tools, LAB_SCENARIO)
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# End-to-end EnvironmentEvaluator reward
# ---------------------------------------------------------------------------


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert [t.id for t in tasks] == ALL_TASK_IDS
    assert set(get_tasks_split()["base"]) == set(ALL_TASK_IDS)
    assert set(get_tasks_split()["base_en"]) == {SCENARIO}
    assert set(get_tasks_split()["base_ja"]) == {JA_SCENARIO}


def test_task_instructions_reference_knowledge_not_sentences():
    for task in get_tasks():
        ins = task.user_scenario.instructions
        assert not (getattr(ins, "known_info", None) or "").strip(), task.id
        ti = (getattr(ins, "task_instructions", None) or "").lower()
        if task.id != "quotation_workflow_1_ja":
            assert "your knowledge" in ti, task.id
        assert "record the quotation request" not in ti


def test_agent_policy_requires_glossary_discipline():
    p = " ".join(BUSINESS_INTERVIEW_POLICY_PATH.read_text().split()).lower()
    assert "glossary" in p
    assert "hypothesized" in p
    assert "confirm_concept" in p
    assert "mention" in p
    assert "terminology" in p
    assert "exact substring" in p
    assert "cycles" in p


def test_policy_does_not_hardcode_domain_terms():
    p = " ".join(BUSINESS_INTERVIEW_POLICY_PATH.read_text().split()).lower()
    for banned in ("quotation", "crm", "tc_"):
        assert banned not in p, f"policy hard-codes domain term: {banned}"


def test_evaluator_rewards_full_reconstruction():
    """The full tau2 EnvironmentEvaluator path (replay + env assertions +
    domain diagnostics) rewards a faithful trajectory carrying private
    assertions."""
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    traj = _reference_trajectory()
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=traj,
        solo_mode=False,
        env_kwargs={},
    )
    assert reward_info is not None
    checks = {
        c.env_assertion.func_name: c.met for c in (reward_info.env_assertions or [])
    }
    assert checks == {
        "assert_finish_interview": True,
        "assert_graph_reconstructed": True,
        "assert_necessity_handled": True,
        "assert_evidence_backed": True,
    }
    diag = (reward_info.info or {})["diagnostics"]
    assert diag["structural_pass"] is True
    assert diag["quality_pass"] is True
    assert diag["glossary_pass"] is True


# ---------------------------------------------------------------------------
# Visibility-safe knowledge projection (requirement 1)
# ---------------------------------------------------------------------------


def test_lab_knowledge_is_a_physical_projection_no_hidden_concepts():
    """StakeholderKnowledge must be a physical projection of the Truth:
    hidden read/write concepts and their labels are ABSENT from the knowledge
    and from the rendered stakeholder prompt."""
    from tau2.domains.business_interview.scenario import lab_sample_truth
    from tau2.domains.business_interview.user_simulator import (
        StakeholderUserSimulator,
    )

    sc = get_scenario(LAB_SCENARIO)
    assert sc is not None
    knowledge = sc.knowledge
    visible_concepts = {
        c.concept_id for c in sc.claims.values() if c.concept_id is not None
    }
    # hidden data concepts: writes/reads artifacts the lab tech cannot see
    for hidden in (
        "tc_accessioned_sample",
        "tc_seasoned_chamber",
        "tc_conditioned_sample",
        "tc_batch_approval",
    ):
        assert hidden not in knowledge.concept_views, hidden
        assert hidden not in visible_concepts, hidden
    # hidden labels never appear in the views either
    for word in ("accessioned sample", "seasoned chamber", "conditioned sample"):
        assert word not in knowledge.concept_views.values(), word
    # visible concepts stay
    assert "tc_sample" in knowledge.concept_views
    assert "tc_system_chamber" in knowledge.concept_views
    # the knowledge block rendered for the LLM contains no hidden concept
    env = get_environment()
    task = next(t for t in get_tasks() if t.id == LAB_SCENARIO)
    sim = StakeholderUserSimulator(
        llm="dummy", task=task, environment=env, instructions="x"
    )
    block = sim._knowledge_block()  # noqa: SLF001 - test-only
    for hidden in ("tc_accessioned_sample", "tc_seasoned_chamber"):
        assert hidden not in block, hidden
    assert "accessioned sample" not in block
    assert "seasoned chamber" not in block
    assert "tc_sample" in block
    # the view list references only visible concepts
    assert "tc_batch_approval" not in block


def test_quotation_knowledge_contexts_use_only_visible_relations():
    """Contextual knowledge never exposes hidden relations: every incoming
    edge in a context is a visible edge of the stakeholder."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    visible_edges = set(sc.stakeholder.visible_edge_ids)
    for nid, ctx in sc.knowledge.contextual_knowledge.items():
        for eid in ctx.incoming_edge_ids:
            assert eid in visible_edges, (nid, eid)
        assert nid in sc.stakeholder.visible_node_ids, nid
    # the full Truth graph itself never leaks into the knowledge
    for nid in sc.truth.nodes:
        if nid not in sc.stakeholder.visible_node_ids:
            assert nid not in sc.knowledge.contextual_knowledge


def test_lab_hidden_assertions_never_enter_visible_claim_ids():
    sc = get_scenario(LAB_SCENARIO)
    assert sc is not None
    for claim_id in sc.knowledge.visible_claim_ids:
        assert claim_id in sc.claims
        # every visible claim must be assertable by this stakeholder
        claim = sc.claims[claim_id]
        assert claim.context_id in sc.stakeholder.visible_node_ids or (
            claim.context_id in sc.stakeholder.visible_edge_ids
        )


# ---------------------------------------------------------------------------
# Graph context prerequisite for claim scoring (requirement 4)
# ---------------------------------------------------------------------------


def test_missing_required_incoming_topology_blocks_contextual_claim_credit():
    """A node/property claim is scoreable only when the reconstructed incoming
    context covers the complete visible Truth incoming context. Deleting one
    incoming edge of a merge node voids EVERY property claim at that node."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # remove e5 (ap->sq): sq is a merge node whose visible incoming context is
    # {e4, e5}; the reconstructed context no longer covers it
    del tools.db.graph.edges["e5"]
    res = _eval(tools)
    # topology: e5 missing -> edge_recall drops, and the sq node (agent 'e')
    # loses contextual credit for ALL its visible properties
    assert res.edge_recall < 1.0
    assert res.activity_correctness < 1.0
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_missing_incoming_edge_alone_does_not_void_other_nodes():
    """Only the node whose context is incomplete loses credit; other nodes
    keep their contextual credit."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    del tools.db.graph.edges["e5"]
    res = _eval(tools)
    # every node still maps (activity provenance), so node_recall stays 1.0;
    # only the merge node's properties lose credit (5/6 nodes keep 1.0)
    assert res.node_recall == 1.0
    assert 0.5 < res.activity_correctness < 1.0
    assert res.structural_pass is False


def test_same_activity_distinct_positions_remain_distinguishable_with_gate():
    """The context gate keeps same-activity positions distinguishable even
    when node mapping is driven by topology (cycles + rework participate)."""
    from tau2.domains.business_interview.claims import build_claims
    from tau2.domains.business_interview.evaluation import EvaluationSpec, evaluate
    from tau2.domains.business_interview.stakeholder import StakeholderFilter

    # truth: a cycle — x1 -> x2 -> x1 (rework back-edge)
    truth = BusinessProcessGraph(
        nodes={
            "s": Node(id="s", activity=ConceptRef(concept_id="act_start")),
            "x1": Node(id="x1", activity=ConceptRef(concept_id="act_x")),
            "x2": Node(id="x2", activity=ConceptRef(concept_id="act_x")),
        },
        edges={
            "e1": Edge(id="e1", from_node="s", to_node="x1"),
            "e2": Edge(id="e2", from_node="x1", to_node="x2"),
            "e3": Edge(id="e3", from_node="x2", to_node="x1"),  # rework back-edge
        },
        concepts={
            "act_start": BusinessConcept(
                id="act_start", kind="activity", display_label="start"
            ),
            "act_x": BusinessConcept(id="act_x", kind="activity", display_label="x"),
        },
        start_node_id="s",
        end_node_ids=["x2"],
    )
    filter_ = StakeholderFilter(
        name="any",
        visible_node_ids=["s", "x1", "x2"],
        visible_edge_ids=["e1", "e2", "e3"],
        visible_attributes=["activity"],
    )
    claims = build_claims(truth, filter_)

    tools = _tools()
    tools.db.graph = BusinessProcessGraph(
        nodes={
            "s": Node(id="s", activity=ConceptRef(concept_id="s_act")),
            "n1": Node(id="n1", activity=ConceptRef(concept_id="x_act")),
            "n2": Node(id="n2", activity=ConceptRef(concept_id="x_act")),
        },
        edges={
            "z1": Edge(id="z1", from_node="s", to_node="n1"),
            "z2": Edge(id="z2", from_node="n1", to_node="n2"),
            "z3": Edge(id="z3", from_node="n2", to_node="n1"),  # rework back-edge
        },
        concepts={
            "s_act": BusinessConcept(
                id="s_act",
                kind="activity",
                display_label="start",
                validation_status="confirmed",
            ),
            "x_act": BusinessConcept(
                id="x_act",
                kind="activity",
                display_label="x",
                validation_status="confirmed",
            ),
        },
        start_node_id="s",
        end_node_ids=["n2"],
    )
    obs_s = _claim_obs(tools, "We start the process.", [_assertion("s.activity", "start the process")])
    obs_x1 = _claim_obs(tools, "First we do X here.", [_assertion("x1.activity", "do X")])
    obs_x2 = _claim_obs(tools, "Then we do X again.", [_assertion("x2.activity", "do X")])
    obs_e1 = _claim_obs(tools, "After the start we do X.", [_assertion("e1.edge_exists", "After the start we do X")])
    obs_e2 = _claim_obs(tools, "Then we do X again.", [_assertion("e2.edge_exists", "Then we do X again")])
    obs_e3 = _claim_obs(tools, "Sometimes we must redo X.", [_assertion("e3.edge_exists", "redo X")])
    tools.db.graph.nodes["s"].activity.evidence.append(
        EvidenceRef(observation_id=obs_s, quote="start the process")
    )
    tools.db.graph.nodes["n1"].activity.evidence.append(
        EvidenceRef(observation_id=obs_x1, quote="do X")
    )
    tools.db.graph.nodes["n2"].activity.evidence.append(
        EvidenceRef(observation_id=obs_x2, quote="do X")
    )
    tools.db.graph.edges["z1"].evidence.append(
        EvidenceRef(observation_id=obs_e1, quote="After the start we do X")
    )
    tools.db.graph.edges["z2"].evidence.append(
        EvidenceRef(observation_id=obs_e2, quote="Then we do X again")
    )
    tools.db.graph.edges["z3"].evidence.append(
        EvidenceRef(observation_id=obs_e3, quote="redo X")
    )
    res = evaluate(
        tools.db,
        truth,
        EvaluationSpec(),
        filter_,
        claims=claims,
        assertions=tools.assertion_ledger.assertions(),
        alignments=tools.assertion_ledger.alignments(),
        terminology=tools.assertion_ledger.terminology(),
    )
    assert res.node_recall == 1.0
    assert res.edge_recall == 1.0
    assert res.activity_correctness == 1.0
    # drop the rework back-edge: x1's context ({e1, e3}) is no longer covered
    del tools.db.graph.edges["z3"]
    res2 = evaluate(
        tools.db,
        truth,
        EvaluationSpec(),
        filter_,
        claims=claims,
        assertions=tools.assertion_ledger.assertions(),
        alignments=tools.assertion_ledger.alignments(),
        terminology=tools.assertion_ledger.terminology(),
    )
    assert res2.edge_recall < 1.0
    assert res2.activity_correctness < 1.0


# ---------------------------------------------------------------------------
# Global span ambiguity: broad clauses cannot cross-credit (requirement 5)
# ---------------------------------------------------------------------------


def test_broad_clause_cannot_independently_ground_activity_system_data():
    """The goal's canonical regression: one broad clause like \"I check
    customer information in CRM\" must not independently ground activity +
    system + data merely because each scoring call examines a different
    property."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "I check customer information in CRM."
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("cc.activity", "I check customer information in CRM"),
            _assertion("cc.system", "CRM"),
            _assertion("cc.reads.tc_customer", "customer information"),
        ],
        alignments=[
            _confirm_alignment_for("tc_activity_check_customer", "I check customer information in CRM"),
            _confirm_alignment_for("tc_system_crm", "CRM"),
            _confirm_alignment_for("tc_customer", "customer information"),
        ],
    )
    # one broad ref cited for ALL THREE slots; the concepts themselves are
    # confirmed on separate dialogue turns (their confirmation spans do not
    # ground any claim, so only the cited broad clause can support the refs)
    broad = _ev(oid, "I check customer information in CRM")
    yes_act = _claim_obs(
        tools,
        "Yes.",
        alignments=[_confirm_alignment_for("tc_activity_check_customer", "Yes.")],
    )
    yes_sys = _claim_obs(
        tools,
        "Yes.",
        alignments=[_confirm_alignment_for("tc_system_crm", "Yes.")],
    )
    yes_data = _claim_obs(
        tools,
        "Yes.",
        alignments=[_confirm_alignment_for("tc_customer", "Yes.")],
    )
    tools.create_concept("broad_act", "activity", "check", evidence=[_ev(yes_act, "Yes.")])
    tools.confirm_concept("broad_act", evidence=[_ev(yes_act, "Yes.")])
    tools.create_concept("broad_sys", "system", "crm", evidence=[_ev(yes_sys, "Yes.")])
    tools.confirm_concept("broad_sys", evidence=[_ev(yes_sys, "Yes.")])
    tools.create_concept(
        "broad_data", "data", "customer", evidence=[_ev(yes_data, "Yes.")]
    )
    tools.confirm_concept("broad_data", evidence=[_ev(yes_data, "Yes.")])
    tools.db.graph.nodes["b"].activity = ConceptRef(
        concept_id="broad_act",
        confidence=1.0,
        evidence=[_evr(oid, "I check customer information in CRM")],
    )
    tools.db.graph.nodes["b"].system = ConceptRef(
        concept_id="broad_sys",
        confidence=1.0,
        evidence=[_evr(oid, "I check customer information in CRM")],
    )
    tools.db.graph.nodes["b"].reads = [
        ConceptRef(
            concept_id="broad_data",
            confidence=1.0,
            evidence=[_evr(oid, "I check customer information in CRM")],
        )
    ]
    res = _eval(tools)
    # the clause is the activity's OWN assertion span, so it grounds activity
    # (one inseparable assertion, not ambiguous); it cannot independently
    # ground system/data
    assert res.activity_correctness == 1.0
    assert res.system_correctness < 1.0
    assert res.read_correctness < 1.0
    assert res.ambiguous_evidence_ref_count == 0


def test_broad_clause_over_several_independent_clauses_grounds_nothing():
    """A clause spanning assertions of several independent claims grounds
    nothing at any slot (global ambiguity, not slot-local)."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "I check the customer in the CRM and I prepare the quotation."
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("cc.activity", "I check the customer in the CRM"),
            _assertion("cc.system", "CRM"),
            _assertion("cq.activity", "prepare the quotation"),
        ],
        alignments=[
            _confirm_alignment_for(
                "tc_activity_check_customer", "I check the customer in the CRM"
            ),
            _confirm_alignment_for("tc_system_crm", "CRM"),
        ],
    )
    clause = _ev(oid, "I check the customer in the CRM and I prepare the quotation.")
    tools.create_concept("wide_act", "activity", "check", evidence=[clause])
    tools.confirm_concept("wide_act", evidence=[clause])
    tools.db.graph.nodes["b"].activity = ConceptRef(
        concept_id="wide_act",
        confidence=1.0,
        evidence=[
            _evr(oid, "I check the customer in the CRM and I prepare the quotation.")
        ],
    )
    res = _eval(tools)
    # the clause covers cc.activity + cq.activity -> globally ambiguous, so
    # node b's activity evidence grounds nothing and the node cannot map
    assert res.node_recall < 1.0
    assert res.ambiguous_evidence_ref_count >= 1


# ---------------------------------------------------------------------------
# Task prose contains no scenario business facts (requirement 6)
# ---------------------------------------------------------------------------


def test_task_prose_contains_no_scenario_business_facts():
    """The semantic knowledge model is the sole business-fact source: task
    prose (unknown_info / task_instructions / description) must not hard-code
    scenario facts or stale DAG terminology."""
    for task in get_tasks():
        ins = task.user_scenario.instructions
        assert not (getattr(ins, "unknown_info", None) or "").strip(), task.id
        assert not (getattr(ins, "known_info", None) or "").strip(), task.id
        ti = (getattr(ins, "task_instructions", None) or "").lower()
        for banned in ("high-value", "credit risk", "month-end", "1,000,000"):
            assert banned not in ti, f"{task.id}: task_instructions contain {banned!r}"
        desc_obj = task.description
        desc = ""
        if desc_obj is not None:
            desc = (
                f"{desc_obj.purpose or ''} {desc_obj.notes or ''}"
            ).lower()
        assert "dag" not in desc, f"{task.id}: description still says DAG"
    # the JA task is also clean of the scenario specifics
    ti_ja = (
        getattr(
            next(
                t for t in get_tasks() if t.id == JA_SCENARIO
            ).user_scenario.instructions,
            "task_instructions",
            None,
        )
        or ""
    )
    for banned in ("高額承認", "与信リスク", "月末", "経理"):
        assert banned not in ti_ja, banned


# ---------------------------------------------------------------------------
# Observation lifecycle (requirement 7)
# ---------------------------------------------------------------------------


def test_start_inference_preserves_observations_and_ledger():
    """Observations are immutable primary evidence: start_inference may reset
    the graph/glossary/completion state but MUST preserve captured
    Observations, the conversation ledger, and the private sidecar ledger."""
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("cc.system", "CRM")],
        alignments=[_confirm_alignment_for("tc_system_crm", "CRM")],
    )
    tools.create_concept("crm", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.confirm_concept("crm", evidence=[_ev(oid, "CRM")])
    tools.create_concept("act", "activity", "check")
    assert tools.db.graph is not None
    tools.db.graph.concepts["act"].validation_status = "confirmed"
    tools.add_node("n1", activity="act", system="crm", evidence=[_ev(oid, "CRM")])
    # restart inference (e.g. the agent decides to redo its graph)
    tools.start_inference("q2")
    # observations + ledger + conversation survive
    assert [o.id for o in tools.db.observations] == [oid]
    assert len(tools.db.messages) == 2  # assistant hello + user statement
    # the OLD observation is still valid as evidence after the restart
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "I check the customer's information in the CRM."
    tools.create_concept("crm2", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.confirm_concept("crm2", evidence=[_ev(oid, "CRM")])
    tools.create_concept("act2", "activity", "check")
    assert tools.db.graph is not None
    tools.db.graph.concepts["act2"].validation_status = "confirmed"
    tools.add_node("n1", activity="act2", system="crm2", evidence=[_ev(oid, "CRM")])
    assert tools.db.graph is not None
    assert tools.db.graph.nodes["n1"].activity.evidence[0].observation_id == oid
    # private ledger entries are preserved too (evaluator still sees them)
    assert tools.assertion_ledger.alignments() != {}
    assert tools.assertion_ledger.assertions() != {}
    res = _eval(tools)
    assert res.authentic_observation_count == 1


# ---------------------------------------------------------------------------
# Private dialogue-event sidecar plumbing
# ---------------------------------------------------------------------------


def test_sidecar_parse_accepts_dialogue_events():
    from tau2.domains.business_interview.user_simulator import parse_sidecar

    sidecar = parse_sidecar(
        '{"message": "Yes.", "assertions": [], "alignments": '
        '[{"truth_concept_id": "tc_quote", "quote": "Yes.", "act": "confirm"}], '
        '"terminology": [{"truth_concept_id": "tc_customer", '
        '"proposed_term": "customer master", "quote": "Yes."}]}'
    )
    assert sidecar["message"] == "Yes."
    assert sidecar["alignments"][0].truth_concept_id == "tc_quote"
    assert sidecar["alignments"][0].act == "confirm"
    assert sidecar["terminology"][0].proposed_term == "customer master"
    # ordinary workflow speech with no events is fine
    plain = parse_sidecar(
        '{"message": "I check the customer.", '
        '"assertions": [{"claim_id": "cc.activity", "quote": "check", '
        '"occurrence": 0}]}'
    )
    assert plain["alignments"] == []
    assert plain["terminology"] == []
    # a bad act is rejected
    with pytest.raises(ValueError):
        parse_sidecar(
            '{"message": "x", "assertions": [], "alignments": '
            '[{"truth_concept_id": "tc_quote", "quote": "x", '
            '"act": "nonsense"}]}'
        )


def test_environment_binds_private_dialogue_events_per_turn():
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    env.on_message(
        UserMessage(
            role="user",
            content="Yes.",
            stakeholder_alignments=[
                {"truth_concept_id": "tc_quote", "quote": "Yes.", "act": "confirm"}
            ],
            stakeholder_terminology=[
                {
                    "truth_concept_id": "tc_quote",
                    "proposed_term": "offer",
                    "quote": "Yes.",
                }
            ],
        )
    )
    assert 0 in env.assertion_ledger.alignments()
    assert 0 in env.assertion_ledger.terminology()
    # invalid events are rejected loudly at ingestion when a catalog is wired
    from tau2.domains.business_interview.user_simulator import StakeholderUserSimulator

    task = next(t for t in get_tasks() if t.id == SCENARIO)
    StakeholderUserSimulator(llm="dummy", task=task, environment=env, instructions="x")
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="Yes.",
                stakeholder_alignments=[
                    {"truth_concept_id": "tc_hidden", "quote": "Yes.", "act": "confirm"}
                ],
            )
        )
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="Yes.",
                stakeholder_terminology=[
                    {
                        "truth_concept_id": "tc_quote",
                        "proposed_term": "offer",
                        "quote": "no such span",
                    }
                ],
            )
        )
