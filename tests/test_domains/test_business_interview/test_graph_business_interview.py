"""Tests for the unified-glossary business_interview domain (v6).

The agent builds a typed **glossary** (BusinessConcept of kind activity /
actor / system / data / condition / rationale) and an inferred
**BusinessProcessGraph** whose node/edge properties reference glossary
concepts. Correctness is grounded ONLY through private provenance:

    ConceptRef -> EvidenceRef -> exact Observation span
        -> private stakeholder assertion -> StakeholderFact -> TruthClaim
        -> Truth BusinessConcept

The evaluator never inspects labels/descriptions/terms/quotes for meaning;
there are no semantic matchers. Cycles are valid. Concepts start
``hypothesized`` and every referenced concept must be resolved before
completion. The same mechanism serves EN, JA and the lab scenario.
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
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
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

# Per-node observation text + private assertions (fact_id, quote) — EN.
# Quotes must be exact substrings of the observation text.
_NODE_OBS_EN = {
    "r": (
        "I receive the quotation request from the customer and record it.",
        [
            ("quotation.receive.activity", "receive the quotation request"),
            ("quotation.receive.actor", "I"),
            ("quotation.receive.writes", "record it"),
        ],
    ),
    "cc": (
        "I check the customer's information in the CRM.",
        [
            ("quotation.check.activity", "check the customer's information"),
            ("quotation.check.actor", "I"),
            ("quotation.check.system", "CRM"),
            ("quotation.check.reads", "customer's information"),
        ],
    ),
    "cq": (
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            ("quotation.create.activity", "create the quotation"),
            ("quotation.create.actor", "I"),
            ("quotation.create.system", "quoting system"),
            ("quotation.create.reads.customer", "customer"),
            ("quotation.create.reads.pricing", "pricing information"),
            ("quotation.create.writes", "quotation"),
        ],
    ),
    "ap": (
        "Quotations over 1,000,000 yen are approved by the manager; the "
        "approval is for credit risk management.",
        [
            ("quotation.approve.activity", "approved"),
            ("quotation.approve.actor", "manager"),
            ("quotation.approve.rationale", "credit risk management"),
        ],
    ),
    "sq": (
        "I send the quotation to the customer by email.",
        [
            ("quotation.send.activity", "send the quotation"),
            ("quotation.send.actor", "I"),
            ("quotation.send.system", "email"),
        ],
    ),
    "me": (
        "At month-end I send the quotation information summary to Accounting "
        "as an Excel file.",
        [
            ("quotation.month_end.activity", "send the quotation information summary"),
            ("quotation.month_end.actor", "I"),
            ("quotation.month_end.system", "Excel"),
            ("quotation.month_end.writes", "summary"),
        ],
    ),
}

_EDGE_OBS_EN = {
    "e1": (
        "After receiving the request, I check the customer information.",
        [("flow.e1.edge_exists", "After receiving the request, I check")],
    ),
    "e2": (
        "After checking the customer information, I create the quotation.",
        [("flow.e2.edge_exists", "After checking the customer information, I create")],
    ),
    "e3": (
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [
            ("flow.e3.edge_exists", "over 1,000,000 yen"),
            ("condition.e3.over_1m", "over 1,000,000 yen"),
        ],
    ),
    "e4": (
        "Quotations at or below 1,000,000 yen are sent directly to the customer.",
        [
            ("flow.e4.edge_exists", "at or below 1,000,000 yen"),
            ("condition.e4.at_or_below_1m", "at or below 1,000,000 yen"),
        ],
    ),
    "e5": (
        "Once approved, the quotation is sent to the customer.",
        [("flow.e5.edge_exists", "Once approved, the quotation is sent")],
    ),
    "e6": (
        "At month-end I also send the summary to Accounting.",
        [
            ("flow.e6.edge_exists", "month-end"),
            ("condition.e6.month_end", "month-end"),
        ],
    ),
}

_NODE_OBS_JA = {
    "r": (
        "私は顧客から見積依頼を受け付けて記録します。",
        [
            ("quotation.receive.activity", "見積依頼を受け付け"),
            ("quotation.receive.actor", "私"),
            ("quotation.receive.writes", "記録"),
        ],
    ),
    "cc": (
        "私はCRMで顧客情報を確認します。",
        [
            ("quotation.check.activity", "確認"),
            ("quotation.check.actor", "私"),
            ("quotation.check.system", "CRM"),
            ("quotation.check.reads", "顧客情報"),
        ],
    ),
    "cq": (
        "私は見積システムで顧客情報と価格情報を使って見積書を作成します。",
        [
            ("quotation.create.activity", "見積書を作成"),
            ("quotation.create.actor", "私"),
            ("quotation.create.system", "見積システム"),
            ("quotation.create.reads.customer", "顧客情報"),
            ("quotation.create.reads.pricing", "価格情報"),
            ("quotation.create.writes", "見積書"),
        ],
    ),
    "ap": (
        "100万円を超える見積書は管理者の承認が必要で、承認は与信リスク管理のためのものです。",
        [
            ("quotation.approve.activity", "承認が必要"),
            ("quotation.approve.actor", "管理者"),
            ("quotation.approve.rationale", "与信リスク管理"),
        ],
    ),
    "sq": (
        "私は見積書をメールで顧客に送付します。",
        [
            ("quotation.send.activity", "送付"),
            ("quotation.send.actor", "私"),
            ("quotation.send.system", "メール"),
        ],
    ),
    "me": (
        "私は月末に見積情報の集計をExcelファイルとして経理チームに送ります。",
        [
            ("quotation.month_end.activity", "送り"),
            ("quotation.month_end.actor", "私"),
            ("quotation.month_end.system", "Excel"),
            ("quotation.month_end.writes", "集計"),
        ],
    ),
}

_EDGE_OBS_JA = {
    "e1": (
        "依頼を受け付けたら、顧客情報を確認します。",
        [("flow.e1.edge_exists", "受け付けたら、顧客情報を確認")],
    ),
    "e2": (
        "顧客情報を確認したら、見積書を作成します。",
        [("flow.e2.edge_exists", "確認したら、見積書を作成")],
    ),
    "e3": (
        "100万円を超える見積書は管理者の承認に回ります。",
        [
            ("flow.e3.edge_exists", "100万円を超える"),
            ("condition.e3.over_1m", "100万円を超える"),
        ],
    ),
    "e4": (
        "100万円以下の見積書はそのまま顧客に送付します。",
        [
            ("flow.e4.edge_exists", "100万円以下"),
            ("condition.e4.at_or_below_1m", "100万円以下"),
        ],
    ),
    "e5": (
        "承認された見積書は顧客に送付します。",
        [("flow.e5.edge_exists", "承認された見積書は顧客に送付")],
    ),
    "e6": (
        "月末には経理チームへの集計も送ります。",
        [
            ("flow.e6.edge_exists", "月末"),
            ("condition.e6.month_end", "月末"),
        ],
    ),
}

# Agent-local glossary ids (kind, label) per truth concept — deliberately
# different from Truth ids, labels are the agent's own wording.
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


def _assertion(fact_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"fact_id": fact_id, "quote": quote, "occurrence": occurrence}


def _claim_obs(
    tools: InterviewTools,
    text: str,
    assertions: Optional[list[dict]] = None,
) -> str:
    """Ingest a stakeholder message, bind its private assertion sidecar at that
    exact turn, and capture it as an Observation."""
    turn = _ingest(tools, "user", text)
    if assertions:
        tools.fact_ledger.bind(
            turn, [StakeholderAssertion(**a) for a in assertions], text
        )
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


def _ev(obs_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"observation_id": obs_id, "quote": quote, "occurrence": occurrence}


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    """Evaluate under the scenario's stakeholder visibility + private assertion
    provenance (runtime behavior: the sidecar ledger of the tools)."""
    sc = get_scenario(scenario)
    assert sc is not None
    return evaluate(
        tools.db,
        sc.truth,
        sc.spec,
        sc.stakeholder,
        claims=sc.claims,
        facts=sc.facts,
        assertions=tools.fact_ledger.assertions(),
    )


def _make_concept(
    tools: InterviewTools,
    truth_cid: str,
    oid: str,
    quote: str,
    ja: bool = False,
) -> str:
    """Create (and confirm) the agent concept for a truth concept id using the
    given observation span, returning the agent concept id."""
    agent_cid, kind, _label = _AGENT_CONCEPTS[truth_cid]
    tools.create_concept(agent_cid, kind, _label, evidence=[_ev(oid, quote)])
    tools.confirm_concept(agent_cid, evidence=[_ev(oid, quote)])
    return agent_cid


def _build(tools: InterviewTools, ja: bool = False) -> None:
    """Build the correct quotation graph with full provenance, asserting only
    stakeholder-visible properties. Every referenced concept is confirmed, so
    completion succeeds and the evaluator returns a full pass."""
    tools.start_inference("quotation")
    _ingest(tools, "assistant", "Hello.")
    node_obs = _NODE_OBS_JA if ja else _NODE_OBS_EN
    edge_obs = _EDGE_OBS_JA if ja else _EDGE_OBS_EN
    created: dict[str, str] = {}
    node_oids: dict[str, str] = {}
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, assertions = node_obs[sid]
        oid = _claim_obs(tools, text, [_assertion(f, q) for f, q in assertions])
        node_oids[sid] = oid

    # concepts first (activity per node; actor/system/data per truth concept)
    for sid, truth_cid in {
        "r": "r",
        "cc": "cc",
        "cq": "cq",
        "ap": "ap",
        "sq": "sq",
        "me": "me",
    }.items():
        oid = node_oids[sid]
        act_quote = next(q for f, q in node_obs[sid][1] if f.endswith(".activity"))
        created[truth_cid] = _make_concept(tools, truth_cid, oid, act_quote, ja)
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
        # find an observation whose assertions mention this concept's claim
        sid_quote = _quote_for_concept(node_obs, truth_cid)
        sid, quote = sid_quote
        oid = node_oids[sid]
        created[truth_cid] = _make_concept(tools, truth_cid, oid, quote, ja)

    for anid, sid in _NODE_MAP.items():
        oid = node_oids[sid]
        # single-property refs: activity + visible actor/system/rationale
        props = {}
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
        # attach per-ref evidence (quotes) so each ref's span matches an assertion
        _attach_ref_evidence(tools, anid, oid, node_obs[sid][1], ja)

    # edges
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, assertions = edge_obs[eid]
        oid = _claim_obs(tools, text, [_assertion(f, q) for f, q in assertions])
        frm, to = {
            "e1": ("a", "b"),
            "e2": ("b", "c"),
            "e3": ("c", "d"),
            "e4": ("c", "e"),
            "e5": ("d", "e"),
            "e6": ("c", "f"),
        }[eid]
        cond = None
        for f, q in assertions:
            if f.startswith("condition."):
                truth_cid = {
                    "e3": "tc_cond_over_1m",
                    "e4": "tc_cond_at_or_below_1m",
                    "e6": "tc_cond_month_end",
                }[eid]
                agent_cid = _make_concept(tools, truth_cid, oid, q, ja)
                cond = agent_cid
        tools.add_edge(
            eid,
            frm,
            to,
            condition=cond,
            evidence=[
                _ev(oid, q)
                for f, q in assertions
                if f.endswith(".edge_exists") or f.endswith(".condition")
            ],
        )
    tools.finish_interview()


def _quote_for_concept(node_obs, truth_cid: str) -> tuple[str, str]:
    """(node sid, quote) for the first observation asserting this concept."""
    for sid, (text, assertions) in node_obs.items():
        for f, q in assertions:
            if _fact_to_concept(f) == truth_cid:
                return sid, q
    raise KeyError(truth_cid)


def _fact_to_concept(fact_id: str) -> Optional[str]:
    mapping = {
        "quotation.receive.activity": "r",
        "quotation.receive.actor": "tc_actor_sales",
        "quotation.receive.writes": "tc_request",
        "quotation.check.activity": "cc",
        "quotation.check.actor": "tc_actor_sales",
        "quotation.check.system": "tc_system_crm",
        "quotation.check.reads": "tc_customer",
        "quotation.create.activity": "cq",
        "quotation.create.actor": "tc_actor_sales",
        "quotation.create.system": "tc_system_quoting",
        "quotation.create.reads.customer": "tc_customer",
        "quotation.create.reads.pricing": "tc_pricing",
        "quotation.create.writes": "tc_quote",
        "quotation.approve.activity": "ap",
        "quotation.approve.actor": "tc_actor_manager",
        "quotation.approve.rationale": "tc_rationale_credit_risk",
        "quotation.send.activity": "sq",
        "quotation.send.actor": "tc_actor_sales",
        "quotation.send.system": "tc_system_email",
        "quotation.month_end.activity": "me",
        "quotation.month_end.actor": "tc_actor_sales",
        "quotation.month_end.system": "tc_system_excel",
        "quotation.month_end.writes": "tc_excel_summary",
    }
    return mapping.get(fact_id)


def _has_quote(node_obs, sid: str, truth_cid: str) -> bool:
    return any(_fact_to_concept(f) == truth_cid for f, q in node_obs[sid][1])


def _visible_for(sid: str, prop: str, ja: bool) -> bool:
    """True when ``prop`` is visible for the TRUTH node id ``sid``."""
    sc = get_scenario(JA_SCENARIO if ja else SCENARIO)
    assert sc is not None
    visible = sc.stakeholder.node_properties_for(sid)
    if prop == "necessity_rationale":
        return "rationale" in visible
    return prop in visible


def _quote_for_sid_activity(assertions) -> str:
    return next(q for f, q in assertions if f.endswith(".activity"))


def _attach_ref_evidence(
    tools: InterviewTools, anid: str, oid: str, assertions, ja: bool
) -> None:
    """Attach per-property evidence spans to the node's refs so each ref's
    evidence exactly matches the corresponding assertion span."""
    dag = tools.db.graph
    assert dag is not None
    node = dag.nodes[anid]
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
        for f, q in assertions:
            if _fact_to_concept(f) == truth_cid:
                ref.evidence.append(
                    EvidenceRef(
                        **{
                            "observation_id": oid,
                            "quote": q,
                            "occurrence": 0,
                        }
                    )
                )
    for axis in ("reads", "writes"):
        for ref in getattr(node, axis):
            for f, q in assertions:
                if _fact_to_concept(f) == _agent_to_truth(ref.concept_id):
                    ref.evidence.append(
                        EvidenceRef(
                            **{"observation_id": oid, "quote": q, "occurrence": 0}
                        )
                    )


def _agent_to_truth(agent_cid: str) -> Optional[str]:
    for truth_cid, (cid, kind, label) in _AGENT_CONCEPTS.items():
        if cid == agent_cid:
            return truth_cid
    return None


def _build_lab(tools: InterviewTools) -> None:
    """Build a lab graph asserting ONLY stakeholder-visible properties with
    full provenance and confirmed concepts."""
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        (
            "n1",
            "When a specimen arrives, I accession it and record it as received.",
            [
                ("lab.accession.activity", "accession"),
                ("lab.accession.actor", "I"),
                ("lab.accession.reads", "specimen"),
            ],
            ("act_accession", "activity", "specimen accession", "lab.tech", "actor"),
        ),
        (
            "n2",
            "I season the environment chamber to prepare it.",
            [
                ("lab.seasoning.activity", "season"),
                ("lab.seasoning.actor", "I"),
                ("lab.seasoning.system", "environment chamber"),
            ],
            ("act_seasoning", "activity", "chamber seasoning", "lab.tech", "actor"),
        ),
        (
            "n3",
            "I run a conditioning cycle that processes the samples inside the chamber.",
            [
                ("lab.conditioning.activity", "conditioning cycle"),
                ("lab.conditioning.actor", "I"),
                ("lab.conditioning.system", "chamber"),
            ],
            ("act_conditioning", "activity", "conditioning cycle", "lab.tech", "actor"),
        ),
        (
            "n4",
            "The lab supervisor approves the conditioned batch before it is released.",
            [
                ("lab.approval.activity", "approves"),
                ("lab.approval.actor", "lab supervisor"),
            ],
            (
                "act_approval",
                "activity",
                "approve conditioned batch",
                "supervisor",
                "actor",
            ),
        ),
    ]
    created: dict[str, str] = {}

    def mk(cid: str, kind: str, label: str, oid: str, quote: str) -> str:
        tools.create_concept(cid, kind, label, evidence=[_ev(oid, quote)])
        tools.confirm_concept(cid, evidence=[_ev(oid, quote)])
        return cid

    for sid, text, assertions, (
        act_cid,
        act_kind,
        act_label,
        actor_cid,
        actor_kind,
    ) in nodes:
        oid = _claim_obs(tools, text, [_assertion(f, q) for f, q in assertions])
        created[f"act_{sid}"] = mk(
            act_cid,
            act_kind,
            act_label,
            oid,
            next(q for f, q in assertions if f.endswith(".activity")),
        )
        actor_truth = {
            "n1": "tc_actor_lab_tech",
            "n2": "tc_actor_lab_tech",
            "n3": "tc_actor_lab_tech",
            "n4": "tc_actor_lab_supervisor",
        }[sid]
        actor_cid_full = (
            "lab_tech" if actor_truth == "tc_actor_lab_tech" else "lab_supervisor"
        )
        if actor_cid_full not in created:
            created[actor_cid_full] = mk(
                actor_cid_full,
                "actor",
                "lab tech" if actor_cid_full == "lab_tech" else "lab supervisor",
                oid,
                next(
                    q
                    for f, q in assertions
                    if f
                    == {
                        "n1": "lab.accession.actor",
                        "n2": "lab.seasoning.actor",
                        "n3": "lab.conditioning.actor",
                        "n4": "lab.approval.actor",
                    }[sid]
                ),
            )
        sample_cid = None
        if sid == "n1":
            sample_cid = mk("sample", "data", "sample", oid, "specimen")
        chamber_cid = None
        if sid in ("n2", "n3"):
            if "chamber" not in created:
                created["chamber"] = mk(
                    "chamber",
                    "system",
                    "environment chamber",
                    oid,
                    "environment chamber" if sid == "n2" else "chamber",
                )
            chamber_cid = "chamber"
        args = {
            "node_id": sid,
            "activity": created[f"act_{sid}"],
            "actor": created[actor_cid_full],
            "evidence": [
                _ev(oid, next(q for f, q in assertions if f.endswith(".activity")))
            ],
        }
        if sid == "n1":
            args["reads"] = [sample_cid]
        if sid in ("n2", "n3"):
            args["system"] = chamber_cid
        tools.add_node(**args)
        assert tools.db.graph is not None
        node = tools.db.graph.nodes[sid]
        if sid == "n1":
            node.reads[0].evidence.append(
                EvidenceRef(observation_id=oid, quote="specimen")
            )
        if sid in ("n2", "n3"):
            node.system.evidence.append(
                EvidenceRef(
                    observation_id=oid,
                    quote="environment chamber" if sid == "n2" else "chamber",
                )
            )
        node.actor.evidence.append(
            EvidenceRef(
                observation_id=oid,
                quote={"n1": "I", "n2": "I", "n3": "I", "n4": "lab supervisor"}[sid],
            )
        )

    edges = [
        (
            "l1",
            "n1",
            "n2",
            "After specimen accession, I prepare the chamber for seasoning.",
            [("lab.flow.l1", "After specimen accession, I prepare")],
        ),
        (
            "l2",
            "n2",
            "n3",
            "After chamber seasoning, I run the conditioning cycle.",
            [("lab.flow.l2", "After chamber seasoning, I run")],
        ),
        (
            "l3",
            "n3",
            "n4",
            "After the conditioning cycle, the lab supervisor approves the batch.",
            [
                (
                    "lab.flow.l3",
                    "After the conditioning cycle, the lab supervisor approves",
                )
            ],
        ),
    ]
    for eid, frm, to, text, assertions in edges:
        oid = _claim_obs(tools, text, [_assertion(f, q) for f, q in assertions])
        tools.add_edge(eid, frm, to, evidence=[_ev(oid, q) for f, q in assertions])
    tools.finish_interview()


def _reference_trajectory() -> list:
    """A full faithful trajectory (messages + tool calls) mirroring ``_build``,
    with the private assertion sidecars carried on the UserMessages.

    Concepts are created + confirmed with their evidence spans and extended
    with ``add_concept_term`` for every additional observation span, so the
    evaluator binds each ref through the concept's evidence alone (ref-level
    evidence is optional).
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

    def say(text, assertions):
        traj.append(
            UM(
                role="user",
                content=text,
                stakeholder_assertions=[_assertion(f, q) for f, q in assertions],
            )
        )
        tools.db.messages.append({"role": "user", "content": text})

    mk("start_inference", {"name": "Quotation creation"})
    created: set[str] = set()
    node_oid: dict[str, str] = {}
    sm = 0
    # node observations -> concepts -> nodes
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, assertions = _NODE_OBS_EN[sid]
        sm += 1
        say(text, assertions)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        node_oid[sid] = oid
        # create/confirm concepts for the claims asserted here (once each)
        for f, q in assertions:
            truth_cid = _fact_to_concept(f)
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
                    "add_concept_term",
                    {"concept_id": agent_cid, "term": q, "evidence": [_ev(oid, q)]},
                )
    # nodes
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
    # edges
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, assertions = _EDGE_OBS_EN[eid]
        sm += 1
        say(text, assertions)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        cond = None
        for f, q in assertions:
            if f.startswith("condition."):
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
            "evidence": [_ev(oid, q) for f, q in assertions],
        }
        if cond:
            eargs["condition"] = cond
        mk("add_edge", eargs)
    mk("finish_interview", {"summary": "Inferred the quotation graph."})
    return traj


# ---------------------------------------------------------------------------
# Model shape (BusinessProcessGraph / cycles)
# ---------------------------------------------------------------------------


def test_cycles_are_valid():
    """Cycles are valid: a graph with a cycle has no structural errors."""
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
            "x": __import__(
                "tau2.domains.business_interview.graph", fromlist=["BusinessConcept"]
            ).BusinessConcept(id="x", kind="activity", preferred_label="x"),
            "y": __import__(
                "tau2.domains.business_interview.graph", fromlist=["BusinessConcept"]
            ).BusinessConcept(id="y", kind="activity", preferred_label="y"),
        },
    )
    assert graph.structure_errors() == []
    assert graph.is_valid


def test_truth_and_agent_result_use_same_graph_class():
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert isinstance(sc.truth, BusinessProcessGraph)
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    assert isinstance(tools.db.graph, BusinessProcessGraph)


def test_all_concept_kinds_present_in_truth_and_glossary():
    """Every ConceptKind appears in the quotation truth and in the agent's
    faithful glossary."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    kinds = {c.kind for c in sc.truth.concepts.values()}
    assert kinds == {"activity", "actor", "system", "data", "condition", "rationale"}
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    agent_kinds = {c.kind for c in tools.db.graph.concepts.values()}
    assert agent_kinds == kinds


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
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0
    assert res.edge_recall == 1.0
    assert res.edge_precision == 1.0
    assert res.concept_correctness == 1.0


def test_missing_node_lowers_node_recall():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    del tools.db.graph.nodes["f"]
    for eid in ("e6",):
        tools.db.graph.edges.pop(eid, None)
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.structural_pass is False


def test_fabricated_node_stays_unmapped():
    """A node whose activity has no provenance support is not mapped to any
    Truth node."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    tools.create_concept(
        "pizza_act", "activity", "eat pizza", evidence=[_ev(pizza, "pizza")]
    )
    tools.confirm_concept("pizza_act", evidence=[_ev(pizza, "pizza")])
    tools.add_node("fab", activity="pizza_act", evidence=[_ev(pizza, "pizza")])
    res = _eval(tools)
    assert res.node_precision < 1.0
    assert res.fabricated_node_count >= 1
    assert res.structural_pass is False


def test_wrong_actor_drops_actor_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # point cc's actor at the manager concept (no provenance for cc.actor)
    tools.db.graph.nodes["b"].actor = ConceptRef(concept_id="manager", confidence=1.0)
    res = _eval(tools)
    assert res.actor_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_system_drops_system_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["b"].system = ConceptRef(concept_id="email", confidence=1.0)
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_read_drops_read_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["c"].reads = [ConceptRef(concept_id="quote", confidence=1.0)]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_write_drops_write_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(concept_id="request", confidence=1.0)
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_rationale_drops_rationale_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["d"].necessity_rationale = ConceptRef(
        concept_id="over_1m", confidence=1.0
    )
    res = _eval(tools)
    assert res.rationale_correctness < 1.0
    assert res.structural_pass is False


def test_missing_visible_property_lowers_score():
    """A visible property left unset (e.g. cc.system) is a recall miss."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.nodes["b"].system = None
    res = _eval(tools)
    assert res.system_correctness < 1.0
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
    assert res.system_correctness == 1.0
    assert res.quality_pass is True


def test_hidden_assertions_fail_regardless_of_provenance():
    """sq.reads asserted with valid quote provenance still fails: hidden axes
    are a prior gate and no claim exists for them."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I send the quotation to the customer by email.",
        [_assertion("quotation.send.activity", "send the quotation")],
    )
    tools.create_concept(
        "quote_dup", "data", "quotation", evidence=[_ev(oid, "quotation")]
    )
    tools.confirm_concept("quote_dup", evidence=[_ev(oid, "quotation")])
    tools.db.graph.nodes["e"].reads = [
        ConceptRef(
            concept_id="quote_dup", confidence=1.0, evidence=[_ev(oid, "quotation")]
        )
    ]
    tools.db.graph.nodes["e"].writes = [
        ConceptRef(
            concept_id="quote_dup", confidence=1.0, evidence=[_ev(oid, "quotation")]
        )
    ]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Provenance-only binding (no labels, no text)
# ---------------------------------------------------------------------------


def test_arbitrary_labels_bind_through_provenance():
    """A concept labeled completely arbitrarily binds through provenance."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [_assertion("quotation.create.writes", "quotation")],
    )
    tools.create_concept("whatever", "data", "zzz", evidence=[_ev(oid, "quotation")])
    tools.confirm_concept("whatever", evidence=[_ev(oid, "quotation")])
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="whatever", confidence=1.0, evidence=[_ev(oid, "quotation")]
        )
    ]
    res = _eval(tools)
    assert res.write_correctness == 1.0
    assert res.concept_correctness == 1.0
    assert res.structural_pass is True


def test_paraphrasing_labels_descriptions_does_not_change_score():
    """Renaming labels/descriptions of the agent glossary does not change any
    semantic score (the evaluator never reads them)."""
    a = _tools()
    _build(a)
    b = _tools()
    _build(b)
    assert b.db.graph is not None
    for concept in b.db.graph.concepts.values():
        concept.preferred_label = "renamed-" + concept.id
        concept.description = "completely different description"
        for term in concept.terms:
            term.text = "renamed-term"
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
    """Identical observation text with different private assertions grounds
    differently: binding follows assertions, never the wording."""
    text = "I handle the paperwork for the deal."
    ok = _tools()
    _build(ok)
    assert ok.db.graph is not None
    oid = _claim_obs(ok, text, [_assertion("quotation.create.writes", "paperwork")])
    ok.create_concept("paper", "data", "paperwork", evidence=[_ev(oid, "paperwork")])
    ok.confirm_concept("paper", evidence=[_ev(oid, "paperwork")])
    ok.db.graph.nodes["c"].writes = [
        ConceptRef(concept_id="paper", confidence=1.0, evidence=[_ev(oid, "paperwork")])
    ]
    res_ok = _eval(ok)
    assert res_ok.write_correctness == 1.0

    bad = _tools()
    _build(bad)
    assert bad.db.graph is not None
    # same text, but the assertion quotes the activity span instead
    oid2 = _claim_obs(
        bad, text, [_assertion("quotation.create.activity", "handle the paperwork")]
    )
    bad.create_concept("paper2", "data", "paperwork", evidence=[_ev(oid2, "paperwork")])
    bad.confirm_concept("paper2", evidence=[_ev(oid2, "paperwork")])
    bad.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="paper2", confidence=1.0, evidence=[_ev(oid2, "paperwork")]
        )
    ]
    res_bad = _eval(bad)
    assert res_bad.write_correctness < 1.0
    assert res_bad.unsupported_ref_count >= 1


def test_invalid_evidence_span_rejected():
    """Evidence whose quote is not an exact span of the Observation is invalid
    (evidence hygiene fails) and cannot ground anything."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = next(o.id for o in tools.db.observations if o.turn == 3)  # cq obs
    # the tool itself rejects a quote that is not an exact span
    tools.create_concept("bogus", "data", "quotation")
    with pytest.raises(ValueError):
        tools.confirm_concept("bogus", evidence=[_ev(oid, "not in the text")])
    # direct mutation with an invalid span fails evidence hygiene
    tools.db.graph.concepts["bogus"].validation_evidence.append(
        EvidenceRef(observation_id=oid, quote="not in the text", occurrence=0)
    )
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="bogus", confidence=1.0, evidence=[_ev(oid, "not in the text")]
        )
    ]
    res = _eval(tools)
    assert res.invalid_evidence_ref_count >= 1
    assert res.evidence_pass is False
    assert res.write_correctness < 1.0


def test_invalid_assertion_quote_rejected_at_ingestion():
    """An assertion whose quote/occurrence do not exactly match the message is
    rejected at ingestion (catalog installed) and at evaluation."""
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    from tau2.domains.business_interview.user_simulator import StakeholderUserSimulator

    task = next(t for t in get_tasks() if t.id == SCENARIO)
    StakeholderUserSimulator(llm="dummy", task=task, environment=env, instructions="x")
    assert env.fact_ledger is not None
    assert env.fact_ledger.catalog is not None
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer.",
                stakeholder_assertions=[_assertion("quotation.check.system", "CRM")],
            )
        )
    # occurrence out of range
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_assertions=[
                    _assertion("quotation.check.system", "CRM", occurrence=3)
                ],
            )
        )
    # unknown fact id
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_assertions=[_assertion("quotation.nope", "CRM")],
            )
        )


def test_second_occurrence_span_matching():
    """occurrence selects the right span when a quote appears twice."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "the quotation is one thing, and the quotation is another"
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("quotation.create.writes", "quotation", occurrence=0),
            _assertion("quotation.create.activity", "quotation", occurrence=1),
        ],
    )
    # evidence at occurrence 1 supports the activity claim only
    tools.create_concept(
        "occ_act",
        "activity",
        "create the quotation",
        evidence=[_ev(oid, "quotation", 1)],
    )
    tools.confirm_concept("occ_act", evidence=[_ev(oid, "quotation", 1)])
    assert tools.db.graph is not None
    tools.db.graph.nodes["c"].activity = ConceptRef(
        concept_id="occ_act", confidence=1.0, evidence=[_ev(oid, "quotation", 1)]
    )
    res = _eval(tools)
    # activity claim still grounded via the second occurrence
    assert res.activity_correctness == 1.0


def test_multi_fact_utterance_cannot_cross_credit_unrelated_concepts():
    """One utterance asserting several facts: a ref citing one span must not
    bind the claims of the other facts."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    text = "I check the customer in the CRM and prepare the quotation."
    oid = _claim_obs(
        tools,
        text,
        [
            _assertion("quotation.check.system", "CRM"),
            _assertion("quotation.create.activity", "prepare the quotation"),
        ],
    )
    # a data concept citing the CRM span must NOT bind the create activity
    tools.create_concept("crm_data", "data", "CRM data", evidence=[_ev(oid, "CRM")])
    tools.confirm_concept("crm_data", evidence=[_ev(oid, "CRM")])
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(concept_id="crm_data", confidence=1.0, evidence=[_ev(oid, "CRM")])
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_ref_count >= 1
    assert res.concept_correctness == 0.0  # crm_data cannot bind tc_quote


# ---------------------------------------------------------------------------
# Concept identity (reuse / split / merge)
# ---------------------------------------------------------------------------


def test_reuse_passes():
    """One actor concept reused across several nodes and one data concept
    reused across cc.reads and cq.reads passes concept binding."""
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.concept_correctness == 1.0
    assert res.structural_pass is True


def test_split_identity_fails_until_merged():
    """Two agent concepts for one Truth actor fail until merged."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("quotation.check.actor", "I")],
    )
    tools.create_concept("sales_dup", "actor", "sales", evidence=[_ev(oid, "I")])
    tools.confirm_concept("sales_dup", evidence=[_ev(oid, "I")])
    tools.db.graph.nodes["b"].actor = ConceptRef(
        concept_id="sales_dup", confidence=1.0, evidence=[_ev(oid, "I")]
    )
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    # repair: merge sales_dup into sales
    tools.merge_concepts("sales", ["sales_dup"])
    res2 = _eval(tools)
    assert res2.concept_correctness == 1.0
    assert res2.structural_pass is True


def test_merge_incompatible_kinds_rejected():
    """Merging concepts of different kinds is rejected by the tool."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "actor", "alpha")
    tools.create_concept("b", "data", "beta")
    with pytest.raises(ValueError):
        tools.merge_concepts("a", ["b"])


def test_merge_distinct_truth_concepts_fails():
    """One agent concept bound to two distinct Truth concepts (of one kind)
    fails concept binding."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            _assertion("quotation.create.reads.customer", "customer"),
            _assertion("quotation.create.reads.pricing", "pricing information"),
        ],
    )
    tools.create_concept(
        "blend", "data", "customer pricing", evidence=[_ev(oid, "customer")]
    )
    tools.add_concept_term(
        "blend", "pricing", evidence=[_ev(oid, "pricing information")]
    )
    tools.confirm_concept("blend", evidence=[_ev(oid, "customer")])
    tools.db.graph.nodes["c"].reads = [
        ConceptRef(concept_id="blend", confidence=1.0, evidence=[_ev(oid, "customer")])
    ]
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


def test_incompatible_kind_binding_fails():
    """An agent data concept used at an actor slot cannot bind (kinds are
    incompatible)."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "I check the customer's information in the CRM.",
        [_assertion("quotation.check.actor", "I")],
    )
    tools.create_concept("actor_data", "data", "some data", evidence=[_ev(oid, "I")])
    tools.confirm_concept("actor_data", evidence=[_ev(oid, "I")])
    tools.db.graph.nodes["b"].actor = ConceptRef(
        concept_id="actor_data", confidence=1.0, evidence=[_ev(oid, "I")]
    )
    res = _eval(tools)
    # the claim is grounded by its evidence, but the binding is incompatible
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


def test_unrelated_observation_cannot_support_a_claim():
    """An authentic but unrelated Observation cannot ground any ref."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    tools.create_concept(
        "pizza_quote", "data", "quotation", evidence=[_ev(pizza, "pizza")]
    )
    tools.confirm_concept("pizza_quote", evidence=[_ev(pizza, "pizza")])
    tools.db.graph.nodes["c"].writes = [
        ConceptRef(
            concept_id="pizza_quote", confidence=1.0, evidence=[_ev(pizza, "pizza")]
        )
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_ref_count >= 1
    assert res.evidence_pass is True  # the obs is real & authentic (hygiene ok)
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Edge grounding
# ---------------------------------------------------------------------------


def test_edge_existence_needs_provenance():
    """An agent edge whose evidence does not support any edge_exists claim is
    not matched (fabricated/unsupported edge)."""
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
    """Conditions bind through provenance; a wrong condition concept fails."""
    good = _tools()
    _build(good)
    res = _eval(good)
    assert res.condition_correctness == 1.0

    bad = _tools()
    _build(bad)
    assert bad.db.graph is not None
    oid = _claim_obs(
        bad,
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [_assertion("flow.e3.edge_exists", "over 1,000,000 yen")],
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
        evidence=[_ev(oid, "over 1,000,000 yen")],
    )
    res2 = _eval(bad)
    assert res2.condition_correctness < 1.0
    assert res2.structural_pass is False


def test_condition_on_unconditional_edge_fails():
    """An asserted condition on an edge with no truth condition has no
    provenance and fails."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _claim_obs(
        tools,
        "After receiving the request, I check the customer information.",
        [_assertion("flow.e1.edge_exists", "After receiving the request, I check")],
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
        evidence=[_ev(oid, "After receiving the request")],
    )
    res = _eval(tools)
    assert res.condition_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Glossary validation / completion
# ---------------------------------------------------------------------------


def test_hypothesized_referenced_concept_blocks_completion():
    """finish_interview refuses while a referenced concept is hypothesized;
    the evaluator's glossary_pass is False."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # revert one concept to hypothesized
    tools.db.graph.concepts["customer"].validation_status = "hypothesized"
    with pytest.raises(ValueError):
        tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is False
    assert "customer" in res.referenced_hypothesized_concepts
    assert res.structural_pass is False


def test_unreferenced_hypothesized_concept_does_not_block():
    """A hypothesized concept that is never referenced does not block
    completion."""
    tools = _tools()
    _build(tools)
    tools.create_concept("unused", "data", "unused concept")
    tools.finish_interview()  # no raise
    res = _eval(tools)
    assert res.glossary_pass is True


def test_unknown_and_disputed_can_complete_when_grounded():
    """Referenced concepts marked unknown/disputed resolve completion."""
    for status, fn in (
        ("unknown", "mark_concept_unknown"),
        ("disputed", "mark_concept_disputed"),
    ):
        tools = _tools()
        _build(tools)
        assert tools.db.graph is not None
        # replace the customer concept's status via the tool
        getattr(tools, fn)("customer")
        # references still fully grounded
        tools.finish_interview()  # no raise
        res = _eval(tools)
        assert res.glossary_pass is True
        assert res.concept_correctness == 1.0
        assert res.structural_pass is True


def test_partially_confirmed_can_complete():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.db.graph.concepts["pricing"].validation_status = "partially_confirmed"
    tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is True


def test_confirm_concept_requires_authentic_evidence():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("c", "data", "thing")
    with pytest.raises(ValueError):
        tools.confirm_concept("c")  # no evidence
    with pytest.raises(ValueError):
        tools.confirm_concept("c", evidence=[_ev("obs_missing", "x")])  # unknown obs
    with pytest.raises(ValueError):
        tools.confirm_concept("c", evidence=[_ev("obs_missing", "x", 0)])  # unknown obs


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
    tools.db.graph.nodes["b"].actor.evidence.append(
        EvidenceRef(observation_id="obs_fake", quote="made up", occurrence=0)
    )
    res = _eval(tools)
    assert res.invalid_observation_source_count >= 1
    assert res.provenance_authenticity_pass is False
    assert res.evidence_pass is False
    assert res.quality_pass is False


def test_zero_evidence_refs_fail_evidence_gate():
    """Asserted refs with no evidence at all fail the evidence gate."""
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    # build a structurally valid but evidence-free graph
    for cid, kind, label in [
        ("act", "activity", "receive"),
        ("sales", "actor", "sales"),
        ("req", "data", "request"),
    ]:
        tools.create_concept(cid, kind, label)
        tools.confirm_concept(cid, evidence=[]) if False else None
    # confirm without evidence is impossible; use direct status via tool marks
    tools.mark_concept_unknown("act")
    tools.mark_concept_unknown("sales")
    tools.mark_concept_unknown("req")
    tools.add_node("a", activity="act", actor="sales", writes=["req"])
    tools.add_edge("e1", "a", "a", evidence=[])
    res = _eval(tools)
    assert res.evidence_pass is False


# ---------------------------------------------------------------------------
# Private-id leakage
# ---------------------------------------------------------------------------


def test_private_ids_absent_from_agent_visible_state():
    """Fact ids, claim ids and truth concept ids never appear in Agent-visible
    tools, serialized DB state or messages."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    listing = tools.list_concepts()
    for forbidden in ("quotation.", "tc_", "claim", "edge_exists"):
        assert forbidden not in listing, forbidden
    dumped = str(tools.db.model_dump(mode="json"))
    sc = get_scenario(SCENARIO)
    assert sc is not None
    for fid in list(sc.facts)[:5]:
        assert fid not in dumped
    for cid in list(sc.claims)[:5]:
        assert cid not in dumped
    assert "fact_ledger" not in InterviewDB.model_fields


def test_user_message_serialization_excludes_assertions():
    msg = UserMessage(
        role="user",
        content="hello",
        stakeholder_assertions=[_assertion("quotation.create.writes", "quotation")],
    )
    dumped = msg.model_dump(mode="json")
    assert "stakeholder_assertions" not in dumped
    assert "quotation.create.writes" not in str(dumped)
    assert "quotation.create.writes" not in str(msg)
    assert "quotation.create.writes" not in repr(msg)


def test_ledger_binds_assertions_per_turn():
    """The environment binds the assertion sidecar against the exact message
    turn; a plain message binds nothing."""
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    env.on_message(UserMessage(role="user", content="plain statement"))
    assert env.fact_ledger is not None
    assert env.fact_ledger.assertions() == {}
    env.on_message(
        UserMessage(
            role="user",
            content="I check the customer in the CRM.",
            stakeholder_assertions=[
                {"fact_id": "quotation.check.system", "quote": "CRM", "occurrence": 0}
            ],
        )
    )
    # catalog is not installed here (no adapter) — raw bind, validated at eval
    assert 1 in env.fact_ledger.assertions()


# ---------------------------------------------------------------------------
# Old semantic matcher machinery is gone
# ---------------------------------------------------------------------------


def test_semantic_matcher_machinery_is_gone():
    import pathlib

    from tau2.domains.business_interview import evaluation, scenario

    assert not hasattr(evaluation, "_predicate_ok")
    assert not hasattr(evaluation, "_necessity_value_ok")
    assert not hasattr(evaluation, "_attribute_match")
    assert not hasattr(evaluation, "_node_overlap")
    assert not hasattr(evaluation, "_STOPWORDS")
    assert not hasattr(evaluation, "_term_covers")
    assert not hasattr(evaluation, "TruthNodeSpec")
    assert not hasattr(scenario, "EvaluationSpec") or "expressions" not in (
        scenario.EvaluationSpec.model_fields
        if hasattr(scenario, "EvaluationSpec")
        else {}
    )
    domain_dir = pathlib.Path(evaluation.__file__).parent
    banned = (
        "preferred_label",
        "_predicate_ok",
        "_necessity_value_ok",
        "_attribute_match",
        "norm_role",
        "norm_system",
        "token overlap",
    )
    for src in domain_dir.glob("*.py"):
        text = src.read_text(encoding="utf-8")
        for token in banned:
            if token in ("preferred_label",):
                # labels exist as data, but must never be *matched*
                continue
            assert token not in text, f"{src.name} still contains {token!r}"
    # deleted modules no longer exist
    assert not (domain_dir / "dag.py").exists()
    assert not (domain_dir / "concepts.py").exists()
    assert not (domain_dir / "aliases.py").exists()


def test_evaluation_spec_is_empty():
    """The scenario EvaluationSpec carries no semantic expression lists."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert sc.spec.model_dump() == {}


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


def test_invalid_and_fabricated_message_ids_rejected():
    tools = _tools()
    _ingest(tools, "user", "Only one real message.")
    for bad in ("sm_0", "sm_2", "sm_99", "turn_3", "foo"):
        with pytest.raises(ValueError):
            tools.observe_message(bad)
    empty = _tools()
    with pytest.raises(ValueError):
        empty.observe_latest_stakeholder_message()


def test_assistant_tool_messages_cannot_be_observed():
    tools = _tools()
    _ingest(tools, "assistant", "I ask a question.")
    _ingest(tools, "user", "A real stakeholder statement.")
    _ingest(tools, "tool", "a tool result")
    listing = tools.list_stakeholder_messages()
    assert listing.count("sm_") == 1
    oid = tools.observe_message("sm_1")
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "A real stakeholder statement."


def test_observation_capture_survives_set_state_replay():
    """Replaying a conversation via environment.set_state restores the same
    stable sm ids and observations (deterministic, append-only ledger)."""
    from tau2.data_model.message import (
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


def test_merge_concepts_repoints_refs_and_folds_terms():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "data", "alpha")
    tools.create_concept("b", "data", "beta")
    tools.create_concept("act", "activity", "first step")
    tools.mark_concept_unknown("act")
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
        tools.mark_concept_unknown(cid)
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
    tools.mark_concept_unknown("act")
    tools.add_node("a", activity="act")
    tools.add_node("b", activity="act")
    tools.add_edge("e1", "a", "b", evidence=[])
    tools.add_edge("e2", "b", "a", evidence=[])
    out = tools.validate_graph()
    assert "validation errors" not in out.lower()
    assert "valid" in out.lower()


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
    assert res.concept_correctness == 1.0


def test_lab_hidden_derived_artifact_assertion_fails():
    tools = _tools()
    _build_lab(tools)
    assert tools.db.graph is not None
    tools.create_concept("seasoned", "data", "seasoned chamber")
    tools.mark_concept_unknown("seasoned")
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


def test_task_instructions_reference_facts_not_known_info():
    """known_info no longer carries business facts; task_instructions refer to
    the structured facts."""
    for task in get_tasks():
        ins = task.user_scenario.instructions
        assert not (ins.known_info or "").strip(), task.id
        ti = (ins.task_instructions or "").lower()
        if task.id != "quotation_workflow_1_ja":
            assert "structured facts" in ti or "your facts" in ti, task.id


def test_agent_policy_requires_glossary_discipline():
    p = " ".join(BUSINESS_INTERVIEW_POLICY_PATH.read_text().split()).lower()
    assert "glossary" in p
    assert "hypothesized" in p
    assert "confirm_concept" in p
    assert "exact substring" in p
    assert "cycles are valid" in p or "cycles are normal" in p


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
