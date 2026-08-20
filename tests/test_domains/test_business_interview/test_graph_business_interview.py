"""Tests for the graph-native business_interview domain (v11).

The graph is the semantic model: Truth = BusinessProcessGraph + TruthConcept[];
the stakeholder's world model is a StakeholderKnowledge (masked graph with
three-valued slots ConceptRef | None | DONT_KNOW + local concepts); the agent
builds an AgentGraph + AgentConcept[]. Correctness is grounded ONLY through
private provenance:

    Agent EvidenceRef -> Observation span -> private annotation
        (stakeholder semantic ID) -> StakeholderKnowledgeGraph element /
        StakeholderKnowledgeConcept

Stakeholder semantic IDs are opaque and stakeholder-local (``skn_001`` /
``ske_001`` / ``skc_001`` style, assigned deterministically from sorted Truth
ids; never derived from Truth ids/labels/terms/positions) — the hand-authored
tables below are keyed by Truth ids and translated via the private mappings
(``_local_sid``) because the tests are evaluator-side.

Property scoring uses property evidence ONLY and scores the three-valued
epistemic states asymmetrically (ConceptRef -> correct grounded ref; None ->
unasserted; DONT_KNOW -> explicit evidenced DONT_KNOW marker). Concept
identity may use mentions + graph provenance (incl. grounding evidence);
validation uses explicit validation/dialogue evidence ONLY. The single
canonical semantic-id resolver is ``StakeholderKnowledgeGraph.resolve``.
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
from tau2.domains.business_interview.evaluation import (
    EvaluationSpec,
    evaluate,
    grounded_semantic_ids,
    resolve_grounding_refs,
)
from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    SemanticAnnotation,
    StakeholderKnowledgeCatalog,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import (
    DONT_KNOW,
    BusinessProcessGraph,
    ConceptRef,
    DontKnowType,
    Edge,
    EvidenceRef,
    InterviewDB,
    Node,
    TruthConcept,
    graph_semantic_ids,
    is_dont_know,
)
from tau2.domains.business_interview.knowledge import (
    project_knowledge,
    slot_concepts,
)
from tau2.domains.business_interview.scenario import (
    Scenario,
    get_scenario,
    quotation_truth,
)
from tau2.domains.business_interview.stakeholder import (
    ConceptKnowledgeOverride,
    StakeholderFilter,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import BUSINESS_INTERVIEW_POLICY_PATH

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
LAB_SCENARIO = "lab_sample_flow"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO, LAB_SCENARIO]


# ---------------------------------------------------------------------------
# Opaque-ID translation (tests are evaluator-side and may use the private
# Truth mappings of the StakeholderKnowledgeGraph)
# ---------------------------------------------------------------------------


def _sc(scenario_id: str = SCENARIO):
    sc = get_scenario(scenario_id)
    assert sc is not None
    return sc


def _kg(scenario_id: str = SCENARIO):
    return _sc(scenario_id).knowledge.graph


def _local_maps(scenario_id: str = SCENARIO):
    """(truth node id -> local, truth edge id -> local, truth concept id ->
    local) via the private evaluator-only mappings."""
    kg = _kg(scenario_id)
    node = {t: k for k, t in kg.node_truth_ids.items()}
    edge = {t: k for k, t in kg.edge_truth_ids.items()}
    concept = {c.truth_concept_id: k for k, c in kg.concepts.items()}
    return node, edge, concept


def _local_sid(sid: str, scenario_id: str = SCENARIO) -> str:
    """Translate a Truth-id semantic id to the scenario's opaque knowledge
    semantic id (e.g. ``node:r:reads:tc_request`` ->
    ``node:skn_005:reads:skc_002``). Already-local ids pass through."""
    node_t2l, edge_t2l, concept_t2l = _local_maps(scenario_id)
    parts = sid.split(":")
    if parts[0] == "node":
        if parts[1].startswith("skn_"):
            if len(parts) == 4 and not parts[3].startswith("skc_"):
                raise KeyError(sid)
            return sid
        local = node_t2l.get(parts[1])
        if local is None:
            raise KeyError(sid)
        out = ["node", local] + parts[2:]
        if len(parts) == 4:
            concept = concept_t2l.get(parts[3])
            if concept is None:
                raise KeyError(sid)
            out[3] = concept
        return ":".join(out)
    if parts[0] == "edge":
        if parts[1].startswith("ske_"):
            return sid
        local = edge_t2l.get(parts[1])
        if local is None:
            raise KeyError(sid)
        return ":".join(["edge", local] + parts[2:])
    if sid.startswith("skc_"):
        return sid
    concept = concept_t2l.get(sid)
    if concept is None:
        raise KeyError(sid)
    return concept


# (agent node id, stakeholder node id)
_NODE_MAP = {"a": "r", "b": "cc", "c": "cq", "d": "ap", "e": "sq", "f": "me"}

# Per-node stakeholder speech: (text, [(semantic_id, quote)]) — EN.
_NODE_OBS = {
    "r": (
        "I receive the quotation request from the customer and record it.",
        [
            ("node:r:activity", "receive the quotation request"),
            ("node:r:actor", "I"),
            ("node:r:writes:tc_request", "record it"),
        ],
    ),
    "cc": (
        "I check the customer information in the CRM.",
        [
            ("node:cc:activity", "check the customer information"),
            ("node:cc:actor", "I"),
            ("node:cc:system", "CRM"),
            ("node:cc:reads:tc_customer", "customer information"),
        ],
    ),
    "cq": (
        "I create the quotation using the customer and pricing information "
        "in the quoting system.",
        [
            ("node:cq:activity", "create the quotation"),
            ("node:cq:actor", "I"),
            ("node:cq:system", "quoting system"),
            ("node:cq:reads:tc_customer", "customer"),
            ("node:cq:reads:tc_pricing", "pricing information"),
            ("node:cq:writes:tc_quote", "quotation"),
        ],
    ),
    "ap": (
        "Quotations over 1,000,000 yen are approved by the manager; the "
        "approval is for credit risk management.",
        [
            ("node:ap:activity", "approved"),
            ("node:ap:actor", "manager"),
            ("node:ap:rationale", "credit risk management"),
        ],
    ),
    "sq": (
        "I send the quotation to the customer by email.",
        [
            ("node:sq:activity", "send the quotation"),
            ("node:sq:actor", "I"),
            ("node:sq:system", "email"),
        ],
    ),
    "me": (
        "At month-end I send the quotation information summary to Accounting "
        "as an Excel file.",
        [
            ("node:me:activity", "send the quotation information summary"),
            ("node:me:actor", "I"),
            ("node:me:system", "Excel"),
            ("node:me:writes:tc_excel_summary", "summary"),
        ],
    ),
}

_EDGE_OBS = {
    "e1": (
        "After receiving the request, I check the customer information.",
        [("edge:e1", "After receiving the request, I check")],
    ),
    "e2": (
        "After checking the customer information, I create the quotation.",
        [("edge:e2", "After checking the customer information, I create")],
    ),
    "e3": (
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [
            ("edge:e3", "go to the manager for approval"),
            ("edge:e3:condition", "over 1,000,000 yen"),
        ],
    ),
    "e4": (
        "Quotations at or below 1,000,000 yen are sent directly to the customer.",
        [
            ("edge:e4", "sent directly to the customer"),
            ("edge:e4:condition", "at or below 1,000,000 yen"),
        ],
    ),
    "e5": (
        "Once approved, the quotation is sent to the customer.",
        [("edge:e5", "Once approved, the quotation is sent")],
    ),
    "e6": (
        "At month-end I also send the summary to Accounting.",
        [
            ("edge:e6", "send the summary to Accounting"),
            ("edge:e6:condition", "month-end"),
        ],
    ),
}

# Agent-local glossary ids (kind, label) per TRUTH concept id (the agent
# ids are agent-local; the knowledge concept ids are opaque scenario ids).
_AGENT_CONCEPTS = {
    "tc_activity_receive_request": (
        "act_receive",
        "activity",
        "receive the quotation request",
    ),
    "tc_activity_check_customer": (
        "act_check",
        "activity",
        "check the customer information",
    ),
    "tc_activity_create_quotation": ("act_create", "activity", "create the quotation"),
    "tc_activity_approve_quotation": (
        "act_approve",
        "activity",
        "approve the high-value quotation",
    ),
    "tc_activity_send_quotation": ("act_send", "activity", "send the quotation"),
    "tc_activity_send_month_end_summary": (
        "act_me",
        "activity",
        "send the month-end summary",
    ),
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
    # lab sample flow
    "tc_activity_accession": ("act_accession", "activity", "specimen accession"),
    "tc_activity_seasoning": ("act_seasoning", "activity", "chamber seasoning"),
    "tc_activity_conditioning": ("act_conditioning", "activity", "conditioning cycle"),
    "tc_activity_batch_approval": (
        "act_approval",
        "activity",
        "approve conditioned batch",
    ),
    "tc_actor_lab_tech": ("lab_tech", "actor", "lab tech"),
    "tc_actor_lab_supervisor": ("lab_supervisor", "actor", "lab supervisor"),
    "tc_system_chamber": ("chamber", "system", "environment chamber"),
    "tc_sample": ("sample", "data", "sample"),
}

# TRUTH concept id per semantic id (semantic ids keyed by Truth ids; the
# knowledge side uses opaque local ids — see ``_local_sid``).
_SEMANTIC_TO_CONCEPT = {
    "node:r:activity": "tc_activity_receive_request",
    "node:r:actor": "tc_actor_sales",
    "node:r:writes:tc_request": "tc_request",
    "node:cc:activity": "tc_activity_check_customer",
    "node:cc:actor": "tc_actor_sales",
    "node:cc:system": "tc_system_crm",
    "node:cc:reads:tc_customer": "tc_customer",
    "node:cq:activity": "tc_activity_create_quotation",
    "node:cq:actor": "tc_actor_sales",
    "node:cq:system": "tc_system_quoting",
    "node:cq:reads:tc_customer": "tc_customer",
    "node:cq:reads:tc_pricing": "tc_pricing",
    "node:cq:writes:tc_quote": "tc_quote",
    "node:ap:activity": "tc_activity_approve_quotation",
    "node:ap:actor": "tc_actor_manager",
    "node:ap:rationale": "tc_rationale_credit_risk",
    "node:sq:activity": "tc_activity_send_quotation",
    "node:sq:actor": "tc_actor_sales",
    "node:sq:system": "tc_system_email",
    "node:me:activity": "tc_activity_send_month_end_summary",
    "node:me:actor": "tc_actor_sales",
    "node:me:system": "tc_system_excel",
    "node:me:writes:tc_excel_summary": "tc_excel_summary",
    "edge:e3:condition": "tc_cond_over_1m",
    "edge:e4:condition": "tc_cond_at_or_below_1m",
    "edge:e6:condition": "tc_cond_month_end",
}


# Stakeholder DONT_KNOW speech: (text, [(truth semantic id, quote)]) — the
# sales stakeholder does not know these properties of existing elements.
_NODE_DONT_KNOW_OBS = {
    "r": (
        "I don't know which system this step uses, what data it reads, or "
        "why it is necessary.",
        [
            ("node:r:system", "which system this step uses"),
            ("node:r:reads", "what data it reads"),
            ("node:r:rationale", "why it is necessary"),
        ],
    ),
    "cc": (
        "I don't know what data it writes or why the check is necessary.",
        [
            ("node:cc:writes", "what data it writes"),
            ("node:cc:rationale", "why the check is necessary"),
        ],
    ),
    "cq": (
        "I don't know why the creation is necessary.",
        [("node:cq:rationale", "why the creation is necessary")],
    ),
    "ap": (
        "I don't know which system the approval uses, what it reads, or "
        "what it writes.",
        [
            ("node:ap:system", "which system the approval uses"),
            ("node:ap:reads", "what it reads"),
            ("node:ap:writes", "what it writes"),
        ],
    ),
    "sq": (
        "I don't know what data the send step reads or writes, or why it is necessary.",
        [
            ("node:sq:reads", "what data the send step reads"),
            ("node:sq:writes", "or writes"),
            ("node:sq:rationale", "why it is necessary"),
        ],
    ),
    "me": (
        "I don't know what data the summary step reads or why it is necessary.",
        [
            ("node:me:reads", "what data the summary step reads"),
            ("node:me:rationale", "why it is necessary"),
        ],
    ),
}

_EDGE_DONT_KNOW_OBS = {
    "e1": (
        "I don't know of any condition before the check.",
        [("edge:e1:condition", "any condition before the check")],
    ),
    "e2": (
        "I don't know of any condition before the creation.",
        [("edge:e2:condition", "any condition before the creation")],
    ),
    "e5": (
        "I don't know of any condition after the approval.",
        [("edge:e5:condition", "any condition after the approval")],
    ),
}


def _record_node_dont_know(tools: InterviewTools, sid: str) -> None:
    """Say and record the DONT_KNOW properties of node ``sid`` (truth id)."""
    text, anns = _NODE_DONT_KNOW_OBS[sid]
    oid = _say(tools, text, [_annotation(s, q) for s, q in anns])
    agent_node = {v: k for k, v in _NODE_MAP.items()}[sid]
    tools.record_dont_know(
        agent_node,
        properties=[
            "rationale" if s == "node:%s:rationale" % sid else s.split(":")[2]
            for s, _ in anns
        ],
        evidence=[_ev(oid, q) for _, q in anns],
    )


def _record_edge_dont_know(tools: InterviewTools, eid: str) -> None:
    """Say and record the DONT_KNOW condition of edge ``eid`` (truth id)."""
    text, anns = _EDGE_DONT_KNOW_OBS[eid]
    oid = _say(tools, text, [_annotation(s, q) for s, q in anns])
    tools.record_edge_condition_dont_know(eid, evidence=[_ev(oid, q) for _, q in anns])


def _tools(scenario_id: str = SCENARIO) -> InterviewTools:
    """Tools with the scenario's StakeholderKnowledgeCatalog installed (the
    binding-aware tools need it to resolve evidence)."""
    tools = InterviewTools(InterviewDB())
    tools.assertion_ledger.install_catalog(
        StakeholderKnowledgeCatalog.from_scenario(_sc(scenario_id))
    )
    return tools


def _ingest(tools: InterviewTools, role: str = "user", content: str = "") -> int:
    tools.db.messages.append({"role": role, "content": content})
    return len(tools.db.messages) - 1


def _annotation(semantic_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"semantic_id": semantic_id, "quote": quote, "occurrence": occurrence}


def _say(
    tools: InterviewTools,
    text: str,
    annotations: Optional[list[dict]] = None,
    alignments: Optional[list[dict]] = None,
    terminology: Optional[list[dict]] = None,
    scenario_id: str = SCENARIO,
) -> str:
    """Ingest a stakeholder message, bind its private sidecar (semantic
    annotations + optional dialogue events) at that exact turn, and capture
    it as an Observation. Semantic ids in the sidecar are translated from
    Truth-id form to the scenario's opaque stakeholder-local ids."""
    turn = _ingest(tools, "user", text)
    if annotations:
        tools.assertion_ledger.bind(
            turn,
            [
                SemanticAnnotation(
                    semantic_id=_local_sid(a["semantic_id"], scenario_id),
                    quote=a["quote"],
                    occurrence=a.get("occurrence", 0),
                )
                for a in annotations
            ],
            text,
        )
    if alignments:
        tools.assertion_ledger.bind_alignment(
            turn,
            [
                ConceptAlignmentAssertion(
                    semantic_id=_local_sid(a["semantic_id"], scenario_id),
                    quote=a["quote"],
                    occurrence=a.get("occurrence", 0),
                    act=a["act"],
                )
                for a in alignments
            ],
            text,
        )
    if terminology:
        tools.assertion_ledger.bind_terminology(
            turn,
            [
                TerminologyConfirmation(
                    semantic_id=_local_sid(t["semantic_id"], scenario_id),
                    proposed_term=t["proposed_term"],
                    quote=t["quote"],
                    occurrence=t.get("occurrence", 0),
                )
                for t in terminology
            ],
            text,
        )
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


def _ev(obs_id: str, quote: str, occurrence: int = 0) -> dict:
    return {"observation_id": obs_id, "quote": quote, "occurrence": occurrence}


def _evr(obs_id: str, quote: str, occurrence: int = 0) -> EvidenceRef:
    """Direct EvidenceRef (for graph mutation); tools accept dicts via _ev."""
    return EvidenceRef(observation_id=obs_id, quote=quote, occurrence=occurrence)


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    """Evaluate under the scenario's StakeholderKnowledge + private semantic
    provenance (runtime behavior: the sidecar ledger of the tools)."""
    sc = get_scenario(scenario)
    assert sc is not None
    return evaluate(
        tools.db,
        sc.knowledge,
        EvaluationSpec(),
        sc.stakeholder,
        truth=sc.truth,
        annotations=tools.assertion_ledger.annotations(),
        alignments=tools.assertion_ledger.alignments(),
        terminology=tools.assertion_ledger.terminology(),
    )


def _make_concept(
    tools: InterviewTools,
    knowledge_cid: str,
    oid: str,
    quote: str,
) -> str:
    """Create (and ground) the agent concept for a knowledge concept id using
    the given observation span, returning the agent concept id."""
    agent_cid, kind, _label = _AGENT_CONCEPTS[knowledge_cid]
    tools.create_concept(agent_cid, kind, _label, evidence=[_ev(oid, quote)])
    tools.ground_concept(agent_cid, evidence=[_ev(oid, quote)])
    return agent_cid


def _quote_for_concept(node_obs, knowledge_cid: str) -> tuple[str, str]:
    for sid, (text, anns) in node_obs.items():
        for semantic_id, q in anns:
            if _SEMANTIC_TO_CONCEPT.get(semantic_id) == knowledge_cid:
                return sid, q
    raise KeyError(knowledge_cid)


def _prop(kcid: str, oid: str, quote: str) -> dict:
    """A property argument with its own evidence."""
    return {"concept_id": _AGENT_CONCEPTS[kcid][0], "evidence": [_ev(oid, quote)]}


def _build(tools: InterviewTools, ja: bool = False) -> None:
    """Build the correct quotation AgentGraph with full provenance: every
    property reference carries its own evidence, every concept is grounded
    with authentic annotation-corresponding spans, every stakeholder
    DONT_KNOW slot is recorded explicitly — so completion succeeds and the
    evaluator returns a full pass."""
    from tau2.domains.business_interview.facts import StakeholderKnowledgeCatalog

    tools.assertion_ledger.install_catalog(
        StakeholderKnowledgeCatalog.from_scenario(_sc(SCENARIO))
    )
    tools.start_inference("quotation")
    _ingest(tools, "assistant", "Hello.")
    node_obs = _NODE_OBS
    edge_obs = _EDGE_OBS
    created: dict[str, str] = {}
    node_oids: dict[str, str] = {}
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, anns = node_obs[sid]
        oid = _say(tools, text, [_annotation(sid_, q) for sid_, q in anns])
        node_oids[sid] = oid

    # concepts first (activity per node; actor/system/data per knowledge
    # concept), each grounded with authentic annotation-corresponding spans
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        kcid = _SEMANTIC_TO_CONCEPT[f"node:{sid}:activity"]
        oid = node_oids[sid]
        quote = next(
            q for sid_, q in node_obs[sid][1] if sid_ == f"node:{sid}:activity"
        )
        created[kcid] = _make_concept(tools, kcid, oid, quote)
    for kcid in (
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
        sid, quote = _quote_for_concept(node_obs, kcid)
        created[kcid] = _make_concept(tools, kcid, node_oids[sid], quote)

    for anid, sid in _NODE_MAP.items():
        oid = node_oids[sid]
        anns = node_obs[sid][1]
        args: dict = {
            "node_id": anid,
            "activity": _prop(
                _SEMANTIC_TO_CONCEPT[f"node:{sid}:activity"],
                oid,
                next(q for s, q in anns if s == f"node:{sid}:activity"),
            ),
        }
        for prop, kcid in (
            ("actor", "tc_actor_sales"),
            ("actor", "tc_actor_manager"),
            ("system", "tc_system_crm"),
            ("system", "tc_system_quoting"),
            ("system", "tc_system_email"),
            ("system", "tc_system_excel"),
            ("necessity_rationale", "tc_rationale_credit_risk"),
        ):
            for semantic_id, q in anns:
                if _SEMANTIC_TO_CONCEPT.get(semantic_id) == kcid:
                    args[prop] = _prop(kcid, oid, q)
        reads = []
        writes = []
        for semantic_id, q in anns:
            kcid = _SEMANTIC_TO_CONCEPT.get(semantic_id)
            if kcid is None:
                continue
            if semantic_id.startswith(f"node:{sid}:reads:"):
                reads.append(_prop(kcid, oid, q))
            elif semantic_id.startswith(f"node:{sid}:writes:"):
                writes.append(_prop(kcid, oid, q))
        if reads:
            args["reads"] = reads
        if writes:
            args["writes"] = writes
        tools.add_node(**args)

    # the stakeholder does NOT know several properties of known elements —
    # record explicit, evidenced DONT_KNOW on every such slot (an unasserted
    # slot does NOT count as DONT_KNOW)
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        _record_node_dont_know(tools, sid)

    # edges
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, anns = edge_obs[eid]
        oid = _say(tools, text, [_annotation(sid_, q) for sid_, q in anns])
        frm, to = {
            "e1": ("a", "b"),
            "e2": ("b", "c"),
            "e3": ("c", "d"),
            "e4": ("c", "e"),
            "e5": ("d", "e"),
            "e6": ("c", "f"),
        }[eid]
        cond = None
        for semantic_id, q in anns:
            if semantic_id == f"edge:{eid}:condition":
                kcid = _SEMANTIC_TO_CONCEPT[semantic_id]
                cond = _make_concept(tools, kcid, oid, q)
                cond = _prop(kcid, oid, q)
        tools.add_edge(
            eid,
            frm,
            to,
            condition=cond,
            evidence=[_ev(oid, q) for sid_, q in anns if sid_ == f"edge:{eid}"],
        )
    for eid in ("e1", "e2", "e5"):
        _record_edge_dont_know(tools, eid)
    tools.set_graph_endpoints(start_node_id="a", end_node_ids=["e", "f"])
    tools.finish_interview()


def _build_lab(tools: InterviewTools) -> None:
    """Build a correct lab AgentGraph asserting ONLY known properties (and
    recording every stakeholder DONT_KNOW slot explicitly) with full
    provenance and grounded concepts."""
    from tau2.domains.business_interview.facts import StakeholderKnowledgeCatalog

    tools.assertion_ledger.install_catalog(
        StakeholderKnowledgeCatalog.from_scenario(_sc(LAB_SCENARIO))
    )
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        (
            "n1",
            "When a specimen arrives, I accession it and record it as received.",
            [
                ("node:n1:activity", "accession"),
                ("node:n1:actor", "I"),
                ("node:n1:reads:tc_sample", "specimen"),
            ],
        ),
        (
            "n2",
            "I season the environment chamber to prepare it.",
            [
                ("node:n2:activity", "season"),
                ("node:n2:actor", "I"),
                ("node:n2:system", "environment chamber"),
            ],
        ),
        (
            "n3",
            "I run a conditioning cycle that processes the samples inside the chamber.",
            [
                ("node:n3:activity", "conditioning cycle"),
                ("node:n3:actor", "I"),
                ("node:n3:system", "chamber"),
            ],
        ),
        (
            "n4",
            "The lab supervisor approves the conditioned batch before it is released.",
            [
                ("node:n4:activity", "approves"),
                ("node:n4:actor", "lab supervisor"),
            ],
        ),
    ]
    created: dict[str, str] = {}
    oids: dict[str, str] = {}
    for sid, text, anns in nodes:
        oid = _say(
            tools, text, [_annotation(s, q) for s, q in anns], scenario_id=LAB_SCENARIO
        )
        oids[sid] = oid
        act_kcid = {
            "n1": "tc_activity_accession",
            "n2": "tc_activity_seasoning",
            "n3": "tc_activity_conditioning",
            "n4": "tc_activity_batch_approval",
        }[sid]
        quote = next(q for s, q in anns if s == f"node:{sid}:activity")
        created[f"act_{sid}"] = _make_concept(tools, act_kcid, oid, quote)
    for sid, text, anns in nodes:
        oid = oids[sid]
        args: dict = {
            "node_id": sid,
            "activity": _prop(
                {
                    "n1": "tc_activity_accession",
                    "n2": "tc_activity_seasoning",
                    "n3": "tc_activity_conditioning",
                    "n4": "tc_activity_batch_approval",
                }[sid],
                oid,
                next(q for s, q in anns if s == f"node:{sid}:activity"),
            ),
        }
        for semantic_id, q in anns:
            if semantic_id == f"node:{sid}:actor":
                kcid = "tc_actor_lab_tech" if sid != "n4" else "tc_actor_lab_supervisor"
                if kcid not in created:
                    created[kcid] = _make_concept(tools, kcid, oid, q)
                args["actor"] = _prop(kcid, oid, q)
            elif semantic_id == f"node:{sid}:system":
                kcid = "tc_system_chamber"
                if kcid not in created:
                    created[kcid] = _make_concept(tools, kcid, oid, q)
                args["system"] = _prop(kcid, oid, q)
            elif semantic_id.startswith(f"node:{sid}:reads:"):
                kcid = semantic_id.rsplit(":", 1)[1]
                if kcid not in created:
                    created[kcid] = _make_concept(tools, kcid, oid, q)
                args.setdefault("reads", []).append(_prop(kcid, oid, q))
        tools.add_node(**args)
    edges = [
        (
            "l1",
            "n1",
            "n2",
            "After specimen accession, I prepare the chamber for seasoning.",
            [("edge:l1", "After specimen accession, I prepare")],
        ),
        (
            "l2",
            "n2",
            "n3",
            "After chamber seasoning, I run the conditioning cycle.",
            [("edge:l2", "After chamber seasoning, I run")],
        ),
        (
            "l3",
            "n3",
            "n4",
            "After the conditioning cycle, the lab supervisor approves the batch.",
            [("edge:l3", "After the conditioning cycle, the lab supervisor approves")],
        ),
    ]
    for eid, frm, to, text, anns in edges:
        oid = _say(
            tools, text, [_annotation(s, q) for s, q in anns], scenario_id=LAB_SCENARIO
        )
        tools.add_edge(eid, frm, to, evidence=[_ev(oid, q) for s, q in anns])
    # the lab tech does NOT know most read/write artifacts or rationales —
    # record every DONT_KNOW slot explicitly (restraint is not an omission)
    lab_dont_know = [
        (
            "n1",
            "I don't know which system the step uses, what it writes, or why "
            "it is necessary.",
            [
                ("node:n1:system", "which system the step uses"),
                ("node:n1:writes", "what it writes"),
                ("node:n1:rationale", "why it is necessary"),
            ],
        ),
        (
            "n2",
            "I don't know what the seasoning reads or writes, or why it is necessary.",
            [
                ("node:n2:reads", "what the seasoning reads"),
                ("node:n2:writes", "or writes"),
                ("node:n2:rationale", "why it is necessary"),
            ],
        ),
        (
            "n3",
            "I don't know what the cycle reads or writes, or why it is necessary.",
            [
                ("node:n3:reads", "what the cycle reads"),
                ("node:n3:writes", "or writes"),
                ("node:n3:rationale", "why it is necessary"),
            ],
        ),
        (
            "n4",
            "I don't know which system the approval uses, what it reads or "
            "writes, or why it is necessary.",
            [
                ("node:n4:system", "which system the approval uses"),
                ("node:n4:reads", "what it reads"),
                ("node:n4:writes", "or writes"),
                ("node:n4:rationale", "why it is necessary"),
            ],
        ),
    ]
    for nid, text, anns in lab_dont_know:
        oid = _say(
            tools, text, [_annotation(s, q) for s, q in anns], scenario_id=LAB_SCENARIO
        )
        tools.record_dont_know(
            nid,
            properties=[
                "rationale" if s == f"node:{nid}:rationale" else s.split(":")[2]
                for s, _ in anns
            ],
            evidence=[_ev(oid, q) for _, q in anns],
        )
    for eid in ("l1", "l2", "l3"):
        text = f"I don't know of any condition on the way from {eid}."
        oid = _say(
            tools,
            text,
            [
                _annotation(
                    f"edge:{eid}:condition", f"any condition on the way from {eid}"
                )
            ],
            scenario_id=LAB_SCENARIO,
        )
        tools.record_edge_condition_dont_know(
            eid, evidence=[_ev(oid, f"any condition on the way from {eid}")]
        )
    tools.set_graph_endpoints(start_node_id="n1", end_node_ids=["n4"])
    tools.finish_interview()


# ---------------------------------------------------------------------------
# Architecture: Truth = graph + concepts; claims are gone
# ---------------------------------------------------------------------------


def test_truthclaim_and_claim_catalogs_are_gone():
    import pathlib

    from tau2.domains.business_interview import evaluation, facts, scenario

    domain_dir = pathlib.Path(evaluation.__file__).parent
    assert not (domain_dir / "claims.py").exists()
    for module in (evaluation, facts, scenario):
        for name in ("TruthClaim", "build_claims", "visible_claim_ids"):
            assert not hasattr(module, name), f"{module.__name__} still has {name}"
    for src in domain_dir.glob("*.py"):
        text = src.read_text(encoding="utf-8")
        assert "TruthClaim" not in text, src.name
        assert "build_claims" not in text, src.name
        assert "visible_claim_ids" not in text, src.name
    # no scenario claim catalog anywhere
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert not hasattr(sc, "claims")


def test_no_stakeholder_semantic_assertion_type():
    from tau2.domains.business_interview import facts

    assert not hasattr(facts, "StakeholderSemanticAssertion")
    assert not hasattr(facts, "StakeholderAssertion")
    assert hasattr(facts, "SemanticAnnotation")


def test_truth_contains_only_graph_and_concepts():
    sc = get_scenario(SCENARIO)
    assert sc is not None
    truth = sc.truth
    assert isinstance(truth, type(quotation_truth()))
    for concept in truth.concepts.values():
        assert isinstance(concept, TruthConcept)
        assert concept.kind in (
            "activity",
            "actor",
            "system",
            "data",
            "condition",
            "rationale",
        )
    # concept descriptions describe the concept, never a workflow position
    assert "node:" not in " ".join(c.description for c in truth.concepts.values())
    assert "edge:" not in " ".join(c.description for c in truth.concepts.values())


# ---------------------------------------------------------------------------
# Stable semantic IDs
# ---------------------------------------------------------------------------


def test_stable_semantic_ids_include_reads_writes_elements():
    truth = quotation_truth()
    ids = graph_semantic_ids(truth.nodes, truth.edges)
    for expected in (
        "node:cq",
        "node:cq:activity",
        "node:cq:reads:tc_customer",
        "node:cq:reads:tc_pricing",
        "node:cq:writes:tc_quote",
        "node:r:writes:tc_request",
        "edge:e3",
        "edge:e3:condition",
        "node:sq:reads",  # slot exists even when known-absent
    ):
        assert expected in ids, expected
    # ids survive reordering: swap the reads list -> same id set
    cq = truth.nodes["cq"]
    assert isinstance(cq.reads, list)
    cq.reads.reverse()
    ids2 = graph_semantic_ids(truth.nodes, truth.edges)
    assert ids == ids2
    # element ids are not list indexes
    assert "reads:0" not in ids and "reads:1" not in ids


def test_conceptref_none_dont_know_are_distinct():
    assert DONT_KNOW is not None
    assert ConceptRef(concept_id="x") is not None
    assert is_dont_know(DONT_KNOW)
    assert not is_dont_know(None)
    assert not is_dont_know(ConceptRef(concept_id="x"))
    assert DONT_KNOW != ConceptRef(concept_id="x")
    assert DONT_KNOW != None  # noqa: E711 - the three-valued distinction


def test_removal_creates_no_shortcut_edge():
    truth = type(quotation_truth())(
        id="t",
        nodes={
            "A": Node(id="A", activity=ConceptRef(concept_id="a1")),
            "B": Node(id="B", activity=ConceptRef(concept_id="a2")),
            "C": Node(id="C", activity=ConceptRef(concept_id="a3")),
        },
        edges={
            "ab": Edge(id="ab", from_node="A", to_node="B"),
            "bc": Edge(id="bc", from_node="B", to_node="C"),
        },
        concepts={
            "a1": TruthConcept(id="a1", kind="activity"),
            "a2": TruthConcept(id="a2", kind="activity"),
            "a3": TruthConcept(id="a3", kind="activity"),
        },
        start_node_id="A",
        end_node_ids=["C"],
    )
    filter_ = StakeholderFilter(
        name="partial",
        visible_node_ids=["A", "C"],  # B unknown -> removed
        visible_edge_ids=["ab", "bc"],
        visible_node_attributes={"A": ["activity"], "C": ["activity"]},
        visible_edge_attributes={"ab": [], "bc": []},
    )
    knowledge = project_knowledge(truth, filter_)
    g = knowledge.graph
    node_t2l = {t: k for k, t in g.node_truth_ids.items()}
    assert set(g.nodes) == {node_t2l["A"], node_t2l["C"]}
    assert set(g.edges) == set()
    # never a shortcut A -> C
    assert not any(
        e.from_node == node_t2l["A"] and e.to_node == node_t2l["C"]
        for e in g.edges.values()
    )
    # start survives when the start node is known; unknown end is dropped
    assert g.start_node_id == node_t2l["A"]
    assert g.end_node_ids == [node_t2l["C"]]


def test_knowledge_masking_three_valued():
    """Known value -> ConceptRef; known absent -> None; unknown -> DONT_KNOW."""
    truth = quotation_truth()
    filter_ = StakeholderFilter(
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
            "e1": ["condition"],
            "e2": [],
            "e3": ["condition"],
            "e4": ["condition"],
            "e5": [],
            "e6": ["condition"],
        },
    )
    knowledge = project_knowledge(truth, filter_)
    g = knowledge.graph
    concept_t2l = {c.truth_concept_id: k for k, c in g.concepts.items()}
    node_t2l = {t: k for k, t in g.node_truth_ids.items()}
    edge_t2l = {t: k for k, t in g.edge_truth_ids.items()}
    cq = g.nodes[node_t2l["cq"]]
    assert isinstance(cq.system, ConceptRef)  # known value
    assert cq.system.concept_id == concept_t2l["tc_system_quoting"]
    assert isinstance(cq.reads, list)  # known list
    assert cq.reads[0].concept_id == concept_t2l["tc_customer"]
    sq = g.nodes[node_t2l["sq"]]
    assert is_dont_know(sq.reads)  # unknown property -> DONT_KNOW
    assert is_dont_know(sq.writes)
    # e1's condition is KNOWN and the Truth has none -> None (known absent)
    assert g.edges[edge_t2l["e1"]].condition is None
    # e2's condition is not known -> DONT_KNOW
    assert is_dont_know(g.edges[edge_t2l["e2"]].condition)
    assert is_dont_know(g.nodes[node_t2l["me"]].necessity_rationale)
    assert is_dont_know(g.nodes[node_t2l["r"]].reads)


def test_description_and_terminology_vary_independently():
    truth = quotation_truth()
    base = StakeholderFilter(
        name="s",
        visible_node_ids=["r"],
        visible_edge_ids=[],
        visible_node_attributes={"r": ["activity"]},
        visible_edge_attributes={},
    )
    # both known (defaults)
    k1 = project_knowledge(truth, base)
    c1 = k1.graph.concepts[
        {c.truth_concept_id: k for k, c in k1.graph.concepts.items()}[  # noqa: SIM118
            "tc_activity_receive_request"
        ]
    ]
    assert c1.has_description() and c1.has_terms()
    assert c1.truth_concept_id == "tc_activity_receive_request"
    # term known, details unknown
    k2 = project_knowledge(
        truth,
        base.model_copy(
            update={
                "concept_overrides": {
                    "tc_activity_receive_request": ConceptKnowledgeOverride(
                        description_known=False
                    )
                }
            }
        ),
    )
    c2 = k2.graph.concepts[
        {c.truth_concept_id: k for k, c in k2.graph.concepts.items()}[  # noqa: SIM118
            "tc_activity_receive_request"
        ]
    ]
    assert not c2.has_description() and c2.has_terms()
    # details known, local/wrong term
    k3 = project_knowledge(
        truth,
        base.model_copy(
            update={
                "concept_overrides": {
                    "tc_activity_receive_request": ConceptKnowledgeOverride(
                        local_terms=["take the order"]
                    )
                }
            }
        ),
    )
    c3 = k3.graph.concepts[
        {c.truth_concept_id: k for k, c in k3.graph.concepts.items()}[  # noqa: SIM118
            "tc_activity_receive_request"
        ]
    ]
    assert c3.has_description() and c3.terms == ["take the order"]
    # details known, term unknown
    k4 = project_knowledge(
        truth,
        base.model_copy(
            update={
                "concept_overrides": {
                    "tc_activity_receive_request": ConceptKnowledgeOverride(
                        terms_known=False
                    )
                }
            }
        ),
    )
    c4 = k4.graph.concepts[
        {c.truth_concept_id: k for k, c in k4.graph.concepts.items()}[  # noqa: SIM118
            "tc_activity_receive_request"
        ]
    ]
    assert c4.has_description() and not c4.has_terms()
    # neither known
    k5 = project_knowledge(
        truth,
        base.model_copy(
            update={
                "concept_overrides": {
                    "tc_activity_receive_request": ConceptKnowledgeOverride(
                        description_known=False, terms_known=False
                    )
                }
            }
        ),
    )
    c5 = k5.graph.concepts[
        {c.truth_concept_id: k for k, c in k5.graph.concepts.items()}[  # noqa: SIM118
            "tc_activity_receive_request"
        ]
    ]
    assert not c5.has_description() and not c5.has_terms()


def test_hidden_concepts_never_enter_knowledge_or_prompt():
    from tau2.domains.business_interview.user_simulator import (
        StakeholderUserSimulator,
    )

    sc = get_scenario(LAB_SCENARIO)
    assert sc is not None
    kg = sc.knowledge.graph
    concepts = kg.concepts
    hidden_tids = (
        "tc_accessioned_sample",
        "tc_seasoned_chamber",
        "tc_conditioned_sample",
        "tc_batch_approval",
    )
    assert not any(c.truth_concept_id in hidden_tids for c in concepts.values())
    # no knowledge concept id encodes a Truth name
    for c in concepts.values():
        assert "accessioned" not in c.id and "seasoned" not in c.id
        assert "conditioned" not in c.id and "approval" not in c.id
    # hidden slots are DONT_KNOW in the knowledge graph
    node_t2l = {t: k for k, t in kg.node_truth_ids.items()}
    assert is_dont_know(kg.nodes[node_t2l["n2"]].writes)
    # the prompt never exposes hidden Truth ids or canonical terms
    env = get_environment()
    task = next(t for t in get_tasks() if t.id == LAB_SCENARIO)
    sim = StakeholderUserSimulator(
        llm="dummy", task=task, environment=env, instructions="x"
    )
    block = sim._knowledge_block()  # noqa: SLF001 - test-only
    for forbidden in (
        "tc_accessioned_sample",
        "tc_seasoned_chamber",
        "tc_conditioned_sample",
        "tc_batch_approval",
        "accessioned sample",
        "seasoned chamber",
        "conditioned sample",
        "batch approval",
    ):
        assert forbidden not in block, forbidden
    # the opaque knowledge concept ids and graph semantic ids ARE in the
    # prompt (no Truth ids, labels or terms)
    concept_t2l = {c.truth_concept_id: k for k, c in concepts.items()}
    assert concept_t2l["tc_sample"] in block
    assert f"node:{node_t2l['n1']}:reads:{concept_t2l['tc_sample']}" in block


# ---------------------------------------------------------------------------
# Provenance: Observation spans -> semantic IDs
# ---------------------------------------------------------------------------


def test_observation_spans_resolve_directly_to_semantic_ids():
    """Private annotations point directly at stakeholder semantic IDs — no
    subject/property/value duplication."""
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    _say(
        tools,
        "I check customer information in CRM.",
        [
            _annotation("node:cc:activity", "check customer information"),
            _annotation("node:cc:reads:tc_customer", "customer information"),
            _annotation("node:cc:system", "CRM"),
        ],
    )
    assert tools.assertion_ledger.annotations() != {}
    # deterministic validation: unknown semantic id rejected at ingestion
    # (an opaque-looking local id that does not exist in the knowledge graph)
    tools2 = _tools()
    tools2.start_inference("q")
    _ingest(tools2, "assistant", "Hello.")
    with pytest.raises(ValueError):
        _say(
            tools2,
            "I check customer information in CRM.",
            [_annotation("node:skn_099:activity", "check customer information")],
        )
    # a property ref whose evidence resolves to the activity semantic id
    # grounds it (full-build below proves the end-to-end path)


def test_edge_and_condition_separately_addressable():
    """edge:e3 and edge:e3:condition are distinct semantic ids; grounding one
    does not ground the other."""
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid = _say(
        tools,
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [
            _annotation("edge:e3", "go to the manager for approval"),
            _annotation("edge:e3:condition", "over 1,000,000 yen"),
        ],
    )
    sc = get_scenario(SCENARIO)
    assert sc is not None

    grounded, invalid, amb = grounded_semantic_ids(
        tools.db,
        tools.assertion_ledger.annotations(),
        [_evr(oid, "go to the manager for approval")],
    )
    assert grounded == {_local_sid("edge:e3")} and invalid == 0 and amb == 0
    grounded2, _, _ = grounded_semantic_ids(
        tools.db,
        tools.assertion_ledger.annotations(),
        [_evr(oid, "over 1,000,000 yen")],
    )
    assert grounded2 == {_local_sid("edge:e3:condition")}
    # a span covering BOTH is globally ambiguous
    grounded3, _, amb3 = grounded_semantic_ids(
        tools.db,
        tools.assertion_ledger.annotations(),
        [_evr(oid, "over 1,000,000 yen go to the manager for approval")],
    )
    assert grounded3 == set() and amb3 == 1


def test_property_scoring_uses_property_evidence_only():
    """Mentions and validation evidence never enter property scoring: a ref
    whose ONLY backing is the concept's mention/validation evidence is
    unsupported."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [_annotation("node:cc:system", "CRM")],
    )
    tools.create_concept("bogus_sys", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.ground_concept("bogus_sys", evidence=[_ev(oid, "CRM")])
    # system ref with NO property evidence of its own
    tools.db.graph.nodes["b"].system = ConceptRef(
        concept_id="bogus_sys", confidence=1.0
    )
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.unsupported_ref_count >= 1
    assert res.ambiguous_evidence_ref_count == 0


def test_broad_clause_cannot_cross_credit_semantic_ids():
    """A broad clause covering several semantic ids grounds nothing (global
    ambiguity), even when each scoring call examines a different slot."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [
            _annotation("node:cc:activity", "check the customer information"),
            _annotation("node:cc:system", "CRM"),
            _annotation("node:cc:reads:tc_customer", "customer information"),
        ],
    )
    tools.db.graph.nodes["b"].system = ConceptRef(
        concept_id="crm",
        confidence=1.0,
        evidence=[_evr(oid, "check the customer information in the CRM")],
    )
    res = _eval(tools)
    # the broad span covers activity+system+reads -> ambiguous -> no credit
    assert res.system_correctness < 1.0
    assert res.ambiguous_evidence_ref_count >= 1


def test_invalid_annotation_quote_rejected_at_ingestion():
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
                stakeholder_annotations=[_annotation("node:cc:system", "CRM")],
            )
        )
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_annotations=[
                    _annotation("node:cc:system", "CRM", occurrence=3)
                ],
            )
        )
    with pytest.raises(ValueError):
        env.on_message(
            UserMessage(
                role="user",
                content="I check the customer in the CRM.",
                stakeholder_annotations=[_annotation("node:zz:system", "CRM")],
            )
        )


# ---------------------------------------------------------------------------
# Full reconstruction
# ---------------------------------------------------------------------------


def test_valid_full_graph_passes():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.graph_valid is True
    assert res.structural_pass is True
    assert res.evidence_pass is True
    assert res.quality_pass is True
    assert res.knowledge_coverage > 0.0


def test_knowledge_coverage_metric_separate():
    """Truth vs StakeholderKnowledge coverage is reported separately and is
    not part of Agent performance."""
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert 0.0 < res.knowledge_coverage < 1.0  # hidden slots lower coverage
    # even a failing agent run keeps the same coverage (scenario-level)
    bad = _tools()
    bad.start_inference("q")
    res_bad = _eval(bad)
    assert res_bad.knowledge_coverage == res.knowledge_coverage


def test_missing_node_lowers_recall():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    del tools.db.graph.nodes["f"]
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.structural_pass is False


def test_fabricated_node_stays_unmapped():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _say(tools, "I like pizza on Fridays.")
    tools.create_concept(
        "pizza_act", "activity", "eat pizza", evidence=[_ev(pizza, "pizza")]
    )
    tools.add_node("fab", activity="pizza_act", evidence=[_ev(pizza, "pizza")])
    res = _eval(tools)
    assert res.node_precision < 1.0
    assert res.fabricated_node_count >= 1
    assert res.structural_pass is False


def test_wrong_actor_drops_actor_correctness():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [_annotation("node:cc:actor", "I")],
    )
    tools.create_concept("manager_wrong", "actor", "manager", evidence=[_ev(oid, "I")])
    tools.ground_concept("manager_wrong", evidence=[_ev(oid, "I")])
    tools.db.graph.nodes["b"].actor = ConceptRef(
        concept_id="manager_wrong",
        confidence=1.0,
        evidence=[_evr(oid, "I")],
    )
    res = _eval(tools)
    assert res.actor_correctness < 1.0
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
# Epistemic restraint
# ---------------------------------------------------------------------------


def test_hidden_truth_guesses_remain_wrong():
    """DONT_KNOW slots: any assertion is wrong; invented elements are
    precision errors."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # sq reads/writes are DONT_KNOW for the sales stakeholder
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [_annotation("node:cc:reads:tc_customer", "customer information")],
    )
    tools.create_concept(
        "guess_data", "data", "something", evidence=[_ev(oid, "customer information")]
    )
    tools.ground_concept("guess_data", evidence=[_ev(oid, "customer information")])
    tools.db.graph.nodes["e"].reads = [
        ConceptRef(
            concept_id="guess_data",
            confidence=1.0,
            evidence=[_evr(oid, "customer information")],
        )
    ]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.structural_pass is False


def test_known_absent_property_asserted_fails():
    """e1's condition is known-absent (None) for the sales stakeholder:
    asserting a condition there is wrong."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    # ground a condition concept with authentic condition-slot evidence
    cond_oid = _say(
        tools,
        "Quotations over 1,000,000 yen go to the manager for approval.",
        [_annotation("edge:e3:condition", "over 1,000,000 yen")],
    )
    tools.create_concept(
        "fake_cond",
        "condition",
        "sometimes",
        evidence=[_ev(cond_oid, "over 1,000,000 yen")],
    )
    tools.ground_concept("fake_cond", evidence=[_ev(cond_oid, "over 1,000,000 yen")])
    # assert it on e1 whose condition the stakeholder knows to be ABSENT
    oid = _say(
        tools,
        "After receiving the request, I check the customer information.",
        [_annotation("edge:e1", "After receiving the request, I check")],
    )
    tools.db.graph.edges["e1"].condition = ConceptRef(
        concept_id="fake_cond",
        confidence=1.0,
        evidence=[_evr(oid, "After receiving the request, I check")],
    )
    res = _eval(tools)
    assert res.condition_correctness < 1.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Glossary: grounded / confirmed / unknown / disputed / terminology
# ---------------------------------------------------------------------------


def test_grounded_concepts_finish_without_confirmation():
    """Concepts resolved via ground_concept (>= grounded) finish the
    interview — explicit confirmation is not required for every concept."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    assert all(
        c.validation_status in ("grounded", "confirmed")
        for c in tools.db.graph.concepts.values()
    )
    assert not any(
        c.validation_status == "confirmed" for c in tools.db.graph.concepts.values()
    )
    res = _eval(tools)
    assert res.glossary_pass is True
    assert res.structural_pass is True


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


def test_ground_concept_requires_private_annotations():
    """Grounding requires the stakeholder's own speech (private semantic
    annotations) — an invented span is not enough."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("c", "data", "thing")
    pizza = _say(tools, "I like pizza on Fridays.")
    with pytest.raises(ValueError):
        tools.ground_concept("c", evidence=[_ev(pizza, "pizza")])
    with pytest.raises(ValueError):
        tools.ground_concept("c", evidence=[_ev("obs_missing", "x")])


def test_ordinary_mention_cannot_confirm_concept():
    """An ordinary workflow mention creates no alignment event, so it cannot
    authorize confirm_concept; a genuine event can."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("quote_c", "data", "quotation")
    oid = _say(
        tools,
        "I create the quotation using the customer information in the quoting system.",
        [_annotation("node:cq:writes:tc_quote", "quotation")],
    )
    with pytest.raises(ValueError):
        tools.confirm_concept("quote_c", evidence=[_ev(oid, "quotation")])
    oid2 = _say(
        tools,
        "Yes.",
        alignments=[{"semantic_id": "tc_quote", "quote": "Yes.", "act": "confirm"}],
    )
    tools.confirm_concept("quote_c", evidence=[_ev(oid2, "Yes.")])
    assert tools.db.graph is not None
    assert tools.db.graph.concepts["quote_c"].validation_status == "confirmed"


def test_unknown_and_disputed_backed_by_events():
    """unknown/disputed need the appropriate private dialogue events and
    resolve completion when referenced."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    dont_know = _say(
        tools,
        "I do not know the reason for that.",
        alignments=[
            {"semantic_id": "tc_pricing", "quote": "do not know", "act": "unknown"}
        ],
    )
    tools.mark_concept_unknown("pricing", evidence=[_ev(dont_know, "do not know")])
    tools.finish_interview()
    res = _eval(tools)
    assert res.glossary_pass is True
    assert res.structural_pass is True

    tools2 = _tools()
    _build(tools2)
    assert tools2.db.graph is not None
    o1 = _say(
        tools2,
        "Actually, they are not the same thing.",
        alignments=[
            {
                "semantic_id": "tc_customer",
                "quote": "not the same thing",
                "act": "dispute",
            }
        ],
    )
    o2 = _say(
        tools2,
        "I keep telling you, those are different.",
        alignments=[
            {"semantic_id": "tc_customer", "quote": "different", "act": "dispute"}
        ],
    )
    tools2.mark_concept_disputed(
        "customer", evidence=[_ev(o1, "not the same thing"), _ev(o2, "different")]
    )
    tools2.finish_interview()
    res2 = _eval(tools2)
    assert res2.glossary_pass is True
    assert res2.structural_pass is True


def test_mention_is_not_terminology():
    """An ordinary authentic mention cannot authorize a terminology
    agreement — only a private terminology-confirmation event (same bound
    knowledge concept + same proposed term + cited span) can."""
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    oid = _say(
        tools,
        "I create the quotation using the customer information in the quoting system.",
        [_annotation("node:cq:writes:tc_quote", "quotation")],
    )
    tools.add_concept_mention("quote", [_ev(oid, "quotation")])
    res_before = _eval(tools)
    assert res_before.structural_pass is True
    # regression: ordinary mention cannot authorize the agreement
    with pytest.raises(ValueError):
        tools.record_terminology_agreement(
            "quote", "the offer document", evidence=[_ev(oid, "quotation")]
        )
    # a genuine private terminology-confirmation event does
    oid_agree = _say(
        tools,
        "Yes, the offer document is fine.",
        terminology=[
            {
                "semantic_id": "tc_quote",
                "proposed_term": "the offer document",
                "quote": "the offer document is fine",
            }
        ],
    )
    tools.record_terminology_agreement(
        "quote",
        "the offer document",
        evidence=[_ev(oid_agree, "the offer document is fine")],
    )
    assert len(tools.db.graph.terminology_agreements) == 1
    res_after = _eval(tools)
    assert res_after.structural_pass is True
    assert res_after.glossary_pass is True


def test_terminology_agreement_requires_matching_event():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("c", "data", "thing")
    with pytest.raises(ValueError):
        tools.record_terminology_agreement("c", "term", evidence=[])
    oid = _say(
        tools,
        "Yes.",
        terminology=[
            {
                "semantic_id": "tc_quote",
                "proposed_term": "the offer document",
                "quote": "Yes.",
            }
        ],
    )
    with pytest.raises(ValueError):
        tools.record_terminology_agreement(
            "c", "something else", evidence=[_ev(oid, "Yes.")]
        )


def test_bulk_self_validation_rejected():
    """One evidence span cannot validate several concepts."""
    tools = _tools()
    tools.start_inference("q")
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [_annotation("node:cc:system", "CRM")],
    )
    tools.create_concept("a", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.create_concept("b", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.ground_concept("a", evidence=[_ev(oid, "CRM")])
    with pytest.raises(ValueError):
        tools.ground_concept("b", evidence=[_ev(oid, "CRM")])


# ---------------------------------------------------------------------------
# Tool-level validation / refinement
# ---------------------------------------------------------------------------


def test_concept_kind_enforcement_in_tools():
    tools = _tools()
    tools.start_inference("q")
    for cid, kind, label in (
        ("act", "activity", "an activity"),
        ("sales", "actor", "sales"),
        ("data_c", "data", "some data"),
        ("sys", "system", "a system"),
        ("cond", "condition", "a condition"),
        ("rat", "rationale", "a rationale"),
    ):
        tools.create_concept(cid, kind, label)
    tools.add_node("n1", activity="act")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="sales")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", actor="data_c")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", system="data_c")
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", reads=["act"])
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", writes=["cond"])
    with pytest.raises(ValueError):
        tools.add_node("n2", activity="act", necessity_rationale="sys")
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
        tools.add_edge("e1", "n1", "n2", condition="data_c")
    tools.add_edge("e1", "n1", "n2", condition="cond")


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


def test_edge_existence_needs_provenance():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    pizza = _say(tools, "I like pizza on Fridays.")
    tools.add_edge("x1", "b", "c", evidence=[_ev(pizza, "pizza")])
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
    assert isinstance(tools.db.graph.nodes["n1"].reads, list)
    assert isinstance(tools.db.graph.nodes["n1"].writes, list)
    assert tools.db.graph.nodes["n1"].reads[0].concept_id == "a"
    assert tools.db.graph.nodes["n1"].writes[0].concept_id == "a"


def test_merge_incompatible_kinds_rejected():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "actor", "alpha")
    tools.create_concept("b", "data", "beta")
    with pytest.raises(ValueError):
        tools.merge_concepts("a", ["b"])


def test_remove_node_after_decomposition_leaves_no_dangling_edge():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("act1", "activity", "coarse")
    tools.create_concept("act2", "activity", "fine")
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


def test_invalid_evidence_span_rejected():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    tools.create_concept("bogus", "data", "quotation")
    with pytest.raises(ValueError):
        tools.ground_concept("bogus", evidence=[_ev("obs_none", "x")])
    res = _eval(tools)
    assert res.invalid_evidence_ref_count >= 0


# ---------------------------------------------------------------------------
# Observation lifecycle / privacy
# ---------------------------------------------------------------------------


def test_start_inference_preserves_observations_and_ledger():
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid = _say(
        tools,
        "I check the customer information in the CRM.",
        [_annotation("node:cc:system", "CRM")],
    )
    tools.create_concept("crm", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.ground_concept("crm", evidence=[_ev(oid, "CRM")])
    tools.create_concept("act", "activity", "check")
    assert tools.db.graph is not None
    tools.db.graph.concepts["act"].validation_status = "grounded"
    tools.add_node(
        "n1",
        activity={"concept_id": "act", "evidence": [_ev(oid, "CRM")]},
        system={"concept_id": "crm", "evidence": [_ev(oid, "CRM")]},
    )
    # restart inference (the agent decides to redo its graph)
    tools.start_inference("q2")
    # observations + ledger + conversation survive
    assert [o.id for o in tools.db.observations] == [oid]
    assert len(tools.db.messages) == 2  # assistant hello + user statement
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "I check the customer information in the CRM."
    # the OLD observation is still valid as evidence after the restart
    tools.create_concept("crm2", "system", "CRM", evidence=[_ev(oid, "CRM")])
    tools.ground_concept("crm2", evidence=[_ev(oid, "CRM")])
    tools.create_concept("act2", "activity", "check")
    assert tools.db.graph is not None
    tools.db.graph.concepts["act2"].validation_status = "grounded"
    tools.add_node(
        "n1",
        activity={"concept_id": "act2", "evidence": [_ev(oid, "CRM")]},
        system={"concept_id": "crm2", "evidence": [_ev(oid, "CRM")]},
    )
    assert tools.db.graph.nodes["n1"].activity.evidence[0].observation_id == oid
    assert tools.assertion_ledger.annotations() != {}
    res = _eval(tools)
    assert res.authentic_observation_count == 1


def test_private_ids_absent_from_agent_visible_state():
    tools = _tools()
    _build(tools)
    assert tools.db.graph is not None
    listing = tools.list_concepts()
    for forbidden in ("skc_", "node:", "edge:", "tc_"):
        assert forbidden not in listing, forbidden
    dumped = str(tools.db.model_dump(mode="json"))
    for forbidden in ("skc_", "node:cc", "edge:e3", "tc_activity"):
        assert forbidden not in dumped, forbidden
    assert "assertion_ledger" not in InterviewDB.model_fields


def test_user_message_serialization_excludes_annotations():
    msg = UserMessage(
        role="user",
        content="hello",
        stakeholder_annotations=[_annotation("node:cc:system", "CRM")],
        stakeholder_alignments=[
            {"semantic_id": "tc_quote", "quote": "yes", "act": "confirm"}
        ],
        stakeholder_terminology=[
            {"semantic_id": "tc_quote", "proposed_term": "offer", "quote": "yes"}
        ],
    )
    dumped = msg.model_dump(mode="json")
    assert "stakeholder_annotations" not in dumped
    assert "stakeholder_alignments" not in dumped
    assert "stakeholder_terminology" not in dumped
    assert "node:cc:system" not in str(dumped)
    assert "skc_quote" not in str(dumped)
    assert "skc_quote" not in repr(msg)


def test_ledger_binds_annotations_per_turn():
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment

    env = get_environment()
    assert isinstance(env, BusinessInterviewEnvironment)
    env.on_message(UserMessage(role="user", content="plain statement"))
    assert env.assertion_ledger is not None
    assert env.assertion_ledger.annotations() == {}
    env.on_message(
        UserMessage(
            role="user",
            content="I check the customer in the CRM.",
            stakeholder_annotations=[_annotation("node:cc:system", "CRM")],
        )
    )
    assert 1 in env.assertion_ledger.annotations()


def test_sidecar_parse_accepts_semantic_annotations():
    from tau2.domains.business_interview.user_simulator import parse_sidecar

    sidecar = parse_sidecar(
        '{"message": "Yes.", "annotations": [], "alignments": '
        '[{"semantic_id": "skc_012", "quote": "Yes.", "act": "confirm"}], '
        '"terminology": [{"semantic_id": "skc_013", '
        '"proposed_term": "customer master", "quote": "Yes."}]}'
    )
    assert sidecar["message"] == "Yes."
    assert sidecar["alignments"][0].semantic_id == "skc_012"
    assert sidecar["terminology"][0].proposed_term == "customer master"
    plain = parse_sidecar(
        '{"message": "I check the customer.", '
        '"annotations": [{"semantic_id": "node:cc:activity", "quote": "check", '
        '"occurrence": 0}]}'
    )
    assert plain["alignments"] == []
    assert plain["terminology"] == []
    with pytest.raises(ValueError):
        parse_sidecar(
            '{"message": "x", "annotations": [], "alignments": '
            '[{"semantic_id": "skc_012", "quote": "x", '
            '"act": "nonsense"}]}'
        )


# ---------------------------------------------------------------------------
# Old semantic matcher machinery is gone
# ---------------------------------------------------------------------------


def test_semantic_matcher_machinery_is_gone():
    import pathlib

    from tau2.domains.business_interview import evaluation

    for name in (
        "_predicate_ok",
        "_necessity_value_ok",
        "_attribute_match",
        "_node_overlap",
        "_STOPWORDS",
        "_term_covers",
        "_supported_claims",
        "_grounded_claim_ids",
        "TruthNodeSpec",
    ):
        assert not hasattr(evaluation, name), name
    domain_dir = pathlib.Path(evaluation.__file__).parent
    for src in domain_dir.glob("*.py"):
        text = src.read_text(encoding="utf-8")
        for token in ("_predicate_ok", "_attribute_match", "norm_role", "norm_system"):
            assert token not in text, f"{src.name} still contains {token!r}"
    assert not (domain_dir / "claims.py").exists()


def test_evaluation_spec_is_empty():
    from tau2.domains.business_interview.evaluation import EvaluationSpec

    assert EvaluationSpec().model_dump() == {}


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------


def test_finish_terminates_episode():
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
    assert env.tools is not None
    _build(env.tools)  # type: ignore[arg-type]
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


def test_lab_hidden_truth_guess_fails():
    """The lab tech does not know the derived read/write artifacts; asserting
    them is wrong (epistemic restraint)."""
    tools = _tools()
    _build_lab(tools)
    assert tools.db.graph is not None
    cant_say = _say(
        tools,
        "I cannot say.",
        alignments=[
            {
                "semantic_id": "tc_sample",
                "quote": "cannot say",
                "act": "unknown",
            }
        ],
        scenario_id=LAB_SCENARIO,
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


def _reference_trajectory() -> list:
    """A full faithful trajectory (messages + tool calls) mirroring ``_build``
    (incl. explicit DONT_KNOW recordings) with the private semantic sidecars
    carried on the UserMessages (ids in opaque local form)."""
    from tau2.data_model.message import AssistantMessage as AM
    from tau2.data_model.message import UserMessage as UM

    traj = []
    tools = InterviewTools(InterviewDB())
    tools.assertion_ledger.install_catalog(
        StakeholderKnowledgeCatalog.from_scenario(_sc(SCENARIO))
    )
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

    def say(text, anns, alignments=None, terminology=None):
        traj.append(
            UM(
                role="user",
                content=text,
                stakeholder_annotations=[
                    _annotation(_local_sid(s), q) for s, q in anns
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
                SemanticAnnotation(semantic_id=_local_sid(s), quote=q, occurrence=0)
                for s, q in anns
            ],
            text,
        )
        if alignments:
            tools.assertion_ledger.bind_alignment(
                turn,
                [
                    ConceptAlignmentAssertion(
                        semantic_id=_local_sid(a["semantic_id"]),
                        quote=a["quote"],
                        occurrence=a.get("occurrence", 0),
                        act=a["act"],
                    )
                    for a in alignments
                ],
                text,
            )
        if terminology:
            tools.assertion_ledger.bind_terminology(
                turn,
                [
                    TerminologyConfirmation(
                        semantic_id=_local_sid(t["semantic_id"]),
                        proposed_term=t["proposed_term"],
                        quote=t["quote"],
                        occurrence=t.get("occurrence", 0),
                    )
                    for t in terminology
                ],
                text,
            )

    mk("start_inference", {"name": "Quotation creation"})
    created: set[str] = set()
    node_oid: dict[str, str] = {}
    sm = 0
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, anns = _NODE_OBS[sid]
        sm += 1
        say(text, anns)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        node_oid[sid] = oid
        for semantic_id, q in anns:
            kcid = _SEMANTIC_TO_CONCEPT.get(semantic_id)
            if kcid is None:
                continue
            agent_cid, kind, label = _AGENT_CONCEPTS[kcid]
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
                    "ground_concept",
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                )
            else:
                mk(
                    "add_concept_mention",
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                )
    for anid, sid in _NODE_MAP.items():
        oid = node_oid[sid]
        anns = _NODE_OBS[sid][1]
        args: dict = {
            "node_id": anid,
            "activity": {
                "concept_id": _AGENT_CONCEPTS[
                    _SEMANTIC_TO_CONCEPT[f"node:{sid}:activity"]
                ][0],
                "evidence": [
                    _ev(oid, next(q for s, q in anns if s == f"node:{sid}:activity"))
                ],
            },
        }
        for semantic_id, q in anns:
            kcid = _SEMANTIC_TO_CONCEPT.get(semantic_id)
            if kcid is None:
                continue
            agent_cid = _AGENT_CONCEPTS[kcid][0]
            if semantic_id.startswith(f"node:{sid}:reads:"):
                args.setdefault("reads", []).append(
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]}
                )
            elif semantic_id.startswith(f"node:{sid}:writes:"):
                args.setdefault("writes", []).append(
                    {"concept_id": agent_cid, "evidence": [_ev(oid, q)]}
                )
            elif semantic_id == f"node:{sid}:actor":
                args["actor"] = {"concept_id": agent_cid, "evidence": [_ev(oid, q)]}
            elif semantic_id == f"node:{sid}:system":
                args["system"] = {"concept_id": agent_cid, "evidence": [_ev(oid, q)]}
            elif semantic_id == f"node:{sid}:rationale":
                args["necessity_rationale"] = {
                    "concept_id": agent_cid,
                    "evidence": [_ev(oid, q)],
                }
        mk("add_node", args)
    # explicit DONT_KNOW recordings (an unasserted slot is NOT DONT_KNOW)
    for sid in ("r", "cc", "cq", "ap", "sq", "me"):
        text, anns = _NODE_DONT_KNOW_OBS[sid]
        sm += 1
        say(text, anns)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        agent_node = {v: k for k, v in _NODE_MAP.items()}[sid]
        mk(
            "record_dont_know",
            {
                "node_id": agent_node,
                "properties": [
                    "rationale" if s == f"node:{sid}:rationale" else s.split(":")[2]
                    for s, _ in anns
                ],
                "evidence": [_ev(oid, q) for _, q in anns],
            },
        )
    for eid in ("e1", "e2", "e3", "e4", "e5", "e6"):
        text, anns = _EDGE_OBS[eid]
        sm += 1
        say(text, anns)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        cond = None
        for semantic_id, q in anns:
            if semantic_id == f"edge:{eid}:condition":
                kcid = _SEMANTIC_TO_CONCEPT[semantic_id]
                agent_cid, kind, label = _AGENT_CONCEPTS[kcid]
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
                        "ground_concept",
                        {"concept_id": agent_cid, "evidence": [_ev(oid, q)]},
                    )
                cond = {"concept_id": agent_cid, "evidence": [_ev(oid, q)]}
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
            "evidence": [_ev(oid, q) for s, q in anns if s == f"edge:{eid}"],
        }
        if cond:
            eargs["condition"] = cond
        mk("add_edge", eargs)
    for eid in ("e1", "e2", "e5"):
        text, anns = _EDGE_DONT_KNOW_OBS[eid]
        sm += 1
        say(text, anns)
        oid = mk("observe_message", {"message_id": f"sm_{sm}"})
        mk(
            "record_edge_condition_dont_know",
            {"edge_id": eid, "evidence": [_ev(oid, q) for _, q in anns]},
        )
    mk("set_graph_endpoints", {"start_node_id": "a", "end_node_ids": ["e", "f"]})
    mk("finish_interview", {"summary": "Inferred the quotation graph."})
    return traj


def test_evaluator_rewards_full_reconstruction():
    """The full tau2 EnvironmentEvaluator path (replay + env assertions +
    domain diagnostics) rewards a faithful trajectory carrying private
    semantic annotations."""
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    traj = _reference_trajectory()
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=traj,
        solo_mode=False,
        env_kwargs={"scenario_id": SCENARIO},
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
# v11: opaque stakeholder-local IDs, canonical resolver, binding-aware
# grounding, explicit Agent DONT_KNOW, knowledge coverage
# ---------------------------------------------------------------------------


def test_opaque_ids_are_stable_and_leak_no_truth_names():
    import re

    sc = _sc(SCENARIO)
    kg = sc.knowledge.graph
    truth = sc.truth
    for nid in kg.nodes:
        assert re.fullmatch(r"skn_\d{3}", nid), nid
    for eid in kg.edges:
        assert re.fullmatch(r"ske_\d{3}", eid), eid
    for cid in kg.concepts:
        assert re.fullmatch(r"skc_\d{3}", cid), cid
    # no Truth id, label, term, node id or edge id leaks into any knowledge
    # id or semantic id (even when descriptions/terms are DONT_KNOW)
    truth_tokens = set(truth.nodes) | set(truth.edges) | set(truth.concepts)
    truth_tokens.update(t for c in truth.concepts.values() for t in c.canonical_terms)
    for sid in kg.semantic_ids():
        assert "tc_" not in sid, sid
        for token in truth_tokens:
            if len(token) < 2:
                continue  # single-char Truth ids are not a leak vector
            assert token not in sid, (token, sid)
        # no Truth node/edge/concept id appears as a full semantic segment
        assert not (set(sid.split(":")) & truth_tokens), sid
    # deterministic for the scenario and invariant to collection reordering
    kg2 = project_knowledge(truth, sc.stakeholder).graph
    assert kg2.node_truth_ids == kg.node_truth_ids
    assert kg2.edge_truth_ids == kg.edge_truth_ids
    assert set(kg2.concepts) == set(kg.concepts)
    sc_again = get_scenario(SCENARIO)
    assert sc_again is not None
    assert sc_again.knowledge.graph.semantic_ids() == kg.semantic_ids()
    reordered = sc.stakeholder.model_copy(
        update={
            "visible_node_ids": list(reversed(sc.stakeholder.visible_node_ids)),
            "visible_edge_ids": list(reversed(sc.stakeholder.visible_edge_ids)),
        }
    )
    kg3 = project_knowledge(truth, reordered).graph
    assert kg3.node_truth_ids == kg.node_truth_ids
    assert kg3.edge_truth_ids == kg.edge_truth_ids
    # private mappings point back at Truth elements
    for local, tid in kg.node_truth_ids.items():
        assert tid in truth.nodes
        assert kg.nodes[local].id == local
    for local, tid in kg.edge_truth_ids.items():
        assert tid in truth.edges
    for c in kg.concepts.values():
        assert c.truth_concept_id in truth.concepts
        assert c.id != c.truth_concept_id


def test_concept_descriptions_contain_no_graph_facts():
    """Truth and knowledge concept descriptions describe only the concept
    itself: no predecessor/successor, reads/writes position, systems,
    branch/timing, input/output relations or hidden graph properties."""
    forbidden = (
        "before",
        "after",
        "then",
        "followed",
        "record it",
        "recorded",
        "from customer and pricing",
        "to accounting",
        "in the crm",
        "in the quoting system",
        "by email",
        "excel file",
        "preparing",
    )
    sc = _sc(SCENARIO)
    for concept in sc.truth.concepts.values():
        desc = concept.description.lower()
        for token in forbidden:
            assert token not in desc, (concept.id, token, desc)
    lab = _sc(LAB_SCENARIO)
    for concept in lab.truth.concepts.values():
        desc = concept.description.lower()
        for token in forbidden:
            assert token not in desc, (concept.id, token, desc)
    for knowledge in (sc.knowledge, lab.knowledge):
        for c in knowledge.graph.concepts.values():
            if not isinstance(c.description, str):
                continue
            desc = c.description.lower()
            for token in forbidden:
                assert token not in desc, (c.id, token, desc)


def test_dont_know_descriptions_cannot_leak_through_prompt():
    """A concept whose description/terms are DONT_KNOW renders as "unknown"
    in the stakeholder prompt — the Truth description can never leak through
    the prompt, and ids stay opaque even when the stakeholder knows nothing."""
    from tau2.domains.business_interview.user_simulator import (
        StakeholderUserSimulator,
    )

    truth = quotation_truth()
    filt = StakeholderFilter(
        name="s",
        visible_node_ids=["r"],
        visible_edge_ids=[],
        visible_node_attributes={"r": ["activity"]},
        visible_edge_attributes={},
        concept_overrides={
            "tc_activity_receive_request": ConceptKnowledgeOverride(
                description_known=False, terms_known=False
            )
        },
    )
    knowledge = project_knowledge(truth, filt)
    scenario = Scenario(
        scenario_id="custom", truth=truth, stakeholder=filt, knowledge=knowledge
    )
    sim = object.__new__(StakeholderUserSimulator)
    sim._scenario = scenario  # noqa: SLF001 - test-only
    sim._catalog = StakeholderKnowledgeCatalog(  # noqa: SLF001 - test-only
        "s", knowledge
    )
    block = sim._knowledge_block()  # noqa: SLF001 - test-only
    concept = next(
        c
        for c in knowledge.graph.concepts.values()
        if c.truth_concept_id == "tc_activity_receive_request"
    )
    assert not concept.has_description() and not concept.has_terms()
    # the DONT_KNOW description renders as unknown; the Truth description
    # and terms never appear anywhere in the prompt
    assert f'id="{concept.id}"' in block
    assert "unknown" in block
    assert "Receive the customer's quotation request" not in block
    assert "receive the quotation request" not in block
    # the opaque id itself leaks no Truth meaning
    assert "activity_receive" not in concept.id and "receive" not in concept.id


def test_every_semantic_id_resolves_uniquely():
    kg = _kg(SCENARIO)
    for sid in kg.semantic_ids():
        resolved = kg.resolve(sid)
        if resolved is None:
            raise AssertionError(sid)
        if resolved.kind == "node":
            assert resolved.node is not None and resolved.node_id is not None
            assert resolved.prop is None and resolved.ref is None
        elif resolved.kind == "node_slot":
            assert resolved.prop is not None and resolved.node is not None
        elif resolved.kind == "node_element":
            assert resolved.ref is not None and resolved.prop in ("reads", "writes")
        elif resolved.kind == "edge":
            assert resolved.edge is not None and resolved.edge_id is not None
        elif resolved.kind == "edge_slot":
            assert resolved.prop == "condition" and resolved.edge is not None
        elif resolved.kind == "concept":
            assert resolved.concept is not None and resolved.value is resolved.concept
    # unresolvable ids -> None (never a partial/lenient parse)
    for bad in (
        "node:skn_999",
        "node:skn_001:bogus",
        "node:skn_001:reads:skc_999",
        "edge:ske_999",
        "edge:ske_001:condition:x",
        "skc_999",
        "node:",
        "edge:",
        "reads:skc_001",
        "",
    ):
        assert kg.resolve(bad) is None, bad
    # semantic_ids() and resolve() agree exactly
    assert kg.semantic_ids() == {
        sid for sid in kg.semantic_ids() if kg.resolve(sid) is not None
    }


def test_node_existence_is_not_activity_slot():
    """node:<id> resolves to node EXISTENCE — never the activity slot — and
    evidence anchored only to the node id never scores the activity."""
    kg = _kg(SCENARIO)
    node_sid = "node:skn_005"  # truth node r
    resolved = kg.resolve(node_sid)
    if resolved is None:
        raise AssertionError(node_sid)
    assert resolved.kind == "node"
    assert resolved.value is resolved.node
    # the activity slot is a separate address with its own value
    act_sid = f"{node_sid}:activity"
    act = kg.resolve(act_sid)
    if act is None:
        raise AssertionError(act_sid)
    assert act.kind == "node_slot" and act.prop == "activity"
    assert isinstance(act.value, ConceptRef)
    # node existence represents NO concept; the slot does
    assert slot_concepts(resolved.value) == set()
    assert slot_concepts(act.value) == {act.value.concept_id}
    # end-to-end: node-existence evidence does not score the activity slot
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid = _say(
        tools,
        "The process starts with the first step.",
        [_annotation(node_sid, "first step")],
    )
    oid_act = _say(
        tools,
        "I receive the quotation request.",
        [_annotation(act_sid, "receive the quotation request")],
    )
    tools.create_concept(
        "act1",
        "activity",
        "first step",
        evidence=[_ev(oid_act, "receive the quotation request")],
    )
    tools.ground_concept(
        "act1", evidence=[_ev(oid_act, "receive the quotation request")]
    )
    tools.add_node(
        "n1",
        activity={"concept_id": "act1", "evidence": [_ev(oid, "first step")]},
    )
    res = _eval(tools)
    assert res.activity_correctness == 0.0


def test_reads_writes_item_resolves_exactly():
    """node:<id>:reads:<k> resolves to the exact ConceptRef — never the
    whole reads list — and element evidence grounds only that element."""
    kg = _kg(SCENARIO)
    elem = _local_sid("node:cq:reads:tc_pricing")
    resolved = kg.resolve(elem)
    if resolved is None:
        raise AssertionError(elem)
    assert resolved.kind == "node_element"
    assert resolved.prop == "reads"
    assert resolved.ref is not None
    assert resolved.ref.concept_id == _local_sid("tc_pricing")
    # the whole-list slot is a DIFFERENT address
    whole = kg.resolve(_local_sid("node:cq:reads"))
    if whole is None:
        raise AssertionError(elem)
    assert whole.kind == "node_slot" and whole.prop == "reads"
    assert isinstance(whole.value, list) and len(whole.value) == 2
    # element evidence grounds exactly the element
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid = _say(
        tools,
        "I use the pricing information.",
        [_annotation("node:cq:reads:tc_pricing", "pricing information")],
    )
    grounded, invalid, amb = grounded_semantic_ids(
        tools.db,
        tools.assertion_ledger.annotations(),
        [_evr(oid, "pricing information")],
    )
    assert grounded == {elem} and invalid == 0 and amb == 0
    # a broad span covering two elements of the same list is ambiguous
    oid2 = _say(
        tools,
        "I use the customer and pricing information.",
        [
            _annotation("node:cq:reads:tc_customer", "customer"),
            _annotation("node:cq:reads:tc_pricing", "pricing"),
        ],
    )
    grounded2, _, amb2 = grounded_semantic_ids(
        tools.db,
        tools.assertion_ledger.annotations(),
        [_evr(oid2, "the customer and pricing information")],
    )
    assert grounded2 == set() and amb2 == 1


def test_ground_concept_rejects_unrelated_ambiguous_kind_wrong():
    """Binding-aware grounding: evidence must resolve to exactly one
    kind-compatible knowledge concept; unrelated, ambiguous and
    kind-incompatible evidence is rejected and leaves the concept
    unresolved. Private stakeholder ids never appear in error messages."""
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    tools.create_concept("act1", "activity", "first step")
    # unrelated: node-existence evidence represents NO concept
    oid = _say(
        tools,
        "The first step happens first.",
        [_annotation("node:skn_005", "first step happens")],
    )
    with pytest.raises(ValueError) as exc:
        tools.ground_concept("act1", evidence=[_ev(oid, "first step happens")])
    assert "skn_" not in str(exc.value) and "skc_" not in str(exc.value)
    # kind-wrong: evidence resolves to a data concept
    oid2 = _say(
        tools,
        "I use the pricing information.",
        [_annotation("node:cq:reads:tc_pricing", "pricing information")],
    )
    with pytest.raises(ValueError):
        tools.ground_concept("act1", evidence=[_ev(oid2, "pricing information")])
    # ambiguous: one broad span covering two distinct ids
    oid3 = _say(
        tools,
        "I check the customer information in the CRM.",
        [
            _annotation("node:cc:activity", "check the customer information"),
            _annotation("node:cc:system", "CRM"),
        ],
    )
    with pytest.raises(ValueError):
        tools.ground_concept(
            "act1", evidence=[_ev(oid3, "check the customer information in the CRM")]
        )
    # ambiguous: two elements of one reads list
    oid4 = _say(
        tools,
        "I use the customer and pricing information.",
        [
            _annotation("node:cq:reads:tc_customer", "customer"),
            _annotation("node:cq:reads:tc_pricing", "pricing"),
        ],
    )
    with pytest.raises(ValueError):
        tools.ground_concept(
            "act1", evidence=[_ev(oid4, "the customer and pricing information")]
        )
    # the concept stays unresolved after every rejection
    assert tools.db.graph is not None
    assert tools.db.graph.concepts["act1"].validation_status == "hypothesized"
    # valid grounding succeeds
    oid5 = _say(
        tools,
        "I receive the quotation request.",
        [_annotation("node:skn_005:activity", "receive the quotation request")],
    )
    tools.ground_concept("act1", evidence=[_ev(oid5, "receive the quotation request")])
    assert tools.db.graph.concepts["act1"].validation_status == "grounded"


def test_grounded_status_agrees_with_evaluator_binding():
    """The tool and the evaluator resolve grounding evidence through the
    same canonical resolver, so the Agent-visible grounded status and the
    evaluator binding never disagree."""
    from tau2.domains.business_interview.evaluation import (  # noqa: PLC2701
        _concept_bindings,
        _knowledge_value_concepts,
        _match_nodes_and_edges,
    )

    tools = _tools()
    _build(tools)
    graph = tools.db.graph
    assert graph is not None
    knowledge = _sc(SCENARIO).knowledge
    annotations = tools.assertion_ledger.annotations()
    mapping, edge_map = _match_nodes_and_edges(graph, knowledge, tools.db, annotations)
    score, agent_to_knowledge = _concept_bindings(
        graph, knowledge, tools.db, annotations, mapping, edge_map
    )
    assert score == 1.0
    for cid, concept in graph.concepts.items():
        if concept.validation_status != "grounded":
            continue
        represented: set[str] = set()
        results, _inv, _amb = resolve_grounding_refs(
            tools.db, annotations, concept.validation_evidence
        )
        for _ref, sid in results:
            represented.update(_knowledge_value_concepts(knowledge, sid))
        # the tool guaranteed exactly one kind-compatible concept; the
        # evaluator binds the SAME concept
        assert len(represented) == 1, cid
        assert agent_to_knowledge[cid] == next(iter(represented)), cid


def test_agent_none_is_not_dont_know():
    """The three-valued agent slots score asymmetrically: stakeholder None
    (known absent) is satisfied by an unasserted slot (DONT_KNOW is NOT
    equivalent); stakeholder DONT_KNOW is satisfied ONLY by an explicit
    evidenced DONT_KNOW marker (unasserted is NOT DONT_KNOW)."""
    truth = quotation_truth()
    # knowledge 1: r.reads is KNOWN-ABSENT (None)
    filt_absent = StakeholderFilter(
        name="s1",
        visible_node_ids=["r"],
        visible_edge_ids=[],
        visible_node_attributes={"r": ["activity", "actor", "writes", "reads"]},
        visible_edge_attributes={},
    )
    # knowledge 2: r.reads is DONT_KNOW
    filt_unknown = StakeholderFilter(
        name="s2",
        visible_node_ids=["r"],
        visible_edge_ids=[],
        visible_node_attributes={"r": ["activity", "actor", "writes"]},
        visible_edge_attributes={},
    )
    k_absent = project_knowledge(truth, filt_absent)
    k_unknown = project_knowledge(truth, filt_unknown)
    tools = _tools()
    tools.assertion_ledger.install_catalog(StakeholderKnowledgeCatalog("s1", k_absent))
    tools.start_inference("q")
    _ingest(tools, "assistant", "Hello.")
    oid_act = _say(
        tools,
        "I receive the quotation request.",
        [_annotation("node:r:activity", "receive the quotation request")],
    )
    tools.create_concept(
        "act1",
        "activity",
        "receive",
        evidence=[_ev(oid_act, "receive the quotation request")],
    )
    tools.ground_concept(
        "act1", evidence=[_ev(oid_act, "receive the quotation request")]
    )
    tools.add_node(
        "a",
        activity={
            "concept_id": "act1",
            "evidence": [_ev(oid_act, "receive the quotation request")],
        },
    )

    def eval_with(knowledge, filt):
        return evaluate(
            tools.db,
            knowledge,
            EvaluationSpec(),
            filt,
            truth=truth,
            annotations=tools.assertion_ledger.annotations(),
            alignments=tools.assertion_ledger.alignments(),
            terminology=tools.assertion_ledger.terminology(),
        )

    # stakeholder None: unasserted agent slot is CORRECT
    res = eval_with(k_absent, filt_absent)
    assert res.read_correctness == 1.0
    # ... but a DONT_KNOW marker is NOT equivalent to None
    assert tools.db.graph is not None
    tools.db.graph.nodes["a"].reads = DontKnowType(evidence=[])
    res_dk = eval_with(k_absent, filt_absent)
    assert res_dk.read_correctness == 0.0
    # stakeholder DONT_KNOW: unasserted is NOT DONT_KNOW (missing/unasserted
    # must not score as both None and DONT_KNOW)
    tools.db.graph.nodes["a"].reads = None
    res_un = eval_with(k_unknown, filt_unknown)
    assert res_un.read_correctness == 0.0
    # ... while an explicit evidenced DONT_KNOW is CORRECT
    tools.assertion_ledger.install_catalog(StakeholderKnowledgeCatalog("s2", k_unknown))
    oid = _say(
        tools,
        "I do not know what the step reads.",
        [_annotation("node:r:reads", "what the step reads")],
    )
    tools.record_dont_know(
        "a", properties=["reads"], evidence=[_ev(oid, "what the step reads")]
    )
    res_dk2 = eval_with(k_unknown, filt_unknown)
    assert res_dk2.read_correctness == 1.0
    assert res_dk2.dont_know_evidence_errors == 0


def test_correct_dont_know_is_rewarded_and_guess_wrong():
    """The full build's explicit DONT_KNOW recordings are rewarded; removing
    them (unasserted) is NOT correct; a hidden-Truth guess on a DONT_KNOW
    slot is wrong."""
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
    assert res.condition_correctness == 1.0
    assert res.dont_know_evidence_errors == 0
    graph = tools.db.graph
    assert graph is not None
    assert is_dont_know(graph.nodes["e"].reads)
    assert is_dont_know(graph.nodes["e"].writes)
    assert is_dont_know(graph.edges["e1"].condition)
    # unasserting a DONT_KNOW slot is NOT correct (missing != DONT_KNOW)
    graph.nodes["e"].reads = None
    graph.nodes["e"].writes = None
    res_missing = _eval(tools)
    assert res_missing.read_correctness < 1.0
    assert res_missing.write_correctness < 1.0
    # a hidden-Truth guess on a DONT_KNOW slot is wrong
    graph.nodes["e"].reads = None
    oid = _say(
        tools,
        "I use the pricing information.",
        [_annotation("node:cq:reads:tc_pricing", "pricing information")],
    )
    graph.nodes["e"].reads = [
        ConceptRef(
            concept_id="pricing",
            confidence=1.0,
            evidence=[_evr(oid, "pricing information")],
        )
    ]
    res_guess = _eval(tools)
    assert res_guess.read_correctness < 1.0


def test_knowledge_coverage_known_absent_unknown_removed():
    """knowledge_coverage: known values AND known absence count as known;
    DONT_KNOW slots and removed nodes/edges count as unknown; node existence
    is never confused with the activity slot."""
    truth = BusinessProcessGraph(
        id="cov",
        name="coverage",
        concepts={
            "a1": TruthConcept(id="a1", kind="activity"),
            "a2": TruthConcept(id="a2", kind="activity"),
            "d1": TruthConcept(id="d1", kind="data"),
        },
        nodes={
            "A": Node(
                id="A",
                activity=ConceptRef(concept_id="a1"),
                reads=[ConceptRef(concept_id="d1")],
            ),
            "B": Node(id="B", activity=ConceptRef(concept_id="a2")),
        },
        edges={"ab": Edge(id="ab", from_node="A", to_node="B")},
        start_node_id="A",
        end_node_ids=["B"],
    )
    ALL_PROPS = ["activity", "actor", "system", "reads", "writes", "rationale"]
    full = StakeholderFilter(
        name="full",
        visible_node_ids=["A", "B"],
        visible_edge_ids=["ab"],
        visible_node_attributes={"A": ALL_PROPS, "B": ALL_PROPS},
        visible_edge_attributes={"ab": ["condition"]},
    )

    def coverage(filt):
        knowledge = project_knowledge(truth, filt)
        db = InterviewDB()
        return evaluate(
            db, knowledge, EvaluationSpec(), filt, truth=truth
        ).knowledge_coverage

    # fully known (incl. known-absent slots: B.reads None, ab condition
    # None) -> 1.0. Total addresses: A(1) + A slots(6) + A element(1) +
    # B(1) + B slots(6) + ab(1) + ab condition(1) = 17.
    assert coverage(full) == 1.0
    # DONT_KNOW: A.reads unknown -> slot + element unknown -> 15/17
    dk = full.model_copy(
        update={
            "visible_node_attributes": {
                "A": [p for p in ALL_PROPS if p != "reads"],
                "B": ALL_PROPS,
            }
        }
    )
    assert coverage(dk) == 15.0 / 17.0
    # removed node + removed edge -> unavailable/unknown -> 7/17
    rm = StakeholderFilter(
        name="rm",
        visible_node_ids=["B"],
        visible_edge_ids=[],
        visible_node_attributes={"B": ALL_PROPS},
        visible_edge_attributes={},
    )
    assert coverage(rm) == 7.0 / 17.0
    # node existence vs activity: removing A costs its existence + slots +
    # elements — the activity slot is not "the node"
    rm2 = StakeholderFilter(
        name="rm2",
        visible_node_ids=["A"],
        visible_edge_ids=[],
        visible_node_attributes={"A": ALL_PROPS},
        visible_edge_attributes={},
    )
    # known: A(1) + A slots(6) + A element(1) = 8 of 17
    assert coverage(rm2) == 8.0 / 17.0


# ---------------------------------------------------------------------------
# Task data / policy hygiene
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
    assert "ground_concept" in p
    assert "mention" in p
    assert "terminology" in p
    assert "exact substring" in p
    assert "cycles" in p


def test_policy_does_not_hardcode_domain_terms():
    p = " ".join(BUSINESS_INTERVIEW_POLICY_PATH.read_text().split()).lower()
    for banned in ("quotation", "crm", "tc_"):
        assert banned not in p, f"policy hard-codes domain term: {banned}"


def test_task_prose_contains_no_scenario_business_facts():
    for task in get_tasks():
        ins = task.user_scenario.instructions
        assert not (getattr(ins, "unknown_info", None) or "").strip(), task.id
        ti = (getattr(ins, "task_instructions", None) or "").lower()
        for banned in ("high-value", "credit risk", "month-end", "1,000,000"):
            assert banned not in ti, f"{task.id}: task_instructions contain {banned!r}"
        desc_obj = task.description
        if desc_obj is not None:
            desc = (f"{desc_obj.purpose or ''} {desc_obj.notes or ''}").lower()
            assert "dag" not in desc, f"{task.id}: description still says DAG"
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
