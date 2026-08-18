"""Tests for the agent-local-concept business_interview domain (v4).

The domain centers on free-text actions + optional generic primitives (with an
explicit ``unclassified`` sentinel for unknown operations) + authentic Observation
provenance. **Reads/writes are agent-local data concepts**: the Agent LLM creates
``DataConcept``\ s, attaches observed ``ConceptTerm``\ s, and reuses concept ids
in ``ConceptRef``\ s; the evaluator binds each local concept to a hidden Truth
concept through **hidden stakeholder provenance** (claim catalog + per-utterance
ledger, ``claims.py``). The evaluator never infers semantic support from
Observation text, never compares labels to Ground Truth, and never rescues a
hidden assertion. DiscoveredConcept and concept-discovery evaluation are removed.
"""

from typing import Optional

import pytest  # type: ignore[reportMissingImports]

import tau2.domains.business_interview.dag as dagmod
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.business_interview.claims import (
    build_provenance_ledger,
    derive_utterance_claims,
    validate_ledger,
)
from tau2.domains.business_interview.concepts import resolve_primitive
from tau2.domains.business_interview.dag import (
    BusinessDAG,
    ConceptRef,
    Edge,
    InferredValue,
    InterviewDB,
    Node,
)
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.evaluation import evaluate
from tau2.domains.business_interview.scenario import (
    get_scenario,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import BUSINESS_INTERVIEW_POLICY_PATH

SCENARIO = "quotation_workflow_1"
JA_SCENARIO = SCENARIO + "_ja"
LAB_SCENARIO = "lab_sample_flow"
ALL_TASK_IDS = [SCENARIO, JA_SCENARIO, LAB_SCENARIO]

# (sid, action, actor, system, read concept ids, write concept ids)
# Truth concept ids are evaluator-only; the tests use the same ids so agent
# concept ids differ from truth ids (exercising the mapping).
_TRUTH_NODES = [
    ("a", "receive quotation request", "sales", None, [], ["tc_request"]),
    ("b", "check customer information in the CRM", "sales", "crm", ["tc_customer"], []),
    (
        "c",
        "create quotation in the quoting system",
        "sales",
        "quoting",
        ["tc_customer", "tc_pricing"],
        ["tc_quote"],
    ),
    (
        "d",
        "approve high-value quotation",
        "manager",
        "quoting",
        ["tc_quote"],
        ["tc_approval"],
    ),
    (
        "e",
        "send quotation to customer",
        "sales",
        "email",
        ["tc_quote"],
        ["tc_sent_quote"],
    ),
    (
        "f",
        "send quotation summary to accounting at month-end",
        "sales",
        "excel",
        ["tc_quote"],
        ["tc_excel_summary"],
    ),
]

_JA_ACTIONS = [
    "見積依頼を受け付ける",
    "CRMで顧客情報を確認する",
    "見積システムで見積を作成する",
    "高額見積を承認する",
    "見積を顧客に送付する",
    "月末に経理へ見積集計を送る",
]

_PRIM_BY_SID = {
    "a": "receive",
    "b": "check",
    "c": "create",
    "d": "approve",
    "e": "send",
    "f": "send",
}

# Agent-local concept ids used by the faithful test builds (deliberately
# different from the truth ids above) and their stakeholder-wording labels.
_AGENT_CONCEPTS = {
    "tc_request": ("request", "quotation request"),
    "tc_customer": ("customer", "customer information"),
    "tc_pricing": ("pricing", "pricing information"),
    "tc_quote": ("quote", "quotation"),
    "tc_approval": ("approval", "approval"),
    "tc_sent_quote": ("sent_quote", "sent quotation"),
    "tc_excel_summary": ("excel_summary", "summary of quotation information"),
}

_AGENT_CONCEPTS_JA = {
    "tc_request": ("request", "見積依頼"),
    "tc_customer": ("customer", "顧客情報"),
    "tc_pricing": ("pricing", "価格情報"),
    "tc_quote": ("quote", "見積書"),
    "tc_approval": ("approval", "承認"),
    "tc_sent_quote": ("sent_quote", "送付済みの見積書"),
    "tc_excel_summary": ("excel_summary", "見積情報の集計"),
}

# Natural phrasing per truth concept used to build observation texts (EN).
_NATURAL_EN = {
    "tc_request": "the quotation request",
    "tc_customer": "the customer information",
    "tc_pricing": "pricing information",
    "tc_quote": "the quotation",
    "tc_approval": "the approval",
    "tc_sent_quote": "the sent quotation",
    "tc_excel_summary": "a summary of the quotation information",
}

_NATURAL_JA = {
    "tc_request": "見積依頼",
    "tc_customer": "顧客情報",
    "tc_pricing": "価格情報",
    "tc_quote": "見積書",
    "tc_approval": "承認",
    "tc_sent_quote": "送付済みの見積書",
    "tc_excel_summary": "見積情報の集計",
}


def _tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    """Evaluate under the scenario's stakeholder visibility + hidden claim
    provenance (runtime behavior)."""
    sc = get_scenario(scenario)
    assert sc is not None
    provenance = build_provenance_ledger(tools.db, sc.claims, sc.stop_phrases)
    return evaluate(
        tools.db,
        sc.truth,
        sc.spec,
        sc.stakeholder,
        claims=sc.claims,
        provenance=provenance,
    )


def _ingest(tools: InterviewTools, role: str = "user", content: str = "") -> int:
    tools.db.messages.append({"role": role, "content": content})
    return len(tools.db.messages) - 1


def _claim_obs(tools: InterviewTools, text: str) -> str:
    _ingest(tools, "user", text)
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


def _node_obs_text(action, actor, system, reads, writes, ja: bool = False) -> str:
    """Natural-language observation text for a node, grounded in the
    stakeholder's wording (the hidden claim derivation matches these)."""
    natural = _NATURAL_JA if ja else _NATURAL_EN
    parts = []
    parts.append(
        f"The {actor} performs: {action}." if actor else f"The process: {action}."
    )
    if system:
        parts.append(f"It uses the {system}.")
    if reads:
        parts.append(f"It reads {', '.join(natural[r] for r in reads)}.")
    if writes:
        parts.append(f"It writes {', '.join(natural[w] for w in writes)}.")
    return " ".join(parts)


def _ensure_concept(
    tools: InterviewTools, truth_cid: str, oid: Optional[str], ja: bool = False
) -> str:
    """Create (or reuse) the agent-local concept for a truth concept id,
    returning the agent concept id. First use creates it with the
    stakeholder-wording label (locale-appropriate); later uses reuse the same
    id (no term churn)."""
    table = _AGENT_CONCEPTS_JA if ja else _AGENT_CONCEPTS
    agent_cid, label = table[truth_cid]
    dag = tools.db.dag
    assert dag is not None
    if agent_cid not in dag.data_concepts:
        tools.create_concept(agent_cid, label, observation_id=oid)
    return agent_cid


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("a", "b", "c", "d", "e", "f"),
    edge_ids: tuple = ("e1", "e2", "e3", "e4", "e5", "e6"),
    evidence: bool = True,
):
    """Build the correct quotation DAG with a quality_pass-quality provenance,
    asserting only **stakeholder-visible** attributes (per the sales
    StakeholderFilter) so the build matches actual runtime behavior.

    Data is recorded as agent-local concepts: one concept per business object
    (``request``, ``customer``, ``pricing``, ``quote``, ``excel_summary``),
    with ``customer`` reused across cc.reads and cq.reads. Hidden attributes
    (ap system/reads/writes, sq reads/writes, me reads) are left unset — the
    correct epistemic-restraint behavior.

    With ``evidence=True`` each node gets one rich authentic Observation that
    supports all its claims, and each edge an Observation that supports the
    relation (+ predicate). With ``evidence=False`` the same topology is built
    with no provenance (for the evidence-gate failures).
    """
    tools.start_inference("quotation")
    i1, i2, i3, i4, i5, i6 = ids
    actions = _JA_ACTIONS if ja else [t[1] for t in _TRUTH_NODES]
    node_data = list(zip(_TRUTH_NODES, actions))
    node_map = {"a": i1, "b": i2, "c": i3, "d": i4, "e": i5, "f": i6}
    sc = get_scenario(JA_SCENARIO if ja else SCENARIO)
    assert sc is not None
    visible_by_sid = {
        sid: sc.stakeholder.visible_attributes_for(tid)
        for sid, tid in {
            "a": "r",
            "b": "cc",
            "c": "cq",
            "d": "ap",
            "e": "sq",
            "f": "me",
        }.items()
    }
    edge_defs = [
        (edge_ids[0], i1, i2, None),
        (edge_ids[1], i2, i3, None),
        (edge_ids[2], i3, i4, "100万円超" if ja else "amount over 1,000,000"),
        (edge_ids[3], i3, i5, "100万円以下" if ja else "amount at or below 1,000,000"),
        (edge_ids[4], i4, i5, None),
        (edge_ids[5], i3, i6, "月末" if ja else "month-end"),
    ]
    rationale = "与信リスク管理のため" if ja else "for credit risk management"

    if evidence:
        _ingest(tools, "assistant", "Hello.")
        for (sid, _, actor, system, reads, writes), action in node_data:
            oid = _claim_obs(
                tools, _node_obs_text(action, actor, system, reads, writes, ja=ja)
            )
            vis = visible_by_sid[sid]
            read_cids = [
                _ensure_concept(tools, c, oid, ja=ja) for c in reads if "reads" in vis
            ]
            write_cids = [
                _ensure_concept(tools, c, oid, ja=ja) for c in writes if "writes" in vis
            ]
            tools.add_node(
                node_map[sid],
                action,
                primitive=_PRIM_BY_SID[sid],
                actor=actor if "actor" in vis else None,
                system=system if "system" in vis else None,
                reads=read_cids or None,
                writes=write_cids or None,
                observation_id=oid,
            )
        doid = _claim_obs(tools, f"The approval is {rationale}.")
        tools.set_node_necessity(i4, rationale=rationale, observation_id=doid)
        tools.set_node_necessity(i6)
        node_id_to_action = {
            node_map[sid]: action for (sid, _, _, _, _, _), action in node_data
        }
        for eid, frm, to, pred in edge_defs:
            fa, ta = node_id_to_action[frm], node_id_to_action[to]
            etext = f"After {fa}, we {ta}." + (f" when {pred}." if pred else "")
            eoid = _claim_obs(tools, etext)
            tools.add_edge(eid, frm, to, predicate=pred, observation_id=eoid)
    else:
        for (sid, _, actor, system, reads, writes), action in node_data:
            vis = visible_by_sid[sid]
            read_cids = [
                _ensure_concept(tools, c, None, ja=ja) for c in reads if "reads" in vis
            ]
            write_cids = [
                _ensure_concept(tools, c, None, ja=ja)
                for c in writes
                if "writes" in vis
            ]
            tools.add_node(
                node_map[sid],
                action,
                primitive=_PRIM_BY_SID[sid],
                actor=actor if "actor" in vis else None,
                system=system if "system" in vis else None,
                reads=read_cids or None,
                writes=write_cids or None,
            )
        tools.set_node_necessity(i4, rationale=rationale)
        tools.set_node_necessity(i6)
        for eid, frm, to, pred in edge_defs:
            tools.add_edge(eid, frm, to, predicate=pred)

    tools.set_dag_endpoints(start_node_id=i1, end_node_ids=[i5, i6])
    tools.finish_interview()


def _wrong_claim(tools: InterviewTools, target: str) -> str:
    """Make one claim **wrong** while giving it an authentic but unrelated
    Observation as provenance.

    Evidence hygiene only checks that the referenced Observation is real and
    authentic; it does not re-interpret its text. A wrong value is therefore
    expected to pass hygiene but fail Ground Truth correctness. For
    reads/writes the ref is re-pointed at a fresh concept citing an
    authentic-but-unrelated Observation, which hidden provenance cannot
    support.
    """
    poison = _claim_obs(tools, "I like pizza on Fridays.")
    dag = tools.db.dag
    assert dag is not None
    if target == "action":
        dag.nodes["b"].action = InferredValue(
            value="perform zebra dance",
            confidence=1.0,
            observation_ids=[poison],
        )
    elif target == "actor":
        dag.nodes["b"].actor = InferredValue(
            value="manager", confidence=1.0, observation_ids=[poison]
        )
    elif target == "system":
        dag.nodes["b"].system = InferredValue(
            value="erp", confidence=1.0, observation_ids=[poison]
        )
    elif target == "read":
        tools.create_concept("wrongdata", "wrong data", observation_id=poison)
        dag.nodes["b"].reads[0] = ConceptRef(
            concept_id="wrongdata",
            confidence=1.0,
            observation_ids=[poison],
        )
    elif target == "write":
        tools.create_concept("wrongdata", "wrong data", observation_id=poison)
        dag.nodes["c"].writes[0] = ConceptRef(
            concept_id="wrongdata",
            confidence=1.0,
            observation_ids=[poison],
        )
    elif target == "predicate":
        dag.edges["e3"].predicate = InferredValue(
            value="a completely wrong condition",
            confidence=1.0,
            observation_ids=[poison],
        )
    elif target == "edge":
        # rewire e5 (d->e) to (b->e): keeps the DAG valid, but is wrong vs truth.
        dag.edges["e5"].from_node = "b"
        dag.edges["e5"].to_node = "e"
    elif target == "necessity":
        assert dag.nodes["d"].necessity is not None
        dag.nodes["d"].necessity.rationale = InferredValue(
            value="a wrong reason", confidence=1.0, observation_ids=[poison]
        )
    return poison

    # ---------------------------------------------------------------------------
    # Model shape (DiscoveredConcept removed)
    # ---------------------------------------------------------------------------
    assert not hasattr(dagmod, "DiscoveredConcept")
    assert "concepts" not in BusinessDAG.model_fields
    assert "concept_id" not in Node.model_fields
    assert not hasattr(InterviewTools, "discover_concept")


def test_truth_and_agent_result_use_same_dag_class():
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert isinstance(sc.truth, BusinessDAG)
    tools = _tools()
    _build(tools)
    assert isinstance(tools.db.dag, BusinessDAG)


def test_cycle_reject():
    dag = BusinessDAG(
        nodes={
            "a": Node(id="a", action=InferredValue(value="x")),
            "b": Node(id="b", action=InferredValue(value="y")),
        },
        edges={
            "e1": Edge(id="e1", from_node="a", to_node="b"),
            "e2": Edge(id="e2", from_node="b", to_node="a"),
        },
        start_node_id="a",
        end_node_ids=["b"],
    )
    assert any("cycle" in e for e in dag.structure_errors())


def test_empty_end_node_ids_invalid():
    dag = BusinessDAG(
        nodes={"a": Node(id="a", action=InferredValue(value="x"))},
        edges={},
        start_node_id="a",
        end_node_ids=[],
    )
    assert not dag.is_valid


# ---------------------------------------------------------------------------
# Generic primitive / unclassified
# ---------------------------------------------------------------------------


def test_resolve_primitive_known():
    assert resolve_primitive("we create the quotation") == "create"
    assert resolve_primitive("the manager approves") == "approve"


def test_resolve_primitive_unknown_is_unclassified():
    assert resolve_primitive("chamber seasoning") == "unclassified"
    assert resolve_primitive("specimen accession") == "unclassified"
    assert resolve_primitive(None) == "unclassified"
    assert resolve_primitive("") == "unclassified"


def test_unknown_operation_with_unclassified_is_valid():
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["b"].primitive = InferredValue(value="unclassified", confidence=1.0)
    res = _eval(tools)
    assert res.quality_pass is True  # unclassified is a normal state, not a failure


# ---------------------------------------------------------------------------
# Valid full DAG + result correctness vs evidence hygiene
# ---------------------------------------------------------------------------


def test_valid_full_dag_passes():
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.quality_pass is True
    assert res.evidence_pass is True
    assert res.provenance_authenticity_pass is True
    assert res.structural_pass is True


def test_wrong_claim_passes_evidence_hygiene_but_fails_ground_truth():
    """A wrong claim backed by an authentic-but-unrelated Observation passes
    evidence hygiene (references are real & authentic) but fails Ground Truth
    correctness comparison."""
    for target in (
        "action",
        "actor",
        "system",
        "read",
        "write",
        "predicate",
        "edge",
        "necessity",
    ):
        tools = _tools()
        _build(tools, evidence=True)
        _wrong_claim(tools, target)
        res = _eval(tools)
        assert res.provenance_authenticity_pass is True, target
        assert res.evidence_pass is True, target  # hygiene passes
        assert res.quality_pass is False, target  # wrong value caught vs GT


def test_wrong_action_drops_node_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "action")
    res = _eval(tools)
    assert res.node_recall < 1.0
    assert res.node_precision < 1.0
    assert res.quality_pass is False


def test_wrong_actor_drops_actor_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "actor")
    res = _eval(tools)
    assert res.actor_correctness < 1.0
    assert res.quality_pass is False


def test_wrong_system_drops_system_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "system")
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.quality_pass is False


def test_wrong_read_drops_read_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "read")
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.quality_pass is False


def test_wrong_write_drops_write_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "write")
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.quality_pass is False


def test_wrong_predicate_drops_predicate_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "predicate")
    res = _eval(tools)
    assert res.predicate_correctness < 1.0
    assert res.quality_pass is False


def test_wrong_edge_drops_edge_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "edge")
    res = _eval(tools)
    assert res.edge_recall < 1.0 or res.edge_precision < 1.0
    assert res.quality_pass is False


def test_wrong_necessity_drops_necessity_correctness():
    tools = _tools()
    _build(tools, evidence=True)
    _wrong_claim(tools, "necessity")
    res = _eval(tools)
    assert res.necessity_correctness < 1.0
    assert res.necessity_pass is False
    assert res.quality_pass is False


def test_correct_action_wrong_known_primitive_lowers_diagnostic():
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["c"].primitive = InferredValue(value="approve", confidence=1.0)
    res = _eval(tools)
    assert res.primitive_correctness < 1.0
    assert res.node_recall == 1.0  # domain concept still correct


# ---------------------------------------------------------------------------
# Evidence / authenticity / endpoints / necessity regressions
# ---------------------------------------------------------------------------


def test_perfect_dag_zero_observations_fails_evidence_gate():
    tools = _tools()
    _build(tools, evidence=False)
    res = _eval(tools)
    assert res.evidence_pass is False
    assert res.quality_pass is False


def test_arbitrary_agent_node_ids_pass():
    tools = _tools()
    _build(tools, ids=("req", "check", "create", "approve", "send", "month_end"))
    res = _eval(tools)
    assert res.quality_pass is True


def test_nonexistent_observation_ref_rejected():
    tools = _tools()
    tools.start_inference("Q")
    with pytest.raises(ValueError):
        tools.add_node("n1", "receive request", observation_id="does_not_exist")


def test_fabricated_observation_reference_fails_authenticity():
    """A DAG claim referencing an observation id that does not exist fails the
    evidence-hygiene gate (invalid reference)."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["b"].actor.observation_ids.append("obs_fabricated")
    res = _eval(tools)
    assert res.invalid_observation_reference_count >= 1
    assert res.provenance_authenticity_pass is False
    assert res.evidence_pass is False
    assert res.quality_pass is False


def test_fabricated_observation_source_fails_authenticity():
    """An Observation that was never captured from a stakeholder message is not
    authentic; a claim referencing it fails evidence hygiene."""
    tools = _tools()
    _build(tools, evidence=True)
    fake = dagmod.Observation(
        id="obs_fake", source_id="stakeholder", text="made up", order=999, turn=999
    )
    tools.db.observations.append(fake)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["b"].actor.observation_ids.append("obs_fake")
    res = _eval(tools)
    assert res.invalid_observation_source_count >= 1
    assert res.provenance_authenticity_pass is False
    assert res.evidence_pass is False
    assert res.quality_pass is False


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
        "predicate_correctness",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "necessity_correctness",
        "primitive_correctness",
        "quality_pass",
    ):
        assert getattr(ren, f) == getattr(rja, f), f


def test_observation_authenticity_invariants():
    tools = _tools()
    turn = _ingest(tools, "user", "We receive requests.")
    assert tools.observe_latest_stakeholder_message() == "sm_1"
    oid = tools.observe_message("sm_1")
    assert oid == f"obs_{turn}"
    assert tools.observe_message("sm_1") == oid  # idempotent
    # fabricated / nonexistent message ids are rejected
    with pytest.raises(ValueError):
        tools.observe_message("sm_99")
    with pytest.raises(ValueError):
        tools.observe_message("sm_0")
    with pytest.raises(ValueError):
        tools.observe_message("not_a_message_id")
    # assistant messages cannot be observed (no sm id exists for them)
    _ingest(tools, "assistant", "I am the agent.")
    assert tools.list_stakeholder_messages().count("sm_") == 1
    with pytest.raises(ValueError):
        tools.observe_message("sm_2")  # only one user message exists


def test_necessity_correct_and_fabricated():
    ok = _tools()
    _build(ok, evidence=True)
    assert _eval(ok).necessity_pass is True
    bad = _tools()
    _build(bad, evidence=True)
    bad.set_node_necessity("f", rationale="for accounting reconciliation")
    assert _eval(bad).fabricated_necessity is True
    assert _eval(bad).necessity_pass is False


def test_declared_endpoints_must_match():
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    dag.end_node_ids = []
    res = _eval(tools)
    assert res.end_precision == 0.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Non-quotation lab scenario (no DiscoveredConcept)
# ---------------------------------------------------------------------------


def _build_lab(tools: InterviewTools):
    """Build a lab DAG asserting ONLY stakeholder-visible lab attributes (per
    the lab StakeholderFilter) with evidence text grounded in the actual
    lab_sample_flow known_info (not benchmark-derived artifact names).

    Hidden derived read/write artifacts (accessioned sample, seasoned chamber,
    conditioned sample, batch approval) are left unset.
    """
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        # (sid, action, prim, actor, system, reads, writes, evidence_text)
        # n1: actor + reads (sample) visible; derived write hidden.
        (
            "n1",
            "specimen accession",
            "receive",
            "lab tech",
            None,
            ["sample"],
            None,
            "When a specimen arrives, I accession it (record it as received).",
        ),
        # n2: actor + system visible; derived write hidden.
        (
            "n2",
            "chamber seasoning",
            "create",
            "lab tech",
            "environment chamber",
            None,
            None,
            "I season the environment chamber to prepare it.",
        ),
        # n3: actor + system visible; derived reads/writes hidden.
        (
            "n3",
            "conditioning cycle",
            "transform",
            "lab tech",
            "environment chamber",
            None,
            None,
            "I run a conditioning cycle that processes the samples inside the chamber.",
        ),
        # n4: actor visible; derived reads/writes hidden.
        (
            "n4",
            "approve conditioned batch",
            "approve",
            "lab supervisor",
            None,
            None,
            None,
            "Finally, the lab supervisor approves the conditioned batch before it is released.",
        ),
    ]
    for sid, action, prim, actor, system, reads, writes, text in nodes:
        oid = _claim_obs(tools, text)
        if reads:
            tools.create_concept("sample", "sample", observation_id=oid)
        tools.add_node(
            sid,
            action,
            primitive=prim,
            actor=actor,
            system=system,
            reads=reads,
            writes=writes,
            observation_id=oid,
        )
    for eid, frm, to, text in [
        (
            "l1",
            "n1",
            "n2",
            "After specimen accession, we prepare the chamber for chamber seasoning.",
        ),
        (
            "l2",
            "n2",
            "n3",
            "After chamber seasoning, we process samples during the conditioning cycle.",
        ),
        (
            "l3",
            "n3",
            "n4",
            "After the conditioning cycle, the lab supervisor approves the conditioned batch.",
        ),
    ]:
        oid = _claim_obs(tools, text)
        tools.add_edge(eid, frm, to, observation_id=oid)
    tools.set_dag_endpoints(start_node_id="n1", end_node_ids=["n4"])
    tools.finish_interview()


def test_non_quotation_lab_scenario_full_pass():
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.quality_pass is True
    assert res.node_recall == 1.0


def test_lab_unknown_primitive_unclassified_valid():
    tools = _tools()
    _build_lab(tools)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["n2"].primitive = InferredValue(value="unclassified", confidence=1.0)
    assert _eval(tools, LAB_SCENARIO).quality_pass is True


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

HIDDEN_TERMS = (
    "receive_request",
    "check_customer",
    "create_quote",
    "approve_quote",
    "send_quote",
    "month_end_summary",
    "credit_risk",
    "pred_amount_over",
)


def test_hidden_truth_not_leaked_to_agent():
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    docs = "\n".join(
        [
            InterviewTools.add_node.__doc__ or "",
            InterviewTools.add_edge.__doc__ or "",
            InterviewTools.observe_message.__doc__ or "",
            InterviewTools.observe_latest_stakeholder_message.__doc__ or "",
            InterviewTools.set_node_necessity.__doc__ or "",
            InterviewTools.create_concept.__doc__ or "",
            InterviewTools.add_concept_term.__doc__ or "",
            InterviewTools.merge_concepts.__doc__ or "",
            InterviewTools.list_concepts.__doc__ or "",
        ]
    ).lower()
    for term in HIDDEN_TERMS:
        assert term not in policy, f"policy leaks {term}"
        assert term not in docs, f"tool doc leaks {term}"
    # claim ids / truth concept ids / hidden provenance never appear in policy
    # or docs (the word "provenance" alone is fine — it is the agent's own
    # observation provenance; the HIDDEN mechanism must not leak)
    for forbidden in (
        "tc_request",
        "tc_",
        "claim_id",
        "claim id",
        "hidden provenance",
    ):
        assert forbidden not in policy, f"policy leaks {forbidden}"
        assert forbidden not in docs, f"tool doc leaks {forbidden}"


# ---------------------------------------------------------------------------
# End-to-end EnvironmentEvaluator reward
# ---------------------------------------------------------------------------


def _mk_tool_message(traj, tools, cid, name, args):
    tc = ToolCall(id=cid, name=name, arguments=args, requestor="assistant")
    traj.append(AssistantMessage(role="assistant", tool_calls=[tc]))
    tools.db.messages.append({"role": "assistant", "content": None})
    res = getattr(tools, name)(**args)
    traj.append(ToolMessage(role="tool", id=cid, content=res, requestor="assistant"))
    tools.db.messages.append({"role": "tool", "content": res})
    return res


def _reference_trajectory(node_ids: tuple = ("a", "b", "c", "d", "e", "f")):
    # Per-node visible attributes for the quotation sales stakeholder (matches
    # tasks.json known_info). Hidden attributes must be left unset so the
    # reference trajectory obeys the stakeholder contract.
    sid_to_truth = {"a": "r", "b": "cc", "c": "cq", "d": "ap", "e": "sq", "f": "me"}
    sc = get_scenario(SCENARIO)
    assert sc is not None
    visible_by_sid = {
        sid: sc.stakeholder.visible_attributes_for(tid)
        for sid, tid in sid_to_truth.items()
    }
    include = set(node_ids)
    traj = []
    tools = InterviewTools(InterviewDB())
    cid = 0
    traj.append(
        AssistantMessage(role="assistant", content="Hello, I'd like to interview you.")
    )
    tools.db.messages.append(
        {"role": "assistant", "content": "Hello, I'd like to interview you."}
    )
    _mk_tool_message(
        traj, tools, "c0", "start_inference", {"name": "Quotation creation"}
    )

    node_specs = [
        (sid, t[1], t) for sid, t in zip("abcdef", _TRUTH_NODES) if sid in include
    ]
    sm = 0
    created: set[str] = set()
    for sid, action, (_, _, actor, system, reads, writes) in node_specs:
        cid += 1
        sm += 1
        statement = _node_obs_text(action, actor, system, reads, writes)
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_message", {"message_id": f"sm_{sm}"}
        )
        cid += 1
        vis = visible_by_sid[sid]
        read_cids, write_cids = [], []
        if "reads" in vis:
            for truth_cid in reads:
                agent_cid, label = _AGENT_CONCEPTS[truth_cid]
                if agent_cid not in created:
                    created.add(agent_cid)
                    cid += 1
                    _mk_tool_message(
                        traj,
                        tools,
                        f"c{cid}",
                        "create_concept",
                        {
                            "concept_id": agent_cid,
                            "label": label,
                            "observation_id": oid,
                        },
                    )
                read_cids.append(agent_cid)
        if "writes" in vis:
            for truth_cid in writes:
                agent_cid, label = _AGENT_CONCEPTS[truth_cid]
                if agent_cid not in created:
                    created.add(agent_cid)
                    cid += 1
                    _mk_tool_message(
                        traj,
                        tools,
                        f"c{cid}",
                        "create_concept",
                        {
                            "concept_id": agent_cid,
                            "label": label,
                            "observation_id": oid,
                        },
                    )
                write_cids.append(agent_cid)
        args = {
            "node_id": sid,
            "action": action,
            "primitive": _PRIM_BY_SID[sid],
            "observation_id": oid,
        }
        if "actor" in vis:
            args["actor"] = actor
        if "system" in vis:
            args["system"] = system
        if "reads" in vis:
            args["reads"] = read_cids
        if "writes" in vis:
            args["writes"] = write_cids
        _mk_tool_message(traj, tools, f"c{cid}", "add_node", args)
    if "d" in include:
        cid += 1
        sm += 1
        statement = "The approval is for credit risk management."
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_message", {"message_id": f"sm_{sm}"}
        )
        cid += 1
        _mk_tool_message(
            traj,
            tools,
            f"c{cid}",
            "set_node_necessity",
            {
                "node_id": "d",
                "rationale": "for credit risk management",
                "observation_id": oid,
            },
        )
    cid += 1
    _mk_tool_message(traj, tools, f"c{cid}", "set_node_necessity", {"node_id": "f"})

    action_by_sid = {sid: t[1] for sid, t in zip("abcdef", _TRUTH_NODES)}
    edge_specs = [
        ("e1", "a", "b", None),
        ("e2", "b", "c", None),
        ("e3", "c", "d", "amount over 1,000,000"),
        ("e4", "c", "e", "amount at or below 1,000,000"),
        ("e5", "d", "e", None),
        ("e6", "c", "f", "month-end"),
    ]
    for eid, frm, to, pred in edge_specs:
        if frm not in include or to not in include:
            continue
        cid += 1
        sm += 1
        fa, ta = action_by_sid[frm], action_by_sid[to]
        statement = f"After {fa}, we {ta}." + (f" when {pred}." if pred else "")
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_message", {"message_id": f"sm_{sm}"}
        )
        cid += 1
        args = {"edge_id": eid, "from_node": frm, "to_node": to, "observation_id": oid}
        if pred:
            args["predicate"] = pred
        _mk_tool_message(traj, tools, f"c{cid}", "add_edge", args)

    cid += 1
    _mk_tool_message(
        traj,
        tools,
        f"c{cid}",
        "set_dag_endpoints",
        {"start_node_id": "a", "end_node_ids": ["e", "f"]},
    )
    cid += 1
    _mk_tool_message(
        traj,
        tools,
        f"c{cid}",
        "finish_interview",
        {"summary": "Inferred the quotation DAG."},
    )
    return traj


def test_evaluator_rewards_full_reconstruction():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_reference_trajectory(),
        solo_mode=False,
        env_kwargs={},
    )
    assert reward_info is not None
    assert reward_info.reward == 1.0
    checks = {
        c.env_assertion.func_name: c.met for c in (reward_info.env_assertions or [])
    }
    assert checks == {
        "assert_finish_interview": True,
        "assert_dag_reconstructed": True,
        "assert_necessity_handled": True,
        "assert_evidence_backed": True,
    }
    diag = (reward_info.info or {})["diagnostics"]
    assert diag["quality_pass"] is True


def test_evaluator_detects_missing_node():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_reference_trajectory(node_ids=("a", "b", "c", "e", "f")),
        solo_mode=False,
        env_kwargs={},
    )
    assert reward_info is not None
    assert reward_info.reward == 0.0
    checks = {
        c.env_assertion.func_name: c.met for c in (reward_info.env_assertions or [])
    }
    assert checks["assert_dag_reconstructed"] is False


def test_evaluator_detects_fabricated_necessity():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    traj = _reference_trajectory()
    for m in traj:
        if isinstance(m, AssistantMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if (
                    tc.name == "set_node_necessity"
                    and tc.arguments.get("node_id") == "f"
                ):
                    tc.arguments["rationale"] = "for accounting reconciliation"
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
    assert checks["assert_necessity_handled"] is False
    assert (reward_info.info or {})["diagnostics"]["fabricated_necessity"] is True


# ---------------------------------------------------------------------------
# Tasks / split
# ---------------------------------------------------------------------------


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert [t.id for t in tasks] == ALL_TASK_IDS
    assert set(get_tasks_split()["base"]) == set(ALL_TASK_IDS)
    assert set(get_tasks_split()["base_en"]) == {SCENARIO}
    assert set(get_tasks_split()["base_ja"]) == {JA_SCENARIO}


def test_lab_task_present():
    assert get_scenario(LAB_SCENARIO) is not None
    assert any(t.id == LAB_SCENARIO for t in get_tasks())


# ---------------------------------------------------------------------------
# Stakeholder prompt fidelity rules (deterministic config/prompt checks)
# ---------------------------------------------------------------------------


def _task_instructions(task_id: str) -> str:
    task = next(t for t in get_tasks() if t.id == task_id)
    instructions = task.user_scenario.instructions
    return instructions.task_instructions or ""  # type: ignore[attr-defined]


def _task_instructions_norm(task_id: str) -> str:
    """task_instructions with whitespace collapsed (line-wrapped fragments)."""
    return " ".join(_task_instructions(task_id).split()).lower()


def _task_scenario_str(task_id: str) -> str:
    task = next(t for t in get_tasks() if t.id == task_id)
    return str(task.user_scenario).lower()


def test_stakeholder_ground_truth_only_no_speculation_rule():
    ti = _task_instructions(SCENARIO).lower()
    assert "answer only from your known info" in ti
    assert "only source of business facts" in ti
    assert "plausible does not mean known" in ti
    assert "do not use general business common sense to fill gaps" in ti
    assert "do not add any process step" in ti
    # The rule is also visible in the full prompt the LLM actually receives.
    assert "plausible does not mean known" in _task_scenario_str(SCENARIO)


def test_stakeholder_no_false_denial_of_known_fact():
    ti = _task_instructions(SCENARIO).lower()
    assert "never deny a fact that is in your known info" in ti


def test_stakeholder_negative_answer_relevant_fact_check():
    ti = _task_instructions(SCENARIO).lower()
    assert "before giving a negative answer" in ti
    assert "re-check whether any known fact is relevant" in ti


def test_stakeholder_progressive_disclosure_no_volunteering():
    ti = _task_instructions(SCENARIO).lower()
    assert "do not dump everything at once" in ti
    assert "keep unasked facts to yourself" in ti
    assert "normal flow" in ti and "month-end" in ti and "approval" in ti
    assert "periodic" in ti or "recurring" in ti or "monthly" in ti


def test_stakeholder_unknown_no_speculation():
    ti = _task_instructions(SCENARIO).lower()
    assert "if asked about something not in your known info, do not guess" in ti
    assert "never invent, guess, or speculate" in ti


def test_stakeholder_quotation_instructions_consistent_with_truth():
    ti = _task_instructions(SCENARIO).lower()
    # approval -> credit risk (a visible Known fact)
    assert "approval" in ti and "credit risk management" in ti
    # month-end -> exists as a known fact but rationale is unknown
    assert "month-end excel" in ti
    assert "you do not know the reason" in ti
    assert "vague impression that accounting needs it" in ti


def test_stakeholder_instructions_ja_present():
    ti = _task_instructions(JA_SCENARIO)
    assert ti  # non-empty
    assert "plausible" in ti.lower() or "plausible は known ではありません" in ti
    assert "否定" in ti  # no-false-denial rule present
    assert "月末" in ti and "与信リスク管理" in ti


def test_stakeholder_instructions_lab_present():
    ti = _task_instructions(LAB_SCENARIO).lower()
    assert "answer only from your known info" in ti
    assert "plausible does not mean known" in ti
    assert "never invent anything you do not know" in ti


# ---------------------------------------------------------------------------
# Terminology alignment contract (Agent policy + stakeholder instructions)
# ---------------------------------------------------------------------------


def _policy() -> str:
    return " ".join(BUSINESS_INTERVIEW_POLICY_PATH.read_text().split()).lower()


def test_agent_policy_requires_terminology_establishment():
    """The Agent policy must instruct terminology discipline: identify concepts,
    use the stakeholder's own terms, establish stable labels, clarify, and use
    one agreed term consistently."""
    p = _policy().lower()
    assert "terminology discipline" in p
    assert "shared working vocabulary" in p
    assert "use the stakeholder's own terminology by default" in p
    assert "establish stable labels early" in p
    assert "may refer to the same concept" in p
    assert "one agreed term consistently" in p
    assert "do not invent synonyms" in p
    assert "do not repeatedly re-confirm" in p


def test_agent_policy_does_not_hardcode_domain_terms():
    """Terminology discipline must stay domain-independent: no quotation or lab
    business-object names hard-coded as *labels to adopt*. The "seasoned
    chamber" example is allowed ONLY as a forbidden-invention example (the
    objective requires the policy to warn against inventing it)."""
    p = _policy().lower()
    for banned in ("quotation", "quote", "sales", "crm"):
        assert banned not in p, f"policy hard-codes domain term: {banned}"
    # forbidden-example mention is required, not a hard-coded label
    assert "seasoned chamber" in p
    assert "must not record an output" in p


def test_agent_policy_first_person_role_not_coreference():
    """The policy must tell the Agent to resolve first-person speech to the
    business role WITHOUT forcing the stakeholder to stop using "I"."""
    p = _policy().lower()
    assert "first-person" in p
    assert "business role" in p
    assert "who are you in this process" in p
    assert "do not make the stakeholder replace" in p


def test_agent_policy_no_derived_artifact_invention():
    """The policy must forbid inventing derived artifacts the stakeholder never
    names (e.g. "seasoned chamber")."""
    p = _policy().lower()
    assert "do not invent derived artifacts" in p
    assert "environment chamber" in p  # the allowed stable system-name example
    assert "seasoned chamber" in p  # mentioned only as the forbidden output
    assert "unless the stakeholder actually names it" in p


def test_stakeholder_instructions_require_honoring_agreed_terms():
    """Stakeholder instructions must require consistent use of explicitly
    agreed terminology."""
    ti = _task_instructions_norm(SCENARIO)
    assert "terminology agreements" in ti
    assert "explicitly proposes a name" in ti
    assert "use that agreed term consistently" in ti


def test_stakeholder_agreement_cannot_add_facts_outside_known_info():
    ti = _task_instructions_norm(SCENARIO)
    assert "agreeing on a name does not create any new business fact" in ti
    assert "only the facts in your known info are true" in ti
    assert "only agree when the proposed name really refers" in ti


def test_stakeholder_ambiguous_identity_not_silently_accepted():
    ti = _task_instructions_norm(SCENARIO)
    assert "if you are not sure whether two things are the same" in ti
    assert "say you do not know instead of agreeing" in ti
    assert "never agree to a name for something that is not in your known info" in ti


def test_stakeholder_instructions_allow_first_person_speech():
    ti = _task_instructions_norm(SCENARIO)
    assert "pronouns are fine" in ti
    assert 'you can keep saying "i" or "we"' in ti
    assert "do not need to replace them with a role name" in ti


def test_stakeholder_instructions_no_gt_vocabulary_exposure():
    """The stakeholder instructions must not reveal hidden GT vocabulary; only
    the known-info vocabulary (approval/credit risk/month-end Excel) may appear."""
    ti = _task_instructions_norm(SCENARIO)
    for banned in (
        "sent_quote",
        "excel_summary",
        "quoting system",
        "month_end_summary",
        "amount over 1,000,000",
    ):
        assert banned not in ti, f"stakeholder instructions expose GT term: {banned}"
    # natural known-info vocabulary still allowed
    assert "quotation" in ti or "approval" in ti


def test_terminology_contract_en_ja_equivalent():
    """EN and JA quotation instructions must carry equivalent terminology
    agreement rules."""
    en = _task_instructions_norm(SCENARIO)
    ja = _task_instructions(JA_SCENARIO)
    assert "## terminology agreements" in _task_instructions(SCENARIO).lower()
    assert "## 用語の合意" in ja
    # each language must cover: consistent use after agreement, no new facts,
    # don't-know on ambiguity, no hidden labels, pronouns ok
    for fragment in (
        "use that agreed term consistently",
        "does not create any new business fact",
        "say you do not know instead of agreeing",
        "hidden or official labels",
        "pronouns are fine",
    ):
        assert fragment in en, fragment
    ja_norm = " ".join(ja.split())
    for fragment in (
        "一貫して使って",
        "新しい業務事実を作り出しません",
        "「分からない」と言って",
        "隠れた正式ラベル",
        "代名詞は問題ありません",
    ):
        assert fragment in ja_norm, fragment


def test_terminology_contract_applies_to_lab():
    """lab_sample_flow must receive the same general terminology contract (no
    quotation-specific wording)."""
    ti = _task_instructions_norm(LAB_SCENARIO)
    assert "## terminology agreements" in _task_instructions(LAB_SCENARIO).lower()
    assert "use that agreed term consistently" in ti
    assert "does not create any new business fact" in ti
    assert "pronouns are fine" in ti
    assert "quotation" not in ti  # lab prompt must not reference quotation
    assert "quotation_workflow" not in ti


# ---------------------------------------------------------------------------
# Conservative node matching (missing nodes must NOT be matched away)
# ---------------------------------------------------------------------------

from tau2.domains.business_interview.evaluation import _match_nodes  # noqa: E402


def _drop_node(tools: InterviewTools, nid: str) -> None:
    """Remove a node and any incident edges from the built DAG."""
    dag = tools.db.dag
    assert dag is not None
    dag.nodes.pop(nid, None)
    for eid in list(dag.edges):
        e = dag.edges[eid]
        if e.from_node == nid or e.to_node == nid:
            del dag.edges[eid]


def _add_node(tools: InterviewTools, nid: str, action: str) -> None:
    oid = _claim_obs(tools, action)
    tools.add_node(nid, action, observation_id=oid)


def test_missing_month_end_node_lowers_node_recall():
    """When the month-end node is absent, node_recall must be < 1.0 (not
    salvaged by fuzzy-matching an unrelated node onto it)."""
    tools = _tools()
    _build(tools, evidence=True)
    _drop_node(tools, "f")  # month-end node removed
    res = _eval(tools)
    assert res.node_recall < 1.0


def test_missing_approval_node_lowers_node_recall():
    """When the approval node is absent, node_recall must be < 1.0."""
    tools = _tools()
    _build(tools, evidence=True)
    _drop_node(tools, "d")  # approval node removed
    res = _eval(tools)
    assert res.node_recall < 1.0


def test_fabricated_node_not_mapped_to_missing_truth_node():
    """An unrelated/fabricated agent node must not be assigned to a Truth node
    that is actually missing (approval here)."""
    tools = _tools()
    _build(tools, evidence=True)
    _drop_node(tools, "d")  # approval absent
    _add_node(tools, "fab", "handle escalation to the legal team")
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert tools.db.dag is not None
    mapping = _match_nodes(tools.db.dag, sc.truth, sc.spec)
    assert "fab" not in mapping  # fabricated node stays unmatched
    assert "d" not in set(mapping.values())  # approval truth node stays unmatched
    assert _eval(tools).node_recall < 1.0


def test_weak_single_token_overlap_does_not_match():
    """A node sharing only one weak common token with a Truth node must not be
    assigned to it."""
    tools = _tools()
    _build(tools, evidence=True)
    _add_node(tools, "weak", "send the file")
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert tools.db.dag is not None
    mapping = _match_nodes(tools.db.dag, sc.truth, sc.spec)
    assert "weak" not in mapping
    assert _eval(tools).node_precision < 1.0


def test_valid_quotation_reconstruction_recall_precision_1():
    """The valid quotation reference reconstruction still maps 1:1."""
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0


def test_arbitrary_ids_and_reasonable_paraphrase_match():
    """Arbitrary agent node ids with reasonable paraphrases (not exact hidden
    expressions) must still match the correct Truth nodes 1:1."""
    tools = _tools()
    node_data = [
        ("n0", "we receive a quotation request from the customer"),
        ("n1", "check the customer details in the CRM"),
        ("n2", "create the quote in the quoting system"),
        ("n3", "get manager approval for high value quotes"),
        ("n4", "send the quotation to the customer by email"),
        ("n5", "at month end send the quotation summary to accounting"),
    ]
    for nid, action in node_data:
        _add_node(tools, nid, action)
    # Build a structurally valid DAG so finish_interview (which validates)
    # succeeds: n0 -> n1 -> n2; n2 -> n3 -> n4 (approval -> send);
    # n2 -> n4 (direct send) and n2 -> n5 (month-end summary).
    for eid, frm, to, pred in [
        ("z1", "n0", "n1", None),
        ("z2", "n1", "n2", None),
        ("z3", "n2", "n3", "amount over 1,000,000"),
        ("z4", "n2", "n4", "amount at or below 1,000,000"),
        ("z5", "n3", "n4", None),
        ("z6", "n2", "n5", "month-end"),
    ]:
        args = {"edge_id": eid, "from_node": frm, "to_node": to}
        if pred:
            args["predicate"] = pred
        tools.add_edge(**args)
    tools.set_dag_endpoints(start_node_id="n0", end_node_ids=["n4", "n5"])
    tools.finish_interview()
    res = _eval(tools)
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0


def test_approval_node_not_mismatched_to_month_end():
    """Regression: the DeepSeek smoke failure. An approval node whose action
    mentions send/quotation/customer must map to the approval Truth node (ap),
    NOT to the month-end node (me), so a missing month-end is not hidden."""
    tools = _tools()
    _build(tools, evidence=True)
    _drop_node(tools, "f")  # month-end absent
    # Re-point the approval node's action at the smoke-style wording.
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["d"].action = InferredValue(
        value="Get approval from a manager before sending the quotation to the customer",
        confidence=1.0,
    )
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert tools.db.dag is not None
    mapping = _match_nodes(tools.db.dag, sc.truth, sc.spec)
    assert mapping["d"] == "ap"  # approval node -> approval Truth node
    assert "me" not in set(mapping.values())  # month-end stays unmatched
    res = _eval(tools)
    assert res.node_recall < 1.0  # missing month-end reflected


def test_edge_metrics_not_inflated_by_node_mismatch():
    """Edge recall/precision must not be inflated by a node being mis-mapped
    onto a missing Truth node. With month-end absent and no month-end node, the
    month-end edge (e6) must not count toward recall."""
    tools = _tools()
    _build(tools, evidence=True)
    _drop_node(tools, "f")  # month-end absent (drops e6 c->f)
    res = _eval(tools)
    # Truth edge e6 (cq->me) is unmatchable; recall is over 6 truth edges.
    assert res.edge_recall <= 5 / 6


def test_en_ja_matching_still_equivalent_after_conservative_gate():
    """EN and JA full reconstructions remain equivalent under the new gate."""
    en = _tools()
    _build(en, ja=False)
    ja = _tools()
    _build(ja, ja=True)
    ren = _eval(en, SCENARIO)
    rja = _eval(ja, JA_SCENARIO)
    assert ren.node_recall == 1.0 and rja.node_recall == 1.0
    assert ren.node_precision == 1.0 and rja.node_precision == 1.0
    assert ren.quality_pass is True and rja.quality_pass is True


# ---------------------------------------------------------------------------
# Observation capture UX (stable stakeholder message ids, no turn_idx)
# ---------------------------------------------------------------------------


def _ingest_and_observe(tools: InterviewTools, text: str) -> str:
    """Add a user message then capture it via latest id + observe_message."""
    _ingest(tools, "user", text)
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


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
    assert "turn" not in listing  # primary id is the stable sm_ id, not a turn index


def test_observe_message_by_id_creates_correct_observation():
    tools = _tools()
    _ingest(tools, "user", "First statement.")
    _ingest(tools, "user", "Second statement.")
    oid = tools.observe_message("sm_2")
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "Second statement."
    assert obs.source_id == "stakeholder"
    # observation's turn points at the real ledger index of that user message
    assert tools.db.messages[obs.turn]["content"] == "Second statement."
    # re-capture is idempotent
    assert tools.observe_message("sm_2") == oid


def test_invalid_and_fabricated_message_ids_rejected():
    tools = _tools()
    _ingest(tools, "user", "Only one real message.")
    for bad in ("sm_0", "sm_2", "sm_99", "turn_3", "foo"):
        with pytest.raises(ValueError):
            tools.observe_message(bad)
    # no user messages at all -> latest fails
    empty = _tools()
    with pytest.raises(ValueError):
        empty.observe_latest_stakeholder_message()


def test_assistant_tool_messages_cannot_be_observed():
    tools = _tools()
    _ingest(tools, "assistant", "I ask a question.")
    _ingest(tools, "user", "A real stakeholder statement.")
    _ingest(tools, "tool", "a tool result")
    listing = tools.list_stakeholder_messages()
    assert listing.count("sm_") == 1  # only the user message got an sm id
    assert "sm_1" in listing
    sm_id = tools.observe_latest_stakeholder_message()
    assert sm_id == "sm_1"
    oid = tools.observe_message(sm_id)
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "A real stakeholder statement."


def test_observe_latest_returns_newest_user_message_id():
    tools = _tools()
    _ingest_and_observe(tools, "First.")
    _ingest(tools, "assistant", "ok")
    assert tools.observe_latest_stakeholder_message() == "sm_1"  # still first
    oid = _ingest_and_observe(tools, "Second.")
    assert tools.observe_latest_stakeholder_message() == "sm_2"
    obs = next(o for o in tools.db.observations if o.id == oid)
    assert obs.text == "Second."
    assert obs.order == 1  # second captured observation


def test_multiple_stakeholder_messages_ids_stable_and_unique():
    tools = _tools()
    texts = [f"message number {i}" for i in range(5)]
    for t in texts:
        _ingest(tools, "user", t)
    listing = tools.list_stakeholder_messages()
    ids = [ln.split(":")[0].strip() for ln in listing.splitlines()]
    assert ids == ["sm_1", "sm_2", "sm_3", "sm_4", "sm_5"]
    assert len(set(ids)) == len(ids)  # unique
    # each maps to the right text
    for i, t in enumerate(texts):
        oid = tools.observe_message(f"sm_{i + 1}")
        obs = next(o for o in tools.db.observations if o.id == oid)
        assert obs.text == t


def test_observation_capture_survives_set_state_replay():
    """Replaying a conversation via environment.set_state restores the same
    stable sm ids and observations (deterministic, append-only ledger)."""
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
        env.on_message(msg)  # mirror the live ledger (all roles appended)
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

    # Build a fresh environment and replay the same trajectory via set_state.
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
    texts = {o.text for o in replay.tools.db.observations}
    assert "First stakeholder statement." in texts
    assert "Second stakeholder statement." in texts
    assert oid1 in {o.id for o in replay.tools.db.observations}
    assert oid2 in {o.id for o in replay.tools.db.observations}


# ---------------------------------------------------------------------------
# DAG refinement / cleanup (working hypothesis, no obsolete coarse nodes)
# ---------------------------------------------------------------------------


def test_remove_node_after_decomposition_leaves_no_dangling_edge():
    tools = _tools()
    tools.start_inference("q")
    tools.add_node("prepare_quotation", "prepare the quotation")
    tools.add_node("record_request", "record the quotation request")
    tools.add_node("check_customer", "check the customer in the CRM")
    tools.add_node("create_quotation", "create the quotation")
    tools.add_edge("e1", "prepare_quotation", "check_customer")
    tools.add_edge("e2", "check_customer", "create_quotation")
    # decompose the coarse node and remove it
    tools.remove_node("prepare_quotation")
    dag = tools.db.dag
    assert dag is not None
    assert "prepare_quotation" not in dag.nodes
    # the incident edge e1 is removed; e2 (between remaining nodes) survives
    assert "e1" not in dag.edges
    assert "e2" in dag.edges
    # no dangling edge references the removed node
    for e in dag.edges.values():
        assert e.from_node != "prepare_quotation"
        assert e.to_node != "prepare_quotation"


def test_remove_node_keeps_observations_and_provenance():
    tools = _tools()
    tools.start_inference("q")
    _ingest(tools, "user", "We prepare the quotation by hand.")
    oid = tools.observe_message("sm_1")
    tools.add_node("coarse", "prepare the quotation", observation_id=oid)
    # attach the same observation to a surviving node too
    tools.add_node("fine", "record the request")
    tools.attach_observation("fine", oid)
    tools.remove_node("coarse")
    # the Observation itself is kept, and its provenance on the surviving node
    # is preserved
    assert any(o.id == oid for o in tools.db.observations)
    dag = tools.db.dag
    assert dag is not None
    assert oid in dag.nodes["fine"].observation_ids
    assert "coarse" not in dag.nodes


def test_remove_node_nonexistent_rejected():
    tools = _tools()
    tools.start_inference("q")
    with pytest.raises(ValueError):
        tools.remove_node("does_not_exist")


def test_finish_rejects_unreachable_node():
    tools = _tools()
    tools.start_inference("q")
    tools.add_node("a", "start")
    tools.add_node("b", "end")
    tools.add_edge("e1", "a", "b")
    tools.add_node("orphan", "orphan step")  # unreachable
    tools.set_dag_endpoints(start_node_id="a", end_node_ids=["b"])
    with pytest.raises(ValueError) as ei:
        tools.finish_interview()
    assert "unreachable node: orphan" in str(ei.value)
    assert tools.db.interview_complete is False


def test_finish_accepts_valid_dag():
    tools = _tools()
    tools.start_inference("q")
    tools.add_node("a", "start")
    tools.add_node("b", "end")
    tools.add_edge("e1", "a", "b")
    tools.set_dag_endpoints(start_node_id="a", end_node_ids=["b"])
    tools.finish_interview()
    assert tools.db.interview_complete is True


def test_finish_rejection_then_fix_and_refinish():
    tools = _tools()
    tools.start_inference("q")
    tools.add_node("a", "start")
    tools.add_node("b", "end")
    tools.add_node("orphan", "orphan step")
    tools.add_edge("e1", "a", "b")
    tools.set_dag_endpoints(start_node_id="a", end_node_ids=["b"])
    with pytest.raises(ValueError):
        tools.finish_interview()
    # fix by removing the orphan, then finish again
    tools.remove_node("orphan")
    tools.finish_interview()
    assert tools.db.interview_complete is True


def test_validate_dag_reports_structural_errors_then_clean():
    tools = _tools()
    tools.start_inference("q")
    tools.add_node("a", "start")
    tools.add_node("b", "end")
    tools.add_edge("e1", "a", "b")
    # start not declared yet -> invalid
    assert "start_node_id must be set" in tools.validate_dag()
    tools.set_dag_endpoints(start_node_id="a", end_node_ids=["b"])
    assert tools.validate_dag() == "DAG is structurally valid."


def test_coarse_node_refinement_end_to_end_valid():
    """Reproduce the smoke failure: coarse receive->prepare->send refined into
    receive->record->check->create->send, removing the coarse prepare node, so
    the final DAG is valid and contains no obsolete node."""
    tools = _tools()
    tools.start_inference("q")
    for nid, action in [
        ("receive_request", "receive quotation request"),
        ("prepare_quotation", "prepare the quotation"),
        ("send_quotation", "send quotation to customer"),
    ]:
        tools.add_node(nid, action)
    tools.add_edge("e1", "receive_request", "prepare_quotation")
    tools.add_edge("e2", "prepare_quotation", "send_quotation")
    # detailed interview reveals sub-steps; add them
    for nid, action in [
        ("record_request", "record the request"),
        ("check_customer", "check customer in the CRM"),
        ("create_quotation", "create quotation in the quoting system"),
    ]:
        tools.add_node(nid, action)
    # reconnect around the refined sub-steps
    tools.add_edge("e3", "receive_request", "record_request")
    tools.add_edge("e4", "record_request", "check_customer")
    tools.add_edge("e5", "check_customer", "create_quotation")
    tools.add_edge("e6", "create_quotation", "send_quotation")
    # remove the now-obsolete coarse node (and its incident edges e1/e2)
    tools.remove_node("prepare_quotation")
    tools.set_dag_endpoints(
        start_node_id="receive_request", end_node_ids=["send_quotation"]
    )
    # no obsolete / unreachable nodes; finish succeeds
    assert tools.validate_dag() == "DAG is structurally valid."
    tools.finish_interview()
    assert tools.db.interview_complete is True
    dag = tools.db.dag
    assert dag is not None
    assert "prepare_quotation" not in dag.nodes
    assert all(
        nid in dag.nodes
        for nid in [
            "receive_request",
            "record_request",
            "check_customer",
            "create_quotation",
            "send_quotation",
        ]
    )


# ---------------------------------------------------------------------------
# Stakeholder-visibility contract (scoring aligned with known_info)
#
# The normal ``_build`` / ``_eval`` helpers above are already stakeholder-aware:
# ``_build`` asserts only stakeholder-visible quotation attributes and ``_eval``
# evaluates under the scenario's StakeholderFilter. These tests pin the contract.
# ---------------------------------------------------------------------------


def test_faithful_contract_reconstruction_achieves_structural_pass():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.structural_pass is True
    assert res.actor_correctness == 1.0
    assert res.system_correctness == 1.0
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
    # hidden attrs left unset should also keep evidence hygiene intact
    assert res.evidence_pass is True
    assert res.quality_pass is True


def test_leaving_ap_system_and_unsupported_reads_writes_unset_is_correct():
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    # ap.system, ap.reads, ap.writes, sq.reads, sq.writes, me.reads are hidden;
    # leaving them unset is correct, so each axis scores 1.0.
    assert res.system_correctness == 1.0
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0


def test_asserting_hidden_ap_system_is_incorrect():
    tools = _tools()
    _build(tools)
    # Asserting ap.system=quoting (which equals the hidden Truth) is STILL wrong:
    # the stakeholder cannot know it, so epistemic restraint is required.
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["d"].system = InferredValue(
        value="quoting", confidence=1.0, observation_ids=["obs_x"]
    )
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_asserting_hidden_ap_reads_is_incorrect():
    tools = _tools()
    _build(tools)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["d"].reads = [
        ConceptRef(concept_id="quote", confidence=1.0, observation_ids=["obs_x"])
    ]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_visible_attribute_still_fails():
    tools = _tools()
    _build(tools)
    # cq.system IS visible; a wrong value must still fail.
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["c"].system = InferredValue(
        value="erp", confidence=1.0, observation_ids=["obs_x"]
    )
    res = _eval(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_contract_preserves_topology_predicate_necessity_evidence():
    # Topology / predicates / necessity / evidence requirements are unchanged:
    # the faithful-contract build still passes all gates.
    tools = _tools()
    _build(tools)
    res = _eval(tools)
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0
    assert res.edge_recall == 1.0
    assert res.edge_precision == 1.0
    assert res.predicate_correctness == 1.0
    assert res.necessity_pass is True
    assert res.evidence_pass is True
    assert res.provenance_authenticity_pass is True
    assert res.structural_pass is True


def test_contract_en_ja_equivalent():
    en = _tools()
    _build(en, ja=False)
    ja = _tools()
    _build(ja, ja=True)
    ren = _eval(en, SCENARIO)
    rja = _eval(ja, JA_SCENARIO)
    for attr in (
        "structural_pass",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "necessity_pass",
        "evidence_pass",
    ):
        assert getattr(ren, attr) == getattr(rja, attr), attr
    assert ren.structural_pass is True


def test_contract_lab_sample_flow_still_works():
    # The lab known_info grounds actor/system and the raw sample input, but the
    # GT read/write artifacts (accessioned sample, seasoned chamber, conditioned
    # sample, batch approval) are hidden derived labels. A faithful build that
    # asserts only visible attrs must pass under the same visibility-aware
    # evaluator.
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.quality_pass is True
    assert res.structural_pass is True
    assert res.actor_correctness == 1.0
    assert res.system_correctness == 1.0
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0


def test_lab_faithful_known_info_reconstruction_passes():
    """A faithful lab reconstruction that asserts only stakeholder-supported
    attributes (per known_info) achieves structural_pass."""
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.structural_pass is True
    assert res.quality_pass is True


def test_lab_hidden_derived_attr_left_unset_is_correct():
    """Hidden derived lab artifacts (seasoned chamber, conditioned sample,
    batch approval, accessioned sample as write) left unset are correct."""
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.write_correctness == 1.0
    assert res.read_correctness == 1.0
    # verify hidden writes really are unset in the faithful build
    dag = tools.db.dag
    assert dag is not None
    assert dag.nodes["n2"].writes == []
    assert dag.nodes["n4"].writes == []


def test_lab_asserting_hidden_derived_attr_is_penalized():
    """Asserting a hidden derived lab artifact (e.g. n2 writes "seasoned
    chamber") is penalized even though it equals the hidden Truth."""
    tools = _tools()
    _build_lab(tools)
    dag = tools.db.dag
    assert dag is not None
    tools.create_concept("seasoned", "seasoned chamber")
    dag.nodes["n2"].writes = [
        ConceptRef(concept_id="seasoned", confidence=1.0, observation_ids=["obs_x"])
    ]
    res = _eval(tools, LAB_SCENARIO)
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


def test_lab_visible_facts_still_required():
    """Genuinely visible lab facts remain required: a wrong actor or a missing
    environment-chamber system must still fail."""
    # wrong actor on a visible node (pick a value that does not normalize to
    # the correct "manager"/"lab supervisor" role)
    tools = _tools()
    _build_lab(tools)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["n4"].actor = InferredValue(
        value="quality intern", confidence=1.0, observation_ids=["obs_x"]
    )
    assert _eval(tools, LAB_SCENARIO).actor_correctness < 1.0
    # missing environment-chamber system on n2 (visible)
    tools2 = _tools()
    _build_lab(tools2)
    dag2 = tools2.db.dag
    assert dag2 is not None
    dag2.nodes["n2"].system = InferredValue()
    assert _eval(tools2, LAB_SCENARIO).system_correctness < 1.0


def test_lab_topology_and_open_world_behavior_still_pass():
    """Topology and open-world primitive behavior remain intact under the lab
    visibility contract."""
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.node_recall == 1.0
    assert res.node_precision == 1.0
    assert res.edge_recall == 1.0
    assert res.edge_precision == 1.0
    assert res.predicate_correctness == 1.0
    assert res.necessity_pass is True
    assert res.evidence_pass is True
    assert res.structural_pass is True
    # open-world: unclassified primitive remains valid (no primitive axis gating)
    dag = tools.db.dag
    assert dag is not None
    dag.nodes["n2"].primitive = InferredValue(value="unclassified", confidence=1.0)
    assert _eval(tools, LAB_SCENARIO).quality_pass is True


def test_default_evaluate_binds_data_via_claim_catalog():
    """INTENTIONAL default-path test.

    Reads/writes bind ONLY through the hidden claim catalog regardless of
    whether a stakeholder is passed: a hidden-asserted ref cannot bind even
    without a stakeholder, because the catalog is visibility-filtered by
    construction (no claim exists for sq.reads / sq.writes / ...).
    (Actor/system keep the all-visible default: the faithful build leaves
    hidden systems unset, so the default path fails those — the old contract.)
    """
    tools = _tools()
    _build(tools, evidence=True)
    sc = get_scenario(SCENARIO)
    assert sc is not None
    prov = build_provenance_ledger(tools.db, sc.claims, sc.stop_phrases)
    res = evaluate(tools.db, sc.truth, sc.spec, claims=sc.claims, provenance=prov)
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
    assert res.concept_correctness == 1.0
    # hidden sq.reads asserted with VALID quote provenance still fails under
    # the default path: no sq.reads claim exists, so nothing can bind
    tools2 = _tools()
    _build(tools2, evidence=True)
    dag2 = tools2.db.dag
    assert dag2 is not None
    oid = _claim_obs(tools2, "I create the quotation in the quoting system.")
    dag2.nodes["e"].reads = [
        ConceptRef(concept_id="quote", confidence=1.0, observation_ids=[oid])
    ]
    sc2 = get_scenario(SCENARIO)
    assert sc2 is not None
    prov2 = build_provenance_ledger(tools2.db, sc2.claims, sc2.stop_phrases)
    res2 = evaluate(tools2.db, sc2.truth, sc2.spec, claims=sc2.claims, provenance=prov2)
    assert res2.structural_pass is False


# ---------------------------------------------------------------------------
# Agent-local data concepts + hidden stakeholder provenance
#
# The Agent LLM, not the evaluator, interprets stakeholder wording variation:
# it creates local DataConcepts, attaches observed ConceptTerms, and reuses
# concept ids in ConceptRefs. The evaluator binds each local concept to a
# hidden Truth concept ONLY through the hidden provenance ledger (claims.py):
# a ref grounds a claim when the Observations it cites privately support that
# claim. Wording alone never creates a match; hidden axes never bind; the
# same local concept must map to ONE Truth concept; one Truth concept must be
# represented by ONE local concept (merge to repair splits).
# ---------------------------------------------------------------------------


def _grounded_ref(
    tools: InterviewTools,
    truth_cid: str,
    obs_ids: list[str],
) -> ConceptRef:
    """A ConceptRef on an existing agent concept citing the given observations."""
    agent_cid, _ = _AGENT_CONCEPTS[truth_cid]
    dag = tools.db.dag
    assert dag is not None
    assert agent_cid in dag.data_concepts
    return ConceptRef(concept_id=agent_cid, confidence=1.0, observation_ids=obs_ids)


def test_claim_catalog_only_contains_visible_axes():
    """Hidden sq.reads / sq.writes / me.reads / ap.reads|writes have NO claims:
    the private catalog is visibility-filtered by construction."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    assert set(sc.claims) == {
        "r.writes.tc_request",
        "cc.reads.tc_customer",
        "cq.reads.tc_customer",
        "cq.reads.tc_pricing",
        "cq.writes.tc_quote",
        "me.writes.tc_excel_summary",
    }
    for cid in (
        "sq.reads.tc_quote",
        "sq.writes.tc_sent_quote",
        "me.reads.tc_quote",
        "ap.reads.tc_quote",
        "ap.writes.tc_approval",
    ):
        assert cid not in sc.claims


def test_agent_label_quotation_binds_to_truth_quote_without_alias():
    """An agent concept labeled "quotation" binds to Truth tc_quote purely via
    hidden provenance — no quote/quotation synonym table exists anywhere."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    oid = _claim_obs(tools, "I create the quotation in the quoting system.")
    # a NEW agent concept with an arbitrary id and the stakeholder's label
    tools.create_concept("q_doc", "quotation", observation_id=oid)
    dag.nodes["c"].writes = [_grounded_ref(tools, "tc_quote", [oid])]
    # concept id differs from the truth concept id (tc_quote) and no label
    # comparison exists
    assert dag.data_concepts["q_doc"].id != "tc_quote"
    res = _eval(tools)
    assert res.write_correctness == 1.0
    assert res.concept_correctness == 1.0
    assert res.structural_pass is True


def test_elaborated_label_binds_via_supporting_observation():
    """'pricing information from the quoting system' binds to pricing when its
    cited Observation privately supports cq.reads.pricing — the label itself
    is never compared to anything."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    oid = _claim_obs(
        tools, "I use the pricing information that is available in the quoting system."
    )
    tools.create_concept(
        "prc_info", "pricing information from the quoting system", observation_id=oid
    )
    dag.nodes["c"].reads = [
        _grounded_ref(
            tools, "tc_customer", list(dag.nodes["c"].reads[0].observation_ids)
        ),
        ConceptRef(concept_id="prc_info", confidence=1.0, observation_ids=[oid]),
    ]
    res = _eval(tools)
    assert res.read_correctness == 1.0
    assert res.concept_correctness == 1.0


def test_wording_alone_cannot_create_a_match():
    """A ref/concept with NO cited observations cannot ground anything: the
    label text is never used by the evaluator."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    tools.create_concept("mystery", "quotation")  # no observation
    dag.nodes["c"].writes = [
        ConceptRef(concept_id="mystery", confidence=1.0, observation_ids=[])
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_concept_ref_count >= 1
    assert res.structural_pass is False


def test_unrelated_observation_cannot_ground_a_concept_ref():
    """An authentic but unrelated Observation id cannot ground a ref: the
    hidden ledger records no claim support for it, so a concept citing only
    that Observation binds nothing."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    tools.create_concept("pizza_quote", "quotation", observation_id=pizza)
    dag.nodes["c"].writes = [
        ConceptRef(concept_id="pizza_quote", confidence=1.0, observation_ids=[pizza])
    ]
    res = _eval(tools)
    assert res.write_correctness < 1.0
    assert res.unsupported_concept_ref_count >= 1
    assert res.evidence_pass is True  # the obs is real & authentic (hygiene ok)
    assert res.structural_pass is False


def test_one_local_concept_as_customer_and_quote_fails():
    """A single agent concept grounded as both customer (cc.reads) and quote
    (cq.writes) must fail concept binding."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    c_oid = _claim_obs(tools, "I check the customer information in the CRM.")
    q_oid = _claim_obs(tools, "I create the quotation in the quoting system.")
    tools.create_concept("blend", "customer information", observation_id=c_oid)
    tools.add_concept_term("blend", "quotation", observation_id=q_oid)
    dag.nodes["b"].reads = [
        ConceptRef(concept_id="blend", confidence=1.0, observation_ids=[c_oid])
    ]
    dag.nodes["c"].writes = [
        ConceptRef(concept_id="blend", confidence=1.0, observation_ids=[q_oid])
    ]
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False


def test_separate_concepts_for_same_visible_object_fail_until_merged():
    """Two agent concept ids both grounded as the same visible Truth customer
    (cc.reads and cq.reads) fail until merge_concepts re-points them."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    cc_oid = _claim_obs(tools, "I check the customer information in the CRM.")
    cq_oid = _claim_obs(tools, "I create the quotation using customer information.")
    tools.create_concept("cust_a", "customer information", observation_id=cc_oid)
    tools.create_concept("cust_b", "customer information", observation_id=cq_oid)
    dag.nodes["b"].reads = [
        ConceptRef(concept_id="cust_a", confidence=1.0, observation_ids=[cc_oid])
    ]
    dag.nodes["c"].reads = [
        ConceptRef(concept_id="cust_b", confidence=1.0, observation_ids=[cq_oid]),
        dag.nodes["c"].reads[1],
    ]
    res = _eval(tools)
    assert res.concept_correctness == 0.0
    assert res.structural_pass is False
    # repair: merge cust_b into cust_a and re-point every reference
    tools.merge_concepts("cust_a", ["cust_b"])
    assert "cust_b" not in dag.data_concepts
    assert all(
        r.concept_id == "cust_a"
        for r in dag.nodes["c"].reads[:1] + dag.nodes["b"].reads
    )
    res2 = _eval(tools)
    assert res2.concept_correctness == 1.0
    assert res2.structural_pass is True


def test_hidden_sq_assertions_fail_even_with_valid_quote_provenance():
    """sq.reads / sq.writes asserted with Observations that genuinely support
    the quote claim still fail: hidden axes are a prior gate."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    oid = _claim_obs(tools, "I create the quotation in the quoting system.")
    dag.nodes["e"].reads = [_grounded_ref(tools, "tc_quote", [oid])]
    dag.nodes["e"].writes = [_grounded_ref(tools, "tc_quote", [oid])]
    res = _eval(tools)
    assert res.read_correctness < 1.0
    assert res.write_correctness < 1.0
    assert res.structural_pass is False


def test_hidden_unset_still_correct():
    """Hidden axes left unset remain correct while concept binding is active."""
    tools = _tools()
    _build(tools, evidence=True)
    res = _eval(tools)
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
    assert res.concept_correctness == 1.0
    assert res.quality_pass is True


def test_lab_uses_same_concept_mechanism():
    """lab_sample_flow binds through the same hidden-provenance mechanism: a
    faithful build passes; an unsupported visible ref fails; a hidden derived
    artifact assertion fails."""
    tools = _tools()
    _build_lab(tools)
    res = _eval(tools, LAB_SCENARIO)
    assert res.quality_pass is True
    assert res.concept_correctness == 1.0
    # unsupported visible ref: a fresh concept citing only an unrelated
    # observation cannot bind n1.reads (no hidden support for the claim)
    dag = tools.db.dag
    assert dag is not None
    pizza = _claim_obs(tools, "I like pizza on Fridays.")
    tools.create_concept("pizza_sample", "sample", observation_id=pizza)
    dag.nodes["n1"].reads = [
        ConceptRef(concept_id="pizza_sample", confidence=1.0, observation_ids=[pizza])
    ]
    res2 = _eval(tools, LAB_SCENARIO)
    assert res2.read_correctness < 1.0
    assert res2.unsupported_concept_ref_count >= 1
    # hidden derived write asserted still fails
    tools2 = _tools()
    _build_lab(tools2)
    dag2 = tools2.db.dag
    assert dag2 is not None
    tools2.create_concept("seasoned", "seasoned chamber")
    dag2.nodes["n2"].writes = [ConceptRef(concept_id="seasoned", confidence=1.0)]
    assert _eval(tools2, LAB_SCENARIO).write_correctness < 1.0


def test_hidden_provenance_absent_from_agent_visible_state():
    """Claim ids, truth concept ids, and the provenance ledger never appear in
    any Agent-visible output or serialized DB state."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    listing = tools.list_concepts()
    assert "tc_" not in listing
    assert "claim" not in listing
    assert "provenance" not in listing
    # tool outputs and the DB model dump carry no claim ids / provenance
    dumped = tools.db.model_dump(mode="json")
    text = str(dumped)
    assert "tc_request" not in text
    assert "claim" not in text
    assert "provenance" not in text
    assert all("tc_" not in str(c) for c in dag.data_concepts)
    # the ledger lives only in the derivation module output, never in the DB
    assert not hasattr(tools.db, "provenance")
    assert not hasattr(tools.db, "claims")


def test_add_node_rejects_unknown_concept_id():
    """add_node/update_node validate that reads/writes reference existing
    agent-local concepts."""
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("known", "known object")
    tools.add_node("n1", "do something", reads=["known"], writes=None)
    with pytest.raises(ValueError):
        tools.add_node("n2", "do something else", reads=["unknown_concept"])
    with pytest.raises(ValueError):
        tools.update_node("n1", writes=["unknown_concept"])


def test_merge_concepts_repoints_refs_and_folds_terms():
    tools = _tools()
    tools.start_inference("q")
    tools.create_concept("a", "alpha")
    tools.create_concept("b", "beta")
    tools.add_node("n1", "first step", reads=["a"], writes=["b"])
    tools.merge_concepts("a", ["b"])
    dag = tools.db.dag
    assert dag is not None
    assert "b" not in dag.data_concepts
    assert dag.nodes["n1"].reads[0].concept_id == "a"
    assert dag.nodes["n1"].writes[0].concept_id == "a"


def test_derive_utterance_claims_longest_phrase_and_stops():
    """Simulator-side derivation: longest phrase wins, stop phrases consume
    spans, and near-collisions never support the wrong claim."""
    sc = get_scenario(SCENARIO)
    assert sc is not None
    d = derive_utterance_claims
    # "quotation request" supports the request claim and consumes "quotation"
    assert d("I record the quotation request", sc.claims, sc.stop_phrases) == {
        "r.writes.tc_request": ["quotation request"]
    }
    # "quotation information" (the summary's data) supports nothing
    assert (
        d("the quotation information is sent to Accounting", sc.claims, sc.stop_phrases)
        == {}
    )
    # the month-end summary supports only the excel_summary claim
    assert d(
        "At month-end I send a summary of the quotation information to Accounting",
        sc.claims,
        sc.stop_phrases,
    ) == {"me.writes.tc_excel_summary": ["summary of the quotation information"]}
    # the create statement supports quote + customer + pricing claims
    out = d(
        "I create the quotation using customer and pricing information",
        sc.claims,
        sc.stop_phrases,
    )
    assert "cq.writes.tc_quote" in out
    assert "cq.reads.tc_customer" in out
    assert "cq.reads.tc_pricing" in out
    # validation rejects unknown / out-of-visibility claim ids
    with pytest.raises(ValueError):
        validate_ledger(
            {3: {"sq.reads.tc_quote": ["quotation"]}}, sc.claims, sc.stakeholder
        )
    with pytest.raises(ValueError):
        validate_ledger({3: {"fake.claim": ["x"]}}, sc.claims, sc.stakeholder)
    validate_ledger(
        {3: {"cq.writes.tc_quote": ["quotation"]}}, sc.claims, sc.stakeholder
    )


def test_en_ja_concept_equivalence():
    """EN and JA faithful reconstructions behave identically under the same
    concept-binding evaluator (JA claims use JA surface terms)."""
    en = _tools()
    _build(en, ja=False)
    ja = _tools()
    _build(ja, ja=True)
    ren = _eval(en, SCENARIO)
    rja = _eval(ja, JA_SCENARIO)
    for attr in (
        "structural_pass",
        "actor_correctness",
        "system_correctness",
        "read_correctness",
        "write_correctness",
        "concept_correctness",
        "necessity_pass",
        "evidence_pass",
    ):
        assert getattr(ren, attr) == getattr(rja, attr), attr
    assert ren.quality_pass is True and rja.quality_pass is True


def test_concept_refs_ground_only_expected_claims_at_slot():
    """A ref whose Observation supports a DIFFERENT slot's claim (e.g. the
    request utterance cited on cq.reads) is unsupported at this slot."""
    tools = _tools()
    _build(tools, evidence=True)
    dag = tools.db.dag
    assert dag is not None
    oid = _claim_obs(tools, "I record the quotation request.")
    # fresh concepts citing only the request utterance: at cq.reads, that
    # Observation supports only r.writes.tc_request — nothing here binds
    tools.create_concept("req_as_cust", "customer", observation_id=oid)
    tools.create_concept("req_as_prc", "pricing", observation_id=oid)
    dag.nodes["c"].reads = [
        ConceptRef(concept_id="req_as_cust", confidence=1.0, observation_ids=[oid]),
        ConceptRef(concept_id="req_as_prc", confidence=1.0, observation_ids=[oid]),
    ]
    res = _eval(tools)
    assert res.read_correctness < 1.0  # neither ref supports cq.reads claims
    assert res.unsupported_concept_ref_count >= 2
