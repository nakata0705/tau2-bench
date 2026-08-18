"""Tests for the simple open-world DAG business_interview domain (v5).

The domain centers on free-text actions + optional generic primitives (with an
explicit ``unclassified`` sentinel for unknown operations) + authentic Observation
provenance. Evaluation separates **result correctness** (inferred DAG vs hidden
Ground Truth) from **evidence hygiene** (references point at real, authentic
stakeholder Observations). The evaluator does NOT re-interpret Observation text
to decide whether it semantically supports a claim. DiscoveredConcept and
concept-discovery evaluation are removed.
"""

import pytest

import tau2.domains.business_interview.dag as dagmod
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.business_interview.concepts import resolve_primitive
from tau2.domains.business_interview.dag import (
    BusinessDAG,
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

_TRUTH_NODES = [
    ("a", "receive quotation request", "sales", None, [], ["request"]),
    ("b", "check customer information in the CRM", "sales", "crm", ["customer"], []),
    (
        "c",
        "create quotation in the quoting system",
        "sales",
        "quoting",
        ["customer", "pricing"],
        ["quote"],
    ),
    (
        "d",
        "approve high-value quotation",
        "manager",
        "quoting",
        ["quote"],
        ["approval"],
    ),
    (
        "e",
        "send quotation to customer",
        "sales",
        "email",
        ["quote"],
        ["sent_quote"],
    ),
    (
        "f",
        "send quotation summary to accounting at month-end",
        "sales",
        "excel",
        ["quote"],
        ["excel_summary"],
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


def _tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


def _eval(tools: InterviewTools, scenario: str = SCENARIO):
    sc = get_scenario(scenario)
    return evaluate(tools.db, sc.truth, sc.spec)


def _ingest(tools: InterviewTools, role: str = "user", content: str = "") -> int:
    tools.db.messages.append({"role": role, "content": content})
    return len(tools.db.messages) - 1


def _claim_obs(tools: InterviewTools, text: str) -> str:
    _ingest(tools, "user", text)
    sm_id = tools.observe_latest_stakeholder_message()
    return tools.observe_message(sm_id)


def _node_obs_text(action, actor, system, reads, writes) -> str:
    parts = []
    parts.append(
        f"The {actor} performs: {action}." if actor else f"The process: {action}."
    )
    if system:
        parts.append(f"It uses the {system}.")
    if reads:
        parts.append(f"It reads {', '.join(reads)}.")
    if writes:
        parts.append(f"It writes {', '.join(writes)}.")
    return " ".join(parts)


def _build(
    tools: InterviewTools,
    ja: bool = False,
    ids: tuple = ("a", "b", "c", "d", "e", "f"),
    edge_ids: tuple = ("e1", "e2", "e3", "e4", "e5", "e6"),
    evidence: bool = True,
):
    """Build the correct quotation DAG with a quality_pass-quality provenance.

    With ``evidence=True`` each node gets one rich authentic Observation that
    supports all its claims, and each edge an Observation that supports the
    relation (+ predicate). With ``evidence=False`` the same topology is built
    with no provenance (for the evidence-gate failures).
    """
    i1, i2, i3, i4, i5, i6 = ids
    actions = _JA_ACTIONS if ja else [t[1] for t in _TRUTH_NODES]
    node_data = list(zip(_TRUTH_NODES, actions))
    node_map = {"a": i1, "b": i2, "c": i3, "d": i4, "e": i5, "f": i6}
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
                tools, _node_obs_text(action, actor, system, reads, writes)
            )
            tools.add_node(
                node_map[sid],
                action,
                primitive=_PRIM_BY_SID[sid],
                actor=actor,
                system=system,
                reads=reads,
                writes=writes,
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
            tools.add_node(
                node_map[sid],
                action,
                primitive=_PRIM_BY_SID[sid],
                actor=actor,
                system=system,
                reads=reads,
                writes=writes,
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
    expected to pass hygiene but fail Ground Truth correctness.
    """
    poison = _claim_obs(tools, "I like pizza on Fridays.")
    dag = tools.db.dag
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
        dag.nodes["b"].reads[0] = InferredValue(
            value="wrongdata", confidence=1.0, observation_ids=[poison]
        )
    elif target == "write":
        dag.nodes["c"].writes[0] = InferredValue(
            value="wrongdata", confidence=1.0, observation_ids=[poison]
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
        dag.nodes["d"].necessity.rationale = InferredValue(
            value="a wrong reason", confidence=1.0, observation_ids=[poison]
        )
    return poison


# ---------------------------------------------------------------------------
# Model shape (DiscoveredConcept removed)
# ---------------------------------------------------------------------------


def test_A_discovered_concept_removed():
    assert not hasattr(dagmod, "DiscoveredConcept")
    assert "concepts" not in BusinessDAG.model_fields
    assert "concept_id" not in Node.model_fields
    assert not hasattr(InterviewTools, "discover_concept")


def test_truth_and_agent_result_use_same_dag_class():
    sc = get_scenario(SCENARIO)
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
    assert any("cycle" in e for e in dag.validate())


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
    tools.db.dag.nodes["b"].primitive = InferredValue(
        value="unclassified", confidence=1.0
    )
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
    tools.db.dag.nodes["c"].primitive = InferredValue(value="approve", confidence=1.0)
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
    tools.db.dag.nodes["b"].actor.observation_ids.append("obs_fabricated")
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
    tools.db.dag.nodes["b"].actor.observation_ids.append("obs_fake")
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
    tools.db.dag.end_node_ids = []
    res = _eval(tools)
    assert res.end_precision == 0.0
    assert res.structural_pass is False


# ---------------------------------------------------------------------------
# Non-quotation lab scenario (no DiscoveredConcept)
# ---------------------------------------------------------------------------


def _build_lab(tools: InterviewTools):
    tools.start_inference("lab")
    _ingest(tools, "assistant", "Hello.")
    nodes = [
        (
            "n1",
            "specimen accession",
            "receive",
            "lab tech",
            None,
            ["sample"],
            ["accessioned sample"],
            "The lab tech receives the specimen during specimen accession. It reads sample. It writes accessioned sample.",
        ),
        (
            "n2",
            "chamber seasoning",
            "create",
            "lab tech",
            "environment chamber",
            [],
            ["seasoned chamber"],
            "The lab tech prepares the chamber during chamber seasoning. It uses the environment chamber. It writes seasoned chamber.",
        ),
        (
            "n3",
            "conditioning cycle",
            "transform",
            "lab tech",
            "environment chamber",
            ["accessioned sample"],
            ["conditioned sample"],
            "The lab tech processes samples during the conditioning cycle. It uses the environment chamber. It reads accessioned sample. It writes conditioned sample.",
        ),
        (
            "n4",
            "approve conditioned batch",
            "approve",
            "lab supervisor",
            None,
            ["conditioned sample"],
            ["batch approval"],
            "The lab supervisor approves the conditioned batch. It reads conditioned sample. It writes batch approval.",
        ),
    ]
    for sid, action, prim, actor, system, reads, writes, text in nodes:
        oid = _claim_obs(tools, text)
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
    sc = get_scenario(LAB_SCENARIO)
    res = evaluate(tools.db, sc.truth, sc.spec)
    assert res.quality_pass is True
    assert res.node_recall == 1.0


def test_lab_unknown_primitive_unclassified_valid():
    tools = _tools()
    _build_lab(tools)
    tools.db.dag.nodes["n2"].primitive = InferredValue(
        value="unclassified", confidence=1.0
    )
    sc = get_scenario(LAB_SCENARIO)
    assert evaluate(tools.db, sc.truth, sc.spec).quality_pass is True


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
        ]
    ).lower()
    for term in HIDDEN_TERMS:
        assert term not in policy, f"policy leaks {term}"
        assert term not in docs, f"tool doc leaks {term}"


# ---------------------------------------------------------------------------
# End-to-end EnvironmentEvaluator reward
# ---------------------------------------------------------------------------


def _mk_tool_message(traj, tools, cid, name, args):
    tc = ToolCall(id=cid, name=name, arguments=args)
    traj.append(AssistantMessage(role="assistant", tool_calls=[tc]))
    tools.db.messages.append({"role": "assistant", "content": None})
    res = getattr(tools, name)(**args)
    traj.append(ToolMessage(role="tool", id=cid, content=res))
    tools.db.messages.append({"role": "tool", "content": res})
    return res


def _reference_trajectory(node_ids: tuple = ("a", "b", "c", "d", "e", "f")):
    # Per-node visible attributes for the quotation sales stakeholder (matches
    # tasks.json known_info). Hidden attributes must be left unset so the
    # reference trajectory obeys the stakeholder contract.
    sid_to_truth = {"a": "r", "b": "cc", "c": "cq", "d": "ap", "e": "sq", "f": "me"}
    sc = get_scenario(SCENARIO)
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
            args["reads"] = reads
        if "writes" in vis:
            args["writes"] = writes
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
    )
    assert reward_info.reward == 1.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks == {
        "assert_finish_interview": True,
        "assert_dag_reconstructed": True,
        "assert_necessity_handled": True,
        "assert_evidence_backed": True,
    }
    diag = reward_info.info["diagnostics"]
    assert diag["quality_pass"] is True


def test_evaluator_detects_missing_node():
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    task = [t for t in get_tasks() if t.id == SCENARIO][0]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=_reference_trajectory(node_ids=("a", "b", "c", "e", "f")),
        solo_mode=False,
    )
    assert reward_info.reward == 0.0
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
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
    )
    checks = {c.env_assertion.func_name: c.met for c in reward_info.env_assertions}
    assert checks["assert_necessity_handled"] is False
    assert reward_info.info["diagnostics"]["fabricated_necessity"] is True


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
    return task.user_scenario.instructions.task_instructions or ""


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
# Conservative node matching (missing nodes must NOT be matched away)
# ---------------------------------------------------------------------------

from tau2.domains.business_interview.evaluation import _match_nodes  # noqa: E402


def _drop_node(tools: InterviewTools, nid: str) -> None:
    """Remove a node and any incident edges from the built DAG."""
    dag = tools.db.dag
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
    tools.db.dag.nodes["d"].action = InferredValue(
        value="Get approval from a manager before sending the quotation to the customer",
        confidence=1.0,
    )
    sc = get_scenario(SCENARIO)
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
        tc = ToolCall(id=f"c{cid}", name=name, arguments=args)
        push(AssistantMessage(role="assistant", tool_calls=[tc]))
        res = getattr(env.tools, name)(**args)
        push(ToolMessage(role="tool", id=f"c{cid}", content=res))
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
    listing = replay.tools.list_stakeholder_messages()
    assert listing.splitlines()[0].startswith("sm_1:")
    assert "sm_2:" in listing
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
    assert "prepare_quotation" not in tools.db.dag.nodes
    # the incident edge e1 is removed; e2 (between remaining nodes) survives
    assert "e1" not in tools.db.dag.edges
    assert "e2" in tools.db.dag.edges
    # no dangling edge references the removed node
    for e in tools.db.dag.edges.values():
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
    assert oid in tools.db.dag.nodes["fine"].observation_ids
    assert "coarse" not in tools.db.dag.nodes


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
    assert "prepare_quotation" not in tools.db.dag.nodes
    assert all(
        nid in tools.db.dag.nodes
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
# ---------------------------------------------------------------------------


def _build_faithful_contract(tools: InterviewTools, ja: bool = False):
    """Build a quotation DAG that asserts ONLY stakeholder-knowable attributes
    (per the sales StakeholderFilter), with authentic Observation provenance.

    Hidden attributes (ap system/reads/writes, sq reads/writes, me reads) are
    left unset — the correct epistemic-restraint behavior.
    """
    actions = _JA_ACTIONS if ja else [t[1] for t in _TRUTH_NODES]
    node_data = list(zip(_TRUTH_NODES, actions))
    # per-node visible axes for the quotation sales stakeholder
    sc = get_scenario(JA_SCENARIO if ja else SCENARIO)
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
    node_map = {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e", "f": "f"}
    _ingest(tools, "assistant", "Hello.")
    for (sid, _, actor, system, reads, writes), action in node_data:
        vis = visible_by_sid[sid]
        oid = _claim_obs(tools, _node_obs_text(action, actor, system, reads, writes))
        tools.add_node(
            node_map[sid],
            action,
            primitive=_PRIM_BY_SID[sid],
            actor=actor if "actor" in vis else None,
            system=system if "system" in vis else None,
            reads=reads if "reads" in vis else None,
            writes=writes if "writes" in vis else None,
            observation_id=oid,
        )
    rationale = "与信リスク管理のため" if ja else "for credit risk management"
    doid = _claim_obs(tools, f"The approval is {rationale}.")
    tools.set_node_necessity("d", rationale=rationale, observation_id=doid)
    tools.set_node_necessity("f")
    action_by_sid = {sid: action for (sid, _, _, _, _, _), action in node_data}
    for eid, frm, to, pred in [
        ("e1", "a", "b", None),
        ("e2", "b", "c", None),
        ("e3", "c", "d", "100万円超" if ja else "amount over 1,000,000"),
        ("e4", "c", "e", "100万円以下" if ja else "amount at or below 1,000,000"),
        ("e5", "d", "e", None),
        ("e6", "c", "f", "月末" if ja else "month-end"),
    ]:
        fa, ta = action_by_sid[frm], action_by_sid[to]
        etext = f"After {fa}, we {ta}." + (f" when {pred}." if pred else "")
        eoid = _claim_obs(tools, etext)
        tools.add_edge(eid, frm, to, predicate=pred, observation_id=eoid)
    tools.set_dag_endpoints(start_node_id="a", end_node_ids=["e", "f"])
    tools.finish_interview()


def _eval_contract(tools: InterviewTools, scenario: str = SCENARIO):
    sc = get_scenario(scenario)
    return evaluate(tools.db, sc.truth, sc.spec, sc.stakeholder)


def test_faithful_contract_reconstruction_achieves_structural_pass():
    tools = _tools()
    _build_faithful_contract(tools)
    res = _eval_contract(tools)
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
    _build_faithful_contract(tools)
    res = _eval_contract(tools)
    # ap.system, ap.reads, ap.writes, sq.reads, sq.writes, me.reads are hidden;
    # leaving them unset is correct, so each axis scores 1.0.
    assert res.system_correctness == 1.0
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0


def test_asserting_hidden_ap_system_is_incorrect():
    tools = _tools()
    _build_faithful_contract(tools)
    # Asserting ap.system=quoting (which equals the hidden Truth) is STILL wrong:
    # the stakeholder cannot know it, so epistemic restraint is required.
    tools.db.dag.nodes["d"].system = InferredValue(
        value="quoting", confidence=1.0, observation_ids=["obs_x"]
    )
    res = _eval_contract(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_asserting_hidden_ap_reads_is_incorrect():
    tools = _tools()
    _build_faithful_contract(tools)
    tools.db.dag.nodes["d"].reads = [
        InferredValue(value="quote", confidence=1.0, observation_ids=["obs_x"])
    ]
    res = _eval_contract(tools)
    assert res.read_correctness < 1.0
    assert res.structural_pass is False


def test_wrong_visible_attribute_still_fails():
    tools = _tools()
    _build_faithful_contract(tools)
    # cq.system IS visible; a wrong value must still fail.
    tools.db.dag.nodes["c"].system = InferredValue(
        value="erp", confidence=1.0, observation_ids=["obs_x"]
    )
    res = _eval_contract(tools)
    assert res.system_correctness < 1.0
    assert res.structural_pass is False


def test_contract_preserves_topology_predicate_necessity_evidence():
    # Topology / predicates / necessity / evidence requirements are unchanged:
    # the faithful-contract build still passes all gates.
    tools = _tools()
    _build_faithful_contract(tools)
    res = _eval_contract(tools)
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
    _build_faithful_contract(en, ja=False)
    ja = _tools()
    _build_faithful_contract(ja, ja=True)
    ren = _eval_contract(en, SCENARIO)
    rja = _eval_contract(ja, JA_SCENARIO)
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
    # Lab known_info provides a defensible basis for every GT read/write
    # artifact, so all axes are visible and the existing faithful build passes
    # under the same visibility-aware evaluator.
    tools = _tools()
    _build_lab(tools)
    res = _eval_contract(tools, LAB_SCENARIO)
    assert res.quality_pass is True
    assert res.structural_pass is True
    assert res.actor_correctness == 1.0
    assert res.system_correctness == 1.0
    assert res.read_correctness == 1.0
    assert res.write_correctness == 1.0
