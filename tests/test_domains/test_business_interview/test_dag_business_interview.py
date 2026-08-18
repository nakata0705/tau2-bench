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
    turn = _ingest(tools, "user", text)
    return tools.observe_turn(turn)


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
    oid = tools.observe_turn(turn)
    assert oid == f"obs_{turn}"
    assert tools.observe_turn(turn) == oid  # idempotent
    with pytest.raises(ValueError):
        tools.observe_turn(99)
    _ingest(tools, "assistant", "I am the agent.")
    with pytest.raises(ValueError):
        tools.observe_turn(len(tools.db.messages) - 1)


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
            InterviewTools.observe_turn.__doc__ or "",
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
    for sid, action, (_, _, actor, system, reads, writes) in node_specs:
        cid += 1
        statement = _node_obs_text(action, actor, system, reads, writes)
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        turn = len(tools.db.messages) - 1
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_turn", {"turn_idx": turn}
        )
        cid += 1
        args = {
            "node_id": sid,
            "action": action,
            "primitive": _PRIM_BY_SID[sid],
            "actor": actor,
            "system": system,
            "reads": reads,
            "writes": writes,
            "observation_id": oid,
        }
        _mk_tool_message(traj, tools, f"c{cid}", "add_node", args)
    if "d" in include:
        cid += 1
        statement = "The approval is for credit risk management."
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        turn = len(tools.db.messages) - 1
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_turn", {"turn_idx": turn}
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
        fa, ta = action_by_sid[frm], action_by_sid[to]
        statement = f"After {fa}, we {ta}." + (f" when {pred}." if pred else "")
        um = UserMessage(role="user", content=statement)
        traj.append(um)
        tools.db.messages.append({"role": "user", "content": statement})
        turn = len(tools.db.messages) - 1
        oid = _mk_tool_message(
            traj, tools, f"c{cid}", "observe_turn", {"turn_idx": turn}
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
