"""Deterministic tests for business_interview LLM-roundtrip reduction and
environment-owned Observations.

These cover the flow changes that cut Agent generations WITHOUT weakening
provenance/semantics:

- an ACCEPTED stakeholder response automatically becomes one Observation
  before the Agent sees it; the Observation id is delivered inline with the
  public text (``[Observation obs_N] ...``) — no observation tool, no
  round trip;
- multiple INDEPENDENT evidence tool calls in one Agent message batch in one
  env round trip (using the already-delivered Observation id), while a
  same-batch FUTURE dependency (an id that does not exist when the batch
  starts) is rejected;
- the stakeholder's Semantic Response Plan (WHAT it answers) is validated
  deterministically against its knowledge before realization: the plan can
  never contradict StakeholderKnowledge, every planned assertion must appear
  in the realized sidecar, and private ids never leak.

No LLM / provider is ever called (scripted stub agent + the real
business_interview environment).
"""

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import TerminationReason
from tau2.domains.business_interview.environment import get_environment
from tau2.domains.business_interview.graph import InterviewDB
from tau2.domains.business_interview.tools import InterviewTools
from tau2.orchestrator.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Stub agent that emits scripted multi-tool-call messages (no LLM)
# ---------------------------------------------------------------------------


class _StubState:
    pass


class _MultiToolAgent:
    """Emits scripted assistant messages, each with a list of tool calls."""

    def __init__(self, messages: list[AssistantMessage]):
        self._messages = list(messages)
        self.tools = []
        self.domain_policy = ""

    def get_init_state(self, message_history=None):
        return _StubState()

    def generate_next_message(self, message, state):
        if not self._messages:
            raise AssertionError("stub agent message queue exhausted")
        return self._messages.pop(0), state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content) and "###STOP###" in message.content

    def stop(self, message=None, state=None):
        pass

    def set_seed(self, seed: int):
        pass


class _StubUser:
    """Minimal user that answers the orchestrator's seeded first message and
    then stays silent (the Agent only makes tool calls in these tests)."""

    def __init__(self):
        self.tools = None
        self._replied = False

    def get_init_state(self, message_history=None):
        return _StubState()

    def generate_next_message(self, message, state):
        from tau2.data_model.message import UserMessage

        if not self._replied:
            self._replied = True
            return (
                UserMessage(
                    role="user",
                    content="Sure, that's fine. What would you like to know?",
                ),
                state,
            )
        raise AssertionError("stub user should not need to reply again in these tests")

    @classmethod
    def is_stop(cls, message) -> bool:
        return False

    def stop(self, message=None, state=None):
        pass

    def set_seed(self, seed: int):
        pass


def _tool_call(name: str, args: dict, idx: int) -> ToolCall:
    return ToolCall(id=f"c{idx}", name=name, arguments=args, requestor="assistant")


def _assistant(tool_calls: list[ToolCall], content: str = "") -> AssistantMessage:
    # An empty tool_calls list must be stored as None so the orchestrator
    # recognizes the message as text (is_tool_call() checks ``tool_calls is
    # not None``).
    return AssistantMessage(
        role="assistant", content=content, tool_calls=tool_calls or None
    )


def _get_tools(env) -> InterviewTools:
    """Narrow the environment's toolkit to InterviewTools (env.tools is typed
    broadly as ToolkitBase)."""
    assert isinstance(env.tools, InterviewTools)
    return env.tools


def _task():
    """The quotation_workflow_1 task (has initial_state the orchestrator
    needs)."""
    from tau2.domains.business_interview.environment import get_tasks

    return next(t for t in get_tasks() if t.id == "quotation_workflow_1")


# ---------------------------------------------------------------------------
# Independent tool calls batch in ONE Agent generation
# ---------------------------------------------------------------------------


def test_independent_tool_calls_execute_as_one_batch():
    """Three INDEPENDENT create_concept calls in one Agent message are all
    executed in a single env round trip (no intermediate Agent generation),
    and all results come back together."""
    env = get_environment()
    tools = _get_tools(env)
    agent = _MultiToolAgent(
        [
            _assistant(
                [
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_actor", "kind": "activity", "label": "clerk"},
                        0,
                    ),
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_sys", "kind": "system", "label": "CRM"},
                        1,
                    ),
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_data", "kind": "data", "label": "order"},
                        2,
                    ),
                ]
            ),
            _assistant([], content="###STOP###"),
        ]
    )
    orch = Orchestrator(
        domain="business_interview",
        agent=agent,  # type: ignore[arg-type]
        user=_StubUser(),  # type: ignore[arg-type]
        environment=env,
        task=_task(),  # type: ignore[arg-type]
        max_steps=20,
        validate_communication=True,
    )
    # run until done, counting orchestrator steps
    orch._run_start_perf = __import__("time").perf_counter()
    orch.initialize()
    guard = 0
    while not orch.done and guard < 30:
        orch.step()
        orch._check_termination()
        guard += 1
    assert orch.termination_reason == TerminationReason.AGENT_STOP
    # the batch was a single AGENT->ENV step: all three concepts created
    db: InterviewDB = tools.db
    assert db.graph is not None
    assert set(db.graph.concepts) >= {"c_actor", "c_sys", "c_data"}
    # the three tool results came back together (one MultiToolMessage -> one
    # subsequent Agent generation, not three)
    messages = orch.get_trajectory()
    assistant_turns = [
        m for m in messages if isinstance(m, AssistantMessage) and m.is_tool_call()
    ]
    assert len(assistant_turns) == 1


def test_dependent_tool_calls_cannot_skip_required_result():
    """A node cannot reference a concept that does not exist yet: the
    dependency must be honored — even when both calls are submitted, the
    dependent one fails (the batch does not silently fabricate the concept)."""
    tools = _get_tools(get_environment())
    # First create the concept, then reference it in add_node.
    result = tools.create_concept("c_actor", "activity", "clerk")
    assert "c_actor" in result
    result = tools.add_node("n1", activity={"concept_id": "c_actor"})
    assert "n1" in result
    # A node referencing an unknown concept is rejected — no auto-creation.
    try:
        tools.add_node("n2", activity={"concept_id": "ghost_concept"})
        created_behind_agents_back = True
    except ValueError:
        created_behind_agents_back = False
    assert created_behind_agents_back is False
    assert tools.db.graph is not None
    assert "n2" not in tools.db.graph.nodes


def test_add_node_requires_existing_concept_via_orchestrator_batch():
    """Even when an Agent submits create_concept + add_node in ONE message,
    the environment executes both in order and the node references the
    concept the Agent created — but a node-only message without the concept
    still fails. (Order inside the batch is preserved, dependency honored.)"""
    env = get_environment()
    tools = _get_tools(env)
    agent = _MultiToolAgent(
        [
            _assistant(
                [
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_actor", "kind": "activity", "label": "clerk"},
                        0,
                    ),
                    _tool_call(
                        "add_node",
                        {"node_id": "n1", "activity": {"concept_id": "c_actor"}},
                        1,
                    ),
                ]
            ),
            _assistant([], content="###STOP###"),
        ]
    )
    orch = Orchestrator(
        domain="business_interview",
        agent=agent,  # type: ignore[arg-type]
        user=_StubUser(),  # type: ignore[arg-type]
        environment=env,
        task=_task(),  # type: ignore[arg-type]
        max_steps=20,
        validate_communication=True,
    )
    orch._run_start_perf = __import__("time").perf_counter()
    orch.initialize()
    guard = 0
    while not orch.done and guard < 30:
        orch.step()
        orch._check_termination()
        guard += 1
    db: InterviewDB = tools.db
    assert db.graph is not None
    assert "n1" in db.graph.nodes
    assert db.graph.nodes["n1"].activity is not None
    assert db.graph.nodes["n1"].activity.concept_id == "c_actor"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Environment-owned Observation creation
# ---------------------------------------------------------------------------


def test_accepted_stakeholder_message_auto_creates_observation():
    """The environment automatically creates exactly one Observation for an
    accepted stakeholder utterance — the Agent never calls an observation
    tool. The Observation id is delivered inline at the front of the public
    text, and the raw text lives in the immutable Observation + ledger."""
    env = get_environment()
    tools = _get_tools(env)
    from tau2.data_model.message import UserMessage

    msg = UserMessage(role="user", content="We do it to manage credit risk.")
    env.on_message(msg)
    assert len(tools.db.observations) == 1
    obs = tools.db.observations[0]
    assert obs.id.startswith("obs_")
    assert obs.text == "We do it to manage credit risk."
    assert obs.source_id == "stakeholder"
    assert obs.order == 0
    # the raw text (not the marker) is what the ledger / Observation store
    assert tools.db.messages[-1]["content"] == "We do it to manage credit risk."
    # the Agent receives the Observation id inline with the public text
    assert msg.content == f"[Observation {obs.id}] We do it to manage credit risk."


def test_accepted_messages_and_observations_remain_1_to_1():
    """one accepted stakeholder utterance <-> one Observation <-> one
    Agent-visible Observation id, in order, never duplicated."""
    env = get_environment()
    tools = _get_tools(env)
    from tau2.data_model.message import UserMessage

    delivered: list[str] = []
    for text in ("Statement one.", "Statement two.", "Statement three."):
        msg = UserMessage(role="user", content=text)
        env.on_message(msg)
        assert msg.content is not None
        delivered.append(msg.content)
    assert len(tools.db.observations) == 3
    ids = [o.id for o in tools.db.observations]
    assert len(set(ids)) == 3
    assert [o.order for o in tools.db.observations] == [0, 1, 2]
    for delivered_content, obs in zip(delivered, tools.db.observations):
        assert delivered_content == f"[Observation {obs.id}] {obs.text}"


def test_failed_sidecar_retry_creates_no_observation_and_consumes_no_id():
    """A stakeholder generation whose private sidecar fails validation must
    not create an Observation or consume an Observation id; only an ACCEPTED
    (validated) utterance does. Repeated failed generations still leave zero
    Observations."""
    env = get_environment()
    tools = _get_tools(env)
    from tau2.data_model.message import UserMessage
    from tau2.domains.business_interview.environment import BusinessInterviewEnvironment
    from tau2.domains.business_interview.user_simulator import StakeholderUserSimulator

    assert isinstance(env, BusinessInterviewEnvironment)
    StakeholderUserSimulator(
        llm="dummy", task=_task(), environment=env, instructions="x"
    )
    assert env.assertion_ledger is not None
    assert env.assertion_ledger.catalog is not None
    for _ in range(2):
        bad = UserMessage(
            role="user",
            content="I check the customer in the CRM.",
            stakeholder_annotations=[
                {
                    "semantic_id": "node:skn_002:system",
                    "quote": "not in the text",
                    "occurrence": 0,
                    "mode": "value",
                }
            ],
        )
        try:
            env.on_message(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sidecar must be rejected")
    assert len(tools.db.observations) == 0
    assert env.assertion_ledger.annotations() == {}


# ---------------------------------------------------------------------------
# Removed observation tools + batching on the delivered Observation id
# ---------------------------------------------------------------------------


def test_obsolete_observation_tool_is_removed():
    """observe_latest_stakeholder_message and observe_message are gone (no
    compatibility shim); the Agent can never create/mutate Observations."""
    env = get_environment()
    tools = _get_tools(env)
    assert not hasattr(tools, "observe_latest_stakeholder_message")
    assert not hasattr(tools, "observe_message")
    names = {t.name for t in env.get_tools()}
    assert "observe_latest_stakeholder_message" not in names
    assert "observe_message" not in names


def test_evidence_ref_immediately_uses_delivered_observation_id():
    """EvidenceRef can use the Observation id delivered with the response
    in the very next tool call — no observation round trip."""
    env = get_environment()
    tools = _get_tools(env)
    from tau2.data_model.message import UserMessage

    msg = UserMessage(role="user", content="The process starts with a request.")
    env.on_message(msg)
    obs = tools.db.observations[0]
    assert msg.content is not None and msg.content.startswith("[Observation ")
    result = tools.create_concept(
        "c_req",
        "activity",
        "receive a request",
        evidence=[
            {
                "observation_id": obs.id,
                "quote": "starts with a request",
                "occurrence": 0,
            }
        ],
    )
    assert "c_req" in result


def test_independent_evidence_tools_batch_with_same_observation_id():
    """Once the Observation id is delivered with the response, independent
    evidence operations batch freely — one Agent turn, many tool calls."""
    env = get_environment()
    tools = _get_tools(env)
    agent = _MultiToolAgent(
        [
            _assistant(
                [
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_a", "kind": "activity", "label": "alpha"},
                        0,
                    ),
                    _tool_call(
                        "create_concept",
                        {"concept_id": "c_b", "kind": "activity", "label": "beta"},
                        1,
                    ),
                ]
            ),
            _assistant([], content="###STOP###"),
        ]
    )
    orch = Orchestrator(
        domain="business_interview",
        agent=agent,  # type: ignore[arg-type]
        user=_StubUser(),  # type: ignore[arg-type]
        environment=env,
        task=_task(),  # type: ignore[arg-type]
        max_steps=20,
        validate_communication=True,
    )
    orch._run_start_perf = __import__("time").perf_counter()
    orch.initialize()
    guard = 0
    while not orch.done and guard < 30:
        orch.step()
        orch._check_termination()
        guard += 1
    db: InterviewDB = tools.db
    assert db.graph is not None
    assert set(db.graph.concepts) >= {"c_a", "c_b"}
    msgs = orch.get_trajectory()
    tool_turns = [
        m for m in msgs if isinstance(m, AssistantMessage) and m.is_tool_call()
    ]
    assert len(tool_turns) == 1


def test_same_batch_future_dependency_is_rejected():
    """A tool in a batch cannot reference an Observation id (or other result)
    that is not yet known when the batch starts — no speculative/future id
    is ever resolved or fabricated."""
    tools = _get_tools(get_environment())
    try:
        tools.create_concept(
            "c_g",
            "activity",
            "ghost",
            evidence=[{"observation_id": "obs_999", "quote": "nope", "occurrence": 0}],
        )
        rejected = False
    except ValueError:
        rejected = True
    assert rejected is True
    assert tools.db.graph is None or "c_g" not in tools.db.graph.concepts
    assert len(tools.db.observations) == 0


# ---------------------------------------------------------------------------
# Semantic Response Plan (WHAT) is validated before realization (HOW)
# ---------------------------------------------------------------------------


def _catalog():
    """The quotation scenario StakeholderKnowledgeCatalog (with knowledge)."""
    from tau2.domains.business_interview.facts import StakeholderKnowledgeCatalog
    from tau2.domains.business_interview.scenario import get_scenario

    sc = get_scenario(_task().id)
    assert sc is not None
    return StakeholderKnowledgeCatalog.from_scenario(sc)


def test_semantic_response_plan_cannot_contradict_knowledge():
    """A plan whose (semantic_id, mode) contradicts the StakeholderKnowledge
    is rejected BEFORE realization."""
    from tau2.domains.business_interview.facts import (
        PlannedResponseItem,
        mode_for_resolved,
    )

    cat = _catalog()
    kg = cat.knowledge.graph
    # find a node with a known-value property slot (e.g. a system slot) and a
    # DONT_KNOW slot
    valueslot = None
    dontknowslot = None
    for nid in sorted(kg.nodes):
        for prop in ("system", "activity"):
            sid = f"node:{nid}:{prop}"
            mode = mode_for_resolved(kg.resolve(sid))
            if mode == "value" and valueslot is None:
                valueslot = sid
            if mode == "dont_know" and dontknowslot is None:
                dontknowslot = sid
    assert valueslot is not None, (
        "expected a known-value slot in the quotation knowledge"
    )
    # a known value cannot be planned as dont_know
    try:
        cat.validate_plan(
            [PlannedResponseItem(semantic_id=valueslot, mode="dont_know")]
        )
        rejected = False
    except ValueError:
        rejected = True
    assert rejected is True
    # a known value planned as absent is also contradicted
    try:
        cat.validate_plan([PlannedResponseItem(semantic_id=valueslot, mode="absent")])
        rejected2 = False
    except ValueError:
        rejected2 = True
    assert rejected2 is True
    # the same id planned with its true mode is accepted
    cat.validate_plan([PlannedResponseItem(semantic_id=valueslot, mode="value")])


def test_dont_know_cannot_become_value_in_plan():
    """A DONT_KNOW slot cannot be planned as a known value."""
    from tau2.domains.business_interview.facts import (
        PlannedResponseItem,
        mode_for_resolved,
    )

    cat = _catalog()
    kg = cat.knowledge.graph
    dontknowslot = None
    for nid in sorted(kg.nodes):
        for prop in ("system", "activity", "rationale"):
            sid = f"node:{nid}:{prop}"
            mode = mode_for_resolved(kg.resolve(sid))
            if mode == "dont_know":
                dontknowslot = sid
                break
        if dontknowslot is not None:
            break
    assert dontknowslot is not None, "expected a DONT_KNOW slot in the scenario"
    try:
        cat.validate_plan([PlannedResponseItem(semantic_id=dontknowslot, mode="value")])
        rejected = False
    except ValueError:
        rejected = True
    assert rejected is True
    # the true dont_know mode is accepted
    cat.validate_plan([PlannedResponseItem(semantic_id=dontknowslot, mode="dont_know")])


def test_plan_cannot_reference_unknown_or_truth_only_id():
    """A plan item whose semantic id is not in the stakeholder's knowledge is
    rejected (the plan can never carry Truth-only information)."""
    from tau2.domains.business_interview.facts import PlannedResponseItem

    cat = _catalog()
    try:
        cat.validate_plan(
            [PlannedResponseItem(semantic_id="node:skn_999:activity", mode="value")]
        )
        rejected = False
    except ValueError:
        rejected = True
    assert rejected is True


def test_every_planned_assertion_must_appear_in_sidecar():
    """check_sidecar_covers_plan requires every planned assertion to be
    present in the realized sidecar with the same id+mode and an exact span."""
    from tau2.domains.business_interview.facts import (
        PlannedResponseItem,
        SemanticAnnotation,
    )

    cat = _catalog()
    kg = cat.knowledge.graph
    valueslot = next(
        sid
        for nid in sorted(kg.nodes)
        for prop in ("system", "activity")
        if (sid := f"node:{nid}:{prop}")
        and __import__(
            "tau2.domains.business_interview.facts", fromlist=["mode_for_resolved"]
        ).mode_for_resolved(kg.resolve(sid))
        == "value"
    )
    plan = [PlannedResponseItem(semantic_id=valueslot, mode="value")]
    text = "We do it to manage credit risk for the order."
    # missing: nothing in the sidecar covers the plan
    try:
        cat.check_sidecar_covers_plan([], text, plan)
        missing_rejected = False
    except ValueError:
        missing_rejected = True
    assert missing_rejected is True
    # correct coverage: same id+mode with an exact quote
    good = [
        SemanticAnnotation(
            semantic_id=valueslot, quote="order", occurrence=0, mode="value"
        )
    ]
    assert "order" in text
    cat.check_sidecar_covers_plan(good, text, plan)
    # an annotation with the RIGHT id but WRONG mode is not coverage
    bad = [
        SemanticAnnotation(
            semantic_id=valueslot, quote="order", occurrence=0, mode="absent"
        )
    ]
    try:
        cat.check_sidecar_covers_plan(bad, text, plan)
        wrongmode_rejected = False
    except ValueError:
        wrongmode_rejected = True
    assert wrongmode_rejected is True
    # an annotation for something NOT in the plan is rejected as unplanned
    otherslot = None
    for nid in sorted(kg.nodes):
        sid = f"node:{nid}:activity"
        if sid != valueslot and kg.resolve(sid) is not None:
            otherslot = sid
            break
    assert otherslot is not None
    extra = good + [
        SemanticAnnotation(
            semantic_id=otherslot, quote="credit risk", occurrence=0, mode="value"
        )
    ]
    try:
        cat.check_sidecar_covers_plan(extra, text, plan)
        extra_rejected = False
    except ValueError:
        extra_rejected = True
    assert extra_rejected is True


def test_stakeholder_generation_plans_then_realizes_with_stubbed_llm():
    """The stakeholder pipeline is two-phase and deterministic: it first builds
    a validated Semantic Response Plan (WHAT), then realizes the validated plan
    into natural language + sidecar, enforcing that every planned assertion
    appears with the same (semantic_id, mode) and an exact public-text span.
    Provable without any live LLM by stubbing the completion."""
    import json as _json

    from tau2.data_model.message import SystemMessage, UserMessage
    from tau2.domains.business_interview.facts import (
        PlannedResponseItem,
        StakeholderKnowledgeCatalog,
        mode_for_resolved,
    )
    from tau2.domains.business_interview.scenario import get_scenario
    from tau2.domains.business_interview.user_simulator import (
        StakeholderUserSimulator,
    )

    env = get_environment()
    sim = StakeholderUserSimulator(
        llm="dummy", task=_task(), environment=env, instructions="x"
    )
    sc = get_scenario(_task().id)
    assert sc is not None
    sim._scenario = sc  # noqa: SLF001
    sim._catalog = StakeholderKnowledgeCatalog.from_scenario(sc)  # noqa: SLF001

    kg = sc.knowledge.graph
    valueslot = next(
        sid
        for nid in sorted(kg.nodes)
        for prop in ("system", "activity")
        if (sid := f"node:{nid}:{prop}")
        and mode_for_resolved(kg.resolve(sid)) == "value"
    )
    plan_payload = _json.dumps({"plan": [{"semantic_id": valueslot, "mode": "value"}]})
    good_sidecar = _json.dumps(
        {
            "message": "We check the order to manage credit risk.",
            "annotations": [
                {
                    "semantic_id": valueslot,
                    "mode": "value",
                    "quote": "the order",
                    "occurrence": 0,
                }
            ],
            "alignments": [],
            "terminology": [],
        }
    )
    bad_sidecar = _json.dumps(
        {
            "message": "We check the order to manage credit risk.",
            "annotations": [],
            "alignments": [],
            "terminology": [],
        }
    )

    msgs = [
        SystemMessage(role="system", content="stakeholder system prompt"),
        UserMessage(role="user", content="What is this step for?"),
    ]

    class _Reply:
        def __init__(self, content):
            self.content = content

    # Phase 1 call -> plan; Phase 2 call -> sidecar (the phase-1 contract is
    # the leading text of the output contract; the realize contract is the
    # extended sidecar contract).
    from tau2.domains.business_interview.user_simulator import (
        _PLAN_CONTRACT as PLAN_CONTRACT,
    )

    def _stub(messages, output_contract_text=None, **kwargs):  # noqa: ANN001
        contract = output_contract_text or ""
        if contract.startswith(PLAN_CONTRACT[:80]):
            return _Reply(plan_payload)
        return _Reply(good_sidecar)

    setattr(sim, "_call_llm", _stub)
    plan = sim._generate_plan(msgs)
    assert plan == [PlannedResponseItem(semantic_id=valueslot, mode="value")]
    sidecar = sim._realize_sidecar(msgs, plan)
    assert sidecar["message"] == "We check the order to manage credit risk."
    assert len(sidecar["annotations"]) == 1

    # deterministic completeness enforcement: an omission is rejected
    setattr(
        sim,
        "_call_llm",
        lambda messages,
        output_contract_text=None,
        call_name=None,
        retry_attempt=False: _Reply(bad_sidecar),
    )
    try:
        sim._realize_sidecar(msgs, plan)
        rejected = False
    except ValueError:
        rejected = True
    assert rejected is True


def test_internal_stakeholder_refusal_retry_is_recorded_at_generate_layer(
    monkeypatch,
):
    """A failed private plan generation remains visible even when the retry
    yields a valid plan and the accepted public answer is ordinary uncertainty."""
    import json as _json

    from tau2.data_model.message import AssistantMessage
    from tau2.domains.business_interview.run_metrics import account_model_refusals
    from tau2.utils.llm_call_metrics import (
        LLMCallMetricsCollector,
        model_refusal_records,
        set_llm_call_metrics_collector,
    )

    class _Message:
        role = "assistant"
        tool_calls = []
        refusal = None

        def __init__(self, content):
            self.content = content

    class _Choice:
        finish_reason = "stop"

        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        model = "fake-model"

        def __init__(self, content):
            self.choices = [_Choice(content)]

        def get(self, key):
            return None

        def to_dict(self):
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": self.choices[0].message.content,
                        },
                    }
                ]
            }

    responses = iter(
        [
            _Response("I can't assist with that request."),
            _Response(_json.dumps({"plan": []})),
            _Response(
                _json.dumps(
                    {
                        "message": "I don't know.",
                        "annotations": [],
                        "alignments": [],
                        "terminology": [],
                    }
                )
            ),
        ]
    )
    monkeypatch.setattr(
        "tau2.utils.llm_utils.completion", lambda **kwargs: next(responses)
    )

    env = get_environment()
    from tau2.domains.business_interview.user_simulator import (
        StakeholderUserSimulator,
    )

    sim = StakeholderUserSimulator(
        llm="openrouter/example-model",
        task=_task(),
        environment=env,
        instructions="x",
    )
    collector = LLMCallMetricsCollector()
    set_llm_call_metrics_collector(collector)
    try:
        public = sim._generate_next_message(
            AssistantMessage(role="assistant", content="Please explain the process."),
            sim.get_init_state(),
        )
        env.on_message(public)
        records = collector.records()
    finally:
        set_llm_call_metrics_collector(None)

    assert [record.call_name for record in records] == [
        "stakeholder_semantic_plan",
        "stakeholder_semantic_plan",
        "stakeholder_realization",
    ]
    assert records[0].explicit_refusal is True
    assert records[1].explicit_refusal is False
    assert records[1].retry_attempt is True
    assert records[1].attempt_index == 1
    assert account_model_refusals([public]) == []
    assert len(model_refusal_records(records)) == 1
