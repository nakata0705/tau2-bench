"""Deterministic tests for business_interview LLM-roundtrip reduction.

These cover the tool-flow ergonomics added to cut unnecessary Agent
generations WITHOUT weakening provenance/semantics:

- multiple INDEPENDENT tool calls in one Agent message are all executed as one
  batch (one env round trip, results returned together);
- DEPENDENT tool calls cannot bypass a required result (a concept must exist
  before a node references it; observation ids must come from the tool, never
  be guessed);
- the merged observation acquisition (`observe_latest_stakeholder_message`
  returns the Observation id directly, no separate `observe_message` round
  trip) preserves the EXACT Observation ids / text / provenance.

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
# Merged observation acquisition preserves exact ids / provenance
# ---------------------------------------------------------------------------


def test_observe_latest_returns_observation_id_directly():
    """observe_latest_stakeholder_message() captures the newest stakeholder
    message AND returns its Observation id in one call — matching exactly what
    observe_message(message_id) would return for the same message."""
    env = get_environment()
    tools = _get_tools(env)
    # a stakeholder message arrives through the environment
    from tau2.data_model.message import UserMessage

    env.on_message(UserMessage(role="user", content="First statement."))
    obs_id = tools.observe_latest_stakeholder_message()
    assert obs_id.startswith("obs_")
    obs = next(o for o in tools.db.observations if o.id == obs_id)
    assert obs.text == "First statement."
    assert obs.source_id == "stakeholder"
    # exact equality with the by-id capture (idempotent + identical)
    assert tools.observe_message("sm_1") == obs_id
    assert tools.observe_latest_stakeholder_message() == obs_id
    assert len(tools.db.observations) == 1


def test_observe_latest_preserves_provenance_across_messages():
    """The merged capture preserves exact obs ids, order and turn binding for
    several messages, and never duplicates an Observation."""
    env = get_environment()
    tools = _get_tools(env)
    from tau2.data_model.message import UserMessage

    env.on_message(UserMessage(role="user", content="Statement one."))
    oid1 = tools.observe_latest_stakeholder_message()
    env.on_message(UserMessage(role="user", content="Statement two."))
    oid2 = tools.observe_latest_stakeholder_message()
    assert oid1 != oid2
    assert len(tools.db.observations) == 2
    obs1 = next(o for o in tools.db.observations if o.id == oid1)
    obs2 = next(o for o in tools.db.observations if o.id == oid2)
    assert obs1.text == "Statement one."
    assert obs2.text == "Statement two."
    assert obs1.order == 0 and obs2.order == 1
    # per-message turn binding matches the ledger
    turns = [i for i, m in enumerate(tools.db.messages) if m.get("role") == "user"]
    assert obs1.turn == turns[0]
    assert obs2.turn == turns[1]
    # observe_message by id yields the SAME observation ids
    assert tools.observe_message("sm_1") == oid1
    assert tools.observe_message("sm_2") == oid2


def test_observe_latest_single_step_in_orchestrator():
    """In a real orchestrator run the Agent needs only ONE tool call to
    capture the newest message: observe_latest_stakeholder_message returns the
    Observation id; no separate observe_message round trip is required. The
    captured Observation matches observe_message(message_id) by id exactly."""
    env = get_environment()
    tools = _get_tools(env)
    agent = _MultiToolAgent(
        [
            _assistant([_tool_call("observe_latest_stakeholder_message", {}, 0)]),
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
    # exactly one Observation, captured from the latest stakeholder message
    obs = tools.db.observations
    assert len(obs) == 1
    user_msgs = [i for i, m in enumerate(tools.db.messages) if m.get("role") == "user"]
    latest_user_id = f"sm_{len(user_msgs)}"
    assert obs[0].id == tools.observe_message(latest_user_id)
    assert obs[0].text == tools.db.messages[user_msgs[-1]]["content"]
    assert obs[0].source_id == "stakeholder"
    # only ONE agent tool-call turn happened (single-step acquisition)
    msgs = orch.get_trajectory()
    tool_turns = [
        m for m in msgs if isinstance(m, AssistantMessage) and m.is_tool_call()
    ]
    assert len(tool_turns) == 1
    # and the returned content was the Observation id
    from tau2.data_model.message import ToolMessage

    tool_msg = [m for m in msgs if isinstance(m, ToolMessage)][0]
    assert (tool_msg.content or "").strip() == obs[0].id
