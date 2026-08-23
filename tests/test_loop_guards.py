"""Deterministic tests for conversation-loop guards.

The guards are a benchmark RUNTIME safeguard (not evaluator semantics): a run
that repeats the same normalized question / response / (question, semantic
answer) interaction terminates early with its own diagnostic termination
reason instead of consuming max_steps. Normalization is cosmetic only — no
semantic similarity, no LLMs, no embeddings.

These tests drive a real Orchestrator with scripted stub participants so no
LLM / provider is ever called.
"""

from typing import Optional

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.utils.normalization import normalize_text

# ---------------------------------------------------------------------------
# Stub participants (scripted, deterministic, no LLM)
# ---------------------------------------------------------------------------


class _StubState:
    pass


class _StubAgent:
    """Emits a scripted list of assistant contents, then distinct filler so
    exhausted messages never collide with the loop counters."""

    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self._fill = 0
        self.tools = []
        self.domain_policy = ""

    def get_init_state(self, message_history=None):
        return _StubState()

    def generate_next_message(self, message, state):
        if self._contents:
            content = self._contents.pop(0)
        else:
            self._fill += 1
            content = f"filler_agent_{self._fill}"
        return AssistantMessage(role="assistant", content=content, cost=0.0), state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return bool(message.content) and "###STOP###" in message.content

    def stop(self, message=None, state=None):
        pass

    def set_seed(self, seed: int):
        pass


class _StubUser:
    """Emits a scripted list of user contents (never tool calls), then
    distinct filler so exhausted messages never collide with the loop
    counters."""

    def __init__(self, contents: list[str], signatures: Optional[list] = None):
        self._contents = list(contents)
        self._signatures = list(signatures or [None] * len(contents))
        self._fill = 0
        self.tools = None

    def get_init_state(self, message_history=None):
        return _StubState()

    def generate_next_message(self, message, state):
        if self._contents:
            content = self._contents.pop(0)
        else:
            self._fill += 1
            content = f"filler_user_{self._fill}"
        return UserMessage(role="user", content=content), state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        return False

    def stop(self, message=None, state=None):
        pass

    def set_seed(self, seed: int):
        pass

    # Optional private semantic fingerprint (mirrors the business_interview
    # StakeholderUserSimulator hook). Scripted so tests are deterministic.
    def interaction_signature(self, message: UserMessage) -> Optional[str]:
        if self._signatures:
            return self._signatures.pop(0)
        return None


class _SigUser(_StubUser):
    """A stub user whose semantic signature is scripted independently of its
    surface text (same words, different semantics / different words, same
    semantics)."""

    def __init__(self, contents: list[str], signatures: list[Optional[str]]):
        super().__init__(contents, signatures)


class _ToolScriptAgent(_StubAgent):
    """Emit deterministic tool calls or public text from a scripted sequence."""

    def __init__(self, script):
        super().__init__([])
        self._script = list(script)
        self._tool_index = 0

    def generate_next_message(self, message, state):
        if self._script:
            item = self._script.pop(0)
        else:
            self._fill += 1
            item = f"filler_agent_{self._fill}"
        if isinstance(item, tuple):
            name, arguments = item
            self._tool_index += 1
            return (
                AssistantMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"tool_{self._tool_index}",
                            name=name,
                            arguments=arguments,
                            requestor="assistant",
                        )
                    ],
                    cost=0.0,
                ),
                state,
            )
        return AssistantMessage(role="assistant", content=item, cost=0.0), state


class _SuccessfulToolEnvironment:
    """Minimal environment that accepts every scripted tool call."""

    def get_response(self, tool_call):
        return ToolMessage(
            id=tool_call.id or "tool_result",
            role="tool",
            content="ok",
            requestor="assistant",
        )

    def on_message(self, message):
        pass

    def sync_tools(self):
        pass

    def get_policy(self):
        return ""

    def get_domain_name(self):
        return "mock"

    def get_info(self, include_tool_info=False):
        return {}

    def set_state(self, **kwargs):
        pass

    def episode_complete(self):
        return False


def _tool_orchestrator(task, script, user_contents=None, max_stalled=6):
    return Orchestrator(
        domain="mock",
        agent=_ToolScriptAgent(script),  # type: ignore[arg-type]
        user=_StubUser(user_contents or ["observation"]),  # type: ignore[arg-type]
        environment=_SuccessfulToolEnvironment(),  # type: ignore[arg-type]
        task=task,
        max_steps=40,
        max_repeated_questions=0,
        max_repeated_responses=0,
        max_repeated_interactions=0,
        max_stalled_tool_operations=max_stalled,
    )


def _run_until_done(orchestrator: Orchestrator, max_loop: int = 60) -> Orchestrator:
    import time

    orchestrator._run_start_perf = time.perf_counter()
    orchestrator.initialize()
    guard = 0
    while not orchestrator.done and guard < max_loop:
        orchestrator.step()
        orchestrator._check_termination()
        guard += 1
    return orchestrator


def _run_steps(orchestrator: Orchestrator, n: int) -> Orchestrator:
    """Run exactly ``n`` steps (init + n step() + termination checks)."""
    import time

    orchestrator._run_start_perf = time.perf_counter()
    orchestrator.initialize()
    for _ in range(n):
        if orchestrator.done:
            break
        orchestrator.step()
        orchestrator._check_termination()
    return orchestrator


@pytest.fixture
def env_and_task(get_environment, base_task):
    return get_environment(), base_task


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_whitespace_case_and_punctuation():
    assert normalize_text("  What do you mean?  ") == "what do you mean"
    assert normalize_text("What do you mean") == "what do you mean"
    assert normalize_text("What\t do  you\nmean?!") == "what do you mean"
    assert normalize_text("WHAT DO YOU MEAN?") == "what do you mean"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    # genuinely different questions do not collide
    assert normalize_text("what is the first step") != normalize_text(
        "what is the second step"
    )
    assert normalize_text("do you read anything") != normalize_text(
        "do you write anything"
    )


# ---------------------------------------------------------------------------
# Repeated Agent question guard
# ---------------------------------------------------------------------------


def test_same_agent_question_twice_does_not_terminate(env_and_task):
    env, task = env_and_task
    # Q1 (twice) interleaved with responses — never reaches 3
    agent = _StubAgent(["Q1", "Q1", "done"])
    user = _StubUser(["A1", "A2"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    # Q1, A1, Q1, A2 = 4 steps; Q1 appears only twice -> no guard fires.
    orch = _run_steps(orch, 4)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


def test_third_identical_agent_question_terminates(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q1", "Q1", "done"])
    user = _StubUser(["A1", "A2", "A3"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    orch = _run_until_done(orch)
    assert orch.done is True
    assert orch.termination_reason == TerminationReason.REPEATED_QUESTION
    assert orch.loop_guard_diagnostics is not None
    diag = orch.loop_guard_diagnostics
    assert diag["type"] == "repeated_question"
    assert diag["threshold"] == 3
    assert diag["count"] == 3
    # loop guard fired BEFORE max_steps
    assert orch.step_count < 50


def test_repetitions_separated_by_other_messages_still_count(env_and_task):
    env, task = env_and_task
    # Q1, Q2, Q1, Q2, Q1 -> Q1 appears 3 times (non-consecutive)
    agent = _StubAgent(["Q1", "Q2", "Q1", "Q2", "Q1"])
    user = _StubUser(["A1", "A2", "A3", "A4", "A5"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    orch = _run_until_done(orch)
    assert orch.termination_reason == TerminationReason.REPEATED_QUESTION


def test_different_questions_do_not_terminate(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "Q3", "Q4", "Q5"])
    user = _StubUser(["A1", "A2", "A3", "A4", "A5"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    # Q1,A1..Q5,A5 = 10 steps; all questions distinct -> no guard fires.
    orch = _run_steps(orch, 10)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


def test_cosmetic_variants_count_as_same_question(env_and_task):
    env, task = env_and_task
    # cosmetic differences (case / punctuation / whitespace) are the SAME
    # normalized question -> 3rd occurrence terminates
    agent = _StubAgent(
        ["What is the first step?", "what  is the first step", "WHAT IS THE FIRST STEP"]
    )
    user = _StubUser(["A1", "A2", "A3"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    orch = _run_until_done(orch)
    assert orch.termination_reason == TerminationReason.REPEATED_QUESTION


# ---------------------------------------------------------------------------
# Repeated stakeholder response guard
# ---------------------------------------------------------------------------


def test_same_user_response_twice_does_not_terminate(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "done"])
    user = _StubUser(["R", "R"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    # R, Q1, R, Q2 = 4 steps; R appears only twice -> no guard fires.
    orch = _run_steps(orch, 4)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


def test_third_identical_user_response_terminates(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "Q3"])
    user = _StubUser(["R", "R", "R"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    orch = _run_until_done(orch)
    assert orch.done is True
    assert orch.termination_reason == TerminationReason.REPEATED_RESPONSE
    lg = orch.loop_guard_diagnostics
    assert lg is not None
    assert lg["type"] == "repeated_response"
    # the third identical response is recorded in the trajectory, then the
    # run terminates BEFORE another agent generation
    assert orch.step_count < 50


# ---------------------------------------------------------------------------
# Tool-only messages do not count
# ---------------------------------------------------------------------------


def test_repeated_successful_write_fires_stalled_tool_operation(env_and_task):
    _, task = env_and_task
    write = ("update_node", {"node_id": "node_a", "actor": "actor_a"})
    orch = _tool_orchestrator(task, [write] * 4)

    _run_until_done(orch)

    assert orch.termination_reason == TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is not None
    assert orch.loop_guard_diagnostics["type"] == "stalled_tool_operation"
    assert orch.loop_guard_diagnostics["cycle_period"] == 1
    assert orch.loop_guard_diagnostics["repetition_count"] == 4
    assert len(orch.loop_guard_diagnostics["fingerprint_hashes"]) == 4


def test_short_write_cycle_fires_stalled_tool_operation(env_and_task):
    _, task = env_and_task
    first = ("update_node", {"node_id": "node_a", "necessity_rationale": "unset"})
    second = (
        "update_node",
        {"node_id": "node_a", "necessity_rationale": "dont_know"},
    )
    orch = _tool_orchestrator(task, [first, second] * 3)

    _run_until_done(orch)

    assert orch.termination_reason == TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is not None
    assert orch.loop_guard_diagnostics["cycle_period"] == 2
    assert orch.loop_guard_diagnostics["repetition_count"] == 3


def test_unset_dont_know_rationale_oscillation_is_detected(env_and_task):
    _, task = env_and_task
    unset = ("update_node", {"node_id": "node_a", "unset": ["necessity_rationale"]})
    dont_know = (
        "record_dont_know",
        {"node_id": "node_a", "properties": ["necessity_rationale"]},
    )
    orch = _tool_orchestrator(task, [unset, dont_know] * 3)

    _run_until_done(orch)

    assert orch.termination_reason == TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is not None
    assert orch.loop_guard_diagnostics["cycle_period"] == 2


def test_new_public_response_resets_tool_operation_candidate(env_and_task):
    _, task = env_and_task
    write = ("update_node", {"node_id": "node_a", "actor": "actor_a"})
    orch = _tool_orchestrator(
        task,
        [write, write, write, "Please answer a new question.", write, write],
        user_contents=["first observation", "new accepted observation"],
    )

    _run_until_done(orch, max_loop=16)

    assert orch.termination_reason != TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is None


def test_different_write_targets_do_not_form_a_cycle(env_and_task):
    _, task = env_and_task
    script = [
        ("update_node", {"node_id": "node_a", "actor": "actor_a"}),
        ("update_node", {"node_id": "node_b", "actor": "actor_b"}),
        ("update_node", {"node_id": "node_a", "system": "system_a"}),
        ("update_node", {"node_id": "node_b", "system": "system_b"}),
    ] * 2
    orch = _tool_orchestrator(task, script)

    _run_until_done(orch, max_loop=20)

    assert orch.termination_reason != TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is None


def test_normal_graph_construction_does_not_stall(env_and_task):
    _, task = env_and_task
    script = [
        ("add_node", {"node_id": "node_a", "activity": "activity_a"}),
        ("add_node", {"node_id": "node_b", "activity": "activity_b"}),
        (
            "add_edge",
            {"edge_id": "edge_ab", "from_node": "node_a", "to_node": "node_b"},
        ),
        ("update_node", {"node_id": "node_a", "actor": "actor_a"}),
        ("update_node", {"node_id": "node_b", "actor": "actor_b"}),
    ]
    orch = _tool_orchestrator(task, script)

    _run_until_done(orch, max_loop=16)

    assert orch.termination_reason != TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is None


def test_read_only_tools_do_not_stall(env_and_task):
    _, task = env_and_task
    script = [("list_concepts", {}), ("validate_graph", {})] * 5
    orch = _tool_orchestrator(task, script)

    _run_until_done(orch, max_loop=24)

    assert orch.termination_reason != TerminationReason.STALLED_TOOL_OPERATION
    assert orch.loop_guard_diagnostics is None


def test_tool_only_agent_messages_do_not_count(env_and_task):
    env, task = env_and_task
    # The agent only makes tool calls (no conversational text) -> the
    # question counter must never trigger even though the run repeats.
    from tau2.data_model.message import ToolCall

    class ToolAgent(_StubAgent):
        def generate_next_message(self, message, state):
            return (
                AssistantMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="t1",
                            name="some_tool",
                            arguments={},
                            requestor="assistant",
                        )
                    ],
                    cost=0.0,
                ),
                state,
            )

    class ToolEnv:
        def __init__(self, env):
            self._env = env

        def get_response(self, tool_call):
            from tau2.data_model.message import ToolMessage

            return ToolMessage(
                id="t1",
                role="tool",
                content="tool_result",
                requestor="assistant",
            )

        def on_message(self, message):
            pass

        def sync_tools(self):
            pass

        def get_policy(self):
            return ""

        def get_domain_name(self):
            return "mock"

        def get_info(self, include_tool_info=False):
            return {}

        def set_state(self, **kwargs):
            pass

        def episode_complete(self):
            return False

    tool_env = ToolEnv(env)
    agent = ToolAgent([])
    user = _StubUser(["R", "R", "R", "R"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=tool_env,  # type: ignore[arg-type]
        task=task,
        max_steps=30,
    )
    orch.initialize()
    # Tool calls route AGENT -> ENV -> AGENT; run a handful of steps and
    # verify the run NEVER terminates for repetition (no conversational
    # question was ever recorded).
    for _ in range(12):
        if orch.done:
            break
        orch.step()
        orch._check_termination()
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


# ---------------------------------------------------------------------------
# Sidecar internal retries do NOT count as public repetitions
# ---------------------------------------------------------------------------


def test_sidecar_internal_retries_do_not_count_as_public_repetitions(env_and_task):
    """A stakeholder sidecar generation retry happens INSIDE one
    ``generate_next_message`` and returns a single public UserMessage. It
    must not advance the repeated-response counter by itself (a sidecar retry
    is a private repair, never a repeated public response; only a response
    that enters the trajectory is counted).

    Here the user internally fails the sidecar twice before producing one
    public answer each turn (mirroring the business_interview sidecar retry
    path), while still emitting "R" on three separate turns. If the internal
    retries were incorrectly counted as if they entered the trajectory, the
    repeated_response guard would fire after the FIRST turn. Correctly it
    fires only when the third PUBLIC response repeats - and the stored count
    is 3 (not 3*retries).
    """
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "Q3"])

    class _RetryingSidecarUser(_StubUser):
        """Simulates a stakeholder whose private sidecar is invalid twice and
        recovers on the retry inside a single generation. Tracks internal
        attempts vs. public emissions so a bug that counts internal retries as
        public repetitions would be observable."""

        def __init__(self, contents: list[str]):
            super().__init__(contents)
            self.internal_sidecar_attempts = 0
            self.public_emissions = 0

        def generate_next_message(self, message, state):
            # Simulate an internal sidecar retry: the first attempts fail and
            # only a later attempt yields a valid sidecar, but this all stays
            # inside the single generation call - the caller observes exactly
            # one UserMessage.
            self.internal_sidecar_attempts += 2  # two failed attempts each turn
            content, state = super().generate_next_message(message, state)
            self.public_emissions += 1
            return content, state

    user = _RetryingSidecarUser(["R", "R", "R"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
        max_repeated_questions=0,  # isolate the response guard
        max_repeated_interactions=0,
    )
    orch = _run_until_done(orch)
    assert orch.done is True
    assert orch.termination_reason == TerminationReason.REPEATED_RESPONSE
    lg = orch.loop_guard_diagnostics
    assert lg is not None
    assert lg["type"] == "repeated_response"
    # The public response repeated only 3 times, even though the user made 2
    # internal sidecar attempts per turn (6 total). Only the 3 public
    # emissions ever enter the trajectory/counter.
    assert lg["count"] == 3
    assert user.internal_sidecar_attempts == 6
    assert user.public_emissions == 3
    # Termination came at the third public emission, well before max_steps.
    assert orch.step_count < 50


# ---------------------------------------------------------------------------
# business_interview stalled_interaction guard
# ---------------------------------------------------------------------------


def test_identical_question_and_semantic_answer_three_times_stalls(env_and_task):
    """Same normalized question + same private semantic sidecar 3 times ->
    STALLED_INTERACTION (the interaction guard is the ONLY active guard here
    so the diagnostic is unambiguous)."""
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q1", "Q1", "Q1"])
    # Surface text differs slightly but the semantic signature is identical
    # (the first response pairs with the DEFAULT greeting; Q1 then pairs
    # with responses 2,3,4 -> the interaction signature repeats 3x)
    user = _SigUser(
        [
            "first reply",
            "I am not sure about that.",
            "Not sure about that.",
            "I really don't know.",
        ],
        [
            "[('skc_000', 'value')]",
            "[('skn_001', 'dont_know')]",
            "[('skn_001', 'dont_know')]",
            "[('skn_001', 'dont_know')]",
        ],
    )
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
        max_repeated_questions=0,  # isolate the interaction guard
        max_repeated_responses=0,
        max_repeated_interactions=3,
    )
    orch = _run_until_done(orch)
    assert orch.done is True
    assert orch.termination_reason == TerminationReason.STALLED_INTERACTION
    lg = orch.loop_guard_diagnostics
    assert lg is not None
    assert lg["type"] == "stalled_interaction"
    assert orch.step_count < 50


def test_same_question_different_semantic_answer_does_not_stall(env_and_task):
    """Q -> A1, Q -> A2, Q -> A3 (different semantic answers) is a legitimate
    clarification sequence: no interaction signature repeats -> no stall."""
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q1", "Q1", "Q1", "Q1"])
    user = _SigUser(
        ["greeting", "answer one", "answer two", "answer three", "answer four"],
        [
            None,
            "[('skc_001', 'value')]",
            "[('skc_002', 'value')]",
            "[('skc_003', 'value')]",
            "[('skc_004', 'value')]",
        ],
    )
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
        max_repeated_questions=0,
        max_repeated_responses=0,
        max_repeated_interactions=3,
    )
    # Q1 + A1..A4: four different semantic answers -> no interaction repeats.
    orch = _run_steps(orch, 8)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


def test_same_semantic_answer_to_different_questions_does_not_stall(env_and_task):
    """Different questions with the same semantic answer are NOT the same
    interaction signature (the pair includes the normalized question)."""
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "Q3", "Q4"])
    user = _SigUser(
        ["greeting", "r1", "r2", "r3"],
        [
            None,
            "[('skc_001', 'value')]",
            "[('skc_001', 'value')]",
            "[('skc_001', 'value')]",
        ],
    )
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
        max_repeated_questions=0,
        max_repeated_responses=0,
        max_repeated_interactions=3,
    )
    # Q1,A1,Q2,A2,Q3,A3: same semantic answer but DIFFERENT questions -> the
    # interaction keys all differ -> no stall.
    orch = _run_steps(orch, 6)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


# ---------------------------------------------------------------------------
# Existing termination behavior unchanged when no loop occurs
# ---------------------------------------------------------------------------


def test_max_steps_still_fires_without_loops(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"])
    user = _StubUser(["A1", "A2", "A3", "A4", "A5", "A6"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=3,
    )
    orch = _run_until_done(orch)
    assert orch.termination_reason == TerminationReason.MAX_STEPS
    assert orch.step_count >= 3


def test_guards_disabled_when_zero(env_and_task):
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q1", "Q1", "Q1", "Q1"])
    user = _StubUser(["R", "R", "R", "R", "R"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=30,
        max_repeated_questions=0,
        max_repeated_responses=0,
        max_repeated_interactions=0,
    )
    # 6 steps (Q1,A1 repeated) with all guards disabled -> nothing fires.
    orch = _run_steps(orch, 6)
    assert orch.termination_reason is None
    assert orch.loop_guard_diagnostics is None


def test_loop_guard_diagnostics_preserved_in_simulation_run(env_and_task):
    """When a guard fires, the finalized SimulationRun carries the exact
    termination reason and loop_guard diagnostics (hashed fingerprint, no
    private ids)."""
    env, task = env_and_task
    agent = _StubAgent(["Q1", "Q1", "Q1", "done"])
    user = _StubUser(["A1", "A2", "A3"])
    orch = Orchestrator(
        domain="mock",
        agent=agent,  # type: ignore[arg-type]
        user=user,  # type: ignore[arg-type]
        environment=env,
        task=task,
        max_steps=50,
    )
    # Run through the full public lifecycle (run() sets timestamps and
    # finalizes): the guard fires inside the loop and _finalize attaches the
    # loop_guard diagnostics to SimulationRun.info.
    import time

    orch._run_start_perf = time.perf_counter()
    orch._run_start_time = "2026-01-01T00:00:00"
    orch = _run_until_done(orch)
    assert orch.termination_reason == TerminationReason.REPEATED_QUESTION
    sim = orch._finalize()
    assert sim.termination_reason == TerminationReason.REPEATED_QUESTION.value
    assert sim.info is not None
    lg = sim.info["loop_guard"]
    assert lg["type"] == "repeated_question"
    assert "fingerprint_hash" in lg
    # the hash is NOT the raw text (no content leak in public output)
    assert "Q1" not in lg["fingerprint_hash"]
