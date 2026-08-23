import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from loguru import logger

from tau2.agent.base_agent import (
    AgentError,
    HalfDuplexAgent,
    is_valid_agent_history_message,
)
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.orchestrator.modes import CommunicationMode
from tau2.user.user_simulator import DummyUser, UserSimulator, UserState
from tau2.user.user_simulator_base import (
    HalfDuplexUser,
    UserError,
    is_valid_user_history_message,
)
from tau2.utils.llm_utils import get_cost
from tau2.utils.normalization import normalize_text
from tau2.utils.utils import format_time, get_now

_OBSERVATION_MARKER_RE = re.compile(r"^\s*\[Observation\s+obs_\d+\]\s*(.*)$", re.S)
_MUTATING_TOOL_PREFIXES = (
    "add_",
    "create_",
    "delete_",
    "finish_",
    "mark_",
    "merge_",
    "record_",
    "remove_",
    "reset_",
    "set_",
    "update_",
)
_STRUCTURAL_TOOL_PREFIXES = (
    "add_",
    "create_",
    "delete_",
    "merge_",
    "remove_",
    "set_",
)


def _strip_observation_marker(text: Optional[str]) -> Optional[str]:
    """Strip a leading ``[Observation obs_N]`` envelope that business_interview
    embeds in front of the stakeholder's public text to deliver the Observation
    id to the Agent; the envelope is not part of the stakeholder's wording.
    Passes through everything else unchanged."""
    if not text:
        return text
    m = _OBSERVATION_MARKER_RE.match(text)
    if m:
        return m.group(1)
    return text


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)

# Type variables for generic orchestrators
# Base types for BaseOrchestrator - unbound to allow both half-duplex and full-duplex
BaseAgentT = TypeVar("BaseAgentT")
BaseUserT = TypeVar("BaseUserT")
TrajectoryItemT = TypeVar(
    "TrajectoryItemT"
)  # Message for half-duplex, Tick for full-duplex

# Half-duplex specific types for Orchestrator
AgentT = TypeVar("AgentT", bound=HalfDuplexAgent)
UserT = TypeVar("UserT", bound=HalfDuplexUser)
AgentInputMessage = UserMessage | ToolMessage | MultiToolMessage
UserInputMessage = AssistantMessage | ToolMessage | MultiToolMessage


def _as_agent_input(message: Optional[Message]) -> AgentInputMessage:
    """Return a message that can be delivered to a half-duplex agent."""
    if isinstance(message, (UserMessage, ToolMessage, MultiToolMessage)):
        return message
    raise ValueError(f"Invalid message for agent: {message}")


def _as_user_input(message: Optional[Message]) -> UserInputMessage:
    """Return a message that can be delivered to a half-duplex user."""
    if isinstance(message, (AssistantMessage, ToolMessage, MultiToolMessage)):
        return message
    raise ValueError(f"Invalid message for user: {message}")


def _message_timestamp(message: Message) -> str:
    """Return a sortable timestamp for a half-duplex trajectory message."""
    if isinstance(message, MultiToolMessage):
        raise ValueError(
            "MultiToolMessage cannot be included in a half-duplex trajectory"
        )
    return message.timestamp or ""


class BaseOrchestrator(ABC, Generic[BaseAgentT, BaseUserT, TrajectoryItemT]):
    """
    Abstract base class for orchestrators.

    Provides the common infrastructure for managing simulations between Agent, User,
    and Environment. Subclasses implement specific communication patterns:
    - Orchestrator: Half-duplex (turn-based) communication, trajectory of Messages
    - FullDuplexOrchestrator: Full-duplex (streaming) communication, trajectory of Ticks

    Type Parameters:
        BaseAgentT: The agent type
        BaseUserT: The user type
        TrajectoryItemT: The trajectory item type (Message for half-duplex, Tick for full-duplex)

    Shared Responsibilities:
        - Environment initialization and tool execution
        - Termination tracking (max steps, max errors, done state)
        - Trajectory management
        - Simulation run lifecycle (initialize, step loop, finalize)

    Subclass Responsibilities:
        - Communication-specific initialization
        - Step implementation for their communication pattern
        - Mode-specific termination checks
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgentT,
        user: BaseUserT,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        simulation_id: Optional[str] = None,
        timeout: Optional[float] = None,
        max_repeated_questions: Optional[int] = 3,
        max_repeated_responses: Optional[int] = 3,
        max_repeated_interactions: Optional[int] = 3,
        max_stalled_tool_operations: Optional[int] = 6,
    ):
        """
        Initialize the base orchestrator.

        Args:
            domain: The domain name of the simulation (e.g., 'airline', 'retail', 'telecom').
            agent: The agent instance.
            user: The user instance.
            environment: The environment instance that handles tool execution.
            task: The task specification containing initial state, goals, and evaluation criteria.
            max_steps: Maximum number of simulation steps before termination. Defaults to 100.
            max_errors: Maximum number of tool execution errors before termination. Defaults to 10.
            seed: Optional random seed for reproducibility. Defaults to None.
            simulation_id: Optional simulation ID. Defaults to generated UUID.
            timeout: Maximum wallclock time in seconds. None means no timeout.
            max_repeated_questions: How many times the same normalized
                conversational Agent question may appear before the run is
                terminated with ``TerminationReason.REPEATED_QUESTION``
                (default 3; ``0``/``None`` disables the guard).
            max_repeated_responses: How many times the same normalized
                stakeholder response may appear before termination with
                ``TerminationReason.REPEATED_RESPONSE`` (default 3;
                ``0``/``None`` disables the guard).
            max_repeated_interactions: How many times the same
                (normalized Agent question, stakeholder semantic answer)
                interaction may appear before termination with
                ``TerminationReason.STALLED_INTERACTION`` (default 3;
                ``0``/``None`` disables the guard). The semantic answer
                fingerprint is provided by the user implementation when
                available (e.g. the business_interview sidecar) and is never
                exposed to the Agent.
            max_stalled_tool_operations: Maximum suffix length considered for
                a successful Agent write-operation cycle (default 6;
                ``0``/``None`` disables this guard). Period-1 cycles fire
                after four identical writes; period-2 cycles fire after three
                repetitions.
        """
        self.domain = domain
        self.agent: BaseAgentT = agent
        self.user: BaseUserT = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.simulation_id = simulation_id or str(uuid.uuid4())

        # State tracking
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None

        # Termination tracking
        self.max_steps: int = max_steps
        self.max_errors: int = max_errors
        self.timeout: Optional[float] = timeout
        self.step_count: int = 0
        self.done: bool = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors: int = 0
        self._run_start_time: Optional[str] = None
        self._run_start_perf: Optional[float] = None

        # Conversation-loop guards (runtime safeguard; not evaluator
        # semantics). Counters keyed by normalized fingerprint; the third
        # identical fingerprint terminates the run early.
        self.max_repeated_questions = max_repeated_questions
        self.max_repeated_responses = max_repeated_responses
        self.max_repeated_interactions = max_repeated_interactions
        self.max_stalled_tool_operations = max_stalled_tool_operations
        self._question_counts: dict[str, int] = {}
        self._response_counts: dict[str, int] = {}
        self._interaction_counts: dict[tuple[str, str], int] = {}
        # last normalized conversational Agent question (for interaction pairing)
        self._last_question_norm: Optional[str] = None
        # per-key first/triggering step for diagnostics
        self._question_first_step: dict[str, int] = {}
        self._response_first_step: dict[str, int] = {}
        self._interaction_first_step: dict[tuple[str, str], int] = {}
        self._question_trigger_step: dict[str, int] = {}
        self._response_trigger_step: dict[str, int] = {}
        self._interaction_trigger_step: dict[tuple[str, str], int] = {}
        # Successful Agent write operations are tracked only as transient
        # structural fingerprints; raw arguments never enter diagnostics.
        self._tool_operation_history: list[tuple[str, str, int]] = []
        # populated when a loop guard fires; attached to SimulationRun.info
        self.loop_guard_diagnostics: Optional[dict] = None

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the orchestrator for simulation.

        Subclasses must implement mode-specific initialization:
        - Set up environment state
        - Initialize agent and user states
        - Set up initial messages/chunks
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Perform one step of the simulation.

        Subclasses implement their communication pattern:
        - Half-duplex: Turn-based message passing
        - Full-duplex: Simultaneous chunk generation
        """
        pass

    @abstractmethod
    def get_trajectory(self) -> list[TrajectoryItemT]:
        """
        Get the trajectory of the simulation.

        Returns:
            List of trajectory items. Type depends on orchestrator mode:
            - Orchestrator (half-duplex): list[Message]
            - FullDuplexOrchestrator: list[Tick]
        """
        pass

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """
        Get all messages from the simulation as a flat list.

        This provides a consistent way to get messages regardless of orchestrator mode.
        For half-duplex, this is the same as get_trajectory().
        For full-duplex, this returns linearized messages from all ticks.

        Returns:
            List of all messages sorted by timestamp with turn_idx assigned.
        """
        pass

    @abstractmethod
    def _validate_mode_compatibility(self) -> None:
        """
        Validate that the agent and user support this communication mode.

        Raises:
            ValueError: If agent or user don't support the required mode.
        """
        pass

    @abstractmethod
    def _check_termination(self) -> None:
        """
        Check for termination conditions specific to this communication mode.

        Sets self.done and self.termination_reason if termination conditions are met.
        """
        pass

    @abstractmethod
    def _finalize(self) -> SimulationRun:
        """
        Finalize the simulation and create the SimulationRun result.

        Called after the simulation loop completes. Should:
        - Send stop signals to agent and user
        - Calculate costs
        - Build and return SimulationRun

        Returns:
            SimulationRun with all simulation data.
        """
        pass

    def _check_timeout(self) -> None:
        if (
            self.timeout is not None
            and self._run_start_perf is not None
            and not self.done
        ):
            elapsed = time.perf_counter() - self._run_start_perf
            if elapsed >= self.timeout:
                self.done = True
                self.termination_reason = TerminationReason.TIMEOUT
                logger.info(
                    f"Simulation timed out after {elapsed:.1f}s (timeout={self.timeout}s)"
                )

    # ------------------------------------------------------------------
    # Conversation-loop guards (runtime safeguard; not evaluator semantics)
    # ------------------------------------------------------------------
    # The guards compare NORMALIZED message fingerprints (cosmetic
    # normalization only — never semantic similarity / LLMs / embeddings).
    # "Two are fine, three terminate": two occurrences of the same
    # normalized fingerprint are allowed (legitimate clarification/repetition);
    # the third occurrence terminates the run early so a broken run does not
    # consume max_steps. A threshold of 0 or None disables a guard.

    def _guard_enabled(self, threshold: Optional[int]) -> bool:
        """True when a loop guard is active (threshold > 0)."""
        return threshold is not None and threshold > 0

    def _record_agent_question(self, text: Optional[str], step_index: int) -> None:
        """Count one conversational Agent message (normalized) sent to the
        stakeholder. Tool-only / empty Agent messages are ignored. The last
        question is tracked whenever the question OR interaction guard is
        active (the interaction guard pairs it with the semantic answer)."""
        norm = normalize_text(text or "")
        if not norm:
            return
        if not self._guard_enabled(
            self.max_repeated_questions
        ) and not self._guard_enabled(self.max_repeated_interactions):
            return
        if norm not in self._question_counts:
            self._question_counts[norm] = 0
            self._question_first_step[norm] = step_index
        self._question_counts[norm] += 1
        self._question_trigger_step[norm] = step_index
        self._last_question_norm = norm

    def _record_user_response(
        self,
        text: Optional[str],
        step_index: int,
        semantic_signature: Optional[str] = None,
    ) -> None:
        """Count one stakeholder natural-language response (normalized), and
        when a semantic signature is available, the (question, semantic
        answer) interaction. Tool messages / private metadata are ignored;
        only the public message text is compared.

        In business_interview the environment delivers each Observation id
        inline at the front of the public text (``[Observation obs_N] ...``);
        that marker is NOT part of the stakeholder's wording, so it is
        stripped before normalization so repeated identical answers still
        collapse to one counter.
        """
        text = _strip_observation_marker(text)
        self._reset_tool_operation_history()
        norm = normalize_text(text or "")
        if not norm:
            return
        if self._guard_enabled(self.max_repeated_responses):
            if norm not in self._response_counts:
                self._response_counts[norm] = 0
                self._response_first_step[norm] = step_index
            self._response_counts[norm] += 1
            self._response_trigger_step[norm] = step_index
        if (
            self._guard_enabled(self.max_repeated_interactions)
            and semantic_signature
            and self._last_question_norm
        ):
            key = (self._last_question_norm, semantic_signature)
            if key not in self._interaction_counts:
                self._interaction_counts[key] = 0
                self._interaction_first_step[key] = step_index
            self._interaction_counts[key] += 1
            self._interaction_trigger_step[key] = step_index

    @staticmethod
    def _fingerprint_hash(key) -> str:
        """Short deterministic hash of a fingerprint key (for diagnostics;
        private semantic ids are never stored verbatim in public output)."""
        import hashlib

        return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _is_mutating_tool_operation(tool_name: str) -> bool:
        """Return whether a tool name conventionally represents a write.

        The guard intentionally uses only the public tool name. Read-oriented
        tools such as ``list_*`` / ``validate_*`` and arbitrary test helpers do
        not enter the state-oscillation history.
        """
        return str(tool_name).startswith(_MUTATING_TOOL_PREFIXES)

    @staticmethod
    def _tool_operation_fingerprint(tool_call: ToolCall) -> str:
        """Canonical transient fingerprint for one tool call."""
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        return json.dumps(
            {"name": tool_call.name, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    @staticmethod
    def _tool_operation_target(tool_call: ToolCall) -> str:
        """Canonical logical target used to reset on real target changes.

        Marker operations and ``update_node``/``update_edge`` share a target
        family, so changing the same node/property from UNSET to DONT_KNOW is
        recognized as an oscillation even when the tool-call spelling differs.
        Values remain in the full operation fingerprint, not this target key.
        """
        name = str(tool_call.name)
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

        if name in {"update_node", "record_dont_know", "record_absent"}:
            properties = {
                key
                for key in (
                    "activity",
                    "actor",
                    "system",
                    "reads",
                    "writes",
                    "necessity_rationale",
                )
                if key in arguments
            }
            for key in ("properties", "unset"):
                value = arguments.get(key)
                if isinstance(value, list):
                    properties.update(str(item) for item in value)
            return json.dumps(
                ["node_property", arguments.get("node_id"), sorted(properties)],
                sort_keys=True,
                default=str,
            )

        if name in {
            "update_edge",
            "record_edge_condition_dont_know",
            "record_edge_condition_absent",
        }:
            return json.dumps(
                ["edge_property", arguments.get("edge_id"), "condition"],
                sort_keys=True,
                default=str,
            )

        target_fields = {
            key: arguments[key]
            for key in (
                "node_id",
                "edge_id",
                "concept_id",
                "from_node",
                "to_node",
            )
            if key in arguments
        }
        operation_fields = sorted(
            key for key in arguments if key not in {"evidence", "summary"}
        )
        return json.dumps(
            [name, target_fields, operation_fields],
            sort_keys=True,
            default=str,
        )

    def _reset_tool_operation_history(self) -> None:
        """Forget a candidate cycle after a public response or real progress."""
        self._tool_operation_history.clear()

    def _record_tool_operation(self, tool_call: ToolCall, step_index: int) -> None:
        """Record a successful Agent write and fire on a short target-local cycle."""
        if not self._guard_enabled(self.max_stalled_tool_operations):
            return
        if getattr(tool_call, "requestor", "assistant") != "assistant":
            return
        if not self._is_mutating_tool_operation(tool_call.name):
            return

        if (
            str(tool_call.name).startswith(_STRUCTURAL_TOOL_PREFIXES)
            or str(tool_call.name) == "finish_interview"
        ):
            self._reset_tool_operation_history()
            return

        fingerprint = self._tool_operation_fingerprint(tool_call)
        target = self._tool_operation_target(tool_call)
        if (
            self._tool_operation_history
            and self._tool_operation_history[-1][1] != target
        ):
            self._reset_tool_operation_history()
        self._tool_operation_history.append((fingerprint, target, step_index))

        configured_limit = self.max_stalled_tool_operations or 0
        # Period 1 allows four identical writes; short period-2 cycles require
        # three complete repetitions. A configured limit caps the suffix size.
        for period in (1, 2, 3):
            repetitions = 4 if period == 1 else 3
            required = period * repetitions
            if (
                required > configured_limit
                or len(self._tool_operation_history) < required
            ):
                continue
            candidate = self._tool_operation_history[-required:]
            pattern = candidate[:period]
            if not all(
                item[1] == pattern[0][1] and item[0] == pattern[index % period][0]
                for index, item in enumerate(candidate)
            ):
                continue
            self.done = True
            self.termination_reason = TerminationReason.STALLED_TOOL_OPERATION
            self.loop_guard_diagnostics = {
                "type": "stalled_tool_operation",
                "reason": TerminationReason.STALLED_TOOL_OPERATION.value,
                "threshold": configured_limit,
                "count": required,
                "cycle_period": period,
                "repetition_count": repetitions,
                "first_step": candidate[0][2],
                "trigger_step": candidate[-1][2],
                "fingerprint_hashes": [
                    self._fingerprint_hash(item[0]) for item in candidate
                ],
            }
            logger.warning(
                "Tool-operation guard fired after {} operations "
                "(period={}, repetitions={}) at step {}; terminating early",
                required,
                period,
                repetitions,
                step_index,
            )
            return

    def _check_loop_guards(self) -> bool:
        """Check repetition counters; if any threshold is exceeded, terminate
        the run with the dedicated termination reason.

        Returns True when a loop guard fired (done + reason set).
        """
        if self.loop_guard_diagnostics is not None:
            return True
        if self._guard_enabled(self.max_repeated_questions):
            threshold = self.max_repeated_questions or 0
            for norm, count in self._question_counts.items():
                if count >= threshold:
                    self._fire_loop_guard(
                        "repeated_question",
                        TerminationReason.REPEATED_QUESTION,
                        threshold,
                        count,
                        norm,
                        self._question_first_step.get(norm),
                        self._question_trigger_step.get(norm),
                    )
                    return True
        if self._guard_enabled(self.max_repeated_responses):
            threshold = self.max_repeated_responses or 0
            for norm, count in self._response_counts.items():
                if count >= threshold:
                    self._fire_loop_guard(
                        "repeated_response",
                        TerminationReason.REPEATED_RESPONSE,
                        threshold,
                        count,
                        norm,
                        self._response_first_step.get(norm),
                        self._response_trigger_step.get(norm),
                    )
                    return True
        if self._guard_enabled(self.max_repeated_interactions):
            threshold = self.max_repeated_interactions or 0
            for key, count in self._interaction_counts.items():
                if count >= threshold:
                    self._fire_loop_guard(
                        "stalled_interaction",
                        TerminationReason.STALLED_INTERACTION,
                        threshold,
                        count,
                        key,
                        self._interaction_first_step.get(key),
                        self._interaction_trigger_step.get(key),
                    )
                    return True
        return False

    def _fire_loop_guard(
        self,
        guard_type: str,
        reason: TerminationReason,
        threshold: int,
        count: int,
        fingerprint,
        first_step: Optional[int] = None,
        trigger_step: Optional[int] = None,
    ) -> None:
        """Terminate the run because a loop guard fired, and record concise
        diagnostics (hashed fingerprint — private semantic ids never leak)."""
        self.done = True
        self.termination_reason = reason
        self.loop_guard_diagnostics = {
            "type": guard_type,
            "reason": reason.value,
            "threshold": threshold,
            "count": count,
            "fingerprint_hash": self._fingerprint_hash(fingerprint),
            "first_step": first_step,
            "trigger_step": trigger_step,
        }
        logger.warning(
            f"Loop guard [{guard_type}] fired after {count} occurrences "
            f"(threshold={threshold}) at step {trigger_step}; terminating "
            f"early: {reason.value}"
        )

    def _cleanup(self) -> None:
        """Best-effort cleanup of agent and user resources.

        Called from the ``finally`` block of :meth:`run` so that WebSocket
        connections, background threads, and other resources are released
        even when ``step()`` raises an unexpected exception.

        On the normal (non-error) path ``_finalize()`` handles cleanup
        as part of building the result, so this method is a no-op.
        """
        try:
            if hasattr(self, "agent") and self.agent is not None:
                stop_agent = getattr(self.agent, "stop", None)
                if callable(stop_agent):
                    stop_agent(None, getattr(self, "agent_state", None))
        except Exception as e:
            logger.warning(f"Error during agent cleanup: {e}")

        try:
            if hasattr(self, "user") and self.user is not None:
                stop_user = getattr(self.user, "stop", None)
                if callable(stop_user):
                    stop_user(None, getattr(self, "user_state", None))
        except Exception as e:
            logger.warning(f"Error during user cleanup: {e}")

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Template method that orchestrates the simulation lifecycle:
        1. Initialize the simulation
        2. Step until done
        3. Check termination conditions after each step
        4. Finalize and return results

        Returns:
            SimulationRun: The simulation run with all results.
        """
        self._run_start_time = get_now()
        self._run_start_perf = time.perf_counter()
        self.initialize()

        finalized = False
        try:
            while not self.done:
                self.step()
                self._check_termination()
            result = self._finalize()
            finalized = True
            return result
        finally:
            if not finalized:
                logger.warning(
                    "Simulation loop exited with an exception — "
                    "running emergency cleanup"
                )
                self._cleanup()

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ) -> None:
        """
        Initialize the environment with the given state.

        Args:
            initialization_data: Optional data to initialize environment state.
            initialization_actions: Optional actions to execute during initialization.
            message_history: Message history for context.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolMessage]:
        """
        Execute tool calls and return results.

        Args:
            tool_calls: List of tool calls to execute.

        Returns:
            List of ToolMessage results from the environment.
        """
        tool_results = []
        for tool_call in tool_calls:
            tool_result = self.environment.get_response(tool_call)
            if tool_result.error:
                self.num_errors += 1
            else:
                self._record_tool_operation(tool_call, self.step_count)
            tool_results.append(tool_result)
        return tool_results

    def _wrap_tool_results(self, tool_results: list[ToolMessage]) -> Message:
        """
        Wrap tool results in appropriate message type.

        Args:
            tool_results: List of tool message results.

        Returns:
            Single ToolMessage if one result, MultiToolMessage if multiple.
        """
        if len(tool_results) > 1:
            return MultiToolMessage(role="tool", tool_messages=tool_results)
        return tool_results[0]

    def _get_environment_info(self) -> EnvironmentInfo:
        """Get the environment info."""
        return self.environment.get_info()


class Orchestrator(BaseOrchestrator[AgentT, UserT, Message]):
    """
    Orchestrator for half-duplex (turn-based) simulation.

    Passes messages between the Agent, User, and Environment in alternating turns.

    Communication Protocol:
        The orchestrator manages message flow between three roles: AGENT, USER, and ENV(ironment).
        Messages are passed in a turn-based manner following these rules:

        Message Types:
            - AssistantMessage: Sent by the agent
            - UserMessage: Sent by the user
            - ToolMessage: Sent by the environment in response to tool calls
            - MultiToolMessage: Wraps multiple tool messages when multiple tool calls are made

        Message Content Rules:
            1. Messages must contain EITHER text content OR tool calls, never both
            2. Messages cannot be empty (must have either text or tool calls)
            3. Tool calls must be followed by corresponding tool messages from the environment

        Communication Flow:
            - AGENT -> USER: Agent sends text response to user
            - AGENT -> ENV: Agent makes tool call(s) to environment
            - USER -> AGENT: User sends text message to agent
            - USER -> ENV: User makes tool call(s) to environment
            - ENV -> AGENT: Environment returns tool results to agent (after agent's tool call)
            - ENV -> USER: Environment returns tool results to user (after user's tool call)

        Solo Mode:
            In solo mode, the user is replaced by a DummyUser and the agent operates autonomously:
            - Agent can ONLY send tool calls (no text messages to user)
            - Exception: Agent can send stop signal (###STOP###) to end simulation
            - Agent interacts exclusively with the environment until completion

        Termination:
            Simulation ends when:
            - Agent sends stop signal (###STOP###)
            - User sends stop signal
            - Maximum steps (max_steps) reached
            - Maximum errors (max_errors) reached
            - Communication protocol violation detected (if validate_communication=True)
    """

    def __init__(
        self,
        domain: str,
        agent: AgentT,
        user: UserT,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        simulation_id: Optional[str] = None,
        validate_communication: bool = False,
        timeout: Optional[float] = None,
        max_repeated_questions: Optional[int] = 3,
        max_repeated_responses: Optional[int] = 3,
        max_repeated_interactions: Optional[int] = 3,
        max_stalled_tool_operations: Optional[int] = 6,
    ):
        """
        Initialize the Orchestrator for managing simulation between Agent, User, and Environment.

        This orchestrator implements half-duplex (turn-based) communication where agent and user
        alternate sending complete messages. For streaming/full-duplex communication, use
        FullDuplexOrchestrator instead.

        Args:
            domain: The domain name of the simulation (e.g., 'airline', 'retail', 'telecom').
            agent: The agent instance that will respond to user requests and make tool calls.
            user: The user instance that interacts with the agent (can be UserSimulator or DummyUser).
            environment: The environment instance that handles tool execution and maintains state.
            task: The task specification containing initial state, goals, and evaluation criteria.
            max_steps: Maximum number of simulation steps before termination. Defaults to 100.
            max_errors: Maximum number of tool execution errors before termination. Defaults to 10.
            seed: Optional random seed for reproducibility of agent and user behavior. Defaults to None.
            solo_mode: If True, agent operates without user interaction (only tool calls allowed).
                      Requires agent to be LLMSoloAgent or GymAgent, and user to be DummyUser.
                      Defaults to False.
            validate_communication: If True, validates communication protocol rules (e.g., no mixed
                                   messages with both text and tool calls). Defaults to False.
            timeout: Maximum wallclock time in seconds. None means no timeout.
            max_repeated_questions: How many times the same normalized
                conversational Agent question may appear before termination
                with ``TerminationReason.REPEATED_QUESTION`` (default 3;
                ``0``/``None`` disables the guard).
            max_repeated_responses: How many times the same normalized
                stakeholder response may appear before termination with
                ``TerminationReason.REPEATED_RESPONSE`` (default 3;
                ``0``/``None`` disables the guard).
            max_repeated_interactions: How many times the same
                (normalized Agent question, stakeholder semantic answer)
                interaction may appear before termination with
                ``TerminationReason.STALLED_INTERACTION`` (default 3;
                ``0``/``None`` disables the guard).
            max_stalled_tool_operations: Maximum suffix length considered for
                successful Agent write-operation cycles (default 6;
                ``0``/``None`` disables the guard).
        """
        # Initialize base class
        super().__init__(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
            simulation_id=simulation_id,
            timeout=timeout,
            max_repeated_questions=max_repeated_questions,
            max_repeated_responses=max_repeated_responses,
            max_repeated_interactions=max_repeated_interactions,
            max_stalled_tool_operations=max_stalled_tool_operations,
        )

        # Half-duplex specific attributes
        self.mode = CommunicationMode.HALF_DUPLEX
        self.trajectory: list[Message] = []
        self.solo_mode = solo_mode
        self.validate_communication = validate_communication

        # Turn-based routing state
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None

        # Validate mode compatibility
        self._validate_mode_compatibility()

    def _validate_mode_compatibility(self):
        """
        Validate that the agent and user support half-duplex communication.

        Raises:
            ValueError: If agent or user don't support half-duplex mode.
        """
        if not hasattr(self.agent, "generate_next_message"):
            raise ValueError(
                f"Agent {self.agent.__class__.__name__} must have 'generate_next_message' method."
            )

        if not hasattr(self.user, "generate_next_message"):
            raise ValueError(
                f"User {self.user.__class__.__name__} must have 'generate_next_message' method."
            )

        logger.info(
            f"Orchestrator initialized in HALF_DUPLEX mode (turn-based) with "
            f"agent={self.agent.__class__.__name__}, "
            f"user={self.user.__class__.__name__}"
        )

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for msg in message_history:
            if isinstance(msg, MultiToolMessage):
                raise ValueError(
                    "MultiToolMessage cannot be included in a half-duplex message history"
                )
            msg.turn_idx = None

        # Add timestamps to the message history
        message_history = self._add_timestamps(message_history)

        if self.solo_mode:
            if not self.environment.solo_mode:
                raise ValueError("Environment should be in solo mode")
            if not (
                isinstance(self.agent, LLMSoloAgent)
                or self.agent.__class__.__name__ == "GymAgent"
            ):
                raise ValueError(
                    "Agent must be a LLMSoloAgent or GymAgent in solo mode"
                )
            if not isinstance(self.user, DummyUser):
                raise ValueError("User must be a DummyUser in solo mode")

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds for the agent, user
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Initialize the agent and user states
        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            # Last message is an assistant message
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():  # Last message is for the user
                    self.to_role = Role.USER
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            # Last message is a user message
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():  # Last message is for the agent
                    self.to_role = Role.AGENT
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            # Last message is a tool message
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history
        else:
            # No message history - initialize fresh
            self.user_state = self.user.get_init_state()
            if not self.solo_mode:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                self.agent_state = self.agent.get_init_state(
                    message_history=[first_message]
                )
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER
            else:
                self.agent_state = self.agent.get_init_state()
                generate_next_message = getattr(self.agent, "generate_next_message")
                first_message, self.agent_state = generate_next_message(
                    None, self.agent_state
                )
                self.trajectory = [first_message]
                self.message = first_message
                # In solo mode, there is no user, so if the message is not a tool call, then we end and report an agent error
                if not first_message.is_tool_call():
                    self.from_role = Role.AGENT
                    self.to_role = Role.USER
                    self.done = True
                    if self.agent.is_stop(first_message):
                        # If the agent is stopping (###STOP###)
                        self.termination_reason = TerminationReason.AGENT_STOP
                    else:
                        self.termination_reason = TerminationReason.AGENT_ERROR
                else:
                    self.from_role = Role.AGENT
                    self.to_role = Role.ENV
                    self.done = self.agent.is_stop(first_message)
                    if self.done:
                        self.to_role = Role.USER  # FIXIT: For now, we assume last message cannot be to the environment
                        self.termination_reason = TerminationReason.AGENT_STOP

        if self.validate_communication:
            self.check_communication_error()
        self.environment.sync_tools()

    def check_communication_error(self) -> None:
        """
        Check the orchestrator state for communication errors and handle them appropriately.

        Communication errors occur when agents or users violate the communication protocol rules:
        - Empty messages (no text content and no tool calls)
        - Mixed messages (both text content and tool calls in the same message)
        - Solo mode violations (agents sending text content instead of tool calls)

        When a communication error is detected:
        - Sets `self.done = True` to terminate the simulation
        - Sets `self.termination_reason` to either `AGENT_ERROR` or `USER_ERROR`
        - Re-raises any other exceptions that are not communication-related
        """
        try:
            self._check_communication_error()
        except Exception as exc:
            if isinstance(exc, AgentError):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_ERROR
            elif isinstance(exc, UserError):
                self.done = True
                self.termination_reason = TerminationReason.USER_ERROR
            else:
                # Re-raise all other exceptions
                raise

    def _check_communication_error(self) -> None:
        """
        Check the orchestrator state for communication protocol violations.

        Validates that messages follow the communication rules:
        1. Messages must have either text content OR tool calls, not both
        2. Messages cannot be empty (no text content and no tool calls)
        3. In solo mode, agents can only send tool calls (except for stop messages)

        Raises:
            AgentError: When the agent violates communication rules
            UserError: When the user violates communication rules
            ValueError: When from_role is invalid
        """
        role = self.from_role
        if role == Role.ENV:
            return
        message = self.message
        if message is None:
            raise ValueError("A participant message is required")
        if role == Role.USER:
            if not isinstance(message, UserMessage):
                raise ValueError(f"Invalid user message: {message}")
            exception_type = UserError
        elif role == Role.AGENT:
            if not isinstance(message, AssistantMessage):
                raise ValueError(f"Invalid agent message: {message}")
            exception_type = AgentError
        else:
            raise ValueError(f"Invalid from role: {role}")
        # Check if the message is empty
        if not message.is_tool_call() and not message.has_text_content():
            raise exception_type(f"{role.value} sent an empty message. {message}")
        # Check if the message has both text content and tool calls
        if message.is_tool_call() and message.has_text_content():
            raise exception_type(
                f"{role.value} sent both text content and tool calls. {message}"
            )

        # Check if the agent is allowed to send a message to the user
        if role == Role.AGENT and self.solo_mode:
            if not isinstance(message, AssistantMessage):
                raise ValueError(f"Invalid agent message: {message}")
            if message.has_text_content() and not self.agent.is_stop(message):
                raise exception_type(
                    f"{role.value} can only send tool calls. {message}"
                )

    def _check_termination(self) -> None:
        """
        Check for half-duplex specific termination conditions.

        Conversation-loop guards fire FIRST (before max_steps/max_errors) so
        a clearly-repeating interaction terminates early with its own
        diagnostic reason instead of consuming max_steps. max_steps /
        max_errors / timeout are only checked when not waiting for an
        environment response.
        """
        # Skip termination checks if we're waiting for environment to respond
        if self.to_role == Role.ENV:
            return
        # A termination reason already set (e.g. EPISODE_COMPLETE) wins.
        if self.termination_reason is not None:
            return

        # Loop guards: same normalized question/response/interaction repeated
        # `max_repeated_*` times terminates early (before max_steps).
        if self._check_loop_guards():
            return

        if self.step_count >= self.max_steps:
            self.done = True
            self.termination_reason = TerminationReason.MAX_STEPS
        if self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        self._check_timeout()

    def _finalize(self) -> SimulationRun:
        """
        Finalize the half-duplex simulation and create the SimulationRun result.

        Sends stop signals to agent and user, calculates costs, and builds the result.

        Returns:
            SimulationRun with all simulation data.
        """
        # Send stop signal to the agent, user, and environment
        has_error = self.termination_reason in [
            TerminationReason.USER_ERROR,
            TerminationReason.AGENT_ERROR,
        ]

        last_msg_to_agent: Optional[AgentInputMessage] = None
        last_msg_to_user: Optional[UserInputMessage] = None
        if self.to_role == Role.AGENT:
            last_msg_to_agent = _as_agent_input(self.message)
        elif self.to_role == Role.USER:
            last_msg_to_user = _as_user_input(self.message)
        elif self.to_role == Role.ENV and not has_error:
            raise ValueError(
                "Environment should not receive the last message. Last message: "
                + str(self.message)
            )
        try:
            self.agent.stop(last_msg_to_agent, self.agent_state)
        except Exception as e:
            logger.warning(f"Error stopping agent during finalization: {e}")
        try:
            self.user.stop(last_msg_to_user, self.user_state)
        except Exception as e:
            logger.warning(f"Error stopping user during finalization: {e}")

        # Wrap up the simulation
        if self._run_start_perf is None:
            raise RuntimeError("Simulation start time is not initialized")
        if self._run_start_time is None:
            raise RuntimeError("Simulation start timestamp is not initialized")
        if self.termination_reason is None:
            raise RuntimeError("Simulation termination reason is not set")
        termination_reason = self.termination_reason
        duration = time.perf_counter() - self._run_start_perf
        messages = self.get_trajectory()
        agent_cost, user_cost = get_cost(messages)
        # Update voice metadata with final turn_idx values
        self._finalize_voice_metadata(messages)

        # Get speech_environment from user's voice_settings if available
        speech_environment = None
        voice_settings = getattr(self.user, "voice_settings", None)
        if voice_settings is not None:
            speech_environment = voice_settings.speech_environment

        simulation_run = SimulationRun(
            id=self.simulation_id,
            task_id=self.task.id,
            start_time=self._run_start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=termination_reason,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
            mode=self.mode.value,
            speech_environment=speech_environment,
        )
        # Loop-guard diagnostics (only present when a guard fired). The stored
        # fingerprint is hashed — private semantic ids never leak into
        # Agent-visible / public output.
        if self.loop_guard_diagnostics is not None:
            simulation_run.info = {"loop_guard": self.loop_guard_diagnostics}
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation using half-duplex (turn-based) communication.

        Sends self.message from self.from_role to self.to_role.
        This can either be a message from agent to user/environment, environment to agent,
        or user to agent. Updates self.trajectory.
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        # AGENT/ENV -> USER
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                _as_user_input(self.message), self.user_state
            )
            user_msg.validate()
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            # Update voice metadata if audio was generated
            self._update_voice_metadata(user_msg)

            self.trajectory.append(user_msg)
            self.environment.on_message(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
            # Conversation-loop guard: count the stakeholder's NATURAL-LANGUAGE
            # response (public text only; tool messages / private metadata are
            # ignored). The user implementation may provide an optional private
            # semantic fingerprint for the stalled_interaction guard.
            if not user_msg.is_tool_call():
                sig_fn = getattr(self.user, "interaction_signature", None)
                semantic_signature: Optional[str] = None
                if callable(sig_fn):
                    sig = sig_fn(user_msg)
                    if isinstance(sig, str):
                        semantic_signature = sig
                self._record_user_response(
                    user_msg.content, self.step_count, semantic_signature
                )
        # USER/ENV -> AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            agent_msg, self.agent_state = self.agent.generate_next_message(
                _as_agent_input(self.message), self.agent_state
            )
            agent_msg.validate()
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP

            self.trajectory.append(agent_msg)
            self.environment.on_message(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.USER
                # Conversation-loop guard: count the Agent's conversational
                # question (tool-only / empty messages are ignored).
                if not self.solo_mode:
                    self._record_agent_question(agent_msg.content, self.step_count)
                # In solo mode, there is no user, so if the message is not a tool call and not a stop, then we end and report an agent error
                if self.solo_mode and not self.agent.is_stop(agent_msg):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_ERROR
        # AGENT/USER -> ENV
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            message = self.message
            if not isinstance(message, (AssistantMessage, UserMessage)):
                raise ValueError("Agent or User should send a participant message")
            tool_calls = message.tool_calls
            if tool_calls is None:
                raise ValueError("Agent or User should send tool call to environment")
            tool_results = self._execute_tool_calls(tool_calls)
            if len(tool_calls) != len(tool_results):
                raise RuntimeError(
                    "Number of tool calls and tool messages should be the same"
                )
            self.trajectory.extend(tool_results)
            for tr in tool_results:
                self.environment.on_message(tr)
            self.message = self._wrap_tool_results(tool_results)
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        if self.validate_communication:
            self.check_communication_error()
        self.step_count += 1
        self.environment.sync_tools()
        # Successful episode completion (e.g. business_interview
        # finish_interview) terminates the episode immediately — distinct
        # from max_steps truncation. First-set termination reason wins.
        if (
            self.termination_reason is None
            and getattr(self.environment, "episode_complete", lambda: False)()
        ):
            self.done = True
            self.termination_reason = TerminationReason.EPISODE_COMPLETE

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=_message_timestamp,
        )
        trajectory: list[Message] = []
        for i, msg in enumerate(messages):
            if isinstance(msg, MultiToolMessage):
                raise ValueError(
                    "MultiToolMessage cannot be included in a half-duplex trajectory"
                )
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    def get_messages(self) -> list[Message]:
        """
        Get all messages from the simulation.

        For half-duplex mode, this is the same as get_trajectory().
        """
        return self.get_trajectory()

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    tool_calls = msg.tool_calls
                    if tool_calls is None:
                        raise ValueError("A tool-call message must contain tool calls")
                    num_expected_tool_messages = len(tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages = 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(self, message_history: list[Message]) -> list[Message]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            if isinstance(msg, MultiToolMessage):
                raise ValueError(
                    "MultiToolMessage cannot be included in a half-duplex message history"
                )
            # Use ISO format (use_compact_format=False) to match get_now() default
            msg.timestamp = format_time(
                time_offset + timedelta(seconds=i), use_compact_format=False
            )
        return message_history

    def _voice_metadata_path(self, audio_path: str) -> Optional[Path]:
        """Return a metadata path constrained to the configured voice output."""
        voice_settings = getattr(self.user, "voice_settings", None)
        output_dir = getattr(voice_settings, "output_dir", None)
        if output_dir is None:
            return None

        output_root = Path(output_dir).expanduser().resolve()
        audio_file = Path(audio_path).expanduser().resolve()
        try:
            relative_audio_path = audio_file.relative_to(output_root)
        except ValueError:
            logger.warning(
                "Ignoring voice metadata path outside the configured output directory: {}",
                audio_path,
            )
            return None
        return output_root / relative_audio_path.parent / "metadata.json"

    def _update_voice_metadata(self, message: UserMessage) -> None:
        """
        Update voice metadata with simulation ID.
        Note: turn_idx is not available until get_trajectory() is called.
        """
        # Check if message has voice UUID (set during synthesis)
        voice_uuid = getattr(message, "_voice_uuid", None)
        if voice_uuid is None or not message.audio_path or not self.simulation_id:
            return

        metadata_path = self._voice_metadata_path(message.audio_path)
        if metadata_path is None:
            return
        metadata = {
            "simulation_id": self.simulation_id,
            "timestamp": message.timestamp,
            "turn_uuid": voice_uuid,
        }

        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Unable to write voice metadata {metadata_path}: {exc}")

    def _finalize_voice_metadata(self, messages: list[Message]) -> None:
        """
        Update all voice metadata files with final turn_idx values.
        """
        for msg in messages:
            voice_uuid = getattr(msg, "_voice_uuid", None)
            if (
                isinstance(msg, UserMessage)
                and voice_uuid is not None
                and msg.audio_path
            ):
                metadata_path = self._voice_metadata_path(msg.audio_path)
                if metadata_path is None or not metadata_path.exists():
                    continue

                if metadata_path.exists():
                    # Read existing metadata
                    try:
                        with metadata_path.open("r", encoding="utf-8") as f:
                            metadata = json.load(f)
                    except (OSError, TypeError, ValueError) as exc:
                        logger.warning(
                            f"Unable to read voice metadata {metadata_path}: {exc}"
                        )
                        continue

                    # Update with turn_idx
                    metadata["turn_idx"] = msg.turn_idx

                    # Write back
                    try:
                        with metadata_path.open("w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)
                    except (OSError, TypeError, ValueError) as exc:
                        logger.warning(
                            f"Unable to update voice metadata {metadata_path}: {exc}"
                        )
