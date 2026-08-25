import re
from pathlib import Path
from typing import Optional

from tau2.data_model.message import Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    SemanticAnnotation,
    SemanticLedger,
    StakeholderKnowledgeCatalog,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import InterviewDB
from tau2.domains.business_interview.scenario import get_scenario
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
    BUSINESS_INTERVIEW_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

# ---------------------------------------------------------------------------
# Environment-owned Observation delivery.
#
# The environment creates one Observation per accepted Stakeholder message
# BEFORE the Agent sees that message. The Agent never calls an observation
# tool and can never create/mutate an Observation: the Observation id is
# delivered deterministically with the public text, embedded at the front of
# the Agent-visible message content:
#
#     [Observation obs_8] We do it to manage credit risk.
#
# The embedded id is a private-of-env formatting aid ONLY: the raw public
# text is what lives in the ledger and in the immutable Observation (quotes
# in the private sidecar are validated against the raw text). The embed/strip
# pair is deterministic so set_state replay of a recorded trajectory recovers
# the exact raw text and never double-embeds.
# ---------------------------------------------------------------------------


_OBSERVATION_EMBED_RE = r"^\s*\[Observation\s+obs_\d+\]\s*(.*)$"


def _embed_observation_id(text: str, obs_id: str) -> str:
    """Front the observation id onto the public text the Agent sees."""
    text = (text or "").strip()
    if not text:
        return text
    return f"[Observation {obs_id}] {text}"


def _strip_observation_id(content: Optional[str]) -> Optional[str]:
    """Recover the raw public text from an already-embedded Agent-visible
    content (idempotent replay); passes through non-embedded content."""
    if not content:
        return content
    m = re.match(_OBSERVATION_EMBED_RE, content, re.S)
    if m:
        return m.group(1)
    return content


def strip_observation_marker(text: Optional[str]) -> Optional[str]:
    """Public: strip a leading ``[Observation obs_N]`` marker if present
    (used by the orchestrator's loop-guard text normalization so the marker
    never affects repeated-response detection)."""
    return _strip_observation_id(text)


class BusinessInterviewEnvironment(Environment):
    """Environment that ingests the conversation into the interview DB and
    owns Observation creation.

    Every conversation message is recorded into ``db.messages`` (an
    environment-controlled audit ledger). Every ACCEPTED stakeholder (user)
    message automatically becomes an immutable Observation BEFORE the Agent
    sees it; the Agent receives the Observation id inline with the public
    text (``[Observation obs_N] <text>``) and never calls an observation
    tool.

    ``assertion_ledger`` is the private semantic sidecar ledger (annotations
    + dialogue events): when a stakeholder (user) message carries private
    sidecar metadata, it is validated deterministically and stored against
    that exact message's turn. Only the public message content ever enters
    the conversation; the ledger is never Agent-visible.

    ``episode_complete`` reports whether the interview was finished — the
    orchestrator uses it to terminate the episode immediately on a successful
    ``finish_interview()`` (distinct from max_steps truncation).
    """

    def __init__(
        self,
        *args,
        assertion_ledger: Optional[SemanticLedger] = None,
        scenario_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scenario_id = scenario_id
        self.assertion_ledger: SemanticLedger = (
            assertion_ledger if assertion_ledger is not None else SemanticLedger()
        )

    def get_evaluation_inputs(self, *, simulation_seed: Optional[int], user=None):
        """Return evaluator-private primary inputs for an offline run artifact.

        The first profile is replaced with the catalog object held by the
        actual stakeholder simulator when available.  This deliberately
        serializes the object used by the run rather than re-projecting Truth
        from a seed after the run has completed.
        """
        from tau2.domains.business_interview.artifact_provenance import (
            build_evaluation_inputs,
        )

        scenario = getattr(user, "_scenario", None)
        if scenario is None:
            scenario = get_scenario(self.scenario_id)
        if scenario is None:
            return None
        actual_knowledge = getattr(getattr(user, "_catalog", None), "knowledge", None)
        profiles = []
        for index, profile in enumerate(scenario.stakeholder_references):
            knowledge = (
                actual_knowledge
                if index == 0 and actual_knowledge is not None
                else profile.knowledge
            )
            profiles.append(
                {
                    "stakeholder_id": profile.stakeholder_id,
                    "stakeholder_name": profile.name,
                    "stakeholder_role": profile.role,
                    "stakeholder": profile.stakeholder,
                    "knowledge": knowledge,
                }
            )
        return build_evaluation_inputs(
            scenario.truth,
            profiles,
            simulation_seed=simulation_seed,
        )

    def episode_complete(self) -> bool:
        """True once the interview finished successfully."""
        tools = self.tools
        if tools is None:
            return False
        db = getattr(tools, "db", None)
        return bool(db is not None and db.interview_complete)

    def on_message(self, message: Message) -> None:
        tools = self.tools
        if tools is None:
            return
        db = getattr(tools, "db", None)
        if db is None:
            return
        turn = len(db.messages)
        content = getattr(message, "content", None)
        if isinstance(message, UserMessage):
            # A recorded trajectory already carries the embedded observation
            # id; recover the RAW public text first (idempotent replay) so
            # sidecar quotes, the ledger and the Observation all use the
            # pristine stakeholder utterance.
            raw = _strip_observation_id(content)
            raw_annotations = getattr(message, "stakeholder_annotations", None)
            if raw_annotations:
                annotations = [
                    SemanticAnnotation(**a) if isinstance(a, dict) else a
                    for a in raw_annotations
                ]
                self.assertion_ledger.bind(turn, annotations, raw)
            raw_alignments = getattr(message, "stakeholder_alignments", None)
            if raw_alignments:
                events = [
                    ConceptAlignmentAssertion(**a) if isinstance(a, dict) else a
                    for a in raw_alignments
                ]
                self.assertion_ledger.bind_alignment(turn, events, raw)
            raw_terminology = getattr(message, "stakeholder_terminology", None)
            if raw_terminology:
                events = [
                    TerminologyConfirmation(**a) if isinstance(a, dict) else a
                    for a in raw_terminology
                ]
                self.assertion_ledger.bind_terminology(turn, events, raw)
            # Environment owns Observation creation: an ACCEPTED stakeholder
            # utterance (sidecar validated above) becomes one Observation here
            # and one Agent-visible Observation id, automatically. A failed
            # sidecar bind raises above and never reaches this point, so no
            # Observation and no id are consumed.
            if raw is not None and str(raw).strip():
                obs_id = tools._capture_user_message(turn, str(raw))  # type: ignore[attr-defined]
                # deliver the Observation id to the Agent inline with the
                # raw public text; only this composed string reaches the Agent.
                message.content = _embed_observation_id(str(raw), obs_id)
                content = str(raw)
        db.messages.append(
            {
                "role": str(getattr(message, "role", "")),
                "content": content,
            }
        )


def get_environment(
    solo_mode: bool = False, scenario_id: Optional[str] = None
) -> Environment:
    """Build the business_interview environment.

    There is no pre-existing data: the database only accumulates the
    conversation, the authentic Observations captured from stakeholder messages,
    and the inferred graph. The private semantic ledger is created here and
    shared with the tools (and, via the environment, with the fact-grounded
    stakeholder simulator adapter). When ``scenario_id`` is given and
    resolvable, the scenario's StakeholderKnowledgeCatalog is installed on the
    ledger at construction (needed by the binding-aware tools; the live
    pipeline also installs it via the stakeholder simulator).
    """
    db = InterviewDB()
    ledger = SemanticLedger()
    if scenario_id is not None:
        scenario = get_scenario(scenario_id)
        if scenario is not None:
            ledger.install_catalog(StakeholderKnowledgeCatalog.from_scenario(scenario))
    tools = InterviewTools(db, assertion_ledger=ledger)
    try:
        with open(BUSINESS_INTERVIEW_POLICY_PATH, "r") as fp:
            policy = fp.read()
    except OSError as exc:
        raise RuntimeError(
            f"cannot read business_interview policy at {BUSINESS_INTERVIEW_POLICY_PATH}: {exc}"
        ) from exc
    env = BusinessInterviewEnvironment(
        domain_name="business_interview",
        policy=policy,
        tools=tools,
        user_tools=None,  # The stakeholder is a plain conversational user.
        assertion_ledger=ledger,
        scenario_id=scenario_id,
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    """Load the business_interview task set, optionally filtered by split."""
    tasks = load_file(BUSINESS_INTERVIEW_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    """Load the task split file (e.g. the 'base' split used by default)."""
    split_file = (
        Path(BUSINESS_INTERVIEW_TASK_SET_PATH).parent
        / f"split_{Path(BUSINESS_INTERVIEW_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
