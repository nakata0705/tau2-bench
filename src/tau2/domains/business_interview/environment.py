from pathlib import Path
from typing import Optional

from tau2.data_model.message import Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.facts import (
    ConceptAlignmentAssertion,
    SemanticAnnotation,
    SemanticLedger,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import InterviewDB
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
    BUSINESS_INTERVIEW_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class BusinessInterviewEnvironment(Environment):
    """Environment that ingests the conversation into the interview DB.

    Every conversation message is recorded into ``db.messages`` (an
    environment-controlled ledger). The agent can then capture stakeholder
    (user) messages as Observations via ``observe_message``; it never writes
    Observation text itself.

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.assertion_ledger: SemanticLedger = (
            assertion_ledger if assertion_ledger is not None else SemanticLedger()
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
            raw_annotations = getattr(message, "stakeholder_annotations", None)
            if raw_annotations:
                annotations = [
                    SemanticAnnotation(**a) if isinstance(a, dict) else a
                    for a in raw_annotations
                ]
                self.assertion_ledger.bind(turn, annotations, content)
            raw_alignments = getattr(message, "stakeholder_alignments", None)
            if raw_alignments:
                events = [
                    ConceptAlignmentAssertion(**a) if isinstance(a, dict) else a
                    for a in raw_alignments
                ]
                self.assertion_ledger.bind_alignment(turn, events, content)
            raw_terminology = getattr(message, "stakeholder_terminology", None)
            if raw_terminology:
                events = [
                    TerminologyConfirmation(**a) if isinstance(a, dict) else a
                    for a in raw_terminology
                ]
                self.assertion_ledger.bind_terminology(turn, events, content)
        db.messages.append(
            {
                "role": str(getattr(message, "role", "")),
                "content": content,
            }
        )


def get_environment(solo_mode: bool = False) -> Environment:
    """Build the business_interview environment.

    There is no pre-existing data: the database only accumulates the
    conversation, the authentic Observations captured from stakeholder messages,
    and the inferred graph. The private semantic ledger is created here and
    shared with the tools (and, via the environment, with the fact-grounded
    stakeholder simulator adapter).
    """
    db = InterviewDB()
    ledger = SemanticLedger()
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
