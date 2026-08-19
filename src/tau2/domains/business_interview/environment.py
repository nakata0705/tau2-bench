from pathlib import Path
from typing import Optional

from tau2.data_model.message import Message, UserMessage
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.facts import (
    StakeholderAssertion,
    StakeholderFactLedger,
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

    ``fact_ledger`` is the private assertion sidecar ledger: when a
    stakeholder (user) message carries private ``assertions`` (from the
    fact-grounded stakeholder simulator), they are validated deterministically
    and stored against that exact message's turn. Only the public message
    content ever enters the conversation; the ledger is never Agent-visible.
    """

    def __init__(
        self, *args, fact_ledger: Optional[StakeholderFactLedger] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.fact_ledger: StakeholderFactLedger = (
            fact_ledger if fact_ledger is not None else StakeholderFactLedger()
        )

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
            raw_assertions = getattr(message, "stakeholder_assertions", None)
            if raw_assertions:
                # Private sidecar: validate deterministically (catalog present
                # in live runs) and store against this exact message's turn.
                # Invalid metadata is rejected loudly.
                assertions = [
                    StakeholderAssertion(**a) if isinstance(a, dict) else a
                    for a in raw_assertions
                ]
                self.fact_ledger.bind(turn, assertions, content)
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
    and the inferred DAG. The private used-fact ledger is created here and
    shared with the tools (and, via the environment, with the fact-grounded
    stakeholder simulator adapter).
    """
    db = InterviewDB()
    ledger = StakeholderFactLedger()
    tools = InterviewTools(db, fact_ledger=ledger)
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
        fact_ledger=ledger,
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
