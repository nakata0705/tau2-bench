from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.business_interview.dag import InterviewDB
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
    BUSINESS_INTERVIEW_TASK_SET_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(solo_mode: bool = False) -> Environment:
    """Build the business_interview environment.

    There is no pre-existing data: the database only accumulates what the
    interviewing agent records (observations + the inferred DAG).
    """
    db = InterviewDB()
    tools = InterviewTools(db)
    with open(BUSINESS_INTERVIEW_POLICY_PATH, "r") as fp:
        policy = fp.read()
    env = Environment(
        domain_name="business_interview",
        policy=policy,
        tools=tools,
        user_tools=None,  # The stakeholder is a plain conversational user.
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
