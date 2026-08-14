"""Tests for the business_interview domain.

Covers the interview record tools, the deterministic env assertions used for
evaluation, task loading, and an end-to-end EnvironmentEvaluator check that
the four evaluation criteria are structurally decidable.
"""

import json

import pytest

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.domains.business_interview.data_model import InterviewDB
from tau2.domains.business_interview.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.business_interview.tools import InterviewTools
from tau2.domains.business_interview.utils import (
    BUSINESS_INTERVIEW_POLICY_PATH,
    BUSINESS_INTERVIEW_TASK_SET_PATH,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.user.user_simulator import UserSimulator

TASK_ID = "quotation_process_interview_1"


@pytest.fixture
def tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


@pytest.fixture
def task() -> Task:
    tasks = get_tasks()
    assert len(tasks) == 1
    return tasks[0]


def test_record_tools_and_assertions(tools: InterviewTools):
    """Recording the reference findings satisfies every env assertion."""
    tools.record_fact(
        content="The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        content="At month-end only, the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        content="The sales employee does not know why the month-end Excel file is necessary."
    )
    tools.finish_interview(
        summary="Month-end Excel reason unknown to the sales employee."
    )

    assert tools.assert_fact_recorded(all_of=["quotation", "system"])
    assert tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
    assert tools.assert_uncertainty_recorded(
        all_of=["month", "excel"],
        any_of=["reason", "why", "unknown", "not know", "unsure"],
    )
    assert tools.assert_no_unsupported_rationale()
    assert tools.assert_interview_complete()


def test_missing_findings_fail_assertions(tools: InterviewTools):
    """An interview that only covers the normal process fails all discovery checks."""
    tools.record_fact(content="Quotations are created in the core business system.")

    assert tools.assert_fact_recorded(all_of=["quotation", "system"])
    assert not tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
    assert not tools.assert_uncertainty_recorded(
        all_of=["month", "excel"],
        any_of=["reason", "why", "unknown", "not know", "unsure"],
    )
    assert not tools.assert_interview_complete()


def test_invented_rationale_is_detected(tools: InterviewTools):
    """An asserted rationale for the month-end exception is flagged even when
    the exception itself was discovered."""
    tools.record_exception(
        content="At month-end the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(content="The Excel file is probably for reconciliation.")

    assert tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
    assert not tools.assert_no_unsupported_rationale()


def test_legitimate_unknown_reason_is_not_a_rationale(tools: InterviewTools):
    """'The reason is unknown' must not be flagged as an invented rationale."""
    tools.record_uncertainty(
        content="The reason for the month-end Excel file is unknown to the sales employee."
    )
    assert tools.assert_no_unsupported_rationale()


def test_normal_process_purpose_phrase_is_not_a_rationale(tools: InterviewTools):
    """A purpose phrase in a finding about the *normal* process is legitimate
    (the interviewee naturally describes why steps are done) and must not be
    flagged. Only rationale phrases attached to the exception are checked."""
    tools.record_fact(
        content="The sales employee reviews the generated quotation to ensure all details are correct."
    )
    tools.record_fact(
        content="The quotation is entered into the core business system so that it can be handled there."
    )
    tools.record_exception(
        content="At month-end only, the sales employee sends an Excel file to Accounting."
    )
    assert tools.assert_no_unsupported_rationale()


def test_rationale_attached_to_exception_is_flagged(tools: InterviewTools):
    """A rationale attached to the exception is still flagged even when the
    purpose phrase is neutral-sounding (e.g. 'to ensure')."""
    tools.record_exception(
        content="At month-end the Excel file is sent to Accounting to ensure records match."
    )
    assert not tools.assert_no_unsupported_rationale()


def test_assertions_are_case_insensitive(tools: InterviewTools):
    tools.record_exception(
        content="At the End of each Month we send an EXCEL file to the accounting team."
    )
    assert tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])


def test_japanese_findings_match_assertions(tools: InterviewTools):
    """Findings recorded in Japanese (when the interview is conducted in
    Japanese) still satisfy the English keyword assertions via the bilingual
    synonym matching."""
    tools.record_fact(content="営業社員が見積書を基幹システムで作成しています。")
    tools.record_exception(
        content="毎月末のみ、営業社員がExcelファイルを経理チームに送付します。"
    )
    tools.record_uncertainty(
        content="月末のExcelファイルが必要な理由は分からないとのことでした。"
    )
    tools.finish_interview(
        summary="通常は基幹システムで見積書を作成。月末のみExcelを経理に送付。理由は不明。"
    )

    assert tools.assert_fact_recorded(all_of=["quotation", "system"])
    assert tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
    assert tools.assert_uncertainty_recorded(
        all_of=["month", "excel"],
        any_of=["reason", "why", "unknown", "not know", "unsure"],
    )
    assert tools.assert_no_unsupported_rationale()
    assert tools.assert_interview_complete()


def test_japanese_invented_rationale_is_detected(tools: InterviewTools):
    """An asserted rationale in Japanese for the month-end exception is
    flagged, even though the exception itself was discovered."""
    tools.record_exception(content="月末にExcelを経理へ送付しています。")
    tools.record_uncertainty(
        content="月末のExcelを経理に送るのは監査のためだと思います。"
    )

    assert tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
    assert not tools.assert_no_unsupported_rationale()


def test_japanese_legitimate_unknown_reason_is_not_a_rationale(tools: InterviewTools):
    """Japanese uncertainty phrasing ('理由は不明', '分からない') must not be
    flagged as an invented rationale."""
    tools.record_exception(content="月末にExcelを経理へ送付しています。")
    tools.record_uncertainty(content="月末のExcelファイルの理由は不明です。")
    assert tools.assert_no_unsupported_rationale()

    tools.record_uncertainty(content="なぜ必要なのか分からないとのことでした。")
    assert tools.assert_no_unsupported_rationale()


def test_language_prompt_injection():
    """The --language setting injects a language requirement into both the
    agent system prompt and the user simulator guidelines."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.persona import PersonaConfig
    from tau2.domains.business_interview.environment import get_environment
    from tau2.user.user_simulator import UserSimulator

    env = get_environment()
    agent = LLMAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="dummy",
        language="ja",
    )
    assert "<language_requirement>" in agent.system_prompt
    assert "Japanese" in agent.system_prompt

    agent_default = LLMAgent(
        tools=env.get_tools(), domain_policy=env.get_policy(), llm="dummy"
    )
    assert "<language_requirement>" not in agent_default.system_prompt

    user = UserSimulator(
        llm="dummy",
        instructions="dummy scenario",
        persona_config=PersonaConfig(language="ja"),
    )
    assert "Japanese" in user.system_prompt
    assert "###STOP###" in user.system_prompt

    user_default = UserSimulator(llm="dummy", instructions="dummy scenario")
    assert "LANGUAGE REQUIREMENT" not in user_default.system_prompt


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == TASK_ID
    assert TASK_ID in get_tasks_split()["base"]
    assert [t.id for t in get_tasks(task_split_name="base")] == [TASK_ID]


def test_initial_state_language_override():
    """The Japanese initial-state override is loaded and applied by the run
    pipeline when --language ja is set."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import _load_run_tasks, _localize_task_initial_state

    tasks = get_tasks()
    task = tasks[0]
    assert task.initial_state is not None
    assert "Hello" in task.initial_state.message_history[0].content

    overrides = task.initial_state_overrides
    assert overrides is not None and "ja" in overrides
    ja_initial = overrides["ja"].message_history[0].content
    assert "お世話になっております" in ja_initial

    # _localize_task_initial_state swaps the initial state for 'ja'
    localized = _localize_task_initial_state(task, "ja")
    assert localized.initial_state.message_history[0].content == ja_initial
    assert localized.id == task.id

    # Unknown languages keep the original initial state
    assert _localize_task_initial_state(task, "es") is task

    # The run pipeline applies the override when --language ja is set
    config = TextRunConfig(
        domain="business_interview", llm_agent="x", llm_user="x", language="ja"
    )
    run_tasks = _load_run_tasks(config)
    assert (
        "お世話になっております"
        in run_tasks[0].initial_state.message_history[0].content
    )

    config_en = TextRunConfig(
        domain="business_interview", llm_agent="x", llm_user="x", language=None
    )
    run_tasks_en = _load_run_tasks(config_en)
    assert "Hello" in run_tasks_en[0].initial_state.message_history[0].content


def test_no_hidden_ground_truth_in_stakeholder_or_policy(task: Task):
    """The stakeholder must not know why the month-end exception exists, so no
    plausible rationale for it may appear in the user scenario or the policy.

    The user scenario is what the stakeholder simulator sees; the policy is
    what the interviewing agent sees. Both must stay free of the hidden ground
    truth (the reason for the exception)."""
    # The stakeholder only knows that the exception exists, not why.
    scenario_text = str(task.user_scenario).lower()
    assert "month" in scenario_text  # the exception itself is known
    assert "reconciliation" not in scenario_text
    assert "audit" not in scenario_text
    assert "tax" not in scenario_text
    assert "compliance" not in scenario_text

    policy_text = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    assert "reconciliation" not in policy_text
    assert "audit" not in policy_text
    assert "tax" not in policy_text
    assert "compliance" not in policy_text


# ---------------------------------------------------------------------------
# End-to-end environment evaluation (deterministic, no LLM required)
# ---------------------------------------------------------------------------


def _tool_call_message(call_id: str, name: str, arguments: dict, result: str) -> list:
    return [
        AssistantMessage(
            role="assistant",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        ),
        ToolMessage(role="tool", id=call_id, content=result),
    ]


GOOD_TRAJECTORY = [
    AssistantMessage(
        role="assistant", content="Hello, I'd like to interview you about quotations."
    ),
    UserMessage(role="user", content="Sure, what would you like to know?"),
    *_tool_call_message(
        "c1",
        "record_fact",
        {
            "content": "The sales employee creates quotations in the core business system."
        },
        "Fact recorded (fact #1).",
    ),
    *_tool_call_message(
        "c2",
        "record_exception",
        {
            "content": "At month-end only, the sales employee sends an Excel file to Accounting."
        },
        "Exception recorded (exception #1).",
    ),
    *_tool_call_message(
        "c3",
        "record_uncertainty",
        {
            "content": "The sales employee does not know why the month-end Excel file is necessary."
        },
        "Uncertainty recorded (uncertainty #1).",
    ),
    *_tool_call_message(
        "c4",
        "finish_interview",
        {"summary": "Month-end Excel file reason unknown."},
        "Interview marked as complete.",
    ),
    AssistantMessage(
        role="assistant", content="Thank you, that's everything I needed."
    ),
]


def test_evaluator_rewards_complete_interview(task: Task):
    """A complete interview passes every env assertion and scores 1.0."""
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=GOOD_TRAJECTORY,
        solo_mode=False,
    )
    assert reward_info.reward == 1.0
    checks = {
        check.env_assertion.func_name: check.met for check in reward_info.env_assertions
    }
    assert checks == {
        "assert_fact_recorded": True,
        "assert_exception_recorded": True,
        "assert_uncertainty_recorded": True,
        "assert_no_unsupported_rationale": True,
        "assert_interview_complete": True,
    }


def test_evaluator_detects_missing_exception(task: Task):
    """An interview that never asks about exceptions fails the exception,
    uncertainty, and completion assertions (reward 0)."""
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="Hello, I'd like to interview you about quotations.",
        ),
        UserMessage(role="user", content="Sure."),
        *_tool_call_message(
            "c1",
            "record_fact",
            {
                "content": "The sales employee creates quotations in the core business system."
            },
            "Fact recorded (fact #1).",
        ),
        AssistantMessage(role="assistant", content="Thank you, that's everything."),
    ]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
        solo_mode=False,
    )
    assert reward_info.reward == 0.0
    checks = {
        check.env_assertion.func_name: check.met for check in reward_info.env_assertions
    }
    assert checks["assert_fact_recorded"] is True
    assert checks["assert_exception_recorded"] is False
    assert checks["assert_uncertainty_recorded"] is False
    assert checks["assert_interview_complete"] is False


def test_evaluator_detects_invented_rationale(task: Task):
    """Recording a guessed rationale for the exception fails the no-unsupported-
    rationale assertion even if the exception was discovered."""
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="Hello, I'd like to interview you about quotations.",
        ),
        UserMessage(role="user", content="Sure."),
        *_tool_call_message(
            "c1",
            "record_fact",
            {
                "content": "The sales employee creates quotations in the core business system."
            },
            "Fact recorded (fact #1).",
        ),
        *_tool_call_message(
            "c2",
            "record_exception",
            {
                "content": "At month-end the sales employee sends an Excel file to Accounting."
            },
            "Exception recorded (exception #1).",
        ),
        *_tool_call_message(
            "c3",
            "record_uncertainty",
            {
                "content": "The Excel file is sent because Accounting needs it for reconciliation."
            },
            "Uncertainty recorded (uncertainty #1).",
        ),
        AssistantMessage(role="assistant", content="Thank you, that's everything."),
    ]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
        solo_mode=False,
    )
    assert reward_info.reward == 0.0
    checks = {
        check.env_assertion.func_name: check.met for check in reward_info.env_assertions
    }
    assert checks["assert_exception_recorded"] is True
    assert checks["assert_no_unsupported_rationale"] is False


def test_task_json_has_expected_structure():
    """Guard against accidental edits that would break evaluation semantics."""
    with open(BUSINESS_INTERVIEW_TASK_SET_PATH) as fp:
        raw = json.load(fp)
    assert len(raw) == 1
    criteria = raw[0]["evaluation_criteria"]
    assert criteria["reward_basis"] == ["ENV_ASSERTION"]
    assert {a["func_name"] for a in criteria["env_assertions"]} == {
        "assert_fact_recorded",
        "assert_exception_recorded",
        "assert_uncertainty_recorded",
        "assert_no_unsupported_rationale",
        "assert_interview_complete",
    }
    # The user scenario must never include a rationale for the exception.
    scenario = json.dumps(raw[0]["user_scenario"]).lower()
    assert "because" not in scenario


# ---------------------------------------------------------------------------
# Orchestrator-level offline run (scripted agent + user, no LLM calls)
# ---------------------------------------------------------------------------


class _ScriptedAgent(LLMAgent):
    """An LLMAgent whose responses come from a script instead of an LLM."""

    def __init__(self, tools, domain_policy, script):
        super().__init__(tools=tools, domain_policy=domain_policy, llm="dummy")
        self.script = list(script)

    def _generate_next_message(self, message, state):
        step = self.script.pop(0)
        if isinstance(step, tuple):
            name, args = step
            return AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id=f"c{len(self.script)}", name=name, arguments=args)
                ],
            )
        return AssistantMessage(role="assistant", content=step)


class _ScriptedUser(UserSimulator):
    """A UserSimulator whose replies come from a script instead of an LLM."""

    def __init__(self, script):
        super().__init__(llm="dummy", instructions="dummy")
        self.script = list(script)

    def _generate_next_message(self, message, state):
        return UserMessage(role="user", content=self.script.pop(0))


def test_orchestrator_offline_run_scores_full_reward():
    """A scripted agent that discovers everything gets USER_STOP termination and
    a full reward from the env assertions (no LLM involved)."""
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.runner.simulation import run_simulation

    task = get_tasks()[0]
    environment = get_environment()

    agent_script = [
        (
            "record_fact",
            {
                "content": "The sales employee creates quotations in the core business system."
            },
        ),
        (
            "record_exception",
            {
                "content": "At month-end only, the sales employee sends an Excel file to Accounting."
            },
        ),
        (
            "record_uncertainty",
            {
                "content": "The sales employee does not know why the month-end Excel file is necessary."
            },
        ),
        ("finish_interview", {"summary": "Month-end Excel reason unknown."}),
        "Thank you for your time. That completes the interview.",
    ]
    agent = _ScriptedAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        script=agent_script,
    )
    user = _ScriptedUser(script=["No problem, glad to help. ###STOP###"])
    orchestrator = Orchestrator(
        domain="business_interview",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=100,
        max_errors=10,
        seed=42,
    )
    result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)

    assert result.termination_reason == "user_stop"
    assert result.reward_info.reward == 1.0
    checks = {
        check.env_assertion.func_name: check.met
        for check in result.reward_info.env_assertions
    }
    assert all(checks.values())
    assert environment.tools.db.interview_complete is True
