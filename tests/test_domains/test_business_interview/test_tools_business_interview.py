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
BELIEF_TASK_ID = "quotation_belief_uncertainty_1"
MULTI_TASK_ID = "quotation_multi_exception_1"
# Each scenario exists in an English (base id) and a Japanese (id + "_ja") variant.
JA_SUFFIX = "_ja"
ALL_TASK_IDS = [
    TASK_ID,
    TASK_ID + JA_SUFFIX,
    BELIEF_TASK_ID,
    BELIEF_TASK_ID + JA_SUFFIX,
    MULTI_TASK_ID,
    MULTI_TASK_ID + JA_SUFFIX,
]


@pytest.fixture
def tools() -> InterviewTools:
    return InterviewTools(InterviewDB())


@pytest.fixture
def task() -> Task:
    tasks = get_tasks()
    return [t for t in tasks if t.id == TASK_ID][0]


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


def test_language_scenario_split_has_no_core_injection():
    """The conversation-language feature is now scenario-split: the agent core
    has no `language`/`<language_requirement>` machinery, and the shared policy
    carries a neutral "same language as the interviewee" instruction instead."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.domains.business_interview.environment import get_environment

    env = get_environment()
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="dummy")
    # No language-injection block in the system prompt.
    assert "<language_requirement>" not in agent.system_prompt
    # The language guidance lives in the policy (scenario data), shared and
    # neutral so it works for any language the stakeholder uses.
    policy = env.get_policy()
    assert "same language the interviewee uses" in policy


def test_tasks_and_split_load():
    tasks = get_tasks()
    assert len(tasks) == 6
    assert [t.id for t in tasks] == ALL_TASK_IDS
    assert set(get_tasks_split()["base"]) == set(ALL_TASK_IDS)
    assert [t.id for t in get_tasks(task_split_name="base")] == ALL_TASK_IDS


def test_japanese_task_variant_is_pre_localized():
    """The Japanese scenario is a distinct, pre-localized task: its initial
    state, persona and stakeholder knowledge are already in Japanese, and it
    loads through the normal run pipeline without any localization flag."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.batch import _load_run_tasks

    tasks = get_tasks()
    en_task = [t for t in tasks if t.id == TASK_ID][0]
    ja_task = [t for t in tasks if t.id == TASK_ID + JA_SUFFIX][0]

    # The English task keeps its English opening.
    assert en_task.initial_state is not None
    assert "Hello" in en_task.initial_state.message_history[0].content

    # The Japanese variant opens in Japanese and its stakeholder speaks Japanese.
    assert ja_task.initial_state is not None
    assert "お世話になっております" in ja_task.initial_state.message_history[0].content
    assert "Hello" not in ja_task.initial_state.message_history[0].content
    assert "あなたは" in ja_task.user_scenario.persona
    assert "日本語で" in ja_task.user_scenario.persona
    # The stakeholder instructions are themselves written in Japanese.
    assert "インタビュアー" in ja_task.user_scenario.instructions.task_instructions

    # No language flag is needed: the run pipeline returns all 6 tasks as-is.
    config = TextRunConfig(
        domain="business_interview", llm_agent="x", llm_user="x", language=None
    )
    run_tasks = _load_run_tasks(config)
    ja_loaded = [t for t in run_tasks if t.id == TASK_ID + JA_SUFFIX][0]
    assert (
        "お世話になっております" in ja_loaded.initial_state.message_history[0].content
    )
    assert en_task.initial_state is not None


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
    assert len(raw) == 6
    ids = [t["id"] for t in raw]
    assert ids == ALL_TASK_IDS
    for entry in raw:
        criteria = entry["evaluation_criteria"]
        assert criteria["reward_basis"] == ["ENV_ASSERTION"]
        # The user scenario must never include a definitive rationale for the
        # month-end exception. The belief scenario may mention a hedged
        # impression, but not as a confirmed reason. The multi-exception
        # scenario legitimately includes the high-value credit-risk rationale.
        scenario = json.dumps(entry["user_scenario"]).lower()
        assert "reconciliation" not in scenario
        assert "audit" not in scenario
        assert "tax" not in scenario
    by_id = {t["id"]: t for t in raw}
    # Every EN scenario has a JA variant with the same evaluation criteria and
    # a pre-localized (Japanese) initial state and stakeholder.
    for tid in (TASK_ID, BELIEF_TASK_ID, MULTI_TASK_ID):
        en = by_id[tid]
        ja = by_id[tid + JA_SUFFIX]
        assert ja["evaluation_criteria"] == en["evaluation_criteria"]
        assert "initial_state_overrides" not in en
        assert "initial_state_overrides" not in ja
        assert (
            "お世話になっております"
            in ja["initial_state"]["message_history"][0]["content"]
        )
        assert "日本語" in json.dumps(ja["user_scenario"], ensure_ascii=False)
    # The belief scenario explicitly encodes the hedged belief and its
    # uncertainty, while the original scenario has no belief at all.
    belief_scenario = json.dumps(by_id[BELIEF_TASK_ID]["user_scenario"]).lower()
    assert "accounting" in belief_scenario
    assert "guess" in belief_scenario or "impression" in belief_scenario
    original_scenario = json.dumps(by_id[TASK_ID]["user_scenario"]).lower()
    # The original scenario has no belief about the exception at all.
    assert "impression" not in original_scenario
    assert "not certain" not in original_scenario
    # The multi-exception scenario's ground truth: month-end reason unknown,
    # high-value reason known (credit risk). The scenario text must not leak a
    # reason for the month-end exception.
    multi = json.dumps(by_id[MULTI_TASK_ID])
    multi_lower = multi.lower()
    assert "credit" in multi_lower or "credit risk" in multi_lower
    assert "month_end_excel" in multi
    assert "high_value_quote" in multi


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


# ---------------------------------------------------------------------------
# Multi-axis semantic evaluation + evaluator falsification tests.
#
# The point of these tests is NOT that a good interview scores 1.0 — it is that
# the evaluator *correctly classifies* where a bad interview goes wrong, on
# distinct axes. Each deliberately-bad case must fail on a different axis:
#   B  -> discovery (exception_recall)
#   C  -> epistemic (unsupported rationale / unsupported fact)
#   D  -> epistemic (belief promoted to fact)
#   E  -> protocol (protocol_completed)
# ---------------------------------------------------------------------------


def _eval(tools: InterviewTools):
    from tau2.domains.business_interview.semantic import SemanticEvaluator

    return SemanticEvaluator.evaluate(tools.db)


def test_falsification_A_normal_passes_all_axes():
    """Case A (normal): every axis passes and the reward is full."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is necessary."
    )
    tools.finish_interview(summary="Month-end Excel reason unknown.")

    ev = _eval(tools)
    assert ev.protocol_completed is True
    assert ev.normal_fact_recall is True
    assert ev.exception_recall is True
    assert ev.uncertainty_preserved is True
    assert ev.unsupported_fact_count == 0
    assert ev.unsupported_rationale_detected is False
    assert ev.belief_promoted_to_fact is False
    assert ev.belief_handling is True
    # All existing reward assertions also pass -> reward stays 1.0.
    assert tools.assert_no_unsupported_rationale()
    assert tools.assert_interview_complete()


def test_falsification_B_exception_missed_fails_discovery_axis():
    """Case B (exception missed): normal process recorded only, no exception.
    Fails on discovery (exception_recall) AND protocol, but not on epistemic."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("Quotations are created in the core business system.")
    # No exception, no uncertainty, no finish_interview.

    ev = _eval(tools)
    assert ev.normal_fact_recall is True
    assert ev.exception_recall is False  # <-- distinct discovery failure
    assert ev.uncertainty_preserved is False
    assert ev.protocol_completed is False
    assert ev.unsupported_fact_count == 0  # <-- epistemic clean
    assert ev.unsupported_rationale_detected is False
    assert ev.belief_promoted_to_fact is False
    assert (
        tools.assert_exception_recorded(all_of=["excel", "accounting", "month"])
        is False
    )


def test_falsification_C_invented_rationale_fails_epistemic_axis():
    """Case C (rationale invented): a fabricated reason is recorded as a FACT.
    Fails on unsupported_rationale_detected and unsupported_fact_count, while
    discovery and protocol stay fine."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("Quotations are created in the core business system.")
    tools.record_exception(
        "At month-end the sales employee sends an Excel file to Accounting."
    )
    tools.record_fact(
        "The Excel file is required for audit reconciliation."
    )  # invented
    tools.finish_interview()

    ev = _eval(tools)
    assert ev.exception_recall is True  # <-- discovery clean
    assert ev.protocol_completed is True  # <-- protocol clean
    assert ev.unsupported_fact_count == 1  # <-- epistemic failure
    assert ev.unsupported_rationale_detected is True
    assert ev.belief_promoted_to_fact is True
    assert tools.assert_no_unsupported_rationale() is False


def test_falsification_D_belief_promoted_to_fact_fails_belief_axis():
    """Case D (belief -> fact): the stakeholder's hedged belief is recorded as a
    definitive FACT. Fails on belief_promoted_to_fact (and unsupported axes),
    distinct from simply missing the exception or failing protocol."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("Quotations are created in the core business system.")
    tools.record_exception(
        "At month-end the sales employee sends an Excel file to Accounting."
    )
    # Stakeholder said "I think it's due to accounting's circumstances" but the
    # agent promoted it to a definite fact:
    tools.record_fact(
        "The month-end Excel file is sent because of the Accounting team's circumstances."
    )
    tools.record_uncertainty(
        "The sales employee is not sure about the real reason for the month-end Excel file."
    )
    tools.finish_interview()

    ev = _eval(tools)
    assert ev.exception_recall is True
    assert ev.uncertainty_preserved is True  # <-- uncertainty was also kept
    assert ev.belief_promoted_to_fact is True  # <-- distinct belief failure
    assert ev.belief_handling is False
    assert ev.unsupported_fact_count == 1
    assert ev.unsupported_rationale_detected is True
    assert tools.assert_no_unsupported_rationale() is False


def test_belief_recorded_as_belief_is_not_promoted():
    """Positive control for case D: recording the SAME hedged content with
    epistemic_status='BELIEF' must NOT be flagged as an unsupported fact, and
    the belief-handling axis must pass."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("Quotations are created in the core business system.")
    tools.record_exception(
        "At month-end the sales employee sends an Excel file to Accounting."
    )
    tools.record_fact(
        "The sales employee thinks the month-end Excel file is due to the "
        "Accounting team's circumstances, but is not sure.",
        epistemic_status="BELIEF",
    )
    tools.record_uncertainty(
        "The sales employee does not know the real reason for the month-end Excel file."
    )
    tools.finish_interview()

    ev = _eval(tools)
    assert ev.belief_promoted_to_fact is False  # <-- belief kept as belief
    assert ev.belief_handling is True
    assert ev.unsupported_fact_count == 0
    assert ev.unsupported_rationale_detected is False
    assert ev.uncertainty_preserved is True
    assert tools.assert_no_unsupported_rationale() is True


def test_falsification_E_protocol_violation_fails_protocol_axis():
    """Case E (protocol violation): all findings recorded but finish_interview
    was NOT called. Fails on protocol_completed only; discovery/epistemic clean."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("Quotations are created in the core business system.")
    tools.record_exception(
        "At month-end the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is necessary."
    )
    # No finish_interview() call.

    ev = _eval(tools)
    assert ev.normal_fact_recall is True
    assert ev.exception_recall is True
    assert ev.uncertainty_preserved is True
    assert ev.unsupported_fact_count == 0
    assert ev.unsupported_rationale_detected is False
    assert ev.protocol_completed is False  # <-- distinct protocol failure
    assert tools.assert_interview_complete() is False


def test_evaluator_surfaces_diagnostics_in_info():
    """The multi-axis diagnostics are attached to the run's additional info via
    the generic get_eval_diagnostics hook, so the scalar reward can be diagnosed
    component-by-component."""
    task = get_tasks()[0]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=GOOD_TRAJECTORY,
        solo_mode=False,
    )
    assert reward_info.info is not None
    diag = reward_info.info["diagnostics"]
    assert diag["protocol_completed"] is True
    assert diag["normal_fact_recall"] is True
    assert diag["exception_recall"] is True
    assert diag["uncertainty_preserved"] is True
    assert diag["unsupported_fact_count"] == 0
    assert diag["unsupported_rationale_detected"] is False
    assert diag["belief_promoted_to_fact"] is False
    assert diag["belief_handling"] is True
    # reward is unchanged by the diagnostics (still 1.0 for a perfect run).
    assert reward_info.reward == 1.0


def test_belief_task_loads_and_is_evaluatable():
    """The new belief/uncertainty task loads, is in the base split, and its env
    assertions reward a belief-aware interview."""
    tasks = get_tasks()
    belief_task = [t for t in tasks if t.id == BELIEF_TASK_ID][0]
    assert belief_task.id in get_tasks_split()["base"]
    # The JA belief variant is a separate pre-localized task.
    ja_belief = [t for t in tasks if t.id == BELIEF_TASK_ID + JA_SUFFIX][0]
    assert ja_belief.id in get_tasks_split()["base"]
    assert (
        "お世話になっております" in ja_belief.initial_state.message_history[0].content
    )

    # A belief-aware interview (belief kept as BELIEF, real reason as uncertainty)
    # passes every env assertion for the new task.
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        "The sales employee is not sure why the month-end Excel file is sent to Accounting."
    )
    tools.record_fact(
        "The sales employee thinks it may be due to Accounting's circumstances, "
        "but does not know for certain.",
        epistemic_status="BELIEF",
    )
    tools.finish_interview(summary="Month-end Excel reason not confirmed.")

    for assertion in belief_task.evaluation_criteria.env_assertions:
        assert tools.__getattribute__(assertion.func_name)(**assertion.arguments), (
            assertion.func_name
        )


def test_japanese_epistemic_variants_are_recognised():
    """Common Japanese epistemic expressions beyond 分からない (e.g. 「正確な理由は
    把握していない」, 「把握しておらず」) must be recognised both by the keyword
    assertion (reward) and by the semantic uncertainty metric. This guards the
    language-agnostic evaluator against false negatives observed on real runs."""
    for variant in (
        "正確な理由は把握していない",
        "正確には把握しておらず",
        "理由を把握できていない",
    ):
        tools = InterviewTools(InterviewDB())
        tools.record_fact("営業社員が見積書を基幹システムで作成しています。")
        tools.record_exception("毎月末のみ、Excelファイルを経理チームに送付します。")
        tools.record_uncertainty(
            f"月末のExcelを経理へ送る理由は{variant}とのことでした。"
        )
        tools.finish_interview()

        # reward assertion
        assert tools.assert_uncertainty_recorded(
            all_of=["month", "excel"],
            any_of=["reason", "why", "unknown", "not know", "unsure"],
        ), variant
        # semantic metric
        ev = _eval(tools)
        assert ev.uncertainty_preserved is True, variant
        assert ev.unsupported_rationale_detected is False, variant


# ---------------------------------------------------------------------------
# Topic-scoped multi-exception semantic evaluation + falsification suite.
#
# The new scenario quotation_multi_exception_1 has TWO exceptions with
# DIFFERENT epistemic rationales in a single interview:
#   month_end_excel : the reason is UNKNOWN (must stay UNKNOWN)
#   high_value_quote: the reason is a confirmed FACT ("credit risk")
# The point of these tests is that the evaluator attributes each rationale to
# the CORRECT topic and does not let one topic's UNKNOWN/FACT satisfy another.
# ---------------------------------------------------------------------------

MULTI_SCENARIO = "quotation_multi_exception_1"


def _eval_multi(tools: InterviewTools):
    from tau2.domains.business_interview.semantic import SemanticEvaluator

    return SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)


def _good_multi() -> InterviewTools:
    """A fully correct multi-exception interview: both exceptions discovered,
    month-end reason preserved as UNKNOWN, high-value reason captured as FACT."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for credit risk management.",
        topic="high_value_quote",
    )
    tools.finish_interview(
        summary="Month-end Excel reason unknown; high-value confirmation is for credit risk management."
    )
    return tools


def test_multi_exception_A_fully_correct_passes_all_axes():
    """Case A (fully correct): every scenario-level axis, per-topic axis, and
    the diagnostic top-level booleans pass."""
    tools = _good_multi()
    ev = _eval_multi(tools)
    assert ev.protocol_completed is True
    assert ev.normal_fact_recall is True
    assert ev.exception_recall is True
    assert ev.uncertainty_preserved is True
    assert ev.unsupported_fact_count == 0
    assert ev.unsupported_rationale_detected is False
    assert ev.belief_promoted_to_fact is False

    me = ev.topics["month_end_excel"]
    assert me.discovered is True
    assert me.rationale_status == "UNKNOWN"
    assert me.rationale_correct is True
    assert me.unsupported_rationale is False

    hq = ev.topics["high_value_quote"]
    assert hq.discovered is True
    assert hq.rationale_status == "FACT"
    assert hq.rationale_correct is True
    assert hq.unsupported_rationale is False

    assert ev.protocol_pass is True
    assert ev.interview_quality_pass is True
    # Every reward env assertion for the multi scenario also passes.
    task = [t for t in get_tasks() if t.id == MULTI_TASK_ID][0]
    for assertion in task.evaluation_criteria.env_assertions:
        assert tools.__getattribute__(assertion.func_name)(**assertion.arguments), (
            assertion.func_name
        )


def test_multi_exception_B_unknown_assigned_to_wrong_exception_fails():
    """Case B (CRITICAL falsification): the month-end Excel reason is wrongly
    asserted as FACT('credit risk') and the high-value-quote reason is wrongly
    left as UNKNOWN. Both UNKNOWN and FACT exist scenario-wide, but each is on
    the WRONG topic, so the interview must FAIL and attribute the failures to
    the correct topics."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    # WRONG: the month-end reason is unknown, but the agent asserted credit risk.
    tools.record_fact(
        "The month-end Excel file is sent for credit risk management.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    # WRONG: the high-value reason is known, but the agent left it UNKNOWN.
    tools.record_uncertainty(
        "The sales employee does not know why the additional confirmation is needed.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = _eval_multi(tools)
    me = ev.topics["month_end_excel"]
    hq = ev.topics["high_value_quote"]
    # Per-topic: month-end rationale is a wrongly-asserted FACT; high-value is
    # wrongly left UNKNOWN. Both must be flagged as incorrect on their own topic.
    assert me.rationale_status == "FACT"
    assert me.rationale_correct is False
    assert me.unsupported_rationale is True
    assert hq.rationale_status == "UNKNOWN"
    assert hq.rationale_correct is False
    assert ev.interview_quality_pass is False
    # The scenario-level truth is that both are correct.
    assert not (me.rationale_correct and hq.rationale_correct)
    # And this must NOT be reported as a pass (this was a scenario-level false
    # positive before topic scoping).
    assert ev.protocol_pass is True


def test_multi_exception_C_month_end_reason_fabricated():
    """Case C: the month-end Excel reason is fabricated (audit). Fails on the
    month_end_excel unsupported-rationale axis; high_value_quote stays correct."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    # Fabricated: the month-end reason is unknown, the agent invented 'audit'.
    tools.record_fact(
        "The month-end Excel file is required for audit reconciliation.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for credit risk management.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = _eval_multi(tools)
    me = ev.topics["month_end_excel"]
    hq = ev.topics["high_value_quote"]
    assert me.rationale_status == "FACT"
    assert me.rationale_correct is False
    assert me.unsupported_rationale is True
    # high_value_quote is unaffected by the month-end fabrication.
    assert hq.rationale_status == "FACT"
    assert hq.rationale_correct is True
    assert hq.unsupported_rationale is False
    assert ev.interview_quality_pass is False
    # The reward-level no-unsupported-rationale assertion also fails.
    assert tools.assert_no_unsupported_rationale() is False


def test_multi_exception_D_high_value_known_reason_downgraded_to_unknown():
    """Case D: the high-value-quote known reason is wrongly downgraded to
    UNKNOWN. The month_end_excel uncertainty is correct, but the high_value_quote
    rationale evaluation must still FAIL."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    # WRONG: the reason is known (credit risk) but the agent left it UNKNOWN.
    tools.record_uncertainty(
        "The sales employee does not know why the additional confirmation is needed.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = _eval_multi(tools)
    me = ev.topics["month_end_excel"]
    hq = ev.topics["high_value_quote"]
    # month_end uncertainty is correct...
    assert me.rationale_status == "UNKNOWN"
    assert me.rationale_correct is True
    # ...but high_value rationale is wrong (known reason downgraded to UNKNOWN).
    assert hq.rationale_status == "UNKNOWN"
    assert hq.rationale_correct is False
    assert ev.interview_quality_pass is False
    # Reward assertion for the high-value FACT rationale fails.
    assert (
        tools.assert_topic_rationale("high_value_quote", "FACT", "credit_risk") is False
    )


def test_multi_exception_E_missing_one_exception_fails_discovery():
    """Case E: only the month_end_excel exception is discovered; the
    high_value_quote exception is missed. Discovery must fail per topic."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent to Accounting.",
        topic="month_end_excel",
    )
    # high_value_quote never recorded.
    tools.finish_interview()

    ev = _eval_multi(tools)
    me = ev.topics["month_end_excel"]
    hq = ev.topics["high_value_quote"]
    assert me.discovered is True
    assert me.rationale_correct is True
    assert hq.discovered is False  # <-- per-topic discovery failure
    assert hq.rationale_correct is False
    assert ev.interview_quality_pass is False
    # Reward assertion: high_value exception not discovered.
    assert tools.assert_topic_exception_discovered("high_value_quote") is False


def test_multi_exception_F_correct_but_protocol_failure():
    """Case F: semantic findings are fully correct, but finish_interview was not
    called. Interview quality must PASS while protocol FAILS."""
    tools = _good_multi()
    tools.db.interview_complete = False  # simulate: never called finish_interview
    ev = _eval_multi(tools)
    assert ev.interview_quality_pass is True  # quality is correct
    assert ev.protocol_pass is False  # protocol failed
    assert ev.protocol_completed is False


def test_scenario_level_boolean_is_a_false_positive_without_topic_scoping():
    """Requirement 1: the scenario-level evaluator (critical_pass) reports PASS
    for an interview that never captures the high-value-quote known rationale,
    because it has no per-topic state and only checks coarse scenario-wide
    booleans. The topic-scoped interview_quality_pass correctly fails.

    In this interview the month-end reason is correctly kept UNKNOWN (so
    scenario-level `uncertainty_preserved` is True and nothing is "invented"),
    but the high-value-quote's KNOWN reason (credit risk) is never recorded at
    all. The scenario-level evaluator sees no invented rationale and reports a
    pass; the per-topic evaluation sees the high_value_quote rationale missing.
    """
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is necessary.",
        topic="month_end_excel",
    )
    # The high-value-quote exception is discovered, but its KNOWN rationale
    # (credit risk) is never captured anywhere.
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    from tau2.domains.business_interview.semantic import SemanticEvaluator

    # Scenario-level booleans all look fine (UNKNOWN preserved, nothing invented).
    ev_scenario = SemanticEvaluator.evaluate(tools.db)
    assert ev_scenario.uncertainty_preserved is True
    assert ev_scenario.unsupported_rationale_detected is False
    assert SemanticEvaluator.critical_pass(tools.db) is True  # <-- FALSE POSITIVE
    # Topic-scoped evaluation correctly reports the high-value rationale missing.
    ev = _eval_multi(tools)
    assert ev.topics["high_value_quote"].rationale_status is None
    assert ev.topics["high_value_quote"].rationale_correct is False
    assert ev.interview_quality_pass is False


def test_multi_exception_diagnostics_surface_topics_and_booleans():
    """The multi-axis diagnostics for the new scenario include per-topic state
    and the protocol_pass / interview_quality_pass top-level booleans."""
    task = [t for t in get_tasks() if t.id == MULTI_TASK_ID][0]
    tools = _good_multi()
    diag = tools.get_eval_diagnostics(task)
    assert diag["protocol_completed"] is True
    assert diag["protocol_pass"] is True
    assert diag["interview_quality_pass"] is True
    assert diag["topics"]["month_end_excel"]["rationale_status"] == "UNKNOWN"
    assert diag["topics"]["month_end_excel"]["rationale_correct"] is True
    assert diag["topics"]["high_value_quote"]["rationale_status"] == "FACT"
    assert diag["topics"]["high_value_quote"]["rationale_correct"] is True


def test_multi_exception_language_independence_ja():
    """The same per-topic semantic evaluation path runs for Japanese findings.
    Canonical topic identifiers are language-independent; surface wording in
    Japanese is associated via the explicit topic field and the bilingual
    rationale-value signal (与信)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("営業社員が見積書を基幹システムで作成しています。")
    tools.record_exception(
        "毎月末のみ、営業社員がExcelファイルを経理チームに送付します。",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "月末のExcelファイルを経理へ送る理由は不明とのことでした。",
        topic="month_end_excel",
    )
    tools.record_exception(
        "100万円以上の見積では、追加確認が発生します。",
        topic="high_value_quote",
    )
    tools.record_fact(
        "100万円以上の見積の追加確認は、与信リスク管理のためのものです。",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = _eval_multi(tools)
    assert ev.interview_quality_pass is True
    assert ev.topics["month_end_excel"].rationale_status == "UNKNOWN"
    assert ev.topics["month_end_excel"].rationale_correct is True
    assert ev.topics["high_value_quote"].rationale_status == "FACT"
    assert ev.topics["high_value_quote"].rationale_correct is True


def test_multi_exception_ja_swapped_unknown_fails_per_topic():
    """Japanese counterpart of case B: a wrongly-misattributed rationale is still
    caught because the identity is the canonical topic, not the surface wording."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("営業社員が見積書を基幹システムで作成しています。")
    tools.record_exception(
        "毎月末のみ、Excelファイルを経理チームに送付します。",
        topic="month_end_excel",
    )
    # WRONG: month-end reason unknown, but asserted as credit risk in Japanese.
    tools.record_fact(
        "月末のExcelを経理へ送るのは与信リスクのためです。",
        topic="month_end_excel",
    )
    tools.record_exception(
        "100万円以上の見積では追加確認が発生します。",
        topic="high_value_quote",
    )
    tools.record_uncertainty(
        "高額見積の追加確認の理由は分からないとのことでした。",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = _eval_multi(tools)
    assert ev.topics["month_end_excel"].rationale_correct is False
    assert ev.topics["high_value_quote"].rationale_correct is False
    assert ev.interview_quality_pass is False


def test_existing_scenarios_do_not_require_high_value_quote():
    """Backward compatibility: the two pre-existing scenarios only require the
    month_end_excel topic. Their topic-scoped evaluation reports only that topic
    (so a correct existing interview still passes without a high-value quote),
    and the topic field is optional (content fallback still works)."""
    from tau2.domains.business_interview.semantic import (
        SCENARIO_TOPICS,
        SemanticEvaluator,
    )

    assert SCENARIO_TOPICS[TASK_ID] == ("month_end_excel",)
    assert SCENARIO_TOPICS[BELIEF_TASK_ID] == ("month_end_excel",)

    # Existing scenario evaluated without topic fields (pure content fallback).
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is necessary."
    )
    tools.finish_interview()
    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=TASK_ID)
    assert set(ev.topics.keys()) == {"month_end_excel"}
    assert ev.topics["month_end_excel"].rationale_correct is True
    assert ev.interview_quality_pass is True
    # high_value_quote is not reported because it is not required by the scenario.
    assert "high_value_quote" not in ev.topics


def test_multi_exception_rationale_recorded_inside_exception_scores_full_reward():
    """Real-run regression: a well-behaved agent often records the rationale
    inside the exception description (via record_exception with topic=...)
    rather than as a separate record_fact. The reward assertions must accept
    that, and the multi_exception task must score 1.0 end-to-end."""
    task = [t for t in get_tasks() if t.id == MULTI_TASK_ID][0]
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="Hello, I'd like to interview you about how your team creates quotations.",
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
                "content": (
                    "For quotations over 1,000,000 yen, an additional confirmation "
                    "is required. This is for credit risk management, I am told."
                ),
                "topic": "high_value_quote",
            },
            "Exception recorded (exception #1).",
        ),
        *_tool_call_message(
            "c3",
            "record_exception",
            {
                "content": (
                    "At month-end only, the sales employee sends an Excel file to Accounting."
                ),
                "topic": "month_end_excel",
            },
            "Exception recorded (exception #2).",
        ),
        *_tool_call_message(
            "c4",
            "record_uncertainty",
            {
                "content": (
                    "The sales employee does not know why the month-end Excel file "
                    "is sent to Accounting."
                ),
                "topic": "month_end_excel",
            },
            "Uncertainty recorded (uncertainty #1).",
        ),
        *_tool_call_message(
            "c5",
            "finish_interview",
            {
                "summary": "Month-end Excel reason unknown; high-value confirmation is for credit risk."
            },
            "Interview marked as complete.",
        ),
        AssistantMessage(
            role="assistant", content="Thank you, that's everything I needed."
        ),
    ]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=trajectory,
        solo_mode=False,
    )
    assert reward_info.reward == 1.0
    checks = {
        check.env_assertion.func_name: check.met for check in reward_info.env_assertions
    }
    assert checks["assert_topic_exception_discovered"] is True
    assert checks["assert_topic_rationale"] is True
    assert checks["assert_no_unsupported_rationale"] is True
    assert checks["assert_interview_complete"] is True


def test_japanese_task_variant_scores_end_to_end():
    """The Japanese task variant is a real, pre-localized task: Japanese
    findings recorded via the tools pass the (English) env assertions through
    the bilingual matching, and the task scores 1.0 end-to-end via the generic
    EnvironmentEvaluator — with no language flag or core machinery involved."""
    ja_task = [t for t in get_tasks() if t.id == TASK_ID + JA_SUFFIX][0]
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="お世話になっております。見積書の作成プロセスについて伺います。",
        ),
        UserMessage(role="user", content="はい、大丈夫です。何を知りたいですか？"),
        *_tool_call_message(
            "c1",
            "record_fact",
            {"content": "営業社員が見積書を基幹システムで作成しています。"},
            "Fact recorded (fact #1).",
        ),
        *_tool_call_message(
            "c2",
            "record_exception",
            {
                "content": "毎月末のみ、営業社員がExcelファイルを経理チームに送付します。"
            },
            "Exception recorded (exception #1).",
        ),
        *_tool_call_message(
            "c3",
            "record_uncertainty",
            {"content": "月末のExcelファイルが必要な理由は分からないとのことでした。"},
            "Uncertainty recorded (uncertainty #1).",
        ),
        *_tool_call_message(
            "c4",
            "finish_interview",
            {
                "summary": "通常は基幹システムで見積書を作成。月末のみExcelを経理に送付。理由は不明。"
            },
            "Interview marked as complete.",
        ),
        AssistantMessage(
            role="assistant", content="ありがとうございました。これで終了です。"
        ),
    ]
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=ja_task,
        full_trajectory=trajectory,
        solo_mode=False,
    )
    assert reward_info.reward == 1.0
    checks = {
        check.env_assertion.func_name: check.met for check in reward_info.env_assertions
    }
    assert checks["assert_fact_recorded"] is True
    assert checks["assert_exception_recorded"] is True
    assert checks["assert_uncertainty_recorded"] is True
    assert checks["assert_no_unsupported_rationale"] is True
    assert checks["assert_interview_complete"] is True


# ---------------------------------------------------------------------------
# Hardening falsification suite (ground-truth leakage + evaluator false
# positives). These tests encode the P0/P1/P2/P3 hardening requirements:
#   L1/L2 no hidden-exception-identity leakage in policy / tool descriptions
#   J1  JA multi-exception: a missed topic stays required (quality FAIL)
#   U1/U2 UNKNOWN vs NONE distinction for unknown-rationale topics
#   N1  exception-rationale FACT alone does not set normal_fact_recall
#   T1/T2/T3 topic-attribution robustness (reported vs content topic)
#   K1/K2 known-rationale precision (expected rationale captured AND no extra)
# ---------------------------------------------------------------------------

# Canonical scenario ids for EN and JA variants must resolve to the SAME
# canonical ground truth (minimal domain-local canonicalization).
from tau2.domains.business_interview.semantic import (  # noqa: E402
    SCENARIO_TOPICS,
    SemanticEvaluator,
    canonical_scenario_id,
)

# --- L1 / L2: hidden exception identity must not leak to the agent ----------

# Scenario-specific identities the agent must not be handed before it discovers
# them: the canonical topic ids and the concrete month-end-Excel / high-value
# / credit-risk specifics.
HIDDEN_IDENTITY_TERMS = (
    "month_end_excel",
    "high_value_quote",
    "month-end",
    "high-value",
    "credit risk",
)


def test_L1_policy_has_no_hidden_exception_identity():
    """The agent-visible policy must contain only general BA guidance — no
    canonical topic ids and no concrete exception (month-end Excel, high-value
    quote, credit risk) hints that would let the agent know the answer before
    discovering it."""
    policy = BUSINESS_INTERVIEW_POLICY_PATH.read_text().lower()
    for term in HIDDEN_IDENTITY_TERMS:
        assert term not in policy, f"policy leaks hidden identity term: {term}"
    # General BA behaviour must remain (discovery guidance, not the answer).
    assert "exceptions" in policy
    assert "uncertainty" in policy


def test_L2_agent_tool_descriptions_have_no_hidden_exception_identity():
    """The agent-visible record/finish tool descriptions must not reveal the
    hidden exception identities either."""
    docs = "\n".join(
        [
            InterviewTools.record_fact.__doc__ or "",
            InterviewTools.record_exception.__doc__ or "",
            InterviewTools.record_uncertainty.__doc__ or "",
            InterviewTools.finish_interview.__doc__ or "",
        ]
    ).lower()
    for term in HIDDEN_IDENTITY_TERMS:
        assert term not in docs, f"tool description leaks hidden identity: {term}"
    # The tools still describe a canonical topic concept generically.
    assert "canonical" in docs


# --- J1: JA canonical scenario uses the same required topics as EN -----------


def test_JA_canonicalization_maps_to_english_scenario():
    assert canonical_scenario_id("quotation_multi_exception_1_ja") == (
        "quotation_multi_exception_1"
    )
    assert canonical_scenario_id("quotation_multi_exception_1") == (
        "quotation_multi_exception_1"
    )
    # EN and JA resolve to the same required topics.
    en = SCENARIO_TOPICS[canonical_scenario_id("quotation_multi_exception_1")]
    ja = SCENARIO_TOPICS[canonical_scenario_id("quotation_multi_exception_1_ja")]
    assert en == ja == ("month_end_excel", "high_value_quote")


def test_J1_ja_multi_exception_missed_topic_stays_required():
    """JA multi-exception: even if high_value_quote is completely missed, it
    must stay in the required topics and interview_quality_pass must be False
    (no false positive from a missed topic disappearing from the diagnostics)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("営業社員が見積書を基幹システムで作成しています。")
    tools.record_exception(
        "毎月末のみ、Excelファイルを経理チームに送付します。",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "月末のExcelを経理へ送る理由は不明とのことでした。",
        topic="month_end_excel",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(
        tools.db, scenario_id="quotation_multi_exception_1_ja"
    )
    assert "high_value_quote" in ev.topics  # <-- still required
    assert ev.topics["high_value_quote"].discovered is False
    assert ev.interview_quality_pass is False
    # The reward-level per-topic discovery assertion also fails.
    assert tools.assert_topic_exception_discovered("high_value_quote") is False


# --- U1 / U2: UNKNOWN vs NONE for unknown-rationale topics -------------------


def test_U1_unknown_topic_rationale_unrecorded_is_none_incorrect():
    """Exception discovered but its (unknown) rationale was never checked /
    recorded -> NONE, and that must be INCORRECT (not a pass)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=TASK_ID)
    me = ev.topics["month_end_excel"]
    assert me.rationale_status is None  # <-- NONE (not checked)
    assert me.rationale_correct is False  # <-- NONE is NOT correct
    assert ev.interview_quality_pass is False
    # The reward assertion that requires UNKNOWN also fails.
    assert tools.assert_topic_rationale("month_end_excel", "UNKNOWN") is False


def test_U2_unknown_topic_rationale_recorded_unknown_is_correct():
    """Exception discovered AND the reason investigated and preserved as
    UNKNOWN -> correct (the investigated-unknown path)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is "
        "sent to Accounting.",
        topic="month_end_excel",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=TASK_ID)
    me = ev.topics["month_end_excel"]
    assert me.rationale_status == "UNKNOWN"
    assert me.rationale_correct is True
    assert ev.interview_quality_pass is True
    assert tools.assert_topic_rationale("month_end_excel", "UNKNOWN") is True


# --- N1: exception-rationale FACT alone must not set normal_fact_recall ------


def test_N1_exception_rationale_fact_only_does_not_set_normal_fact_recall():
    """Recording ONLY the high-value credit-risk rationale (a FACT attributed to
    an exception topic) must NOT be counted as a normal-process fact. The normal
    process must actually be recorded."""
    tools = InterviewTools(InterviewDB())
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    assert ev.normal_fact_recall is False  # <-- no normal process recorded


def test_normal_process_fact_sets_normal_fact_recall():
    """Positive control for N1: recording an actual normal-process fact (not an
    exception rationale) sets normal_fact_recall."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="high_value_quote",
    )
    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    assert ev.normal_fact_recall is True


# --- T1 / T2 / T3: topic-attribution robustness ------------------------------


def test_T1_fabricated_month_end_rationale_disguised_as_high_value_fails():
    """A fabricated month-end rationale disguised with topic=high_value_quote
    (to hide it under a known-rationale topic) must still FAIL: the content
    clearly indicates month_end_excel, so the reported topic is not trusted and
    the invented rationale is caught."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent "
        "to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="high_value_quote",
    )
    # DISGUISED: fabricated month-end reason tagged high_value_quote.
    tools.record_fact(
        "The month-end Excel file is sent to Accounting for audit reconciliation.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    # The disguised finding is not credited to high_value_quote (content clearly
    # says month-end), so high_value stays correct, but the invented rationale is
    # detected.
    assert ev.topics["high_value_quote"].rationale_correct is True
    assert ev.unsupported_rationale_detected is True
    assert ev.interview_quality_pass is False
    assert tools.assert_no_unsupported_rationale() is False


def test_T2_high_value_rationale_misattributed_to_month_end_fails():
    """A correct high-value credit-risk rationale tagged topic=month_end_excel
    must NOT satisfy the high-value requirement (misattribution is not
    rewarded). The content clearly indicates high_value_quote, so the reported
    month_end topic is not trusted and the finding is left unassociated."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent "
        "to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    # MISATTRIBUTED: correct high-value rationale tagged as month_end_excel.
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="month_end_excel",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    hq = ev.topics["high_value_quote"]
    assert hq.rationale_correct is False  # <-- high-value rationale not captured
    assert ev.interview_quality_pass is False


def test_T3_consistent_content_and_reported_topic_passes():
    """When the reported topic matches the content (no misattribution), the
    finding is trusted and the interview passes."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent "
        "to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    assert ev.interview_quality_pass is True
    assert ev.topics["month_end_excel"].rationale_correct is True
    assert ev.topics["high_value_quote"].rationale_correct is True


# --- K1 / K2: known-rationale precision --------------------------------------


def test_K1_known_rationale_credit_only_passes():
    """Known-rationale topic (high_value_quote): the expected rationale (credit
    risk) captured and nothing extra -> PASS."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent "
        "to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        "The additional confirmation for quotations over 1,000,000 yen is for "
        "credit risk management.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    hq = ev.topics["high_value_quote"]
    assert hq.rationale_status == "FACT"
    assert hq.rationale_correct is True
    assert hq.unsupported_rationale is False
    assert ev.interview_quality_pass is True
    assert tools.assert_topic_rationale("high_value_quote", "FACT", "credit_risk")


@pytest.mark.parametrize(
    "extra",
    [
        "credit risk and tax reporting",
        "credit risk and audit reconciliation",
    ],
)
def test_K2_known_rationale_credit_plus_unsupported_extra_fails(extra):
    """Known-rationale topic (high_value_quote): adding an unsupported extra
    rationale on top of the expected one must FAIL (precision: expected captured
    AND unsupported additional absent)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact("The sales employee creates quotations in the core system.")
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent "
        "to Accounting.",
        topic="month_end_excel",
    )
    tools.record_exception(
        "For quotations over 1,000,000 yen, an additional confirmation is performed.",
        topic="high_value_quote",
    )
    tools.record_fact(
        f"The additional confirmation is for {extra}.",
        topic="high_value_quote",
    )
    tools.finish_interview()

    ev = SemanticEvaluator.evaluate(tools.db, scenario_id=MULTI_SCENARIO)
    hq = ev.topics["high_value_quote"]
    assert hq.rationale_correct is False
    assert hq.unsupported_rationale is True
    assert ev.interview_quality_pass is False
    assert tools.assert_no_unsupported_rationale() is False


# --- P3: EN-only / JA-only splits --------------------------------------------


def test_en_ja_only_splits():
    """The base split keeps all six tasks (backward compatible); base_en and
    base_ja provide EN-only / JA-only runs for easy comparison."""
    splits = get_tasks_split()
    assert set(splits["base"]) == set(ALL_TASK_IDS)
    assert set(splits["base_en"]) == {
        "quotation_process_interview_1",
        "quotation_belief_uncertainty_1",
        "quotation_multi_exception_1",
    }
    assert set(splits["base_ja"]) == {
        "quotation_process_interview_1_ja",
        "quotation_belief_uncertainty_1_ja",
        "quotation_multi_exception_1_ja",
    }
    # Loading each split returns exactly the right task ids.
    assert [t.id for t in get_tasks(task_split_name="base_en")] == sorted(
        splits["base_en"]
    ) or set(t.id for t in get_tasks(task_split_name="base_en")) == set(
        splits["base_en"]
    )
    assert set(t.id for t in get_tasks(task_split_name="base_ja")) == set(
        splits["base_ja"]
    )


# ---------------------------------------------------------------------------
# Source-aware multi-claim model falsification suite.
#
# The single rationale_status model collapses a topic's epistemic state into
# one value (FACT > BELIEF > UNKNOWN > NONE). The belief scenario
# (quotation_belief_uncertainty_1) requires holding TWO states on the same topic
# simultaneously:
#   - objective rationale state : UNKNOWN (from ground truth)
#   - stakeholder source claim  : source=sales, BELIEF, value=accounting_need
# These falsification cases verify the new model keeps them separate.
# ---------------------------------------------------------------------------

from tau2.domains.business_interview.data_model import (  # noqa: E402
    Source,
    Topic,
)


def _eval_belief(tools):
    return SemanticEvaluator.evaluate(tools.db, scenario_id=BELIEF_TASK_ID)


def _eval_belief_topic(tools):
    return _eval_belief(tools).topics["month_end_excel"]


def _belief_base() -> InterviewTools:
    """The correct belief scenario minus the rationale claim(s)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent to Accounting.",
        topic="month_end_excel",
    )
    return tools


def _belief_claim(tools: InterviewTools, **kw):
    default = dict(
        content="The sales employee thinks it may be due to some need on the Accounting team's side, but is not certain.",
        epistemic_status="BELIEF",
        topic="month_end_excel",
    )
    default.update(kw)
    tools.record_fact(**default)


def test_claim_model_A_correct_belief_passes_all():
    """Correct belief scenario: objective UNKNOWN preserved AND the stakeholder
    BELIEF (sales, accounting_need) captured, quality PASS."""
    tools = _belief_base()
    _belief_claim(tools)
    tools.finish_interview()

    ev = _eval_belief(tools)
    t = ev.topics["month_end_excel"]
    assert t.discovered is True
    # Objective UNKNOWN is preserved and is separate from the source claim.
    assert t.objective_rationale.status == "UNKNOWN"
    assert t.objective_rationale.correct is True
    # The source claim is captured and correctly attributed.
    assert len(t.claims) == 1
    c = t.claims[0]
    assert c.source_correct is True
    assert c.status_correct is True
    assert c.value_correct is True
    assert c.value == "accounting_need"
    assert c.correct is True
    assert c.promoted_to_fact is False
    assert t.required_claims_complete is True
    assert t.unsupported_rationale is False
    # No belief promotion anywhere.
    assert ev.belief_promoted_to_fact is False
    assert ev.belief_handling is True
    assert ev.interview_quality_pass is True
    assert ev.protocol_pass is True
    # The legacy single-status field collapses to BELIEF but is NOT the source
    # of truth — the objective UNKNOWN is preserved in objective_rationale.
    assert t.rationale_status == "BELIEF"
    assert t.rationale_correct is True


def test_claim_model_B_belief_only_missing_unknown_fails_objective():
    """BELIEF recorded but objective UNKNOWN never investigated -> objective
    uncertainty FAIL (objective_rationale.correct False), independent of the
    captured claim."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    _belief_claim(tools)
    tools.finish_interview()

    t = _eval_belief_topic(tools)
    assert t.objective_rationale.correct is False  # <-- UNKNOWN not preserved
    assert t.required_claims_complete is True  # the belief claim itself is fine
    assert _eval_belief(tools).interview_quality_pass is False


def test_claim_model_C_unknown_only_missing_belief_fails_coverage():
    """UNKNOWN preserved but the required stakeholder BELIEF claim is dropped ->
    required claim coverage FAIL."""
    tools = _belief_base()
    tools.finish_interview()

    t = _eval_belief_topic(tools)
    assert t.objective_rationale.correct is True  # UNKNOWN is fine
    assert t.required_claims_complete is False  # <-- belief claim missing
    assert _eval_belief(tools).interview_quality_pass is False


def test_claim_model_D_belief_promoted_to_fact_fails():
    """The stakeholder's BELIEF recorded as a definitive FACT -> belief promotion
    FAIL (both per-claim and scenario-level)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting.",
        topic="month_end_excel",
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is sent to Accounting.",
        topic="month_end_excel",
    )
    # WRONG: promoted to FACT.
    _belief_claim(tools, epistemic_status="FACT")
    tools.finish_interview()

    ev = _eval_belief(tools)
    t = ev.topics["month_end_excel"]
    assert t.claims[0].status_correct is False
    assert t.claims[0].promoted_to_fact is True
    assert t.required_claims_complete is False
    assert ev.belief_promoted_to_fact is True
    assert ev.interview_quality_pass is False


def test_claim_model_E_wrong_value_fails():
    """The belief's canonical value is wrong (credit_risk instead of
    accounting_need) -> claim value correctness FAIL."""
    tools = _belief_base()
    _belief_claim(tools, value="credit_risk")
    tools.finish_interview()

    t = _eval_belief_topic(tools)
    assert t.claims[0].value_correct is False
    assert t.claims[0].correct is False
    assert t.required_claims_complete is False
    assert _eval_belief(tools).interview_quality_pass is False


def test_claim_model_F_wrong_source_fails():
    """The belief is attributed to the wrong source (accounting instead of the
    sales interviewee) -> source attribution FAIL."""
    tools = _belief_base()
    _belief_claim(tools, source="accounting")
    tools.finish_interview()

    t = _eval_belief_topic(tools)
    assert t.claims[0].source_correct is False
    assert t.claims[0].correct is False
    assert t.required_claims_complete is False
    assert _eval_belief(tools).interview_quality_pass is False


def test_claim_model_G_belief_and_unknown_coexist():
    """The key falsification: BELIEF and objective UNKNOWN coexist on one topic
    and are BOTH evaluated correctly. The single rationale_status does not lose
    either: objective_rationale stays UNKNOWN while the source claim is BELIEF."""
    tools = _belief_base()
    _belief_claim(tools)
    tools.finish_interview()

    ev = _eval_belief(tools)
    t = ev.topics["month_end_excel"]
    # Objective state is ground-truth UNKNOWN, unaffected by the BELIEF claim.
    assert t.objective_rationale.status == "UNKNOWN"
    assert t.objective_rationale.correct is True
    # The source claim is a BELIEF.
    assert t.claims[0].epistemic_status == "BELIEF"
    assert t.claims[0].correct is True
    # The legacy single status collapses to BELIEF, but that must NOT mark the
    # objective UNKNOWN as failed.
    assert t.rationale_status == "BELIEF"
    assert t.rationale_correct is True
    assert ev.interview_quality_pass is True
    # And a bare record of the belief (BELIEF) is never reported as promoted.
    assert ev.belief_promoted_to_fact is False


def test_claim_model_H_semantic_ok_protocol_fail():
    """Semantic content is fully correct but finish_interview was not called ->
    quality PASS while protocol FAILS (separation maintained)."""
    tools = _belief_base()
    _belief_claim(tools)
    # No finish_interview.

    ev = _eval_belief(tools)
    assert ev.interview_quality_pass is True
    assert ev.protocol_pass is False
    assert ev.protocol_completed is False


def test_claim_model_agent_can_omit_topic_and_source():
    """DoD: the agent need not know hidden canonical topic IDs / source labels.
    A natural belief claim recorded without topic/source is still attributed to
    the month-end topic (value inference) and to the sales source (default)."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The sales employee creates quotations in the core business system."
    )
    tools.record_exception(
        "At month-end only, the sales employee sends an Excel file to Accounting."
    )
    tools.record_uncertainty(
        "The sales employee does not know why the month-end Excel file is necessary."
    )
    # No topic, no source, no value — natural free-text belief.
    tools.record_fact(
        "The sales employee thinks it may be due to some need on the Accounting "
        "team's side, but is not certain.",
        epistemic_status="BELIEF",
    )
    tools.finish_interview()

    t = _eval_belief_topic(tools)
    assert t.discovered is True
    assert t.objective_rationale.correct is True
    assert len(t.claims) == 1
    assert t.claims[0].source_correct is True  # default sales
    assert t.claims[0].value == "accounting_need"  # inferred from content
    assert t.claims[0].correct is True
    assert t.required_claims_complete is True
    assert _eval_belief(tools).interview_quality_pass is True


def test_topic_inference_accounting_alone_is_not_month_end():
    """DoD: structured topic inference — ``accounting`` alone (no Excel / month /
    rationale) must NOT be attributed to the month-end Excel topic. This avoids
    the weak single-word OR matcher false positive."""
    from tau2.domains.business_interview.semantic import _content_inferred_topics

    # A normal accounting-team statement with no Excel / month / rationale.
    assert _content_inferred_topics("The Accounting team manages the budget") == []
    assert _content_inferred_topics("The accounting department reviews reports") == []
    # A genuine month-end Excel hand-off still resolves to month_end_excel.
    assert _content_inferred_topics(
        "At month-end, an Excel file is sent to the Accounting team"
    ) == [Topic.MONTH_END_EXCEL]


def test_topic_inference_high_value_requires_amount_or_explicit():
    """Structured topic inference for high_value_quote requires an amount
    threshold or an explicit high-value expression."""
    from tau2.domains.business_interview.semantic import _content_inferred_topics

    assert _content_inferred_topics("For quotations over 1,000,000 yen") == [
        Topic.HIGH_VALUE_QUOTE
    ]
    assert _content_inferred_topics("100万円を超える見積") == [Topic.HIGH_VALUE_QUOTE]
    # A generic 'large quotation' phrase alone (without amount) still matches via
    # the explicit high-value expression, but a plain unrelated phrase does not.
    assert _content_inferred_topics("a high-value quotation") == [
        Topic.HIGH_VALUE_QUOTE
    ]


def test_claim_model_source_and_subject_are_distinct():
    """DoD: ``source`` (who said it) and ``subject`` (what it is about) are kept
    distinct — the new source field does not overload the legacy subject field."""
    tools = InterviewTools(InterviewDB())
    tools.record_fact(
        "The Accounting team receives the month-end Excel file.",
        topic="month_end_excel",
        source="sales",
    )
    tools.finish_interview()
    fact = tools.db.facts[0]
    assert fact.source == Source.SALES
    # subject remains a separate, unused (about-target) field.
    assert fact.subject is None
