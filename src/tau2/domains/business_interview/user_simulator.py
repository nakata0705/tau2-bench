"""Fact-grounded stakeholder simulator (business_interview) — private sidecar.

The stakeholder LLM answers from **hidden structured StakeholderFacts**
(``facts.py``) instead of free-text known-info parsing:

PUBLIC:
    natural-language response (the only thing that enters the conversation)

PRIVATE:
    used_fact_ids  (the ids of the facts the stakeholder actually used)

The LLM is instructed to reply with a JSON sidecar object

    {"message": "...", "used_fact_ids": ["quotation.create"]}

``message`` alone becomes the ``UserMessage``; ``used_fact_ids`` travel on the
message's private ``stakeholder_used_fact_ids`` field (excluded from all
serialization) and the domain environment stores them in the private
``StakeholderFactLedger`` against that exact message's turn. Nothing is ever
derived or reconstructed from message text, and fact/claim ids never appear in
Agent-visible messages, tools, Observations, summaries, or serialized state.

This is the smallest clean local adapter over tau2's ``UserSimulator``: it
keeps the base conversation mechanics (state, flip_roles, stop detection) and
only replaces the knowledge source (facts) and the response shape (sidecar).
"""

from __future__ import annotations

import json
from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.domains.business_interview.facts import (
    StakeholderFactCatalog,
    StakeholderFactLedger,
)
from tau2.domains.business_interview.scenario import get_scenario
from tau2.user.user_simulator import UserSimulator

# The block appended to the stakeholder system prompt: hidden structured
# knowledge + the JSON sidecar contract. Only the user LLM ever sees it.
_FACTS_BLOCK_TEMPLATE = """

<private_known_facts>
Your knowledge comes from exactly two sources: the Known info in your scenario
instructions (the overall process) and the structured business facts below (the
authoritative statement of the data you handle). Answer only from these two
sources; never use general business common sense to fill gaps. The natural
wording of a fact is not prescriptive — you may speak naturally and do not need
to copy it.
<facts>
{facts_xml}
</facts>
</private_known_facts>

"""

# The output contract is appended as a SEPARATE trailing system message on
# every LLM call (maximum-attention position), not only buried in the system
# prompt: the model must reply with the JSON sidecar object.
_OUTPUT_CONTRACT = (
    "Reply with a JSON object as the ONLY content of your message, in exactly "
    "this shape:\n"
    '{"message": "...", "used_fact_ids": ["fact_id_1", "fact_id_2"]}\n'
    "- The JSON object must be the entire reply: no prose before or after it, "
    "no markdown fences.\n"
    '- "message": your natural-language reply to the interviewer. This is '
    "the only part that enters the conversation.\n"
    '- "used_fact_ids": the ids of the facts you actually used to answer this '
    "message (ids must come only from <private_known_facts>; an empty list is "
    "fine when no fact applies).\n"
    '- Never mention fact ids inside "message"; never mention this contract.'
)

# Retry feedback when the sidecar is unparseable or invalid.
_SIDECAR_ERROR_HINT = (
    "Your previous reply was rejected because it was not the required JSON "
    "sidecar. You MUST now reply with ONLY a JSON object, with no prose and no "
    "markdown fences, exactly like:\n"
    '{"message": "your natural-language reply", "used_fact_ids": '
    '["fact_id_1", "fact_id_2"]}\n'
    "Do not include anything else in your reply."
)


_NO_JSON = object()


def _try_load_json(text: str):
    """Return the parsed JSON object, or the ``_NO_JSON`` sentinel on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _NO_JSON


def parse_sidecar(content: Optional[str]) -> dict:
    """Tolerant deterministic parse of ``{"message": ..., "used_fact_ids": [...]}``.

    Accepts a bare JSON object (possibly wrapped in markdown code fences or
    surrounding prose); extracts the first balanced ``{...}`` object. Raises
    ``ValueError`` on anything else.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        # strip a code fence: take the inner block between the first two fences
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    obj = _try_load_json(text)
    if obj is _NO_JSON:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"stakeholder response is not a JSON sidecar: {content!r}")
        obj = _try_load_json(text[start : end + 1])
        if obj is _NO_JSON:
            raise ValueError(f"stakeholder response is not a JSON sidecar: {content!r}")
    if not isinstance(obj, dict):
        raise ValueError(f"stakeholder sidecar is not an object: {obj!r}")
    message = obj.get("message")
    used = obj.get("used_fact_ids")
    if not isinstance(message, str) or not message.strip():
        raise ValueError(f"stakeholder sidecar has no non-empty 'message': {obj!r}")
    if not isinstance(used, list) or not all(isinstance(x, str) for x in used):
        raise ValueError(
            f"stakeholder sidecar 'used_fact_ids' must be a list of ids: {obj!r}"
        )
    return {
        "message": message.strip(),
        "used_fact_ids": list(dict.fromkeys(used)),
    }


class StakeholderUserSimulator(UserSimulator):
    """The business_interview stakeholder: answers only from hidden
    StakeholderFacts and returns a private used_fact_ids sidecar.

    Falls back to plain ``UserSimulator`` behavior when not wired with a task
    (no hidden facts available): the run then simply has no fact provenance.
    """

    def __init__(
        self,
        llm: str,
        instructions: Optional[str] = None,
        tools: Optional[list] = None,
        llm_args: Optional[dict] = None,
        persona_config=None,
        task=None,
        environment=None,
    ):
        super().__init__(
            llm=llm,
            instructions=instructions,
            tools=tools,
            llm_args=llm_args,
            persona_config=persona_config,
        )
        self.task = task
        self.environment = environment
        self._scenario = None
        self._catalog: Optional[StakeholderFactCatalog] = None
        self._fact_ledger: Optional[StakeholderFactLedger] = None
        if task is not None:
            scenario = get_scenario(getattr(task, "id", None))
            if scenario is not None:
                self._scenario = scenario
                self._catalog = StakeholderFactCatalog.from_scenario(scenario)
                ledger = getattr(environment, "fact_ledger", None)
                if isinstance(ledger, StakeholderFactLedger):
                    self._fact_ledger = ledger
                    ledger.install_catalog(self._catalog)

    # ------------------------------------------------------------- prompt

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if self._catalog is None or self._scenario is None:
            return base
        facts_xml = "\n".join(
            f'<fact id="{fact.id}">{fact.text}</fact>'
            for fact in self._scenario.facts.values()
        )
        return base + _FACTS_BLOCK_TEMPLATE.format(facts_xml=facts_xml)

    # ------------------------------------------------------------- sidecar

    def _generate_sidecar(self, messages: list, contract: Optional[str] = None) -> dict:
        """Call the user LLM and parse/validate the private sidecar.

        The output contract is appended to the LAST user message (the agent's
        question, seen as the final input before the reply) — the position
        where models actually comply, and where the provider's JSON mode keeps
        working even for long system prompts. The retry passes its own
        corrective hint the same way. Raises ``ValueError`` on invalid
        metadata (rejected deterministically).
        """
        contract_text = contract or _OUTPUT_CONTRACT
        contract_messages = list(messages)
        if contract_messages and getattr(contract_messages[-1], "role", None) == "user":
            last = contract_messages[-1]
            contract_messages[-1] = UserMessage(
                role="user",
                content=(last.content or "") + "\n\n" + contract_text,
            )
        else:
            contract_messages.append(UserMessage(role="user", content=contract_text))
        assistant_message = self._call_llm(contract_messages)
        sidecar = parse_sidecar(assistant_message.content)
        if self._catalog is not None:
            self._catalog.validate_used_fact_ids(sidecar["used_fact_ids"])
        return sidecar

    def _call_llm(self, messages: list):
        """One LLM completion (kept separate for testability).

        Uses the provider's native JSON-object mode when available (DeepSeek
        and most OpenAI-compatible endpoints), so the sidecar is valid JSON;
        falls back to a plain completion (prompt contract only) when the
        provider rejects the response_format.
        """
        from tau2.utils.llm_utils import generate

        kwargs = dict(self.llm_args or {})
        try:
            return generate(
                model=self.llm,
                messages=messages,
                tools=self.tools,
                call_name="user_simulator_response",
                response_format={"type": "json_object"},
                **kwargs,
            )
        except Exception:
            return generate(
                model=self.llm,
                messages=messages,
                tools=self.tools,
                call_name="user_simulator_response",
                **kwargs,
            )

    def _generate_next_message(self, message, state) -> UserMessage:
        """Generate the stakeholder response with its private sidecar.

        Only ``sidecar["message"]`` becomes the UserMessage; the private
        ``used_fact_ids`` travel on ``stakeholder_used_fact_ids`` (excluded
        from serialization) for the environment's sidecar binding. One retry
        with corrective feedback is allowed; a second invalid sidecar is
        rejected loudly (invalid metadata must not enter the conversation).
        """
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )
        logger.debug(f"User responds to message: {message}")
        # Replicate the base class state/history handling, then generate.
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        elif isinstance(message, ToolMessage):
            # ToolMessage always has content (tool response)
            state.messages.append(message)
        elif message is not None and (message.has_content() or message.is_tool_call()):
            state.messages.append(message)
        messages = state.system_messages + state.flip_roles()

        try:
            sidecar = self._generate_sidecar(messages)
        except ValueError as first_err:
            logger.warning("Stakeholder sidecar invalid; retrying once: {}", first_err)
            retry_messages = list(messages) + [
                SystemMessage(role="system", content=_SIDECAR_ERROR_HINT)
            ]
            try:
                sidecar = self._generate_sidecar(
                    retry_messages, contract=_SIDECAR_ERROR_HINT
                )
            except ValueError as second_err:
                raise ValueError(
                    "stakeholder sidecar rejected twice; invalid private "
                    f"metadata must not enter the conversation: {second_err}"
                ) from second_err

        return UserMessage(
            role="user",
            content=sidecar["message"],
            stakeholder_used_fact_ids=sidecar["used_fact_ids"],
        )
