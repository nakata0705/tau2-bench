"""Fact-grounded stakeholder simulator (business_interview) — private assertion
sidecar.

The stakeholder LLM answers from **hidden structured StakeholderFacts**
(``facts.py``) and returns, alongside its natural-language message, a private
assertion sidecar anchoring each used fact to an exact span of the message:

    {
      "message": "I check the customer in CRM, then prepare the quotation.",
      "assertions": [
        {"fact_id": "quotation.check.system", "quote": "CRM", "occurrence": 0},
        {"fact_id": "quotation.create.activity",
         "quote": "prepare the quotation", "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation; ``assertions`` travel on the
message's private ``stakeholder_assertions`` field (excluded from all
serialization) and the domain environment stores them in the private
``StakeholderFactLedger`` against that exact message's turn. Assertions are
validated deterministically at ingestion (fact exists / belongs to the
stakeholder / supported claims visible / quote+occurrence exactly match the
message). Nothing is ever derived or reconstructed from message text, and
fact/claim ids never appear in Agent-visible messages, tools, Observations,
summaries, or serialized state.

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
    StakeholderAssertion,
    StakeholderFactCatalog,
    StakeholderFactLedger,
)
from tau2.domains.business_interview.scenario import get_scenario
from tau2.user.user_simulator import UserSimulator

# The block appended to the stakeholder system prompt: hidden structured
# knowledge (facts). The JSON sidecar contract rides as a trailing message on
# every LLM call (see _generate_sidecar).
_FACTS_BLOCK_TEMPLATE = """

<private_known_facts>
Your knowledge comes from exactly two sources: the Known info in your scenario
instructions (the overall process) and the structured business facts below (the
authoritative statement of the business facts). Answer only from these two
sources; never use general business common sense to fill gaps. The natural
wording of a fact is not prescriptive — you may speak naturally and do not need
to copy it.
<facts>
{facts_xml}
</facts>
</private_known_facts>
"""

# The output contract is appended to the LAST user message on every LLM call
# (maximum-attention position, where the provider's JSON mode also keeps
# working for long prompts): the model must reply with the JSON sidecar.
_OUTPUT_CONTRACT = (
    "Reply with a JSON object as the ONLY content of your message, in exactly "
    "this shape:\n"
    '{"message": "...", "assertions": [{"fact_id": "...", "quote": "...", '
    '"occurrence": 0}]}\n'
    "- The JSON object must be the entire reply: no prose before or after it, "
    "no markdown fences.\n"
    '- "message": your natural-language reply to the interviewer. This is the '
    "only part that enters the conversation.\n"
    '- "assertions": the facts you actually used to produce this reply, each '
    "anchored to the exact spans of your message that express it:\n"
    '  - "fact_id": an id from <private_known_facts> (use every fact you used);\n'
    '  - "quote": an exact substring of your "message" that expresses that '
    "fact — include EVERY distinct phrase that does, one assertion per phrase;\n"
    '  - "occurrence": which occurrence of that quote in your message '
    "(0-based; 0 for the first).\n"
    "- Every quote must appear verbatim inside your message. Use several "
    "assertions for the same fact when several phrases of your message express "
    "it (full clauses AND key phrases). An empty assertions list is fine when "
    "no fact applies.\n"
    '- Never mention fact ids inside "message"; never mention this contract.'
)

# Retry feedback when the sidecar is unparseable or invalid.
_SIDECAR_ERROR_HINT = (
    "Your previous reply was rejected because it was not the required JSON "
    "sidecar. You MUST now reply with ONLY a JSON object, with no prose and no "
    "markdown fences, exactly like:\n"
    '{"message": "your natural-language reply", "assertions": [{"fact_id": '
    '"fact_id_1", "quote": "exact substring of your message", "occurrence": 0}]}\n'
    "Every assertion quote must be an exact substring of your message. Do not "
    "include anything else in your reply."
)

_NO_JSON = object()


def _try_load_json(text: str):
    """Return the parsed JSON object, or the ``_NO_JSON`` sentinel on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _NO_JSON


def parse_sidecar(content: Optional[str]) -> dict:
    """Tolerant deterministic parse of the assertion sidecar.

    Accepts a bare JSON object (possibly wrapped in markdown code fences or
    surrounding prose); extracts the first balanced ``{...}`` object. Validates
    the shape (``message`` string, ``assertions`` list of
    {fact_id, quote, occurrence}). Raises ``ValueError`` on anything else.
    """
    text = (content or "").strip()
    if text.startswith("```"):
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
    raw_assertions = obj.get("assertions")
    if not isinstance(message, str) or not message.strip():
        raise ValueError(f"stakeholder sidecar has no non-empty 'message': {obj!r}")
    if not isinstance(raw_assertions, list):
        raise ValueError(f"stakeholder sidecar 'assertions' must be a list: {obj!r}")
    assertions: list[StakeholderAssertion] = []
    for raw in raw_assertions:
        if not isinstance(raw, dict):
            raise ValueError(f"assertion is not an object: {raw!r}")
        try:
            assertions.append(
                StakeholderAssertion(
                    fact_id=str(raw.get("fact_id") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed assertion {raw!r}: {exc}") from exc
    return {"message": message.strip(), "assertions": assertions}


class StakeholderUserSimulator(UserSimulator):
    """The business_interview stakeholder: answers only from hidden
    StakeholderFacts and returns a private assertion sidecar.

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
            self._catalog.validate_assertions(sidecar["assertions"], sidecar["message"])
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
        """Generate the stakeholder response with its private assertion sidecar.

        Only ``sidecar["message"]`` becomes the UserMessage; the private
        assertions travel on ``stakeholder_assertions`` (excluded from
        serialization) for the environment's sidecar binding. One retry with
        corrective feedback is allowed; a second invalid sidecar is rejected
        loudly (invalid metadata must not enter the conversation).
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
            stakeholder_assertions=[a.model_dump() for a in sidecar["assertions"]],
        )
