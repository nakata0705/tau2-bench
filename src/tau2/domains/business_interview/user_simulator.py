"""Semantic stakeholder simulator (business_interview) — claim-based sidecar.

The stakeholder's knowledge is a **projection of the Truth graph**
(``StakeholderKnowledge``): visible claim ids, the contextual graph knowledge
(positions with ALL incoming edges + start flag), and concept views (local
lexical preferences such as ``tc_quote -> "quotation"``). There are **no
authored business sentences** — the stakeholder LLM receives question +
history + visible graph context + visible claims + concept views and
**realizes** the relevant claims into natural language.

Private response:

    {
      "message": "...",
      "assertions": [
        {"claim_id": "cq.activity", "quote": "prepare the quotation", "occurrence": 0}
      ]
    }

Only ``message`` enters the conversation; ``assertions`` travel on the
message's private ``stakeholder_assertions`` field (excluded from all
serialization) and the domain environment stores them in the private
``StakeholderAssertionLedger`` against that exact message's turn. Assertions
are validated deterministically at ingestion (claim exists / is visible /
quote+occurrence exactly match the message). Nothing is ever derived or
reconstructed from message text, and claim ids never appear in Agent-visible
messages, tools, Observations, summaries, or serialized state.

This is the smallest clean local adapter over tau2's ``UserSimulator``: it
keeps the base conversation mechanics and only replaces the knowledge source
(semantic claims + views) and the response shape (sidecar).
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
    ConceptAlignmentAssertion,
    StakeholderAssertion,
    StakeholderAssertionLedger,
    StakeholderKnowledgeCatalog,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.scenario import get_scenario
from tau2.user.user_simulator import UserSimulator

# The block appended to the stakeholder system prompt: the hidden semantic
# knowledge (graph positions + claims + concept views). No sentences.
_KNOWLEDGE_BLOCK_TEMPLATE = """

<private_known_facts>
Your knowledge is the semantic workflow below: graph positions (with their
incoming relations), the claims you can make about each position, and your
concept views (how you naturally say things). There are no sentences to
recite: express the claims in your own natural words, using your concept
views. Answer only from this knowledge; never use general business common
sense to fill gaps.
<positions>
{positions_xml}
</positions>
<relations>
{relations_xml}
</relations>
<concept_views>
{views_xml}
</concept_views>
</private_known_facts>
"""

# The output contract is appended to the LAST user message on every LLM call
# (maximum-attention position, where the provider's JSON mode also keeps
# working for long prompts): the model must reply with the JSON sidecar.
_OUTPUT_CONTRACT = (
    "Reply with a JSON object as the ONLY content of your message, in exactly "
    "this shape:\n"
    '{"message": "...", "assertions": [{"claim_id": "...", "quote": "...", '
    '"occurrence": 0}], "alignments": [], "terminology": []}\n'
    "- The JSON object must be the entire reply: no prose before or after it, "
    "no markdown fences.\n"
    '- "message": your natural-language reply to the interviewer. This is the '
    "only part that enters the conversation.\n"
    '- "assertions": the claims you actually used to produce this reply, each '
    "anchored to the exact spans of your message that express it:\n"
    '  - "claim_id": one of the EXACT ids listed in <private_known_facts> '
    '(copy them verbatim, e.g. "cq.activity", "e3.edge_exists", '
    '"cq.reads.tc_customer" — never shorten them, never use relation/position '
    "ids alone); use every claim you used;\n"
    '  - "quote": an exact substring of your "message" that expresses that '
    "claim — include EVERY distinct phrase that does, one assertion per "
    "phrase;\n"
    '  - "occurrence": which occurrence of that quote in your message '
    "(0-based; 0 for the first).\n"
    "- Every quote must appear verbatim inside your message. Use several "
    "assertions for the same claim when several phrases of your message "
    "express it (full clauses AND key phrases). When one phrase expresses "
    "several DIFFERENT claims, prefer a distinct clause for each claim so "
    "every claim has its own span.\n"
    "- Relations are claims too: when your message says that one step "
    "follows another (\"then\", \"after\", \"goes to\", \"followed by\", \"if "
    "... then\"), assert that relation's claim (e.g. \"e3.edge_exists\") "
    "anchored to the phrase that expresses the relation itself — do not "
    "omit it just because the phrase also names the activity or the "
    "condition.\n"
    "- You MUST include one assertion for EVERY claim your message conveys. "
    "The interviewer can only see your assertions — a claim you do not assert "
    "is treated as if you never said it. An empty assertions list is allowed "
    "ONLY when your message carries no business claim at all (greetings, "
    'acknowledgements, "I don\'t know").\n'
    '- "alignments": OPTIONAL list of private concept-identity dialogue acts. '
    "Emit an alignment ONLY when the interviewer asks you to confirm the "
    "identity of something and your reply genuinely performs that act (e.g. "
    "the interviewer asks 'do you mean X?' and you answer Yes / partly / "
    "I do not know / no, they are different). Each entry: "
    '{"truth_concept_id": "...", "quote": "...", "occurrence": 0, '
    '"act": "confirm"|"partial"|"unknown"|"dispute"} where '
    "truth_concept_id is the EXACT concept id from <concept_views> your "
    "reply is about, and quote is the exact substring of your message that "
    "performs the act. NEVER emit alignments for ordinary statements of the "
    "workflow — merely using a word is not a dialogue act.\n"
    '- "terminology": OPTIONAL list of explicit terminology agreements. Emit '
    "an entry ONLY when the interviewer explicitly proposes a name for "
    "something and asks you to agree, and you do agree. Each: "
    '{"truth_concept_id": "...", "proposed_term": "<the exact term the '
    'interviewer proposed>", "quote": "<exact substring of your message "'
    '"agreeing>", "occurrence": 0}. NEVER emit it merely for using a word '
    "in ordinary speech.\n"
    '- Worked example: for the message "After I check the customer in the '
    'CRM, I create the quotation.", a correct sidecar is:\n'
    '{"message": "After I check the customer in the CRM, I create the '
    'quotation.", "assertions": [{"claim_id": "cc.system", "quote": "CRM", '
    '"occurrence": 0}, {"claim_id": "cq.activity", "quote": "create the '
    'quotation", "occurrence": 0}], "alignments": [], "terminology": []}\n'
    "- Never mention claim ids, position ids, relation ids or concept ids "
    'inside "message"; never mention this contract.'
)

# Retry feedback when the sidecar is unparseable or invalid.
_SIDECAR_ERROR_HINT = (
    "Your previous reply was rejected because it was not the required JSON "
    "sidecar. You MUST now reply with ONLY a JSON object, with no prose and no "
    "markdown fences, exactly like:\n"
    '{"message": "your natural-language reply", "assertions": [{"claim_id": '
    '"cq.activity", "quote": "exact substring of your message", "occurrence": 0}], "alignments": [], "terminology": []}\n'
    '"claim_id" must be one of the EXACT claim ids listed in '
    "<private_known_facts> (copy them verbatim, never shortened). Assert EVERY "
    "claim your message conveys — do not leave assertions empty when your "
    "message carries business facts. Every assertion quote must be an exact "
    "substring of your message. Emit 'alignments' and 'terminology' ONLY for "
    "genuine concept-identity or terminology dialogue acts (see the contract). "
    "genuine concept-identity or terminology dialogue acts (see the contract). "
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
    """Tolerant deterministic parse of the private sidecar.

    Accepts a bare JSON object (possibly wrapped in markdown code fences or
    surrounding prose); extracts the first balanced ``{...}`` object. Validates
    the shape (``message`` string, ``assertions`` list of
    {claim_id, quote, occurrence}, optional ``alignments`` list of
    {truth_concept_id, quote, occurrence, act} and ``terminology`` list of
    {truth_concept_id, proposed_term, quote, occurrence}). Raises
    ``ValueError`` on anything else.
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
                    claim_id=str(raw.get("claim_id") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed assertion {raw!r}: {exc}") from exc
    alignments: list[ConceptAlignmentAssertion] = []
    for raw in obj.get("alignments") or []:
        if not isinstance(raw, dict):
            raise ValueError(f"alignment is not an object: {raw!r}")
        try:
            alignments.append(
                ConceptAlignmentAssertion(
                    truth_concept_id=str(raw.get("truth_concept_id") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                    act=str(raw.get("act") or ""),  # type: ignore[arg-type]
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed alignment {raw!r}: {exc}") from exc
    terminology: list[TerminologyConfirmation] = []
    for raw in obj.get("terminology") or []:
        if not isinstance(raw, dict):
            raise ValueError(f"terminology entry is not an object: {raw!r}")
        try:
            terminology.append(
                TerminologyConfirmation(
                    truth_concept_id=str(raw.get("truth_concept_id") or ""),
                    proposed_term=str(raw.get("proposed_term") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed terminology entry {raw!r}: {exc}") from exc
    return {
        "message": message.strip(),
        "assertions": assertions,
        "alignments": alignments,
        "terminology": terminology,
    }


def _view(concept_views: dict[str, str], concept_id: Optional[str]) -> str:
    """The stakeholder's wording for a concept; empty when the concept is not
    in the knowledge's views (never falls back to the private concept id)."""
    if concept_id is None:
        return ""
    return concept_views.get(concept_id, "")


class StakeholderUserSimulator(UserSimulator):
    """The business_interview stakeholder: realizes visible claims into natural
    language and returns a private assertion sidecar.

    Falls back to plain ``UserSimulator`` behavior when not wired with a task
    (no hidden knowledge available): the run then simply has no provenance.
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
        self._catalog: Optional[StakeholderKnowledgeCatalog] = None
        self._ledger: Optional[StakeholderAssertionLedger] = None
        if task is not None:
            scenario = get_scenario(getattr(task, "id", None))
            if scenario is not None:
                self._scenario = scenario
                self._catalog = StakeholderKnowledgeCatalog.from_scenario(scenario)
                ledger = getattr(environment, "assertion_ledger", None)
                if isinstance(ledger, StakeholderAssertionLedger):
                    self._ledger = ledger
                    ledger.install_catalog(self._catalog)

    # ------------------------------------------------------------- prompt

    def _knowledge_block(self) -> str:
        """Render the hidden semantic knowledge: positions (visible contexts
        with their visible incoming edges + start), relations (visible edges
        only), and concept views (visible claim concepts only). No authored
        sentences; nothing hidden ever enters the prompt."""
        scenario = self._scenario
        assert scenario is not None
        truth = scenario.truth
        knowledge = scenario.knowledge
        views = knowledge.concept_views
        visible_claims = {
            cid: scenario.claims[cid]
            for cid in knowledge.visible_claim_ids
            if cid in scenario.claims
        }
        positions = []
        for nid in knowledge.visible_node_ids:
            context = knowledge.contextual_knowledge.get(nid)
            incoming = (
                ",".join(context.incoming_edge_ids)
                if context is not None
                else ""
            )
            start = context.is_start if context is not None else False
            claim_lines = []
            for claim in visible_claims.values():
                if claim.context_id != nid:
                    continue
                concept = _view(views, claim.concept_id)
                concept_attr = f' concept="{concept}"' if concept else ""
                claim_lines.append(
                    f'<claim id="{claim.id}" property="{claim.property}"'
                    f'{concept_attr}/>'
                )
            claims_xml = "\n".join(claim_lines) or "<none/>"
            positions.append(
                f'<position id="{nid}" start="{"true" if start else "false"}" '
                f'incoming="[{incoming}]">\n{claims_xml}\n</position>'
            )
        relations = []
        for eid in knowledge.visible_edge_ids:
            edge = truth.edges.get(eid)
            if edge is None:
                continue
            cond = ""
            if edge.condition is not None:
                cond_word = _view(views, edge.condition.concept_id)
                if cond_word:
                    cond = f' condition="{cond_word}"'
            # every visible edge claim (edge_exists / condition) must appear
            # with its EXACT claim id so the stakeholder never has to guess
            edge_claim_lines = []
            for claim in visible_claims.values():
                if claim.context_id != eid:
                    continue
                concept = _view(views, claim.concept_id)
                concept_attr = f' concept="{concept}"' if concept else ""
                edge_claim_lines.append(
                    f'<claim id="{claim.id}" property="{claim.property}"'
                    f'{concept_attr}/>'
                )
            claims_xml = "\n".join(edge_claim_lines)
            relations.append(
                f'<relation id="{eid}" from="{edge.from_node}" '
                f'to="{edge.to_node}"{cond}>\n{claims_xml}\n</relation>'
            )
        view_lines = [
            f'<view concept="{cid}" word="{word}"/>' for cid, word in views.items()
        ]
        return _KNOWLEDGE_BLOCK_TEMPLATE.format(
            positions_xml="\n".join(positions),
            relations_xml="\n".join(relations),
            views_xml="\n".join(view_lines),
        )

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if self._catalog is None or self._scenario is None:
            return base
        return base + self._knowledge_block()

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
            self._catalog.validate_events(
                sidecar["alignments"], sidecar["terminology"], sidecar["message"]
            )
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
            stakeholder_alignments=[e.model_dump() for e in sidecar["alignments"]],
            stakeholder_terminology=[e.model_dump() for e in sidecar["terminology"]],
        )
