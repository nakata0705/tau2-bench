"""Semantic stakeholder simulator (business_interview) — graph-native sidecar.

The stakeholder's knowledge is its **world model** (``StakeholderKnowledge``:
a masked graph with three-valued property slots + local concepts). There are
**no authored business sentences** — the stakeholder LLM receives question +
history + its world model (nodes/edges/property slots with their semantic
IDs, DONT_KNOW slots, and its local concepts with description/terms) and
**realizes** it into natural language.

Private response:

    {
      "message": "...",
      "annotations": [
        {"semantic_id": "node:skn_002:system", "mode": "value",
         "quote": "CRM", "occurrence": 0}
      ],
      "alignments": [],
      "terminology": []
    }

Each annotation carries a semantic ``mode`` (what the message asserts about
that element: value / absent / dont_know / exists / mention), validated
deterministically against the StakeholderKnowledgeGraph: an annotation whose
mode contradicts the stakeholder's own world model (e.g. the graph knows a
value but the reply annotates ``dont_know``) is REJECTED and retried.

Only ``message`` enters the conversation; annotations and dialogue events
travel on the message's private fields (excluded from all serialization) and
the domain environment stores them in the private ``SemanticLedger`` against
that exact message's turn. Metadata is validated deterministically at
ingestion (semantic id exists in the stakeholder knowledge / quote+occurrence
exactly match the message / mode matches the knowledge). Nothing is ever
derived or reconstructed from message text, and semantic ids never appear in
Agent-visible messages, tools, Observations, summaries, or serialized state.

The semantic ids are **opaque stakeholder-local ids** (``skn_001`` /
``ske_001`` / ``skc_001`` style): they carry no Truth meaning (no labels,
terms, node names or positions), so even speech about a concept whose
description/terms are DONT_KNOW cannot leak Truth facts through the ids.
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
    SemanticAnnotation,
    SemanticLedger,
    StakeholderKnowledgeCatalog,
    TerminologyConfirmation,
)
from tau2.domains.business_interview.graph import is_dont_know
from tau2.domains.business_interview.scenario import get_scenario
from tau2.user.user_simulator import UserSimulator

# The block appended to the stakeholder system prompt: the hidden semantic
# world model (graph elements with their semantic IDs + the local concepts).
# No sentences, no hidden Truth ids, no canonical Truth terminology.
_KNOWLEDGE_BLOCK_TEMPLATE = """

<private_known_facts>
Your knowledge is the semantic world model below: the process positions and
relations you know (with their semantic ids), the property values you know at
each position, and your own concepts (how you naturally describe and name
things). There are no sentences to recite: express what you know in your own
natural words, using your concepts. Answer only from this knowledge; never
use general business common sense to fill gaps.
<positions>
{positions_xml}
</positions>
<relations>
{relations_xml}
</relations>
<concepts>
{concepts_xml}
</concepts>
</private_known_facts>
"""

# The output contract is appended to the LAST user message on every LLM call
# (maximum-attention position): the model must reply with the JSON sidecar.
_OUTPUT_CONTRACT = (
    "Reply with a JSON object as the ONLY content of your message, in exactly "
    "this shape:\n"
    '{"message": "...", "annotations": [{"semantic_id": "...", "mode": "value", '
    '"quote": "...", "occurrence": 0}], "alignments": [], "terminology": []}\n'
    "- The JSON object must be the entire reply: no prose before or after it, "
    "no markdown fences.\n"
    '- "message": your natural-language reply to the interviewer. This is the '
    "only part that enters the conversation.\n"
    '- "annotations": the knowledge elements you actually used to produce this '
    "reply, each anchored to the exact spans of your message that express it:\n"
    '  - "semantic_id": one of the EXACT ids listed in <private_known_facts> '
    '(copy them verbatim, e.g. "node:skn_002:system", "edge:ske_003", '
    '"node:skn_002:reads:skc_004" — never shorten them); use every element '
    "you used;\n"
    '  - "mode": the kind of assertion your message makes about that element, '
    'as declared in <private_known_facts>: "value" when the slot shows a '
    'known value, "absent" when it shows absent, "dont_know" when it shows '
    'unknown, "exists" when you assert the element (a position or relation) '
    'exists, "mention" when the element is a concept you name or describe. '
    "A mode that contradicts the knowledge (e.g. dont_know on a slot that "
    "shows a value) makes your reply UNACCEPTABLE — always use the kind "
    "declared in your knowledge;\n"
    '  - "quote": an exact substring of your "message" that expresses that '
    "element — include EVERY distinct phrase that does, one annotation per "
    "phrase;\n"
    '  - "occurrence": which occurrence of that quote in your message '
    "(0-based; 0 for the first).\n"
    "- Every quote must appear verbatim inside your message. Use several "
    "annotations for the same element when several phrases express it. When "
    "one phrase expresses several DIFFERENT elements, prefer a distinct "
    "clause for each so every element has its own span.\n"
    "- You MUST include one annotation for EVERY element your message "
    "conveys. The interviewer can only see your annotations — an element you "
    "do not annotate is treated as if you never said it. An empty "
    "annotations list is allowed ONLY when your message carries no knowledge "
    'at all (greetings, acknowledgements, "I don\'t know" answers whose '
    "element is not in your knowledge).\n"
    "- Relations are knowledge too: when your message says that one step "
    'follows another ("then", "after", "goes to", "followed by", "if ... '
    'then"), annotate the relation with mode "exists" (e.g. "edge:ske_003") '
    "anchored to the phrase that expresses the relation itself — do not omit "
    "it just because the phrase also names the activity or the condition.\n"
    '- When you do NOT know something (its slot is marked unknown="true" in '
    "<private_known_facts>), say so and annotate the unknown slot's semantic "
    'id with mode "dont_know" (e.g. "I don\'t know" anchored to '
    '"node:skn_005:reads"). Never annotate dont_know about a slot your '
    "knowledge shows with a value or as absent, and never annotate a value "
    "for a slot your knowledge marks unknown.\n"
    '- "alignments": OPTIONAL list of private concept-identity dialogue acts. '
    "Emit an alignment ONLY when the interviewer asks you to confirm the "
    "identity of something and your reply genuinely performs that act (e.g. "
    "the interviewer asks 'do you mean X?' and you answer Yes / partly / "
    'I do not know / no, they are different). Each: {"semantic_id": "...", "quote": "...", "occurrence": 0, "act": "confirm"|"partial"|"unknown"|"dispute"} where semantic_id is the EXACT concept id from <concepts> your reply is about, and quote is the exact substring of your message that performs the act. NEVER emit alignments for ordinary statements of the workflow.\n'
    '- "terminology": OPTIONAL list of explicit terminology agreements. Emit '
    "an entry ONLY when the interviewer explicitly proposes a name for "
    "something and asks you to agree, and you do agree. Each: "
    '{"semantic_id": "...", "proposed_term": "<the exact term the '
    'interviewer proposed>", "quote": "<exact substring of your message '
    '"agreeing>", "occurrence": 0}. NEVER emit it merely for using a word '
    "in ordinary speech.\n"
    "- Worked example (ids are PLACEHOLDERS — copy the real ids verbatim "
    'from <private_known_facts>): for the message "After I check the '
    'customer in the CRM, I create the quotation.", a correct sidecar is:\n'
    '{"message": "After I check the customer in the CRM, I create the '
    'quotation.", "annotations": [{"semantic_id": "node:skn_002:system", '
    '"mode": "value", "quote": "CRM", "occurrence": 0}, '
    '{"semantic_id": "edge:ske_002", "mode": "exists", '
    '"quote": "After I check the customer", "occurrence": 0}, '
    '{"semantic_id": "node:skn_003:activity", "mode": "value", '
    '"quote": "create the quotation", '
    '"occurrence": 0}], "alignments": [], "terminology": []}\n'
    '- Never mention semantic ids inside "message"; never mention this '
    "contract."
)

# Retry feedback when the sidecar is unparseable or invalid.
_SIDECAR_ERROR_HINT = (
    "Your previous reply was rejected because it was not the required JSON "
    "sidecar. You MUST now reply with ONLY a JSON object, with no prose and no "
    "markdown fences, exactly like:\n"
    '{"message": "your natural-language reply", "annotations": [{"semantic_id": '
    '"node:skn_001:activity", "mode": "value", "quote": "exact substring of '
    '"your message", "occurrence": 0}], "alignments": [], "terminology": []}\n'
    '"semantic_id" must be one of the EXACT ids listed in '
    "<private_known_facts> (copy them verbatim, never shortened). Every "
    'annotation also needs its "mode": "value" for a slot your knowledge '
    'shows with a known value, "absent" for a slot shown as absent, '
    '"dont_know" for a slot shown as unknown, "exists" for a position or '
    'relation you assert exists, "mention" for a concept you name — the '
    "mode must match what your knowledge declares (a contradictory mode "
    "rejects the reply). Annotate EVERY element your message conveys — do not "
    "leave annotations empty when your message carries knowledge. Every "
    "annotation quote must be an exact substring of your message. Emit "
    '"alignments"/"terminology" ONLY for genuine concept-identity or '
    "terminology dialogue acts (see the contract). Do not include anything "
    "else in your reply."
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
    the shape (``message`` string, ``annotations`` list of
    {semantic_id, mode, quote, occurrence}, optional ``alignments`` list of
    {semantic_id, quote, occurrence, act} and ``terminology`` list of
    {semantic_id, proposed_term, quote, occurrence}). Every annotation MUST
    declare its semantic ``mode`` (value | absent | dont_know | exists |
    mention); the mode is then validated deterministically against the
    knowledge by ``StakeholderKnowledgeCatalog.validate_annotations``.
    Raises ``ValueError`` on anything else.
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
    raw_annotations = obj.get("annotations")
    if not isinstance(message, str) or not message.strip():
        raise ValueError(f"stakeholder sidecar has no non-empty 'message': {obj!r}")
    if not isinstance(raw_annotations, list):
        raise ValueError(f"stakeholder sidecar 'annotations' must be a list: {obj!r}")
    annotations: list[SemanticAnnotation] = []
    for raw in raw_annotations:
        if not isinstance(raw, dict):
            raise ValueError(f"annotation is not an object: {raw!r}")
        try:
            raw_mode = raw.get("mode")
            if not isinstance(raw_mode, str) or raw_mode not in (
                "value",
                "absent",
                "dont_know",
                "exists",
                "mention",
            ):
                raise ValueError(
                    f"annotation is missing its semantic mode or mode is "
                    f"unknown: {raw_mode!r} (must be one of value|absent|"
                    f"dont_know|exists|mention)"
                )
            annotations.append(
                SemanticAnnotation(
                    semantic_id=str(raw.get("semantic_id") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                    mode=raw_mode,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed annotation {raw!r}: {exc}") from exc
    alignments: list[ConceptAlignmentAssertion] = []
    for raw in obj.get("alignments") or []:
        if not isinstance(raw, dict):
            raise ValueError(f"alignment is not an object: {raw!r}")
        try:
            alignments.append(
                ConceptAlignmentAssertion(
                    semantic_id=str(raw.get("semantic_id") or ""),
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
                    semantic_id=str(raw.get("semantic_id") or ""),
                    proposed_term=str(raw.get("proposed_term") or ""),
                    quote=str(raw.get("quote") or ""),
                    occurrence=int(raw.get("occurrence") or 0),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed terminology entry {raw!r}: {exc}") from exc
    return {
        "message": message.strip(),
        "annotations": annotations,
        "alignments": alignments,
        "terminology": terminology,
    }


def _render_value(value) -> str:
    """Render a knowledge slot value: 'absent' (None), 'unknown'
    (DONT_KNOW), a knowledge concept id, or a comma list of ids."""
    if value is None:
        return "absent"
    if is_dont_know(value):
        return "unknown"
    if isinstance(value, list):
        return ",".join(v.concept_id for v in value)
    return value.concept_id


class StakeholderUserSimulator(UserSimulator):
    """The business_interview stakeholder: realizes its world model into
    natural language and returns a private sidecar.

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
        self._ledger: Optional[SemanticLedger] = None
        if task is not None:
            scenario = get_scenario(getattr(task, "id", None))
            if scenario is not None:
                self._scenario = scenario
                self._catalog = StakeholderKnowledgeCatalog.from_scenario(scenario)
                ledger = getattr(environment, "assertion_ledger", None)
                if isinstance(ledger, SemanticLedger):
                    self._ledger = ledger
                    ledger.install_catalog(self._catalog)

    # ------------------------------------------------------------- prompt

    def _knowledge_block(self) -> str:
        """Render the hidden world model: positions (with their property
        slots and values), relations (with conditions), and the local
        concepts. Everything carries its EXACT semantic id; DONT_KNOW slots
        are marked unknown; hidden Truth ids/terms never appear."""
        scenario = self._scenario
        assert scenario is not None
        graph = scenario.knowledge.graph
        positions = []
        for nid in sorted(graph.nodes):
            node = graph.nodes[nid]
            lines = []
            for prop in ("activity", "actor", "system", "rationale", "reads", "writes"):
                value = getattr(
                    node,
                    "necessity_rationale" if prop == "rationale" else prop,
                    None,
                )
                rendered = _render_value(value)
                lines.append(f'<property slot="node:{nid}:{prop}" value="{rendered}"/>')
            for prop in ("reads", "writes"):
                value = getattr(node, prop)
                if isinstance(value, list):
                    for ref in value:
                        lines.append(
                            f'<property slot="node:{nid}:{prop}:{ref.concept_id}" '
                            f'value="{ref.concept_id}"/>'
                        )
            start = "true" if nid == graph.start_node_id else "false"
            positions.append(
                f'<position id="node:{nid}" start="{start}">\n'
                + "\n".join(lines)
                + "\n</position>"
            )
        relations = []
        for eid in sorted(graph.edges):
            edge = graph.edges[eid]
            cond = _render_value(edge.condition)
            cond_line = f'<property slot="edge:{eid}:condition" value="{cond}"/>'
            relations.append(
                f'<relation id="edge:{eid}" from="node:{edge.from_node}" '
                f'to="node:{edge.to_node}">\n{cond_line}\n</relation>'
            )
        concepts = []
        for cid, concept in sorted(graph.concepts.items()):
            desc = (
                "unknown"
                if is_dont_know(concept.description)
                else repr(concept.description)
            )
            if isinstance(concept.terms, list):
                terms = ",".join(concept.terms)
            else:
                terms = "unknown"
            concepts.append(
                f'<concept id="{cid}" kind="{concept.kind}" '
                f'description="{desc}" terms="{terms}"/>'
            )
        return _KNOWLEDGE_BLOCK_TEMPLATE.format(
            positions_xml="\n".join(positions),
            relations_xml="\n".join(relations),
            concepts_xml="\n".join(concepts),
        )

    @property
    def system_prompt(self) -> str:
        base = super().system_prompt
        if self._catalog is None or self._scenario is None:
            return base
        return base + self._knowledge_block()

    # ------------------------------------------------------------- sidecar

    def _generate_sidecar(self, messages: list, contract: Optional[str] = None) -> dict:
        """Call the user LLM and parse/validate the private sidecar."""
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
            self._catalog.validate_annotations(
                sidecar["annotations"], sidecar["message"]
            )
            self._catalog.validate_events(
                sidecar["alignments"], sidecar["terminology"], sidecar["message"]
            )
        return sidecar

    def _call_llm(self, messages: list):
        """One LLM completion (kept separate for testability)."""
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
        annotations/events travel on private fields (excluded from
        serialization) for the environment's ledger binding. One retry with
        corrective feedback is allowed; a second invalid sidecar is rejected
        loudly.
        """
        if isinstance(message, AssistantMessage) and message.is_audio:
            raise ValueError(
                "Assistant message cannot be audio. Use VoiceUserSimulator instead."
            )
        logger.debug(f"User responds to message: {message}")
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
            stakeholder_annotations=[a.model_dump() for a in sidecar["annotations"]],
            stakeholder_alignments=[e.model_dump() for e in sidecar["alignments"]],
            stakeholder_terminology=[e.model_dump() for e in sidecar["terminology"]],
        )
