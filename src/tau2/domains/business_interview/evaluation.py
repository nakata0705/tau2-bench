"""Evaluator for the graph-native business_interview benchmark (v13 — Truth
reconstruction is the primary score).

The agent's inferred ``AgentGraph`` + ``AgentConcept[]`` is compared to the
**Truth** (``BusinessProcessGraph`` + ``TruthConcept[]``). Reconstruction is
the primary benchmark concern: a structurally and semantically faithful Agent
graph scores correctly even when exact conversational evidence annotations
(``EvidenceRef`` quote spans) are missing or ambiguous, or the fact was
inferred rather than explicitly exposed. Unsupported-but-correct
reconstruction counts as correct; fabricated/wrong content does not.

Concept identity is matched deterministically by **content** over the
agent-local glossary (``display_label`` / ``description``) against the Truth
concept signatures (``canonical_terms`` / ``description``), scoped to the
same ``kind``. Node identity falls out of the activity/slot content
signature; edge identity falls out of the node alignment plus existence of a
matching Truth edge.

Epistemic belief recording keeps explicit four-state Agent slots (see
``tools.py``), but the *score* evaluates the final belief against Truth
without requiring conversational proof. The scoring rule is asymmetric and
explicit:

* Truth ``ConceptRef``: only a content-matching asserted Agent ``ConceptRef``
  is correct; ``UNSET``, ``ABSENT``, ``DONT_KNOW`` and a wrong concept are
  incorrect.
* Truth ``None`` (canonical absence): only an explicit Agent ``ABSENT``
  marker is correct; ``UNSET``, ``DONT_KNOW`` and any ``ConceptRef`` are
  incorrect.

The same rule applies to reads/writes known-empty properties and unconditional
edge conditions. The private provenance ledger
(annotations/alignments/terminology) and evidence-hygiene metrics remain as
**diagnostics** only. They are reported but never gate ``quality_pass``.

StakeholderKnowledge stays a simulator constraint, not the scored target: it
limits what the conversation can reveal, while the scored target is the Truth.
"""

import re
import unicodedata
from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentGraph,
    ConceptRef,
    DontKnowType,
    EvidenceRef,
    InterviewDB,
    business_edge_ids,
    business_entry_node_ids,
    business_graph_projection,
    business_node_ids,
    canonical_structure_errors,
    edge_is_structural,
    is_absent,
    is_dont_know,
    is_unset,
    node_is_structural,
)
from tau2.domains.business_interview.grounding import (
    grounded_ids as grounded_semantic_ids,
)
from tau2.domains.business_interview.grounding import (
    grounded_refs as resolve_grounding_refs,
)

from .joint_structural_alignment import (  # pyright: ignore[reportMissingImports]
    JointStructuralAlignmentDiagnostics,
    build_joint_structural_alignment_diagnostics,
)
from .usage_alignment import (  # pyright: ignore[reportMissingImports]
    UsageAlignmentDiagnostics,
    build_usage_alignment_diagnostics,
)

# Re-exported provenance helpers kept for diagnostic callers (tests/diagnostics).
__all__ = [
    "EvaluationSpec",
    "EvaluationResult",
    "EvaluationDiagnostics",
    "FailureAttribution",
    "evaluate",
    "grounded_semantic_ids",
    "resolve_grounding_refs",
]

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")


class ConceptSummary(BaseModel):
    """A Truth or Agent concept identity rendered for offline inspection."""

    concept_id: str
    kind: str
    label: Optional[str] = None
    labels: list[str] = Field(default_factory=list)


class ConceptPairDiagnostic(BaseModel):
    """One deterministic candidate considered by the concept matcher."""

    truth_concept_id: str
    truth_kind: str
    truth_label: Optional[str] = None
    truth_labels: list[str] = Field(default_factory=list)
    agent_concept_id: str
    agent_kind: str
    agent_label: Optional[str] = None
    exact_label_match: bool = False
    exact_label_match_path: Optional[str] = None
    lexical_similarity_score: float = 0.0
    threshold: float
    eligible: bool = False
    selected_mapping: bool = False


class ConceptMappingDiagnostic(BaseModel):
    """A selected Agent-local -> Truth concept mapping."""

    agent_concept_id: str
    truth_concept_id: str
    agent_kind: str
    truth_kind: str
    agent_label: Optional[str] = None
    truth_label: Optional[str] = None
    exact_label_match_path: Optional[str] = None
    lexical_similarity_score: float = 0.0


class ConceptDiagnostics(BaseModel):
    """Complete trace of the concept alignment inputs and selected mapping."""

    threshold: float
    expected_truth_concept_ids: list[str] = Field(default_factory=list)
    attempted_agent_concept_ids: list[str] = Field(default_factory=list)
    agent_to_truth: dict[str, str] = Field(default_factory=dict)
    candidate_pairs: list[ConceptPairDiagnostic] = Field(default_factory=list)
    selected_mappings: list[ConceptMappingDiagnostic] = Field(default_factory=list)
    unmatched_truth_concepts: list[ConceptSummary] = Field(default_factory=list)
    unmatched_agent_concepts: list[ConceptSummary] = Field(default_factory=list)


class SlotItemDiagnostic(BaseModel):
    """Trace for one expected or asserted reads/writes list element."""

    item_type: str
    truth_concept_id: Optional[str] = None
    agent_concept_id: Optional[str] = None
    mapped_truth_concept_id: Optional[str] = None
    truth_label: Optional[str] = None
    agent_label: Optional[str] = None
    matched: bool = False
    reason: str


class SlotDiagnostic(BaseModel):
    """Per-property scoring explanation for one Truth node/edge slot."""

    property: str
    slot_kind: str
    truth_state: str
    agent_state: str
    truth_concept_id: Optional[str] = None
    agent_concept_id: Optional[str] = None
    truth_concept_ids: list[str] = Field(default_factory=list)
    agent_concept_ids: list[str] = Field(default_factory=list)
    mapped_truth_concept_ids: list[str] = Field(default_factory=list)
    truth_label: Optional[str] = None
    agent_label: Optional[str] = None
    truth_labels: list[str] = Field(default_factory=list)
    agent_labels: list[str] = Field(default_factory=list)
    matched: bool = False
    score_contribution: float = 0.0
    reason: str
    reason_codes: list[str] = Field(default_factory=list)
    missing_truth_concept_ids: list[str] = Field(default_factory=list)
    extra_agent_concept_ids: list[str] = Field(default_factory=list)
    items: list[SlotItemDiagnostic] = Field(default_factory=list)


class NodeDiagnostic(BaseModel):
    """Node alignment plus all six Truth property-slot diagnostics."""

    truth_node_id: str
    agent_node_id: Optional[str] = None
    matched: bool = False
    reason: str
    slots: dict[str, SlotDiagnostic] = Field(default_factory=dict)


class EdgeDiagnostic(BaseModel):
    """Edge alignment, endpoint structural checks, and condition diagnostic."""

    truth_edge_id: str
    agent_edge_id: Optional[str] = None
    matched: bool = False
    reason: str
    truth_from_node: str
    truth_to_node: str
    agent_from_node: Optional[str] = None
    agent_to_node: Optional[str] = None
    from_node_match: bool = False
    to_node_match: bool = False
    structural_match: bool = False
    condition: SlotDiagnostic


ROOT_CAUSE_CATEGORIES = (
    "stakeholder_disclosure",
    "agent_elicitation",
    "agent_recording",
    "evaluator_matching",
    "insufficient_evidence_to_classify",
)


class FailureAttribution(BaseModel):
    """Conservative offline attribution for one failed scored slot."""

    target_id: str
    dimension: str
    property: str
    category: Literal[
        "stakeholder_disclosure",
        "agent_elicitation",
        "agent_recording",
        "evaluator_matching",
        "insufficient_evidence_to_classify",
    ]
    reason: str
    evidence: list[str] = Field(default_factory=list)


class EvaluationDiagnostics(BaseModel):
    """Evaluator-only Truth/Agent reconstruction trace.

    ``canonical_contract`` is diagnostic metadata only.  Structural SOURCE,
    SINK, boundary edges, and shortcut provenance are never folded into the
    ordinary business node/edge score fields.
    """

    schema_version: str = "business_interview.evaluation_diagnostics.v4"
    canonical_contract: dict[str, object] = Field(default_factory=dict)
    score_fields_unchanged: bool = True
    node_diagnostics: list[NodeDiagnostic] = Field(default_factory=list)
    unmatched_agent_nodes: list[str] = Field(default_factory=list)
    edge_diagnostics: list[EdgeDiagnostic] = Field(default_factory=list)
    unmatched_agent_edges: list[str] = Field(default_factory=list)
    concepts: ConceptDiagnostics
    usage_alignment: UsageAlignmentDiagnostics = Field(
        default_factory=UsageAlignmentDiagnostics
    )
    joint_structural_alignment: JointStructuralAlignmentDiagnostics = Field(
        default_factory=JointStructuralAlignmentDiagnostics
    )


class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    v12: the spec carries no semantic matchers and no provenance matchers.
    Everything semantic lives in the Truth graph. Kept as a model for API
    stability; scenarios may leave it empty.
    """


class EvaluationResult(BaseModel):
    protocol_completed: bool
    graph_created: bool
    graph_valid: bool

    node_recall: float
    node_precision: float
    edge_recall: float
    edge_precision: float
    start_correct: bool
    end_recall: float
    end_precision: float
    activity_correctness: float
    actor_correctness: float
    system_correctness: float
    read_correctness: float
    write_correctness: float
    rationale_correctness: float
    condition_correctness: float
    concept_correctness: float
    concept_recall: float
    concept_precision: float
    unsupported_ref_count: int
    fabricated_node_count: int
    fabricated_edge_count: int

    # glossary completeness (concept reconstruction completeness)
    glossary_complete: bool

    # evidence hygiene (diagnostic only — never gates quality anymore)
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
    marker_evidence_errors_surrogate: int
    invalid_observation_reference_count: int
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool
    evidence_pass: bool
    reconstruction_pass: bool

    structural_pass: bool
    protocol_pass: bool
    quality_pass: bool

    # informational: what share of the Truth the conversation's stakeholder
    # could actually expose (never part of Agent performance scoring)
    knowledge_coverage: float

    # evaluator-only Truth/Agent reconstruction trace. This is deliberately
    # not part of InterviewDB or any Agent/Stakeholder-visible surface.
    diagnostics: EvaluationDiagnostics


# ---------------------------------------------------------------------------
# Content signatures (deterministic, language-tolerant; never semantic NLP)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "is",
        "are",
        "was",
        "be",
        "it",
        "as",
        "that",
        "this",
        "we",
        "do",
        "does",
        "doesn",
        "don",
        "via",
        "into",
        "from",
        "then",
        "after",
        "before",
        "when",
        "if",
        "so",
        "also",
        "using",
        "use",
        "used",
        "has",
        "have",
        "had",
        "there",
        "their",
        "i",
        "my",
        "you",
        "your",
        "he",
        "she",
        "they",
        "who",
        "what",
        "all",
        "any",
        "some",
        "not",
        "no",
        "yes",
        "but",
        "or",
        "same",
        "other",
        "about",
        "would",
        "will",
        "can",
        "could",
        "should",
        "just",
        "very",
        "much",
        "more",
        "most",
        "than",
        "up",
        "down",
        "out",
        "over",
        "again",
        "once",
        "day",
        "time",
        "things",
        "thing",
    }
)

# Broad nouns carry little identity information when they are the only token
# in a label. They are removed from lexical overlap, but exact canonical/local
# label equality is handled separately by ``_concept_similarity``. This keeps
# an exact ``CRM``/``SAP``/``Excel`` (or exact generic label) match possible
# without allowing ``system`` to match every ``... system`` concept.
_GENERIC_TOKENS = frozenset(
    {
        "system",
        "systems",
        "document",
        "documents",
        "information",
        "quotation",
        "quotations",
        "process",
        "processes",
        "data",
    }
)

# CJK / full-width ranges: Hiragana, Katakana, CJK ideographs, CJK ext.
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\U00020000-\U0002a6df\U0002b740-\U0002b81f]"
)


def _normalize(text: Optional[str]) -> str:
    """NFKC normalize + lowercase; language-neutral (keeps Japanese text)."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).lower()


def _split_runs(text: str):
    """Yield ``(is_cjk, chunk)`` runs over the normalized text."""
    if not text:
        return
    cur_is_cjk = bool(_CJK_RE.match(text[0]))
    start = 0
    for i, ch in enumerate(text):
        is_cjk = bool(_CJK_RE.match(ch))
        if is_cjk != cur_is_cjk:
            yield cur_is_cjk, text[start:i]
            start = i
            cur_is_cjk = is_cjk
    yield cur_is_cjk, text[start:]


def _char_bigrams(chunk: str) -> set[str]:
    """Character bigrams of a CJK run (single char when len 1)."""
    if len(chunk) == 1:
        return {chunk}
    return {chunk[i : i + 2] for i in range(len(chunk) - 1)}


def _tokens(text: Optional[str]) -> set[str]:
    """Language-tolerant deterministic signature tokenization:

    - NFKC normalize + lowercase (so Japanese/full-width text is preserved);
    - Latin/alphanumeric runs: whole-word tokens with deterministic
      stop-word and low-information generic-token sets removed;
    - CJK runs (Hiragana/Katakana/ideographs): character bigrams (no
      language-specific word splitter needed).

    A single generic token like "the"/"system" therefore cannot make two
    unrelated concepts look equivalent.
    """
    norm = _normalize(text)
    if not norm:
        return set()
    toks: set[str] = set()
    for is_cjk, chunk in _split_runs(norm):
        if is_cjk:
            toks |= _char_bigrams(chunk)
        else:
            for w in re.findall(r"[a-z0-9]+", chunk):
                if len(w) >= 2 and w not in _STOP_WORDS and w not in _GENERIC_TOKENS:
                    toks.add(w)
    return toks


def _similarity(a: "set", b: "set") -> float:
    """Symmetric Dice coefficient over token sets; 0 on empty/disjoint input.
    Dice (not Jaccard) is the standard score for n-gram signatures and is
    far more lenient for short labels than token Jaccard."""
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return 2.0 * len(inter) / (len(a) + len(b))


# Minimum similarity for two concepts to be considered the same business
# thing. A single shared generic token lands well below this.
_CONCEPT_MATCH_THRESHOLD = 0.4


def _truth_label_tokens(concept, extra_terms=None) -> set[str]:
    """Signature of the Truth concept's canonical terms (plus any extra
    local terms from the stakeholder's vocabulary, e.g. Japanese terms)
    only — the optional description is excluded so a long description never
    dilutes a strong label match."""
    toks: set[str] = set()
    for t in concept.canonical_terms:
        toks |= _tokens(t)
    for t in extra_terms or ():
        toks |= _tokens(t)
    return toks


def _truth_concept_tokens(concept, extra_terms=None) -> set[str]:
    """Signature of canonical terms + description (+ extra local terms),
    used as a fallback when the labels do not overlap enough."""
    return _truth_label_tokens(concept, extra_terms) | _tokens(concept.description)


def _agent_label_tokens(concept) -> set[str]:
    return _tokens(concept.display_label)


def _agent_concept_tokens(concept) -> set[str]:
    """Signature of the agent label + description (fallback signature)."""
    return _agent_label_tokens(concept) | _tokens(concept.description)


def _label_key(text: Optional[str]) -> str:
    """Return a compact normalized label key for exact/near-exact labels.

    Punctuation and spacing are ignored here (so ``month-end`` and
    ``month end`` agree), while the lexical Dice path remains responsible for
    partial/multi-token overlap. The key is Unicode-aware and deterministic.
    """
    return "".join(ch for ch in _normalize(text) if ch.isalnum())


def _has_exact_label_match(agent, truth_concept, extra_terms=None) -> bool:
    """Whether the Agent label equals a canonical or local Truth label.

    Exact label equality is intentionally checked before generic-token
    filtering. It preserves legitimate short identifiers and exact generic
    labels while preventing a one-token generic label from matching a longer
    label that merely contains it.
    """
    agent_key = _label_key(agent.display_label)
    if not agent_key:
        return False
    candidates = list(truth_concept.canonical_terms) + list(extra_terms or ())
    return any(agent_key == _label_key(term) for term in candidates)


def _concept_similarity(agent, truth_concept, extra_terms=None) -> float:
    """Best-of exact-label and label/description Dice matching.

    Exact canonical/local label equality is a full match. Otherwise the
    lexical path compares label-only and label+description signatures after
    removing stop and low-information generic tokens. Each side is computed
    against canonical Truth terms and canonical + stakeholder-extra terms
    (e.g. Japanese). Taking the max preserves English-vs-English matches
    while allowing a JA agent label to match a JA extra term. The result is
    thresholded against ``_CONCEPT_MATCH_THRESHOLD`` by the caller.

    This is lexical reconstruction, not language-independent semantic
    equivalence: paraphrases must share tokens or be supplied as a
    scenario-local Truth/knowledge term.
    """
    if _has_exact_label_match(agent, truth_concept, extra_terms):
        return 1.0
    a_label = _agent_label_tokens(agent)
    a_full = _agent_concept_tokens(agent)
    t_label = _truth_label_tokens(truth_concept)
    t_full = _truth_concept_tokens(truth_concept)
    t_label_ext = _truth_label_tokens(truth_concept, extra_terms)
    t_full_ext = _truth_concept_tokens(truth_concept, extra_terms)
    return max(
        _similarity(a_label, t_label),
        _similarity(a_label, t_label_ext),
        _similarity(a_full, t_full),
        _similarity(a_full, t_full_ext),
    )


def _max_weight_assignment(
    weights: dict[tuple[str, str], float],
    left: list[str],
    right: list[str],
    threshold: float,
) -> dict[str, str]:
    """Deterministic maximum-weight bipartite matching (Hungarian / Kuhn-
    Munkres) mapping rows (``left``) to distinct columns (``right``).

    Only pairs with ``weight >= threshold`` are allowed; every other pair is
    treated as forbidden. The matrix is padded with explicit dummy rows and
    dummy columns at cost 0, so every real row always has a zero-cost
    fallback: a real row matched to a real column is kept, anything matched
    to a dummy (or a forbidden pair) is left unmatched. Ties are broken by
    the sorted input order, so the optimum is independent of Agent-local
    concept/node ids.

    Returns ``{left_id: right_id}`` for the matched pairs.
    """
    n, m = len(left), len(right)
    if n == 0 or m == 0:
        return {}
    size = n + m  # real rows + dummy rows, real cols + dummy cols
    BIG = 1e15

    # cost[col][row] for min-cost assignment; row/col 1..size (1-indexed).
    # Real row i in 1..n, real col j in 1..m.
    cost = [[0.0] * (size + 1) for _ in range(size + 1)]

    row_idx = {cid: i for i, cid in enumerate(left, start=1)}
    col_idx = {tid: j for j, tid in enumerate(right, start=1)}
    allowed: set[tuple[int, int]] = set()
    for (lft, rgt), w in weights.items():
        if w >= threshold and lft in row_idx and rgt in col_idx:
            i, j = row_idx[lft], col_idx[rgt]
            cost[j][i] = -w
            allowed.add((i, j))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (i, j) not in allowed:
                cost[j][i] = BIG

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [1e18] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 1e18
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                cur = cost[j][i0] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # augment along the found path
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment: dict[str, str] = {}
    for j in range(1, m + 1):
        i = p[j]
        if 1 <= i <= n and (i, j) in allowed:
            assignment[left[i - 1]] = right[j - 1]
    return assignment


def _prop_from_node(node, prop):
    """Read a property slot from a Truth or Agent node (agent may carry
    four-state markers; Truth two-valued)."""
    if prop in ("reads", "writes"):
        return getattr(node, prop)
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


def _node_concept_refs(node, agent_concepts) -> list[tuple[str, ConceptRef]]:
    """All asserted (concept_id, ConceptRef) of an Agent node."""
    out: list[tuple[str, ConceptRef]] = []
    for prop in _NODE_PROPS:
        for ref in node.asserted_refs(prop):
            out.append((prop, ref))
    return out


def _node_signature(
    node, concepts, is_truth: bool, agent_to_truth: Optional[dict[str, str]] = None
) -> set[tuple[str, str]]:
    """Node signature from its (prop, truth_concept_id) references.

    For a Truth node this is derived straight from its concept refs; for an
    Agent node each ref's Agent-local concept is first mapped to its aligned
    Truth concept (unmapped/fabricated refs contribute nothing, so they
    cannot help a fabricated node get matched). This makes node identity
    depend on the aligned concept content, never on Agent-local ids.
    """
    sig: set[tuple[str, str]] = set()
    for prop in _NODE_PROPS:
        for ref in node.refs(prop):
            cid = ref.concept_id
            if is_truth:
                if cid in concepts:
                    sig.add((prop, cid))
                continue
            mapped = agent_to_truth.get(cid) if agent_to_truth is not None else None
            if mapped is not None:
                sig.add((prop, mapped))
    return sig


def _map_nodes_and_edges(
    agent: AgentGraph,
    truth,
    agent_to_truth: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Stable global node mapping by aligned (prop, truth_concept_id) pairs;
    edge mapping by matching endpoint pair on the Truth graph.

    Node matching is a maximum-weight bipartite assignment (not greedy by
    Agent-local id order), so the mapping is invariant to arbitrary
    Agent-node-id reorderings.
    """
    agent_sigs = {
        anid: _node_signature(
            node, agent.concepts, is_truth=False, agent_to_truth=agent_to_truth
        )
        for anid, node in agent.nodes.items()
    }
    truth_sigs = {
        tnid: _node_signature(node, truth.concepts, is_truth=True)
        for tnid, node in truth.nodes.items()
    }
    weights: dict[tuple[str, str], float] = {}
    for anid in agent_sigs:
        a = agent_sigs[anid]
        if not a:
            continue
        for tnid in truth_sigs:
            t = truth_sigs[tnid]
            inter = a & t
            if not inter:
                continue
            # A shared activity pair is the node's primary identity; sharing
            # only a minor property (e.g. one actor) is NOT identity. This
            # keeps the mapping stable and content-driven without accepting
            # tiny incidental overlaps.
            shares_activity = any(prop == "activity" for prop, _ in inter)
            if not shares_activity and len(inter) < 2:
                continue
            s = _similarity(a, t)
            if s > 0.0:
                weights[(anid, tnid)] = s
    mapping = _max_weight_assignment(
        weights, sorted(agent_sigs), sorted(truth_sigs), threshold=0.0
    )

    edge_map: dict[str, str] = {}
    for eid, edge in agent.edges.items():
        a = mapping.get(edge.from_node)
        b = mapping.get(edge.to_node)
        if a is None or b is None:
            continue
        cands = [
            tid
            for tid, te in truth.edges.items()
            if te.from_node == a and te.to_node == b
        ]
        if cands:
            edge_map[eid] = sorted(cands)[0]
    return mapping, edge_map


_NO_VALUE = object()


def _truth_scalar_value(value):
    """The Truth concept id a scalar slot value stands for (None -> None)."""
    if isinstance(value, ConceptRef):
        return value.concept_id
    return None


def _score_scalar_slot(
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
) -> int:
    """Epistemic-aware scalar slot score (1 correct / 0 otherwise).

    Truth ``ConceptRef`` -> only a matching asserted Agent ``ConceptRef`` is
    correct; every other Agent state is incorrect. Truth ``None`` -> only an
    explicit Agent ``ABSENT`` marker is correct; ``UNSET``, ``DONT_KNOW`` and
    any ``ConceptRef`` are incorrect. The same rule applies to edge
    conditions; no-answer is never rewarded as a lucky guess.
    """
    tcid = _truth_scalar_value(truth_value)
    if tcid is not None:
        if not isinstance(agent_value, ConceptRef) or not agent_value.asserted:
            return 0
        return 1 if agent_to_truth.get(agent_value.concept_id) == tcid else 0
    # Truth absent: explicit ABSENT is the only correct state
    if isinstance(agent_value, AbsentType):
        return 1
    return 0


def _score_list_slot(
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
) -> tuple[float, int]:
    """Epistemic-aware reads/writes slot score, plus unsupported count.

    Truth list -> recall*precision over the normal element set.
    Truth None / known-empty -> only an explicit Agent ABSENT marker scores
    1.0; UNSET and DONT_KNOW are incomplete; asserted list refs are
    fabricated.
    """
    expected: set[str] = (
        {ref.concept_id for ref in truth_value}
        if isinstance(truth_value, list)
        else set()
    )
    if not expected:
        if isinstance(agent_value, AbsentType):
            return 1.0, 0
        if isinstance(agent_value, list):
            unsupported = sum(1 for r in agent_value if r.asserted)
            return 0.0, unsupported
        return 0.0, 0
    if not isinstance(agent_value, list):
        return 0.0, 0
    claimed = {
        cid
        for r in agent_value
        if r.asserted and (cid := agent_to_truth.get(r.concept_id)) is not None
    }
    if not claimed:
        return 0.0, 0
    recall = len(claimed & expected) / len(expected)
    precision = len(claimed & expected) / len(claimed)
    unsupported = sum(
        1
        for r in agent_value
        if r.asserted and agent_to_truth.get(r.concept_id) is None
    )
    return recall * precision, unsupported


# ---------------------------------------------------------------------------
# Concept identity (content-based)
# ---------------------------------------------------------------------------


def _truth_referenced_concept_ids(truth) -> set[str]:
    """Truth concept ids referenced by the Truth graph."""
    ids: set[str] = set()
    for node in truth.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                ids.add(ref.concept_id)
    for edge in truth.edges.values():
        if isinstance(edge.condition, ConceptRef):
            ids.add(edge.condition.concept_id)
    return ids


def _agent_referenced_concept_ids(agent: AgentGraph) -> set[str]:
    return agent.referenced_concepts()


def _align_concepts(
    agent: AgentGraph,
    truth,
    term_extras: Optional[dict[str, list[str]]] = None,
) -> tuple[float, float, dict[str, str]]:
    """Content-based bijective concept alignment (per kind, global optimum).

    Expected = Truth concepts the Truth graph references. Attempted = agent
    concepts the Agent graph references. For each concept kind separately, a
    maximum-weight bipartite matching maps attempted -> expected concepts
    (``_CONCEPT_MATCH_THRESHOLD`` gates weak pairs). This is deterministic
    and invariant to Agent-local concept ids.

    Returns ``(concept_recall, concept_precision,
    agent_concept_id -> truth_concept_id)``; precision is over the attempted
    set, recall over expected.
    """
    expected: set[str] = _truth_referenced_concept_ids(truth)
    attempted: set[str] = _agent_referenced_concept_ids(agent)

    # per-kind bipartite assignment (maximum-weight, deterministic)
    by_kind = sorted({ac.kind for ac in agent.concepts.values() if ac.id in attempted})
    recognized: dict[str, str] = {}
    for kind in by_kind:
        left = sorted(
            cid
            for cid in attempted
            if cid in agent.concepts and agent.concepts[cid].kind == kind
        )
        right = sorted(
            tid
            for tid in expected
            if tid in truth.concepts and truth.concepts[tid].kind == kind
        )
        weights: dict[tuple[str, str], float] = {}
        for acid in left:
            for tid in right:
                s = _concept_similarity(
                    agent.concepts[acid],
                    truth.concepts[tid],
                    (term_extras or {}).get(tid),
                )
                if s >= _CONCEPT_MATCH_THRESHOLD:
                    weights[(acid, tid)] = s
        recognized.update(
            _max_weight_assignment(
                weights, left, right, threshold=_CONCEPT_MATCH_THRESHOLD
            )
        )

    recalled = set(recognized.values()) & expected
    concept_recall = len(recalled) / len(expected) if expected else 1.0
    concept_precision = len(recognized) / len(attempted) if attempted else 1.0
    agent_to_truth = {acid: tid for acid, tid in recognized.items() if tid in expected}
    return concept_recall, concept_precision, agent_to_truth


# ---------------------------------------------------------------------------
# Node / edge identity (content-based)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evidence hygiene (diagnostics only)
# ---------------------------------------------------------------------------


def _evidence_metrics(
    db: InterviewDB,
    agent: AgentGraph,
) -> tuple[int, int, float, float, float]:
    """(invalid, invalid_obs_refs, node_cov, ref_cov, edge_cov)."""
    invalid = invalid_obs = 0
    ref_total = ref_hit = 0
    node_total = node_hit = 0
    edge_total = edge_hit = 0
    obs_text = {o.id: o.text for o in db.observations}

    def span_ok(ev: EvidenceRef) -> bool:
        nonlocal invalid, invalid_obs
        text = obs_text.get(ev.observation_id)
        if text is None:
            invalid += 1
            invalid_obs += 1
            return False
        if ev.quote and ev.resolve_span(text) is None:
            invalid += 1
            return False
        return True

    def all_ok(evs) -> bool:
        ok = True
        for ev in evs:
            if not span_ok(ev):
                ok = False
        return ok

    for node in agent.nodes.values():
        refs: list[ConceptRef] = []
        markers: list[EvidenceRef] = []
        for prop in _NODE_PROPS:
            refs.extend(node.refs(prop))
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                markers.extend(slot.evidence)
        node_total += 1
        if any(
            r.asserted and any(span_ok(ev) for ev in r.evidence) for r in refs
        ) or any(span_ok(ev) for ev in markers):
            node_hit += 1
        for r in refs:
            if not r.asserted:
                continue
            ref_total += 1
            if r.evidence and all_ok(r.evidence):
                ref_hit += 1
    for edge in agent.edges.values():
        edge_total += 1
        if edge.evidence and all_ok(edge.evidence):
            edge_hit += 1
    node_cov = node_hit / node_total if node_total else 1.0
    ref_cov = ref_hit / ref_total if ref_total else 1.0
    edge_cov = edge_hit / edge_total if edge_total else 1.0
    return invalid, invalid_obs, node_cov, ref_cov, edge_cov


def _all_ref_obs(agent: AgentGraph) -> set[str]:
    ids: set[str] = set()
    for node in agent.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                for ev in ref.evidence:
                    ids.add(ev.observation_id)
        for prop in _NODE_PROPS:
            slot = node.slot_value(prop)
            if isinstance(slot, (AbsentType, DontKnowType)):
                for ev in slot.evidence:
                    ids.add(ev.observation_id)
    for edge in agent.edges.values():
        for ev in edge.evidence:
            ids.add(ev.observation_id)
        if isinstance(edge.condition, ConceptRef):
            for ev in edge.condition.evidence:
                ids.add(ev.observation_id)
    return ids


# ---------------------------------------------------------------------------
# Knowledge coverage (informational)
# ---------------------------------------------------------------------------


def _knowledge_coverage(truth, knowledge) -> float:
    """Share of Truth addresses the conversation's stakeholder knows
    (informational only; never part of Agent scoring). Tree: node existence
    (1) + each node property slot (6) + each element of a known reads/writes
    list (per Truth element) + edge existence (1) + edge condition slot (1).
    Known values AND known absence count as known; DONT_KNOW slots and
    removed nodes/edges count as unknown; node existence is never confused
    with the activity slot."""
    kg = knowledge.graph if knowledge is not None else None
    if truth is None or kg is None:
        return 0.0
    node_t2l = {t: k for k, t in kg.node_truth_ids.items()}
    edge_t2l = {t: k for k, t in kg.edge_truth_ids.items()}
    concept_t2l = {c.truth_concept_id: k for k, c in kg.concepts.items()}
    total = known = 0

    for nid in business_node_ids(truth):
        node = truth.nodes[nid]
        local = node_t2l.get(nid)
        kn = kg.nodes.get(local) if local is not None else None
        total += 1
        if kn is not None:
            known += 1  # node existence is known (removed nodes are not)
        for prop in _NODE_PROPS:
            total += 1
            if kn is None:
                if prop in ("reads", "writes"):
                    for _ref in getattr(node, prop) or []:
                        total += 1  # removed node -> elements unknown
                continue
            value = _prop_value(kn, prop)
            if not is_dont_know(value):
                known += 1  # known value OR known absence
            if prop in ("reads", "writes"):
                for ref in getattr(node, prop) or []:
                    total += 1
                    if is_dont_know(value):
                        continue
                    if value is None:
                        known += 1  # known-absent whole property
                    elif isinstance(value, list) and concept_t2l.get(
                        ref.concept_id
                    ) in {r.concept_id for r in value}:
                        known += 1
    for eid in business_edge_ids(truth):
        local = edge_t2l.get(eid)
        ke = kg.edges.get(local) if local is not None else None
        total += 1
        if ke is not None:
            known += 1  # edge existence is known (removed edges are not)
        total += 1  # the condition slot itself
        if ke is not None and not is_dont_know(ke.condition):
            known += 1
    return known / total if total else 0.0


def _prop_value(node, prop):
    if prop in ("reads", "writes"):
        return getattr(node, prop)
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


def _canonical_contract_diagnostic(graph, knowledge=None) -> dict[str, object]:
    """Describe boundary/shortcut handling without affecting any score."""
    has_explicit_boundary = bool(
        getattr(graph, "source_node_id", None)
        and getattr(graph, "sink_node_id", None)
        and any(node_is_structural(node) for node in graph.nodes.values())
    )
    structural_nodes = [
        node_id for node_id, node in graph.nodes.items() if node_is_structural(node)
    ]
    structural_edges = [
        edge_id for edge_id, edge in graph.edges.items() if edge_is_structural(edge)
    ]
    shortcuts = [
        {
            "edge_id": edge_id,
            "contracted_nodes": list(getattr(edge, "contracted_nodes", [])),
            "derived_from_edges": list(getattr(edge, "derived_from_edges", [])),
        }
        for edge_id, edge in sorted(graph.edges.items())
        if getattr(edge, "is_shortcut", False)
    ]
    errors = canonical_structure_errors(graph) if has_explicit_boundary else []
    stakeholder_graph = getattr(knowledge, "graph", None)
    stakeholder_shortcuts = []
    if stakeholder_graph is not None:
        stakeholder_shortcuts = [
            {
                "edge_id": edge_id,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "contracted_nodes": list(getattr(edge, "contracted_nodes", [])),
                "derived_from_edges": list(getattr(edge, "derived_from_edges", [])),
            }
            for edge_id, edge in sorted(stakeholder_graph.edges.items())
            if getattr(edge, "is_shortcut", False)
        ]
    return {
        "explicit_source_node_id": getattr(graph, "source_node_id", None),
        "explicit_sink_node_id": getattr(graph, "sink_node_id", None),
        "canonical": has_explicit_boundary and not errors,
        "invariant_errors": errors,
        "structural_node_count": len(structural_nodes),
        "structural_edge_count": len(structural_edges),
        "business_node_count": len(business_node_ids(graph)),
        "business_edge_count": len(business_edge_ids(graph)),
        "structural_node_ids": sorted(structural_nodes),
        "structural_edge_ids": sorted(structural_edges),
        "truth_shortcut_edges": shortcuts,
        "stakeholder_shortcut_edges": stakeholder_shortcuts,
        "existing_evaluator_shortcut_policy": (
            "shortcut edges are currently ordinary Agent business edges; "
            "provenance is diagnostic and no score credit is granted"
        ),
        "scoring_excludes_structural_elements": True,
        "topology_derived_end_inference_used": False,
    }


# ---------------------------------------------------------------------------
# Offline reconstruction diagnostics
# ---------------------------------------------------------------------------


_DIAGNOSTIC_NODE_PROPS = (
    ("activity", "activity"),
    ("actor", "actor"),
    ("system", "system"),
    ("reads", "reads"),
    ("writes", "writes"),
    ("necessity_rationale", "rationale"),
)


def _concept_labels(concept, is_truth: bool) -> list[str]:
    if concept is None:
        return []
    if is_truth:
        return [str(label) for label in (concept.canonical_terms or [])]
    label = getattr(concept, "display_label", None)
    return [str(label)] if label else []


def _concept_label(concept, is_truth: bool) -> Optional[str]:
    labels = _concept_labels(concept, is_truth)
    return labels[0] if labels else None


def _exact_label_match_path(agent, truth_concept, extra_terms=None) -> Optional[str]:
    """Return which existing exact-label branch selected, without matching."""
    agent_key = _label_key(getattr(agent, "display_label", None))
    if not agent_key:
        return None
    for term in truth_concept.canonical_terms or []:
        if agent_key == _label_key(term):
            return "canonical_term"
    for term in extra_terms or ():
        if agent_key == _label_key(term):
            return "stakeholder_extra_term"
    return None


def _concept_diagnostics(
    agent: AgentGraph,
    truth,
    term_extras: dict[str, list[str]],
    agent_to_truth: dict[str, str],
) -> ConceptDiagnostics:
    """Trace the already-selected concept mapping without changing it."""
    expected = sorted(_truth_referenced_concept_ids(truth))
    attempted = sorted(_agent_referenced_concept_ids(agent))
    pairs: list[ConceptPairDiagnostic] = []
    for acid in attempted:
        aconcept = agent.concepts.get(acid)
        if aconcept is None:
            continue
        for tid in expected:
            tconcept = truth.concepts.get(tid)
            if tconcept is None or aconcept.kind != tconcept.kind:
                continue
            path = _exact_label_match_path(aconcept, tconcept, term_extras.get(tid))
            similarity = _concept_similarity(aconcept, tconcept, term_extras.get(tid))
            pairs.append(
                ConceptPairDiagnostic(
                    truth_concept_id=tid,
                    truth_kind=tconcept.kind,
                    truth_label=_concept_label(tconcept, True),
                    truth_labels=_concept_labels(tconcept, True),
                    agent_concept_id=acid,
                    agent_kind=aconcept.kind,
                    agent_label=_concept_label(aconcept, False),
                    exact_label_match=path is not None,
                    exact_label_match_path=path,
                    lexical_similarity_score=similarity,
                    threshold=_CONCEPT_MATCH_THRESHOLD,
                    eligible=similarity >= _CONCEPT_MATCH_THRESHOLD,
                    selected_mapping=agent_to_truth.get(acid) == tid,
                )
            )

    selected: list[ConceptMappingDiagnostic] = []
    for acid, tid in sorted(agent_to_truth.items()):
        aconcept = agent.concepts.get(acid)
        tconcept = truth.concepts.get(tid)
        if aconcept is None or tconcept is None:
            continue
        selected.append(
            ConceptMappingDiagnostic(
                agent_concept_id=acid,
                truth_concept_id=tid,
                agent_kind=aconcept.kind,
                truth_kind=tconcept.kind,
                agent_label=_concept_label(aconcept, False),
                truth_label=_concept_label(tconcept, True),
                exact_label_match_path=_exact_label_match_path(
                    aconcept, tconcept, term_extras.get(tid)
                ),
                lexical_similarity_score=_concept_similarity(
                    aconcept, tconcept, term_extras.get(tid)
                ),
            )
        )

    unmatched_truth = [
        ConceptSummary(
            concept_id=tid,
            kind=truth.concepts[tid].kind,
            label=_concept_label(truth.concepts[tid], True),
            labels=_concept_labels(truth.concepts[tid], True),
        )
        for tid in expected
        if tid not in set(agent_to_truth.values()) and tid in truth.concepts
    ]
    unmatched_agent = [
        ConceptSummary(
            concept_id=acid,
            kind=agent.concepts[acid].kind,
            label=_concept_label(agent.concepts[acid], False),
            labels=_concept_labels(agent.concepts[acid], False),
        )
        for acid in attempted
        if acid not in agent_to_truth and acid in agent.concepts
    ]
    return ConceptDiagnostics(
        threshold=_CONCEPT_MATCH_THRESHOLD,
        expected_truth_concept_ids=expected,
        attempted_agent_concept_ids=attempted,
        agent_to_truth={acid: agent_to_truth[acid] for acid in sorted(agent_to_truth)},
        candidate_pairs=pairs,
        selected_mappings=selected,
        unmatched_truth_concepts=unmatched_truth,
        unmatched_agent_concepts=unmatched_agent,
    )


def _agent_state(value) -> str:
    if value is _NO_VALUE:
        return "missing"
    if value is None or is_unset(value):
        return "unset"
    if is_absent(value):
        return "absent"
    if is_dont_know(value):
        return "dont_know"
    if isinstance(value, ConceptRef):
        return "value" if value.asserted else "value_unasserted"
    if isinstance(value, list):
        return "value" if value else "value_empty"
    return type(value).__name__


def _truth_state(value) -> str:
    if isinstance(value, ConceptRef):
        return "value"
    if isinstance(value, list) and value:
        return "value"
    return "absent"


def _concept_for_ref(ref, concepts):
    if isinstance(ref, ConceptRef):
        return concepts.get(ref.concept_id)
    return None


def _reason_for_epistemic_states(truth_state: str, agent_state: str) -> str:
    if truth_state == "value":
        return {
            "unset": "truth_value_agent_unset",
            "absent": "truth_value_agent_absent",
            "dont_know": "truth_value_agent_dont_know",
            "missing": "unmatched_node",
            "value_unasserted": "truth_value_agent_unasserted",
        }.get(agent_state, "wrong_concept")
    return {
        "absent": "truth_absent_agent_absent",
        "unset": "truth_absent_agent_unset",
        "dont_know": "truth_absent_agent_dont_know",
        "missing": "unmatched_node",
    }.get(agent_state, "truth_absent_agent_value")


def _slot_diagnostic(
    property_name: str,
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
    agent_concepts,
    truth_concepts,
    *,
    unmatched_reason: Optional[str] = None,
) -> SlotDiagnostic:
    """Explain one scalar/list score using the unchanged scoring helpers."""
    slot_kind = "list" if property_name in ("reads", "writes") else "scalar"
    truth_refs = list(truth_value) if isinstance(truth_value, list) else [truth_value]
    truth_refs = [ref for ref in truth_refs if isinstance(ref, ConceptRef)]
    agent_refs = list(agent_value) if isinstance(agent_value, list) else [agent_value]
    agent_refs = [ref for ref in agent_refs if isinstance(ref, ConceptRef)]
    truth_ids = sorted({ref.concept_id for ref in truth_refs})
    agent_ids = sorted({ref.concept_id for ref in agent_refs})
    mapped_ids = sorted(
        {
            agent_to_truth[ref.concept_id]
            for ref in agent_refs
            if ref.asserted and ref.concept_id in agent_to_truth
        }
    )
    truth_state = _truth_state(truth_value)
    agent_state = _agent_state(agent_value)
    truth_ref = truth_refs[0] if slot_kind == "scalar" and truth_refs else None
    agent_ref = agent_refs[0] if slot_kind == "scalar" and agent_refs else None
    truth_concept = _concept_for_ref(truth_ref, truth_concepts)
    agent_concept = _concept_for_ref(agent_ref, agent_concepts)

    if unmatched_reason is not None:
        return SlotDiagnostic(
            property=property_name,
            slot_kind=slot_kind,
            truth_state=truth_state,
            agent_state="missing",
            truth_concept_id=truth_ref.concept_id if truth_ref else None,
            truth_concept_ids=truth_ids,
            truth_label=_concept_label(truth_concept, True),
            truth_labels=_concept_labels(truth_concept, True),
            matched=False,
            score_contribution=0.0,
            reason=unmatched_reason,
            reason_codes=[unmatched_reason],
            missing_truth_concept_ids=truth_ids,
        )

    if slot_kind == "list":
        score, _unsupported = _score_list_slot(agent_value, truth_value, agent_to_truth)
        expected = set(truth_ids)
        claimed = {
            agent_to_truth[ref.concept_id]
            for ref in agent_refs
            if ref.asserted and ref.concept_id in agent_to_truth
        }
        missing = sorted(expected - claimed)
        extra_refs = [
            ref
            for ref in agent_refs
            if ref.asserted and agent_to_truth.get(ref.concept_id) not in expected
        ]
        extra_ids = sorted({ref.concept_id for ref in extra_refs})
        items: list[SlotItemDiagnostic] = []
        for tid in sorted(expected):
            tconcept = truth_concepts.get(tid)
            items.append(
                SlotItemDiagnostic(
                    item_type="truth",
                    truth_concept_id=tid,
                    truth_label=_concept_label(tconcept, True),
                    matched=tid in claimed,
                    reason="correct_value" if tid in claimed else "missing_list_item",
                )
            )
        for index, ref in sorted(
            enumerate(agent_refs), key=lambda pair: (pair[1].concept_id, pair[0])
        ):
            mapped = agent_to_truth.get(ref.concept_id)
            aconcept = agent_concepts.get(ref.concept_id)
            matched = bool(ref.asserted and mapped in expected)
            item_reason = "correct_value" if matched else "extra_list_item"
            if not ref.asserted:
                item_reason = "truth_value_agent_unasserted"
            items.append(
                SlotItemDiagnostic(
                    item_type="agent",
                    agent_concept_id=ref.concept_id,
                    mapped_truth_concept_id=mapped,
                    agent_label=_concept_label(aconcept, False),
                    truth_label=(
                        _concept_label(truth_concepts.get(mapped), True)
                        if mapped is not None
                        else None
                    ),
                    matched=matched,
                    reason=item_reason,
                )
            )

        if not truth_ids:
            if score == 1.0:
                reason = "truth_absent_agent_absent"
                codes = [reason]
            elif extra_ids:
                reason = "extra_list_item"
                codes = [reason]
            else:
                reason = _reason_for_epistemic_states(truth_state, agent_state)
                codes = [reason]
        elif not isinstance(agent_value, list):
            reason = _reason_for_epistemic_states(truth_state, agent_state)
            codes = [reason, "missing_list_item"]
        elif score == 1.0 and not missing and not extra_ids:
            reason = "correct_value"
            codes = [reason]
        else:
            codes = []
            if missing:
                codes.append("missing_list_item")
            if extra_ids:
                codes.append("extra_list_item")
            if not codes:
                codes.append(_reason_for_epistemic_states(truth_state, agent_state))
            if len(codes) > 1 and set(codes) == {
                "missing_list_item",
                "extra_list_item",
            }:
                reason = "missing_and_extra_list_items"
            else:
                reason = codes[0]
        if extra_ids and any(
            agent_to_truth.get(ref.concept_id) is None for ref in extra_refs
        ):
            codes.append("wrong_concept")
        return SlotDiagnostic(
            property=property_name,
            slot_kind=slot_kind,
            truth_state=truth_state,
            agent_state=agent_state,
            truth_concept_ids=truth_ids,
            agent_concept_ids=agent_ids,
            mapped_truth_concept_ids=mapped_ids,
            truth_labels=[
                label
                for tid in truth_ids
                for label in _concept_labels(truth_concepts.get(tid), True)
            ],
            agent_labels=[
                label
                for aid in agent_ids
                for label in _concept_labels(agent_concepts.get(aid), False)
            ],
            matched=score == 1.0,
            score_contribution=score,
            reason=reason,
            reason_codes=list(dict.fromkeys(codes)),
            missing_truth_concept_ids=missing,
            extra_agent_concept_ids=extra_ids,
            items=items,
        )

    score = _score_scalar_slot(agent_value, truth_value, agent_to_truth)
    if score == 1:
        reason = (
            "truth_absent_agent_absent"
            if truth_state == "absent" and agent_state == "absent"
            else "correct_value"
        )
        codes = [reason]
    elif isinstance(truth_value, ConceptRef) and isinstance(agent_value, ConceptRef):
        reason = "wrong_concept"
        codes = [reason]
    else:
        reason = _reason_for_epistemic_states(truth_state, agent_state)
        codes = [reason]
        if isinstance(agent_value, ConceptRef) and truth_state == "absent":
            codes.append("wrong_concept")
    return SlotDiagnostic(
        property=property_name,
        slot_kind=slot_kind,
        truth_state=truth_state,
        agent_state=agent_state,
        truth_concept_id=truth_ref.concept_id if truth_ref else None,
        agent_concept_id=agent_ref.concept_id if agent_ref else None,
        truth_concept_ids=truth_ids,
        agent_concept_ids=agent_ids,
        mapped_truth_concept_ids=mapped_ids,
        truth_label=_concept_label(truth_concept, True),
        agent_label=_concept_label(agent_concept, False),
        truth_labels=_concept_labels(truth_concept, True),
        agent_labels=_concept_labels(agent_concept, False),
        matched=score == 1,
        score_contribution=score,
        reason=reason,
        reason_codes=list(dict.fromkeys(codes)),
    )


def _build_evaluation_diagnostics(
    agent: AgentGraph,
    target,
    term_extras: dict[str, list[str]],
    agent_to_truth: dict[str, str],
    node_mapping: dict[str, str],
    edge_mapping: dict[str, str],
) -> EvaluationDiagnostics:
    concept_trace = _concept_diagnostics(agent, target, term_extras, agent_to_truth)
    usage_trace = build_usage_alignment_diagnostics(
        agent,
        target,
        node_mapping=node_mapping,
        edge_mapping=edge_mapping,
        current_agent_to_truth=agent_to_truth,
        lexical_pairs=concept_trace.candidate_pairs,
    )
    try:
        # The production and usage mappings are comparison-only inputs.  The
        # joint builder performs its structural search before copying them into
        # the private diagnostic section and never uses them as constraints.
        joint_trace = build_joint_structural_alignment_diagnostics(
            agent,
            target,
            production_node_to_truth=node_mapping,
            production_concept_to_truth=agent_to_truth,
            usage_concept_to_truth=usage_trace.usage_agent_to_truth,
        )
    except Exception as exc:
        # Diagnostics must never turn a matcher experiment failure into a
        # production evaluation failure or alter any score/pass field.
        joint_trace = JointStructuralAlignmentDiagnostics(
            status="error",
            error_type=type(exc).__name__,
        )
    truth_to_agent = {tid: aid for aid, tid in node_mapping.items()}
    nodes: list[NodeDiagnostic] = []
    for tid in sorted(target.nodes):
        aid = truth_to_agent.get(tid)
        truth_node = target.nodes[tid]
        slots: dict[str, SlotDiagnostic] = {}
        for output_prop, score_prop in _DIAGNOSTIC_NODE_PROPS:
            truth_value = _prop_value(truth_node, score_prop)
            if aid is None:
                slots[output_prop] = _slot_diagnostic(
                    output_prop,
                    _NO_VALUE,
                    truth_value,
                    agent_to_truth,
                    agent.concepts,
                    target.concepts,
                    unmatched_reason="unmatched_node",
                )
            else:
                slots[output_prop] = _slot_diagnostic(
                    output_prop,
                    agent.nodes[aid].slot_value(score_prop),
                    truth_value,
                    agent_to_truth,
                    agent.concepts,
                    target.concepts,
                )
        nodes.append(
            NodeDiagnostic(
                truth_node_id=tid,
                agent_node_id=aid,
                matched=aid is not None,
                reason="matched_node" if aid is not None else "unmatched_node",
                slots=slots,
            )
        )

    edges: list[EdgeDiagnostic] = []
    for tid in sorted(target.edges):
        truth_edge = target.edges[tid]
        # Match the same first Agent edge that the existing condition score
        # uses.  This is explanatory only, but keeps the trace auditable.
        aid = next(
            (eid for eid in agent.edges if edge_mapping.get(eid) == tid),
            None,
        )
        if aid is not None:
            agent_edge = agent.edges[aid]
            from_match = node_mapping.get(agent_edge.from_node) == truth_edge.from_node
            to_match = node_mapping.get(agent_edge.to_node) == truth_edge.to_node
            condition = _slot_diagnostic(
                "condition",
                agent_edge.condition,
                truth_edge.condition,
                agent_to_truth,
                agent.concepts,
                target.concepts,
            )
            structural_match = from_match and to_match
        else:
            agent_edge = None
            from_match = to_match = structural_match = False
            condition = _slot_diagnostic(
                "condition",
                _NO_VALUE,
                truth_edge.condition,
                agent_to_truth,
                agent.concepts,
                target.concepts,
                unmatched_reason="unmatched_edge",
            )
        edges.append(
            EdgeDiagnostic(
                truth_edge_id=tid,
                agent_edge_id=aid,
                matched=structural_match,
                reason="matched_edge" if structural_match else "unmatched_edge",
                truth_from_node=truth_edge.from_node,
                truth_to_node=truth_edge.to_node,
                agent_from_node=(agent_edge.from_node if agent_edge else None),
                agent_to_node=(agent_edge.to_node if agent_edge else None),
                from_node_match=from_match,
                to_node_match=to_match,
                structural_match=structural_match,
                condition=condition,
            )
        )

    return EvaluationDiagnostics(
        node_diagnostics=nodes,
        unmatched_agent_nodes=sorted(
            aid for aid in agent.nodes if aid not in node_mapping
        ),
        edge_diagnostics=edges,
        unmatched_agent_edges=sorted(
            eid for eid in agent.edges if eid not in edge_mapping
        ),
        concepts=concept_trace,
        usage_alignment=usage_trace,
        joint_structural_alignment=joint_trace,
    )


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate(
    db: InterviewDB,
    knowledge,
    spec: EvaluationSpec,
    stakeholder=None,
    *,
    truth=None,
    annotations=None,
    alignments=None,
    terminology=None,
) -> EvaluationResult:
    """Evaluate the inferred AgentGraph against the Truth (content-based).

    ``truth`` (the BusinessProcessGraph) is the primary target. ``knowledge``
    (StakeholderKnowledge) and ``stakeholder`` are used only for the
    informational coverage metric. The provenance ledgers are diagnostics and
    never gate quality.
    """
    raw_target = truth if truth is not None else knowledge.graph
    # Canonical SOURCE/SINK and boundary edges are structural-only.  The
    # scoring target is an immutable business projection; the full contract is
    # retained separately in diagnostics.
    target = business_graph_projection(raw_target)
    canonical_contract = _canonical_contract_diagnostic(raw_target, knowledge)
    agent = db.graph if db.graph is not None else AgentGraph()
    protocol = db.interview_complete
    graph_created = len(agent.nodes) > 0
    graph_valid = agent.is_valid

    # Stakeholder-local concept terms (e.g. Japanese terms for the JA locale)
    # enrich the Truth concept signatures so an agent vocabulary in the
    # stakeholder's language still matches. Diagnostic vocabulary, never
    # provenance.
    term_extras: dict[str, list[str]] = {}
    if knowledge is not None and getattr(knowledge, "concepts", None):
        for c in knowledge.concepts.values():
            t = getattr(c, "terms", None)
            if isinstance(t, list) and t:
                term_extras.setdefault(c.truth_concept_id, list(t))

    # concept alignment first (property + concept scoring need the mapping)
    concept_recall, concept_precision, agent_to_truth = _align_concepts(
        agent, target, term_extras
    )

    mapping, edge_map = _map_nodes_and_edges(agent, target, agent_to_truth)

    target_node_ids = list(target.nodes)
    agent_node_ids = list(agent.nodes)
    matched_nodes = set(mapping.values())
    node_recall = len(matched_nodes) / len(target_node_ids) if target.nodes else 0.0
    node_precision = len(mapping) / len(agent_node_ids) if agent.nodes else 0.0
    fabricated_node_count = len(agent_node_ids) - len(mapping)

    target_edge_list = list(target.edges.values())
    agent_edge_list = list(agent.edges.values())
    matched_edges = set(edge_map.values())
    edge_recall = (
        len(matched_edges) / len(target_edge_list) if list(target.edges) else 1.0
    )
    edge_precision = len(edge_map) / len(agent_edge_list) if agent.edges else 0.0
    fabricated_edge_count = len(agent_edge_list) - len(edge_map)

    agent_start_ids = set(getattr(agent, "start_node_ids", []))
    if not agent_start_ids and agent.start_node_id is not None:
        agent_start_ids = {agent.start_node_id}
    mapped_agent_starts = {
        mapping[node_id] for node_id in agent_start_ids if node_id in mapping
    }
    target_entries = set(business_entry_node_ids(raw_target))
    start_correct = mapped_agent_starts == target_entries
    agent_ends = {mapping.get(eid) for eid in agent.end_node_ids if mapping.get(eid)}
    target_ends = set(target.end_node_ids)
    end_recall = (
        len(agent_ends & target_ends) / len(target_ends) if target_ends else 1.0
    )
    end_precision = (
        len(agent_ends & target_ends) / len(agent_ends) if agent_ends else 0.0
    )

    hits = {p: 0.0 for p in _NODE_PROPS}
    unsupported = 0
    for anid, tnid in mapping.items():
        anode = agent.nodes[anid]
        tnode = target.nodes[tnid]
        for prop in _NODE_PROPS:
            aval = anode.slot_value(prop)
            tval = _prop_value(tnode, prop)
            if prop in ("reads", "writes"):
                s, un = _score_list_slot(aval, tval, agent_to_truth)
                hits[prop] += s
                unsupported += un
            else:
                hits[prop] += _score_scalar_slot(aval, tval, agent_to_truth)
    nm = len(mapping) or 1
    activity_correctness = hits["activity"] / nm
    actor_correctness = hits["actor"] / nm
    system_correctness = hits["system"] / nm
    read_correctness = hits["reads"] / nm
    write_correctness = hits["writes"] / nm
    rationale_correctness = hits["rationale"] / nm

    cond_hits = cond_total = 0
    for tid, te in target.edges.items():
        cond_total += 1
        ae = next(
            (edge for eid, edge in agent.edges.items() if edge_map.get(eid) == te.id),
            None,
        )
        if ae is None:
            continue
        if _score_scalar_slot(ae.condition, te.condition, agent_to_truth):
            cond_hits += 1
    condition_correctness = cond_hits / cond_total if cond_total else 1.0

    concept_correctness = concept_recall * concept_precision
    glossary_complete = bool(concept_recall == 1.0 and concept_precision == 1.0)
    # No validation/grounding lifecycle: concept status is an Agent belief
    # record that nothing gates on. glossary_complete = reconstruction.

    authentic_ids: set[str] = set()
    invalid_source = 0
    for o in db.observations:
        m = db.messages[o.turn] if 0 <= o.turn < len(db.messages) else None
        if (
            m is not None
            and m.get("role") == "user"
            and (m.get("content") or "") == (o.text or "")
        ):
            authentic_ids.add(o.id)
        else:
            invalid_source += 1
    authentic_count = len(authentic_ids)
    orphan_count = sum(1 for o in db.observations if o.id not in _all_ref_obs(agent))
    invalid_refs, invalid_obs_refs, node_cov, ref_cov, edge_cov = _evidence_metrics(
        db, agent
    )

    provenance_authenticity_pass = bool(
        invalid_refs == 0 and invalid_obs_refs == 0 and invalid_source == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and node_cov == 1.0
        and ref_cov == 1.0
        and edge_cov == 1.0
    )

    structural_pass = bool(
        graph_created
        and graph_valid
        and node_recall == 1.0
        and node_precision == 1.0
        and edge_recall == 1.0
        and edge_precision == 1.0
        and start_correct
        and end_recall == 1.0
        and end_precision == 1.0
        and activity_correctness == 1.0
        and actor_correctness == 1.0
        and system_correctness == 1.0
        and read_correctness == 1.0
        and write_correctness == 1.0
        and rationale_correctness == 1.0
        and condition_correctness == 1.0
        and concept_recall == 1.0
        and concept_precision == 1.0
        and concept_correctness == 1.0
        and glossary_complete
    )
    protocol_pass = protocol
    reconstruction_pass = structural_pass
    # quality = reconstruction ONLY; provenance is never a hard gate
    quality_pass = reconstruction_pass

    # Build private explanatory metadata only after all existing score fields
    # have been computed. The trace calls the same score helpers but does not
    # feed any value back into the score or pass/fail calculations.
    diagnostics = _build_evaluation_diagnostics(
        agent,
        target,
        term_extras,
        agent_to_truth,
        mapping,
        edge_map,
    )
    diagnostics.canonical_contract = canonical_contract

    return EvaluationResult(
        protocol_completed=protocol,
        graph_created=graph_created,
        graph_valid=graph_valid,
        node_recall=node_recall,
        node_precision=node_precision,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        start_correct=start_correct,
        end_recall=end_recall,
        end_precision=end_precision,
        activity_correctness=activity_correctness,
        actor_correctness=actor_correctness,
        system_correctness=system_correctness,
        read_correctness=read_correctness,
        write_correctness=write_correctness,
        rationale_correctness=rationale_correctness,
        condition_correctness=condition_correctness,
        concept_correctness=concept_correctness,
        concept_recall=concept_recall,
        concept_precision=concept_precision,
        unsupported_ref_count=unsupported,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_complete=glossary_complete,
        node_evidence_coverage=node_cov,
        ref_evidence_coverage=ref_cov,
        edge_evidence_coverage=edge_cov,
        invalid_evidence_ref_count=invalid_refs,
        ambiguous_evidence_ref_count=0,
        marker_evidence_errors_surrogate=0,
        invalid_observation_reference_count=invalid_obs_refs,
        authentic_observation_count=authentic_count,
        invalid_observation_source_count=invalid_source,
        orphan_observation_count=orphan_count,
        provenance_authenticity_pass=provenance_authenticity_pass,
        evidence_pass=evidence_pass,
        reconstruction_pass=reconstruction_pass,
        structural_pass=structural_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
        knowledge_coverage=_knowledge_coverage(truth, knowledge),
        diagnostics=diagnostics,
    )
