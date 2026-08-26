"""Pure Agent-to-Truth alignment and shared aligned-graph comparison.

Alignment adapters own identity semantics.  ``compare_aligned_graphs`` only
consumes explicit mappings, so Agent lexical matching and Stakeholder private-id
mapping can share score arithmetic without sharing epistemic models.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .graph import (
    AbsentType,
    AgentGraph,
    ConceptRef,
    business_edge_ids,
    business_node_ids,
    is_dont_know,
)

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")


def slot_value(node: Any, prop: str) -> Any:
    """Read the same business slot from Truth, Stakeholder, or Agent nodes."""
    attr = "necessity_rationale" if prop == "rationale" else prop
    return getattr(node, attr)


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


def _exact_label_match_path(
    agent_concept, truth_concept, extra_terms=None
) -> Optional[str]:
    """Return which exact-label branch selected a pair."""
    agent_key = _label_key(getattr(agent_concept, "display_label", None))
    if not agent_key:
        return None
    for term in truth_concept.canonical_terms or []:
        if agent_key == _label_key(term):
            return "canonical_term"
    for term in extra_terms or ():
        if agent_key == _label_key(term):
            return "stakeholder_extra_term"
    return None


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
        agent_node_id: _node_signature(
            node, agent.concepts, is_truth=False, agent_to_truth=agent_to_truth
        )
        for agent_node_id, node in agent.nodes.items()
    }
    truth_sigs = {
        tnid: _node_signature(node, truth.concepts, is_truth=True)
        for tnid, node in truth.nodes.items()
    }
    weights: dict[tuple[str, str], float] = {}
    for agent_node_id in agent_sigs:
        a = agent_sigs[agent_node_id]
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
                weights[(agent_node_id, tnid)] = s
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
    *,
    known_absent=None,
) -> int:
    """Epistemic-aware scalar slot score (1 correct / 0 otherwise).

    The Truth-side rule is shared by Agent and Stakeholder comparisons:
    ``ConceptRef`` requires the matching Truth concept, while canonical
    absence requires an explicit known-absence state.  Agent comparisons use
    ``ABSENT`` as that state; StakeholderKnowledge comparisons pass
    ``known_absent=lambda value: value is None``.  ``DONT_KNOW`` is never a
    correct answer to a Truth value and never a lucky match for a Truth
    absence.
    """
    tcid = _truth_scalar_value(truth_value)
    if tcid is not None:
        if not isinstance(agent_value, ConceptRef) or not agent_value.asserted:
            return 0
        return 1 if agent_to_truth.get(agent_value.concept_id) == tcid else 0
    # Truth absent: explicit candidate known-absence is the only correct state.
    if isinstance(agent_value, AbsentType):
        return 1
    if known_absent is not None and known_absent(agent_value):
        return 1
    return 0


def _score_list_slot(
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
    *,
    known_absent=None,
) -> tuple[float, int]:
    """Shared reads/writes completeness score, plus unsupported count.

    Truth lists use recall*precision over the normal element set.  Truth
    ``None`` / known-empty requires an explicit candidate known-absence state
    (Agent ``ABSENT`` or Stakeholder ``None``); ``UNSET`` and ``DONT_KNOW`` are
    incomplete and asserted list refs are fabricated.
    """
    expected: set[str] = (
        {ref.concept_id for ref in truth_value}
        if isinstance(truth_value, list)
        else set()
    )
    if not expected:
        if isinstance(agent_value, AbsentType):
            return 1.0, 0
        if known_absent is not None and known_absent(agent_value):
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


@dataclass(frozen=True)
class GraphAlignment:
    """Explicit candidate-local to Truth mappings plus concept completeness."""

    concept_to_truth: dict[str, str]
    node_to_truth: dict[str, str]
    edge_to_truth: dict[str, str]
    concept_recall: float
    concept_precision: float


@dataclass(frozen=True)
class AlignedGraphComparison:
    """Shared score components after model-specific identity alignment."""

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
    glossary_complete: bool


def align_agent_to_truth(
    agent: AgentGraph,
    truth: Any,
    terminology_terms: Optional[dict[str, list[str]]] = None,
) -> GraphAlignment:
    """Align Agent concepts lexically, then nodes and edges structurally."""
    recall, precision, concepts = _align_concepts(agent, truth, terminology_terms)
    nodes, edges = _map_nodes_and_edges(agent, truth, concepts)
    return GraphAlignment(
        concept_to_truth=concepts,
        node_to_truth=nodes,
        edge_to_truth=edges,
        concept_recall=recall,
        concept_precision=precision,
    )


def compare_aligned_graphs(
    *,
    candidate: Any,
    truth: Any,
    alignment: GraphAlignment,
    candidate_node_ids: list[str],
    candidate_edge_ids: list[str],
    candidate_start_node_ids: set[str],
    candidate_end_node_ids: set[str],
    truth_entry_node_ids: set[str],
    graph_valid: bool,
    known_absent: Optional[Callable[[Any], bool]] = None,
    empty_node_recall: float = 0.0,
) -> AlignedGraphComparison:
    """Compare explicitly aligned business graphs using the current contract.

    Candidate identity and epistemic interpretation are supplied by the caller.
    This function intentionally preserves the historical denominator and first
    mapped-edge condition semantics used by the benchmark.
    """
    node_mapping = alignment.node_to_truth
    edge_mapping = alignment.edge_to_truth
    concept_mapping = alignment.concept_to_truth

    target_node_ids = list(truth.nodes)
    matched_nodes = set(node_mapping.values())
    node_recall = (
        len(matched_nodes) / len(target_node_ids) if truth.nodes else empty_node_recall
    )
    node_precision = (
        len(node_mapping) / len(candidate_node_ids) if candidate_node_ids else 0.0
    )
    fabricated_node_count = len(candidate_node_ids) - len(node_mapping)

    target_edge_ids = list(truth.edges)
    matched_edges = set(edge_mapping.values())
    edge_recall = len(matched_edges) / len(target_edge_ids) if target_edge_ids else 1.0
    edge_precision = (
        len(edge_mapping) / len(candidate_edge_ids) if candidate_edge_ids else 0.0
    )
    fabricated_edge_count = len(candidate_edge_ids) - len(edge_mapping)

    mapped_starts = {
        node_mapping[node_id]
        for node_id in candidate_start_node_ids
        if node_id in node_mapping
    }
    start_correct = mapped_starts == truth_entry_node_ids
    mapped_ends = {
        node_mapping[node_id]
        for node_id in candidate_end_node_ids
        if node_id in node_mapping
    }
    truth_ends = set(truth.end_node_ids)
    end_recall = len(mapped_ends & truth_ends) / len(truth_ends) if truth_ends else 1.0
    end_precision = (
        len(mapped_ends & truth_ends) / len(mapped_ends) if mapped_ends else 0.0
    )

    hits = {prop: 0.0 for prop in _NODE_PROPS}
    unsupported = 0
    for candidate_id, truth_id in node_mapping.items():
        candidate_node = candidate.nodes[candidate_id]
        truth_node = truth.nodes[truth_id]
        for prop in _NODE_PROPS:
            candidate_value = slot_value(candidate_node, prop)
            truth_value = slot_value(truth_node, prop)
            if prop in ("reads", "writes"):
                score, unsupported_here = _score_list_slot(
                    candidate_value,
                    truth_value,
                    concept_mapping,
                    known_absent=known_absent,
                )
                hits[prop] += score
                unsupported += unsupported_here
            else:
                hits[prop] += _score_scalar_slot(
                    candidate_value,
                    truth_value,
                    concept_mapping,
                    known_absent=known_absent,
                )
    denominator = len(node_mapping) or 1

    condition_hits = 0
    for truth_edge_id, truth_edge in truth.edges.items():
        candidate_edge = next(
            (
                edge
                for edge_id, edge in candidate.edges.items()
                if edge_mapping.get(edge_id) == truth_edge_id
            ),
            None,
        )
        if candidate_edge is None:
            continue
        if _score_scalar_slot(
            candidate_edge.condition,
            truth_edge.condition,
            concept_mapping,
            known_absent=known_absent,
        ):
            condition_hits += 1
    condition_correctness = (
        condition_hits / len(target_edge_ids) if target_edge_ids else 1.0
    )

    concept_correctness = alignment.concept_recall * alignment.concept_precision
    glossary_complete = bool(
        alignment.concept_recall == 1.0 and alignment.concept_precision == 1.0
    )
    return AlignedGraphComparison(
        graph_created=bool(candidate_node_ids),
        graph_valid=graph_valid,
        node_recall=node_recall,
        node_precision=node_precision,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        start_correct=start_correct,
        end_recall=end_recall,
        end_precision=end_precision,
        activity_correctness=hits["activity"] / denominator,
        actor_correctness=hits["actor"] / denominator,
        system_correctness=hits["system"] / denominator,
        read_correctness=hits["reads"] / denominator,
        write_correctness=hits["writes"] / denominator,
        rationale_correctness=hits["rationale"] / denominator,
        condition_correctness=condition_correctness,
        concept_correctness=concept_correctness,
        concept_recall=alignment.concept_recall,
        concept_precision=alignment.concept_precision,
        unsupported_ref_count=unsupported,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_complete=glossary_complete,
    )


def reconstruction_complete(metrics: AlignedGraphComparison) -> bool:
    """Current structural/quality completeness predicate shared by both lanes."""
    return bool(
        metrics.graph_created
        and metrics.graph_valid
        and metrics.node_recall == 1.0
        and metrics.node_precision == 1.0
        and metrics.edge_recall == 1.0
        and metrics.edge_precision == 1.0
        and metrics.start_correct
        and metrics.end_recall == 1.0
        and metrics.end_precision == 1.0
        and metrics.activity_correctness == 1.0
        and metrics.actor_correctness == 1.0
        and metrics.system_correctness == 1.0
        and metrics.read_correctness == 1.0
        and metrics.write_correctness == 1.0
        and metrics.rationale_correctness == 1.0
        and metrics.condition_correctness == 1.0
        and metrics.concept_recall == 1.0
        and metrics.concept_precision == 1.0
        and metrics.concept_correctness == 1.0
        and metrics.glossary_complete
    )


def knowledge_coverage(truth: Any, knowledge: Any) -> float:
    """Share of Truth addresses known by one StakeholderKnowledge view."""
    graph = knowledge.graph if knowledge is not None else None
    if truth is None or graph is None:
        return 0.0
    node_truth_to_local = {
        truth_id: local for local, truth_id in graph.node_truth_ids.items()
    }
    edge_truth_to_local = {
        truth_id: local for local, truth_id in graph.edge_truth_ids.items()
    }
    concept_truth_to_local = {
        concept.truth_concept_id: local for local, concept in graph.concepts.items()
    }
    total = known = 0
    for node_id in business_node_ids(truth):
        truth_node = truth.nodes[node_id]
        local_id = node_truth_to_local.get(node_id)
        stakeholder_node = graph.nodes.get(local_id) if local_id is not None else None
        total += 1
        if stakeholder_node is not None:
            known += 1
        for prop in _NODE_PROPS:
            total += 1
            if stakeholder_node is None:
                if prop in ("reads", "writes"):
                    total += len(getattr(truth_node, prop) or [])
                continue
            value = slot_value(stakeholder_node, prop)
            if not is_dont_know(value):
                known += 1
            if prop in ("reads", "writes"):
                for ref in getattr(truth_node, prop) or []:
                    total += 1
                    if is_dont_know(value):
                        continue
                    if value is None:
                        known += 1
                    elif isinstance(value, list) and concept_truth_to_local.get(
                        ref.concept_id
                    ) in {item.concept_id for item in value}:
                        known += 1
    for edge_id in business_edge_ids(truth):
        local_id = edge_truth_to_local.get(edge_id)
        stakeholder_edge = graph.edges.get(local_id) if local_id is not None else None
        total += 1
        if stakeholder_edge is not None:
            known += 1
        total += 1
        if stakeholder_edge is not None and not is_dont_know(
            stakeholder_edge.condition
        ):
            known += 1
    return known / total if total else 0.0


def truth_referenced_concept_ids(truth: Any) -> set[str]:
    return _truth_referenced_concept_ids(truth)


def agent_referenced_concept_ids(agent: AgentGraph) -> set[str]:
    return _agent_referenced_concept_ids(agent)
