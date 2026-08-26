"""Evaluator-private usage-based concept-alignment diagnostics.

This module deliberately does not participate in production scoring.  It
compares graph addresses after the evaluator's existing node/edge mapping has
been applied, so its conclusion is explicitly conditional on that mapping.
The experimental assignment never reads labels, descriptions, translations,
embeddings, or LLM output.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import AgentGraph, ConceptRef

_USAGE_NODE_PROPERTIES = (
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "rationale",
)


class UsageConceptPairDiagnostic(BaseModel):
    """Usage-only evidence for one same-kind Truth/Agent concept pair."""

    truth_concept_id: str
    truth_kind: str
    truth_label: Optional[str] = None
    truth_labels: list[str] = Field(default_factory=list)
    agent_concept_id: str
    agent_kind: str
    agent_label: Optional[str] = None
    truth_usage_addresses: list[str] = Field(default_factory=list)
    agent_usage_addresses: list[str] = Field(default_factory=list)
    agent_unmapped_usage_addresses: list[str] = Field(default_factory=list)
    usages_only_in_truth: list[str] = Field(default_factory=list)
    usages_only_in_agent: list[str] = Field(default_factory=list)
    overlap_count: int = 0
    truth_usage_count: int = 0
    agent_usage_count: int = 0
    usage_recall: float = 0.0
    usage_precision: float = 0.0
    usage_f1: float = 0.0
    usage_jaccard: float = 0.0
    exact_usage_match: bool = False
    truth_usage_strict_superset_of_agent: bool = False
    agent_usage_strict_superset_of_truth: bool = False
    broader_narrower_relation: str = "none"


class UsageAssignmentDiagnostic(BaseModel):
    """One deterministic internal usage assignment.

    ``ambiguous`` is important: the ids are retained only so the artifact is
    reproducible; the diagnostic must not imply that this pair's identity was
    resolved when the usage signatures are indistinguishable.
    """

    agent_concept_id: str
    truth_concept_id: str
    kind: str
    usage_similarity_score: float = 0.0
    exact_usage_match: bool = False
    ambiguous: bool = False
    ambiguity_class_id: Optional[str] = None


class UsageAmbiguityClass(BaseModel):
    """An equivalence class whose usage evidence cannot resolve identity."""

    class_id: str
    kind: str
    truth_concept_ids: list[str] = Field(default_factory=list)
    agent_concept_ids: list[str] = Field(default_factory=list)
    usage_addresses: list[str] = Field(default_factory=list)
    reason: str = "identical_usage_signature"


class UsageComparisonDiagnostic(BaseModel):
    """Comparison of current lexical mapping and usage-only assignment."""

    agent_concept_id: str
    agent_kind: str
    agent_label: Optional[str] = None
    current_truth_concept_id: Optional[str] = None
    current_truth_label: Optional[str] = None
    usage_truth_concept_id: Optional[str] = None
    usage_truth_label: Optional[str] = None
    classification: str
    current_lexical_similarity_score: float = 0.0
    current_exact_label_match: bool = False
    current_exact_label_match_path: Optional[str] = None
    current_usage_recall: float = 0.0
    current_usage_precision: float = 0.0
    current_usage_f1: float = 0.0
    current_usage_jaccard: float = 0.0
    current_exact_usage_match: bool = False
    usage_similarity_score: float = 0.0
    usage_exact_usage_match: bool = False
    usage_ambiguous: bool = False
    ambiguity_class_id: Optional[str] = None
    labels_differ: bool = False
    labels_agree_but_usage_does_not: bool = False
    usage_evidence_substantially_stronger: bool = False


class UsageKindSummary(BaseModel):
    """Compact usage experiment counts for one concept kind."""

    kind: str
    truth_referenced_concept_count: int = 0
    agent_referenced_concept_count: int = 0
    truth_concepts_with_usage_count: int = 0
    agent_concepts_with_usage_count: int = 0
    exact_usage_matches: int = 0
    partial_usage_matches: int = 0
    structurally_ambiguous_concepts: int = 0
    disagreements_with_current_mapping: int = 0
    usage_substantially_stronger_disagreements: int = 0
    insufficient_usage: int = 0


class UsageAlignmentDiagnostics(BaseModel):
    """Complete evaluator-private usage-based alignment experiment."""

    schema_version: str = "business_interview.usage_alignment.v1"
    method: str = "usage_alignment_conditioned_on_current_node_mapping"
    interpretation: str = (
        "Given the current node/edge correspondence, test whether graph usage "
        "is a better concept-identity signal than labels"
    )
    circularity_limitation: str = (
        "Node/edge correspondence is a scaffold that can itself depend partly "
        "on concept alignment; this is not a fully label-independent evaluator"
    )
    assignment_algorithm: str = (
        "per-kind one-to-one maximum-weight assignment; exact non-empty "
        "usage equality has priority, then usage F1"
    )
    assignment_uses_labels: bool = False
    node_mapping_scaffold_used: bool = True
    edge_mapping_scaffold_used: bool = True
    truth_referenced_concept_count: int = 0
    agent_referenced_concept_count: int = 0
    exact_usage_match_count: int = 0
    partial_usage_match_count: int = 0
    structurally_ambiguous_concept_count: int = 0
    disagreement_count: int = 0
    usage_substantially_stronger_disagreement_count: int = 0
    labels_differ_usage_agrees_count: int = 0
    labels_agree_usage_does_not_count: int = 0
    current_agent_to_truth: dict[str, str] = Field(default_factory=dict)
    usage_agent_to_truth: dict[str, str] = Field(default_factory=dict)
    candidate_pairs: list[UsageConceptPairDiagnostic] = Field(default_factory=list)
    assignments: list[UsageAssignmentDiagnostic] = Field(default_factory=list)
    ambiguity_classes: list[UsageAmbiguityClass] = Field(default_factory=list)
    comparisons: list[UsageComparisonDiagnostic] = Field(default_factory=list)
    kind_summaries: list[UsageKindSummary] = Field(default_factory=list)
    unmatched_truth_concept_ids: list[str] = Field(default_factory=list)
    unmatched_agent_concept_ids: list[str] = Field(default_factory=list)
    unmapped_agent_usage_addresses: dict[str, list[str]] = Field(default_factory=dict)


def _truth_referenced_concept_ids(truth) -> set[str]:
    ids: set[str] = set()
    for node in truth.nodes.values():
        for prop in _USAGE_NODE_PROPERTIES:
            for ref in node.asserted_refs(prop):
                ids.add(ref.concept_id)
    for edge in truth.edges.values():
        condition = edge.condition
        if isinstance(condition, ConceptRef) and condition.asserted:
            ids.add(condition.concept_id)
    return ids


def _agent_referenced_concept_ids(agent: AgentGraph) -> set[str]:
    return agent.referenced_concepts()


def _truth_usage_addresses(truth) -> dict[str, set[str]]:
    addresses: dict[str, set[str]] = defaultdict(set)
    for node_id in sorted(truth.nodes):
        node = truth.nodes[node_id]
        for prop in _USAGE_NODE_PROPERTIES:
            for ref in node.asserted_refs(prop):
                addresses[ref.concept_id].add(f"node:{node_id}:{prop}")
    for edge_id in sorted(truth.edges):
        condition = truth.edges[edge_id].condition
        if isinstance(condition, ConceptRef) and condition.asserted:
            addresses[condition.concept_id].add(f"edge:{edge_id}:condition")
    return addresses


def _agent_usage_addresses(
    agent: AgentGraph,
    node_mapping: Mapping[str, str],
    edge_mapping: Mapping[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return mapped and unmapped Agent usage addresses per concept.

    Only mapped addresses enter the comparison signature.  Unmapped addresses
    are retained separately so an empty mapped signature is not mistaken for
    proof that a concept was never used.
    """

    mapped: dict[str, set[str]] = defaultdict(set)
    unmapped: dict[str, set[str]] = defaultdict(set)
    for node_id in sorted(agent.nodes):
        node = agent.nodes[node_id]
        mapped_node_id = node_mapping.get(node_id)
        for prop in _USAGE_NODE_PROPERTIES:
            for ref in node.asserted_refs(prop):
                if mapped_node_id is None:
                    unmapped[ref.concept_id].add(f"node:{node_id}:{prop}")
                else:
                    mapped[ref.concept_id].add(f"node:{mapped_node_id}:{prop}")
    for edge_id in sorted(agent.edges):
        condition = agent.edges[edge_id].condition
        if not isinstance(condition, ConceptRef) or not condition.asserted:
            continue
        mapped_edge_id = edge_mapping.get(edge_id)
        if mapped_edge_id is None:
            unmapped[condition.concept_id].add(f"edge:{edge_id}:condition")
        else:
            mapped[condition.concept_id].add(f"edge:{mapped_edge_id}:condition")
    return mapped, unmapped


class _PairMetrics(TypedDict):
    overlap_count: int
    truth_usage_count: int
    agent_usage_count: int
    usage_recall: float
    usage_precision: float
    usage_f1: float
    usage_jaccard: float
    exact_usage_match: bool
    truth_usage_strict_superset_of_agent: bool
    agent_usage_strict_superset_of_truth: bool
    broader_narrower_relation: str


def _metrics(truth_addresses: set[str], agent_addresses: set[str]) -> _PairMetrics:
    truth_count = len(truth_addresses)
    agent_count = len(agent_addresses)
    overlap = len(truth_addresses & agent_addresses)
    recall = overlap / truth_count if truth_count else 0.0
    precision = overlap / agent_count if agent_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = truth_addresses | agent_addresses
    jaccard = overlap / len(union) if union else 0.0
    exact = bool(truth_addresses) and truth_addresses == agent_addresses
    truth_superset = truth_addresses > agent_addresses
    agent_superset = agent_addresses > truth_addresses
    if truth_addresses and truth_addresses == agent_addresses:
        relation = "none"
    elif truth_superset:
        relation = "truth_broader_agent_narrower"
    elif agent_superset:
        relation = "agent_broader_truth_narrower"
    elif truth_addresses and agent_addresses:
        relation = "incomparable"
    else:
        relation = "none"
    return {
        "overlap_count": overlap,
        "truth_usage_count": truth_count,
        "agent_usage_count": agent_count,
        "usage_recall": recall,
        "usage_precision": precision,
        "usage_f1": f1,
        "usage_jaccard": jaccard,
        "exact_usage_match": exact,
        "truth_usage_strict_superset_of_agent": truth_superset,
        "agent_usage_strict_superset_of_truth": agent_superset,
        "broader_narrower_relation": relation,
    }


def _signature_key(addresses: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(addresses))


def _max_weight_assignment(
    weights: Mapping[tuple[str, str], float],
    left: list[str],
    right: list[str],
) -> dict[str, str]:
    """Deterministic maximum-weight matching for positive usage weights.

    The implementation is intentionally local to this experiment.  Its only
    tie-breaker is sorted opaque concept ids; labels never enter the matrix.
    """

    n, m = len(left), len(right)
    if n == 0 or m == 0:
        return {}
    size = n + m
    big = 1e15
    cost = [[0.0] * (size + 1) for _ in range(size + 1)]
    row_idx = {value: index for index, value in enumerate(left, start=1)}
    col_idx = {value: index for index, value in enumerate(right, start=1)}
    allowed: set[tuple[int, int]] = set()
    for (left_id, right_id), weight in weights.items():
        if weight <= 0.0:
            continue
        i = row_idx[left_id]
        j = col_idx[right_id]
        cost[j][i] = -weight
        allowed.add((i, j))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (i, j) not in allowed:
                cost[j][i] = big

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    parent = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        parent[0] = i
        column = 0
        min_value = [1e18] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            row = parent[column]
            delta = 1e18
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                current = cost[candidate][row] - u[row] - v[candidate]
                if current < min_value[candidate]:
                    min_value[candidate] = current
                    way[candidate] = column
                if min_value[candidate] < delta:
                    delta = min_value[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[parent[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_value[candidate] -= delta
            column = next_column
            if parent[column] == 0:
                break
        while True:
            previous = way[column]
            parent[column] = parent[previous]
            column = previous
            if column == 0:
                break

    assignment: dict[str, str] = {}
    for column in range(1, m + 1):
        row = parent[column]
        if 1 <= row <= n and (row, column) in allowed:
            assignment[left[row - 1]] = right[column - 1]
    return assignment


def _concept_label(concept, *, truth: bool) -> Optional[str]:
    if concept is None:
        return None
    if truth:
        terms = list(getattr(concept, "canonical_terms", None) or [])
        return str(terms[0]) if terms else None
    label = getattr(concept, "display_label", None)
    return str(label) if label else None


def _concept_labels(concept, *, truth: bool) -> list[str]:
    if concept is None:
        return []
    if truth:
        return [str(term) for term in (getattr(concept, "canonical_terms", None) or [])]
    label = _concept_label(concept, truth=False)
    return [label] if label else []


def _lexical_pair_map(lexical_pairs: Iterable[Any]) -> dict[tuple[str, str], Any]:
    return {
        (pair.agent_concept_id, pair.truth_concept_id): pair for pair in lexical_pairs
    }


def build_usage_alignment_diagnostics(
    agent: AgentGraph,
    truth,
    *,
    node_mapping: Mapping[str, str],
    edge_mapping: Mapping[str, str],
    current_agent_to_truth: Mapping[str, str],
    lexical_pairs: Iterable[Any] = (),
) -> UsageAlignmentDiagnostics:
    """Build the diagnostic-only usage alignment experiment.

    ``node_mapping`` and ``edge_mapping`` are explicitly treated as a
    scaffold.  The function never changes them or the production concept
    mapping, and the assignment matrix is constructed solely from usage-set
    metrics and concept kinds.
    """

    truth_ids = sorted(_truth_referenced_concept_ids(truth))
    agent_ids = sorted(_agent_referenced_concept_ids(agent))
    truth_usage = _truth_usage_addresses(truth)
    agent_usage, agent_unmapped_usage = _agent_usage_addresses(
        agent, node_mapping, edge_mapping
    )

    pair_by_ids: dict[tuple[str, str], UsageConceptPairDiagnostic] = {}
    pairs: list[UsageConceptPairDiagnostic] = []
    for agent_id in agent_ids:
        agent_concept = agent.concepts.get(agent_id)
        if agent_concept is None:
            continue
        for truth_id in truth_ids:
            truth_concept = truth.concepts.get(truth_id)
            if truth_concept is None or agent_concept.kind != truth_concept.kind:
                continue
            truth_addresses = set(truth_usage.get(truth_id, set()))
            agent_addresses = set(agent_usage.get(agent_id, set()))
            values = _metrics(truth_addresses, agent_addresses)
            pair = UsageConceptPairDiagnostic(
                truth_concept_id=truth_id,
                truth_kind=truth_concept.kind,
                truth_label=_concept_label(truth_concept, truth=True),
                truth_labels=_concept_labels(truth_concept, truth=True),
                agent_concept_id=agent_id,
                agent_kind=agent_concept.kind,
                agent_label=_concept_label(agent_concept, truth=False),
                truth_usage_addresses=sorted(truth_addresses),
                agent_usage_addresses=sorted(agent_addresses),
                agent_unmapped_usage_addresses=sorted(
                    agent_unmapped_usage.get(agent_id, set())
                ),
                usages_only_in_truth=sorted(truth_addresses - agent_addresses),
                usages_only_in_agent=sorted(agent_addresses - truth_addresses),
                **values,
            )
            pairs.append(pair)
            pair_by_ids[(agent_id, truth_id)] = pair
    pairs.sort(
        key=lambda pair: (pair.truth_kind, pair.agent_concept_id, pair.truth_concept_id)
    )

    kinds = sorted(
        {
            agent.concepts[agent_id].kind
            for agent_id in agent_ids
            if agent_id in agent.concepts
        }
        | {
            truth.concepts[truth_id].kind
            for truth_id in truth_ids
            if truth_id in truth.concepts
        }
    )
    usage_assignment: dict[str, str] = {}
    for kind in kinds:
        left = sorted(
            agent_id
            for agent_id in agent_ids
            if agent_id in agent.concepts and agent.concepts[agent_id].kind == kind
        )
        right = sorted(
            truth_id
            for truth_id in truth_ids
            if truth_id in truth.concepts and truth.concepts[truth_id].kind == kind
        )
        exact_bonus = max(len(left), len(right)) + 1
        weights: dict[tuple[str, str], float] = {}
        for agent_id in left:
            for truth_id in right:
                pair = pair_by_ids[(agent_id, truth_id)]
                weight = pair.usage_f1
                if pair.exact_usage_match:
                    weight += exact_bonus
                if weight > 0.0:
                    weights[(agent_id, truth_id)] = weight
        usage_assignment.update(_max_weight_assignment(weights, left, right))

    truth_groups: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    agent_groups: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for truth_id in truth_ids:
        concept = truth.concepts.get(truth_id)
        if concept is None:
            continue
        addresses = frozenset(truth_usage.get(truth_id, set()))
        if addresses:
            truth_groups[(concept.kind, "truth", _signature_key(addresses))].append(
                truth_id
            )
    for agent_id in agent_ids:
        concept = agent.concepts.get(agent_id)
        if concept is None:
            continue
        addresses = frozenset(agent_usage.get(agent_id, set()))
        if addresses:
            agent_groups[(concept.kind, "agent", _signature_key(addresses))].append(
                agent_id
            )

    # A duplicate usage signature on either side is ambiguous whenever it has
    # at least one positive usage candidate on the other side.  This catches
    # partial ties as well as the simpler exact-signature case: two Agent
    # concepts with the same usage set have identical usage score rows, so an
    # id-sorted Hungarian pairing cannot establish which Truth candidate owns
    # which Agent identity.  Raw classes are merged when their candidate sets
    # overlap, producing one explicit equivalence class rather than silently
    # asserting a pairwise identity.
    raw_ambiguity_groups: list[tuple[str, set[str], set[str], set[str], str]] = []
    for (kind, _side, signature), members in sorted(agent_groups.items()):
        agent_members = set(members)
        if len(agent_members) <= 1:
            continue
        truth_candidates = {
            truth_id
            for agent_id in agent_members
            for truth_id in truth_ids
            if (pair := pair_by_ids.get((agent_id, truth_id))) is not None
            and pair.usage_f1 > 0.0
        }
        if truth_candidates:
            raw_ambiguity_groups.append(
                (
                    kind,
                    truth_candidates,
                    agent_members,
                    set(signature),
                    "identical_agent_usage_signature_with_multiple_candidates",
                )
            )
    for (kind, _side, signature), members in sorted(truth_groups.items()):
        truth_members = set(members)
        if len(truth_members) <= 1:
            continue
        agent_candidates = {
            agent_id
            for truth_id in truth_members
            for agent_id in agent_ids
            if (pair := pair_by_ids.get((agent_id, truth_id))) is not None
            and pair.usage_f1 > 0.0
        }
        if agent_candidates:
            raw_ambiguity_groups.append(
                (
                    kind,
                    truth_members,
                    agent_candidates,
                    set(signature),
                    "identical_truth_usage_signature_with_multiple_candidates",
                )
            )

    raw_ambiguity_groups.sort(
        key=lambda group: (
            group[0],
            tuple(sorted(group[1])),
            tuple(sorted(group[2])),
            tuple(sorted(group[3])),
            group[4],
        )
    )
    ambiguity_components: list[tuple[str, set[str], set[str], set[str], set[str]]] = []
    for kind, truth_members, agent_members, addresses, reason in raw_ambiguity_groups:
        matching = [
            index
            for index, component in enumerate(ambiguity_components)
            if component[0] == kind
            and (component[1] & truth_members or component[2] & agent_members)
        ]
        if not matching:
            ambiguity_components.append(
                (kind, set(truth_members), set(agent_members), set(addresses), {reason})
            )
            continue
        first = matching[0]
        (
            component_kind,
            component_truth,
            component_agent,
            component_addresses,
            reasons,
        ) = ambiguity_components[first]
        component_truth.update(truth_members)
        component_agent.update(agent_members)
        component_addresses.update(addresses)
        reasons.add(reason)
        for index in reversed(matching[1:]):
            _, other_truth, other_agent, other_addresses, other_reasons = (
                ambiguity_components.pop(index)
            )
            component_truth.update(other_truth)
            component_agent.update(other_agent)
            component_addresses.update(other_addresses)
            reasons.update(other_reasons)
        ambiguity_components[first] = (
            component_kind,
            component_truth,
            component_agent,
            component_addresses,
            reasons,
        )

    ambiguity_components.sort(
        key=lambda component: (
            component[0],
            tuple(sorted(component[1])),
            tuple(sorted(component[2])),
            tuple(sorted(component[3])),
        )
    )
    ambiguity_classes: list[UsageAmbiguityClass] = []
    class_by_truth: dict[str, str] = {}
    class_by_agent: dict[str, str] = {}
    for index, (kind, truth_members, agent_members, addresses, reasons) in enumerate(
        ambiguity_components, start=1
    ):
        class_id = f"usage_ambiguity:{kind}:{index}"
        ambiguity_classes.append(
            UsageAmbiguityClass(
                class_id=class_id,
                kind=kind,
                truth_concept_ids=sorted(truth_members),
                agent_concept_ids=sorted(agent_members),
                usage_addresses=sorted(addresses),
                reason=";".join(sorted(reasons)),
            )
        )
        for truth_id in truth_members:
            class_by_truth[truth_id] = class_id
        for agent_id in agent_members:
            class_by_agent[agent_id] = class_id

    assignments: list[UsageAssignmentDiagnostic] = []
    for agent_id, truth_id in sorted(usage_assignment.items()):
        pair = pair_by_ids[(agent_id, truth_id)]
        class_id = class_by_agent.get(agent_id) or class_by_truth.get(truth_id)
        assignments.append(
            UsageAssignmentDiagnostic(
                agent_concept_id=agent_id,
                truth_concept_id=truth_id,
                kind=pair.agent_kind,
                usage_similarity_score=pair.usage_f1,
                exact_usage_match=pair.exact_usage_match,
                ambiguous=class_id is not None,
                ambiguity_class_id=class_id,
            )
        )

    lexical_by_ids = _lexical_pair_map(lexical_pairs)
    comparisons: list[UsageComparisonDiagnostic] = []
    for agent_id in agent_ids:
        agent_concept = agent.concepts.get(agent_id)
        if agent_concept is None:
            continue
        current_truth_id = current_agent_to_truth.get(agent_id)
        usage_truth_id = usage_assignment.get(agent_id)
        current_pair = (
            pair_by_ids.get((agent_id, current_truth_id))
            if current_truth_id is not None
            else None
        )
        usage_pair = (
            pair_by_ids.get((agent_id, usage_truth_id))
            if usage_truth_id is not None
            else None
        )
        ambiguity_class_id = class_by_agent.get(agent_id)
        usage_ambiguous = ambiguity_class_id is not None
        if usage_ambiguous:
            classification = "usage_ambiguous"
        elif usage_truth_id is None:
            if current_truth_id is not None and (
                current_pair is None or current_pair.usage_f1 == 0.0
            ):
                classification = "current_mapping_has_no_usage_support"
            else:
                classification = "insufficient_usage"
        elif usage_truth_id == current_truth_id:
            classification = "same_mapping"
        elif usage_pair is not None and usage_pair.exact_usage_match:
            classification = "usage_exact_but_current_different"
        else:
            classification = "usage_prefers_different_mapping"

        lexical_pair = (
            lexical_by_ids.get((agent_id, current_truth_id))
            if current_truth_id is not None
            else None
        )
        labels_differ = bool(
            current_truth_id is not None
            and lexical_pair is not None
            and not bool(getattr(lexical_pair, "exact_label_match", False))
        )
        current_exact_usage = bool(
            current_pair is not None and current_pair.exact_usage_match
        )
        labels_agree_but_usage_does_not = bool(
            lexical_pair is not None
            and bool(getattr(lexical_pair, "exact_label_match", False))
            and not current_exact_usage
        )
        current_usage_f1 = current_pair.usage_f1 if current_pair else 0.0
        usage_stronger = bool(
            usage_pair is not None
            and usage_truth_id != current_truth_id
            and (
                usage_pair.exact_usage_match
                or (usage_pair.usage_f1 > 0.0 and current_usage_f1 == 0.0)
            )
        )
        comparisons.append(
            UsageComparisonDiagnostic(
                agent_concept_id=agent_id,
                agent_kind=agent_concept.kind,
                agent_label=_concept_label(agent_concept, truth=False),
                current_truth_concept_id=current_truth_id,
                current_truth_label=_concept_label(
                    truth.concepts.get(current_truth_id), truth=True
                ),
                usage_truth_concept_id=usage_truth_id,
                usage_truth_label=_concept_label(
                    truth.concepts.get(usage_truth_id), truth=True
                ),
                classification=classification,
                current_lexical_similarity_score=getattr(
                    lexical_pair, "lexical_similarity_score", 0.0
                ),
                current_exact_label_match=bool(
                    getattr(lexical_pair, "exact_label_match", False)
                ),
                current_exact_label_match_path=getattr(
                    lexical_pair, "exact_label_match_path", None
                ),
                current_usage_recall=(
                    current_pair.usage_recall if current_pair else 0.0
                ),
                current_usage_precision=(
                    current_pair.usage_precision if current_pair else 0.0
                ),
                current_usage_f1=current_pair.usage_f1 if current_pair else 0.0,
                current_usage_jaccard=(
                    current_pair.usage_jaccard if current_pair else 0.0
                ),
                current_exact_usage_match=current_exact_usage,
                usage_similarity_score=usage_pair.usage_f1 if usage_pair else 0.0,
                usage_exact_usage_match=(
                    usage_pair.exact_usage_match if usage_pair else False
                ),
                usage_ambiguous=usage_ambiguous,
                ambiguity_class_id=ambiguity_class_id,
                labels_differ=labels_differ,
                labels_agree_but_usage_does_not=labels_agree_but_usage_does_not,
                usage_evidence_substantially_stronger=usage_stronger,
            )
        )

    assignments_by_kind: dict[str, list[UsageAssignmentDiagnostic]] = defaultdict(list)
    comparisons_by_kind: dict[str, list[UsageComparisonDiagnostic]] = defaultdict(list)
    for assignment in assignments:
        assignments_by_kind[assignment.kind].append(assignment)
    for comparison in comparisons:
        comparisons_by_kind[comparison.agent_kind].append(comparison)

    kind_summaries: list[UsageKindSummary] = []
    for kind in kinds:
        truth_kind_ids = [
            truth_id
            for truth_id in truth_ids
            if truth_id in truth.concepts and truth.concepts[truth_id].kind == kind
        ]
        agent_kind_ids = [
            agent_id
            for agent_id in agent_ids
            if agent_id in agent.concepts and agent.concepts[agent_id].kind == kind
        ]
        kind_assignments = assignments_by_kind[kind]
        kind_comparisons = comparisons_by_kind[kind]
        disagreement_classes = {
            "usage_exact_but_current_different",
            "usage_prefers_different_mapping",
            "current_mapping_has_no_usage_support",
        }
        kind_summaries.append(
            UsageKindSummary(
                kind=kind,
                truth_referenced_concept_count=len(truth_kind_ids),
                agent_referenced_concept_count=len(agent_kind_ids),
                truth_concepts_with_usage_count=sum(
                    bool(truth_usage.get(truth_id)) for truth_id in truth_kind_ids
                ),
                agent_concepts_with_usage_count=sum(
                    bool(agent_usage.get(agent_id)) for agent_id in agent_kind_ids
                ),
                exact_usage_matches=sum(
                    assignment.exact_usage_match for assignment in kind_assignments
                ),
                partial_usage_matches=sum(
                    not assignment.exact_usage_match for assignment in kind_assignments
                ),
                structurally_ambiguous_concepts=len(
                    {
                        *{
                            truth_id
                            for ambiguity in ambiguity_classes
                            if ambiguity.kind == kind
                            for truth_id in ambiguity.truth_concept_ids
                        },
                        *{
                            agent_id
                            for ambiguity in ambiguity_classes
                            if ambiguity.kind == kind
                            for agent_id in ambiguity.agent_concept_ids
                        },
                    }
                ),
                disagreements_with_current_mapping=sum(
                    comparison.classification in disagreement_classes
                    for comparison in kind_comparisons
                ),
                usage_substantially_stronger_disagreements=sum(
                    comparison.usage_evidence_substantially_stronger
                    for comparison in kind_comparisons
                ),
                insufficient_usage=sum(
                    comparison.classification
                    in {
                        "insufficient_usage",
                        "current_mapping_has_no_usage_support",
                    }
                    for comparison in kind_comparisons
                ),
            )
        )

    unmatched_truth = sorted(set(truth_ids) - set(usage_assignment.values()))
    unmatched_agent = sorted(set(agent_ids) - set(usage_assignment))
    ambiguity_concepts = {
        *{
            truth_id
            for ambiguity in ambiguity_classes
            for truth_id in ambiguity.truth_concept_ids
        },
        *{
            agent_id
            for ambiguity in ambiguity_classes
            for agent_id in ambiguity.agent_concept_ids
        },
    }
    disagreement_classes = {
        "usage_exact_but_current_different",
        "usage_prefers_different_mapping",
        "current_mapping_has_no_usage_support",
    }
    return UsageAlignmentDiagnostics(
        truth_referenced_concept_count=len(truth_ids),
        agent_referenced_concept_count=len(agent_ids),
        exact_usage_match_count=sum(
            assignment.exact_usage_match for assignment in assignments
        ),
        partial_usage_match_count=sum(
            not assignment.exact_usage_match for assignment in assignments
        ),
        structurally_ambiguous_concept_count=len(ambiguity_concepts),
        disagreement_count=sum(
            comparison.classification in disagreement_classes
            for comparison in comparisons
        ),
        usage_substantially_stronger_disagreement_count=sum(
            comparison.usage_evidence_substantially_stronger
            for comparison in comparisons
        ),
        labels_differ_usage_agrees_count=sum(
            comparison.labels_differ
            and comparison.usage_exact_usage_match
            and comparison.current_truth_concept_id == comparison.usage_truth_concept_id
            for comparison in comparisons
        ),
        labels_agree_usage_does_not_count=sum(
            comparison.labels_agree_but_usage_does_not for comparison in comparisons
        ),
        current_agent_to_truth={
            agent_id: current_agent_to_truth[agent_id]
            for agent_id in sorted(current_agent_to_truth)
        },
        usage_agent_to_truth={
            assignment.agent_concept_id: assignment.truth_concept_id
            for assignment in assignments
        },
        candidate_pairs=pairs,
        assignments=assignments,
        ambiguity_classes=ambiguity_classes,
        comparisons=comparisons,
        kind_summaries=kind_summaries,
        unmatched_truth_concept_ids=unmatched_truth,
        unmatched_agent_concept_ids=unmatched_agent,
        unmapped_agent_usage_addresses={
            concept_id: sorted(addresses)
            for concept_id, addresses in sorted(agent_unmapped_usage.items())
            if concept_id in agent_ids
        },
    )


__all__ = [
    "UsageAlignmentDiagnostics",
    "UsageAmbiguityClass",
    "UsageAssignmentDiagnostic",
    "UsageComparisonDiagnostic",
    "UsageConceptPairDiagnostic",
    "UsageKindSummary",
    "build_usage_alignment_diagnostics",
]
