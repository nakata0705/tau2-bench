"""Diagnostic-only joint structural alignment for business_interview.

This module is deliberately independent from the production evaluator matcher.
It treats an AgentGraph and a BusinessProcessGraph as typed heterogeneous
graphs and searches for a pair of partial one-to-one mappings:

* Agent nodes -> Truth nodes
* asserted Agent concepts -> asserted Truth concepts

Only graph structure is inspected.  In particular, this module never reads a
label, description, canonical term, translation, embedding, sidecar semantic
id, EvidenceRef text, StakeholderKnowledge mapping, or LLM output.  Concept
``kind`` is a hard constraint.  UNSET, ABSENT, and DONT_KNOW are epistemic
states, not Concept nodes; only asserted ConceptRef values contribute graph
incidences.

The search is exact for the bounded candidate space when its deterministic
state limits are not reached.  Node mappings are enumerated with a
branch-and-bound-style bounded backtracking search.  For each node mapping,
non-condition concept assignments use exact additive partial assignment, while
condition concepts are enumerated so parallel process edges can choose the
best endpoint-compatible edge correspondence.  If a limit is reached, the
result says so instead of pretending that the optimum or uniqueness is proven.
IDs are sorted only to make traversal, representative serialization, and
ambiguity artifacts reproducible; they are never an objective signal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.graph import AgentGraph, ConceptRef

_NODE_RELATIONS: tuple[str, ...] = (
    "activity",
    "actor",
    "system",
    "reads",
    "writes",
    "rationale",
)
_RELATIONS: tuple[str, ...] = (*_NODE_RELATIONS, "condition")

# Equal component weights keep the diagnostic easy to audit.  Every component
# is a bounded agreement score, so extra/missing structure lowers that
# component even though the search itself only rewards preserved structure.
_COMPONENT_WEIGHT = Fraction(1, 1)


class JointStructuralComponent(BaseModel):
    """One normalized component of the joint structural objective."""

    matched_count: int = 0
    agent_count: int = 0
    truth_count: int = 0
    agreement: float = 0.0
    weight: float = 1.0
    contribution: float = 0.0


class JointStructuralObjective(BaseModel):
    """Auditable objective components for one winning alignment.

    Each component is a bounded F1-style agreement ``2*matched/(Agent+Truth)``
    (empty on both sides is exact agreement), and the total is the unweighted
    sum of Node, Concept, six node-slot, condition, process-edge, start, and
    end components.  A concept/node pair with no positive typed structural
    support is outside the admissible match domain and remains unmatched; it
    is not an unreported zero-evidence tie.
    """

    total_score: float = 0.0
    max_score: float = 0.0
    normalized_score: float = 0.0
    components: dict[str, JointStructuralComponent] = Field(default_factory=dict)


class JointStructuralAmbiguityClass(BaseModel):
    """A structural equivalence class among equally optimal mappings."""

    class_id: str
    entity_type: str
    agent_ids: list[str] = Field(default_factory=list)
    truth_ids: list[str] = Field(default_factory=list)
    unmatched_agent_ids: list[str] = Field(default_factory=list)
    unmatched_truth_ids: list[str] = Field(default_factory=list)
    reason: str = "equal_optimal_structural_alignment"


class JointStructuralSearchStats(BaseModel):
    """Deterministic search-size and bound information."""

    algorithm: str = "bounded_node_backtracking_with_exact_per_kind_concept_search"
    exact_search: bool = True
    bound_hit: bool = False
    bound_reason: Optional[str] = None
    max_node_search_states: int = 250_000
    max_concept_search_states: int = 1_000_000
    max_optimal_alternatives: int = 2_048
    node_candidate_pair_count: int = 0
    node_search_states: int = 0
    node_leaves_evaluated: int = 0
    node_states_pruned: int = 0
    concept_search_states: int = 0
    concept_states_pruned: int = 0
    concept_assignments_evaluated: int = 0
    edge_assignment_states: int = 0
    optimal_solution_count: int = 0
    optimum_is_unique: bool = False
    optimal_solution_count_is_exact: bool = False
    # A bounded incumbent is not a lower bound on the number of global optima.
    optimal_solution_count_is_lower_bound: bool = False
    objective_lower_bound: float = 0.0
    objective_upper_bound: float = 0.0
    # A deterministic work estimate is used instead of wall-clock time so
    # regenerated JSON/report artifacts remain byte-for-byte reproducible.
    runtime_estimate_ms: int = 0


class JointStructuralAlignmentDiagnostics(BaseModel):
    """Evaluator-private result of the joint structural experiment."""

    schema_version: str = "business_interview.joint_structural_alignment.v1"
    status: str = "ok"
    error_type: Optional[str] = None
    method: str = "joint_structural_branch_and_bound"
    interpretation: str = (
        "Jointly choose partial one-to-one Node and asserted Concept mappings "
        "to maximize typed graph agreement; this is diagnostic only."
    )
    assignment_uses_labels: bool = False
    forbidden_signals: list[str] = Field(
        default_factory=lambda: [
            "labels_or_canonical_terms",
            "descriptions",
            "lexical_similarity",
            "translations_or_embeddings",
            "StakeholderKnowledge_mappings",
            "sidecar_semantic_ids",
            "EvidenceRef_text",
            "LLM_judgment",
        ]
    )
    concept_kind_is_hard_constraint: bool = True
    asserted_concepts_only: bool = True
    epistemic_markers_are_not_concepts: bool = True
    unsupported_pairs_are_not_matches: bool = True
    candidate_policy: str = (
        "Only same-kind pairs with positive typed structural support are "
        "admissible; unsupported entities remain unmatched rather than being "
        "assigned by opaque-id order."
    )
    representative_agent_node_to_truth_node: dict[str, str] = Field(
        default_factory=dict
    )
    representative_agent_concept_to_truth_concept: dict[str, str] = Field(
        default_factory=dict
    )
    representative_agent_edge_to_truth_edge: dict[str, str] = Field(
        default_factory=dict
    )
    representative_mapping_is_not_identity_resolution: bool = True
    invariant_agent_node_to_truth_node: dict[str, str] = Field(default_factory=dict)
    invariant_agent_concept_to_truth_concept: dict[str, str] = Field(
        default_factory=dict
    )
    invariants_proven: bool = False
    node_ambiguity_classes: list[JointStructuralAmbiguityClass] = Field(
        default_factory=list
    )
    concept_ambiguity_classes: list[JointStructuralAmbiguityClass] = Field(
        default_factory=list
    )
    unmatched_representative_agent_nodes: list[str] = Field(default_factory=list)
    unmatched_representative_truth_nodes: list[str] = Field(default_factory=list)
    unmatched_representative_agent_concepts: list[str] = Field(default_factory=list)
    unmatched_representative_truth_concepts: list[str] = Field(default_factory=list)
    objective: JointStructuralObjective = Field(
        default_factory=JointStructuralObjective
    )
    search: JointStructuralSearchStats = Field(
        default_factory=JointStructuralSearchStats
    )
    production_agent_node_to_truth_node: dict[str, str] = Field(default_factory=dict)
    production_agent_concept_to_truth_concept: dict[str, str] = Field(
        default_factory=dict
    )
    conditioned_usage_agent_concept_to_truth_concept: dict[str, str] = Field(
        default_factory=dict
    )
    production_vs_joint_node_differences: list[dict[str, Optional[str]]] = Field(
        default_factory=list
    )
    production_vs_joint_concept_differences: list[dict[str, Optional[str]]] = Field(
        default_factory=list
    )
    usage_vs_joint_concept_differences: list[dict[str, Optional[str]]] = Field(
        default_factory=list
    )


@dataclass
class _GraphIndex:
    node_ids: list[str]
    edge_ids: list[str]
    concept_ids: list[str]
    concept_ids_by_kind: dict[str, list[str]]
    node_refs: dict[tuple[str, str], set[str]]
    edge_condition_refs: dict[str, set[str]]
    relation_totals: dict[str, int]
    incoming_degree: dict[str, int]
    outgoing_degree: dict[str, int]
    end_nodes: set[str]
    start_node: Optional[str]
    node_kind_counts: dict[tuple[str, str], dict[str, int]]


@dataclass
class _AssignmentResult:
    best_score: Fraction
    mappings: list[dict[str, str]]
    truncated: bool = False


@dataclass
class _Solution:
    objective: Fraction
    components: dict[str, tuple[int, int, int, Fraction]]
    node_mapping: dict[str, str]
    concept_mapping: dict[str, str]
    edge_mapping: dict[str, str]


@dataclass
class _MutableSearch:
    max_node_states: int
    max_concept_states: int
    max_alternatives: int
    node_candidate_pair_count: int = 0
    node_search_states: int = 0
    node_leaves_evaluated: int = 0
    node_states_pruned: int = 0
    concept_search_states: int = 0
    concept_states_pruned: int = 0
    concept_assignments_evaluated: int = 0
    edge_assignment_states: int = 0
    bound_hit: bool = False
    bound_reasons: set[str] | None = None

    def __post_init__(self) -> None:
        if self.bound_reasons is None:
            self.bound_reasons = set()

    def hit(self, reason: str) -> None:
        self.bound_hit = True
        if self.bound_reasons is None:
            self.bound_reasons = set()
        self.bound_reasons.add(reason)


# ---------------------------------------------------------------------------
# Label-free graph indexing
# ---------------------------------------------------------------------------


def _asserted_ref_ids(value: Any, property_name: str) -> set[str]:
    """Return asserted ConceptRef ids, excluding all epistemic markers."""
    refs = value.asserted_refs(property_name)
    return {
        ref.concept_id
        for ref in refs
        if isinstance(ref, ConceptRef) and ref.asserted and ref.concept_id
    }


def _build_index(graph) -> _GraphIndex:
    node_ids = sorted(graph.nodes)
    edge_ids = sorted(graph.edges)
    node_refs: dict[tuple[str, str], set[str]] = {}
    edge_condition_refs: dict[str, set[str]] = {}
    relation_totals = {relation: 0 for relation in _RELATIONS}
    referenced: set[str] = set()

    for node_id in node_ids:
        node = graph.nodes[node_id]
        for relation in _NODE_RELATIONS:
            refs = _asserted_ref_ids(node, relation)
            node_refs[(node_id, relation)] = refs
            relation_totals[relation] += len(refs)
            referenced.update(refs)

    for edge_id in edge_ids:
        condition = graph.edges[edge_id].condition
        refs = (
            {condition.concept_id}
            if isinstance(condition, ConceptRef)
            and condition.asserted
            and condition.concept_id
            else set()
        )
        edge_condition_refs[edge_id] = refs
        relation_totals["condition"] += len(refs)
        referenced.update(refs)

    concept_ids = sorted(cid for cid in referenced if cid in graph.concepts)
    concept_ids_by_kind: dict[str, list[str]] = defaultdict(list)
    for concept_id in concept_ids:
        concept_ids_by_kind[graph.concepts[concept_id].kind].append(concept_id)
    for kind in concept_ids_by_kind:
        concept_ids_by_kind[kind].sort()

    incoming_degree = {node_id: 0 for node_id in node_ids}
    outgoing_degree = {node_id: 0 for node_id in node_ids}
    for edge in graph.edges.values():
        if edge.from_node in outgoing_degree:
            outgoing_degree[edge.from_node] += 1
        if edge.to_node in incoming_degree:
            incoming_degree[edge.to_node] += 1

    end_nodes = set(graph.end_node_ids) & set(node_ids)
    node_kind_counts: dict[tuple[str, str], dict[str, int]] = {}
    for node_id in node_ids:
        for relation in _NODE_RELATIONS:
            counts: dict[str, int] = defaultdict(int)
            for concept_id in node_refs[(node_id, relation)]:
                concept = graph.concepts.get(concept_id)
                if concept is not None:
                    counts[concept.kind] += 1
            node_kind_counts[(node_id, relation)] = dict(counts)

    return _GraphIndex(
        node_ids=node_ids,
        edge_ids=edge_ids,
        concept_ids=concept_ids,
        concept_ids_by_kind={
            kind: list(ids) for kind, ids in concept_ids_by_kind.items()
        },
        node_refs=node_refs,
        edge_condition_refs=edge_condition_refs,
        relation_totals=relation_totals,
        incoming_degree=incoming_degree,
        outgoing_degree=outgoing_degree,
        end_nodes=end_nodes,
        start_node=(graph.start_node_id if graph.start_node_id in node_ids else None),
        node_kind_counts=node_kind_counts,
    )


def _node_candidate_potential(
    agent: _GraphIndex,
    truth: _GraphIndex,
    agent_node_id: str,
    truth_node_id: str,
) -> Fraction:
    """A label-free eligibility/order score, not an objective tie-breaker."""
    potential = Fraction(0, 1)
    for relation in _NODE_RELATIONS:
        a_counts = agent.node_kind_counts[(agent_node_id, relation)]
        t_counts = truth.node_kind_counts[(truth_node_id, relation)]
        for kind in set(a_counts) | set(t_counts):
            potential += min(a_counts.get(kind, 0), t_counts.get(kind, 0))

    if agent.incoming_degree[agent_node_id] == truth.incoming_degree[truth_node_id]:
        potential += Fraction(1, 2)
    if agent.outgoing_degree[agent_node_id] == truth.outgoing_degree[truth_node_id]:
        potential += Fraction(1, 2)
    if (agent.start_node == agent_node_id) == (truth.start_node == truth_node_id):
        potential += 1
    if (agent_node_id in agent.end_nodes) == (truth_node_id in truth.end_nodes):
        potential += 1

    # Two completely bare, equally-role'd nodes are still structurally
    # eligible.  Their eventual mapping is intentionally reported ambiguous.
    agent_bare = (
        all(
            not agent.node_kind_counts[(agent_node_id, relation)]
            for relation in _NODE_RELATIONS
        )
        and agent.incoming_degree[agent_node_id] == 0
        and agent.outgoing_degree[agent_node_id] == 0
    )
    truth_bare = (
        all(
            not truth.node_kind_counts[(truth_node_id, relation)]
            for relation in _NODE_RELATIONS
        )
        and truth.incoming_degree[truth_node_id] == 0
        and truth.outgoing_degree[truth_node_id] == 0
    )
    if agent_bare and truth_bare:
        potential += Fraction(1, 4)
    return potential


def _node_candidates(
    agent: _GraphIndex,
    truth: _GraphIndex,
    *,
    allowed_node_pairs: Mapping[str, Iterable[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], Fraction]]:
    candidates: dict[str, list[str]] = {}
    potentials: dict[tuple[str, str], Fraction] = {}
    allowed_sets = (
        {agent_id: set(truth_ids) for agent_id, truth_ids in allowed_node_pairs.items()}
        if allowed_node_pairs is not None
        else None
    )
    for agent_node_id in agent.node_ids:
        pairs: list[tuple[Fraction, str]] = []
        allowed_truth_ids = (
            allowed_sets.get(agent_node_id, set()) if allowed_sets is not None else None
        )
        for truth_node_id in truth.node_ids:
            if allowed_truth_ids is not None and truth_node_id not in allowed_truth_ids:
                continue
            potential = _node_candidate_potential(
                agent, truth, agent_node_id, truth_node_id
            )
            if potential <= 0:
                continue
            potentials[(agent_node_id, truth_node_id)] = potential
            pairs.append((potential, truth_node_id))
        pairs.sort(key=lambda item: (-item[0], item[1]))
        candidates[agent_node_id] = [truth_node_id for _, truth_node_id in pairs]
    return candidates, potentials


# ---------------------------------------------------------------------------
# Assignment and edge helpers
# ---------------------------------------------------------------------------


def _enumerate_best_additive_assignments(
    weights: Mapping[tuple[str, str], Fraction],
    left: list[str],
    right: list[str],
    search: _MutableSearch,
) -> _AssignmentResult:
    """Exact maximum-weight partial assignment for positive pair weights."""
    if not left or not right:
        return _AssignmentResult(Fraction(0, 1), [{}])

    candidates: dict[str, list[str]] = {}
    for left_id in left:
        candidate_ids = [
            right_id
            for candidate_left, right_id in weights
            if candidate_left == left_id and weights[(candidate_left, right_id)] > 0
        ]
        candidate_ids.sort(
            key=lambda right_id: (-weights[(left_id, right_id)], right_id)
        )
        candidates[left_id] = candidate_ids
    ordered_left = sorted(left, key=lambda left_id: (len(candidates[left_id]), left_id))
    suffix_max: list[Fraction] = [Fraction(0, 1)] * (len(ordered_left) + 1)
    for index in range(len(ordered_left) - 1, -1, -1):
        left_id = ordered_left[index]
        largest = Fraction(0, 1)
        for right_id in candidates[left_id]:
            largest = max(largest, weights[(left_id, right_id)])
        suffix_max[index] = suffix_max[index + 1] + largest

    best: Optional[Fraction] = None
    mappings: list[dict[str, str]] = []
    mapping_keys: set[tuple[tuple[str, str], ...]] = set()
    truncated = False

    def visit(
        position: int,
        used_right: set[str],
        mapping: dict[str, str],
        score: Fraction,
    ) -> None:
        nonlocal best, mappings, truncated
        if search.concept_search_states >= search.max_concept_states:
            search.hit("concept_search_state_limit")
            truncated = True
            return
        search.concept_search_states += 1
        if best is not None and score + suffix_max[position] < best:
            search.concept_states_pruned += 1
            return
        if position == len(ordered_left):
            search.concept_assignments_evaluated += 1
            key = tuple(sorted(mapping.items()))
            if best is None or score > best:
                best = score
                mappings = [dict(mapping)]
                mapping_keys.clear()
                mapping_keys.add(key)
            elif score == best and key not in mapping_keys:
                if len(mappings) < search.max_alternatives:
                    mappings.append(dict(mapping))
                    mapping_keys.add(key)
                else:
                    truncated = True
                    search.hit("optimal_alternative_limit")
            return

        left_id = ordered_left[position]
        # A positive pair always dominates leaving the left entity unmatched
        # when that pair remains available, but the skip branch is required for
        # one-to-one conflicts and for partial mappings.
        for right_id in candidates[left_id]:
            if right_id in used_right:
                continue
            mapping[left_id] = right_id
            used_right.add(right_id)
            visit(
                position + 1,
                used_right,
                mapping,
                score + weights[(left_id, right_id)],
            )
            used_right.remove(right_id)
            del mapping[left_id]
        visit(position + 1, used_right, mapping, score)

    visit(0, set(), {}, Fraction(0, 1))
    if best is None:
        return _AssignmentResult(Fraction(0, 1), [{}], truncated=truncated)
    mappings.sort(key=lambda mapping: tuple(sorted(mapping.items())))
    return _AssignmentResult(best, mappings, truncated=truncated)


def _enumerate_candidate_assignments(
    left: list[str],
    right: list[str],
    allowed: Mapping[tuple[str, str], bool],
    search: _MutableSearch,
) -> tuple[list[dict[str, str]], bool]:
    """Enumerate partial mappings for condition concepts with a hard bound."""
    if not left or not right:
        return [{}], False
    candidates: dict[str, list[str]] = {
        left_id: sorted(
            right_id for right_id in right if allowed.get((left_id, right_id), False)
        )
        for left_id in left
    }
    ordered_left = sorted(left, key=lambda left_id: (len(candidates[left_id]), left_id))
    results: list[dict[str, str]] = []
    result_keys: set[tuple[tuple[str, str], ...]] = set()
    truncated = False

    def visit(position: int, used_right: set[str], mapping: dict[str, str]) -> None:
        nonlocal truncated
        if search.concept_search_states >= search.max_concept_states:
            search.hit("concept_search_state_limit")
            truncated = True
            return
        search.concept_search_states += 1
        if position == len(ordered_left):
            search.concept_assignments_evaluated += 1
            key = tuple(sorted(mapping.items()))
            if key not in result_keys:
                result_keys.add(key)
                if len(results) < search.max_alternatives:
                    results.append(dict(mapping))
                else:
                    truncated = True
                    search.hit("optimal_alternative_limit")
            return
        left_id = ordered_left[position]
        for right_id in candidates[left_id]:
            if right_id in used_right:
                continue
            mapping[left_id] = right_id
            used_right.add(right_id)
            visit(position + 1, used_right, mapping)
            used_right.remove(right_id)
            del mapping[left_id]
        # Unmatched concepts are explicitly allowed.
        visit(position + 1, used_right, mapping)

    visit(0, set(), {})
    results.sort(key=lambda mapping: tuple(sorted(mapping.items())))
    return results or [{}], truncated


def _endpoint_edge_groups(
    agent_graph,
    truth_graph,
    node_mapping: Mapping[str, str],
) -> dict[tuple[str, str], tuple[list[str], list[str]]]:
    grouped_agent: dict[tuple[str, str], list[str]] = defaultdict(list)
    grouped_truth: dict[tuple[str, str], list[str]] = defaultdict(list)
    for edge_id in sorted(agent_graph.edges):
        edge = agent_graph.edges[edge_id]
        from_node = node_mapping.get(edge.from_node)
        to_node = node_mapping.get(edge.to_node)
        if from_node is not None and to_node is not None:
            grouped_agent[(from_node, to_node)].append(edge_id)
    # Truth groups are populated for every endpoint pair that an Agent edge can
    # reach; keeping this restricted avoids accidentally rewarding unrelated
    # Truth edges.
    reachable_pairs = set(grouped_agent)
    for edge_id in sorted(truth_graph.edges):
        edge = truth_graph.edges[edge_id]
        pair = (edge.from_node, edge.to_node)
        if pair in reachable_pairs:
            grouped_truth[pair].append(edge_id)
    return {
        pair: (sorted(grouped_agent[pair]), sorted(grouped_truth[pair]))
        for pair in sorted(reachable_pairs)
    }


def _best_edge_group_mapping(
    agent_graph,
    truth_graph,
    agent_edge_ids: list[str],
    truth_edge_ids: list[str],
    concept_mapping: Mapping[str, str],
    search: _MutableSearch,
) -> tuple[dict[str, str], int]:
    """Map all possible parallel endpoint edges, maximizing condition hits."""
    needed = min(len(agent_edge_ids), len(truth_edge_ids))
    if needed == 0:
        return {}, 0
    agent_edge_ids = sorted(agent_edge_ids)
    truth_edge_ids = sorted(truth_edge_ids)
    best_count = -1
    best_mapping: dict[str, str] = {}
    best_key: Optional[tuple[tuple[str, str], ...]] = None
    must_skip_agent = len(agent_edge_ids) > len(truth_edge_ids)

    def condition_match(agent_edge_id: str, truth_edge_id: str) -> int:
        a = agent_graph.edges[agent_edge_id].condition
        t = truth_graph.edges[truth_edge_id].condition
        if not isinstance(a, ConceptRef) or not a.asserted:
            return 0
        if not isinstance(t, ConceptRef) or not t.asserted:
            return 0
        return concept_mapping.get(a.concept_id) == t.concept_id

    def visit(
        position: int,
        remaining_truth: tuple[str, ...],
        mapping: dict[str, str],
        score: int,
    ) -> None:
        nonlocal best_count, best_mapping, best_key
        if search.edge_assignment_states >= search.max_concept_states:
            search.hit("edge_assignment_state_limit")
            return
        search.edge_assignment_states += 1
        remaining_agents = len(agent_edge_ids) - position
        if len(mapping) + min(remaining_agents, len(remaining_truth)) < needed:
            return
        if position == len(agent_edge_ids):
            if len(mapping) != needed:
                return
            key = tuple(sorted(mapping.items()))
            if score > best_count or (
                score == best_count and (best_key is None or key < best_key)
            ):
                best_count = score
                best_mapping = dict(mapping)
                best_key = key
            return

        agent_edge_id = agent_edge_ids[position]
        for truth_edge_id in remaining_truth:
            mapping[agent_edge_id] = truth_edge_id
            visit(
                position + 1,
                tuple(item for item in remaining_truth if item != truth_edge_id),
                mapping,
                score + condition_match(agent_edge_id, truth_edge_id),
            )
            del mapping[agent_edge_id]
        if must_skip_agent or len(mapping) + remaining_agents > needed:
            visit(position + 1, remaining_truth, mapping, score)

    visit(0, tuple(truth_edge_ids), {}, 0)
    if best_count < 0:
        return {}, 0
    return best_mapping, best_count


def _edge_mapping_and_condition_hits(
    agent_graph,
    truth_graph,
    node_mapping: Mapping[str, str],
    concept_mapping: Mapping[str, str],
    search: _MutableSearch,
) -> tuple[dict[str, str], int, int]:
    edge_mapping: dict[str, str] = {}
    condition_matches = 0
    for agent_edge_ids, truth_edge_ids in _endpoint_edge_groups(
        agent_graph, truth_graph, node_mapping
    ).values():
        group_mapping, group_condition_matches = _best_edge_group_mapping(
            agent_graph,
            truth_graph,
            agent_edge_ids,
            truth_edge_ids,
            concept_mapping,
            search,
        )
        edge_mapping.update(group_mapping)
        condition_matches += group_condition_matches
    return edge_mapping, len(edge_mapping), condition_matches


# ---------------------------------------------------------------------------
# Joint objective
# ---------------------------------------------------------------------------


def _pair_support(
    agent_graph,
    truth_graph,
    agent: _GraphIndex,
    truth: _GraphIndex,
    node_mapping: Mapping[str, str],
) -> tuple[dict[tuple[str, str], dict[str, int]], dict[tuple[str, str], bool]]:
    """Return same-kind node/edge condition support for concept candidates."""
    support: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    condition_possible: dict[tuple[str, str], bool] = {}
    for agent_node_id, truth_node_id in node_mapping.items():
        for relation in _NODE_RELATIONS:
            for agent_concept_id in agent.node_refs[(agent_node_id, relation)]:
                if agent_concept_id not in agent_graph.concepts:
                    continue
                agent_kind = agent_graph.concepts[agent_concept_id].kind
                for truth_concept_id in truth.node_refs[(truth_node_id, relation)]:
                    if truth_concept_id not in truth_graph.concepts:
                        continue
                    if agent_kind != truth_graph.concepts[truth_concept_id].kind:
                        continue
                    support[(agent_concept_id, truth_concept_id)][relation] += 1

    for agent_edge_ids, truth_edge_ids in _endpoint_edge_groups(
        agent_graph, truth_graph, node_mapping
    ).values():
        for agent_edge_id in agent_edge_ids:
            agent_condition = agent_graph.edges[agent_edge_id].condition
            if (
                not isinstance(agent_condition, ConceptRef)
                or not agent_condition.asserted
            ):
                continue
            if agent_condition.concept_id not in agent_graph.concepts:
                continue
            agent_kind = agent_graph.concepts[agent_condition.concept_id].kind
            for truth_edge_id in truth_edge_ids:
                truth_condition = truth_graph.edges[truth_edge_id].condition
                if (
                    not isinstance(truth_condition, ConceptRef)
                    or not truth_condition.asserted
                ):
                    continue
                if truth_condition.concept_id not in truth_graph.concepts:
                    continue
                if agent_kind != truth_graph.concepts[truth_condition.concept_id].kind:
                    continue
                pair = (agent_condition.concept_id, truth_condition.concept_id)
                condition_possible[pair] = True
    return {key: dict(value) for key, value in support.items()}, condition_possible


def _component(
    matched: int,
    agent_count: int,
    truth_count: int,
) -> tuple[int, int, int, Fraction]:
    denominator = agent_count + truth_count
    agreement = Fraction(2 * matched, denominator) if denominator else Fraction(1, 1)
    return matched, agent_count, truth_count, agreement


def _solution_for_mapping(
    agent_graph,
    truth_graph,
    agent: _GraphIndex,
    truth: _GraphIndex,
    node_mapping: Mapping[str, str],
    concept_mapping: Mapping[str, str],
    search: _MutableSearch,
) -> _Solution:
    support, _ = _pair_support(agent_graph, truth_graph, agent, truth, node_mapping)
    edge_mapping, edge_matches, condition_matches = _edge_mapping_and_condition_hits(
        agent_graph, truth_graph, node_mapping, concept_mapping, search
    )
    relation_matches = {relation: 0 for relation in _RELATIONS}
    for (agent_concept_id, truth_concept_id), values in support.items():
        if concept_mapping.get(agent_concept_id) != truth_concept_id:
            continue
        for relation, count in values.items():
            relation_matches[relation] += count
    relation_matches["condition"] = condition_matches

    components: dict[str, tuple[int, int, int, Fraction]] = {}
    components["nodes"] = _component(
        len(node_mapping), len(agent.node_ids), len(truth.node_ids)
    )
    components["concepts"] = _component(
        len(concept_mapping), len(agent.concept_ids), len(truth.concept_ids)
    )
    for relation in _RELATIONS:
        components[relation] = _component(
            relation_matches[relation],
            agent.relation_totals[relation],
            truth.relation_totals[relation],
        )
    components["process_edges"] = _component(
        edge_matches, len(agent.edge_ids), len(truth.edge_ids)
    )

    both_start_absent = agent.start_node is None and truth.start_node is None
    start_match = (
        1
        if (
            both_start_absent
            or (
                agent.start_node is not None
                and truth.start_node is not None
                and node_mapping.get(agent.start_node) == truth.start_node
            )
        )
        else 0
    )
    if both_start_absent:
        components["start_node"] = (0, 0, 0, Fraction(1, 1))
    else:
        components["start_node"] = (
            start_match,
            1 if agent.start_node is not None else 0,
            1 if truth.start_node is not None else 0,
            Fraction(start_match, 1),
        )

    mapped_agent_ends = {
        node_mapping[agent_end]
        for agent_end in agent.end_nodes
        if agent_end in node_mapping
    }
    components["end_nodes"] = _component(
        len(mapped_agent_ends & truth.end_nodes),
        len(agent.end_nodes),
        len(truth.end_nodes),
    )

    objective = sum(
        (agreement * _COMPONENT_WEIGHT for _, _, _, agreement in components.values()),
        Fraction(0, 1),
    )
    return _Solution(
        objective=objective,
        components=components,
        node_mapping=dict(sorted(node_mapping.items())),
        concept_mapping=dict(sorted(concept_mapping.items())),
        edge_mapping=dict(sorted(edge_mapping.items())),
    )


def _pair_weights(
    support: Mapping[tuple[str, str], Mapping[str, int]],
    agent_graph,
    truth_graph,
    agent: _GraphIndex,
    truth: _GraphIndex,
) -> dict[tuple[str, str], Fraction]:
    concept_denominator = len(agent.concept_ids) + len(truth.concept_ids)
    concept_reward = (
        Fraction(2, concept_denominator) if concept_denominator else Fraction(0, 1)
    )
    relation_rewards = {
        relation: (
            Fraction(
                2, agent.relation_totals[relation] + truth.relation_totals[relation]
            )
            if agent.relation_totals[relation] + truth.relation_totals[relation]
            else Fraction(0, 1)
        )
        for relation in _NODE_RELATIONS
    }
    weights: dict[tuple[str, str], Fraction] = {}
    for pair, values in support.items():
        if pair[0] not in agent_graph.concepts or pair[1] not in truth_graph.concepts:
            continue
        if agent_graph.concepts[pair[0]].kind != truth_graph.concepts[pair[1]].kind:
            continue
        incidence_reward = sum(
            count * relation_rewards[relation] for relation, count in values.items()
        )
        if incidence_reward > 0:
            weights[pair] = concept_reward + incidence_reward
    return weights


def _condition_pair_allowed(
    condition_possible: Mapping[tuple[str, str], bool],
    agent: _GraphIndex,
    truth: _GraphIndex,
) -> dict[tuple[str, str], bool]:
    return {
        (agent_concept_id, truth_concept_id): True
        for (agent_concept_id, truth_concept_id), possible in condition_possible.items()
        if possible
        and agent_concept_id in agent.concept_ids
        and truth_concept_id in truth.concept_ids
    }


def _mapping_key(solution: _Solution) -> tuple:
    return (
        tuple(solution.node_mapping.items()),
        tuple(solution.concept_mapping.items()),
    )


# ---------------------------------------------------------------------------
# Ambiguity and comparison reporting
# ---------------------------------------------------------------------------


def _ambiguity_classes(
    solutions: list[_Solution],
    entity_type: str,
    entity_ids: Iterable[str],
    exact: bool,
) -> tuple[dict[str, str], list[JointStructuralAmbiguityClass]]:
    if not solutions:
        return {}, []
    mapping_by_solution = [
        solution.node_mapping if entity_type == "node" else solution.concept_mapping
        for solution in solutions
    ]
    ids = sorted(set(entity_ids))
    invariant: dict[str, str] = {}
    possible: dict[str, set[str]] = {}
    unmapped: dict[str, bool] = {}
    for entity_id in ids:
        values = {mapping.get(entity_id) for mapping in mapping_by_solution}
        non_null = {value for value in values if value is not None}
        possible[entity_id] = set(non_null)
        unmapped[entity_id] = None in values
        if exact and len(values) == 1 and next(iter(values), None) is not None:
            invariant[entity_id] = next(iter(values))  # type: ignore[assignment]

    reverse: dict[str, set[str]] = defaultdict(set)
    for entity_id, targets in possible.items():
        for target_id in targets:
            reverse[target_id].add(entity_id)

    ambiguous_seed_agents = {
        entity_id
        for entity_id in ids
        if len(possible[entity_id]) > 1
        or (unmapped[entity_id] and possible[entity_id])
        or any(len(reverse[target_id]) > 1 for target_id in possible[entity_id])
    }
    ambiguous_seed_truths = {
        target_id for target_id, source_ids in reverse.items() if len(source_ids) > 1
    }

    adjacency_agent: dict[str, set[str]] = defaultdict(set)
    adjacency_truth: dict[str, set[str]] = defaultdict(set)
    for entity_id, targets in possible.items():
        for target_id in targets:
            adjacency_agent[entity_id].add(target_id)
            adjacency_truth[target_id].add(entity_id)

    classes: list[JointStructuralAmbiguityClass] = []
    visited_agents: set[str] = set()
    visited_truths: set[str] = set()
    seeds = sorted(ambiguous_seed_agents) + [
        f"\x00{target_id}" for target_id in sorted(ambiguous_seed_truths)
    ]
    for seed in seeds:
        if seed.startswith("\x00"):
            truth_seed = seed[1:]
            if truth_seed in visited_truths:
                continue
            queue_truth = [truth_seed]
            queue_agent: list[str] = []
        else:
            if seed in visited_agents:
                continue
            queue_agent = [seed]
            queue_truth = []
        class_agents: set[str] = set()
        class_truths: set[str] = set()
        while queue_agent or queue_truth:
            while queue_agent:
                agent_id = queue_agent.pop()
                if agent_id in visited_agents:
                    continue
                visited_agents.add(agent_id)
                class_agents.add(agent_id)
                for target_id in adjacency_agent.get(agent_id, set()):
                    if target_id not in visited_truths:
                        queue_truth.append(target_id)
            while queue_truth:
                truth_id = queue_truth.pop()
                if truth_id in visited_truths:
                    continue
                visited_truths.add(truth_id)
                class_truths.add(truth_id)
                for agent_id in adjacency_truth.get(truth_id, set()):
                    if agent_id not in visited_agents:
                        queue_agent.append(agent_id)
        if not class_agents and not class_truths:
            continue
        unmatched_agents = sorted(
            entity_id for entity_id in class_agents if unmapped.get(entity_id, False)
        )
        unmatched_truths = sorted(
            truth_id
            for truth_id in class_truths
            if any(
                truth_id not in candidate_mapping.values()
                for candidate_mapping in mapping_by_solution
            )
        )
        classes.append(
            JointStructuralAmbiguityClass(
                class_id=f"joint_ambiguity:{entity_type}:{len(classes) + 1}",
                entity_type=entity_type,
                agent_ids=sorted(class_agents),
                truth_ids=sorted(class_truths),
                unmatched_agent_ids=unmatched_agents,
                unmatched_truth_ids=unmatched_truths,
                reason=(
                    "equal_optimal_structural_alignment"
                    if exact
                    else "observed_equal_optimal_alignments_under_search_bound"
                ),
            )
        )
    classes.sort(key=lambda item: item.class_id)
    return dict(sorted(invariant.items())), classes


def _mapping_differences(
    left: Mapping[str, str], right: Mapping[str, str]
) -> list[dict[str, Optional[str]]]:
    differences: list[dict[str, Optional[str]]] = []
    for agent_id in sorted(set(left) | set(right)):
        left_value = left.get(agent_id)
        right_value = right.get(agent_id)
        if left_value == right_value:
            continue
        differences.append(
            {
                "agent_id": agent_id,
                "left_truth_id": left_value,
                "right_truth_id": right_value,
            }
        )
    return differences


def _fraction_to_float(value: Fraction) -> float:
    """Convert an exact objective fraction for Pydantic/report serialization."""
    return value.numerator / value.denominator


def _objective_model(solution: _Solution) -> JointStructuralObjective:
    components: dict[str, JointStructuralComponent] = {}
    for name, (
        matched,
        agent_count,
        truth_count,
        agreement,
    ) in solution.components.items():
        components[name] = JointStructuralComponent(
            matched_count=matched,
            agent_count=agent_count,
            truth_count=truth_count,
            agreement=_fraction_to_float(agreement),
            weight=_fraction_to_float(_COMPONENT_WEIGHT),
            contribution=_fraction_to_float(agreement * _COMPONENT_WEIGHT),
        )
    total = sum(
        (
            agreement * _COMPONENT_WEIGHT
            for _, _, _, agreement in solution.components.values()
        ),
        Fraction(0, 1),
    )
    max_score = _COMPONENT_WEIGHT * len(components)
    return JointStructuralObjective(
        total_score=_fraction_to_float(total),
        max_score=_fraction_to_float(max_score),
        normalized_score=(_fraction_to_float(total / max_score) if max_score else 1.0),
        components=components,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def evaluate_joint_structural_mapping_objective(
    agent: AgentGraph,
    truth,
    *,
    node_mapping: Mapping[str, str],
    concept_mapping: Mapping[str, str],
    max_edge_search_states: int = 1_000_000,
) -> JointStructuralObjective:
    """Evaluate the existing objective for a fixed mapping, without searching.

    This helper is for evaluator-private counterfactual diagnostics.  It calls
    the same objective implementation as the joint search and does not add a
    candidate, alter admissibility, or participate in tie-breaking.  The
    fixed mappings are intentionally accepted as-is so an audit can measure
    what would happen if a production mapping were forced, including an
    unsupported assignment.
    """
    if max_edge_search_states <= 0:
        raise ValueError("max_edge_search_states must be positive")
    agent_index = _build_index(agent)
    truth_index = _build_index(truth)
    search = _MutableSearch(
        max_node_states=1,
        max_concept_states=max_edge_search_states,
        max_alternatives=1,
    )
    solution = _solution_for_mapping(
        agent,
        truth,
        agent_index,
        truth_index,
        dict(node_mapping),
        dict(concept_mapping),
        search,
    )
    if search.bound_hit:
        raise ValueError(
            "fixed mapping objective evaluation exceeded the edge search bound"
        )
    return _objective_model(solution)


def build_joint_structural_alignment_diagnostics(
    agent: AgentGraph,
    truth,
    *,
    production_node_to_truth: Mapping[str, str] | None = None,
    production_concept_to_truth: Mapping[str, str] | None = None,
    usage_concept_to_truth: Mapping[str, str] | None = None,
    node_candidate_allowlist: Mapping[str, Iterable[str]] | None = None,
    max_node_search_states: int = 250_000,
    max_concept_search_states: int = 1_000_000,
    max_optimal_alternatives: int = 2_048,
) -> JointStructuralAlignmentDiagnostics:
    """Search a diagnostic-only joint structural alignment.

    The optional production/usage mappings are copied into comparison fields
    *after* the structural search.  They never constrain candidates, score
    pairs, or break ties.  This makes the matcher independent of the
    production lexical and conditioned-usage experiments while allowing one
    private diagnostic section to compare all three mappings.

    ``node_candidate_allowlist`` is an additional diagnostic-only restriction.
    It is intended for representation experiments such as a fixed virtual
    boundary anchor; the production call path leaves it as ``None``.  The
    allowlist filters structural candidate pairs before the unchanged joint
    objective/search is run and never reads labels or concept text.
    """
    if max_node_search_states <= 0 or max_concept_search_states <= 0:
        raise ValueError("search state limits must be positive")
    if max_optimal_alternatives <= 0:
        raise ValueError("max_optimal_alternatives must be positive")

    agent_index = _build_index(agent)
    truth_index = _build_index(truth)
    candidates, potentials = _node_candidates(
        agent_index,
        truth_index,
        allowed_node_pairs=node_candidate_allowlist,
    )
    search = _MutableSearch(
        max_node_states=max_node_search_states,
        max_concept_states=max_concept_search_states,
        max_alternatives=max_optimal_alternatives,
        node_candidate_pair_count=len(potentials),
    )

    # Process constrained nodes first; candidate ordering is structural only.
    node_order = sorted(
        agent_index.node_ids,
        key=lambda node_id: (len(candidates[node_id]), node_id),
    )
    best_objective: Optional[Fraction] = None
    optimal_solutions: dict[tuple, _Solution] = {}

    def consider_node_mapping(node_mapping: dict[str, str]) -> None:
        nonlocal best_objective, optimal_solutions
        search.node_leaves_evaluated += 1
        support, condition_possible = _pair_support(
            agent, truth, agent_index, truth_index, node_mapping
        )
        per_kind_maps: list[list[dict[str, str]]] = []
        for kind in sorted(
            set(agent_index.concept_ids_by_kind) | set(truth_index.concept_ids_by_kind)
        ):
            left = list(agent_index.concept_ids_by_kind.get(kind, []))
            right = list(truth_index.concept_ids_by_kind.get(kind, []))
            if kind == "condition":
                allowed = _condition_pair_allowed(
                    condition_possible, agent_index, truth_index
                )
                maps, truncated = _enumerate_candidate_assignments(
                    left, right, allowed, search
                )
                if truncated:
                    search.hit("condition_assignment_bound")
                per_kind_maps.append(maps)
                continue
            weights = {
                pair: value
                for pair, value in _pair_weights(
                    support, agent, truth, agent_index, truth_index
                ).items()
                if pair[0] in left and pair[1] in right
            }
            result = _enumerate_best_additive_assignments(weights, left, right, search)
            if result.truncated:
                search.hit("additive_concept_assignment_bound")
            per_kind_maps.append(result.mappings)

        combinations: list[dict[str, str]] = [{}]
        for kind_maps in per_kind_maps:
            next_combinations: list[dict[str, str]] = []
            for base in combinations:
                for extra in kind_maps:
                    merged = dict(base)
                    merged.update(extra)
                    next_combinations.append(merged)
                    if len(next_combinations) >= max_optimal_alternatives:
                        search.hit("joint_alternative_limit")
                        break
                if len(next_combinations) >= max_optimal_alternatives:
                    break
            combinations = next_combinations or [{}]

        for concept_mapping in combinations:
            solution = _solution_for_mapping(
                agent,
                truth,
                agent_index,
                truth_index,
                node_mapping,
                concept_mapping,
                search,
            )
            if best_objective is None or solution.objective > best_objective:
                best_objective = solution.objective
                optimal_solutions = {_mapping_key(solution): solution}
            elif solution.objective == best_objective:
                key = _mapping_key(solution)
                if key not in optimal_solutions:
                    if len(optimal_solutions) < max_optimal_alternatives:
                        optimal_solutions[key] = solution
                    else:
                        search.hit("optimal_alternative_limit")

    def visit_node_mappings(
        position: int,
        node_mapping: dict[str, str],
        used_truth_nodes: set[str],
    ) -> None:
        if search.node_search_states >= search.max_node_states:
            search.hit("node_search_state_limit")
            return
        search.node_search_states += 1
        if position == len(node_order):
            consider_node_mapping(node_mapping)
            return
        agent_node_id = node_order[position]
        for truth_node_id in candidates[agent_node_id]:
            if truth_node_id in used_truth_nodes:
                continue
            node_mapping[agent_node_id] = truth_node_id
            used_truth_nodes.add(truth_node_id)
            visit_node_mappings(position + 1, node_mapping, used_truth_nodes)
            used_truth_nodes.remove(truth_node_id)
            del node_mapping[agent_node_id]
        # Unmatched Agent nodes are allowed and are explored after structural
        # candidates so a full exact reconstruction is found early.
        visit_node_mappings(position + 1, node_mapping, used_truth_nodes)

    visit_node_mappings(0, {}, set())

    if not optimal_solutions:
        # This is reachable only when a search bound fires before its first
        # leaf.  Return an explicit empty diagnostic rather than fabricating a
        # mapping or claiming an optimum.
        empty_objective = JointStructuralObjective(
            total_score=0.0,
            max_score=len(_RELATIONS) + 5,
            normalized_score=0.0,
        )
        selected_solution = _Solution(
            objective=Fraction(0, 1),
            components={},
            node_mapping={},
            concept_mapping={},
            edge_mapping={},
        )
    else:
        selected_solution = sorted(optimal_solutions.values(), key=_mapping_key)[0]
        empty_objective = _objective_model(selected_solution)

    exact_search = not search.bound_hit
    solutions = sorted(optimal_solutions.values(), key=_mapping_key)
    invariant_nodes, node_ambiguities = _ambiguity_classes(
        solutions,
        "node",
        agent_index.node_ids,
        exact=exact_search,
    )
    invariant_concepts, concept_ambiguities = _ambiguity_classes(
        solutions,
        "concept",
        agent_index.concept_ids,
        exact=exact_search,
    )

    global_max_objective = Fraction(len(_RELATIONS) + 5, 1)
    incumbent_objective = best_objective or Fraction(0, 1)
    search_stats = JointStructuralSearchStats(
        exact_search=exact_search,
        bound_hit=search.bound_hit,
        bound_reason=(
            ";".join(sorted(search.bound_reasons or set()))
            if search.bound_hit
            else None
        ),
        max_node_search_states=max_node_search_states,
        max_concept_search_states=max_concept_search_states,
        max_optimal_alternatives=max_optimal_alternatives,
        node_candidate_pair_count=search.node_candidate_pair_count,
        node_search_states=search.node_search_states,
        node_leaves_evaluated=search.node_leaves_evaluated,
        node_states_pruned=search.node_states_pruned,
        concept_search_states=search.concept_search_states,
        concept_states_pruned=search.concept_states_pruned,
        concept_assignments_evaluated=search.concept_assignments_evaluated,
        edge_assignment_states=search.edge_assignment_states,
        optimal_solution_count=len(solutions),
        optimum_is_unique=exact_search and len(solutions) == 1,
        optimal_solution_count_is_exact=exact_search,
        optimal_solution_count_is_lower_bound=False,
        objective_lower_bound=_fraction_to_float(incumbent_objective),
        objective_upper_bound=_fraction_to_float(
            incumbent_objective if exact_search else global_max_objective
        ),
        runtime_estimate_ms=max(
            1,
            (
                search.node_search_states
                + search.concept_search_states
                + search.edge_assignment_states
            )
            // 1000,
        )
        if search.node_search_states
        or search.concept_search_states
        or search.edge_assignment_states
        else 0,
    )

    production_nodes = dict(sorted((production_node_to_truth or {}).items()))
    production_concepts = dict(sorted((production_concept_to_truth or {}).items()))
    usage_concepts = dict(sorted((usage_concept_to_truth or {}).items()))
    return JointStructuralAlignmentDiagnostics(
        representative_agent_node_to_truth_node=selected_solution.node_mapping,
        representative_agent_concept_to_truth_concept=selected_solution.concept_mapping,
        representative_agent_edge_to_truth_edge=selected_solution.edge_mapping,
        invariant_agent_node_to_truth_node=invariant_nodes,
        invariant_agent_concept_to_truth_concept=invariant_concepts,
        invariants_proven=exact_search,
        node_ambiguity_classes=node_ambiguities,
        concept_ambiguity_classes=concept_ambiguities,
        unmatched_representative_agent_nodes=sorted(
            set(agent_index.node_ids) - set(selected_solution.node_mapping)
        ),
        unmatched_representative_truth_nodes=sorted(
            set(truth_index.node_ids) - set(selected_solution.node_mapping.values())
        ),
        unmatched_representative_agent_concepts=sorted(
            set(agent_index.concept_ids) - set(selected_solution.concept_mapping)
        ),
        unmatched_representative_truth_concepts=sorted(
            set(truth_index.concept_ids)
            - set(selected_solution.concept_mapping.values())
        ),
        objective=empty_objective,
        search=search_stats,
        production_agent_node_to_truth_node=production_nodes,
        production_agent_concept_to_truth_concept=production_concepts,
        conditioned_usage_agent_concept_to_truth_concept=usage_concepts,
        production_vs_joint_node_differences=_mapping_differences(
            production_nodes, selected_solution.node_mapping
        ),
        production_vs_joint_concept_differences=_mapping_differences(
            production_concepts, selected_solution.concept_mapping
        ),
        usage_vs_joint_concept_differences=_mapping_differences(
            usage_concepts, selected_solution.concept_mapping
        ),
    )


__all__ = [
    "JointStructuralAlignmentDiagnostics",
    "JointStructuralAmbiguityClass",
    "JointStructuralComponent",
    "JointStructuralObjective",
    "JointStructuralSearchStats",
    "evaluate_joint_structural_mapping_objective",
    "build_joint_structural_alignment_diagnostics",
]
