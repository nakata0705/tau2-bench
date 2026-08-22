"""Evaluator for the graph-native business_interview benchmark (v12 — Truth
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

Epistemic belief recording keeps conservative semantics on the *creation*
side (see tools.py), but the *score* evaluates the final belief against Truth
without requiring conversational proof. For scoring, UNSET / ABSENT /
DONT_KNOW are all "no value claimed": a Truth value slot must be claimed as a
content-matching ConceptRef, a Truth-absent slot must not assert a value.

The private provenance ledger (annotations/alignments/terminology) and the
evidence-hygiene metrics remain as **diagnostics** only. They are reported
but never gate ``quality_pass``.

StakeholderKnowledge stays a simulator constraint, not the scored target: it
limits what the conversation can reveal, while the scored target is the Truth.
"""

import re
from typing import Optional

from pydantic import BaseModel

from tau2.domains.business_interview.graph import (
    AbsentType,
    AgentGraph,
    ConceptRef,
    DontKnowType,
    EvidenceRef,
    InterviewDB,
    is_dont_know,
)
from tau2.domains.business_interview.grounding import (
    grounded_ids as grounded_semantic_ids,
)
from tau2.domains.business_interview.grounding import (
    grounded_refs as resolve_grounding_refs,
)

# Re-exported provenance helpers kept for diagnostic callers (tests/diagnostics).
__all__ = [
    "EvaluationSpec",
    "EvaluationResult",
    "evaluate",
    "grounded_semantic_ids",
    "resolve_grounding_refs",
]

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")


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

    # glossary completion + genuine validation
    glossary_pass: bool
    glossary_complete: bool
    referenced_hypothesized_concepts: list[str]
    glossary_validation_errors: list[str]

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


# ---------------------------------------------------------------------------
# Content signatures (deterministic; never semantic NLP)
# ---------------------------------------------------------------------------


def _tokens(text: Optional[str]) -> set[str]:
    """Lowercased alphanumeric tokens (deterministic)."""
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def _similarity(a: set[str], b: set[str]) -> float:
    """Symmetric Jaccard similarity; 0 on empty/disjoint input."""
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / len(a | b)


def _truth_concept_tokens(concept) -> set[str]:
    toks: set[str] = set()
    for t in concept.canonical_terms:
        toks |= _tokens(t)
    if concept.description:
        toks |= _tokens(concept.description)
    return toks


def _agent_concept_tokens(concept) -> set[str]:
    toks = _tokens(concept.display_label)
    if concept.description:
        toks |= _tokens(concept.description)
    return toks


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


def _node_signature(node, concepts, is_truth: bool) -> set[str]:
    """Content signature of a node from its referenced concept contents."""
    toks: set[str] = set()
    for prop in _NODE_PROPS:
        for ref in node.refs(prop):
            cid = ref.concept_id
            if cid not in concepts:
                continue
            concept = concepts[cid]
            if is_truth:
                toks |= _truth_concept_tokens(concept)
            else:
                toks |= _agent_concept_tokens(concept)
    return toks


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
) -> tuple[float, float, dict[str, str]]:
    """Content-based bijective concept alignment.

    Expected = Truth concepts the Truth graph references. Attempted = agent
    concepts the Agent graph references. Each attempted concept aligns to the
    best-compatible (same kind, content-overlapping) Truth concept,
    bijectively. Returns ``(concept_recall, concept_precision,
    agent_concept_id -> truth_concept_id)``; precision is over the attempted
    set, recall over expected.
    """
    expected: set[str] = _truth_referenced_concept_ids(truth)
    attempted: set[str] = _agent_referenced_concept_ids(agent)

    candidates: dict[str, dict[str, float]] = {}
    for acid in attempted:
        ac = agent.concepts.get(acid)
        if ac is None:
            continue
        a_toks = _agent_concept_tokens(ac)
        scores: dict[str, float] = {}
        for tid in expected:
            tc = truth.concepts.get(tid)
            if tc is None or tc.kind != ac.kind:
                continue
            s = _similarity(a_toks, _truth_concept_tokens(tc))
            if s > 0.0:
                scores[tid] = s
        candidates[acid] = scores

    claimed: set[str] = set()
    recognized: dict[str, str] = {}
    for acid in sorted(candidates):
        pool = {tid: s for tid, s in candidates[acid].items() if tid not in claimed}
        if not pool:
            continue
        best = max(pool, key=lambda tid: (pool[tid], tid))
        recognized[acid] = best
        claimed.add(best)

    recalled = set(recognized.values()) & expected
    concept_recall = len(recalled) / len(expected) if expected else 1.0
    concept_precision = len(recognized) / len(attempted) if attempted else 1.0
    agent_to_truth = {acid: tid for acid, tid in recognized.items() if tid in expected}
    return concept_recall, concept_precision, agent_to_truth


def expected_attempted(attempted: set[str], agent: AgentGraph) -> set[str]:
    """Compat shim: the attempted referenced concept ids are the agent
    referenced concept ids."""
    return attempted


# ---------------------------------------------------------------------------
# Node / edge identity (content-based)
# ---------------------------------------------------------------------------


def _map_nodes_and_edges(
    agent: AgentGraph,
    truth,
) -> tuple[dict[str, str], dict[str, str]]:
    """Node mapping by content-signature similarity; edge mapping by matching
    endpoint pair."""
    agent_sigs = {
        anid: _node_signature(node, agent.concepts, is_truth=False)
        for anid, node in agent.nodes.items()
    }
    truth_sigs = {
        tnid: _node_signature(node, truth.concepts, is_truth=True)
        for tnid, node in truth.nodes.items()
    }
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for anid in sorted(agent_sigs):
        a = agent_sigs[anid]
        if not a:
            continue
        pool = {
            tnid: _similarity(a, t)
            for tnid, t in truth_sigs.items()
            if tnid not in used
        }
        if not pool:
            continue
        best = max(pool, key=lambda tnid: (pool[tnid], tnid))
        if pool[best] <= 0.0:
            continue
        mapping[anid] = best
        used.add(best)

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
    """1 if the agent's scalar belief matches the Truth slot.

    Truth ConceptRef -> the agent must claim a matching value concept.
    Truth None -> the agent must claim NO value (UNSET/ABSENT/DONT_KNOW or
    a fabricated value all score 0 is wrong).
    """
    tcid = _truth_scalar_value(truth_value)
    if tcid is not None:
        if not isinstance(agent_value, ConceptRef) or not agent_value.asserted:
            return 0
        return 1 if agent_to_truth.get(agent_value.concept_id) == tcid else 0
    # Truth has no value here: agent must claim none
    if isinstance(agent_value, ConceptRef) and agent_value.asserted:
        return 0
    return 1


def _score_list_slot(
    agent_value,
    truth_value,
    agent_to_truth: dict[str, str],
) -> tuple[float, int]:
    """Recall*precision over the read/write element set, plus unsupported."""
    expected: set[str] = (
        {ref.concept_id for ref in truth_value}
        if isinstance(truth_value, list)
        else set()
    )
    if not expected:
        if isinstance(agent_value, list) and agent_value:
            unsupported = sum(
                1
                for r in agent_value
                if r.asserted and agent_to_truth.get(r.concept_id) is not None
            )
            return 0.0, unsupported
        return 1.0, 0
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

    for nid, node in truth.nodes.items():
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
    for eid, edge in truth.edges.items():
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
    target = truth if truth is not None else knowledge.graph
    agent = db.graph if db.graph is not None else AgentGraph()
    protocol = db.interview_complete
    graph_created = len(agent.nodes) > 0
    graph_valid = agent.is_valid

    # concept alignment first (property + concept scoring need the mapping)
    concept_recall, concept_precision, agent_to_truth = _align_concepts(agent, target)

    mapping, edge_map = _map_nodes_and_edges(agent, target)

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

    start_correct = bool(
        agent.start_node_id is not None
        and mapping.get(agent.start_node_id) == target.start_node_id
    )
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
    # No provenance/grounding gate on the glossary: hypothesis states are Agent
    # belief records only. glossary_pass = reconstruction completeness.
    hypothesized: list[str] = []
    glossary_pass = glossary_complete
    glossary_claims = []

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
        and glossary_pass
        and glossary_complete
    )
    protocol_pass = protocol
    reconstruction_pass = structural_pass
    # quality = reconstruction ONLY; provenance is never a hard gate
    quality_pass = reconstruction_pass

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
        glossary_pass=glossary_pass,
        glossary_complete=glossary_complete,
        referenced_hypothesized_concepts=hypothesized,
        glossary_validation_errors=glossary_claims,
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
    )
