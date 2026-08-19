"""Evaluator for the open-world business_interview benchmark (v8 — unified
glossary, provenance-only).

The agent's inferred ``BusinessProcessGraph`` is compared to the scenario Truth
graph — both use the same class. **Correctness is grounded ONLY through private
provenance**:

    ConceptRef
      -> EvidenceRef (observation_id, quote, occurrence)
      -> exact Observation span
      -> private stakeholder assertion (fact_id, quote, occurrence)
      -> StakeholderFact
      -> TruthClaim (subject_kind, subject_id, property, concept_id)
      -> Truth BusinessConcept

The evaluator performs **no semantic NLP**: it never inspects
``preferred_label``, ``description``, ``ConceptTerm.text``, ``Observation.text``
or ``EvidenceRef.quote`` for meaning; there are no action-expression lists, no
predicate/necessity expression lists, no actor/system alias tables, no token
overlap, no synonym dictionaries, no WordNet, no embeddings and no evaluator
LLM. Quotes are matched to private assertions by exact string span only.

Node identity is established by the agent node's **activity** claim binding;
edge identity by from/to structure + the private **edge_exists** claim. Concept
identity is a per-kind unique perfect matching between used Agent concepts and
visible Truth concepts. Visibility stays prior: hidden unset -> correct,
hidden asserted -> incorrect (private provenance never rescues a hidden
assertion). Interview completion additionally requires that every referenced
Agent concept is no longer ``hypothesized``.
"""

from typing import Optional

from pydantic import BaseModel

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.facts import (
    StakeholderAssertion,
    StakeholderFact,
    message_contains_span,
)
from tau2.domains.business_interview.graph import (
    BusinessConcept,
    BusinessProcessGraph,
    ConceptRef,
    EvidenceRef,
    InterviewDB,
)
from tau2.domains.business_interview.stakeholder import StakeholderFilter

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")


class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    v8: the spec carries NO semantic matchers. Node/edge/property correctness
    is expressed entirely through hidden TruthClaims + private
    StakeholderFacts. Kept as a model for API stability; scenarios may leave
    it empty.
    """


class EvaluationResult(BaseModel):
    protocol_completed: bool
    graph_created: bool
    graph_valid: bool

    node_recall: float
    node_precision: float
    edge_recall: float
    edge_precision: float
    activity_correctness: float
    actor_correctness: float
    system_correctness: float
    read_correctness: float
    write_correctness: float
    rationale_correctness: float
    condition_correctness: float
    concept_correctness: float
    unsupported_ref_count: int
    fabricated_node_count: int
    fabricated_edge_count: int

    # glossary completion
    glossary_pass: bool
    referenced_hypothesized_concepts: list[str]

    # evidence hygiene
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    invalid_observation_reference_count: int
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool
    evidence_pass: bool

    structural_pass: bool
    protocol_pass: bool
    quality_pass: bool


# ---------------------------------------------------------------------------
# Private metadata validation (deterministic)
# ---------------------------------------------------------------------------


def _validate_assertions(
    assertions: dict[int, list[StakeholderAssertion]],
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    stakeholder: Optional[StakeholderFilter],
    messages: list[dict],
) -> None:
    """Deterministically reject invalid private assertion metadata.

    Every assertion's fact must exist (existence implies it belongs to this
    stakeholder), every supported TruthClaim must exist and be
    stakeholder-visible, and the quote/occurrence must exactly match the
    stakeholder message at that turn. Raises ``ValueError``.
    """
    for turn, turn_assertions in (assertions or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for assertion in turn_assertions:
            fact = (facts or {}).get(assertion.fact_id)
            if fact is None:
                raise ValueError(
                    f"assertion fact_id {assertion.fact_id!r} (turn {turn}) is not "
                    f"in the scenario's StakeholderFact catalog"
                )
            for cid in fact.supported_claim_ids:
                claim = (claims or {}).get(cid)
                if claim is None:
                    raise ValueError(
                        f"fact {assertion.fact_id!r} (turn {turn}) supports unknown "
                        f"TruthClaim id {cid!r}"
                    )
                if stakeholder is not None and not _claim_visible(claim, stakeholder):
                    raise ValueError(
                        f"fact {assertion.fact_id!r} (turn {turn}) supports claim "
                        f"{cid!r} which is outside stakeholder visibility"
                    )
            if not message_contains_span(
                message, assertion.quote, assertion.occurrence
            ):
                raise ValueError(
                    f"assertion {assertion.fact_id!r} (turn {turn}): quote "
                    f"{assertion.quote!r} occurrence {assertion.occurrence} does "
                    f"not exactly match the stakeholder message"
                )


def _claim_visible(claim: TruthClaim, stakeholder: StakeholderFilter) -> bool:
    if claim.subject_kind == "node":
        return claim.property in stakeholder.node_properties_for(claim.subject_id)
    if claim.subject_id not in stakeholder.visible_edge_ids:
        return False
    if claim.property == "condition":
        return "condition" in stakeholder.edge_properties_for(claim.subject_id)
    return claim.property == "edge_exists"


# ---------------------------------------------------------------------------
# Evidence chain (provenance only)
# ---------------------------------------------------------------------------


def _obs_by_id(db: InterviewDB, obs_id: str):
    for obs in db.observations:
        if obs.id == obs_id:
            return obs
    return None


def _ref_evidence(agent: BusinessProcessGraph, ref: ConceptRef) -> list[EvidenceRef]:
    """The evidence cited by a ref: the ref's own plus its concept's terms and
    validation evidence. Only these spans are inspected — never their meaning.
    """
    out = list(ref.evidence)
    concept = agent.concepts.get(ref.concept_id)
    if concept is not None:
        for term in concept.terms:
            out.extend(term.evidence)
        out.extend(concept.validation_evidence)
    return out


def _supported_claims(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    ref: ConceptRef,
    subject_kind: str,
    subject_id: str,
    property: str,
) -> list[TruthClaim]:
    """The TruthClaims at this subject+property supported by one ConceptRef.

    Chain: EvidenceRef -> exact Observation span -> private assertion ->
    StakeholderFact -> TruthClaims (kept to those expected at this slot).
    Nothing about the quote's meaning is ever inspected.
    """
    expected = [
        c
        for c in claims.values()
        if (subject_kind == "*" or c.subject_kind == subject_kind)
        and (subject_id in (None, "*") or c.subject_id == subject_id)
        and (property == "*" or c.property == property)
    ]
    if not expected:
        return []
    supported: set[str] = set()
    for ev in _ref_evidence(agent, ref):
        obs = _obs_by_id(db, ev.observation_id)
        if obs is None:
            continue
        for assertion in assertions.get(obs.turn, []):
            if assertion.quote == ev.quote and assertion.occurrence == ev.occurrence:
                fact = facts.get(assertion.fact_id)
                if fact is not None:
                    supported.update(fact.supported_claim_ids)
    return [c for c in expected if c.id in supported]


# ---------------------------------------------------------------------------
# Node / edge identity
# ---------------------------------------------------------------------------


def _match_nodes(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
) -> dict[str, str]:
    """Map agent node id -> truth node id through the activity claim binding.

    An agent node maps to a Truth node when its activity ref's provenance
    supports exactly that node's activity claim. Ambiguous or unsupported
    activity refs leave the node unmapped. Assignments are injective
    (deterministic sorted order).
    """
    candidates: dict[str, set[str]] = {}
    for nid, node in agent.nodes.items():
        supported = _supported_claims(
            agent, db, facts, claims, assertions, node.activity, "node", "*", "activity"
        )
        tnids = {c.subject_id for c in supported if c.subject_id != "*"}
        if len(tnids) == 1:
            candidates[nid] = tnids
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for nid in sorted(candidates):
        tnid = next(iter(candidates[nid]))
        if tnid in used:
            continue
        mapping[nid] = tnid
        used.add(tnid)
    return mapping


def _truth_edge_between(truth: BusinessProcessGraph, tf: str, tt: str):
    for e in truth.edges.values():
        if e.from_node == tf and e.to_node == tt:
            return e
    return None


# ---------------------------------------------------------------------------
# Property scoring
# ---------------------------------------------------------------------------


def _property_score(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    tnid: str,
    anid: str,
    prop: str,
    visible: bool,
) -> tuple[float, int]:
    """Score one node property (activity/actor/system/reads/writes/rationale).

    Visible property: recall over expected hidden claims (each grounded by at
    least one agent ref) times precision over agent refs (each must support at
    least one expected claim). Hidden property (prior gate): correct only when
    nothing is asserted — private provenance never rescues a hidden assertion.
    Returns (score, unsupported_ref_count).
    """
    refs = agent.nodes[anid].asserted_refs(prop)
    if not visible:
        return (1.0 if not refs else 0.0), 0
    expected = [
        c
        for c in claims.values()
        if c.subject_kind == "node" and c.subject_id == tnid and c.property == prop
    ]
    if not expected:
        return (1.0 if not refs else 0.0), 0
    if not refs:
        return 0.0, 0
    grounded = {c.id: False for c in expected}
    valid = 0
    unsupported = 0
    for ref in refs:
        supported = _supported_claims(
            agent, db, facts, claims, assertions, ref, "node", tnid, prop
        )
        if not supported:
            unsupported += 1
            continue
        valid += 1
        for c in supported:
            grounded[c.id] = True
    recall = sum(grounded.values()) / len(expected)
    precision = valid / len(refs)
    return recall * precision, unsupported


# ---------------------------------------------------------------------------
# Concept identity (per kind, unique perfect matching)
def _concept_bindings(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    mapping: dict[str, str],
    stakeholder: Optional[StakeholderFilter],
) -> float:
    """Concept-level binding integrity across all matched visible slots.

    For every ConceptKind:
    - one Agent concept may bind to only one Truth data concept of its own
      kind (incompatible kinds and ambiguous bindings are violations);
    - one visible Truth concept must be represented by one Agent concept
      (splits fail until merged; merging distinct Truth concepts fails);
    - the binding must be a UNIQUE perfect matching per kind.

    Returns concept_correctness in [0, 1].
    """
    # candidate truth concepts per agent concept: intersection over its
    # visible-slot refs of the supported claim concepts
    candidates: dict[str, set[str]] = {}
    referenced: set[str] = set()
    for anid, tnid in mapping.items():
        visible = (
            set(_NODE_PROPS)
            if stakeholder is None
            else stakeholder.node_properties_for(tnid)
        )
        for prop in visible:
            for ref in agent.nodes[anid].asserted_refs(prop):
                referenced.add(ref.concept_id)
                supported = _supported_claims(
                    agent, db, facts, claims, assertions, ref, "node", tnid, prop
                )
                truth_ids = {c.concept_id for c in supported if c.concept_id}
                if truth_ids:
                    prev = candidates.setdefault(ref.concept_id, set(truth_ids))
                    candidates[ref.concept_id] = prev & truth_ids
    # kind of a truth concept = kind of the claims that reference it
    truth_kind: dict[str, str] = {}
    property_kind: dict[str, str] = {
        "activity": "activity",
        "actor": "actor",
        "system": "system",
        "reads": "data",
        "writes": "data",
        "condition": "condition",
        "rationale": "rationale",
    }
    for c in claims.values():
        if c.concept_id is not None:
            truth_kind.setdefault(c.concept_id, property_kind.get(c.property, "data"))

    # every referenced agent concept must bind to EXACTLY ONE Truth concept of
    # its own kind (empty => unsupported/unassignable; >1 => ambiguous/merged
    # distinct Truth concepts; kind mismatch => incompatible binding)
    agent_to_truth: dict[str, str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts.get(cid)
        if concept is None:
            return 0.0
        cand = candidates.get(cid, set())
        if len(cand) != 1:
            return 0.0
        truth_id = next(iter(cand))
        if truth_kind.get(truth_id) != concept.kind:
            return 0.0
        agent_to_truth[cid] = truth_id

    # injective + coverage per kind: one visible Truth concept is represented
    # by exactly one Agent concept (splits fail; missing concepts fail)
    truth_by_kind: dict[str, list[str]] = {}
    for t, k in truth_kind.items():
        truth_by_kind.setdefault(k, []).append(t)
    by_kind: dict[str, list[str]] = {}
    for cid, tid in agent_to_truth.items():
        by_kind.setdefault(agent.concepts[cid].kind, []).append(cid)
    for kind, agents in by_kind.items():
        covered = {agent_to_truth[c] for c in agents}
        if len(covered) != len(agents):
            return 0.0  # two Agent concepts bound the same Truth concept
        truths = set(truth_by_kind.get(kind, []))
        if covered != truths:
            return 0.0  # a visible Truth concept has no Agent identity
    return 1.0


def _truth_concept_by_id(concept_id: str) -> BusinessConcept:
    """Placeholder resolver kept for symmetry; kinds come from claim
    properties (see ``_concept_bindings``)."""
    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Evidence hygiene
# ---------------------------------------------------------------------------


def _all_evidence_refs(agent: BusinessProcessGraph) -> list[EvidenceRef]:
    out: list[EvidenceRef] = []
    for node in agent.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                out.extend(ref.evidence)
        for concept_id in {
            r.concept_id for prop in _NODE_PROPS for r in node.refs(prop)
        }:
            concept = agent.concepts.get(concept_id)
            if concept is not None:
                for term in concept.terms:
                    out.extend(term.evidence)
                out.extend(concept.validation_evidence)
    for edge in agent.edges.values():
        out.extend(edge.evidence)
        if edge.condition is not None:
            out.extend(edge.condition.evidence)
    return out


def _evidence_metrics(
    db: InterviewDB,
    agent: BusinessProcessGraph,
    all_obs_ids: set[str],
) -> tuple[int, int, float, float, float]:
    """(invalid_evidence, invalid_obs_refs, node_cov, ref_cov, edge_cov)."""
    invalid_evidence = 0
    invalid_obs_refs = 0
    ref_total = ref_hit = 0
    node_total = node_hit = 0
    edge_total = edge_hit = 0

    obs_text = {o.id: o.text for o in db.observations}

    def span_ok(ev: EvidenceRef) -> bool:
        nonlocal invalid_evidence, invalid_obs_refs
        text = obs_text.get(ev.observation_id)
        if text is None:
            invalid_evidence += 1
            if ev.observation_id not in all_obs_ids:
                invalid_obs_refs += 1
            return False
        if not _span_in(text, ev.quote, ev.occurrence):
            invalid_evidence += 1
            return False
        return True

    for node in agent.nodes.values():
        node_refs: list[ConceptRef] = []
        for prop in _NODE_PROPS:
            node_refs.extend(node.refs(prop))
        node_total += 1
        if any(
            r.asserted and any(span_ok(ev) for ev in _ref_evidence(agent, r))
            for r in node_refs
        ):
            node_hit += 1
        for r in node_refs:
            if not r.asserted:
                continue
            ref_total += 1
            evs = _ref_evidence(agent, r)
            if evs and all(span_ok(ev) for ev in evs):
                ref_hit += 1
    for edge in agent.edges.values():
        edge_total += 1
        if edge.evidence and all(span_ok(ev) for ev in edge.evidence):
            edge_hit += 1
        for ev in edge.evidence:
            span_ok(ev)
        if edge.condition is not None:
            for ev in _ref_evidence(agent, edge.condition):
                span_ok(ev)

    node_cov = node_hit / node_total if node_total else 1.0
    ref_cov = ref_hit / ref_total if ref_total else 1.0
    edge_cov = edge_hit / edge_total if edge_total else 1.0
    return invalid_evidence, invalid_obs_refs, node_cov, ref_cov, edge_cov


def _span_in(text: str, quote: str, occurrence: int) -> bool:
    if not quote:
        return False
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start == -1:
            return False
    return True


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate(
    db: InterviewDB,
    truth: BusinessProcessGraph,
    spec: EvaluationSpec,
    stakeholder: Optional[StakeholderFilter] = None,
    *,
    claims: Optional[dict[str, TruthClaim]] = None,
    facts: Optional[dict[str, StakeholderFact]] = None,
    assertions: Optional[dict[int, list[StakeholderAssertion]]] = None,
) -> EvaluationResult:
    """Evaluate the inferred graph against the hidden Truth.

    ``claims`` is the hidden TruthClaim catalog (``Scenario.claims``),
    ``facts`` the private StakeholderFact catalog (``Scenario.facts``) and
    ``assertions`` the private sidecar ledger (``{turn: [assertions]}``,
    ``StakeholderFactLedger.assertions()``). All three are evaluator-only; the
    Agent never sees them. Invalid private metadata is rejected
    deterministically. Without provenance, no ConceptRef can ground a claim.
    """
    agent = db.graph if db.graph is not None else BusinessProcessGraph()
    protocol = db.interview_complete
    graph_created = len(agent.nodes) > 0
    graph_valid = agent.is_valid
    claims = claims or {}
    facts = facts or {}
    ledger = assertions or {}
    _validate_assertions(ledger, facts, claims, stakeholder, db.messages)

    # ---- node mapping (activity claim binding) -----------------------------
    mapping = _match_nodes(agent, db, facts, claims, ledger)
    truth_node_ids = list(truth.nodes)
    agent_node_ids = list(agent.nodes)
    matched_truth = set(mapping.values())
    matched_agent = set(mapping.keys())
    node_recall = len(matched_truth) / len(truth_node_ids) if truth_node_ids else 0.0
    node_precision = len(matched_agent) / len(agent_node_ids) if agent_node_ids else 0.0
    fabricated_node_count = len(agent_node_ids) - len(matched_agent)

    # ---- edges -------------------------------------------------------------
    truth_edge_list = list(truth.edges.values())
    agent_edge_list = list(agent.edges.values())
    matched_agent_edges: set[str] = set()
    for eid, edge in agent.edges.items():
        tf = mapping.get(edge.from_node)
        tt = mapping.get(edge.to_node)
        if tf is None or tt is None:
            continue
        te = _truth_edge_between(truth, tf, tt)
        if te is None:
            continue
        supported = _edge_exists_supported(
            agent, db, facts, claims, ledger, edge, te.id
        )
        if supported:
            matched_agent_edges.add(eid)
    edge_recall = (
        len(matched_agent_edges) / len(truth_edge_list) if truth_edge_list else 1.0
    )
    edge_precision = (
        len(matched_agent_edges) / len(agent_edge_list) if agent_edge_list else 0.0
    )
    fabricated_edge_count = len(agent_edge_list) - len(matched_agent_edges)

    # ---- property correctness ----------------------------------------------
    hits = {p: 0.0 for p in _NODE_PROPS}
    unsupported_concept_ref_count = 0
    for anid, tnid in mapping.items():
        visible = (
            set(_NODE_PROPS)
            if stakeholder is None
            else stakeholder.node_properties_for(tnid)
        )
        for prop in _NODE_PROPS:
            score, unsup = _property_score(
                agent, db, facts, claims, ledger, tnid, anid, prop, prop in visible
            )
            hits[prop] += score
            unsupported_concept_ref_count += unsup
    nm = len(mapping) or 1
    activity_correctness = hits["activity"] / nm
    actor_correctness = hits["actor"] / nm
    system_correctness = hits["system"] / nm
    read_correctness = hits["reads"] / nm
    write_correctness = hits["writes"] / nm
    rationale_correctness = hits["rationale"] / nm

    # ---- edge conditions ----------------------------------------------------
    cond_hits = cond_total = 0
    for te in truth_edge_list:
        ae = None
        for eid in agent.edges:
            edge = agent.edges[eid]
            if (
                mapping.get(edge.from_node) == te.from_node
                and mapping.get(edge.to_node) == te.to_node
            ):
                ae = edge
                break
        expected = [
            c
            for c in claims.values()
            if c.subject_kind == "edge"
            and c.subject_id == te.id
            and c.property == "condition"
        ]
        if not expected:
            # no truth condition: an asserted condition has no provenance -> wrong
            cond_total += 1
            cond_hits += (
                1
                if (ae is None or ae.condition is None or not ae.condition.asserted)
                else 0
            )
            continue
        cond_total += 1
        if ae is None or ae.condition is None or not ae.condition.asserted:
            continue
        supported = _supported_claims(
            agent, db, facts, claims, ledger, ae.condition, "edge", te.id, "condition"
        )
        if supported:
            cond_hits += 1
    condition_correctness = cond_hits / cond_total if cond_total else 1.0

    # ---- concept identity ---------------------------------------------------
    concept_correctness = _concept_bindings(
        agent, db, facts, claims, ledger, mapping, stakeholder
    )

    # ---- glossary completion ------------------------------------------------
    referenced = agent.referenced_concepts()
    hypothesized = sorted(
        cid
        for cid in referenced
        if agent.concepts.get(cid) is not None
        and agent.concepts[cid].validation_status == "hypothesized"
    )
    glossary_pass = not hypothesized

    # ---- evidence hygiene ---------------------------------------------------
    all_obs_ids = {o.id for o in db.observations}
    authentic_obs_ids: set[str] = set()
    invalid_observation_source_count = 0
    for o in db.observations:
        m = db.messages[o.turn] if 0 <= o.turn < len(db.messages) else None
        if (
            m is not None
            and m.get("role") == "user"
            and (m.get("content") or "") == (o.text or "")
        ):
            authentic_obs_ids.add(o.id)
        else:
            invalid_observation_source_count += 1
    authentic_observation_count = len(authentic_obs_ids)
    orphan_observation_count = sum(
        1 for o in db.observations if o.id not in _all_referenced_observation_ids(agent)
    )

    (
        invalid_evidence_ref_count,
        invalid_observation_reference_count,
        node_evidence_coverage,
        ref_evidence_coverage,
        edge_evidence_coverage,
    ) = _evidence_metrics(db, agent, all_obs_ids)

    provenance_authenticity_pass = bool(
        invalid_evidence_ref_count == 0
        and invalid_observation_reference_count == 0
        and invalid_observation_source_count == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and node_evidence_coverage == 1.0
        and ref_evidence_coverage == 1.0
        and edge_evidence_coverage == 1.0
    )

    # ---- gates -------------------------------------------------------------
    structural_pass = bool(
        graph_created
        and graph_valid
        and node_recall == 1.0
        and node_precision == 1.0
        and edge_recall == 1.0
        and edge_precision == 1.0
        and activity_correctness == 1.0
        and actor_correctness == 1.0
        and system_correctness == 1.0
        and read_correctness == 1.0
        and write_correctness == 1.0
        and rationale_correctness == 1.0
        and condition_correctness == 1.0
        and concept_correctness == 1.0
        and glossary_pass
    )
    protocol_pass = protocol
    quality_pass = bool(structural_pass and evidence_pass)

    return EvaluationResult(
        protocol_completed=protocol,
        graph_created=graph_created,
        graph_valid=graph_valid,
        node_recall=node_recall,
        node_precision=node_precision,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        activity_correctness=activity_correctness,
        actor_correctness=actor_correctness,
        system_correctness=system_correctness,
        read_correctness=read_correctness,
        write_correctness=write_correctness,
        rationale_correctness=rationale_correctness,
        condition_correctness=condition_correctness,
        concept_correctness=concept_correctness,
        unsupported_ref_count=unsupported_concept_ref_count,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_pass=glossary_pass,
        referenced_hypothesized_concepts=hypothesized,
        node_evidence_coverage=node_evidence_coverage,
        ref_evidence_coverage=ref_evidence_coverage,
        edge_evidence_coverage=edge_evidence_coverage,
        invalid_evidence_ref_count=invalid_evidence_ref_count,
        invalid_observation_reference_count=invalid_observation_reference_count,
        authentic_observation_count=authentic_observation_count,
        invalid_observation_source_count=invalid_observation_source_count,
        orphan_observation_count=orphan_observation_count,
        provenance_authenticity_pass=provenance_authenticity_pass,
        evidence_pass=evidence_pass,
        structural_pass=structural_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
    )


def _edge_exists_supported(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    facts: dict[str, StakeholderFact],
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    edge,
    truth_edge_id: str,
) -> bool:
    """True when the agent edge's evidence supports the truth edge's
    ``edge_exists`` claim (provenance only)."""
    expected = [
        c
        for c in claims.values()
        if c.subject_kind == "edge"
        and c.subject_id == truth_edge_id
        and c.property == "edge_exists"
    ]
    if not expected:
        return False
    supported: set[str] = set()
    for ev in edge.evidence:
        obs = _obs_by_id(db, ev.observation_id)
        if obs is None:
            continue
        for assertion in assertions.get(obs.turn, []):
            if assertion.quote == ev.quote and assertion.occurrence == ev.occurrence:
                fact = facts.get(assertion.fact_id)
                if fact is not None:
                    supported.update(fact.supported_claim_ids)
    return expected[0].id in supported


def _all_referenced_observation_ids(agent: BusinessProcessGraph) -> set[str]:
    ids: set[str] = set()
    for node in agent.nodes.values():
        for prop in _NODE_PROPS:
            for ref in node.refs(prop):
                for ev in _ref_evidence(agent, ref):
                    ids.add(ev.observation_id)
    for edge in agent.edges.values():
        for ev in edge.evidence:
            ids.add(ev.observation_id)
        if edge.condition is not None:
            for ev in _ref_evidence(agent, edge.condition):
                ids.add(ev.observation_id)
    return ids
