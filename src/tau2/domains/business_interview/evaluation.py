"""Evaluator for the open-world business_interview benchmark (v9 — graph
context, provenance-only).

The agent's inferred ``BusinessProcessGraph`` is compared to the scenario Truth
graph — both use the same class. **Correctness is grounded ONLY through private
provenance**:

    ConceptRef
      -> EvidenceRef (observation_id, quote, occurrence)
      -> concrete character span in the immutable Observation
      -> private stakeholder assertion (claim_id + span)
      -> TruthClaim (context_id = graph position, property, concept_id)
      -> Truth BusinessConcept

Spans **correspond** when one contains the other (deterministic character
relation). An evidence span that corresponds to assertions of several
different claims is **ambiguous and fails** (no cross-credit). The evaluator
performs no semantic NLP: no action expressions, no predicate/necessity
matchers, no actor/system aliases, no label/description/mention comparison.

**Node identity uses topology**: an agent node maps to a Truth node through a
deterministic consistent assignment of activity provenance (the node's
activity claim grounding) plus the reconstructed incoming topology — every
agent edge with edge-existence provenance must land on a Truth edge that
exists and matches, and edge conditions must be consistent. The same activity
may occur at several graph positions; topology disambiguates.

Concept identity is per-kind (activity/actor/system/data/condition/rationale):
every referenced agent concept must bind to exactly one Truth concept of its
own kind, one visible Truth concept must be represented by one Agent concept,
reuse across slots is required, and **edge conditions participate** in the
same identity validation. Visibility stays prior: hidden unset -> correct,
hidden asserted -> incorrect. Glossary completion: referenced concepts must be
resolved, and `confirmed`/`unknown`/`disputed` statuses must be backed by
appropriate stakeholder evidence (no bulk self-confirmation).
"""

from typing import Optional

from pydantic import BaseModel

from tau2.domains.business_interview.claims import TruthClaim
from tau2.domains.business_interview.facts import (
    StakeholderAssertion,
    message_contains_span,
)
from tau2.domains.business_interview.graph import (
    BusinessProcessGraph,
    ConceptRef,
    Edge,
    EvidenceRef,
    InterviewDB,
    spans_correspond,
)
from tau2.domains.business_interview.stakeholder import StakeholderFilter

_NODE_PROPS = ("activity", "actor", "system", "reads", "writes", "rationale")

# kind of a Truth concept = kind of the claims that reference it
_PROPERTY_KIND: dict[str, str] = {
    "activity": "activity",
    "actor": "actor",
    "system": "system",
    "reads": "data",
    "writes": "data",
    "condition": "condition",
    "rationale": "rationale",
}


class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    v9: the spec carries NO semantic matchers. Everything semantic is
    expressed as graph-contextual TruthClaims + private assertions. Kept as a
    model for API stability; scenarios may leave it empty.
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
    unsupported_ref_count: int
    fabricated_node_count: int
    fabricated_edge_count: int

    # glossary completion + genuine validation
    glossary_pass: bool
    referenced_hypothesized_concepts: list[str]
    glossary_validation_errors: list[str]

    # evidence hygiene
    node_evidence_coverage: float
    ref_evidence_coverage: float
    edge_evidence_coverage: float
    invalid_evidence_ref_count: int
    ambiguous_evidence_ref_count: int
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
    claims: dict[str, TruthClaim],
    stakeholder: Optional[StakeholderFilter],
    messages: list[dict],
) -> None:
    """Deterministically reject invalid private assertion metadata.

    Every assertion's claim must exist and be stakeholder-visible, and the
    quote/occurrence must exactly match the stakeholder message at that turn.
    Raises ``ValueError``.
    """
    for turn, turn_assertions in (assertions or {}).items():
        message = messages[turn].get("content") if 0 <= turn < len(messages) else None
        for assertion in turn_assertions:
            claim = (claims or {}).get(assertion.claim_id)
            if claim is None:
                raise ValueError(
                    f"assertion claim_id {assertion.claim_id!r} (turn {turn}) is "
                    f"not in the scenario's visible claim catalog"
                )
            if stakeholder is not None and not _claim_visible(claim, stakeholder):
                raise ValueError(
                    f"assertion claim_id {assertion.claim_id!r} (turn {turn}) is "
                    f"outside stakeholder visibility"
                )
            if not message_contains_span(
                message, assertion.quote, assertion.occurrence
            ):
                raise ValueError(
                    f"assertion {assertion.claim_id!r} (turn {turn}): quote "
                    f"{assertion.quote!r} occurrence {assertion.occurrence} does "
                    f"not exactly match the stakeholder message"
                )


def _claim_visible(claim: TruthClaim, stakeholder: StakeholderFilter) -> bool:
    if claim.context_id in stakeholder.visible_node_ids:
        return claim.property in stakeholder.node_properties_for(claim.context_id)
    if claim.context_id in stakeholder.visible_edge_ids:
        if claim.property == "condition":
            return "condition" in stakeholder.edge_properties_for(claim.context_id)
        return claim.property == "edge_exists"
    return False


# ---------------------------------------------------------------------------
# Span-based grounding (provenance only)
# ---------------------------------------------------------------------------


def _obs_by_id(db: InterviewDB, obs_id: str):
    for obs in db.observations:
        if obs.id == obs_id:
            return obs
    return None


def _ref_evidence(agent: BusinessProcessGraph, ref: ConceptRef) -> list[EvidenceRef]:
    """The evidence cited by a ref: the ref's own plus its concept's mentions
    and validation evidence. Only spans are inspected — never their meaning."""
    out = list(ref.evidence)
    concept = agent.concepts.get(ref.concept_id)
    if concept is not None:
        out.extend(concept.mentions)
        out.extend(concept.validation_evidence)
    return out


def _grounded_claim_ids(
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    evidence: list[EvidenceRef],
    context_id: str = "*",
    property: str = "*",
) -> tuple[set[str], int, int]:
    """Claims grounded by a list of EvidenceRefs via span correspondence,
    scoped to the expected claims at one slot (``context_id``/``property``).

    For each evidence span, the assertions of the same Observation whose spans
    correspond (containment either way) are collected. Among the **expected**
    claims at this slot, exactly one correspondence grounds that claim; two or
    more make the evidence span **ambiguous (it grounds nothing — no
    cross-credit within a slot)**; correspondences to claims of other slots
    are irrelevant here. A span that is not an exact span of the Observation
    is invalid.

    Returns (grounded claim ids, invalid count, ambiguous count).
    """
    expected = {
        c.id
        for c in claims.values()
        if (context_id == "*" or c.context_id == context_id)
        and (property == "*" or c.property == property)
    }
    grounded: set[str] = set()
    invalid = 0
    ambiguous = 0
    for ev in evidence:
        obs = _obs_by_id(db, ev.observation_id)
        if obs is None:
            invalid += 1
            continue
        ev_span = ev.resolve_span(obs.text)
        if ev_span is None:
            invalid += 1
            continue
        corresponding: set[str] = set()
        for assertion in assertions.get(obs.turn, []):
            ass_span = _resolve_assertion_span(obs.text, assertion)
            if ass_span is None:
                continue
            if spans_correspond(ev_span, ass_span):
                corresponding.add(assertion.claim_id)
        slot_candidates = corresponding & expected
        if len(slot_candidates) == 1:
            grounded.update(slot_candidates)
        elif len(slot_candidates) > 1:
            ambiguous += 1
    return grounded, invalid, ambiguous


def _resolve_assertion_span(
    text: str, assertion: StakeholderAssertion
) -> Optional[tuple[int, int]]:
    """Resolve an assertion's quote/occurrence to (start, end) in ``text``."""
    if not assertion.quote:
        return None
    start = -1
    for _ in range(assertion.occurrence + 1):
        start = text.find(assertion.quote, start + 1)
        if start == -1:
            return None
    return (start, start + len(assertion.quote))


def _corresponding_claim_ids(
    db: InterviewDB,
    assertions: dict[int, list[StakeholderAssertion]],
    evidence: list[EvidenceRef],
) -> set[str]:
    """ALL claim ids whose assertion spans correspond (containment) to the
    given evidence spans — no ambiguity filtering. Used for genuine
    confirmation checks; claim grounding uses ``_grounded_claim_ids``."""
    corresponding: set[str] = set()
    for ev in evidence:
        obs = _obs_by_id(db, ev.observation_id)
        if obs is None:
            continue
        ev_span = ev.resolve_span(obs.text)
        if ev_span is None:
            continue
        for assertion in assertions.get(obs.turn, []):
            ass_span = _resolve_assertion_span(obs.text, assertion)
            if ass_span is not None and spans_correspond(ev_span, ass_span):
                corresponding.add(assertion.claim_id)
    return corresponding


def _claims_at(
    claims: dict[str, TruthClaim],
    context_id: str,
    property: str,
) -> list[TruthClaim]:
    return [
        c
        for c in claims.values()
        if c.context_id == context_id and (property == "*" or c.property == property)
    ]


# ---------------------------------------------------------------------------
# Node / edge identity (provenance + topology)
# ---------------------------------------------------------------------------


def _condition_compatible(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    edge: Edge,
    truth_edge: Edge,
) -> bool:
    """True when the agent edge's condition (if asserted) does not contradict
    the truth edge (provenance only).

    The condition is INCOMPATIBLE only when its evidence positively grounds a
    condition claim of a DIFFERENT truth edge (it points at another branch).
    Condition evidence that grounds nothing is weak (penalized by condition
    scoring) but does not break the correspondence.
    """
    if edge.condition is None or not edge.condition.asserted:
        return True
    # ref-level evidence only: concept mentions participate in identity
    # validation, not in the topology correspondence
    grounded, _invalid, _amb = _grounded_claim_ids(
        db,
        claims,
        assertions,
        edge.condition.evidence,
        property="condition",
    )
    grounded_edges = {
        c.context_id
        for c in claims.values()
        if c.id in grounded and c.property == "condition"
    }
    return not (grounded_edges - {truth_edge.id})


def _match_nodes_and_edges(
    agent: BusinessProcessGraph,
    truth: BusinessProcessGraph,
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Deterministic node/edge correspondence.

    Node candidates come from activity-claim grounding; edge candidates from
    edge_exists-claim grounding. The assignment maximizes the number of
    satisfied topology constraints: an agent edge with edge-existence
    provenance is satisfied when the mapped endpoints land on a Truth edge
    whose edge_exists claim it supports, with conditions consistent. Ties are
    broken deterministically (lexicographic order). An agent edge whose
    topology contradicts its own provenance is scored as an unsupported edge
    (precision miss) instead of voiding the whole correspondence. The same
    activity at several Truth positions is disambiguated by topology.

    Returns (node mapping, agent edge id -> truth edge id).
    """
    tn: dict[str, set[str]] = {}
    for nid, node in agent.nodes.items():
        grounded, _i, _a = _grounded_claim_ids(
            db,
            claims,
            assertions,
            _ref_evidence(agent, node.activity),
            property="activity",
        )
        tnids = {c.context_id for c in claims.values() if c.id in grounded}
        if tnids:
            tn[nid] = tnids
    te: dict[str, set[str]] = {}
    for eid, edge in agent.edges.items():
        grounded, _i, _a = _grounded_claim_ids(
            db, claims, assertions, edge.evidence, property="edge_exists"
        )
        teids = {c.context_id for c in claims.values() if c.id in grounded}
        if teids:
            te[eid] = teids

    order = sorted(tn)

    def edge_satisfied(mapping: dict[str, str], eid: str, edge: Edge) -> bool:
        a = mapping.get(edge.from_node)
        b = mapping.get(edge.to_node)
        if a is None or b is None:
            return False
        return any(
            truth.edges[t].from_node == a
            and truth.edges[t].to_node == b
            and _condition_compatible(
                agent, db, claims, assertions, edge, truth.edges[t]
            )
            for t in te.get(eid, set())
            if t in truth.edges
        )

    best_mapping: dict[str, str] = {}
    best_score = -1

    def enumerate_assignments(i: int, mapping: dict[str, str]) -> None:
        nonlocal best_mapping, best_score
        if i == len(order):
            score = sum(
                1
                for eid, edge in agent.edges.items()
                if edge_satisfied(mapping, eid, edge)
            )
            if score > best_score:
                best_score = score
                best_mapping = dict(mapping)
            return
        nid = order[i]
        for tnid in sorted(tn[nid]):
            if tnid in mapping.values():
                continue
            mapping[nid] = tnid
            enumerate_assignments(i + 1, mapping)
            del mapping[nid]

    if order:
        enumerate_assignments(0, {})

    mapping = best_mapping
    # agent edge -> truth edge assignment (matching the chosen mapping)
    edge_map: dict[str, str] = {}
    for eid, edge in agent.edges.items():
        a = mapping.get(edge.from_node)
        b = mapping.get(edge.to_node)
        if a is None or b is None:
            continue
        for t in sorted(te.get(eid, set())):
            te_obj = truth.edges.get(t)
            if (
                te_obj is not None
                and te_obj.from_node == a
                and te_obj.to_node == b
                and _condition_compatible(agent, db, claims, assertions, edge, te_obj)
            ):
                edge_map[eid] = t
                break
    return mapping, edge_map


# ---------------------------------------------------------------------------
# Property scoring
# ---------------------------------------------------------------------------


def _property_score(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    tnid: str,
    anid: str,
    prop: str,
    visible: bool,
) -> tuple[float, int]:
    """Score one node property (activity/actor/system/reads/writes/rationale).

    Visible property: recall over expected hidden claims (each grounded by at
    least one agent ref) times precision over agent refs (each must ground at
    least one expected claim). Hidden property (prior gate): correct only when
    nothing is asserted. Returns (score, unsupported_ref_count).
    """
    refs = agent.nodes[anid].asserted_refs(prop)
    if not visible:
        return (1.0 if not refs else 0.0), 0
    expected = _claims_at(claims, tnid, prop)
    if not expected:
        return (1.0 if not refs else 0.0), 0
    if not refs:
        return 0.0, 0
    grounded = {c.id: False for c in expected}
    valid = 0
    unsupported = 0
    for ref in refs:
        g, _i, _a = _grounded_claim_ids(
            db,
            claims,
            assertions,
            _ref_evidence(agent, ref),
            context_id=tnid,
            property=prop,
        )
        supported = [c for c in expected if c.id in g]
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
# Concept identity (per kind, incl. edge conditions)
# ---------------------------------------------------------------------------


def _concept_bindings(
    agent: BusinessProcessGraph,
    truth: BusinessProcessGraph,
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    mapping: dict[str, str],
    edge_map: dict[str, str],
    stakeholder: Optional[StakeholderFilter],
) -> tuple[float, dict[str, str]]:
    """Concept-level binding integrity across all matched visible slots.

    For every ConceptKind (conditions included):
    - one Agent concept may bind to exactly ONE Truth concept of its own kind
      (unsupported => unassignable; >1 => ambiguous/merged distinct concepts;
      kind mismatch => incompatible);
    - one visible Truth concept must be represented by exactly one Agent
      concept (splits fail; missing concepts fail).

    Returns (concept_correctness, agent concept id -> Truth concept id).
    """
    candidates: dict[str, set[str]] = {}
    referenced: set[str] = set()

    def add_ref(ref: ConceptRef, context_id: str, prop: str) -> None:
        if not ref.asserted:
            return
        referenced.add(ref.concept_id)
        g, _i, _a = _grounded_claim_ids(
            db,
            claims,
            assertions,
            _ref_evidence(agent, ref),
            context_id=context_id,
            property=prop,
        )
        truth_ids = {
            c.concept_id
            for c in _claims_at(claims, context_id, prop)
            if c.concept_id is not None and c.id in g
        }
        if truth_ids:
            prev = candidates.setdefault(ref.concept_id, set(truth_ids))
            candidates[ref.concept_id] = prev & truth_ids

    for anid, tnid in mapping.items():
        visible = (
            set(_NODE_PROPS)
            if stakeholder is None
            else stakeholder.node_properties_for(tnid)
        )
        for prop in visible:
            for ref in agent.nodes[anid].asserted_refs(prop):
                add_ref(ref, tnid, prop)
    for eid, teid in edge_map.items():
        edge = agent.edges[eid]
        if edge.condition is not None and edge.condition.asserted:
            add_ref(edge.condition, teid, "condition")

    truth_kind: dict[str, str] = {}
    for c in claims.values():
        if c.concept_id is not None:
            truth_kind.setdefault(c.concept_id, _PROPERTY_KIND.get(c.property, "data"))

    agent_to_truth: dict[str, str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts.get(cid)
        if concept is None:
            return 0.0, {}
        cand = candidates.get(cid, set())
        if len(cand) != 1:
            return 0.0, {}
        truth_id = next(iter(cand))
        if truth_kind.get(truth_id) != concept.kind:
            return 0.0, {}
        agent_to_truth[cid] = truth_id

    truth_by_kind: dict[str, list[str]] = {}
    for t, k in truth_kind.items():
        truth_by_kind.setdefault(k, []).append(t)
    by_kind: dict[str, list[str]] = {}
    for cid, tid in agent_to_truth.items():
        by_kind.setdefault(agent.concepts[cid].kind, []).append(cid)
    for kind, agents in by_kind.items():
        covered = {agent_to_truth[c] for c in agents}
        if len(covered) != len(agents):
            return 0.0, {}
        if covered != set(truth_by_kind.get(kind, [])):
            return 0.0, {}
    return 1.0, agent_to_truth


# ---------------------------------------------------------------------------
# Glossary validation (genuine confirmation)
# ---------------------------------------------------------------------------


def _glossary_validation(
    agent: BusinessProcessGraph,
    db: InterviewDB,
    claims: dict[str, TruthClaim],
    assertions: dict[int, list[StakeholderAssertion]],
    agent_to_truth: dict[str, str],
) -> tuple[bool, list[str], list[str]]:
    """Completion + genuine-validation rules.

    - every referenced concept must be resolved (not hypothesized);
    - ``confirmed`` / ``partially_confirmed`` concepts need validation
      evidence that corresponds to a private assertion of a claim of the
      concept's bound Truth concept (actual stakeholder usage of the
      concept's identity — not just any authentic earlier speech);
    - ``unknown`` concepts need validation evidence that does NOT correspond
      to any assertion of the concept's claims (the stakeholder did not
      assert the concept — e.g. "I don't know");
    - ``disputed`` concepts need evidence from >= 2 distinct Observations
      corresponding to assertions of the concept's claims;
    - **no bulk self-confirmation**: one (observation, quote, occurrence)
      span may back at most one concept's validation_evidence.

    Returns (pass, referenced_hypothesized, validation_errors).
    """
    referenced = agent.referenced_concepts()
    hypothesized = sorted(
        cid
        for cid in referenced
        if agent.concepts.get(cid) is not None
        and agent.concepts[cid].validation_status == "hypothesized"
    )
    if hypothesized:
        return False, hypothesized, []

    errors: list[str] = []
    span_usage: dict[tuple[str, str, int], str] = {}
    for cid in sorted(referenced):
        concept = agent.concepts[cid]
        bound = agent_to_truth.get(cid)
        for ev in concept.validation_evidence:
            key = (ev.observation_id, ev.quote, ev.occurrence)
            prev = span_usage.get(key)
            if prev is not None and prev != cid:
                errors.append(
                    f"evidence span {key} backs both {prev} and {cid} "
                    f"(bulk self-confirmation)"
                )
            span_usage[key] = cid
            obs = _obs_by_id(db, ev.observation_id)
            if obs is None:
                errors.append(
                    f"concept {cid}: validation evidence references unknown observation"
                )
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                errors.append(f"concept {cid}: validation evidence span is not exact")
                continue
            corresponding = set()
            for a in assertions.get(obs.turn, []):
                ass_span = _resolve_assertion_span(obs.text, a)
                if ass_span is not None and spans_correspond(ev_span, ass_span):
                    corresponding.add(a.claim_id)
            same_concept_claims = {
                c.id
                for c in claims.values()
                if bound is not None and c.concept_id == bound
            }
            if concept.validation_status in ("confirmed", "partially_confirmed"):
                if not concept.validation_evidence:
                    errors.append(
                        f"concept {cid}: confirmed requires validation evidence"
                    )
                    continue
                if bound is None or not (corresponding & same_concept_claims):
                    errors.append(
                        f"concept {cid}: confirmation evidence does not correspond "
                        f"to a stakeholder assertion of the concept's claims"
                    )
            elif concept.validation_status == "unknown":
                if not concept.validation_evidence:
                    errors.append(
                        f"concept {cid}: unknown requires validation evidence"
                    )
                    continue
                if corresponding & same_concept_claims:
                    errors.append(
                        f"concept {cid}: unknown evidence corresponds to an "
                        f"assertion of the concept's claims"
                    )
            elif concept.validation_status == "disputed":
                pass  # handled below (>=2 distinct observations)
    # disputed needs >= 2 distinct observations with corresponding assertions
    for cid in sorted(referenced):
        concept = agent.concepts[cid]
        if concept.validation_status != "disputed":
            continue
        bound = agent_to_truth.get(cid)
        same_concept_claims = {
            c.id for c in claims.values() if bound is not None and c.concept_id == bound
        }
        obs_with_support = set()
        for ev in concept.validation_evidence:
            obs = _obs_by_id(db, ev.observation_id)
            if obs is None:
                continue
            ev_span = ev.resolve_span(obs.text)
            if ev_span is None:
                continue
            corresponding = set()
            for a in assertions.get(obs.turn, []):
                ass_span = _resolve_assertion_span(obs.text, a)
                if ass_span is not None and spans_correspond(ev_span, ass_span):
                    corresponding.add(a.claim_id)
            if corresponding & same_concept_claims:
                obs_with_support.add(ev.observation_id)
        if len(obs_with_support) < 2:
            errors.append(
                f"concept {cid}: disputed needs evidence from at least two "
                f"distinct Observations asserting the concept's claims"
            )
    return (not errors), hypothesized, errors


# ---------------------------------------------------------------------------
# Evidence hygiene
# ---------------------------------------------------------------------------


def _evidence_metrics(
    db: InterviewDB,
    agent: BusinessProcessGraph,
) -> tuple[int, int, float, float, float]:
    """(invalid, invalid_obs_refs, node_cov, ref_cov, edge_cov)."""
    invalid = 0
    invalid_obs_refs = 0
    ref_total = ref_hit = 0
    node_total = node_hit = 0
    edge_total = edge_hit = 0
    obs_text = {o.id: o.text for o in db.observations}

    def span_ok(ev: EvidenceRef) -> bool:
        nonlocal invalid, invalid_obs_refs
        text = obs_text.get(ev.observation_id)
        if text is None:
            invalid += 1
            if ev.observation_id not in obs_text:
                invalid_obs_refs += 1
            return False
        if ev.resolve_span(text) is None:
            invalid += 1
            return False
        return True

    def check_evidence(evs: list[EvidenceRef]) -> bool:
        ok = True
        for ev in evs:
            if not span_ok(ev):
                ok = False
        return ok

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
            if evs and check_evidence(evs):
                ref_hit += 1
    for edge in agent.edges.values():
        edge_total += 1
        if edge.evidence and check_evidence(edge.evidence):
            edge_hit += 1
        if edge.condition is not None:
            check_evidence(_ref_evidence(agent, edge.condition))
    node_cov = node_hit / node_total if node_total else 1.0
    ref_cov = ref_hit / ref_total if ref_total else 1.0
    edge_cov = edge_hit / edge_total if edge_total else 1.0
    return invalid, invalid_obs_refs, node_cov, ref_cov, edge_cov


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
    assertions: Optional[dict[int, list[StakeholderAssertion]]] = None,
) -> EvaluationResult:
    """Evaluate the inferred graph against the hidden Truth.

    ``claims`` is the hidden graph-contextual TruthClaim catalog
    (``Scenario.claims``) and ``assertions`` the private sidecar ledger
    (``{turn: [assertions]}``, ``StakeholderAssertionLedger.assertions()``).
    Both are evaluator-only; the Agent never sees them. Invalid private
    metadata is rejected deterministically. Without provenance, no ConceptRef
    can ground a claim.
    """
    agent = db.graph if db.graph is not None else BusinessProcessGraph()
    protocol = db.interview_complete
    graph_created = len(agent.nodes) > 0
    graph_valid = agent.is_valid
    claims = claims or {}
    ledger = assertions or {}
    _validate_assertions(ledger, claims, stakeholder, db.messages)

    # ---- node/edge correspondence (provenance + topology) ------------------
    mapping, edge_map = _match_nodes_and_edges(agent, truth, db, claims, ledger)
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
    matched_truth_edges = {teid for teid in edge_map.values()}
    edge_recall = (
        len(matched_truth_edges) / len(truth_edge_list) if truth_edge_list else 1.0
    )
    edge_precision = len(edge_map) / len(agent_edge_list) if agent_edge_list else 0.0
    fabricated_edge_count = len(agent_edge_list) - len(edge_map)

    # ---- start / end --------------------------------------------------------
    start_correct = bool(
        agent.start_node_id is not None
        and mapping.get(agent.start_node_id) == truth.start_node_id
    )
    agent_ends = {mapping.get(eid) for eid in agent.end_node_ids if mapping.get(eid)}
    truth_ends = set(truth.end_node_ids)
    end_recall = len(agent_ends & truth_ends) / len(truth_ends) if truth_ends else 1.0
    end_precision = (
        len(agent_ends & truth_ends) / len(agent_ends) if agent_ends else 0.0
    )

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
                agent, db, claims, ledger, tnid, anid, prop, prop in visible
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
        expected = _claims_at(claims, te.id, "condition")
        if not expected:
            cond_total += 1
            ae = next(
                (
                    edge
                    for edge in agent.edges.values()
                    if edge_map.get(edge.id) == te.id
                ),
                None,
            )
            cond_hits += (
                1
                if (ae is None or ae.condition is None or not ae.condition.asserted)
                else 0
            )
            continue
        cond_total += 1
        ae = next(
            (edge for edge in agent.edges.values() if edge_map.get(edge.id) == te.id),
            None,
        )
        if ae is None or ae.condition is None or not ae.condition.asserted:
            continue
        g, _i, _a = _grounded_claim_ids(
            db,
            claims,
            ledger,
            _ref_evidence(agent, ae.condition),
            context_id=te.id,
            property="condition",
        )
        if expected[0].id in g:
            cond_hits += 1
    condition_correctness = cond_hits / cond_total if cond_total else 1.0

    # ---- concept identity ---------------------------------------------------
    concept_correctness, agent_to_truth = _concept_bindings(
        agent, truth, db, claims, ledger, mapping, edge_map, stakeholder
    )

    # ---- glossary completion + genuine validation ---------------------------
    glossary_pass, hypothesized, glossary_errors = _glossary_validation(
        agent, db, claims, ledger, agent_to_truth
    )

    # ---- evidence hygiene ---------------------------------------------------
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
    ) = _evidence_metrics(db, agent)

    # ambiguous evidence refs: slot-scoped, aggregated from the scoring paths
    ambiguous_evidence_ref_count = 0
    for anid, tnid in mapping.items():
        visible = (
            set(_NODE_PROPS)
            if stakeholder is None
            else stakeholder.node_properties_for(tnid)
        )
        for prop in visible:
            for ref in agent.nodes[anid].asserted_refs(prop):
                _g, _i, amb = _grounded_claim_ids(
                    db,
                    claims,
                    ledger,
                    _ref_evidence(agent, ref),
                    context_id=tnid,
                    property=prop,
                )
                ambiguous_evidence_ref_count += amb
    for edge in agent.edges.values():
        _g, _i, amb = _grounded_claim_ids(
            db, claims, ledger, edge.evidence, property="edge_exists"
        )
        ambiguous_evidence_ref_count += amb
        if edge.condition is not None and edge.condition.asserted:
            for teid in edge_map.values():
                _g, _i, amb2 = _grounded_claim_ids(
                    db,
                    claims,
                    ledger,
                    _ref_evidence(agent, edge.condition),
                    context_id=teid,
                    property="condition",
                )
                ambiguous_evidence_ref_count += amb2

    provenance_authenticity_pass = bool(
        invalid_evidence_ref_count == 0
        and invalid_observation_reference_count == 0
        and invalid_observation_source_count == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and ambiguous_evidence_ref_count == 0
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
        unsupported_ref_count=unsupported_concept_ref_count,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        glossary_pass=glossary_pass,
        referenced_hypothesized_concepts=hypothesized,
        glossary_validation_errors=glossary_errors,
        node_evidence_coverage=node_evidence_coverage,
        ref_evidence_coverage=ref_evidence_coverage,
        edge_evidence_coverage=edge_evidence_coverage,
        invalid_evidence_ref_count=invalid_evidence_ref_count,
        ambiguous_evidence_ref_count=ambiguous_evidence_ref_count,
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
