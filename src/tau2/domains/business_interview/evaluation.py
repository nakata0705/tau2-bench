"""Evaluator for the open-world business_interview benchmark (v5 — simple).

The agent's inferred ``BusinessDAG`` (open-world free-text actions, optional
generic primitives, authentic Observation provenance) is compared to the scenario
Truth DAG — both use the same ``BusinessDAG`` class.

**Benchmark vs production separation.** In production there is no ground truth;
here a hidden ground truth exists and only the evaluator uses it to score
deterministically. Scenario-specific domain concepts live in a **hidden
scenario-local ``EvaluationSpec``** (evaluator-only). The global resolver holds
only reusable generic primitives (unknown operations resolve to ``unclassified``).

Metrics cover structure (node/edge recall + precision, declared start/end),
predicate/actor/system/read/write correctness, necessity correctness,
**primitive correctness** (diagnostic; ``unclassified`` is not a failure),
evidence-backedness + Observation authenticity, and **claim-level evidence
relevance** (each claim is evaluated independently against its own value:
action / primitive / actor / system / each read / each write / edge relation /
predicate / necessity properties).
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.aliases import norm_role, norm_system
from tau2.domains.business_interview.concepts import resolve_primitive
from tau2.domains.business_interview.dag import BusinessDAG, InferredValue, InterviewDB

_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")

_NECESSITY_PROPS = ("rationale", "owner", "evidence", "removal_impact")


class TruthNodeSpec(BaseModel):
    """Hidden, scenario-local spec for one Truth node.

    ``expressions`` are evaluator-only aliases/paraphrases used to match the
    agent's free-text action to this Truth node. ``primitive`` is the expected
    generic operation (optional). Never shown to the agent.
    """

    expressions: list[str] = Field(default_factory=list)
    primitive: Optional[str] = None


class EvaluationSpec(BaseModel):
    """Evaluator-only, scenario-local annotations (hidden from the agent)."""

    truth_nodes: dict[str, TruthNodeSpec] = Field(default_factory=dict)
    predicate_expressions: dict[str, list[str]] = Field(default_factory=dict)
    necessity_expressions: dict[str, list[str]] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    protocol_completed: bool
    dag_created: bool
    dag_valid: bool

    node_recall: float
    node_precision: float
    domain_concept_correctness: float
    edge_recall: float
    edge_precision: float
    start_correct: bool
    end_recall: float
    end_precision: float
    predicate_correctness: float
    actor_correctness: float
    system_correctness: float
    read_correctness: float
    write_correctness: float
    necessity_correctness: float
    primitive_correctness: float
    fabricated_node_count: int
    fabricated_edge_count: int
    fabricated_necessity: bool

    # evidence-backedness
    node_evidence_coverage: float
    attribute_provenance_coverage: float
    edge_evidence_coverage: float
    predicate_provenance_coverage: float
    necessity_provenance_coverage: float
    invalid_observation_reference_count: int
    evidence_pass: bool

    # observation authenticity
    authentic_observation_count: int
    invalid_observation_source_count: int
    orphan_observation_count: int
    provenance_authenticity_pass: bool

    # claim-level evidence relevance
    relevance_pass: bool

    structural_pass: bool
    necessity_pass: bool
    protocol_pass: bool
    quality_pass: bool


# ---------------------------------------------------------------------------
# Matching helpers (open-world, scenario-local)
# ---------------------------------------------------------------------------


def _tokens(text: Optional[str]) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _match_score(text: Optional[str], spec: TruthNodeSpec) -> int:
    if not text:
        return 0
    t = text.lower()
    at = _tokens(text)
    best = 0
    for expr in spec.expressions:
        e = expr.lower()
        ov = len(at & _tokens(expr))
        if e in t or t in e:
            ov += 1
        best = max(best, ov)
    return best


def _pick_tiebreak(agent_node, cands: list[str], truth: BusinessDAG) -> str:
    rec_actor = norm_role(agent_node.actor.value)
    rec_system = norm_system(agent_node.system.value)
    best = cands[0]
    best_score = -1
    for tid in cands:
        tn = truth.nodes[tid]
        score = 0
        if rec_actor == norm_role(tn.actor.value):
            score += 2
        if rec_system == norm_system(tn.system.value):
            score += 1
        if score > best_score:
            best_score = score
            best = tid
    return best


def _match_nodes(agent: BusinessDAG, truth: BusinessDAG, spec: EvaluationSpec):
    """Map agent node id -> truth node id using the hidden EvaluationSpec."""
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for nid, node in agent.nodes.items():
        text = node.action.value
        if not text:
            continue
        scored = [
            (tid, _match_score(text, spec.truth_nodes[tid]))
            for tid in spec.truth_nodes
            if tid not in used
        ]
        scored = [(t, s) for t, s in scored if s > 0]
        if not scored:
            continue
        best = max(s for _, s in scored)
        cands = [t for t, s in scored if s == best]
        chosen = _pick_tiebreak(node, cands, truth) if len(cands) > 1 else cands[0]
        used.add(chosen)
        mapping[nid] = chosen
    return mapping


# ---------------------------------------------------------------------------
# Data / predicate / necessity
# ---------------------------------------------------------------------------


def _data_recall(agent_items, truth_items) -> float:
    if not truth_items:
        return 1.0 if not agent_items else 0.0
    hits = 0
    for gi in range(len(truth_items)):
        if not truth_items[gi]:
            hits += 1
            continue
        matched = any(
            bool(_tokens(truth_items[gi]) & _tokens(a))
            or truth_items[gi].lower() in (a or "").lower()
            for a in agent_items
        )
        if matched:
            hits += 1
    return hits / len(truth_items)


_PREDICATE_STOP = {
    "amount",
    "at",
    "or",
    "and",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "yen",
    "more",
    "than",
    "less",
    "quotation",
    "quote",
}


def _significant_tokens(text) -> set[str]:
    return {t for t in _tokens(text) if t not in _PREDICATE_STOP and not t.isdigit()}


def _predicate_ok(agent_pred, truth_pred, spec: EvaluationSpec) -> bool:
    if truth_pred is None:
        return True
    if not agent_pred:
        return False
    expressions = spec.predicate_expressions.get(truth_pred) or [truth_pred]
    at = _significant_tokens(agent_pred)
    a_low = (agent_pred or "").lower()
    for expr in expressions:
        et = _significant_tokens(expr)
        if at and et and (at & et):
            return True
        e_low = expr.lower()
        if e_low in a_low or a_low in e_low:
            return True
    return False


def _necessity_value_ok(aval: str, tval: str, spec: EvaluationSpec) -> bool:
    expressions = spec.necessity_expressions.get(tval) or [tval]
    a_low = (aval or "").lower()
    a_tokens = _tokens(aval)
    for expr in expressions:
        e_low = expr.lower()
        if e_low in a_low or a_low in e_low:
            return True
        if a_tokens & _tokens(expr):
            return True
    return False


# ---------------------------------------------------------------------------
# Evidence (Observation provenance) analysis
# ---------------------------------------------------------------------------


def _iter_node_inferred(node):
    yield node.action
    # an 'unclassified' primitive is a sentinel (unknown operation), not a claim
    if node.primitive is not None and node.primitive.value != "unclassified":
        yield node.primitive
    yield node.actor
    yield node.system
    for r in node.reads:
        yield r
    for w in node.writes:
        yield w


def _evidence_metrics(
    agent: BusinessDAG, all_obs_ids: set[str], authentic_obs_ids: set[str]
):
    invalid = 0
    nonauthentic = 0
    node_total = node_hit = 0
    attr_total = attr_hit = 0
    edge_total = edge_hit = 0
    pred_total = pred_hit = 0
    nec_total = nec_hit = 0

    def classify(ids: list[str]) -> None:
        nonlocal invalid, nonauthentic
        for i in ids:
            if i not in all_obs_ids:
                invalid += 1
            elif i not in authentic_obs_ids:
                nonauthentic += 1

    def iv_has_valid_ref(iv: InferredValue) -> bool:
        return any(o in authentic_obs_ids for o in iv.observation_ids)

    for node in agent.nodes.values():
        classify(node.observation_ids)
        node_total += 1
        if any(o in authentic_obs_ids for o in node.observation_ids):
            node_hit += 1
        for iv in _iter_node_inferred(node):
            classify(iv.observation_ids)
            if iv.asserted:
                attr_total += 1
                if iv_has_valid_ref(iv):
                    attr_hit += 1
        if node.necessity is not None:
            for prop in _NECESSITY_PROPS:
                iv = getattr(node.necessity, prop)
                classify(iv.observation_ids)
                if iv.asserted:
                    nec_total += 1
                    if iv_has_valid_ref(iv):
                        nec_hit += 1

    for edge in agent.edges.values():
        classify(edge.observation_ids)
        edge_total += 1
        if any(o in authentic_obs_ids for o in edge.observation_ids):
            edge_hit += 1
        if edge.predicate is not None:
            classify(edge.predicate.observation_ids)
            if edge.predicate.asserted:
                pred_total += 1
                if iv_has_valid_ref(edge.predicate):
                    pred_hit += 1

    node_evidence_coverage = node_hit / node_total if node_total else 1.0
    attribute_provenance_coverage = attr_hit / attr_total if attr_total else 1.0
    edge_evidence_coverage = edge_hit / edge_total if edge_total else 1.0
    predicate_provenance_coverage = pred_hit / pred_total if pred_total else 1.0
    necessity_provenance_coverage = nec_hit / nec_total if nec_total else 1.0
    return (
        node_evidence_coverage,
        attribute_provenance_coverage,
        edge_evidence_coverage,
        predicate_provenance_coverage,
        necessity_provenance_coverage,
        invalid,
        nonauthentic,
    )


# ---------------------------------------------------------------------------
# Claim-level evidence relevance (SUPPORTED / CONTRADICTED / UNKNOWN)
# ---------------------------------------------------------------------------


_NEGATION = (
    "not ",
    " no ",
    "never",
    "doesn't",
    "don't",
    " no evidence",
    "no such",
    "しない",
    " ない",
    "なし",
)
_RELATION_WORDS = (
    "after",
    "then",
    "followed by",
    "before",
    "next",
    "subsequently",
    "次に",
    "後に",
    "その後",
    "の後",
    "それから",
)
_DIRECTION_UP = (
    "over",
    "above",
    "exceed",
    "exceeds",
    "more than",
    "greater than",
    "超",
    "超過",
    "以上",
)
_DIRECTION_DOWN = (
    "below",
    "under",
    "less than",
    "at or below",
    "以下",
    "未満",
)


def _is_negated(text: str) -> bool:
    return any(n in text for n in _NEGATION)


def _mentions(text: str, value: Optional[str]) -> bool:
    """True if ``value`` (substring or a shared token) appears in ``text``."""
    if not value:
        return False
    vl = value.lower()
    if vl in text:
        return True
    return bool(_tokens(value) & _tokens(text))


def _relation_supported(text: str, from_action, to_action) -> bool:
    if not (from_action and to_action):
        return False
    if not any(w in text for w in _RELATION_WORDS):
        return False
    return _mentions(text, from_action) and _mentions(text, to_action)


def _predicate_supported(text: str, claim_value: Optional[str]) -> bool:
    cv = (claim_value or "").lower()
    if not cv:
        return False
    if cv in text:
        return True
    up = any(w in cv for w in _DIRECTION_UP)
    down = any(w in cv for w in _DIRECTION_DOWN)
    if up and any(w in text for w in _DIRECTION_UP):
        return True
    if down and any(w in text for w in _DIRECTION_DOWN):
        return True
    return False


def support(
    obs_text: str, claim_kind: str, claim_value: Optional[str], context=None
) -> str:
    """SUPPORTED / CONTRADICTED / UNKNOWN for one observation vs one claim.

    Each claim is matched against its **own** value (actor against the actor
    value, system against the system value, predicate against the predicate
    itself, etc.); the node-wide signal set is not reused. CONTRADICTED wins
    over SUPPORTED when the observation is both relevant and negated.
    """
    ot = (obs_text or "").lower()
    if claim_kind == "primitive":
        if not claim_value or claim_value == "unclassified":
            return "UNKNOWN"
        if resolve_primitive(obs_text) == claim_value:
            return "SUPPORTED"
        return "UNKNOWN"
    if claim_kind == "edge":
        ctx = context or {}
        ok = _relation_supported(ot, ctx.get("from_action"), ctx.get("to_action"))
    elif claim_kind == "predicate":
        ok = _predicate_supported(ot, claim_value)
    else:
        ok = _mentions(ot, claim_value)
    if not ok:
        return "UNKNOWN"
    return "CONTRADICTED" if _is_negated(ot) else "SUPPORTED"


def _obs_ids_supported(
    obs_ids: list[str], claim_kind: str, claim_value: Optional[str], context, obs_by_id
) -> bool:
    if not obs_ids:
        return False
    for oid in obs_ids:
        obs = obs_by_id.get(oid)
        if obs is None:
            continue
        if support(obs.text, claim_kind, claim_value, context) == "SUPPORTED":
            return True
    return False


def _relevance_pass(agent: BusinessDAG, obs_by_id) -> bool:
    """Every asserted claim must have at least one SUPPORTED provenance.

    Claims are evaluated independently: action, primitive (unless
    ``unclassified``), actor, system, each read, each write, each necessity
    property, edge relation, and predicate. An unrelated / partially-poisoned
    observation cannot support a claim it does not actually mention.
    """
    for node in agent.nodes.values():
        claims = [
            ("action", node.action, "action", None),
            ("actor", node.actor, "actor", None),
            ("system", node.system, "system", None),
        ]
        if node.primitive is not None and node.primitive.asserted:
            claims.append(("primitive", node.primitive, "primitive", None))
        for i, r in enumerate(node.reads):
            claims.append((f"read[{i}]", r, "read", None))
        for i, w in enumerate(node.writes):
            claims.append((f"write[{i}]", w, "write", None))
        if node.necessity is not None:
            for p in _NECESSITY_PROPS:
                claims.append(
                    (f"necessity.{p}", getattr(node.necessity, p), "necessity", None)
                )
        for _name, iv, kind, ctx in claims:
            if not iv.asserted:
                continue
            if iv.value == "unclassified":
                continue  # sentinel for an unknown primitive; not a claim to fail
            if not _obs_ids_supported(
                iv.observation_ids, kind, iv.value, ctx, obs_by_id
            ):
                return False

    for edge in agent.edges.values():
        fa = agent.nodes.get(edge.from_node)
        ta = agent.nodes.get(edge.to_node)
        ctx = {
            "from_action": fa.action.value if fa else None,
            "to_action": ta.action.value if ta else None,
        }
        if edge.observation_ids and not _obs_ids_supported(
            edge.observation_ids, "edge", None, ctx, obs_by_id
        ):
            return False
        if edge.predicate is not None and edge.predicate.asserted:
            if not _obs_ids_supported(
                edge.predicate.observation_ids,
                "predicate",
                edge.predicate.value,
                None,
                obs_by_id,
            ):
                return False
    return True


def _all_referenced_observation_ids(agent: BusinessDAG) -> set[str]:
    ids: set[str] = set()
    for node in agent.nodes.values():
        ids.update(node.observation_ids)
        for iv in _iter_node_inferred(node):
            ids.update(iv.observation_ids)
        if node.necessity is not None:
            for p in _NECESSITY_PROPS:
                ids.update(getattr(node.necessity, p).observation_ids)
    for edge in agent.edges.values():
        ids.update(edge.observation_ids)
        if edge.predicate is not None:
            ids.update(edge.predicate.observation_ids)
    return ids


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def _truth_has_edge(truth: BusinessDAG, tf: str, tt: str) -> bool:
    return any(e.from_node == tf and e.to_node == tt for e in truth.edges.values())


def _agent_edge(agent: BusinessDAG, af: str, at: str):
    for e in agent.edges.values():
        if e.from_node == af and e.to_node == at:
            return e
    return None


def evaluate(
    db: InterviewDB, truth: BusinessDAG, spec: EvaluationSpec
) -> EvaluationResult:
    agent = db.dag if db.dag is not None else BusinessDAG()
    protocol = db.interview_complete
    dag_created = len(agent.nodes) > 0
    dag_valid = agent.is_valid

    # ---- node matching -----------------------------------------------------
    mapping = _match_nodes(agent, truth, spec)
    truth_node_ids = list(truth.nodes)
    agent_node_ids = list(agent.nodes)
    matched_truth = set(mapping.values())
    matched_agent = set(mapping.keys())
    node_recall = len(matched_truth) / len(truth_node_ids) if truth_node_ids else 0.0
    node_precision = len(matched_agent) / len(agent_node_ids) if agent_node_ids else 0.0
    domain_concept_correctness = node_precision
    fabricated_node_count = len(agent_node_ids) - len(matched_agent)
    rev = {t: a for a, t in mapping.items()}

    # ---- edges -------------------------------------------------------------
    truth_edge_list = list(truth.edges.values())
    agent_edge_list = list(agent.edges.values())
    matched_truth_edges = 0
    predicate_hits = 0
    predicate_total = 0
    for te in truth_edge_list:
        af = rev.get(te.from_node)
        at = rev.get(te.to_node)
        if af is None or at is None:
            continue
        ae = _agent_edge(agent, af, at)
        if ae is None:
            continue
        matched_truth_edges += 1
        tpred = te.predicate.value if te.predicate else None
        apred = ae.predicate.value if ae.predicate else None
        predicate_total += 1
        if _predicate_ok(apred, tpred, spec):
            predicate_hits += 1
    edge_recall = matched_truth_edges / len(truth_edge_list) if truth_edge_list else 1.0
    matched_agent_edge_ids = set()
    for ae in agent_edge_list:
        tf = mapping.get(ae.from_node)
        tt = mapping.get(ae.to_node)
        if tf is not None and tt is not None and _truth_has_edge(truth, tf, tt):
            matched_agent_edge_ids.add(ae.id)
    edge_precision = (
        len(matched_agent_edge_ids) / len(agent_edge_list) if agent_edge_list else 0.0
    )
    fabricated_edge_count = len(agent_edge_list) - len(matched_agent_edge_ids)
    predicate_correctness = predicate_hits / predicate_total if predicate_total else 1.0

    # ---- declared start / end ----------------------------------------------
    start_correct = bool(
        agent.start_node_id and mapping.get(agent.start_node_id) == truth.start_node_id
    )
    truth_ends = set(truth.end_node_ids)
    agent_declared_ends = {
        mapping.get(eid) for eid in agent.end_node_ids if mapping.get(eid) is not None
    }
    end_recall = (
        len(agent_declared_ends & truth_ends) / len(truth_ends) if truth_ends else 1.0
    )
    end_precision = (
        len(agent_declared_ends & truth_ends) / len(agent_declared_ends)
        if agent_declared_ends
        else 0.0
    )

    # ---- attribute correctness ---------------------------------------------
    actor_hits = system_hits = read_hits = write_hits = 0
    for anid, tnid in mapping.items():
        an = agent.nodes[anid]
        tn = truth.nodes[tnid]
        if norm_role(an.actor.value) == norm_role(tn.actor.value):
            actor_hits += 1
        if norm_system(an.system.value) == norm_system(tn.system.value):
            system_hits += 1
        read_hits += _data_recall(
            [r.value for r in an.reads], [r.value for r in tn.reads]
        )
        write_hits += _data_recall(
            [w.value for w in an.writes], [w.value for w in tn.writes]
        )
    nm = len(mapping) or 1
    actor_correctness = actor_hits / nm
    system_correctness = system_hits / nm
    read_correctness = read_hits / nm
    write_correctness = write_hits / nm

    # ---- necessity correctness ---------------------------------------------
    nec_total = nec_hits = 0
    fabricated_necessity = False
    for tnid, tn in truth.nodes.items():
        anid = rev.get(tnid)
        if anid is None or tn.necessity is None:
            continue
        an = agent.nodes[anid]
        for prop in _NECESSITY_PROPS:
            tval = getattr(tn.necessity, prop).value
            aval = getattr(an.necessity, prop) if an.necessity else InferredValue()
            nec_total += 1
            if tval is None:
                if aval.value is not None:
                    fabricated_necessity = True
                else:
                    nec_hits += 1
            else:
                if aval.asserted and _necessity_value_ok(aval.value, tval, spec):
                    nec_hits += 1
    necessity_correctness = nec_hits / nec_total if nec_total else 1.0

    # ---- primitive correctness (diagnostic) --------------------------------
    primitive_hits = primitive_total = 0
    for anid, tnid in mapping.items():
        expected = spec.truth_nodes[tnid].primitive
        if expected is None:
            continue
        ap = agent.nodes[anid].primitive
        if ap is None or not ap.asserted:
            continue
        resolved = resolve_primitive(ap.value)
        if resolved == "unclassified":
            continue  # valid open-world state; not penalized
        primitive_total += 1
        if resolved == expected:
            primitive_hits += 1
    primitive_correctness = primitive_hits / primitive_total if primitive_total else 1.0

    # ---- observation authenticity -------------------------------------------
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
    obs_by_id = {o.id: o for o in db.observations}
    orphan_observation_count = sum(
        1 for o in db.observations if o.id not in _all_referenced_observation_ids(agent)
    )

    # ---- evidence ----------------------------------------------------------
    (
        node_evidence_coverage,
        attribute_provenance_coverage,
        edge_evidence_coverage,
        predicate_provenance_coverage,
        necessity_provenance_coverage,
        invalid_observation_reference_count,
        nonauthentic_reference_count,
    ) = _evidence_metrics(agent, all_obs_ids, authentic_obs_ids)
    provenance_authenticity_pass = bool(
        invalid_observation_reference_count == 0 and nonauthentic_reference_count == 0
    )
    evidence_pass = bool(
        provenance_authenticity_pass
        and node_evidence_coverage == 1.0
        and attribute_provenance_coverage == 1.0
        and edge_evidence_coverage == 1.0
        and predicate_provenance_coverage == 1.0
        and necessity_provenance_coverage == 1.0
    )

    # ---- claim-level relevance ---------------------------------------------
    relevance_pass = _relevance_pass(agent, obs_by_id)

    # ---- gates -------------------------------------------------------------
    structural_pass = bool(
        dag_created
        and dag_valid
        and node_recall == 1.0
        and node_precision == 1.0
        and edge_recall == 1.0
        and edge_precision == 1.0
        and start_correct
        and end_recall == 1.0
        and end_precision == 1.0
        and predicate_correctness == 1.0
        and actor_correctness == 1.0
        and system_correctness == 1.0
        and read_correctness == 1.0
        and write_correctness == 1.0
    )
    necessity_pass = necessity_correctness == 1.0 and not fabricated_necessity
    protocol_pass = protocol
    quality_pass = bool(
        structural_pass
        and necessity_pass
        and evidence_pass
        and provenance_authenticity_pass
        and relevance_pass
    )

    return EvaluationResult(
        protocol_completed=protocol,
        dag_created=dag_created,
        dag_valid=dag_valid,
        node_recall=node_recall,
        node_precision=node_precision,
        domain_concept_correctness=domain_concept_correctness,
        edge_recall=edge_recall,
        edge_precision=edge_precision,
        start_correct=start_correct,
        end_recall=end_recall,
        end_precision=end_precision,
        predicate_correctness=predicate_correctness,
        actor_correctness=actor_correctness,
        system_correctness=system_correctness,
        read_correctness=read_correctness,
        write_correctness=write_correctness,
        necessity_correctness=necessity_correctness,
        primitive_correctness=primitive_correctness,
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        fabricated_necessity=fabricated_necessity,
        node_evidence_coverage=node_evidence_coverage,
        attribute_provenance_coverage=attribute_provenance_coverage,
        edge_evidence_coverage=edge_evidence_coverage,
        predicate_provenance_coverage=predicate_provenance_coverage,
        necessity_provenance_coverage=necessity_provenance_coverage,
        invalid_observation_reference_count=invalid_observation_reference_count,
        evidence_pass=evidence_pass,
        authentic_observation_count=authentic_observation_count,
        invalid_observation_source_count=invalid_observation_source_count,
        orphan_observation_count=orphan_observation_count,
        provenance_authenticity_pass=provenance_authenticity_pass,
        relevance_pass=relevance_pass,
        structural_pass=structural_pass,
        necessity_pass=necessity_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
    )
