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
and **evidence hygiene** (evidence-backedness + Observation authenticity).

**Result correctness vs evidence hygiene.** Result correctness compares the
inferred DAG against the hidden Ground Truth and is computed by exact Ground
Truth comparison. Evidence hygiene deterministically guarantees only that each
asserted claim references a *real, authentic stakeholder Observation* (captured
from an actual user message, not fabricated). The evaluator never re-interprets
Observation *text* to decide whether it semantically supports a claim — the
natural language of the stakeholder is expected to vary, and measuring whether
an Observation's wording matches a claim is out of scope.
"""

from typing import Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.aliases import norm_role, norm_system
from tau2.domains.business_interview.concepts import resolve_primitive
from tau2.domains.business_interview.dag import BusinessDAG, InferredValue, InterviewDB
from tau2.domains.business_interview.stakeholder import StakeholderFilter

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
    """Evaluator-only, scenario-local annotations (hidden from the agent).

    ``data_expressions`` maps a canonical Ground Truth data value (a read/write
    value) to the small set of **scenario-local** accepted expressions for the
    same business concept. It is a narrow, deterministic equivalence layer for
    **stakeholder-visible** reads/writes only: it never changes the canonical
    Truth value and it never applies to hidden attributes (a hidden assertion
    stays incorrect even when its wording matches an expression). A scenario
    without ``data_expressions`` keeps the baseline token/substring matching
    exactly.
    """

    truth_nodes: dict[str, TruthNodeSpec] = Field(default_factory=dict)
    predicate_expressions: dict[str, list[str]] = Field(default_factory=dict)
    necessity_expressions: dict[str, list[str]] = Field(default_factory=dict)
    data_expressions: dict[str, list[str]] = Field(default_factory=dict)


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

    structural_pass: bool
    necessity_pass: bool
    protocol_pass: bool
    quality_pass: bool


# ---------------------------------------------------------------------------
# Matching helpers (open-world, scenario-local)
# ---------------------------------------------------------------------------


def _tokens(text: Optional[str]) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


# Common function words / connectors that carry no node-identity signal. These
# are dropped before counting action overlap so a single weak shared token (e.g.
# "the", "to", "customer") can never, by itself, justify a match. Single-char
# tokens (possessive remnants like ``s``) are also ignored.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "into",
        "onto",
        "before",
        "after",
        "during",
        "we",
        "you",
        "us",
        "our",
        "ours",
        "i",
        "me",
        "my",
        "mine",
        "it",
        "its",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "done",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "that",
        "this",
        "these",
        "those",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "how",
        "why",
        "when",
        "then",
        "than",
        "if",
        "else",
        "not",
        "no",
        "none",
        "yes",
        "also",
        "too",
        "very",
        "so",
        "such",
        "about",
        "around",
        "over",
        "under",
        "up",
        "down",
        "out",
        "again",
        "still",
        "yet",
        "already",
        "all",
        "any",
        "each",
        "every",
        "some",
        "both",
        "few",
        "more",
        "most",
        "other",
        "another",
        "been",
        "get",
        "gets",
        "got",
        "use",
        "uses",
        "used",
        "using",
        "make",
        "makes",
        "made",
    ]
)


# Minimum number of *significant* (non-stopword) shared tokens required before
# an action can be considered a candidate match. A single weak shared token is
# never enough. (Full-expression containment and actor/system agreement can
# still qualify/rank a candidate — see below.)
_MIN_NODE_OVERLAP = 2


def _sig_tokens(text: Optional[str]) -> set[str]:
    """Significant (non-stopword, len>1) tokens of ``text``."""
    return {t for t in _tokens(text) if len(t) > 1 and t not in _STOPWORDS}


def _has_cjk(text: str) -> bool:
    """True if ``text`` contains CJK/kana characters (which ``_TOKEN_RE`` can't
    tokenize, so JA matching relies on substring containment instead)."""
    return any(ord(c) > 0x2E7F for c in text)


def _node_overlap(text: Optional[str], spec: TruthNodeSpec) -> int:
    """Best significant-token overlap between the agent action and any hidden
    scenario-local expression for a Truth node."""
    if not text:
        return 0
    at = _sig_tokens(text)
    best = 0
    for expr in spec.expressions:
        ov = len(at & _sig_tokens(expr))
        if ov > best:
            best = ov
    return best


def _expression_contained(text: Optional[str], spec: TruthNodeSpec) -> bool:
    """True if a whole hidden expression is a substring of the agent action (or
    vice versa). This is the primary signal for JA / non-Latin text where token
    overlap is 0, and a strong standalone signal for EN. Only meaningful phrases
    (>=2 significant tokens, or CJK content) are considered, so a single common
    token can never match by substring alone."""
    t = (text or "").lower()
    for expr in spec.expressions:
        e = (expr or "").lower()
        if not e:
            continue
        sig = _sig_tokens(e)
        if not sig and not _has_cjk(e):
            continue
        if e in t or t in e:
            return True
    return False


def _attribute_match(node, tnode) -> int:
    """Actor/system agreement as a **reinforcement** bonus. It is used only to
    rank candidates that already cleared the action gate — it never rescues a
    weak/ambiguous action match on its own."""
    bonus = 0
    ra = norm_role(node.actor.value)
    if ra and ra == norm_role(tnode.actor.value):
        bonus += 2
    rs = norm_system(node.system.value)
    if rs and rs == norm_system(tnode.system.value):
        bonus += 1
    return bonus


def _match_nodes(agent: BusinessDAG, truth: BusinessDAG, spec: EvaluationSpec):
    """Map agent node id -> truth node id using the hidden EvaluationSpec.

    Conservative, deterministic matching:
    - An agent node is a candidate for a Truth node iff it shares >=2 significant
      tokens with some hidden expression, OR contains a whole hidden expression.
    - Actor/system agreement adds a ranking bonus (reinforcement, not rescue).
    - Candidates are assigned greedily by (overlap, bonus) descending, so an
      agent node with low confidence is left unmatched rather than forced onto a
      Truth node it barely resembles. Agent node ids are ignored (arbitrary ids
      allowed). No embeddings / LLM judge are used.
    """
    candidates: list[tuple[int, int, str, str]] = []
    for nid, node in agent.nodes.items():
        text = node.action.value
        if not text:
            continue
        for tid, tspec in spec.truth_nodes.items():
            overlap = _node_overlap(text, tspec)
            if overlap < _MIN_NODE_OVERLAP and not _expression_contained(text, tspec):
                continue
            bonus = _attribute_match(node, truth.nodes[tid])
            candidates.append((overlap, bonus, nid, tid))
    # Highest overlap first; actor/system agreement breaks ties. Deterministic.
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    used_truth: set[str] = set()
    assigned: set[str] = set()
    mapping: dict[str, str] = {}
    for overlap, bonus, nid, tid in candidates:
        if nid in assigned or tid in used_truth:
            continue
        mapping[nid] = tid
        assigned.add(nid)
        used_truth.add(tid)
    return mapping


# ---------------------------------------------------------------------------
# Data / predicate / necessity
# ---------------------------------------------------------------------------


def _norm_data_value(value: Optional[str]) -> str:
    """Case-fold + whitespace normalization only (harmless formatting).

    Used for read/write concept identity: no tokenization, no stemming, no
    substring containment. Two labels are the same concept iff their
    normalized forms are byte-identical.
    """
    return " ".join((value or "").lower().split())


def _data_item_ok(tvalue: str, avalue: str, spec: EvaluationSpec) -> bool:
    """Concept-identity match for one read/write value (precision-first).

    Contract:

        normalized exact canonical value
        OR
        normalized exact scenario-local accepted expression

    ``spec.data_expressions[tvalue]`` lists **complete labels** that identify
    the same scenario concept as the canonical Truth value ``tvalue``; a label
    matches only when its normalized form equals the agent value's normalized
    form exactly. Shared tokens or substring containment are NEVER used to
    infer concept identity ("quotation request" / "quotation information" /
    "price quotation" do not match "quote" unless explicitly declared). The
    canonical Truth value is never rewritten.
    """
    na = _norm_data_value(avalue)
    if na == _norm_data_value(tvalue):
        return True
    for expr in spec.data_expressions.get(tvalue, ()):
        if na == _norm_data_value(expr):
            return True
    return False


def _data_recall(
    agent_items, truth_items, spec: Optional[EvaluationSpec] = None
) -> float:
    if spec is None:
        spec = EvaluationSpec()
    if not truth_items:
        return 1.0 if not agent_items else 0.0
    hits = 0
    for gi in range(len(truth_items)):
        if not truth_items[gi]:
            hits += 1
            continue
        matched = any(_data_item_ok(truth_items[gi], a, spec) for a in agent_items)
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
# Evidence hygiene (Observation provenance analysis)
# ---------------------------------------------------------------------------


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


def _attribute_ok(
    axis: str,
    agent_value,
    truth_value,
    spec: EvaluationSpec,
    visible: bool,
) -> float:
    """Score one actor/system/reads/writes axis for one matched node.

    ``visible=True`` (stakeholder can know it): compare the agent value to the
    Truth value using the current matching behavior.

    ``visible=False`` (hidden from the stakeholder): the correct behavior is to
    leave it unset/empty; an asserted/invented value is incorrect. We return 1.0
    for unset and 0.0 for any asserted value (even if it happens to equal the
    hidden Truth).

    ``agent_value``/``truth_value`` are ``InferredValue`` for actor/system and
    lists of ``InferredValue`` for reads/writes.
    """
    if axis in ("actor", "system"):
        if visible:
            if axis == "actor":
                return (
                    1.0
                    if norm_role(agent_value.value) == norm_role(truth_value.value)
                    else 0.0
                )
            return (
                1.0
                if norm_system(agent_value.value) == norm_system(truth_value.value)
                else 0.0
            )
        # hidden: correct only when unset/not asserted
        return 1.0 if not agent_value.asserted else 0.0
    # reads / writes
    if visible:
        return _data_recall(
            [v.value for v in agent_value], [v.value for v in truth_value], spec
        )
    # hidden: correct only when the list has no asserted entry
    return 1.0 if not any(v.asserted for v in agent_value) else 0.0


def evaluate(
    db: InterviewDB,
    truth: BusinessDAG,
    spec: EvaluationSpec,
    stakeholder: Optional[StakeholderFilter] = None,
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
        visible = (
            set(("actor", "system", "reads", "writes"))
            if stakeholder is None
            else stakeholder.visible_attributes_for(tnid)
        )
        actor_hits += _attribute_ok(
            "actor", an.actor, tn.actor, spec, "actor" in visible
        )
        system_hits += _attribute_ok(
            "system", an.system, tn.system, spec, "system" in visible
        )
        read_hits += _attribute_ok(
            "reads", an.reads, tn.reads, spec, "reads" in visible
        )
        write_hits += _attribute_ok(
            "writes", an.writes, tn.writes, spec, "writes" in visible
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
                if (
                    aval.asserted
                    and aval.value is not None
                    and _necessity_value_ok(aval.value, tval, spec)
                ):
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
        structural_pass=structural_pass,
        necessity_pass=necessity_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
    )
