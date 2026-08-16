"""Evaluator for the evidence-backed DAG business_interview benchmark (v3).

The agent's inferred ``BusinessDAG`` (from ``InterviewDB``) is compared to the
scenario Truth DAG — both use the **same** ``BusinessDAG`` class. Concepts
(``EvaluationSpec``) are evaluator-only annotations that make action matching
language-independent and hide agent node ids.

Metrics: node/edge recall + precision, start correctness, end recall + precision,
predicate correctness, actor/system/read/write correctness, necessity correctness,
and fabricated node / edge counts. There is no Step / Transition / Branch-specific
evaluator; control-flow conditions live only on ``Edge.predicate``.
"""

from collections import defaultdict

from pydantic import BaseModel, Field

from tau2.domains.business_interview.aliases import norm_role, norm_system
from tau2.domains.business_interview.concepts import (
    DATA_CONCEPTS,
    NECESSITY_CONCEPTS,
    NODE_CONCEPTS,
    PREDICATE_CONCEPTS,
    resolve,
)
from tau2.domains.business_interview.dag import BusinessDAG, InterviewDB


class EvaluationSpec(BaseModel):
    """Evaluator-only annotations for a scenario.

    ``truth_node_concepts`` maps each Truth node id to its language-independent
    concept id (``concepts.py``). This is how "approve high-value quotation" and
    "100万円超の見積を営業部長が承認" are judged equivalent. Never shown to the agent.
    """

    truth_node_concepts: dict[str, str] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    protocol_completed: bool
    dag_created: bool
    dag_valid: bool

    node_recall: float
    node_precision: float
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
    fabricated_node_count: int
    fabricated_edge_count: int
    fabricated_necessity: bool

    structural_pass: bool
    necessity_pass: bool
    protocol_pass: bool
    quality_pass: bool


def _data_concepts(items) -> set[str]:
    return {c for c in (resolve(i, DATA_CONCEPTS) for i in items) if c is not None}


def _pick_candidate(agent_node, truth_ids: list[str], truth: BusinessDAG) -> str:
    """Deterministic same-concept tie-break by actor/system, then data."""
    rec_actor = norm_role(agent_node.actor.value)
    rec_system = norm_system(agent_node.system.value)
    rec_reads = _data_concepts([r.value for r in agent_node.reads])
    rec_writes = _data_concepts([w.value for w in agent_node.writes])

    def score(tid: str) -> int:
        tn = truth.nodes[tid]
        sc = 0
        if rec_actor == (tn.actor.value or ""):
            sc += 2
        if rec_system == (tn.system.value or ""):
            sc += 1
        sc += len(rec_reads & _data_concepts([r.value for r in tn.reads]))
        sc += len(rec_writes & _data_concepts([w.value for w in tn.writes]))
        return sc

    scores = [score(t) for t in truth_ids]
    return truth_ids[scores.index(max(scores))]


def _match_nodes(agent: BusinessDAG, truth: BusinessDAG, spec: EvaluationSpec):
    """Map agent node id -> truth node id by action concept (arbitrary ids ok)."""
    gt_by_concept: dict[str, list[str]] = defaultdict(list)
    for nid, cid in spec.truth_node_concepts.items():
        gt_by_concept[cid].append(nid)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for nid, node in agent.nodes.items():
        cid = resolve(
            node.action.value,
            NODE_CONCEPTS,
            actor=node.actor.value,
            system=node.system.value,
        )
        if not cid:
            continue
        cands = [t for t in gt_by_concept.get(cid, []) if t not in used]
        if not cands:
            continue
        chosen = cands[0] if len(cands) == 1 else _pick_candidate(node, cands, truth)
        used.add(chosen)
        mapping[nid] = chosen
    return mapping


def _data_recall(agent_items, truth_items) -> float:
    if not truth_items:
        return 1.0 if not agent_items else 0.0
    gt = _data_concepts(truth_items)
    if not gt:
        return 0.0
    rec = _data_concepts(agent_items)
    return sum(1 for g in gt if g in rec) / len(truth_items)


def _predicate_ok(agent_pred, truth_pred) -> bool:
    if truth_pred is None:
        return True
    if not agent_pred:
        return False
    return resolve(agent_pred, PREDICATE_CONCEPTS) == resolve(
        truth_pred, PREDICATE_CONCEPTS
    )


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
        if _predicate_ok(apred, tpred):
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

    # ---- start / end -------------------------------------------------------
    start_correct = bool(
        agent.start_node_id and mapping.get(agent.start_node_id) == truth.start_node_id
    )
    end_hits = 0
    for teid in truth.end_node_ids:
        for anid, tnid in mapping.items():
            if tnid == teid and agent.out_degree(anid) == 0:
                end_hits += 1
                break
    end_recall = end_hits / len(truth.end_node_ids) if truth.end_node_ids else 1.0
    agent_ends = [nid for nid in agent_node_ids if agent.out_degree(nid) == 0]
    end_precision = (
        sum(1 for a in agent_ends if mapping.get(a) in set(truth.end_node_ids))
        / len(agent_ends)
        if agent_ends
        else 0.0
    )

    # ---- attribute correctness over matched nodes --------------------------
    actor_hits = system_hits = read_hits = write_hits = 0
    for anid, tnid in mapping.items():
        an = agent.nodes[anid]
        tn = truth.nodes[tnid]
        if norm_role(an.actor.value) == (tn.actor.value or ""):
            actor_hits += 1
        if norm_system(an.system.value) == (tn.system.value or ""):
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
        for prop in ("rationale", "owner", "evidence", "removal_impact"):
            tval = getattr(tn.necessity, prop).value
            aval = getattr(an.necessity, prop).value if an.necessity else None
            nec_total += 1
            if tval is None:
                # expected unknown: agent must not fabricate a value
                if aval:
                    fabricated_necessity = True
                else:
                    nec_hits += 1
            else:
                if aval and _necessity_value_ok(aval, tval, prop):
                    nec_hits += 1
    necessity_correctness = nec_hits / nec_total if nec_total else 1.0

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
    quality_pass = structural_pass and necessity_pass

    return EvaluationResult(
        protocol_completed=protocol,
        dag_created=dag_created,
        dag_valid=dag_valid,
        node_recall=node_recall,
        node_precision=node_precision,
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
        fabricated_node_count=fabricated_node_count,
        fabricated_edge_count=fabricated_edge_count,
        fabricated_necessity=fabricated_necessity,
        structural_pass=structural_pass,
        necessity_pass=necessity_pass,
        protocol_pass=protocol_pass,
        quality_pass=quality_pass,
    )


def _necessity_value_ok(aval: str, tval: str, prop: str) -> bool:
    if prop == "rationale":
        concept = NECESSITY_CONCEPTS.get("credit_risk")
        return concept is not None and resolve(aval, [concept]) == concept.id
    return aval.strip().lower() == tval.strip().lower()
