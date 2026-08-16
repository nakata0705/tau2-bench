"""Structural workflow-reconstruction evaluator for business_interview (v2).

The agent records a reconstructed ``Workflow`` (steps, transitions, branches,
per-step necessity) using its own arbitrary step ids. The evaluator compares it
against the evaluator-only ground truth (``ground_truth.py``) and produces a
``WorkflowEvaluation``.

Design principles implemented here:

- **Arbitrary step ids**: step matching resolves each reconstructed step's
  action to a language-independent *concept* (``concepts.py``) and maps the
  agent's step id to the ground-truth step id. Every graph evaluation
  (transitions, branches, challenge, improvement) runs over that mapping, so
  the agent never has to guess hidden canonical ids.
- **EN/JA equivalence**: concept resolution is bilingual, so an equivalent
  Japanese reconstruction is scored identically to an English one.
- **Action-first matching**: a step matches a ground-truth step only by its
  action resolving to the same concept. actor/system are auxiliary (used for
  tie-breaking and for accuracy metrics), so a wrong action with the same
  actor/system is a missing + unexpected step, never a false match.
- **Precision-aware data**: read/write data are scored by recall AND precision.
- **Condition-aware transitions**: transitions are judged on from/to AND their
  condition concept; a missing / reversed / different condition is a fail.
- **NONE vs UNKNOWN**: an uninvestigated necessity is a fail; an investigated
  UNKNOWN is a pass; fabricating a UNKNOWN as FACT is an epistemic fail.
- **Confirmed rationale content**: epistemic status, source and semantic
  content are all compared to the ground truth.
- **Necessity challenge quality**: the questionable step must be investigated
  on why / owner / evidence / removal-impact and its deletion considered,
  before any improvement is accepted.
"""

from typing import Optional

from tau2.domains.business_interview.concepts import (
    CONDITION_CONCEPTS,
    DATA_CONCEPTS,
    RATIONALE_CONCEPTS,
    STEP_CONCEPTS,
    resolve,
    resolve_metadata,
)
from tau2.domains.business_interview.data_model import (
    EpistemicStatus,
    WorkflowDB,
    WorkflowEvaluation,
)
from tau2.domains.business_interview.ground_truth import (
    GTStep,
    get_ground_truth,
)

_IMPROVEMENT_RANK = {
    "question": 1,
    "delete": 2,
    "simplify": 3,
    "accelerate": 4,
    "automate": 5,
}

_ROLE_ALIASES = {
    "sales": (
        "sales",
        "sales employee",
        "sales representative",
        "sales rep",
        "salesperson",
        "quotation handler",
        "quotation preparer",
        "handler",
        "interviewee",
        "営業",
        "営業担当者",
        "営業社員",
        "見積担当",
    ),
    "manager": (
        "manager",
        "approver",
        "supervisor",
        "approval",
        "authority",
        "承認者",
        "上司",
        "承認",
    ),
}

_SYSTEM_ALIASES = {
    "crm": ("crm", "customer relationship management"),
    "quoting": (
        "quoting",
        "quote system",
        "quoting system",
        "quotation system",
        "見積システム",
    ),
    "email": ("email", "e-mail", "mail", "メール"),
    "excel": ("excel", "spreadsheet", "エクセル"),
}

_TOKEN_RE = __import__("re").compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _normalize_label(label: Optional[str], aliases: dict[str, tuple[str, ...]]) -> str:
    if label is None:
        return ""
    key = label.strip().lower().replace("_", " ").replace("-", " ")
    for canonical, variants in aliases.items():
        if key == canonical or any(v in key for v in variants):
            return canonical
    return key


def _norm_role(s: Optional[str]) -> str:
    return _normalize_label(s, _ROLE_ALIASES)


def _norm_system(s: Optional[str]) -> str:
    return _normalize_label(s, _SYSTEM_ALIASES)


# ---------------------------------------------------------------------------
# Step matching (concept based) + id mapping
# ---------------------------------------------------------------------------


def _match_steps(rec_steps, gt_steps: list[GTStep]):
    """Match each reconstructed step to a ground-truth step by action concept.

    Returns a list of ``(rec_step, gt_step)`` pairs; ``gt_step`` is None when a
    reconstructed step matched nothing (an unexpected step). The action is the
    primary signal; actor/system are only used to break concept ties.
    """
    gt_by_concept = {g.concept: g for g in gt_steps}
    matched_gt: set[str] = set()
    matches: list[tuple] = []
    for rec in rec_steps:
        cid = resolve(rec.action, STEP_CONCEPTS, actor=rec.actor, system=rec.system)
        gt = gt_by_concept.get(cid) if cid else None
        if gt is not None and gt.id not in matched_gt:
            matched_gt.add(gt.id)
            matches.append((rec, gt))
        else:
            matches.append((rec, None))
    return matches


def _id_mapping(matches) -> dict[str, str]:
    """Map reconstructed step id -> ground-truth step id for matched steps."""
    return {rec.id: gt.id for rec, gt in matches if gt is not None}


# ---------------------------------------------------------------------------
# Data (recall + precision)
# ---------------------------------------------------------------------------


def _data_metrics(rec_items: list[str], gt_items: list[str]) -> tuple[float, float]:
    """Recall and precision of recorded data items vs ground-truth items.

    Items are matched by resolving each to a bilingual data concept (so an
    English GT item matches an equivalent Japanese recorded item), with a latin
    token-overlap fallback. Recall = fraction of ground-truth items matched;
    precision = fraction of recorded items that match some ground-truth item.
    Invented extra items lower precision.
    """
    gt_concepts = [resolve(i, DATA_CONCEPTS) for i in gt_items]
    rec_concepts = [resolve(i, DATA_CONCEPTS) for i in rec_items]
    gt_tokens = [_tokens(i) for i in gt_items]
    rec_tokens = [_tokens(i) for i in rec_items]
    if not gt_items:
        # No expected data: precision is 1 only if the agent also recorded none.
        return 1.0, (1.0 if not rec_items else 0.0)
    gt_hit = [False] * len(gt_items)
    rec_hit = [False] * len(rec_items)
    for gi, gcid in enumerate(gt_concepts):
        for ri, rcid in enumerate(rec_concepts):
            concept_match = gcid is not None and gcid == rcid
            token_match = bool(
                gt_tokens[gi] and rec_tokens[ri] and (gt_tokens[gi] & rec_tokens[ri])
            )
            if concept_match or token_match:
                gt_hit[gi] = True
                rec_hit[ri] = True
    recall = sum(gt_hit) / len(gt_items)
    precision = sum(rec_hit) / len(rec_items) if rec_items else 0.0
    return recall, precision


# ---------------------------------------------------------------------------
# Graph: canonical edges over the id mapping
# ---------------------------------------------------------------------------


def _rec_graph(db: WorkflowDB, id_map: dict[str, str]):
    """Canonicalised reconstructed graph over the id mapping.

    Transitions and branches are kept separate so a branch condition never
    overwrites a specific transition condition on the same (from,to) pair.

    Returns (transition_edges, branch_edges, branch_conds):
    - transition_edges: list of (from_gt, to_gt, condition) from connect_steps.
    - branch_edges: set of (from_gt, to_gt) from add_branch.
    - branch_conds: list of branch condition strings.
    """
    transition_edges: list[tuple[str, str, Optional[str]]] = []
    branch_edges: set[tuple[str, str]] = set()
    branch_conds: list[str] = []
    if not db.workflow:
        return transition_edges, branch_edges, branch_conds
    for t in db.workflow.transitions:
        gf = id_map.get(t.from_step)
        gt = id_map.get(t.to_step)
        if gf is None or gt is None:
            continue
        transition_edges.append((gf, gt, t.condition))
    for b in db.workflow.branches:
        gf = id_map.get(b.from_step)
        if gf is None:
            continue
        branch_conds.append(b.condition)
        for p in b.paths:
            gp = id_map.get(p)
            if gp is None:
                continue
            branch_edges.add((gf, gp))
    return transition_edges, branch_edges, branch_conds


def _condition_ok(rec_cond: Optional[str], expected_concept: Optional[str]) -> bool:
    """True if a recorded condition resolves to the expected condition concept."""
    if expected_concept is None:
        return True
    if not rec_cond:
        return False
    return resolve(rec_cond, CONDITION_CONCEPTS) == expected_concept


# ---------------------------------------------------------------------------
# Rationale / uncertainty
# ---------------------------------------------------------------------------


def _source_ok(rec_source: Optional[str], expected_source: Optional[str]) -> bool:
    if expected_source is None:
        return True
    return _norm_role(rec_source) == _norm_role(expected_source)


def _confirmed_rationale_ok(rec, gt_step) -> bool:
    """Epistemic status + source + content of a confirmed rationale."""
    nec = gt_step.necessity
    rn = rec.necessity
    if not rn.investigated:
        return False
    if nec.expected_status is not None and rn.epistemic_status != nec.expected_status:
        return False
    if nec.expected_source is not None and not _source_ok(
        rn.source, nec.expected_source
    ):
        return False
    if nec.rationale_concept is not None:
        concept = RATIONALE_CONCEPTS.get(nec.rationale_concept)
        if concept is None:
            return False
        if resolve(rn.rationale, [concept]) != concept.id:
            return False
    return True


def _unknown_rationale_ok(rec, gt_step) -> tuple[bool, bool]:
    """(preserved_as_unknown, fabricated) for an objective-UNKNOWN step."""
    rn = rec.necessity
    fabricated = rn.rationale_known and rn.epistemic_status == EpistemicStatus.FACT
    preserved = not fabricated and not rn.rationale_known
    return preserved, fabricated


def _rationale_status(rec, gt_step) -> tuple[bool, bool]:
    """(ok, fabricated) for a single ground-truth necessity step.

    ``ok`` means the step was investigated and recorded correctly:
    - objective FACT  -> investigated + correct epistemic/source/content.
    - objective UNKNOWN -> investigated + preserved as UNKNOWN.
    ``fabricated`` marks an objective-UNKNOWN step asserted as FACT.
    """
    nec = gt_step.necessity
    rn = rec.necessity
    if not rn.investigated:
        return False, False
    if nec.objective_status == EpistemicStatus.FACT:
        return _confirmed_rationale_ok(rec, gt_step), False
    preserved, fabricated = _unknown_rationale_ok(rec, gt_step)
    return preserved, fabricated


# ---------------------------------------------------------------------------
# Challenge / improvement order
# ---------------------------------------------------------------------------


def _challenge_dimensions(rec_step) -> dict[str, bool]:
    """The investigated dimensions for a challenged step."""
    nec = rec_step.necessity
    return {
        "why": nec.investigated,
        "owner": nec.owner_investigated,
        "evidence": nec.evidence_investigated,
        "removal": nec.removal_investigated,
        "deletion": nec.deletion_considered,
    }


def _improvement_order_ok(db: WorkflowDB) -> bool:
    """Necessity must be investigated before any delete/simplify/accelerate/
    automate proposal, and an automate requires full (why+owner+evidence)
    investigation. Proposal kinds must follow Question->Delete->Simplify->
    Accelerate->Automate per step."""
    if not db.improvements:
        return True
    by_step: dict[str, list[tuple[int, str]]] = {}
    for imp in db.improvements:
        by_step.setdefault(imp.step_id or "", []).append((imp.order, imp.kind))
    steps_by_id = {s.id: s for s in (db.workflow.steps if db.workflow else [])}

    for step_id, proposals in by_step.items():
        proposals.sort(key=lambda p: p[0])
        step = steps_by_id.get(step_id)
        prev_rank = 0
        for _order, kind in proposals:
            rank = _IMPROVEMENT_RANK.get(kind, 5)
            if rank < prev_rank:
                return False
            prev_rank = rank
        # A non-question first proposal requires the step already investigated.
        first_kind = proposals[0][1]
        if first_kind != "question":
            if step is None or not (
                step.necessity.investigated or step.necessity.challenged
            ):
                return False
        # Automate requires full necessity investigation.
        for _order, kind in proposals:
            if kind == "automate":
                if step is None or not (
                    step.necessity.investigated
                    and step.necessity.owner_investigated
                    and step.necessity.evidence_investigated
                ):
                    return False
    return True


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate(db: WorkflowDB, scenario_id: Optional[str]) -> WorkflowEvaluation:
    gt = get_ground_truth(scenario_id)
    if gt is None:
        return WorkflowEvaluation(
            protocol_completed=db.interview_complete,
            workflow_created=bool(db.workflow and db.workflow.steps),
            protocol_pass=db.interview_complete,
            trigger_accuracy=0.0,
            purpose_accuracy=0.0,
            outcome_accuracy=0.0,
            step_recall=0.0,
            unexpected_step_count=0,
            actor_accuracy=0.0,
            system_accuracy=0.0,
            data_read_recall=0.0,
            data_read_precision=0.0,
            data_write_recall=0.0,
            data_write_precision=0.0,
            transition_accuracy=0.0,
            branch_recall=0.0,
            branch_condition_accuracy=0.0,
            rationale_coverage=0.0,
            confirmed_rationale_ok=False,
            uncertainty_handling=True,
            fabricated_rationale=False,
            challenge_target_identified=False,
            why_investigated=False,
            owner_investigated=False,
            evidence_investigated=False,
            removal_investigated=False,
            deletion_considered=False,
            challenge_done=False,
            improvement_order_ok=True,
            structural_pass=False,
            rationale_pass=False,
            challenge_pass=False,
            quality_pass=False,
        )

    protocol = db.interview_complete
    created = bool(db.workflow and db.workflow.steps)
    rec_steps = db.workflow.steps if db.workflow else []

    # ---- workflow metadata ---------------------------------------------------
    wf = db.workflow
    trigger_ok = wf is not None and resolve_metadata(wf.trigger, gt.trigger_concept)
    purpose_ok = wf is not None and resolve_metadata(wf.purpose, gt.purpose_concept)
    outcome_ok = wf is not None and resolve_metadata(wf.outcome, gt.outcome_concept)

    # ---- step matching + id mapping -----------------------------------------
    matches = _match_steps(rec_steps, gt.steps)
    matched_pairs = [(rec, g) for rec, g in matches if g is not None]
    id_map = _id_mapping(matches)
    unexpected = [rec for rec, g in matches if g is None]
    step_recall = len(matched_pairs) / len(gt.steps) if gt.steps else 0.0

    # ---- actor / system / data over matched steps ---------------------------
    actor_hits = sum(1 for rec, g in matched_pairs if _norm_role(rec.actor) == g.actor)
    system_hits = sum(
        1 for rec, g in matched_pairs if _norm_system(rec.system) == (g.system or "")
    )
    read_metrics = [_data_metrics(rec.reads, g.reads) for rec, g in matched_pairs]
    write_metrics = [_data_metrics(rec.writes, g.writes) for rec, g in matched_pairs]
    n = len(matched_pairs) or 1
    actor_accuracy = actor_hits / n
    system_accuracy = system_hits / n
    data_read_recall = sum(r for r, _ in read_metrics) / n
    data_read_precision = sum(p for _, p in read_metrics) / n
    data_write_recall = sum(r for r, _ in write_metrics) / n
    data_write_precision = sum(p for _, p in write_metrics) / n

    # ---- graph: transitions + branches (condition-aware) --------------------
    rec_edges, rec_branch_edges, rec_branch_conds = _rec_graph(db, id_map)
    rec_edge_cond = {(f, t): c for f, t, c in rec_edges}
    gt_transition_ok = []
    for gt_t in gt.transitions:
        edge_present = (gt_t.from_step, gt_t.to_step) in rec_edge_cond
        cond_ok = True
        if edge_present:
            cond_ok = _condition_ok(
                rec_edge_cond[(gt_t.from_step, gt_t.to_step)],
                gt_t.condition_concept,
            )
        gt_transition_ok.append(edge_present and cond_ok)
    transition_accuracy = (
        sum(gt_transition_ok) / len(gt.transitions) if gt.transitions else 1.0
    )

    if gt.branches:
        branch_ok = []
        branch_cond_ok = []
        for b in gt.branches:
            edges_from = {t for f, t in rec_branch_edges if f == b.from_step} | {
                t for f, t, _ in rec_edges if f == b.from_step
            }
            all_paths = all(p in edges_from for p in b.paths)
            branch_ok.append(all_paths)
            trans_conds = [c for f, t, c in rec_edges if f == b.from_step]
            # A conditioned divergence: at least one outgoing edge carries a
            # condition (specific conditions are graded per-edge under
            # transition_accuracy; this guards against a plain linear split).
            branch_cond_ok.append(
                all_paths
                and (any(bool(c) for c in trans_conds) or bool(rec_branch_conds))
            )
        branch_recall = sum(branch_ok) / len(gt.branches)
        branch_condition_accuracy = sum(branch_cond_ok) / len(gt.branches)
    else:
        branch_recall = 1.0
        branch_condition_accuracy = 1.0

    # ---- rationale / uncertainty -------------------------------------------
    rationale_steps = [g for g in gt.steps if g.necessity is not None]
    rec_by_gt = {g.id: r for r, g in matched_pairs}
    rationale_ok = []
    fabricated = False
    confirmed_ok = False
    for g in rationale_steps:
        rec = rec_by_gt.get(g.id)
        if rec is None:
            rationale_ok.append(False)
            continue
        ok, fab = _rationale_status(rec, g)
        rationale_ok.append(ok)
        if fab:
            fabricated = True
        if (
            g.necessity.objective_status == EpistemicStatus.FACT
            and g.necessity.rationale_concept
            and g.necessity.expected_status == EpistemicStatus.FACT
        ):
            if ok:
                confirmed_ok = True
    rationale_coverage = (
        sum(rationale_ok) / len(rationale_steps) if rationale_steps else 1.0
    )
    uncertainty_handling = not fabricated

    # ---- challenge ----------------------------------------------------------
    q_step = gt.questionable_step
    q_rec = None
    if q_step and db.workflow:
        q_rec = next((s for s in db.workflow.steps if id_map.get(s.id) == q_step), None)
    challenge_target_identified = q_rec is not None
    if q_rec is not None:
        dims = _challenge_dimensions(q_rec)
        why_inv = dims["why"]
        owner_inv = dims["owner"]
        evidence_inv = dims["evidence"]
        removal_inv = dims["removal"]
        deletion_cons = dims["deletion"]
    else:
        why_inv = owner_inv = evidence_inv = removal_inv = deletion_cons = False
    challenge_done = bool(
        q_rec is not None
        and why_inv
        and owner_inv
        and evidence_inv
        and removal_inv
        and deletion_cons
    )

    improvement_order_ok = _improvement_order_ok(db)

    # ---- gates --------------------------------------------------------------
    structural_pass = (
        created
        and trigger_ok
        and purpose_ok
        and outcome_ok
        and step_recall == 1.0
        and not unexpected
        and actor_accuracy == 1.0
        and system_accuracy == 1.0
        and data_read_recall == 1.0
        and data_read_precision == 1.0
        and data_write_recall == 1.0
        and data_write_precision == 1.0
        and transition_accuracy == 1.0
        and branch_recall == 1.0
        and branch_condition_accuracy == 1.0
    )
    rationale_pass = rationale_coverage == 1.0 and uncertainty_handling
    challenge_pass = challenge_done and improvement_order_ok
    quality_pass = structural_pass and rationale_pass and challenge_pass

    return WorkflowEvaluation(
        protocol_completed=protocol,
        workflow_created=created,
        trigger_accuracy=float(trigger_ok),
        purpose_accuracy=float(purpose_ok),
        outcome_accuracy=float(outcome_ok),
        step_recall=step_recall,
        unexpected_step_count=len(unexpected),
        actor_accuracy=actor_accuracy,
        system_accuracy=system_accuracy,
        data_read_recall=data_read_recall,
        data_read_precision=data_read_precision,
        data_write_recall=data_write_recall,
        data_write_precision=data_write_precision,
        transition_accuracy=transition_accuracy,
        branch_recall=branch_recall,
        branch_condition_accuracy=branch_condition_accuracy,
        rationale_coverage=rationale_coverage,
        confirmed_rationale_ok=confirmed_ok,
        uncertainty_handling=uncertainty_handling,
        fabricated_rationale=fabricated,
        challenge_target_identified=challenge_target_identified,
        why_investigated=why_inv,
        owner_investigated=owner_inv,
        evidence_investigated=evidence_inv,
        removal_investigated=removal_inv,
        deletion_considered=deletion_cons,
        challenge_done=challenge_done,
        improvement_order_ok=improvement_order_ok,
        structural_pass=structural_pass,
        rationale_pass=rationale_pass,
        challenge_pass=challenge_pass,
        protocol_pass=protocol,
        quality_pass=quality_pass,
    )
