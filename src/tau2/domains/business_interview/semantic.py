"""Structural workflow-reconstruction evaluator for business_interview (v2).

The agent records a reconstructed ``Workflow`` (steps, transitions, branches,
and per-step necessity). The evaluator compares it against the evaluator-only
ground truth (``ground_truth.py``) and produces a ``WorkflowEvaluation`` with
structural, epistemic and challenge metrics.

Steps are matched by content (action text similarity plus actor/system match),
because the agent uses its own step labels — the benchmark is not a game of
guessing hidden canonical ids.
"""

import re
from typing import Optional

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

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# CJK ranges so character-bigram matching works for Japanese actions/conditions.
_CJK_RE = re.compile(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9faf]")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _action_sig(text: str) -> set[str]:
    """Language-agnostic action/condition signature: latin tokens plus CJK
    character bigrams, so Japanese and English texts with the same meaning can
    be matched (the benchmark is not a hidden-id guessing game)."""
    s = (text or "").lower()
    toks = set(_TOKEN_RE.findall(s))
    norm = _CJK_RE.sub("", s)
    big = {norm[i : i + 2] for i in range(len(norm) - 1)}
    return toks | big


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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


def _data_recall(rec_items: list[str], gt_items: list[str]) -> float:
    """Fraction of ground-truth data items covered by recorded items (token overlap)."""
    if not gt_items:
        return 1.0
    gt_tokens = [_tokens(i) for i in gt_items]
    rec_tokens = [_tokens(i) for i in rec_items]
    hit = 0
    for gt in gt_tokens:
        if any(gt & rt for rt in rec_tokens):
            hit += 1
    return hit / len(gt_items)


def _step_score(rec, gt: GTStep) -> float:
    """Similarity between a reconstructed step and a ground-truth step."""
    score = _jaccard(_action_sig(rec.action), _action_sig(gt.action))
    if _norm_role(rec.actor) and _norm_role(rec.actor) == gt.actor:
        score += 0.3
    if rec.system and _norm_system(rec.system) == (gt.system or ""):
        score += 0.3
    return score


def _match_steps(rec_steps, gt_steps: list[GTStep]):
    """Greedily match reconstructed steps to ground-truth steps.

    Returns a list of (rec_step, gt_step) pairs; gt_step is None when a
    reconstructed step matched nothing (an unexpected step).
    """
    threshold = 0.45
    matched_gt: set[int] = set()
    matches: list[tuple] = []
    for rec in rec_steps:
        best = None
        best_score = threshold
        for gi, gt in enumerate(gt_steps):
            if gi in matched_gt:
                continue
            s = _step_score(rec, gt)
            if s > best_score:
                best_score = s
                best = gi
        if best is not None:
            matched_gt.add(best)
            matches.append((rec, gt_steps[best]))
        else:
            matches.append((rec, None))
    return matches


def _rec_edges(db: WorkflowDB) -> set[tuple[str, str, Optional[str]]]:
    """All reconstructed directed edges (from, to, condition)."""
    edges: set[tuple[str, str, Optional[str]]] = set()
    if not db.workflow:
        return edges
    for t in db.workflow.transitions:
        edges.add((t.from_step, t.to_step, t.condition))
    for b in db.workflow.branches:
        for p in b.paths:
            edges.add((b.from_step, p, b.condition))
    return edges


def _cond_sig(c: str) -> set[str]:
    """Condition signature: latin tokens when present, else CJK bigrams."""
    toks = _tokens(c)
    return toks if toks else _action_sig(c)


def _cond_matches(rec_cond: Optional[str], gt_cond: str) -> bool:
    """Lenient branch-condition match: the reconstructed condition should cover
    the ground-truth condition's core tokens (e.g. 'amount over 1,000,000'
    covers the canonical branch condition 'amount threshold')."""
    if not rec_cond:
        return False
    gt_s = _cond_sig(gt_cond)
    rec_s = _cond_sig(rec_cond)
    if not gt_s:
        return True
    return len(gt_s & rec_s) / len(gt_s) >= 0.3


def _improvement_order_ok(db: WorkflowDB) -> bool:
    """Proposals for each step must follow Question→Delete→Simplify→Accelerate→
    Automate. Automating a step without first questioning it (or challenging its
    necessity) is a poor improvement order."""
    if not db.improvements:
        return True
    by_step: dict[str, list[tuple[int, str]]] = {}
    for imp in db.improvements:
        by_step.setdefault(imp.step_id or "", []).append((imp.order, imp.kind))
    challenged_ids: set[str] = set()
    if db.workflow:
        challenged_ids = {s.id for s in db.workflow.steps if s.necessity.challenged}
    for step_id, proposals in by_step.items():
        proposals.sort(key=lambda p: p[0])
        first_rank = _IMPROVEMENT_RANK.get(proposals[0][1], 5)
        if step_id in challenged_ids:
            continue  # a challenge counts as questioning first
        if first_rank > _IMPROVEMENT_RANK["simplify"]:
            return False
    return True


def evaluate(db: WorkflowDB, scenario_id: Optional[str]) -> WorkflowEvaluation:
    gt = get_ground_truth(scenario_id)
    if gt is None:
        return WorkflowEvaluation(
            protocol_completed=db.interview_complete,
            workflow_created=bool(db.workflow and db.workflow.steps),
            protocol_pass=db.interview_complete,
        )

    protocol = db.interview_complete
    created = bool(db.workflow and db.workflow.steps)
    rec_steps = db.workflow.steps if db.workflow else []

    # ---- Step matching ------------------------------------------------------
    matches = _match_steps(rec_steps, gt.steps)
    matched_pairs = [(rec, g) for rec, g in matches if g is not None]
    unexpected = [rec for rec, g in matches if g is None]
    step_recall = len(matched_pairs) / len(gt.steps) if gt.steps else 0.0

    # ---- actor / system / data accuracy over matched steps -----------------
    actor_hits = sum(1 for rec, g in matched_pairs if _norm_role(rec.actor) == g.actor)
    system_hits = sum(
        1 for rec, g in matched_pairs if _norm_system(rec.system) == (g.system or "")
    )
    read_scores = [_data_recall(rec.reads, g.reads) for rec, g in matched_pairs]
    write_scores = [_data_recall(rec.writes, g.writes) for rec, g in matched_pairs]
    n = len(matched_pairs) or 1
    actor_accuracy = actor_hits / n
    system_accuracy = system_hits / n
    data_read_accuracy = sum(read_scores) / n
    data_write_accuracy = sum(write_scores) / n

    # ---- transitions --------------------------------------------------------
    rec_edges = _rec_edges(db)
    gt_edges = {(t.from_step, t.to_step) for t in gt.transitions}
    rec_edge_pairs = {(f, t) for f, t, _ in rec_edges}
    transition_accuracy = (
        len(gt_edges & rec_edge_pairs) / len(gt_edges) if gt_edges else 1.0
    )

    # ---- branches -----------------------------------------------------------
    if gt.branches:
        branch_ok = []
        branch_cond_ok = []
        for b in gt.branches:
            edges_from = {t for f, t, _ in rec_edges if f == b.from_step}
            all_paths = all(p in edges_from for p in b.paths)
            branch_ok.append(all_paths)
            conds = [c for f, t, c in rec_edges if f == b.from_step]
            branch_cond_ok.append(
                any(_cond_matches(c, b.condition) for c in conds)
                if all_paths
                else False
            )
        branch_recall = sum(branch_ok) / len(gt.branches)
        branch_condition_accuracy = sum(branch_cond_ok) / len(gt.branches)
    else:
        branch_recall = 1.0
        branch_condition_accuracy = 1.0

    # ---- rationale / uncertainty -------------------------------------------
    rationale_steps = [g for g in gt.steps if g.necessity is not None]
    covered = 0
    fabricated = False
    for g in rationale_steps:
        rec = next((r for r, gg in matched_pairs if gg.id == g.id), None)
        if rec is None:
            continue
        if g.necessity.objective_status == EpistemicStatus.FACT:
            if rec.necessity.rationale_known:
                covered += 1
        else:  # UNKNOWN
            preserved = not rec.necessity.rationale_known or (
                rec.necessity.epistemic_status == EpistemicStatus.UNKNOWN
            )
            if preserved:
                covered += 1
            if rec.necessity.rationale_known and (
                rec.necessity.epistemic_status == EpistemicStatus.FACT
            ):
                fabricated = True
    rationale_coverage = covered / len(rationale_steps) if rationale_steps else 1.0
    uncertainty_handling = not fabricated

    # ---- challenge / improvement order -------------------------------------
    q_step = gt.questionable_step
    challenge_done = False
    if q_step and db.workflow:
        q_rec = next((s for s in db.workflow.steps if s.id == q_step), None)
        challenge_done = bool(q_rec and q_rec.necessity.challenged)

    improvement_order_ok = _improvement_order_ok(db)

    # ---- gates --------------------------------------------------------------
    structural_pass = (
        created
        and step_recall == 1.0
        and not unexpected
        and actor_accuracy == 1.0
        and system_accuracy == 1.0
        and data_read_accuracy == 1.0
        and data_write_accuracy == 1.0
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
        step_recall=step_recall,
        unexpected_step_count=len(unexpected),
        actor_accuracy=actor_accuracy,
        system_accuracy=system_accuracy,
        data_read_accuracy=data_read_accuracy,
        data_write_accuracy=data_write_accuracy,
        transition_accuracy=transition_accuracy,
        branch_recall=branch_recall,
        branch_condition_accuracy=branch_condition_accuracy,
        rationale_coverage=rationale_coverage,
        uncertainty_handling=uncertainty_handling,
        fabricated_rationale=fabricated,
        challenge_done=challenge_done,
        improvement_order_ok=improvement_order_ok,
        structural_pass=structural_pass,
        rationale_pass=rationale_pass,
        challenge_pass=challenge_pass,
        protocol_pass=protocol,
        quality_pass=quality_pass,
    )
