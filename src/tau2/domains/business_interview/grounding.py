"""Shared deterministic provenance machinery (tools + evaluator).

The stakeholder's private annotations (``SemanticAnnotation``) attach exact
character spans of its messages to stakeholder semantic IDs. Agent evidence
spans resolve to semantic IDs through the GLOBAL span rule implemented here — the single implementation used by the tools
(evidence-ref shape validation, ABSENT/DONT_KNOW recording) and the
evaluator (evidence hygiene diagnostics). No other span resolution logic
exists.

Global span rule (deterministic, never semantic):

- equal-span annotations win: when the evidence span is exactly an
  annotation span, it covers exactly the semantic ids of those annotations;
- otherwise it covers the ids of the maximal annotation spans it contains
  plus the ids of annotation spans strictly containing it.

A span covering several DISTINCT semantic ids is globally ambiguous and
grounds nothing (no cross-credit). Semantic-id interpretation itself is the
canonical resolver in ``StakeholderKnowledgeGraph.resolve`` — provenance here
only produces the id sets.
"""

from typing import Optional

from tau2.domains.business_interview.facts import SemanticAnnotation
from tau2.domains.business_interview.graph import EvidenceRef, InterviewDB


def resolve_span_text(
    text: str, quote: str, occurrence: int
) -> Optional[tuple[int, int]]:
    """Resolve (quote, occurrence) to a character span in ``text``, or None."""
    if not quote:
        return None
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(quote, start + 1)
        if start == -1:
            return None
    return (start, start + len(quote))


def covered_ids_for_span(
    annotations: list[SemanticAnnotation],
    text: str,
    ev_span: tuple[int, int],
) -> set[str]:
    """The semantic IDs an evidence span covers, globally (across ALL
    elements).

    Deterministic containment rule:
    - equal-span annotations win: when the evidence span is exactly an
      annotation span, it covers exactly the semantic ids of those
      annotations;
    - otherwise it covers the ids of the maximal annotation spans it
      contains plus the ids of annotation spans strictly containing it.

    A span covering several DISTINCT semantic ids is globally ambiguous and
    must not be reused across slots.
    """
    resolved: list[tuple[SemanticAnnotation, tuple[int, int]]] = []
    for annotation in annotations:
        span = resolve_span_text(text, annotation.quote, annotation.occurrence)
        if span is not None:
            resolved.append((annotation, span))
    equal = {a.semantic_id for a, s in resolved if s == ev_span}
    if equal:
        return equal
    contained = [
        (a, s) for a, s in resolved if s[0] >= ev_span[0] and s[1] <= ev_span[1]
    ]
    containing = [
        (a, s)
        for a, s in resolved
        if s[0] <= ev_span[0] and s[1] >= ev_span[1] and s != ev_span
    ]
    maximal = [
        (a, s)
        for a, s in contained
        if not any(s2 != s and s2[0] <= s[0] and s[1] <= s2[1] for _, s2 in contained)
    ]
    return {a.semantic_id for a, _ in maximal} | {a.semantic_id for a, _ in containing}


def obs_by_id(db: InterviewDB, obs_id: str):
    for obs in db.observations:
        if obs.id == obs_id:
            return obs
    return None


def grounded_ids(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    evidence: list[EvidenceRef],
) -> tuple[set[str], int, int]:
    """Semantic IDs grounded by a list of EvidenceRefs via the GLOBAL span
    rule. Returns (grounded ids, invalid count, ambiguous count)."""
    grounded: set[str] = set()
    invalid = 0
    ambiguous = 0
    for ev in evidence:
        obs = obs_by_id(db, ev.observation_id)
        if obs is None:
            invalid += 1
            continue
        ev_span = ev.resolve_span(obs.text)
        if ev_span is None:
            invalid += 1
            continue
        covered = covered_ids_for_span(annotations.get(obs.turn, []), obs.text, ev_span)
        if len(covered) > 1:
            ambiguous += 1
            continue
        if len(covered) == 1:
            grounded.update(covered)
    return grounded, invalid, ambiguous


def grounded_refs(
    db: InterviewDB,
    annotations: dict[int, list[SemanticAnnotation]],
    evidence: list[EvidenceRef],
) -> tuple[list[tuple[EvidenceRef, str]], int, int]:
    """Per-ref single-id resolution via the GLOBAL span rule.

    Returns ([(ref, semantic_id)] for refs resolving to exactly one id,
    invalid count, ambiguous count). Every ref of the input must appear in
    the result for the evidence to be usable (invalid + ambiguous == 0 and
    len(results) == len(evidence)).
    """
    results: list[tuple[EvidenceRef, str]] = []
    invalid = 0
    ambiguous = 0
    for ev in evidence:
        obs = obs_by_id(db, ev.observation_id)
        if obs is None:
            invalid += 1
            continue
        ev_span = ev.resolve_span(obs.text)
        if ev_span is None:
            invalid += 1
            continue
        covered = covered_ids_for_span(annotations.get(obs.turn, []), obs.text, ev_span)
        if len(covered) > 1:
            ambiguous += 1
            continue
        if len(covered) == 1:
            results.append((ev, next(iter(covered))))
        # len(covered) == 0: valid span, grounds nothing (unrelated speech)
    return results, invalid, ambiguous
