"""Hidden Truth-claim catalog and stakeholder provenance (evaluator-only).

This module is the **simulator-side private metadata emitter**. The benchmark
privately records which Truth claim each stakeholder utterance came from:

- ``Claim`` — one evaluator-only Truth claim: a stakeholder-visible
  read/write fact (Truth node + axis + Truth data concept), e.g.
  ``cq.writes.tc_quote``. Only claims allowed by the ``StakeholderFilter``
  enter the private catalog.
- ``build_claims`` — derive the private claim catalog from the Truth DAG +
  StakeholderFilter + the scenario's surface-term table.
- ``build_provenance_ledger`` — for every stakeholder (user) message,
  deterministically record which claims the utterance expressed and which
  surface terms it used. This is a pure function of the simulator's own
  output (the natural-language text) and its private catalog — it is the
  metadata the stakeholder simulator emits together with its response.
- ``validate_ledger`` — reject unknown claim ids and claims outside
  stakeholder visibility.

The Agent never sees claim ids, the catalog, or the ledger: they live only in
this module and in the evaluator's inputs. The evaluator never infers
semantic support by reading Observation text — it consumes the ledger.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.business_interview.dag import BusinessDAG, InterviewDB
from tau2.domains.business_interview.stakeholder import StakeholderFilter

DataAxis = Literal["reads", "writes"]


class Claim(BaseModel):
    """One hidden Truth claim: a stakeholder-visible read/write fact.

    ``id`` binds a Truth node + axis + Truth data concept (e.g.
    ``cq.writes.tc_quote``). ``surface_terms`` are the natural phrases the
    stakeholder uses for this fact (from its private known-info wording);
    they are simulator-side vocabulary, never Ground Truth canonical labels
    required of the Agent.
    """

    id: str
    node_id: str
    axis: DataAxis
    concept_id: str
    surface_terms: list[str] = Field(default_factory=list)


def _phrases(claims: dict[str, Claim]) -> list[tuple[str, list[str]]]:
    """(normalized phrase, claim ids) pairs sorted longest-first.

    A phrase may belong to several claims (e.g. ``customer information`` is
    the surface term of the customer claim on both cc.reads and cq.reads).
    """
    by_phrase: dict[str, list[str]] = {}
    for cid, claim in claims.items():
        for term in claim.surface_terms:
            norm = " ".join(term.lower().split())
            by_phrase.setdefault(norm, []).append(cid)
    return sorted(by_phrase.items(), key=lambda kv: -len(kv[0]))


def _norm_text(text: Optional[str]) -> str:
    return " ".join((text or "").lower().split())


def derive_utterance_claims(
    text: Optional[str],
    claims: dict[str, Claim],
    stop_phrases: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    """Deterministic derivation of the claims expressed by one utterance.

    Greedy longest-phrase matching over the normalized text: the longest
    catalog phrase starting at the earliest position wins and consumes its
    character span, so ``"quotation request"`` supports the request claim
    (and consumes ``"quotation"``), while ``"I create the quotation"``
    supports the quote claim. ``stop_phrases`` (e.g. ``quotation
    information``) consume their span without supporting any claim.

    Returns ``{claim_id: [surface terms matched]}``. This is the private
    metadata the stakeholder simulator emits with its response; it contains
    no Ground Truth canonical labels.
    """
    text_norm = _norm_text(text)
    if not text_norm:
        return {}
    phrases = _phrases(claims)
    # combined candidates: (length desc, phrase, claim ids or [] for stops)
    candidates = [(len(phrase), phrase, cids) for phrase, cids in phrases] + [
        (len(stop), stop, [])
        for stop in sorted(
            (" ".join(p.lower().split()) for p in stop_phrases), key=len, reverse=True
        )
    ]
    candidates.sort(key=lambda c: (-c[0], not c[2]))
    result: dict[str, list[str]] = {}
    pos = 0
    n = len(text_norm)
    while pos < n:
        best: Optional[tuple[int, str, list[str]]] = None
        for length, phrase, cids in candidates:
            if phrase and text_norm.startswith(phrase, pos):
                best = (length, phrase, cids)
                break
        if best is None:
            pos += 1
            continue
        length, phrase, cids = best
        for cid in cids:
            result.setdefault(cid, [])
            if phrase not in result[cid]:
                result[cid].append(phrase)
        pos += max(length, 1)
    return result


def build_claims(
    truth: BusinessDAG,
    stakeholder: StakeholderFilter,
    surface_terms: dict[tuple[str, str, str], list[str]],
) -> dict[str, Claim]:
    """Build the private claim catalog from the Truth DAG + visibility.

    One claim per Truth read/write concept ref on a **stakeholder-visible**
    axis. Claim ids look like ``cq.writes.tc_quote``. Hidden axes (sq.reads,
    sq.writes, ...) produce no claims and can never enter the catalog.
    ``surface_terms`` keys are ``(truth_node_id, axis, truth_concept_id)``.
    """
    claims: dict[str, Claim] = {}
    for nid, node in truth.nodes.items():
        visible = stakeholder.visible_attributes_for(nid)
        for axis in ("reads", "writes"):
            if axis not in visible:
                continue
            for ref in getattr(node, axis):
                cid = f"{nid}.{axis}.{ref.concept_id}"
                terms = surface_terms.get((nid, axis, ref.concept_id), [])
                claims[cid] = Claim(
                    id=cid,
                    node_id=nid,
                    axis=axis,
                    concept_id=ref.concept_id,
                    surface_terms=list(terms),
                )
    return claims


def build_provenance_ledger(
    db: InterviewDB,
    claims: dict[str, Claim],
    stop_phrases: tuple[str, ...] = (),
) -> dict[int, dict[str, list[str]]]:
    """The stakeholder simulator's hidden provenance, keyed by message turn.

    For every user (stakeholder) message in the conversation ledger,
    deterministically record which visible Truth claims the utterance
    expressed and which surface terms it used. Returns
    ``{turn_index: {claim_id: [surface terms]}}``.
    """
    ledger: dict[int, dict[str, list[str]]] = {}
    for turn, msg in enumerate(db.messages):
        if msg.get("role") != "user":
            continue
        derived = derive_utterance_claims(msg.get("content"), claims, stop_phrases)
        if derived:
            ledger[turn] = derived
    return ledger


def validate_ledger(
    ledger: dict[int, dict[str, list[str]]],
    claims: dict[str, Claim],
    stakeholder: StakeholderFilter,
) -> None:
    """Deterministically reject invalid provenance metadata.

    Raises ``ValueError`` on an unknown claim id or a claim whose node/axis
    is outside stakeholder visibility (defense in depth — the derivation
    above can only ever emit valid entries).
    """
    for turn, entry in ledger.items():
        for cid in entry:
            if cid not in claims:
                raise ValueError(
                    f"provenance for turn {turn} references unknown claim id {cid!r}"
                )
            claim = claims[cid]
            if claim.axis not in stakeholder.visible_attributes_for(claim.node_id):
                raise ValueError(
                    f"provenance for turn {turn} references claim {cid!r} which "
                    f"is outside stakeholder visibility"
                )
