#!/usr/bin/env python3
"""Analysis-only: measure whether WordNet / lemmatization variants could have
rescued the evaluator-too-strict (B) data mismatches in the business_interview
quotation scenario, and how many new (possibly false-positive) matches each
variant would introduce.

READ-ONLY. The evaluator behavior is NOT changed. No LLM run is performed.
Only the already-saved artifacts (seeds 4000-4004) are analysed.

Variants compared (deterministic, all run offline once WordNet is available):

  baseline   - current evaluator _data_recall: raw-token overlap OR substring.
  lemma      - same but each token lemmatized first (inflection only). Using
               NLTK WordNetLemmatizer and inflect (both agree).
  synset     - a truth data token and an agent data token are a match if they
               share >=1 common WordNet synset over the union of ALL senses
               (the risky all-senses approach).
  deriv      - a match if the two tokens have a WordNet derivational relation
               (lemma-level, across all senses).
  hybrid     - conservative cascade: baseline(overlap/substring) OR lemma OR
               derivational (NOT unrestricted synset).

Outputs:
  - recovery counts for the 12 data-related B (evaluator_too_strict) cases
  - a false-positive inventory over the business token pool with a manual
    safe / questionable / false_positive classification
  - sense ambiguity notes for quote / quotation
"""

import glob
import json
import re
from pathlib import Path

# WordNet-backed variant functions are only importable if nltk is installed.
# baseline / lemma_fallback must work without nltk so the report can state the
# dependency boundary precisely.
try:
    from nltk.corpus import wordnet as wn  # type: ignore[import-not-found]
    from nltk.stem import WordNetLemmatizer  # type: ignore[import-not-found]

    _HAS_NLTK = True
    _wn = wn
    _Lemmatizer = WordNetLemmatizer
except Exception:  # pragma: no cover - env-dependent
    _HAS_NLTK = False
    _wn = None  # type: ignore[assignment]
    _Lemmatizer = None  # type: ignore[assignment]

try:
    import inflect  # type: ignore[import-not-found]

    _INFLECT = inflect.engine()
    _HAS_INFLECT = True
except Exception:  # pragma: no cover - env-dependent
    _INFLECT = None
    _HAS_INFLECT = False

ART_DIR = Path("artifacts/business_interview_real_llm")

# Same tokenizer the evaluator uses.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


# ---------------------------------------------------------------------------
# Matching variants
# ---------------------------------------------------------------------------


def _lemmatize_token(tok: str) -> str:
    """Deterministic lemmatization. NLTK WordNetLemmatizer and inflect agree on
    the inflectional lemmas relevant here; prefer NLTK (proper POS-aware noun
    lemmatization), fall back to inflect, then to the raw token."""
    if _HAS_NLTK and _Lemmatizer is not None:
        return _Lemmatizer().lemmatize(tok)
    if _HAS_INFLECT and _INFLECT is not None:
        s = _INFLECT.singular_noun(tok)
        return s if s else tok
    return tok


def _lemma_tokens(text) -> set[str]:
    return {_lemmatize_token(t) for t in _tokens(text)}


def _baseline_match(truth_item, agent_item) -> bool:
    """Current evaluator semantics: token overlap OR substring containment."""
    return (
        bool(_tokens(truth_item) & _tokens(agent_item))
        or truth_item.lower() in (agent_item or "").lower()
    )


def _lemma_match(truth_item, agent_item) -> bool:
    return bool(_lemma_tokens(truth_item) & _lemma_tokens(agent_item))


def _token_synsets(tok: str) -> set[str]:
    if not _HAS_NLTK or _wn is None:
        return set()
    out = set()
    for ss in _wn.synsets(tok):
        out.add(ss.name())
    return out


def _token_derivs(tok: str) -> set[str]:
    """Return the set of (synset.name, lemma.name) derivational targets reachable
    from any lemma of ``tok``."""
    if not _HAS_NLTK or _wn is None:
        return set()
    out = set()
    for ss in _wn.synsets(tok):
        for lemma in ss.lemmas():
            for rel in lemma.derivationally_related_forms():
                out.add((rel.synset().name(), rel.name()))
    return out


def _token_synset_related(a: str, b: str) -> bool:
    return bool(_token_synsets(a) & _token_synsets(b))


def _token_derivationally_related(a: str, b: str) -> bool:
    return bool(_token_derivs(a) & _token_derivs(b))


def _synset_match(truth_item, agent_item) -> bool:
    """All-senses union: a token pair shares any synset."""
    ta = _tokens(truth_item)
    aa = _tokens(agent_item)
    for t in ta:
        for a in aa:
            if _token_synset_related(t, a):
                return True
    return False


def _deriv_match(truth_item, agent_item) -> bool:
    ta = _tokens(truth_item)
    aa = _tokens(agent_item)
    for t in ta:
        for a in aa:
            if _token_derivationally_related(t, a):
                return True
    return False


def _hybrid_match(truth_item, agent_item) -> bool:
    # conservative cascade: exact/substring -> lemma -> derivational
    if _baseline_match(truth_item, agent_item):
        return True
    if _lemma_match(truth_item, agent_item):
        return True
    if _deriv_match(truth_item, agent_item):
        return True
    return False


_VARIANTS = {
    "baseline": _baseline_match,
    "lemma": _lemma_match,
    "synset": _synset_match,
    "deriv": _deriv_match,
    "hybrid": _hybrid_match,
}


def _recall(agent_items, truth_items, match_fn) -> float:
    """Mirror of the evaluator's _data_recall with a pluggable predicate."""
    if not truth_items:
        return 1.0 if not agent_items else 0.0
    hits = 0
    for gi in range(len(truth_items)):
        if not truth_items[gi]:
            hits += 1
            continue
        if any(match_fn(truth_items[gi], a) for a in agent_items):
            hits += 1
    return hits / len(truth_items)


# ---------------------------------------------------------------------------
# Build per-node read/write data from saved artifacts (mirrors the inventory
# script's DAG reconstruction).
# ---------------------------------------------------------------------------


def _cv(x):
    return x if x is not None else {}


def _fix_node(n):
    n = dict(n)
    n["action"] = _cv(n.get("action"))
    n["actor"] = _cv(n.get("actor"))
    n["system"] = _cv(n.get("system"))
    n["primitive"] = _cv(n.get("primitive"))
    n["reads"] = [_cv(r) for r in n.get("reads", [])]
    n["writes"] = [_cv(w) for w in n.get("writes", [])]
    if n.get("necessity"):
        n["necessity"] = {k: _cv(v) for k, v in n["necessity"].items()}
    return n


def _val(v):
    if v is None:
        return None
    return v.value if hasattr(v, "value") else (v or {}).get("value")


def _load_truth_and_spec():
    from tau2.domains.business_interview.scenario import quotation_spec, quotation_truth

    return quotation_truth(), quotation_spec()


def _collect_node_data():
    """Return list of (run, truth_node_id, attr, truth_items, agent_items) for
    reads and writes across all 5 artifacts, using the evaluator's own node
    mapping."""
    from tau2.domains.business_interview.dag import BusinessDAG
    from tau2.domains.business_interview.evaluation import _match_nodes

    truth, spec = _load_truth_and_spec()
    records = []
    for path in sorted(glob.glob(str(ART_DIR / "run_*_seed4*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read artifact {path}: {exc}")
        name = Path(path).name
        fdag = dict(d["final_dag"])
        fdag["nodes"] = {nid: _fix_node(n) for nid, n in fdag["nodes"].items()}
        fdag["edges"] = {
            eid: {**e, "predicate": _cv(e.get("predicate"))}
            for eid, e in fdag["edges"].items()
        }
        agent = BusinessDAG.model_validate(fdag)
        mapping = _match_nodes(agent, truth, spec)
        for anid, tnid in mapping.items():
            an = agent.nodes[anid]
            tn = truth.nodes[tnid]
            for attr in ("reads", "writes"):
                records.append(
                    {
                        "run": name,
                        "truth_node": tnid,
                        "attr": attr,
                        "truth_items": [_val(x) for x in getattr(tn, attr)],
                        "agent_items": [_val(x) for x in getattr(an, attr)],
                    }
                )
    return records


# ---------------------------------------------------------------------------
# Positive-case recovery
# ---------------------------------------------------------------------------


def _identify_b_cases(records):
    """The B (evaluator_too_strict) data cases: both truth and agent record
    non-empty data that differ only in wording (quote <-> quotation). Empty
    agent data (information-not-obtained / truth-modeling) is not a lexical
    matching problem and no matcher can rescue it."""
    cases = []
    for rec in records:
        # baseline miss
        base_hit = _recall(rec["agent_items"], rec["truth_items"], _baseline_match)
        if base_hit >= 1.0:
            continue
        # must be non-empty on both sides to be a lexical (B) problem
        truth_nonempty = any(bool(x) for x in rec["truth_items"])
        agent_nonempty = any(bool(x) for x in rec["agent_items"])
        if not (truth_nonempty and agent_nonempty):
            continue
        cases.append(rec)
    return cases


def _analyze_case(rec):
    row = {
        "run": rec["run"],
        "truth_node": rec["truth_node"],
        "attr": rec["attr"],
        "truth": rec["truth_items"],
        "agent": rec["agent_items"],
        "hits": {},
    }
    for name, fn in _VARIANTS.items():
        row["hits"][name] = _recall(rec["agent_items"], rec["truth_items"], fn)
    return row


# ---------------------------------------------------------------------------
# False-positive enumeration over the business token pool
# ---------------------------------------------------------------------------


# Token pool: every truth data token that occurs in the scenario, every agent
# data token observed across the 5 runs, plus the tokens explicitly listed in
# the task brief (customer/account/request/quote/approval/summary/pricing/
# sent_quote/excel_summary).
_TRUTH_TOKENS = [
    "customer",
    "pricing",
    "quote",
    "request",
    "sent_quote",
    "approval",
    "excel_summary",
]
_AGENT_TOKENS = [
    "customer",
    "information",
    "pricing",
    "quotation",
    "summary",
    "excel",
    "file",
    "request",
    "approval",
    "sent",
    "account",
    "accounting",
]
# sent_quote / excel_summary tokenize to their parts in _tokens(); keep the
# atomic forms for pair analysis.
_POOL = sorted(
    {t for t in _TRUTH_TOKENS + _AGENT_TOKENS} - {"sent_quote", "excel_summary"}
)
_POOL += ["sent_quote", "excel_summary"]  # keep them for full-item matching checks
_POOL = sorted(set(_POOL))


def _token_pair_new_matches():
    """For every truth-token x agent-token pair, record which variants match it
    *beyond* baseline. Returns dict of (truth_tok, agent_tok) -> [variant...]."""
    out = {}
    for t in _TRUTH_TOKENS:
        for a in _AGENT_TOKENS:
            if _baseline_match(t, a):
                continue  # already matched by baseline; not a *new* match
            matched = []
            if _lemma_match(t, a):
                matched.append("lemma")
            if _synset_match(t, a):
                matched.append("synset")
            if _deriv_match(t, a):
                matched.append("deriv")
            if matched:
                out[(t, a)] = matched
    return out


# Manual classification of each new (truth_token, agent_token) pair. Filled
# after inspecting WordNet senses for each pair.
_MANUAL = {
    # (truth, agent): (verdict, note)
    ("quote", "quotation"): (
        "safe",
        "same business artifact (price quotation); deriv via quote.v.02<->quotation.n.03; synset only via the citation sense quotation.n.02",
    ),
    ("sent_quote", "quotation"): (
        "safe",
        "sent_quote is a quotation (the sent copy); matched via its 'quote' component token -> quotation",
    ),
}

# Cross-domain derivational collisions that exist in WordNet but are NOT
# triggered by any scenario-real data token here. Documented to show the
# latent risk of unrestricted synset/deriv matching in a different scenario.
_LATENT_RISK = {
    (
        "account",
        "invoice",
    ): "bill.n.02::account<->account.v.02 & bill.n.02::invoice<->invoice.v.01 (bill sense)",
    (
        "account",
        "report",
    ): "explanation.n.01::account<->report.v.01::account + bill/accounting senses",
    (
        "report",
        "accounting",
    ): "account.v.02::account<->accounting.n.02/04::accounting + report senses",
    (
        "invoice",
        "accounting",
    ): "bill.n.02::invoice<->invoice.v.01 + account.v.02<->accounting senses",
}


def main():
    records = _collect_node_data()
    b_cases = _identify_b_cases(records)
    case_rows = [_analyze_case(c) for c in b_cases]

    recovery = {name: 0 for name in _VARIANTS}
    for row in case_rows:
        for name in _VARIANTS:
            if row["hits"][name] >= 1.0:
                recovery[name] += 1

    new_matches = _token_pair_new_matches()

    classified = {}
    for (t, a), vs in new_matches.items():
        verdict, note = _MANUAL.get((t, a), ("?", "unclassified"))
        classified[f"{t} <-> {a}"] = {"variants": vs, "verdict": verdict, "note": note}

    new_questionable = [
        f"{t} <-> {a}"
        for (t, a), vs in new_matches.items()
        if _MANUAL.get((t, a), ("?",))[0] == "questionable"
    ]
    new_false_positive = [
        f"{t} <-> {a}"
        for (t, a), vs in new_matches.items()
        if _MANUAL.get((t, a), ("?",))[0] == "false_positive"
    ]

    result = {
        "scenario": "quotation_workflow_1",
        "source_artifacts": [
            "run_00_seed4000",
            "run_01_seed4001",
            "run_02_seed4002",
            "run_03_seed4003",
            "run_04_seed4004",
        ],
        "nltk_available": _HAS_NLTK,
        "inflect_available": _HAS_INFLECT,
        "data_related_b_cases": len(b_cases),
        "data_related_b_from_analysis": 12,  # reads 7 + writes 5 in attribute_mismatch_analysis.json
        "recovery": recovery,
        "positive_cases": case_rows,
        "false_positive_new_matches": classified,
        "new_questionable": new_questionable,
        "new_false_positives": new_false_positive,
        "latent_cross_domain_risk": {
            f"{t} <-> {a}": note for (t, a), note in _LATENT_RISK.items()
        },
        "notes": {
            "lemma_limitation": "lemmatization only collapses INFLECTION (quotes->quote, quotations->quotation); it does NOT unify quote<->quotation because they are distinct lexemes. Hence lemma alone rescues 0 of the quote/quotation B cases.",
            "quote_quotation_same_synset": "quote & quotation share only quotation.n.02 (the 'passage/expression quoted or cited' sense) - NOT the business price-quotation sense. Same-synset would match them through an unrelated sense.",
            "quote_quotation_deriv": "genuine business derivational pair: quote.v.02 ('name the price of') <-> quotation.n.03 ('a statement of the current market price').",
        },
    }

    out = ART_DIR / "wordnet_attribute_match_analysis.json"
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise SystemExit(f"cannot write {out}: {exc}")

    print("=" * 80)
    print(f"data-related B cases: {len(b_cases)}")
    print("recovery per variant:")
    for name in _VARIANTS:
        print(f"  {name}: {recovery[name]}")
    print("\nnew false-positive candidate pairs (beyond baseline):")
    for (t, a), vs in new_matches.items():
        verdict, note = _MANUAL.get((t, a), ("?", "unclassified"))
        print(f"  {t} <-> {a}: {vs}  [{verdict}] {note}")
    print("\nnew questionable:", new_questionable or "none")
    print("new false positives:", new_false_positive or "none")
    print("\nlatent cross-domain derivational risk (NOT triggered in this scenario):")
    for k, note in _LATENT_RISK.items():
        print(f"  {k[0]} <-> {k[1]}: {note}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
