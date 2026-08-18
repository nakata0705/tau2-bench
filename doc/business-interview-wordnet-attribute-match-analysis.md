# business_interview — WordNet / Lemmatization Attribute-Match Analysis

**Date:** 2026-08-18
**Branch:** `business-interview`
**Type:** **Analysis only.** Evaluator, Agent prompt, Stakeholder prompt, Ground
Truth, tool API, and matching logic are **unchanged**. No new LLM runs. Uses only
the already-committed DeepSeek artifacts (seeds 4000–4004) and the existing
`attribute_mismatch_inventory.json` / `attribute_mismatch_analysis.json`.

Question: is there value in adding **lemmatization** or **WordNet** (same-synset /
derivationally-related-forms) to the evaluator's data-attribute matching? Measured
offline over the existing artifacts, without changing any evaluator behavior.

---

## 1. Current problem

The prior analysis (`attribute_mismatch_analysis.json`) found **22 / 50** (44%) of
attribute mismatches are `B_evaluator_too_strict`. The data-related subset is:

| source | count |
| -------- | ------- |
| reads, GT `quote` vs agent `quotation` / `quotation information` | **7** |
| writes, GT `quote` vs agent `quotation` | **5** |
| **data-related B total** | **12** |

All 12 are the same pair: Truth token `quote` vs agent token `quotation`
(or `quotation information`). The current `_data_recall` matches by **raw token
overlap** (`_tokens(a) & _tokens(b)`) **or substring**. `quote` and `quotation`
share no raw token (`_TOKEN_RE = [a-z0-9]+`), and `"quote"` is **not** a substring
of `"quotation"` (q-u-o-t-*a*-t-i-o-n). So every one of the 12 misses.

Note: the two **actor** mismatches (`"I (the stakeholder)"` → `sales`) are a
first-person/role-alias problem in `norm_role`, **not** a lexical/WordNet problem.
They are out of scope here and would not be helped by WordNet.

## 2. Matching variants tested

Deterministic, offline, no LLM / embeddings:

| variant | predicate |
| --------- | ----------- |
| **baseline** | current `_data_recall`: token overlap OR substring |
| **lemma** | tokenize, then lemmatize each token before overlap (NLTK `WordNetLemmatizer`; `inflect` agrees) |
| **synset** | a truth token & an agent token match if they share ≥1 common WordNet synset over the **union of all senses** |
| **deriv** | a match if the two tokens have a WordNet **derivationally-related-form** relation (lemma-level, all senses) |
| **hybrid** | conservative cascade: baseline(overlap/substring) → lemma → deriv (NOT unrestricted synset) |

Each variant was run as a drop-in predicate inside the evaluator's own
`_data_recall` aggregation (`any(agent item matches truth item)`), so the recovery
counts are directly comparable.

## 3. `quote` ↔ `quotation` — the specific result

The single most important empirical fact:

> **Lemmatization alone rescues 0 of the 12 cases.**

Both `WordNetLemmatizer` and `inflect` collapse only **inflection**:
`quotes → quote`, `quotations → quotation`. They do **not** unify `quote` and
`quotation`, because those are **distinct lexemes** (verb/noun `quote` vs. noun
`quotation`) — not inflectional forms of each other. Porter/Lancaster stemmers
also fail: `quote → quot`, `quotation → quotat`.

| case | baseline | lemma | synset | deriv | hybrid |
|------|:--------:|:-----:|:------:|:-----:|:------:|
| GT `quote` vs agent `quotation` (12) | miss | **miss** | hit | hit | hit |

WordNet **same-synset** and **derivationally-related-forms** both match the pair
(all 12), so `synset`/`deriv`/`hybrid` recover 12/12. The `quote ↔ quotation`
pair is **safe** as a business match either way, but the two mechanisms resolve it
through **different senses** (see §5).

## 4. Recovery counts (positive cases)

Auto-detection over the 5 artifacts found **13** recoverable cases with non-empty
agent data (12 genuine B + 1 additional run_01 send-write `sent_quote` vs
`quotation`, which the manual analysis classed `C` but is lexically recoverable).

| metric | value |
| -------- | ------- |
| data-related B (manual, reads 7 + writes 5) | 12 |
| data-related recoverable cases (auto, non-empty agent) | 13 |
| **baseline recovery** | 0 |
| **lemma recovery** | **0** |
| **synset recovery** | 13 |
| **deriv recovery** | 13 |
| **hybrid recovery** | 13 |

## 5. WordNet sense ambiguity (`quote` / `quotation`)

- `quote` synsets: `quotation_mark.n.01`, `quotation.n.02`, `quote.v.01`,
  `quote.v.02`, `quote.v.03`, `quote.v.04`.
- `quotation` synsets: `citation.n.03`, `quotation.n.02`, `quotation.n.03`,
  `quotation.n.04`.

**Same-synset** matches `quote ↔ quotation` only through **`quotation.n.02`**
("a passage or expression that is quoted or cited") — the **citation** sense,
**not** the business price-quotation sense. So the same-synset match succeeds via
an **unrelated** sense; it happens to be correct here only because the scenario
uses the same spelling.

**Derivational** relation resolves it through the **business** sense:
`quote.v.02` ("name the price of") ↔ `quotation.n.03` ("a statement of the current
market price"). This is the correct, on-topic resolution.

**Conclusion:** `quote` and `quotation` are **not** in the same synset in the
business sense; they are derivationally related *through* the business sense. So
**deriv is the semantically correct WordNet mechanism**, and same-synset works
only by luck (via the citation sense).

## 6. False-positive analysis (most important)

Enumerated every truth data token × agent data token pair that is **not** already
matched by baseline, across all scenario-real tokens plus the task's flagged
candidates (`customer`, `account`, `request`, `quote`, `approval`, `summary`,
`pricing`, `sent_quote`, `excel_summary`, `quotation`, `information`, …).

**New matches introduced beyond baseline** (over scenario-real data tokens):

| pair | variants | verdict |
|------|----------|---------|
| `quote` ↔ `quotation` | synset, deriv | **safe** (same business artifact; deriv via price sense) |
| `sent_quote` ↔ `quotation` | synset, deriv | **safe** (`sent_quote` is the sent quotation; matched via its `quote` component token) |

- **new questionable: 0**
- **new false positives over scenario-real data: 0**

The task's feared cross-domain collisions did **not** fire in this scenario's
data: `customer`↔`account`, `request`↔`quote`, `approval`↔`summary`, `pricing`↔
`quotation`, `account`↔`report`, etc. produced **no** new match (they share no
synset/deriv under the all-senses union).

**However** the all-senses approach is genuinely risky in general. A broader probe
(`account`/`invoice`/`report`/`accounting`) shows WordNet *does* derivationally
link cross-domain terms — e.g. `account ↔ invoice` via `bill.n.02::account ↔
account.v.02` and `bill.n.02::invoice ↔ invoice.v.01`; `report ↔ accounting` via
`account.v.02 ↔ accounting.n.02/04`. These are **latent** risks that would become
**false positives** if such tokens ever coexisted as distinct data artifacts in a
future scenario (e.g. a finance workflow with both `account` and `invoice`
records). They do not fire here, but the mechanism is unsound.

**Why not unrestricted synonym matching:** same-synset over the full sense union
matches `quote↔quotation` through the *citation* sense and would, in other
scenarios, match unrelated business terms (the `account`/`invoice`/`report` family)
through bill/explanation senses. It is **not** explainable ("which sense?") and
is scenario-unsafe. This confirms the all-senses-union approach should be rejected
even though it happened to be clean on this dataset.

## 7. Dependency / offline reproducibility

| option | package | runtime download | offline-reproducible | notes |
| -------- | --------- | :----------------: | :--------------------: | ------- |
| baseline (current) | stdlib | none | **yes** | zero deps |
| lemma (NLTK) | `nltk` | **yes** (`nltk.download('wordnet')`) | no* | corpus not bundled |
| lemma (`inflect`) | `inflect` (pure-Python) | none | **yes** | singular/plural only → rescues 0 anyway |
| synset / deriv (NLTK WordNet) | `nltk` | **yes** | no* | corpus not bundled |
| scenario-local alias | stdlib | none | **yes** | tiny lookup table |

\* NLTK WordNet corpus is a **runtime download** (`nltk.download('wordnet')`,
`omw-1.4`). It is not part of the wheel; a CI/offline box must fetch and pin it
(`nltk_data/corpora/wordnet`, `omw-1.4`) or vendor it. That breaks the project's
preference for **fully offline, externally-reproducible** evaluation. It also
adds `nltk` + corpus as a hard runtime dependency and ~38MB of corpus data
(`wordnet.zip` 10MB + `omw-1.4.zip` 27MB). Startup cost of loading WordNet is a
one-time ~1s; per-pair lookup is cheap but requires both corpora present.

`inflect` is pure-Python, offline, and dependency-light — but as §3 shows it only
handles inflection, so it rescues **0** of these cases. It is therefore not worth
adding for this problem either.

## 8. Recommended matcher architecture

Given the data (100% of recoverable cases are the single `quote↔quotation` family)
and the constraints (deterministic / explainable / reproducible / no large hand
dictionary / offline-first):

```
_data_recall match(item_truth, item_agent):
  1. baseline: token overlap OR substring            (unchanged)
  2. lemma:    lemmatized-token overlap              (cheap, offline, near-free)
  3. alias:    scenario-local synonym table look-up  (tiny, curated, explicit)
```

- **Do not** add WordNet.
- Prefer a **tiny scenario-local alias map** (a handful of curated pairs like
  `quote↔quotation`, `pricing↔price`) over a general WordNet resource. It is
  explicit, explainable, and cannot introduce cross-domain false positives.
- Optionally keep lemmatization (a pure-Python `inflect`-style, or a small suffix
  rule) as a low-cost generic layer — but **do not** rely on it for `quote↔
  quotation`; it does not help.

## 9. Precision-oriented recommendation

| policy | choice |
| -------- | -------- |
| **safest** | **E. lemma + small scenario-local aliases** (or even alias-only). Zero new dependency, fully offline, no WordNet sense ambiguity, no cross-domain false positives. Recovers 13/13. |
| **highest recall** | **C. derivational-relation only** (recovers 13/13 via the correct business sense) — but adds the NLTK WordNet corpus download dependency and carries latent cross-domain risk. |
| **best tradeoff** | **E. lemma + small scenario-local aliases.** Same recall as WordNet on this data, none of the dependency/offline/sense risk. |

Recommended overall: **E. lemma + small scenario-local aliases** (equivalent to
the prior report's "deterministic quote↔quotation normalization").

## 10. Should WordNet be introduced?

**No.** For the sole actual need (`quote↔quotation`), WordNet buys nothing that a
2-line scenario-local alias does not, while adding:

- a runtime corpus download (`wordnet` + `omw-1.4`) → breaks offline/CI
  reproducibility unless vendored;
- the all-senses union that matches the pair through an **irrelevant** (citation)
  sense and can match unrelated business terms in other scenarios (latent FP).

Lemmatization **alone is insufficient** (rescues 0). WordNet derivational matching
is semantically correct for this pair but overkill and non-offline. The
deterministic, explainable, reproducible fix is a **small scenario-local alias**
(option E), not WordNet.

## 11. What should be implemented next

- Add a **small data-token normalization table** to `_data_recall` (e.g.
  `quote → quotation`, `quotation → quote`) so the token-overlap check treats
  them as equivalent — the minimal change that recovers all 12–13 cases.
- Optionally, a lightweight **pure-Python lemmatizer** (no corpus) as a generic
  layer, clearly subordinate to the explicit alias map.
- Add first-person/self-reference **actor** aliases (`i`, `me`, `stakeholder`,
  `the interviewee`) to `norm_role` to fix the 10 actor `B` cases (a `norm_role`
  change, not a WordNet/lexical one).

## 12. What should NOT be changed yet

- **Do not** add WordNet (same-synset or derivational) to the evaluator.
- **Do not** switch `_data_recall` to unrestricted semantic/synset matching.
- **Do not** add embeddings / an LLM judge.
- **Do not** change `_match_nodes`, the node-matching gate, Agent/Stakeholder
  prompts, Ground Truth, or tool APIs.
- **Do not** add `nltk` / WordNet corpus as a project dependency.

## 13. Verification

- **Analysis only**: no evaluator/agent/stakeholder/GT/tool/matching changes.
- `scripts/wordnet_attribute_match_experiment.py` parses seeds 4000–4004, runs all
  5 variants through the evaluator's own `_data_recall` aggregation, and emits
  `artifacts/business_interview_real_llm/wordnet_attribute_match_analysis.json`.
- **Output is deterministic**: identical SHA-1 (`32cf8923…`) across repeated runs.
- `make check-all` (ruff lint + format) clean on the new script.
- `pytest tests/test_domains/test_business_interview/` → **67 passed** (existing
  suite unaffected; no behavior change).

## 14. Deliverables

- Experiment: `scripts/wordnet_attribute_match_experiment.py`
- Result: `artifacts/business_interview_real_llm/wordnet_attribute_match_analysis.json`
- This report: `doc/business-interview-wordnet-attribute-match-analysis.md`

Note: `nltk` + `inflect` were installed **only** into the local `.venv` to run the
analysis and are **not** declared project dependencies.
