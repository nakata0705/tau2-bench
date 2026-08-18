# business_interview — Stakeholder-Backed Terminology: Design Note

**Date:** 2026-08-19
**Branch:** `business-interview`
**Type:** **Design only — no mechanism implemented.** This note explores
replacing/reducing the static scenario-local `EvaluationSpec.data_expressions`
synonym table with terminology established **during the interview**, and
decides whether a small deterministic design is safe enough to implement
later.

---

## 0. Problem

Semantic equivalence for reads/writes is currently achieved with
**agent-local data concepts + hidden stakeholder provenance** (see the domain
README): the Agent LLM creates local concepts and the evaluator binds them to
Truth concepts through private per-utterance claim support (`claims.py`). The
earlier static author-declared label table (`EvaluationSpec.data_expressions`)
was removed.

> **Update (private-fact-provenance refactor):** the simulator-side surface-term
> table this note targets no longer exists. Surface terms, stop phrases and
> text parsing were removed entirely; the stakeholder simulator now answers
> from hidden structured **StakeholderFacts** and returns a private
> **used_fact_ids** sidecar (`facts.py`, `user_simulator.py`). The evaluator
> binds Agent-local concepts through that private provenance only. The
> terminology-agreement questions below remain relevant as a future mechanism
> for *changing* the facts' wording during the interview, but the static
> synonym-table premise of section 0 is obsolete.

The conversation itself produces **terminology agreements** the
stakeholder explicitly confirms ("Is 'the quotation' the same thing you send
later?" → "Yes."), and those agreements are currently ignored by the
evaluator.

Desired direction:

```text
stakeholder term
  -> interviewer clarifies identity
  -> stakeholder confirms
  -> machine-readable terminology record
  -> same local business concept can be referenced consistently
```

## 1. Design questions

### Q1. Is the identity attached to a node, node+axis, or a reusable local object?

**Answer: a reusable local object (concept identity), instantiated per
node+axis slot.**

The same business object flows through several slots in the Truth DAG: the
`quote` object is `cq.writes`, `ap.reads`, `sq.reads`, `me.reads`. A
terminology record must therefore be keyed by a **concept id** (a
scenario-local object identity, e.g. `obj:quote`), not by `(node, axis)`.
Each node+axis slot then *references* the concept; the set of accepted labels
lives on the concept, not on the slot.

In today's implementation this is exactly what `data_expressions` keys on: the
canonical Truth *value* (a per-concept label), shared by all slots whose Truth
value equals it. A terminology mechanism would keep that shape and add
interview-confirmed labels to the concept's accepted set.

### Q2. What does a terminology agreement record?

```python
class TerminologyAgreement(BaseModel):
    concept_id: str            # scenario-local object identity (e.g. "quote")
    canonical_label: str       # GT author's canonical label (unchanged)
    term: str                  # the stakeholder's term (complete label)
    proposed_by: str           # "stakeholder" | "interviewer"
    confirming_observation_ids: list[str]  # the affirmative exchange(s)
    term_observation_ids: list[str]        # where the term was used
    status: Literal["confirmed"]           # only confirmed agreements count
```

The record captures **what was agreed** (term ↔ concept) and **why the
evaluator trusts it** (the exact observations where the term appeared and
where identity was confirmed). It does **not** store the semantic mapping —
the term→concept mapping is the agreement; the concept→canonical mapping
remains the GT author's declaration.

### Q3. What Observation/provenance should it carry?

Two kinds of provenance, both mandatory:

1. **Term usage observations** — the stakeholder messages in which the term
   appears verbatim (`term_observation_ids`).
2. **Confirmation observations** — the stakeholder message(s) that
   affirmatively answer the interviewer's identity question
   (`confirming_observation_ids`).

The second kind is the load-bearing one: a term the stakeholder merely used
once is not an agreement. The confirmation must be an **explicit, minimal
response to an explicit identity question** (see Q4).

**Important caveat (from the goal): an Observation id alone is NOT proof of
semantic support.** Recording an Observation id proves *that a message
exists*; it does not prove *that the message supports the claim*. The
deterministic verification in Q4 therefore verifies the **protocol** (question
asked → affirmative answer), and the **semantic mapping stays
GT-author-declared**. Interview confirmation can make the agent's *label*
acceptable; it can never declare *which concept a label means* — that is
always the scenario spec's job.

### Q4. How can confirmation be deterministically verified without an LLM judge?

Protocol-level verification, not semantics-level:

1. The interviewer must have issued a **dedicated tool call** (e.g.
   `confirm_terminology(concept_hint, term)`) or an assistant message that
   asks an explicit yes/no identity question containing the term.
2. The **immediately following stakeholder message** must match a small,
   strict confirmation pattern (case/whitespace normalized):
   `"yes"`, `"yes, ..."`, `"that's correct"`, `"that's right"`, `"correct"`,
   `"right"`, `"はい"`, `"そうです"` … (a bounded, deterministic list — no
   embeddings, no LLM judge).
3. The term must appear **verbatim** in at least one *prior* authentic
   stakeholder Observation (so the agent cannot invent a term the stakeholder
   never used).
4. The stakeholder simulator already guarantees (via its policy + `known_info`
   grounding) that it only confirms identity when the proposed term
   accurately refers to something it knows. The evaluator verifies the
   exchange happened; the simulator's policy guarantees the *truth* of the
   confirmation.

Limitations to state plainly:

- Deterministic verification proves **protocol compliance**, not semantic
  truth. It cannot detect a lying/confused stakeholder — but the stakeholder
  simulator is not adversarial (policy-grounded), so protocol compliance is
  sufficient.
- A term confirmed in conversation becomes an accepted label only for
  **visible** slots of the confirmed concept; hidden axes are untouched (Q6).

### Q5. How can one object flow through multiple nodes?

Concept-level keying (Q1): the agreement record references `concept_id`, and
every visible slot whose canonical value is that concept accepts any confirmed
term of the concept. Example: if "quotation" is confirmed for `obj:quote`,
then `cq.writes="quotation"`, `sq.reads="quotation"` (if visible), etc. all
match — without per-slot duplication.

In the quotation scenario today, the only visible slot of `quote` is
`cq.writes` (the other `quote` slots are hidden), so the mechanism's
multi-slot benefit would be visible in a scenario with multiple visible slots
of one concept.

### Q6. How are homonyms handled?

- **Same term, two concepts** (e.g. "request" the object vs "request" the
  action, or "quote" as noun vs verb): the interviewer must disambiguate by
  asking with a **contextual anchor** (which system / which flow), and the
  confirmation exchange must contain that anchor. The evaluator binds the
  agreement to exactly one `concept_id` — the term is never a global alias.
  If the stakeholder's confirmation cannot be attributed to a single concept,
  the agreement is rejected (no partial credit).
- **Same concept, two terms** ("quotation" and "quote"): both become accepted
  labels of the same concept; no conflict.
- In practice the static `data_expressions` remains the **disambiguator of
  record**: the concept id is derived from the canonical value, so homonym
  risk is the same as today (per-concept tables), and interview confirmation
  can only *add* labels to an existing concept — it can never merge concepts.

### Q7. How do we prevent hidden-Truth leakage?

The StakeholderFilter already guarantees the stakeholder **never states**
hidden facts (the simulator sees only the filtered DAG). Therefore:

1. No authentic Observation can contain a hidden concept's canonical value,
   so no confirmation exchange can reference a hidden concept — the agent
   cannot obtain a confirmed term for a hidden slot.
2. The evaluator applies terminology records **only inside the visibility
   gate** (`_attribute_ok`'s visible branch), exactly like
   `data_expressions` today. A hidden slot with an asserted value — even one
   carrying a confirmed term of the *same* concept — still scores 0.0.
3. `StakeholderFilter`, `known_info`, and the GT are unchanged; the mechanism
   adds accepted labels for visible concepts only.

### Q8. How does it generalize to lab_sample_flow?

Identically. The lab scenario's visible concepts (`sample`, `environment
chamber`, actors) can accumulate confirmed terms; its hidden derived artifacts
(`accessioned sample`, `seasoned chamber`, `conditioned sample`, `batch
approval`) are never stated by the stakeholder, so no agreement can reference
them and they remain hidden. Because the mechanism is concept-keyed and
scenario-local, it needs no per-scenario specialization.

### Q9. Do we actually need a glossary model/tool?

**Not yet — and possibly never.** Two findings:

1. The current static `data_expressions` + exact matching already covers the
   demonstrated need (`quote ↔ quotation`) with zero runtime machinery.
2. A minimal terminology mechanism needs: one tool call
   (`confirm_terminology` or a special form of an existing message), a
   `TerminologyAgreement` record, a deterministic confirmation-pattern check,
   and a merge of confirmed labels into the visible-slots matching. That is a
   **small** addition — no glossary model, no ontology, no new inference.

But the honest cost-benefit is: **the mechanism's value is proportional to how
often the agent and stakeholder actually negotiate terms**, which the current
policy already encourages. Before implementing, a measurable criterion should
exist (e.g. "confirmed labels change ≥N matched visible axes across M real
runs"). The design is safe enough to implement (deterministic, visibility-
gated, leak-proof by construction), but it is **not** needed to make the
current benchmark correct — it is a future goal, gated on evidence from real
runs.

## 2. Conclusion

- **Identity is concept-level** (reusable local object), instantiated per
  node+axis slot.
- **Agreement record**: concept id + canonical label + term + who proposed +
  confirming/usage Observation ids + status.
- **Provenance**: term-usage observations AND the affirmative confirmation
  observation; Observation ids prove the exchange happened, not the semantics
  (the semantics stay GT-author-declared).
- **Deterministic verification**: dedicated identity question + strict
  confirm-pattern on the next stakeholder message + term must appear verbatim
  in a prior authentic Observation; no LLM judge.
- **Flow across nodes**: concept-keyed accepted-label sets.
- **Homonyms**: per-concept binding with a contextual anchor; never global
  aliases.
- **Leakage**: impossible by construction (filtered stakeholder + visibility
  gate); hidden slots never use confirmed labels.
- **Lab generalization**: mechanism-agnostic; hidden derived artifacts stay
  hidden.
- **Glossary model/tool: NOT implemented.** The static `data_expressions`
  mechanism remains the evaluator's source of semantic identity; a future
  goal may add the confirm-tool + deterministic verification, gated on
  real-run evidence that negotiated terms change scoring outcomes.

**Recommendation:** keep the current static, exact, scenario-local
`data_expressions` for now; do not implement the terminology tool in this
change.
