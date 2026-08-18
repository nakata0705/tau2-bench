#!/usr/bin/env python3
"""Analyze stakeholder fidelity in business_interview smoke runs.

Heuristic, deterministic keyword/co-occurrence scan over the STAKEHOLDER (user)
messages in saved smoke artifacts to classify fidelity failures vs the quotation
Ground Truth:
  - unsupported_fact       : invented a process step/branch not in GT (e.g. what
                             happens when customer info is missing / lookup fails)
  - contradicted_known_fact: denied a fact that is in the Known info
  - improper_negative_answer: "no other steps/branches" when approval or
                             month-end had not yet been disclosed
  - premature_disclosure   : volunteered approval/month-end when NOT asked about
                             conditions/exceptions/month-end/periodic
  - fabricated_unknown     : made up a rationale for the (unknown) month-end step

This is a lightweight manual-diagnostic heuristic; the final classification is
also cross-checked by reading the conversations. Not part of the test suite/CI.
"""

import glob
import json

GT_KNOWN = {
    "approval": ["approval", "approve", "credit risk"],
    "month_end": ["month-end", "monthly", "month end", "month_end"],
    "unknown_month_end_reason": ["don't know why", "not certain", "no documentation"],
}


def _user_messages(path):
    d = json.load(open(path))
    return [
        m["content"] for m in d["conversation"] if m["role"] == "user" and m["content"]
    ]


def _agent_questions(path):
    d = json.load(open(path))
    return [
        m["content"]
        for m in d["conversation"]
        if m["role"] == "assistant" and m["content"] and not m["tool_calls"]
    ]


def _mentions(text, needles):
    low = text.lower()
    return any(n in low for n in needles)


def classify(path):
    users = _user_messages(path)
    agent_qs = _agent_questions(path)
    all_agent = " ".join(agent_qs).lower()

    # unsupported_fact: stakeholder invents a missing-info / error branch.
    unsupported = any(
        _mentions(
            u,
            [
                "isn't found",
                "is not found",
                "not found in the crm",
                "missing customer",
                "look into it",
                "ask the customer for the missing",
                "get the correct details",
            ],
        )
        for u in users
    )

    # approval disclosure: was the approval branch ever mentioned by stakeholder?
    approval_mentioned = any(
        _mentions(u, ["1,000,000", "manager", "approval", "approve"]) for u in users
    )
    # month-end disclosure
    month_end_mentioned = any(
        _mentions(u, ["month-end", "month end", "monthly"]) for u in users
    )

    # improper_negative_answer: "no other steps/branches/exceptions" appears
    # BEFORE approval+month-end have both been disclosed.
    improper_negative = False
    approval_seen = month_seen = False
    for u in users:
        if _mentions(u, ["1,000,000", "approval", "approve"]):
            approval_seen = True
        if _mentions(u, ["month-end", "month end", "monthly"]):
            month_seen = True
        if _mentions(
            u,
            [
                "no other",
                "no additional",
                "none that",
                "that's all",
                "nothing else",
                "only the normal",
                "no branches",
                "no other steps",
                "no special cases",
                "no other branches",
            ],
        ):
            if not (approval_seen and month_seen):
                # negative before both known conditional/periodic facts disclosed
                improper_negative = True

    # premature_disclosure: month-end mentioned but the agent never asked about
    # month-end / periodic / recurring.
    asked_month_end = _mentions(
        all_agent,
        ["month-end", "month end", "monthly", "periodic", "recurring", "month-end"],
    )
    premature_month = month_end_mentioned and not asked_month_end
    # approval mentioned before agent asked about conditions/exceptions/approval
    asked_approval = _mentions(
        all_agent,
        [
            "exception",
            "condition",
            "special",
            "branch",
            "approval",
            "approve",
            "threshold",
            "amount over",
            "1,000,000",
            "variation",
        ],
    )
    premature_approval = approval_mentioned and not asked_approval

    # fabricated_unknown: stakeholder asserted a reason for the month-end step.
    fabricated_unknown = any(
        u.strip()
        .lower()
        .startswith(("because", "the reason", "it's because", "it is because"))
        and _mentions(
            u, ["month", "accounting", "reconciliation", "track", "record", "for audit"]
        )
        and not _mentions(u, ["don't know", "not sure", "not certain"])
        for u in users
    )

    codes = []
    if unsupported:
        codes.append("unsupported_fact")
    if improper_negative:
        codes.append("improper_negative_answer")
    if premature_month or premature_approval:
        codes.append("premature_disclosure")
    if fabricated_unknown:
        codes.append("fabricated_unknown")
    if not codes:
        codes.append("none")

    return {
        "approval_mentioned": approval_mentioned,
        "month_end_mentioned": month_end_mentioned,
        "unknown_month_end_reason_handled": any(
            _mentions(
                u,
                [
                    "don't know why",
                    "not certain",
                    "no documentation",
                    "vague impression",
                ],
            )
            for u in users
        ),
        "failures": codes,
    }


def main():
    results = {}
    for p in sorted(
        glob.glob("artifacts/business_interview_real_llm/run_*_seed3*.json")
    ):
        name = p.split("/")[-1]
        results[name] = classify(p)
        print(f"{name:26}", json.dumps(results[name]))

    out = (
        "artifacts/business_interview_real_llm/stakeholder_fidelity_classification.json"
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
