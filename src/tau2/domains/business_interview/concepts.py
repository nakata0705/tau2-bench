"""Generic, scenario-independent primitives for business_interview (v4, open-world).

The global resolver here contains only **reusable generic operation primitives**
(e.g. CHECK / CREATE / APPROVE / SEND ...) and signal words used to guess an
abstract operation from free text. It is **not** a quotation-specific ontology
and it is **not** the agent's business-understanding mechanism — it is only a
benchmark scoring aid for the optional ``primitive`` axis. Unseen / unknown
operations are allowed (resolution returns None).

Scenario-specific domain concepts and their expressions live in the scenario's
hidden ``EvaluationSpec`` (evaluator-only), never here.
"""

from typing import Optional

# A primitive is (id, signal_words). Signals are case-insensitive substrings.
_PRIMITIVES: list[tuple[str, tuple[str, ...]]] = [
    (
        "create",
        (
            "create",
            "build",
            "generate",
            "produce",
            "make",
            "prepare",
            "作成",
            "生成",
            "作成する",
            "準備",
        ),
    ),
    (
        "check",
        ("check", "verify", "validate", "confirm", "inspect", "照会", "確認", "検証"),
    ),
    ("approve", ("approve", "authorize", "sanction", "承認", "許可", "認可")),
    ("reject", ("reject", "decline", "refuse", "却下", "拒否")),
    ("send", ("send", "deliver", "transmit", "dispatch", "送付", "送信", "発送")),
    ("receive", ("receive", "intake", "accept", "受領", "受け付", "受付")),
    ("record", ("record", "log", "enter", "register", "記録", "登録")),
    ("update", ("update", "edit", "modify", "change", "更新", "変更")),
    ("transform", ("transform", "convert", "process", "変換", "加工")),
    ("reconcile", ("reconcile", "match", "align", "照合", "突合")),
    ("notify", ("notify", "alert", "inform", "通知")),
    ("move", ("move", "transfer", "route", "移送", "移動")),
    ("review", ("review", "examine", "レビュー", "精査")),
]


def resolve_primitive(text: Optional[str]) -> Optional[str]:
    """Guess an abstract generic primitive from free text, or None if unknown.

    Returns the best-scoring primitive id (substring signal scoring) or None,
    so that an unseen operation never fails the agent.
    """
    if not text:
        return None
    t = text.lower()
    scored: list[tuple[int, str]] = []
    for pid, signals in _PRIMITIVES:
        hits = sum(1 for s in signals if s in t)
        if hits > 0:
            scored.append((hits, pid))
    if not scored:
        return None
    best = max(h for h, _ in scored)
    cands = sorted(p for h, p in scored if h == best)
    return cands[0]
