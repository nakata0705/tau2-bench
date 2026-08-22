"""Deterministic message normalization for conversation-loop detection.

This is a **runtime safeguard** utility: it exists ONLY to catch effectively
identical messages with cosmetic formatting differences (whitespace, case,
line breaks, trivial terminal punctuation). It is deliberately NOT a semantic
normalizer:

- no semantic similarity
- no synonym dictionaries
- no translation
- no stemming / embeddings / LLMs
- meaningful words are never removed

Original message text is always kept in artifacts; normalization is used only
for loop-detection counters and diagnostics.
"""

from __future__ import annotations

import re
from typing import Optional

# Trivial terminal punctuation removed by default (e.g. "Hello?" == "Hello").
_TERMINAL_PUNCT = ".!?。！？"

_WS = re.compile(r"\s+")


def normalize_text(
    text: Optional[str],
    *,
    strip_terminal_punctuation: bool = True,
) -> str:
    """Deterministic normalization of one message for equality comparison.

    Rules (in order):
    - ``None`` -> ``""``;
    - normalize line breaks (CRLF / CR -> LF);
    - casefold (lowercase + Unicode folding);
    - strip leading/trailing whitespace;
    - collapse repeated whitespace (incl. tabs/newlines) to single spaces;
    - optionally strip trivial terminal punctuation (``.!?。！？``) so that
      "What do you mean?" and "What do you mean" compare equal.

    The result is the fingerprint key for loop detection: two messages with
    the same fingerprint are treated as the same repeated interaction.
    """
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.casefold()
    text = _WS.sub(" ", text).strip()
    if strip_terminal_punctuation:
        text = text.rstrip(_TERMINAL_PUNCT)
    return text
