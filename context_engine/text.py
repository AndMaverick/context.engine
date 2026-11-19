"""Cutting a chunk down to a token limit without cutting it in the middle of a thought, where that can be avoided."""
from __future__ import annotations

import re

from .tokens import Counter

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")
ELLIPSIS = "…"


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def truncate_to(text: str, limit: int, count: Counter) -> tuple[str, bool]:
    """Return ``(text, was_cut)`` where ``count(text) <= limit`` whenever that is possible at all.

    Whole sentences are kept while they fit. If not even the first sentence fits, the text is cut at the last word boundary that
    fits (or at the last character, for text with no spaces) and an ellipsis is appended. Returns ``("", True)`` when not even
    one character fits. ``count`` is called on candidate prefixes, so it need not be additive."""
    if count(text) <= limit:
        return text, False
    kept = ""
    for sentence in split_sentences(text):
        glue = "" if not kept or kept[-1] in "。！？" else " "
        candidate = kept + glue + sentence
        if count(candidate) > limit:
            break
        kept = candidate
    if kept:
        return kept, True
    lo, hi = 0, len(text)
    while lo < hi:  # largest prefix that still fits
        mid = (lo + hi + 1) // 2
        if count(text[:mid].rstrip() + ELLIPSIS) <= limit:
            lo = mid
        else:
            hi = mid - 1
    prefix = text[:lo]
    if lo < len(text) and text[lo:lo + 1].strip() and " " in prefix:  # landed mid-word: back up to a boundary
        prefix = prefix[:prefix.rfind(" ")]
    prefix = prefix.rstrip()
    return (prefix + ELLIPSIS, True) if prefix else ("", True)
