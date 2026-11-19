"""How alike two chunks are, for penalising redundancy. Every function returns 0 (nothing shared) to 1 (the same)."""
from __future__ import annotations

import functools
import re
from collections.abc import Callable

Similarity = Callable[[str, str], float]


def overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets."""
    left, right = set(a.lower().split()), set(b.lower().split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@functools.lru_cache(maxsize=8192)
def _shingles(text: str, n: int) -> frozenset[str]:
    flat = re.sub(r"\s+", " ", text.lower()).strip()
    if len(flat) < n:
        return frozenset({flat}) if flat else frozenset()
    return frozenset(flat[i:i + n] for i in range(len(flat) - n + 1))


def shingle_similarity(a: str, b: str, n: int = 4) -> float:
    """Jaccard similarity of character n-grams. Works for text with no spaces (Chinese, Japanese) and catches near-duplicates
    that differ by inflection or a typo, where word overlap sees two different words."""
    left, right = _shingles(a, n), _shingles(b, n)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment(a: str, b: str) -> float:
    """How much of the shorter text's words appear in the other: 1 when one chunk is contained in the other, which Jaccard
    underrates (a short chunk repeated inside a long one scores low on Jaccard)."""
    left, right = set(a.lower().split()), set(b.lower().split())
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


NAMED: dict[str, Similarity] = {"jaccard": overlap, "shingle": shingle_similarity, "containment": containment}


def resolve(similarity: str | Similarity) -> Similarity:
    if callable(similarity):
        return similarity
    try:
        return NAMED[similarity]
    except KeyError:
        raise ValueError(f"unknown similarity {similarity!r}; use one of {', '.join(NAMED)} or a function") from None
