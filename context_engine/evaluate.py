"""Measuring a packing against the alternatives, on the same chunks and the same budget."""
from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .model import Chunk, Decision, Packed
from .pack import pack
from .similarity import Similarity
from .tokens import Counter, estimate_tokens


@dataclass(frozen=True)
class Row:
    strategy: str
    used: int
    chunks: int
    score: float  # the plain sum of scores of what was packed
    covered: int | None  # gold facts present in the packed chunks, when chunks carry ``meta["facts"]``
    gold: int | None
    packed: Packed


def topk_stop(chunks: Sequence[Chunk], budget: int, count: Counter | None = None, overhead: int = 0) -> Packed:
    """The baseline most pipelines use: sort by score, add chunks until the next one does not fit, then stop. A chunk that is a
    little too big ends the packing even if smaller, lower-scored chunks would still fit."""
    counter = count or estimate_tokens
    used = 0
    picked: list[Chunk] = []
    decisions: list[Decision] = []
    for chunk in sorted(chunks, key=lambda c: (-c.score, c.id)):
        tok = counter(chunk.text) + overhead
        if used + tok > budget:
            break
        picked.append(chunk)
        used += tok
    taken = {c.id for c in picked}
    for chunk in chunks:
        tok = counter(chunk.text) + overhead
        decisions.append(Decision(chunk, "selected" if chunk.id in taken else "dropped",
                                  "top-k" if chunk.id in taken else "after the first chunk that did not fit", tok))
    return Packed(picked, used, budget, [c for c in chunks if c.id not in taken], decisions)


def _facts(chunk: Chunk) -> set[str]:
    raw = chunk.meta.get("facts", ())
    return {str(f) for f in raw} if isinstance(raw, (list, tuple, set, frozenset)) else set()


def coverage(chunks: Iterable[Chunk], gold: Iterable[str] | None = None, universe: Iterable[Chunk] | None = None) -> tuple[int, int]:
    """``(covered, total)``: how many of the gold facts appear in ``chunks``. ``gold`` defaults to every fact that appears in any
    chunk of ``universe`` (or, without it, of ``chunks``)."""
    have: set[str] = set()
    for chunk in chunks:
        have |= _facts(chunk)
    if gold is None:
        wanted: set[str] = set()
        for chunk in (universe if universe is not None else chunks):
            wanted |= _facts(chunk)
    else:
        wanted = {str(g) for g in gold}
    return len(have & wanted), len(wanted)


def compare_strategies(
    chunks: Sequence[Chunk], budget: int, diversity: float = 0.0, *, similarity: str | Similarity = "jaccard",
    count: Counter | None = None, overhead: int = 0, gold: Iterable[str] | None = None,
) -> list[Row]:
    """Pack the same chunks with each approach. With ``diversity`` above 0 the ``mmr`` and ``best`` rows use it; the others do not."""
    items = list(chunks)
    gold_set = None if gold is None else {str(g) for g in gold}
    has_facts = any(_facts(c) for c in items)
    kw: dict[str, Any] = {"count": count, "similarity": similarity, "overhead": overhead}
    runs: list[tuple[str, Packed]] = [
        ("topk-stop", topk_stop(items, budget, count, overhead)),
        ("greedy", pack(items, budget, strategy="greedy", **kw)),
        ("density", pack(items, budget, strategy="density", **kw)),
    ]
    with contextlib.suppress(ValueError):  # too large for the exact search: the row is left out rather than faked
        runs.append(("optimal", pack(items, budget, strategy="optimal", **kw)))
    if diversity > 0:
        runs.append(("mmr", pack(items, budget, diversity, strategy="greedy", **kw)))
        runs.append(("best+mmr", pack(items, budget, diversity, strategy="best", **kw)))
    rows = []
    for name, packed in runs:
        covered, total = coverage(packed.chunks, gold_set, items) if has_facts else (None, None)
        rows.append(Row(name, packed.used, len(packed.chunks), packed.value, covered, total, packed))
    return rows
