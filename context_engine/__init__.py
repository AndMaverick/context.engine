"""context.engine: pack the best, least redundant context into a fixed token budget."""

from __future__ import annotations

from dataclasses import dataclass

__version__ = "0.1.0"


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    score: float
    source: str = ""


def tokens(text: str) -> int:
    """Cheap token estimate: about 1.33 tokens per word. Swap in a real tokenizer when you have one."""
    return max(1, round(len(text.split()) * 1.33))


def overlap(a: str, b: str) -> float:
    """Jaccard similarity of word sets."""
    left, right = set(a.lower().split()), set(b.lower().split())
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


@dataclass
class Packed:
    chunks: list
    used: int
    budget: int
    dropped: list

    def text(self, separator: str = "\n\n") -> str:
        return separator.join(c.text for c in self.chunks)


def pack(chunks, budget: int, diversity: float = 0.0, pinned=()) -> Packed:
    """Greedy selection by score, penalized by redundancy with what is already chosen.

    pinned chunk ids are always included first. If they alone exceed the budget,
    that is a caller error and ValueError is raised.
    """
    by_id = {c.id: c for c in chunks}
    selected: list = []
    used = 0
    for chunk_id in pinned:
        chunk = by_id[chunk_id]
        used += tokens(chunk.text)
        selected.append(chunk)
    if used > budget:
        raise ValueError(f"pinned chunks need {used} tokens, budget is {budget}")

    remaining = [c for c in chunks if c.id not in set(pinned)]
    while remaining:
        def adjusted(chunk):
            penalty = max((overlap(chunk.text, s.text) for s in selected), default=0.0)
            return chunk.score - diversity * penalty

        best = max(remaining, key=lambda c: (adjusted(c), c.score, c.id))
        remaining.remove(best)
        cost = tokens(best.text)
        if used + cost <= budget:
            selected.append(best)
            used += cost
    chosen = {c.id for c in selected}
    dropped = [c for c in chunks if c.id not in chosen]
    return Packed(selected, used, budget, dropped)
