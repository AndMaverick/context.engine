"""Where the packed chunks sit in the prompt. Long-context models are reported to use the start and end of a prompt better than the
middle (Liu et al., "Lost in the Middle", 2023), so the order is a choice, not a leftover of the selection."""
from __future__ import annotations

from collections.abc import Sequence

from .model import Chunk

ORDERS = ("selection", "original", "ends", "ascending", "source")


def arrange(chunks: Sequence[Chunk], how: str, original: Sequence[str] | None = None) -> list[Chunk]:
    """Reorder ``chunks``.

    selection  the order they were chosen in: pinned first, then best first.
    original   the order the caller supplied them in (pass ``original`` as the input ids): keeps a document's own flow.
    ends       best at the start and the end, weakest in the middle.
    ascending  weakest first, best last: the best evidence sits right before the question.
    source     grouped by source, sources in order of first appearance, each group in original order.
    """
    items = list(chunks)
    if how == "selection":
        return items
    rank = {cid: i for i, cid in enumerate(original)} if original is not None else {c.id: i for i, c in enumerate(items)}
    if how == "original":
        return sorted(items, key=lambda c: rank.get(c.id, len(rank)))
    if how == "ascending":
        return sorted(items, key=lambda c: (c.score, c.id))
    if how == "ends":
        best_first = sorted(items, key=lambda c: (-c.score, c.id))
        front: list[Chunk] = []
        back: list[Chunk] = []
        for i, chunk in enumerate(best_first):
            (front if i % 2 == 0 else back).append(chunk)
        return front + back[::-1]
    if how == "source":
        by_pos = sorted(items, key=lambda c: rank.get(c.id, len(rank)))
        first: dict[str, int] = {}
        for c in by_pos:
            first.setdefault(c.source, len(first))
        return sorted(by_pos, key=lambda c: first[c.source])
    raise ValueError(f"unknown order {how!r}; use one of {', '.join(ORDERS)}")
