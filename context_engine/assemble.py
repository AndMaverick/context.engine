"""Building the whole prompt inside a model's context window: system text, packed context, the question, and room left for the answer."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .model import Chunk, Packed
from .pack import pack
from .tokens import Counter, estimate_tokens


class BudgetError(ValueError):
    """The parts that must be in the prompt do not leave room for any context."""


@dataclass
class Assembled:
    prompt: str
    context: str
    tokens: int  # the real count of ``prompt``
    window: int
    reserve_output: int
    breakdown: dict[str, int]
    packed: Packed
    citations: dict[int, str] = field(default_factory=dict)  # position in the context -> chunk id

    @property
    def headroom(self) -> int:
        return self.window - self.reserve_output - self.tokens


def assemble(
    chunks: Sequence[Chunk], *, window: int, reserve_output: int = 0, system: str = "", question: str = "",
    chunk_format: str = "[{n}] {text}", separator: str = "\n\n", count: Counter | None = None, **pack_options: Any,
) -> Assembled:
    """Pack ``chunks`` into what is left of ``window`` after the system text, the question, the reserved answer space and the
    per-chunk formatting, then check the finished prompt with the real counter and repack tighter if it is over.

    ``chunk_format`` may use ``{n}`` (position, from 1), ``{id}``, ``{source}`` and ``{text}``. Extra keyword arguments go to
    ``pack`` (``diversity``, ``pinned``, ``strategy``, ``order`` and the rest). ``Assembled.citations`` maps each ``n`` to a chunk id."""
    counter = count or estimate_tokens
    sections = {"system": counter(system) if system else 0, "question": counter(question) if question else 0}
    fixed = sections["system"] + sections["question"]
    available = window - reserve_output - fixed
    if available <= 0:
        raise BudgetError(f"window {window} minus {reserve_output} for the answer minus {fixed} for system and question leaves nothing for context")
    longest = max((c.id for c in chunks), key=len, default="")
    sample = chunk_format.format(n=len(chunks) or 1, id=longest, source="", text="")
    overhead = counter(sample) + counter(separator) if sample or separator else 0
    margin = 0
    for _ in range(8):
        packed = pack(chunks, max(0, available - margin), count=count, overhead=overhead, **pack_options)
        body = separator.join(chunk_format.format(n=n, id=c.id, source=c.source, text=c.text) for n, c in enumerate(packed.chunks, 1))
        parts = [p for p in (system, body, question) if p]
        prompt = separator.join(parts)
        total = counter(prompt)
        if total <= window - reserve_output:
            break
        margin += total - (window - reserve_output) + 1
    else:
        raise BudgetError("could not fit the prompt after repacking; the counter may be too irregular to budget against")
    sections["context"] = counter(body) if body else 0
    sections["reserved for answer"] = reserve_output
    return Assembled(prompt, body, total, window, reserve_output, sections, packed,
                     {n: c.id for n, c in enumerate(packed.chunks, 1)})
