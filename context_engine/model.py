"""The records: a chunk going in, a packed context coming out, and the reason for every decision in between."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    score: float
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)  # offsets, headings, anything you want to keep

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "score": self.score, "source": self.source, "meta": self.meta}


@dataclass(frozen=True)
class Decision:
    chunk: Chunk
    status: str  # pinned, selected, or dropped
    reason: str
    tokens: int
    adjusted: float | None = None  # the score after the redundancy penalty, when one applied


@dataclass
class Packed:
    chunks: list[Chunk]
    used: int
    budget: int
    dropped: list[Chunk]
    decisions: list[Decision] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # things the caller should know, e.g. a strategy that fell back

    def text(self, separator: str = "\n\n") -> str:
        return separator.join(c.text for c in self.chunks)

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.chunks]

    @property
    def value(self) -> float:
        """The total score of what was packed."""
        return sum(c.score for c in self.chunks)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def explain(self) -> str:
        """One line per chunk: what happened to it and why. The first thing to read when the answer is missing a fact."""
        lines = [f"{self.used} of {self.budget} tokens used, {len(self.chunks)} chunks packed, {len(self.dropped)} dropped"]
        for d in self.decisions:
            adj = f" (adjusted {d.adjusted:.3f})" if d.adjusted is not None else ""
            lines.append(f"  {d.status:<9}{d.chunk.id:<16}{d.tokens:>5} tok  score {d.chunk.score:.3f}{adj}  {d.reason}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"used": self.used, "budget": self.budget, "chunks": [c.to_dict() for c in self.chunks], "dropped": [c.id for c in self.dropped],
                "decisions": [{"id": d.chunk.id, "status": d.status, "reason": d.reason, "tokens": d.tokens} for d in self.decisions],
                "notes": self.notes}
