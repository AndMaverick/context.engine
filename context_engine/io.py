"""Reading chunks from JSON Lines, with errors that name the line."""
from __future__ import annotations

import json
import sys
from collections.abc import Iterable

from .model import Chunk


class ChunkFormatError(ValueError):
    pass


def parse_chunks(lines: Iterable[str], origin: str = "input") -> list[Chunk]:
    """Each non-blank line is a JSON object with ``id`` (string), ``text`` (string) and ``score`` (number); ``source`` and
    ``meta`` are optional."""
    out: list[Chunk] = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        where = f"{origin} line {number}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ChunkFormatError(f"{where}: not valid JSON ({exc.msg})") from None
        if not isinstance(obj, dict):
            raise ChunkFormatError(f"{where}: expected a JSON object, got {type(obj).__name__}")
        for key, kind in (("id", str), ("text", str)):
            if not isinstance(obj.get(key), kind):
                raise ChunkFormatError(f"{where}: {key!r} must be a string")
        score = obj.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ChunkFormatError(f"{where}: 'score' must be a number")
        meta = obj.get("meta", {})
        if not isinstance(meta, dict):
            raise ChunkFormatError(f"{where}: 'meta' must be an object")
        out.append(Chunk(obj["id"], obj["text"], float(score), str(obj.get("source", "")), meta))
    return out


def read_chunks(path: str) -> list[Chunk]:
    if path == "-":
        return parse_chunks(sys.stdin, "stdin")
    with open(path, encoding="utf-8") as handle:
        return parse_chunks(handle, path)
