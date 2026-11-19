"""Command line: pack a JSONL file of scored chunks into a token budget and see why each one was kept or dropped.

Exit codes: 0 success, 1 a gate was hit (a required chunk was not packed, or a dropped chunk scored above the limit), 2 error."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .evaluate import compare_strategies
from .io import ChunkFormatError, read_chunks
from .order import ORDERS
from .pack import STRATEGIES, pack
from .similarity import NAMED
from .tokens import estimate_tokens


def _pack_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("file", help="JSONL of chunks ({id, text, score, source?, meta?}), or - for stdin")
    p.add_argument("--budget", type=int, required=True, help="token budget")
    p.add_argument("--diversity", type=float, default=0.0)
    p.add_argument("--similarity", choices=sorted(NAMED), default="jaccard")
    p.add_argument("--overhead", type=int, default=0, help="extra tokens each chunk costs (header, separator)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-engine", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pack", help="pack chunks into a budget")
    _pack_options(p)
    p.add_argument("--strategy", choices=STRATEGIES, default="best")
    p.add_argument("--pin", action="append", default=[], metavar="ID", help="always include this chunk (repeatable)")
    p.add_argument("--require", action="append", default=[], metavar="ID", help="exit 1 if this chunk is not packed (repeatable)")
    p.add_argument("--fail-on-drop-score", type=float, metavar="S", help="exit 1 if any dropped chunk scored at least S")
    p.add_argument("--order", choices=ORDERS, default="selection")
    p.add_argument("--min-score", type=float)
    p.add_argument("--max-per-source", type=int)
    p.add_argument("--dedupe-above", type=float, metavar="SIM")
    p.add_argument("--truncate", action="store_true", help="cut the best chunk that did not fit to fill the last tokens")
    p.add_argument("--output", choices=("explain", "text", "json", "ids"), default="explain")

    p = sub.add_parser("compare", help="pack with each strategy and compare, including a top-k baseline")
    _pack_options(p)
    p.add_argument("--gold", help="comma-separated fact names that matter, for coverage (chunks list theirs in meta.facts)")

    p = sub.add_parser("dedupe", help="list near-duplicate chunks")
    p.add_argument("file")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--similarity", choices=sorted(NAMED), default="shingle")

    p = sub.add_parser("count", help="estimate tokens in a text file (or - for stdin)")
    p.add_argument("file")
    return parser


def _cmd_pack(a: argparse.Namespace) -> int:
    chunks = read_chunks(a.file)
    packed = pack(chunks, a.budget, a.diversity, a.pin, similarity=a.similarity, strategy=a.strategy, overhead=a.overhead,
                  min_score=a.min_score, max_per_source=a.max_per_source, dedupe_above=a.dedupe_above, truncate=a.truncate, order=a.order)
    if a.output == "text":
        print(packed.text())
    elif a.output == "ids":
        print("\n".join(packed.ids))
    elif a.output == "json":
        print(json.dumps(packed.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(packed.explain())
    code = 0
    missing = [i for i in a.require if i not in packed.ids]
    for cid in missing:
        print(f"gate: required chunk {cid!r} was not packed", file=sys.stderr)
        code = 1
    if a.fail_on_drop_score is not None:
        for chunk in packed.dropped:
            if chunk.score >= a.fail_on_drop_score:
                print(f"gate: dropped {chunk.id!r} scored {chunk.score:.3f}, at or above {a.fail_on_drop_score}", file=sys.stderr)
                code = 1
    return code


def _cmd_compare(a: argparse.Namespace) -> int:
    chunks = read_chunks(a.file)
    gold = [g.strip() for g in a.gold.split(",") if g.strip()] if a.gold else None
    rows = compare_strategies(chunks, a.budget, a.diversity, similarity=a.similarity, overhead=a.overhead, gold=gold)
    has_cov = rows[0].covered is not None
    print(f"{'strategy':<12}{'tokens':>8}{'chunks':>8}{'score':>9}" + (f"{'facts':>10}" if has_cov else ""))
    for r in rows:
        line = f"{r.strategy:<12}{r.used:>8}{r.chunks:>8}{r.score:>9.3f}"
        if has_cov:
            line += f"{r.covered:>6}/{r.gold:<3}"
        print(line)
    return 0


def _cmd_dedupe(a: argparse.Namespace) -> int:
    from .similarity import resolve
    chunks = read_chunks(a.file)
    sim = resolve(a.similarity)
    ranked = sorted(chunks, key=lambda c: (-c.score, c.id))
    kept: list[str] = []
    found = 0
    for chunk in ranked:
        twin = max(((sim(chunk.text, k.text), k.id) for k in ranked if k.id in kept), default=(0.0, ""))
        if twin[0] >= a.threshold:
            print(f"{chunk.id}  duplicates {twin[1]}  (similarity {twin[0]:.2f}, score {chunk.score:.3f})")
            found += 1
        else:
            kept.append(chunk.id)
    print(f"{found} near-duplicates among {len(chunks)} chunks at similarity >= {a.threshold}")
    return 0


def _cmd_count(a: argparse.Namespace) -> int:
    if a.file == "-":
        text = sys.stdin.read()
    else:
        with open(a.file, encoding="utf-8") as handle:
            text = handle.read()
    print(estimate_tokens(text))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"pack": _cmd_pack, "compare": _cmd_compare, "dedupe": _cmd_dedupe, "count": _cmd_count}[args.command](args)
    except (ChunkFormatError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
