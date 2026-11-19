"""Choosing which chunks go into the budget, and recording why every other one did not."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from .model import Chunk, Decision, Packed
from .order import arrange
from .similarity import Similarity, resolve
from .text import truncate_to
from .tokens import Counter, estimate_tokens

STRATEGIES = ("greedy", "density", "optimal", "best")
KNAPSACK_CELL_LIMIT = 3_000_000  # items x (budget + 1); beyond this the exact search is skipped and ``best`` says so


def set_value(chunks: Iterable[Chunk], diversity: float, similarity: Similarity) -> float:
    """What a set of chunks is worth: the sum of scores, minus ``diversity`` times each chunk's similarity to a higher-scored
    chunk in the set. Order-independent, so strategies that pick in different orders can be compared on one number."""
    ranked = sorted(chunks, key=lambda c: (-c.score, c.id))
    total = 0.0
    for i, chunk in enumerate(ranked):
        penalty = max((similarity(chunk.text, other.text) for other in ranked[:i]), default=0.0)
        total += chunk.score - diversity * penalty
    return total


def pack(
    chunks: Sequence[Chunk],
    budget: int,
    diversity: float = 0.0,
    pinned: Sequence[str] = (),
    *,
    count: Counter | None = None,
    similarity: str | Similarity = "jaccard",
    strategy: str = "greedy",
    overhead: int = 0,
    min_score: float | None = None,
    max_per_source: int | None = None,
    source_tokens: int | None = None,
    dedupe_above: float | None = None,
    truncate: bool = False,
    min_fragment: int = 24,
    order: str = "selection",
) -> Packed:
    """Choose chunks for a token ``budget``.

    pinned          ids that are always included, first. Over budget is a caller error (ValueError). They skip every filter below.
    diversity       how hard to push away from redundancy: each chunk's score loses ``diversity`` times its similarity to what
                    is already chosen.
    count           token counter, ``str -> int``. Defaults to a tokenizer-free estimate; pass your model's tokenizer.
    similarity      "jaccard", "shingle", "containment", or a function ``(a, b) -> 0..1``.
    strategy        greedy: best adjusted score first (the 0.1 behaviour). density: best score per token first. optimal: the exact
                    best total score (knapsack), which needs diversity 0 and no quotas or dedupe. best: run greedy, density and,
                    when it applies and is small enough, optimal, and keep the set worth the most.
    overhead        tokens each chunk costs beyond its text (a citation header, a separator).
    min_score       chunks scoring below this are dropped.
    max_per_source  most chunks one source may contribute. source_tokens: most tokens one source may use.
    dedupe_above    a chunk at least this similar to one already chosen is dropped outright (a hard rule; diversity is soft).
    truncate        if room is left, cut the best chunk that did not fit down to fit (at a sentence boundary where possible).
                    Its score is scaled by the share of its tokens kept, and ``meta["truncated"]`` records the cut.
    min_fragment    the smallest truncated fragment worth including, in tokens.
    order           where chunks sit in the result: see ``context_engine.order``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; use one of {', '.join(STRATEGIES)}")
    if budget < 0:
        raise ValueError(f"budget must not be negative, got {budget}")
    if diversity < 0:
        raise ValueError(f"diversity must not be negative, got {diversity}")
    counter = count or estimate_tokens
    sim = resolve(similarity)
    items = list(chunks)
    ids = [c.id for c in items]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"chunk ids must be unique; repeated: {', '.join(dup)}")
    by_id = {c.id: c for c in items}
    cost = {c.id: counter(c.text) + overhead for c in items}

    selected: list[Chunk] = []
    why: dict[str, Decision] = {}
    used = 0
    for chunk_id in pinned:
        if chunk_id not in by_id:
            raise KeyError(f"pinned id {chunk_id!r} is not among the chunks")
        if any(s.id == chunk_id for s in selected):
            continue
        chunk = by_id[chunk_id]
        selected.append(chunk)
        used += cost[chunk_id]
        why[chunk_id] = Decision(chunk, "pinned", "pinned by the caller", cost[chunk_id])
    if used > budget:
        raise ValueError(f"pinned chunks need {used} tokens, budget is {budget}")

    pinned_ids = {c.id for c in selected}
    pool: list[Chunk] = []
    for chunk in items:
        if chunk.id in pinned_ids:
            continue
        if min_score is not None and chunk.score < min_score:
            why[chunk.id] = Decision(chunk, "dropped", f"score {chunk.score:.3f} is below min_score {min_score:.3f}", cost[chunk.id])
        elif cost[chunk.id] > budget:
            why[chunk.id] = Decision(chunk, "dropped", f"needs {cost[chunk.id]} tokens, more than the whole budget of {budget}", cost[chunk.id])
        else:
            pool.append(chunk)

    notes: list[str] = []
    constrained = bool(max_per_source is not None or source_tokens is not None or dedupe_above is not None)

    def run(kind: str) -> tuple[list[Chunk], dict[str, Decision]]:
        return _greedy(kind, pool, selected, used, budget, diversity, sim, cost, max_per_source, source_tokens, dedupe_above)

    if strategy in ("greedy", "density"):
        picked, reasons = run(strategy)
    elif strategy == "optimal":
        if diversity > 0 or constrained:
            raise ValueError("strategy='optimal' maximises total score, so it needs diversity=0 and no quotas or dedupe; use 'best'")
        found = _optimal(pool, budget - used, cost)
        if found is None:
            raise ValueError(f"too large for the exact search ({len(pool)} chunks x {budget - used} tokens); use 'best' or 'greedy'")
        picked, reasons = found
    else:  # best
        candidates = [run("greedy"), run("density")]
        if diversity == 0 and not constrained:
            exact = _optimal(pool, budget - used, cost)
            if exact is None:
                notes.append(f"exact search skipped: {len(pool)} chunks x {budget - used} tokens is over the {KNAPSACK_CELL_LIMIT:,} cell limit")
            else:
                candidates.append(exact)
        value = [set_value(selected + p, diversity, sim) for p, _ in candidates]
        picked, reasons = candidates[value.index(max(value))]  # ties go to the earlier (greedy) result

    final = list(selected) + picked
    final_used = used + sum(cost[c.id] for c in picked)
    for chunk in picked:
        why[chunk.id] = _pick_decision(chunk, diversity, sim, final, cost[chunk.id])
    why.update(reasons)

    if truncate and budget - final_used >= max(min_fragment, 1) + overhead:
        leftovers = sorted((d for d in why.values() if d.status == "dropped" and d.reason.startswith(("no room", "needs"))),
                           key=lambda d: (-d.chunk.score, d.chunk.id))
        if leftovers:
            target = leftovers[0].chunk
            room = budget - final_used - overhead
            text, cut = truncate_to(target.text, room, counter)
            if text and cut and counter(text) >= min_fragment:
                kept = counter(text)
                meta = {**target.meta, "truncated": True, "original_tokens": cost[target.id] - overhead, "kept_tokens": kept}
                piece = Chunk(target.id, text, target.score * kept / max(1, cost[target.id] - overhead), target.source, meta)
                final.append(piece)
                final_used += kept + overhead
                why[target.id] = Decision(piece, "truncated", f"cut from {cost[target.id] - overhead} to {kept} tokens to fill the last {room}", kept + overhead)

    final = arrange(final, order, ids)
    chosen = {c.id for c in final}
    dropped = [c for c in items if c.id not in chosen]
    decisions = [why[c.id] for c in items]
    return Packed(final, final_used, budget, dropped, decisions, notes)


def _pick_decision(chunk: Chunk, diversity: float, sim: Similarity, chosen: Sequence[Chunk], tokens: int) -> Decision:
    above = [c for c in chosen if c.id != chunk.id and (-c.score, c.id) < (-chunk.score, chunk.id)]
    penalty = max((sim(chunk.text, c.text) for c in above), default=0.0)
    adjusted = chunk.score - diversity * penalty if diversity else None
    return Decision(chunk, "selected", "fits and ranked above what was left", tokens, adjusted)


def _greedy(
    kind: str, pool: Sequence[Chunk], pinned: Sequence[Chunk], used: int, budget: int, diversity: float, sim: Similarity,
    cost: dict[str, int], max_per_source: int | None, source_tokens: int | None, dedupe_above: float | None,
) -> tuple[list[Chunk], dict[str, Decision]]:
    picked: list[Chunk] = []
    reasons: dict[str, Decision] = {}
    per_count: dict[str, int] = {}
    per_tokens: dict[str, int] = {}
    for c in pinned:
        per_count[c.source] = per_count.get(c.source, 0) + 1
        per_tokens[c.source] = per_tokens.get(c.source, 0) + cost[c.id]
    # each candidate's similarity to its nearest chosen chunk, kept up to date as chunks are chosen (one similarity call per pair)
    near: dict[str, tuple[float, Chunk | None]] = {}
    for chunk in pool:
        best_sim, best_with = 0.0, None
        for s in pinned:
            value = sim(chunk.text, s.text)
            if best_with is None or value > best_sim:
                best_sim, best_with = value, s
        near[chunk.id] = (best_sim, best_with)

    remaining = list(pool)
    while remaining:
        best: Chunk | None = None
        best_key: tuple[float, float, str] | None = None
        for chunk in list(remaining):
            tok = cost[chunk.id]
            nearest, twin = near[chunk.id]
            if used + tok > budget:
                cost_of_overlap = diversity * nearest
                note = f"; ranked down {cost_of_overlap:.2f} for similarity {nearest:.2f} to {twin.id}" if twin is not None and cost_of_overlap >= 0.1 else ""
                reasons[chunk.id] = Decision(chunk, "dropped", f"no room: needs {tok} tokens, {budget - used} left{note}", tok)
            elif dedupe_above is not None and twin is not None and nearest >= dedupe_above:
                reasons[chunk.id] = Decision(chunk, "dropped", f"near-duplicate of {twin.id} (similarity {nearest:.2f})", tok)
            elif max_per_source is not None and per_count.get(chunk.source, 0) >= max_per_source:
                reasons[chunk.id] = Decision(chunk, "dropped", f"source {chunk.source!r} already has {max_per_source} chunks", tok)
            elif source_tokens is not None and per_tokens.get(chunk.source, 0) + tok > source_tokens:
                reasons[chunk.id] = Decision(chunk, "dropped", f"source {chunk.source!r} would pass its {source_tokens} token share", tok)
            else:
                adj = chunk.score - diversity * nearest
                key = (adj / tok if kind == "density" and tok > 0 else adj, chunk.score, chunk.id)
                if best_key is None or key > best_key:
                    best, best_key = chunk, key
                continue
            remaining.remove(chunk)
        if best is None:
            break
        remaining.remove(best)
        picked.append(best)
        used += cost[best.id]
        per_count[best.source] = per_count.get(best.source, 0) + 1
        per_tokens[best.source] = per_tokens.get(best.source, 0) + cost[best.id]
        for chunk in remaining:
            value = sim(chunk.text, best.text)
            if value > near[chunk.id][0] or near[chunk.id][1] is None:
                near[chunk.id] = (value, best)
    return picked, reasons


def _optimal(pool: Sequence[Chunk], capacity: int, cost: dict[str, int]) -> tuple[list[Chunk], dict[str, Decision]] | None:
    """Exact 0/1 knapsack on total score. Returns None when it is too large to run in reasonable time."""
    if capacity < 0:
        return [], {}
    items = [c for c in pool if c.score > 0]
    if len(items) * (capacity + 1) > KNAPSACK_CELL_LIMIT:
        return None
    best = [0.0] * (capacity + 1)
    took: list[bytearray] = []
    for chunk in items:
        w_cost, value = cost[chunk.id], chunk.score
        row = bytearray(capacity + 1)
        if w_cost <= capacity:
            nxt = best[:]
            for w in range(w_cost, capacity + 1):
                cand = best[w - w_cost] + value
                if cand > nxt[w] + 1e-12:
                    nxt[w] = cand
                    row[w] = 1
            best = nxt
        took.append(row)
    w = capacity
    picked_idx: list[int] = []
    for i in range(len(items) - 1, -1, -1):
        if took[i][w]:
            picked_idx.append(i)
            w -= cost[items[i].id]
    picked = sorted((items[i] for i in picked_idx), key=lambda c: (-c.score, c.id))
    taken = {c.id for c in picked}
    reasons = {}
    for chunk in pool:
        if chunk.id not in taken:
            tok = cost[chunk.id]
            why = ("score is not positive" if chunk.score <= 0 else
                   f"no room: needs {tok} tokens and the exact search found a set worth more without it")
            reasons[chunk.id] = Decision(chunk, "dropped", why, tok)
    return picked, reasons
