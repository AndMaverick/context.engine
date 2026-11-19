"""Reproduce the README numbers: python examples/benchmark.py

200 seeded synthetic retrieval corpora (about 160 chunks each, 8 of 40 facts are the ones the question needs, the best documents come
back as several near-duplicate passages). Budget is one fifth of the corpus. For each packer: mean gold facts covered, and the paired
difference from the top-k baseline with a 95% interval from a bootstrap over seeds."""
from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_engine import coverage, estimate_tokens, pack, topk_stop  # noqa: E402
from context_engine.synthetic import corpus  # noqa: E402

SEEDS = 200
PACKERS = {
    "top-k, stop at first overflow": lambda ch, b: topk_stop(ch, b),
    "greedy by score (0.1 behaviour)": lambda ch, b: pack(ch, b),
    "best (greedy / density / exact)": lambda ch, b: pack(ch, b, strategy="best"),
    "greedy + diversity 0.9 (jaccard)": lambda ch, b: pack(ch, b, 0.9),
    "greedy + diversity 0.9 (shingle)": lambda ch, b: pack(ch, b, 0.9, similarity="shingle"),
    "best + diversity 0.9 (shingle)": lambda ch, b: pack(ch, b, 0.9, similarity="shingle", strategy="best"),
    "diversity 0.9 + dedupe at 0.8": lambda ch, b: pack(ch, b, 0.9, similarity="shingle", dedupe_above=0.8),
}


def main() -> None:
    results: dict[str, list[int]] = {name: [] for name in PACKERS}
    scores: dict[str, list[float]] = {name: [] for name in PACKERS}
    started = time.time()
    for seed in range(SEEDS):
        chunks, gold = corpus(seed)
        budget = sum(estimate_tokens(c.text) for c in chunks) // 5
        for name, packer in PACKERS.items():
            packed = packer(chunks, budget)
            assert packed.used <= budget
            results[name].append(coverage(packed.chunks, gold, chunks)[0])
            scores[name].append(packed.value)
    base = results["top-k, stop at first overflow"]
    rng = random.Random(0)
    print(f"{SEEDS} corpora, gold facts per corpus: 8, ran in {time.time() - started:.1f}s\n")
    print(f"{'packer':<36}{'facts/8':>9}{'vs top-k':>10}{'95% interval':>18}{'sum score':>11}")
    for name, got in results.items():
        diffs = [g - b for g, b in zip(got, base, strict=True)]
        boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        print(f"{name:<36}{statistics.fmean(got):>9.2f}{statistics.fmean(diffs):>+10.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}{statistics.fmean(scores[name]):>11.2f}")


if __name__ == "__main__":
    main()
