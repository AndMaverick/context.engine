"""A seeded synthetic retrieval corpus, for measuring packers. It is built to look like retrieval output: the best documents come
back several times as near-duplicate passages, and scores are noisy. Results on it show what diversity-aware packing does when
redundancy is present. They say nothing about your corpus: run ``context-engine compare`` on yours."""
from __future__ import annotations

import random

from .model import Chunk


def corpus(seed: int, facts: int = 40, gold: int = 8, filler: int = 80) -> tuple[list[Chunk], set[str]]:
    """Return ``(chunks, gold_facts)``. Every chunk carries ``meta["facts"]``."""
    rng = random.Random(seed)
    fill = [f"w{n}" for n in range(filler)]
    gold_facts = {f"fact{n}" for n in range(gold)}
    chunks: list[Chunk] = []

    def words_for(fact_ids: list[int], size: int) -> list[str]:
        keys = [f"k{f}x{j}" for f in fact_ids for j in range(10)]
        body = rng.sample(keys, min(len(keys), 6 * len(fact_ids))) + rng.choices(fill, k=size)
        rng.shuffle(body)
        return body

    for fact in range(facts):
        is_gold = f"fact{fact}" in gold_facts
        for doc in range(rng.randint(1, 3)):
            group = [fact] + ([rng.randrange(facts)] if rng.random() < 0.2 else [])
            base = words_for(group, rng.randint(34, 60))
            base_score = (0.75 + 0.15 * rng.random()) if is_gold else (0.30 + 0.35 * rng.random())
            copies = 1 + (rng.randint(2, 5) if is_gold else rng.randint(0, 1))
            for copy in range(copies):
                words = [w for w in base if rng.random() > 0.10]
                words = [rng.choice(fill) if rng.random() < 0.10 else w for w in words]
                chunks.append(Chunk(f"f{fact}d{doc}c{copy}", " ".join(words), round(base_score + rng.gauss(0, 0.02), 4), f"src{fact % 7}",
                                    {"facts": [f"fact{g}" for g in group]}))
    rng.shuffle(chunks)
    return chunks, gold_facts
