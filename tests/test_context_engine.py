import unittest

from context_engine import Chunk, overlap, pack, tokens


class ContextTests(unittest.TestCase):
    def test_token_estimate_scales_with_words(self):
        self.assertGreater(tokens("one two three four five six"), tokens("one two"))
        self.assertEqual(tokens(""), 1)

    def test_budget_is_never_exceeded(self):
        chunks = [Chunk(str(i), "word " * 20, score=1.0 - i / 10) for i in range(6)]
        packed = pack(chunks, budget=60)
        self.assertLessEqual(packed.used, 60)
        self.assertEqual(len(packed.chunks) + len(packed.dropped), 6)

    def test_highest_score_goes_first(self):
        chunks = [Chunk("low", "alpha beta", 0.1), Chunk("high", "gamma delta", 0.9)]
        self.assertEqual(pack(chunks, budget=100).chunks[0].id, "high")

    def test_pinned_always_included_even_if_low_score(self):
        chunks = [Chunk("policy", "legal notice text", 0.0), Chunk("a", "something else entirely", 0.9)]
        packed = pack(chunks, budget=100, pinned=("policy",))
        self.assertEqual(packed.chunks[0].id, "policy")

    def test_pinned_over_budget_raises(self):
        with self.assertRaises(ValueError):
            pack([Chunk("p", "word " * 50, 1.0)], budget=5, pinned=("p",))

    def test_diversity_prefers_novel_chunk(self):
        chunks = [
            Chunk("a", "the quarterly revenue grew twelve percent", 0.90),
            Chunk("b", "the quarterly revenue grew twelve percent yearly", 0.89),
            Chunk("c", "customer churn fell after the redesign", 0.80),
        ]
        plain = [c.id for c in pack(chunks, budget=17).chunks]
        diverse = [c.id for c in pack(chunks, budget=17, diversity=1.0).chunks]
        self.assertEqual(plain, ["a", "b"])
        self.assertEqual(diverse, ["a", "c"])

    def test_overlap_bounds(self):
        self.assertEqual(overlap("a b c", "a b c"), 1.0)
        self.assertEqual(overlap("a b", "c d"), 0.0)


if __name__ == "__main__":
    unittest.main()
