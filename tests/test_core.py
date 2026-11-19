import itertools
import random
import unittest

from context_engine import (
    Chunk,
    arrange,
    containment,
    estimate_tokens,
    overlap,
    pack,
    set_value,
    shingle_similarity,
    tokens,
    truncate_to,
)
from context_engine.similarity import resolve


def sized(cid, size, score, source="", **meta):
    """A chunk whose cost is exactly ``size`` under ``len`` as the counter."""
    return Chunk(cid, "x" * size, score, source, meta)


class TokenTests(unittest.TestCase):
    def test_latin_matches_the_old_estimate(self):
        self.assertEqual(tokens("one two three"), 4)
        self.assertEqual(tokens(""), 1)

    def test_cjk_counts_per_character(self):
        self.assertEqual(estimate_tokens("日本語のテキスト"), 8)

    def test_non_latin_words_cost_more_than_latin(self):
        self.assertGreater(estimate_tokens("привет мир привет мир"), estimate_tokens("hello world hello world"))

    def test_mixed_text(self):
        self.assertGreater(estimate_tokens("hello 世界 world"), estimate_tokens("hello world"))


class SimilarityTests(unittest.TestCase):
    def test_bounds_and_identity(self):
        for fn in (overlap, shingle_similarity, containment):
            self.assertEqual(fn("alpha beta gamma", "alpha beta gamma"), 1.0)
            self.assertEqual(fn("alpha", ""), 0.0)
            self.assertEqual(fn("aaaa bbbb", "cccc dddd"), 0.0)

    def test_shingles_see_inflection_that_words_miss(self):
        a, b = "the revenues increased sharply", "the revenue increases sharply"
        self.assertGreater(shingle_similarity(a, b), overlap(a, b))

    def test_shingles_work_without_spaces(self):
        self.assertGreater(shingle_similarity("今年の売上は大きく伸びた", "今年の売上は大きく伸びました"), 0.6)
        self.assertEqual(overlap("今年の売上", "今年の売上は"), 0.0)

    def test_containment_sees_a_short_chunk_inside_a_long_one(self):
        short, long = "profit grew", "profit grew by twelve percent in the third quarter of the year"
        self.assertEqual(containment(short, long), 1.0)
        self.assertLess(overlap(short, long), 0.3)

    def test_resolve(self):
        self.assertIs(resolve("jaccard"), overlap)
        self.assertEqual(resolve(lambda a, b: 0.5)("a", "b"), 0.5)
        with self.assertRaises(ValueError):
            resolve("cosine")


class PackBasicsTests(unittest.TestCase):
    def test_budget_never_exceeded_on_random_inputs(self):
        rng = random.Random(7)
        for trial in range(200):
            chunks = [sized(f"c{i}", rng.randint(1, 30), rng.random(), f"s{i % 3}") for i in range(rng.randint(0, 12))]
            budget, overhead = rng.randint(0, 80), rng.randint(0, 3)
            for strategy in ("greedy", "density", "best"):
                p = pack(chunks, budget, rng.random(), count=len, strategy=strategy, overhead=overhead)
                self.assertLessEqual(p.used, budget, (trial, strategy))
                self.assertEqual(p.used, sum(len(c.text) + overhead for c in p.chunks))

    def test_every_chunk_has_exactly_one_decision(self):
        chunks = [sized(f"c{i}", 5 + i, 1 - i / 10) for i in range(8)]
        p = pack(chunks, 30, count=len)
        self.assertEqual([d.chunk.id for d in p.decisions], [c.id for c in chunks])
        self.assertEqual({d.chunk.id for d in p.decisions if d.status == "dropped"}, {c.id for c in p.dropped})
        for d in p.decisions:
            self.assertTrue(d.reason)

    def test_unknown_pinned_id_raises(self):
        with self.assertRaises(KeyError):
            pack([Chunk("a", "x", 1.0)], 10, pinned=("nope",))

    def test_duplicate_ids_are_refused(self):
        with self.assertRaises(ValueError):
            pack([Chunk("a", "x", 1.0), Chunk("a", "y", 0.5)], 10)

    def test_bad_arguments(self):
        for kwargs in ({"budget": -1}, {"budget": 5, "diversity": -0.1}, {"budget": 5, "strategy": "magic"}):
            with self.assertRaises(ValueError):
                pack([Chunk("a", "x", 1.0)], **kwargs)

    def test_custom_counter_is_used(self):
        p = pack([sized("a", 10, 1.0)], 10, count=len)
        self.assertEqual(p.used, 10)
        self.assertEqual(pack([sized("a", 10, 1.0)], 9, count=len).chunks, [])

    def test_overhead_is_charged_per_chunk(self):
        chunks = [sized("a", 4, 1.0), sized("b", 4, 0.9)]
        self.assertEqual(len(pack(chunks, 8, count=len).chunks), 2)
        self.assertEqual(len(pack(chunks, 8, count=len, overhead=1).chunks), 1)

    def test_pinned_skip_filters_but_count_toward_the_budget(self):
        chunks = [sized("keep", 4, 0.0, "a"), sized("b", 4, 0.9, "a")]
        p = pack(chunks, 10, count=len, pinned=("keep",), min_score=0.5, max_per_source=1)
        self.assertEqual(p.ids, ["keep"])
        self.assertEqual(p.used, 4)

    def test_chunk_bigger_than_the_whole_budget_says_so(self):
        p = pack([sized("big", 50, 1.0), sized("ok", 5, 0.1)], 10, count=len)
        self.assertEqual(p.ids, ["ok"])
        self.assertIn("more than the whole budget", p.decisions[0].reason)

    def test_min_score(self):
        p = pack([sized("a", 2, 0.9), sized("b", 2, 0.2)], 50, count=len, min_score=0.5)
        self.assertEqual(p.ids, ["a"])
        self.assertIn("below min_score", p.decisions[1].reason)

    def test_empty_input(self):
        p = pack([], 10)
        self.assertEqual((p.chunks, p.used, p.dropped), ([], 0, []))


class ConstraintTests(unittest.TestCase):
    def test_max_per_source(self):
        chunks = [sized("a1", 2, 0.9, "A"), sized("a2", 2, 0.8, "A"), sized("b1", 2, 0.1, "B")]
        p = pack(chunks, 50, count=len, max_per_source=1)
        self.assertEqual(sorted(p.ids), ["a1", "b1"])
        self.assertIn("source 'A' already has 1", p.decisions[1].reason)

    def test_source_token_share(self):
        chunks = [sized("a1", 6, 0.9, "A"), sized("a2", 6, 0.8, "A"), sized("b1", 6, 0.1, "B")]
        p = pack(chunks, 50, count=len, source_tokens=8)
        self.assertEqual(sorted(p.ids), ["a1", "b1"])

    def test_dedupe_drops_the_lower_scored_twin_and_names_it(self):
        chunks = [Chunk("a", "revenue grew twelve percent", 0.9), Chunk("b", "revenue grew twelve percent", 0.8), Chunk("c", "churn fell", 0.1)]
        p = pack(chunks, 100, dedupe_above=0.9)
        self.assertEqual(sorted(p.ids), ["a", "c"])
        self.assertIn("near-duplicate of a", p.decisions[1].reason)

    def test_diversity_adjusted_score_is_reported(self):
        chunks = [Chunk("a", "alpha beta gamma", 0.9), Chunk("b", "alpha beta gamma", 0.8)]
        p = pack(chunks, 100, diversity=1.0)
        adjusted = {d.chunk.id: d.adjusted for d in p.decisions}
        self.assertIsNone(pack(chunks, 100).decisions[0].adjusted)
        self.assertAlmostEqual(adjusted["a"], 0.9)
        self.assertAlmostEqual(adjusted["b"], -0.2)


class StrategyTests(unittest.TestCase):
    # density fails here: the 1-token chunk has the best score per token but leaves no room for the 10-token chunk worth far more.
    DENSITY_TRAP = [sized("small", 1, 2.0), sized("big", 10, 9.0)]
    # greedy by score fails here: the 6-token chunk blocks two 5-token chunks that are worth more together.
    GREEDY_TRAP = [sized("a", 6, 10.0), sized("b", 5, 7.0), sized("c", 5, 7.0)]

    def value(self, chunks, budget, strategy):
        return round(pack(chunks, budget, count=len, strategy=strategy).value, 6)

    def test_density_trap(self):
        self.assertEqual(self.value(self.DENSITY_TRAP, 10, "greedy"), 9.0)
        self.assertEqual(self.value(self.DENSITY_TRAP, 10, "density"), 2.0)
        self.assertEqual(self.value(self.DENSITY_TRAP, 10, "best"), 9.0)

    def test_greedy_trap(self):
        self.assertEqual(self.value(self.GREEDY_TRAP, 10, "greedy"), 10.0)
        self.assertEqual(self.value(self.GREEDY_TRAP, 10, "optimal"), 14.0)
        self.assertEqual(self.value(self.GREEDY_TRAP, 10, "best"), 14.0)

    def test_optimal_equals_brute_force(self):
        rng = random.Random(11)
        for trial in range(120):
            chunks = [sized(f"c{i}", rng.randint(1, 9), round(rng.random(), 3)) for i in range(rng.randint(1, 9))]
            budget = rng.randint(1, 30)
            brute = 0.0
            for r in range(len(chunks) + 1):
                for combo in itertools.combinations(chunks, r):
                    if sum(len(c.text) for c in combo) <= budget:
                        brute = max(brute, sum(c.score for c in combo))
            got = pack(chunks, budget, count=len, strategy="optimal")
            self.assertAlmostEqual(got.value, brute, msg=trial)
            self.assertLessEqual(got.used, budget)
            for other in ("greedy", "density", "best"):
                self.assertLessEqual(pack(chunks, budget, count=len, strategy=other).value, got.value + 1e-9)

    def test_best_is_never_worse_than_greedy_or_density(self):
        rng = random.Random(5)
        for _ in range(100):
            chunks = [sized(f"c{i}", rng.randint(1, 12), rng.random()) for i in range(rng.randint(1, 10))]
            budget = rng.randint(1, 40)
            best = pack(chunks, budget, count=len, strategy="best").value
            for other in ("greedy", "density"):
                self.assertGreaterEqual(best + 1e-9, pack(chunks, budget, count=len, strategy=other).value)

    def test_optimal_refuses_what_it_cannot_do_exactly(self):
        with self.assertRaises(ValueError):
            pack(self.GREEDY_TRAP, 10, 0.5, count=len, strategy="optimal")
        with self.assertRaises(ValueError):
            pack(self.GREEDY_TRAP, 10, count=len, strategy="optimal", max_per_source=1)

    def test_optimal_too_large_is_refused_and_best_says_it_skipped(self):
        chunks = [sized(f"c{i}", 1 + i % 7, 0.5) for i in range(400)]
        with self.assertRaises(ValueError):
            pack(chunks, 10_000, count=len, strategy="optimal")
        p = pack(chunks, 10_000, count=len, strategy="best")
        self.assertTrue(any("exact search skipped" in n for n in p.notes))

    def test_best_with_diversity_picks_the_higher_set_value(self):
        chunks = [Chunk("a", "alpha beta gamma delta", 0.9), Chunk("b", "alpha beta gamma delta", 0.85), Chunk("c", "epsilon zeta eta", 0.5)]
        p = pack(chunks, 12, 1.0, strategy="best")
        self.assertEqual(sorted(p.ids), ["a", "c"])

    def test_set_value_penalises_the_lower_scored_twin(self):
        a, b = Chunk("a", "x y z", 0.9), Chunk("b", "x y z", 0.8)
        self.assertAlmostEqual(set_value([a, b], 1.0, overlap), 0.9 + 0.8 - 1.0)
        self.assertAlmostEqual(set_value([b, a], 1.0, overlap), 0.9 + 0.8 - 1.0)


class TruncateTests(unittest.TestCase):
    def test_sentences_are_kept_whole(self):
        text = "First sentence here. Second sentence follows. Third one ends it."
        cut, was = truncate_to(text, 8, estimate_tokens)
        self.assertTrue(was)
        self.assertEqual(cut, "First sentence here. Second sentence follows.")
        self.assertLessEqual(estimate_tokens(cut), 8)

    def test_fits_already(self):
        self.assertEqual(truncate_to("short text", 50, estimate_tokens), ("short text", False))

    def test_falls_back_to_a_word_cut_with_an_ellipsis(self):
        cut, was = truncate_to("one very long sentence with no full stop until much much later on", 5, estimate_tokens)
        self.assertTrue(was)
        self.assertTrue(cut.endswith("…"))
        self.assertLessEqual(estimate_tokens(cut), 5)
        self.assertNotIn("  ", cut)

    def test_cjk_without_spaces(self):
        cut, _ = truncate_to("今日は天気が良いので散歩に行きます。明日は雨が降るでしょう。", 12, estimate_tokens)
        self.assertLessEqual(estimate_tokens(cut), 12)
        self.assertTrue(cut)

    def test_nothing_fits(self):
        self.assertEqual(truncate_to("hello", 0, len), ("", True))

    def test_pack_truncates_the_best_dropped_chunk_into_the_last_tokens(self):
        a = Chunk("a", "alpha " * 20, 0.9)
        b = Chunk("b", "First point. Second point. Third point. Fourth point.", 0.8)
        budget = estimate_tokens(a.text) + 6
        p = pack([a, b], budget, truncate=True, min_fragment=4)
        self.assertEqual(p.ids[0], "a")
        piece = p.chunks[1]
        self.assertTrue(piece.meta["truncated"])
        self.assertLessEqual(p.used, budget)
        self.assertLess(piece.score, b.score)
        self.assertEqual(p.decisions[1].status, "truncated")
        self.assertEqual(p.dropped, [])

    def test_no_truncation_unless_asked(self):
        a = Chunk("a", "alpha " * 20, 0.9)
        b = Chunk("b", "First point. Second point. Third point. Fourth point.", 0.8)
        self.assertEqual(pack([a, b], estimate_tokens(a.text) + 6).ids, ["a"])


class OrderTests(unittest.TestCase):
    CH = [Chunk(c, "t", s, src) for c, s, src in [("a", 0.1, "X"), ("b", 0.9, "Y"), ("c", 0.5, "X"), ("d", 0.7, "Y"), ("e", 0.3, "Z")]]

    def ids(self, how):
        return [c.id for c in arrange(self.CH, how, [c.id for c in self.CH])]

    def test_original(self):
        self.assertEqual(self.ids("original"), list("abcde"))

    def test_ascending_puts_the_best_last(self):
        self.assertEqual(self.ids("ascending"), ["a", "e", "c", "d", "b"])

    def test_ends_puts_the_two_best_at_the_edges_and_the_weakest_inside(self):
        got = self.ids("ends")
        self.assertEqual((got[0], got[-1]), ("b", "d"))
        self.assertEqual(got[2], "a")

    def test_source_groups_by_first_appearance(self):
        self.assertEqual(self.ids("source"), ["a", "c", "b", "d", "e"])

    def test_pack_applies_the_order(self):
        chunks = [Chunk("lo", "alpha", 0.2), Chunk("hi", "beta", 0.9)]
        self.assertEqual(pack(chunks, 50, order="original").ids, ["lo", "hi"])
        self.assertEqual(pack(chunks, 50).ids, ["hi", "lo"])

    def test_unknown_order(self):
        with self.assertRaises(ValueError):
            arrange(self.CH, "shuffle")


if __name__ == "__main__":
    unittest.main()
