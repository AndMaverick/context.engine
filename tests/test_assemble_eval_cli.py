import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from context_engine import BudgetError, Chunk, assemble, compare_strategies, coverage, estimate_tokens, pack, topk_stop
from context_engine.cli import main
from context_engine.io import ChunkFormatError, parse_chunks
from context_engine.synthetic import corpus


def docs(n=6, words=12):
    return [Chunk(f"d{i}", " ".join(f"w{i}x{j}" for j in range(words)), 1 - i / 10, f"s{i % 2}") for i in range(n)]


class AssembleTests(unittest.TestCase):
    def test_the_finished_prompt_fits_the_window_with_the_answer_reserved(self):
        a = assemble(docs(), window=120, reserve_output=30, system="You answer from the context only.", question="What changed?")
        self.assertLessEqual(a.tokens, 120 - 30)
        self.assertEqual(a.tokens, estimate_tokens(a.prompt))
        self.assertGreaterEqual(a.headroom, 0)
        self.assertEqual(a.breakdown["reserved for answer"], 30)
        self.assertTrue(a.prompt.startswith("You answer"))
        self.assertTrue(a.prompt.endswith("What changed?"))

    def test_citations_follow_the_final_order(self):
        a = assemble(docs(), window=400, order="ascending", chunk_format="[{n}] {id}: {text}")
        self.assertEqual(list(a.citations), list(range(1, len(a.packed.chunks) + 1)))
        self.assertEqual(list(a.citations.values()), a.packed.ids)
        first = a.context.split("\n\n")[0]
        self.assertTrue(first.startswith(f"[1] {a.citations[1]}:"))

    def test_no_room_raises(self):
        with self.assertRaises(BudgetError):
            assemble(docs(), window=20, reserve_output=15, system="a long system prompt that eats the whole budget of this window")

    def test_formatting_overhead_is_budgeted(self):
        plain = assemble(docs(10), window=100, chunk_format="{text}", separator=" ")
        fancy = assemble(docs(10), window=100, chunk_format="### Source {id} ({source})\n{text}", separator="\n\n")
        self.assertLessEqual(fancy.tokens, 100)
        self.assertGreater(len(plain.packed.chunks), len(fancy.packed.chunks))

    def test_irregular_counter_is_repacked_until_it_fits(self):
        def wobbly(text: str) -> int:  # not additive: the whole costs more than the sum of its parts
            return len(text) + len(text) // 7
        a = assemble(docs(8, 5), window=200, reserve_output=10, count=wobbly, chunk_format="{text}")
        self.assertLessEqual(wobbly(a.prompt), 190)

    def test_pack_options_pass_through(self):
        a = assemble(docs(), window=400, pinned=("d5",), chunk_format="{id}")
        self.assertEqual(a.packed.ids[0], "d5")


class EvaluateTests(unittest.TestCase):
    def test_topk_stop_halts_at_the_first_overflow(self):
        chunks = [Chunk("a", "x" * 6, 0.9), Chunk("b", "x" * 6, 0.8), Chunk("c", "x" * 2, 0.7)]
        self.assertEqual(topk_stop(chunks, 8, count=len).ids, ["a"])
        self.assertEqual(pack(chunks, 8, count=len).ids, ["a", "c"])

    def test_coverage(self):
        chunks = [Chunk("a", "t", 1.0, meta={"facts": ["f1", "f2"]}), Chunk("b", "t", 1.0, meta={"facts": ["f3"]})]
        self.assertEqual(coverage(chunks[:1], universe=chunks), (2, 3))
        self.assertEqual(coverage(chunks, gold=["f1", "f9"]), (1, 2))

    def test_compare_rows_and_fairness(self):
        chunks, gold = corpus(3)
        rows = {r.strategy: r for r in compare_strategies(chunks, 700, 0.8, gold=gold)}
        self.assertEqual(set(rows), {"topk-stop", "greedy", "density", "optimal", "mmr", "best+mmr"})
        for r in rows.values():
            self.assertLessEqual(r.used, 700)
            self.assertEqual(r.gold, len(gold))
        self.assertGreaterEqual(rows["optimal"].score, rows["greedy"].score - 1e-9)
        self.assertGreaterEqual(rows["greedy"].score, rows["topk-stop"].score - 1e-9)

    def test_without_facts_there_is_no_coverage(self):
        rows = compare_strategies(docs(), 60)
        self.assertIsNone(rows[0].covered)

    def test_diversity_covers_more_gold_facts_than_plain_retrieval_on_redundant_corpora(self):
        plain = diverse = 0
        for seed in range(15):
            chunks, gold = corpus(seed)
            budget = sum(estimate_tokens(c.text) for c in chunks) // 5
            plain += coverage(pack(chunks, budget).chunks, gold, chunks)[0]
            diverse += coverage(pack(chunks, budget, 0.9, similarity="shingle").chunks, gold, chunks)[0]
        self.assertGreater(diverse, plain)

    def test_corpus_is_deterministic(self):
        a, _ = corpus(1)
        b, _ = corpus(1)
        self.assertEqual([(c.id, c.text, c.score) for c in a], [(c.id, c.text, c.score) for c in b])


class IoTests(unittest.TestCase):
    def test_parses_and_skips_blank_lines(self):
        got = parse_chunks(['{"id":"a","text":"hi","score":1,"source":"s","meta":{"k":1}}', "", '{"id":"b","text":"yo","score":0.5}'])
        self.assertEqual([(c.id, c.score, c.source) for c in got], [("a", 1.0, "s"), ("b", 0.5, "")])
        self.assertEqual(got[0].meta, {"k": 1})

    def test_errors_name_the_line(self):
        for bad, fragment in [
            ('{"id":"a","text":"t"}', "'score' must be a number"),
            ('{"id":1,"text":"t","score":1}', "'id' must be a string"),
            ('{"id":"a","text":"t","score":true}', "'score' must be a number"),
            ('[1]', "expected a JSON object"),
            ('{oops', "not valid JSON"),
            ('{"id":"a","text":"t","score":1,"meta":3}', "'meta' must be an object"),
        ]:
            with self.assertRaises(ChunkFormatError) as ctx:
                parse_chunks(["", bad], "f.jsonl")
            self.assertIn("f.jsonl line 2", str(ctx.exception))
            self.assertIn(fragment, str(ctx.exception))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "chunks.jsonl")
        rows = [
            {"id": "policy", "text": "refunds within thirty days", "score": 0.4, "source": "kb", "meta": {"facts": ["refund"]}},
            {"id": "a", "text": "revenue grew twelve percent this quarter", "score": 0.9, "source": "report", "meta": {"facts": ["rev"]}},
            {"id": "b", "text": "revenue grew twelve percent this quarter overall", "score": 0.88, "source": "report", "meta": {"facts": ["rev"]}},
            {"id": "c", "text": "churn fell after the redesign shipped", "score": 0.7, "source": "report", "meta": {"facts": ["churn"]}},
        ]
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows))

    def tearDown(self):
        self.dir.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_pack_explain(self):
        code, out, _ = self.run_cli("pack", self.path, "--budget", "20")
        self.assertEqual(code, 0)
        self.assertIn("tokens used", out)
        self.assertIn("dropped", out)

    def test_require_gate(self):
        code, _, err = self.run_cli("pack", self.path, "--budget", "9", "--require", "policy")
        self.assertEqual(code, 1)
        self.assertIn("required chunk 'policy' was not packed", err)
        self.assertEqual(self.run_cli("pack", self.path, "--budget", "9", "--pin", "policy", "--require", "policy")[0], 0)

    def test_fail_on_drop_score_gate(self):
        code, _, err = self.run_cli("pack", self.path, "--budget", "11", "--fail-on-drop-score", "0.6")
        self.assertEqual(code, 1)
        self.assertIn("scored", err)
        self.assertEqual(self.run_cli("pack", self.path, "--budget", "200", "--fail-on-drop-score", "0.6")[0], 0)

    def test_output_formats(self):
        _, out, _ = self.run_cli("pack", self.path, "--budget", "200", "--output", "ids")
        self.assertEqual(sorted(out.split()), ["a", "b", "c", "policy"])
        _, out, _ = self.run_cli("pack", self.path, "--budget", "200", "--output", "json")
        self.assertEqual(json.loads(out)["budget"], 200)
        _, out, _ = self.run_cli("pack", self.path, "--budget", "200", "--output", "text")
        self.assertIn("churn fell", out)

    def test_dedupe_flag_drops_the_twin(self):
        _, out, _ = self.run_cli("pack", self.path, "--budget", "200", "--dedupe-above", "0.7", "--output", "ids")
        self.assertNotIn("b", out.split())

    def test_compare_table(self):
        code, out, _ = self.run_cli("compare", self.path, "--budget", "17", "--diversity", "0.9")
        self.assertEqual(code, 0)
        for name in ("topk-stop", "greedy", "density", "optimal", "mmr", "best+mmr", "facts"):
            self.assertIn(name, out)

    def test_dedupe_command(self):
        code, out, _ = self.run_cli("dedupe", self.path, "--threshold", "0.6")
        self.assertEqual(code, 0)
        self.assertIn("b  duplicates a", out)

    def test_count_command(self):
        code, out, _ = self.run_cli("count", self.path)
        self.assertEqual(code, 0)
        self.assertGreater(int(out), 10)

    def test_errors_exit_2(self):
        self.assertEqual(self.run_cli("pack", os.path.join(self.dir.name, "missing.jsonl"), "--budget", "5")[0], 2)
        code, _, err = self.run_cli("pack", self.path, "--budget", "5", "--pin", "nope")
        self.assertEqual(code, 2)
        self.assertIn("nope", err)
        code, _, err = self.run_cli("pack", self.path, "--budget", "1", "--pin", "a")
        self.assertEqual(code, 2)
        self.assertIn("budget is 1", err)

    def test_module_entry_point(self):
        proc = subprocess.run([sys.executable, "-m", "context_engine", "pack", self.path, "--budget", "50"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("tokens used", proc.stdout)


if __name__ == "__main__":
    unittest.main()
