# context.engine

Context windows are a budget, and most retrieval pipelines spend it badly: five nearly identical passages come back, the one fact that mattered gets cut, and nobody can say why. context.engine packs the most valuable, least redundant chunks into a fixed token budget, keeps the chunks that must be there, and records the reason for every chunk it did not use. Standard library only, Python 3.10 to 3.13, typed, 76 tests.

## Use

```python
from context_engine import Chunk, pack

chunks = [
    Chunk("policy", "Refunds are allowed within 30 days.", score=0.0),
    Chunk("a", "Quarterly revenue grew twelve percent.", score=0.90),
    Chunk("b", "Quarterly revenue grew twelve percent year on year.", score=0.89),
    Chunk("c", "Customer churn fell after the redesign.", score=0.80),
]

packed = pack(chunks, budget=40, diversity=1.0, pinned=("policy",))
packed.text()        # the context, ready for the prompt
print(packed.explain())  # one line per chunk: kept, pinned or dropped, and why
```

The 0.1 API (`Chunk`, `pack`, `tokens`, `overlap`, `Packed.text()`) works unchanged.

## What it does

| | |
|---|---|
| **Strategies** | `greedy` (best adjusted score first, the 0.1 behaviour), `density` (best score per token), `optimal` (exact 0/1 knapsack on total score), `best` (runs the others and keeps the set worth the most). Greedy by score and greedy by density each have a case where they lose to the other; `best` covers both, and the test suite checks `optimal` against brute force on 120 random instances. |
| **Redundancy** | `diversity` pushes each chunk's score down by its similarity to what is already chosen. Similarity is `jaccard` (word sets), `shingle` (character n grams, works for Chinese and Japanese and sees inflection), `containment` (a short chunk inside a long one), or your own function, for example one built on embeddings. |
| **Hard rules** | `dedupe_above` drops a chunk outright when it is too like one already chosen. `max_per_source` and `source_tokens` stop one source from taking the window. `min_score` sets a floor. Pinned chunks skip all of these. |
| **Real accounting** | `count=` takes your model's tokenizer. `overhead=` charges each chunk for its citation header and separator. Over budget pins raise instead of truncating silently. |
| **Truncation** | `truncate=True` fills the last tokens with the best chunk that did not fit, cut at a sentence boundary where one fits. Its score is scaled by the share kept and `meta["truncated"]` says it was cut. |
| **Order** | `order="ends"` puts the best chunks at the start and end of the context, `"ascending"` puts the best last, `"original"` keeps document order, `"source"` groups by source. Long context models are reported to use the start and end of a prompt better than the middle (Liu et al., 2023); the library gives you the ordering, it does not measure the effect on your model. |
| **Whole prompt** | `assemble(chunks, window=..., reserve_output=..., system=..., question=...)` budgets the system text, the question, the formatting and the answer space, then counts again the finished prompt and repacks tighter if it came out over. It returns citations mapping `[n]` to chunk ids. |
| **Explanations** | `Packed.decisions` has one record per input chunk with its status, its token cost, its adjusted score and the reason. `explain()` prints them. |

## Command line

`context-engine` (or `python -m context_engine`) reads JSON Lines of `{"id", "text", "score", "source"?, "meta"?}`. Exit codes: 0 success, 1 a gate was hit, 2 error. `--require ID` fails the run if a chunk is not packed, and `--fail-on-drop-score S` fails it if any dropped chunk scored at least `S`, so a pipeline can refuse to answer from a context that lost something important.

```
$ context-engine compare examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle
strategy      tokens  chunks    score     facts
topk-stop         75       3    2.740     1/5  
greedy            75       3    2.740     1/5  
density           69       3    2.550     2/5  
optimal           75       3    2.740     1/5  
mmr               72       3    2.280     3/5  
best+mmr          72       3    2.280     3/5  

$ context-engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --pin refund-policy --require refund-policy
75 of 90 tokens used, 3 chunks packed, 5 dropped
  pinned   refund-policy      27 tok  score 0.520  pinned by the caller
  selected q3-a               27 tok  score 0.930 (adjusted 0.930)  fits and ranked above what was left
  dropped  q3-b               25 tok  score 0.910  no room: needs 25 tokens, 15 left; ranked down 0.66 for similarity 0.74 to q3-a
  dropped  q3-c               23 tok  score 0.900  no room: needs 23 tokens, 15 left; ranked down 0.39 for similarity 0.43 to q3-a
  selected churn              21 tok  score 0.740 (adjusted 0.678)  fits and ranked above what was left
  dropped  sla                24 tok  score 0.610  no room: needs 24 tokens, 15 left
  dropped  hiring             25 tok  score 0.380  no room: needs 25 tokens, 15 left
  dropped  q3-d               25 tok  score 0.880  no room: needs 25 tokens, 15 left; ranked down 0.35 for similarity 0.39 to q3-a
exit 0

$ context-engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --require refund-policy
gate: required chunk 'refund-policy' was not packed
exit 1
```

The first table is the point of `compare`: on these eight chunks the top k baseline and plain greedy fill the budget with near copies of the same revenue sentence and cover 1 of the 5 facts in the corpus. Diversity covers 3, at a lower total score, which is the trade it makes.

## Measured

`python examples/benchmark.py` packs 200 seeded synthetic retrieval corpora (about 160 chunks each, 8 of 40 facts are the ones the question needs, the best documents come back as several nearly identical passages, scores are noisy) into one fifth of their tokens and counts how many of the 8 needed facts are present. The interval is a 95% bootstrap over corpora of the difference from the top k baseline. It takes about three and a half minutes.

```
200 corpora, gold facts per corpus: 8, ran in 210.7s

packer                                facts/8  vs top-k      95% interval  sum score
top-k, stop at first overflow            6.70     +0.00    [+0.00, +0.00]      28.65
greedy by score (0.1 behaviour)          6.72     +0.02    [+0.01, +0.04]      28.92
best (greedy / density / exact)          6.84     +0.14    [-0.03, +0.30]      31.56
greedy + diversity 0.9 (jaccard)         8.00     +1.30    [+1.18, +1.42]      26.10
greedy + diversity 0.9 (shingle)         8.00     +1.30    [+1.18, +1.43]      26.83
best + diversity 0.9 (shingle)           8.00     +1.30    [+1.19, +1.42]      28.16
diversity 0.9 + dedupe at 0.8            8.00     +1.30    [+1.18, +1.43]      26.83
```

Read it for what it is. The corpus is built to contain redundancy, so it shows what diversity aware packing does when redundancy is there, and it hits the ceiling of 8 of 8. Diversity lowers the summed score (26 against 29) because it declines high scoring near copies on purpose. `best` raises the summed score by about 9% over greedy but does not move fact coverage in a way this sample can distinguish from zero. The synthetic text reuses exact words, so `jaccard` and `shingle` tie here. The unit tests show where they differ (inflection, text with no spaces). None of this says anything about your corpus. Run `compare` on yours.

## Honest limits

- **Token counts are estimates** unless you pass `count=`. The rules that ship with the tool (about 1.33 tokens per Latin word, 2 per word in Arabic, Cyrillic and similar scripts, 1 per CJK character) are rules of thumb and real counts can differ by tens of percent. `assemble` checks the finished prompt with whatever counter you give it, so the check is only as good as the counter.
- **Scores are yours.** Retrieval scores from different retrievers are not on one scale and the engine does not calibrate them.
- **Similarity is lexical** unless you supply a function. Two passages that say the same thing in different words look unrelated to `jaccard` and `shingle`.
- **Diversity is a heuristic.** It is the maximal marginal relevance idea applied greedily. Only `optimal` is exact, and only for total score with no diversity, quotas or dedupe; it refuses otherwise, and `best` says in `Packed.notes` when the search was too big to run (over 3,000,000 item by token cells).
- **Truncation cuts meaning.** A sentence boundary cut is cleaner than a mid sentence one and still drops what came after.
- **No semantics, no retrieval, no reranking.** It decides what fits. It does not decide what is true.

## Tests

```
python -m unittest discover -s tests -t . -v
ruff check . && mypy
```

**Related repositories:** [guard.rails](https://github.com/AndMaverick/guard.rails) can redact personal data and secrets from chunks before they are packed. [mao-framework](https://github.com/AndMaverick/mao-framework) has a smaller module, `mao.context`, that packs retrieved chunks into a token budget inside a governed run.

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Read the thinking:** [The Job Nobody Has Yet](https://blog.andmaverick.com/the-job-nobody-has-yet?utm_source=github&utm_medium=repo&utm_campaign=context.engine)

**If this is your problem.** If you are deciding what to build, buy, and fund, I help leadership teams make that call with evidence. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=context.engine).
