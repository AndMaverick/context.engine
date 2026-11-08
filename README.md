# context.engine

Context windows are a budget, and most retrieval pipelines spend it badly: five near-duplicate chunks and the one fact that mattered gets cut. context.engine packs the highest-value, least redundant chunks into a fixed token budget, with pinned items that always make it in. Standard library only.

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
prompt_context = packed.text()
packed.used, packed.dropped
```

## Design

- **Greedy by adjusted score.** Each round picks the chunk with the best score minus a redundancy penalty against what is already selected.
- **Diversity knob.** `diversity=0` is plain top-k by score. Raise it to prefer novel information over near duplicates. Redundancy is measured as word-set overlap.
- **Pinned chunks.** Instructions, policy text, or anything that must be present is included first. If pinned items alone exceed the budget, it raises instead of silently truncating.
- **Honest accounting.** The result reports tokens used, the budget, and exactly which chunks were dropped.
- **Token estimate.** A cheap word-based heuristic. Swap `tokens()` for your real tokenizer in production.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).
