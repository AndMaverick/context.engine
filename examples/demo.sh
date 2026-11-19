#!/bin/sh
# python -m context_engine on examples/chunks.jsonl: 8 chunks, 197 estimated tokens, a budget of 90.
cd "$(dirname "$0")/.."
echo '$ context-engine compare examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle'
python -m context_engine compare examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle
echo
echo '$ context-engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --pin refund-policy --require refund-policy'
python -m context_engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --pin refund-policy --require refund-policy
echo "exit $?"
echo
echo '$ context-engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --require refund-policy'
python -m context_engine pack examples/chunks.jsonl --budget 90 --diversity 0.9 --similarity shingle --require refund-policy >/dev/null
echo "exit $?"
