"""context.engine: pack the best, least redundant context into a fixed token budget, and say why every chunk was kept or dropped."""
from __future__ import annotations

from .assemble import Assembled, BudgetError, assemble
from .evaluate import Row, compare_strategies, coverage, topk_stop
from .model import Chunk, Decision, Packed
from .order import ORDERS, arrange
from .pack import STRATEGIES, pack, set_value
from .similarity import containment, overlap, shingle_similarity
from .text import truncate_to
from .tokens import estimate_tokens, tokens

__version__ = "0.2.0"

__all__ = [
    "ORDERS", "STRATEGIES", "Assembled", "BudgetError", "Chunk", "Decision", "Packed", "Row", "arrange", "assemble",
    "compare_strategies", "containment", "coverage", "estimate_tokens", "overlap", "pack", "set_value", "shingle_similarity",
    "tokens", "topk_stop", "truncate_to",
]
