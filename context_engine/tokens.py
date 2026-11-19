"""Token counting. Real tokenizers differ by model, so everything here takes a ``count`` function and defaults to an estimate."""
from __future__ import annotations

from collections.abc import Callable

Counter = Callable[[str], int]

_CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF))
_NON_LATIN = ((0x0370, 0x03FF), (0x0400, 0x04FF), (0x0590, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0x0900, 0x097F),
              (0x0E00, 0x0E7F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))


def _in(code: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= code <= hi for lo, hi in ranges)


def estimate_tokens(text: str) -> int:
    """A tokenizer-free estimate. Space-separated Latin text counts about 1.33 tokens per word (the 0.1 behaviour). Words in
    Arabic, Cyrillic, Greek, Hebrew, Devanagari or Thai count 2, since tokenizers split them finer. Han, kana and hangul count
    one token per character, since they have no spaces to count. These are rules of thumb, not measurements of any model's
    tokenizer, and real counts can differ by tens of percent: pass ``count=`` wherever you have the model's own tokenizer."""
    latin = non_latin = cjk = 0
    for word in text.split():
        han = sum(1 for ch in word if _in(ord(ch), _CJK))
        if han:
            cjk += han
            latin += 1 if han < len(word) else 0
        elif any(_in(ord(ch), _NON_LATIN) for ch in word):
            non_latin += 1
        else:
            latin += 1
    return max(1, round(latin * 1.33 + non_latin * 2 + cjk))


tokens = estimate_tokens  # the 0.1 name
