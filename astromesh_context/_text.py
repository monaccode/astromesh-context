"""Text helpers shared by deduplication and lexical scoring."""

from __future__ import annotations

import hashlib
import re

_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Lower-cased Unicode word tokens."""
    return _WORD.findall(text.lower())


def shingles(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    """Sets of ``n`` consecutive words; shorter texts yield one shingle of all their words."""
    toks = words(text)
    if len(toks) <= n:
        return frozenset({tuple(toks)})
    return frozenset(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def content_hash(text: str) -> str:
    """sha256 of the text, insensitive to case and whitespace."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
