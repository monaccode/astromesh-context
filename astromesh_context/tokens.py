"""Token counting. The default is an estimate; install the `tiktoken` extra for exact counts."""

from __future__ import annotations

import math
from typing import Protocol

from .errors import MissingExtraError


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...


class CharEstimator:
    """`ceil(len(text) / 4)`. An estimate that can miss by ~10-20 %; leave headroom."""

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class TiktokenTokenizer:
    """Exact counts for OpenAI encodings. `tiktoken` is imported lazily."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        self.encoding = encoding
        self._enc = None

    def count(self, text: str) -> int:
        if self._enc is None:
            try:
                import tiktoken
            except ImportError as exc:
                raise MissingExtraError("install astromesh-context[tiktoken]") from exc
            self._enc = tiktoken.get_encoding(self.encoding)
        return len(self._enc.encode(text))
