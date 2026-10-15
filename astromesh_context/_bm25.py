"""Okapi BM25 over pre-tokenized documents."""

from __future__ import annotations

import math
from collections import Counter


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg = 0.0

    def fit(self, docs: list[list[str]]) -> None:
        self._tf = [Counter(d) for d in docs]
        self._len = [len(d) for d in docs]
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avg = sum(self._len) / len(docs) if docs else 0.0

    def scores(self, query_terms: list[str]) -> list[float]:
        n = len(self._tf)
        out = [0.0] * n
        for term in set(query_terms):
            df = self._df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (self._len[i] / self._avg if self._avg else 0.0)
                out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out
