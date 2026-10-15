"""Level-1 metrics: pure functions of a window's contents."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import pairwise

from astromesh_context.types import ContextItem


def recall_at_budget(window_doc_ids: set[str], relevant: frozenset[str]) -> float:
    """Share of the relevant documents present in the window (1.0 when none are expected)."""
    if not relevant:
        return 1.0
    return len(relevant & window_doc_ids) / len(relevant)


def precision_per_token(
    items: Iterable[ContextItem],
    doc_id_of: Callable[[ContextItem], str | None],
    relevant: frozenset[str],
) -> float:
    """Tokens that come from relevant documents over all tokens in the window."""
    total = 0
    useful = 0
    for it in items:
        n = it.tokens or 0
        total += n
        if doc_id_of(it) in relevant:
            useful += n
    return useful / total if total else 0.0


def distractor_rate(windows_doc_ids: Sequence[set[str]], distractors: frozenset[str]) -> float:
    """Share of windows that contain at least one distractor document."""
    if not windows_doc_ids:
        return 0.0
    return sum(1 for ids in windows_doc_ids if ids & distractors) / len(windows_doc_ids)


def prefix_reuse(hashes: Sequence[str]) -> float:
    """Share of consecutive calls with an identical stable-prefix hash (1.0 below two calls)."""
    if len(hashes) < 2:
        return 1.0
    same = sum(1 for a, b in pairwise(hashes) if a == b)
    return same / (len(hashes) - 1)
