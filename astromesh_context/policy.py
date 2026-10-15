"""Selection policies: which scored candidates fill the token budget."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from astromesh_context.errors import BudgetExceededError
from astromesh_context.types import Budget, ScoredItem, Selection


class Policy(Protocol):
    def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection: ...


def _best_first(items: Sequence[ScoredItem]) -> list[ScoredItem]:
    return sorted(items, key=lambda s: (-s.final, s.item.id))


def _split_pinned(
    items: Sequence[ScoredItem], budget: Budget, include_deprecated: bool = False
) -> tuple[list[ScoredItem], list[ScoredItem], int, list[tuple[ScoredItem, str]]]:
    """Separate pinned from the rest and drop deprecated items.

    Returns (pinned, rest, pinned_tokens, excluded). Pinned items keep input order and win over the
    deprecated filter. Raises BudgetExceededError when pinned tokens exceed the budget.
    """
    pinned = [s for s in items if s.item.pinned]
    candidates = [s for s in items if not s.item.pinned]
    excluded: list[tuple[ScoredItem, str]] = []
    rest: list[ScoredItem] = []
    for s in sorted(candidates, key=lambda s: s.item.id):
        if s.item.signals.deprecated and not include_deprecated:
            excluded.append((s, "deprecated"))
        else:
            rest.append(s)
    pinned_tokens = sum(s.tokens for s in pinned)
    if pinned_tokens > budget.total:
        raise BudgetExceededError(budget.total, [(s.item.id, s.tokens) for s in pinned])
    return pinned, rest, pinned_tokens, excluded


class ScoredKnapsack:
    def __init__(
        self,
        beta: float = 0.0,
        min_score: float | None = None,
        include_deprecated: bool = False,
    ) -> None:
        self.beta = beta
        self.min_score = min_score
        self.include_deprecated = include_deprecated

    def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection:
        pinned, rest, used, excluded = _split_pinned(items, budget, self.include_deprecated)
        included = list(pinned)
        per_source: dict[str, int] = {}  # non-pinned tokens per source

        def add(s: ScoredItem) -> None:
            nonlocal used
            included.append(s)
            used += s.tokens
            per_source[s.item.source] = per_source.get(s.item.source, 0) + s.tokens

        # 1. Forced minimums: ignore min_score and caps, respect the overall budget.
        forced: set[str] = set()
        for source in sorted(budget.min_items):
            pool = _best_first([s for s in rest if s.item.source == source])
            for s in pool[: budget.min_items[source]]:
                if used + s.tokens <= budget.total:
                    add(s)
                    forced.add(s.item.id)

        # 2. Density order.
        remaining = [s for s in rest if s.item.id not in forced]
        remaining.sort(key=lambda s: (-s.final / s.tokens**self.beta, s.item.id))
        over_budget: list[ScoredItem] = []
        for s in remaining:
            cap = budget.caps.get(s.item.source)
            if self.min_score is not None and s.final < self.min_score:
                excluded.append((s, "below_floor"))
            elif cap is not None and per_source.get(s.item.source, 0) + s.tokens > cap:
                excluded.append((s, "source_cap"))
            elif used + s.tokens > budget.total:
                over_budget.append(s)
            else:
                add(s)

        # 3. Back-fill: one pass over budget exclusions, in the same order.
        for s in over_budget:
            if used + s.tokens <= budget.total:
                add(s)
            else:
                excluded.append((s, "budget"))
        return Selection(included, excluded)


class FixedSlots:
    def __init__(self, shares: Mapping[str, float], include_deprecated: bool = False) -> None:
        if any(v < 0 for v in shares.values()):
            raise ValueError(f"shares must be >= 0, got {dict(shares)}")
        if sum(shares.values()) > 1.0 + 1e-9:
            raise ValueError(f"shares must sum to at most 1.0, got {sum(shares.values())}")
        self.shares = dict(shares)
        self.include_deprecated = include_deprecated

    def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection:
        pinned, rest, pinned_tokens, excluded = _split_pinned(
            items, budget, self.include_deprecated
        )
        free = budget.total - pinned_tokens
        room = {src: math.floor(share * free) for src, share in self.shares.items()}
        included = list(pinned)
        for s in _best_first(rest):
            src = s.item.source
            if src in room and s.tokens <= room[src]:
                room[src] -= s.tokens
                included.append(s)
            else:
                excluded.append((s, "share_exhausted"))
        return Selection(included, excluded)


class Everything:
    def __init__(self, truncate: bool = False, include_deprecated: bool = False) -> None:
        self.truncate = truncate
        self.include_deprecated = include_deprecated

    def select(self, items: Sequence[ScoredItem], budget: Budget) -> Selection:
        pinned, rest, used, excluded = _split_pinned(items, budget, self.include_deprecated)
        ordered = _best_first(rest)
        if used + sum(s.tokens for s in ordered) <= budget.total:
            return Selection(pinned + ordered, excluded)
        if not self.truncate:
            raise BudgetExceededError(
                budget.total, [(s.item.id, s.tokens) for s in pinned + ordered]
            )
        included = list(pinned)
        for s in ordered:
            if used + s.tokens <= budget.total:
                included.append(s)
                used += s.tokens
            else:
                excluded.append((s, "truncated"))
        return Selection(included, excluded)
