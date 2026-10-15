"""Cache-stable ordering of the selected items."""

from __future__ import annotations

from collections.abc import Sequence

from astromesh_context.types import ContextItem, ScoredItem


def _edges_first(ranked: list[ContextItem]) -> list[ContextItem]:
    """Place best-first items at positions 0, -1, 1, -2, ... so the best sit at the edges."""
    left: list[ContextItem] = []
    right: list[ContextItem] = []
    for i, item in enumerate(ranked):
        (left if i % 2 == 0 else right).append(item)
    return left + right[::-1]


def order_items(items: Sequence[ScoredItem], source_order: Sequence[str]) -> list[ContextItem]:
    """Order as stable -> session -> volatile, then history; stable prefix is deterministic."""
    rank = {name: i for i, name in enumerate(source_order)}
    body = [s for s in items if s.item.kind != "history"]
    history = [s.item for s in items if s.item.kind == "history"]

    stable = sorted((s.item for s in body if s.item.stability == "stable"), key=lambda i: i.id)
    session = sorted(
        (s.item for s in body if s.item.stability == "session"),
        key=lambda i: (
            rank.get(i.source, len(rank)),
            i.order_key is not None,
            i.order_key or "",
            i.id,
        ),
    )
    volatile = _edges_first(
        [
            s.item
            for s in sorted(
                (s for s in body if s.item.stability == "volatile"),
                key=lambda s: (-s.final, s.item.id),
            )
        ]
    )
    history.sort(key=lambda i: (i.order_key is not None, i.order_key or "", i.id))
    return [*stable, *session, *volatile, *history]
