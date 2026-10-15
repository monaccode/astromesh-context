"""Removal of exact and near-duplicate items before selection."""

from __future__ import annotations

from collections.abc import Sequence

from ._text import content_hash, shingles
from .types import ScoredItem


def dedupe(
    items: Sequence[ScoredItem], threshold: float = 0.8
) -> tuple[list[ScoredItem], list[tuple[ScoredItem, str]]]:
    """Drop items repeating a better one; returns (kept in input order, dropped with reason).

    Items are visited pinned first, then by ``final`` descending, ties by id, so the strongest
    copy survives. An item is dropped as ``duplicate_of:<kept id>`` when its id, its normalized
    content hash, or (Jaccard over 5-word shingles >= ``threshold``) its content matches a kept one.
    """
    order = sorted(items, key=lambda s: (not s.item.pinned, -s.final, s.item.id))
    kept_ids: set[str] = set()
    kept_hashes: dict[str, str] = {}
    kept_shingles: list[tuple[str, frozenset[tuple[str, ...]]]] = []
    kept: set[int] = set()
    dropped: dict[int, tuple[ScoredItem, str]] = {}
    for scored in order:
        item = scored.item
        digest = content_hash(item.content)
        dup_of = item.id if item.id in kept_ids else kept_hashes.get(digest)
        sh = shingles(item.content)
        if dup_of is None:
            for kept_id, other in kept_shingles:
                if len(sh & other) / len(sh | other) >= threshold:
                    dup_of = kept_id
                    break
        if dup_of is not None:
            dropped[id(scored)] = (scored, f"duplicate_of:{dup_of}")
            continue
        kept.add(id(scored))
        kept_ids.add(item.id)
        kept_hashes[digest] = item.id
        kept_shingles.append((item.id, sh))
    return (
        [s for s in items if id(s) in kept],
        [dropped[id(s)] for s in order if id(s) in dropped],
    )
