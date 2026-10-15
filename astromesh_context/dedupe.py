"""Removal of exact and near-duplicate items before selection."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ._text import content_hash, shingles
from .types import ScoredItem

_NO_WORDS = frozenset({()})


def _contained(a: frozenset, b: frozenset) -> float:
    """Overlap coefficient ``|a & b| / min(|a|, |b|)``; 0.0 when either text has no words.

    Containment, not Jaccard: a retrieval chunk is a fragment of the document it came from, and
    Jaccard never reaches the threshold for fragment against whole.
    """
    if a == _NO_WORDS or b == _NO_WORDS:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _inherit(kept: ScoredItem, dropped: ScoredItem) -> ScoredItem:
    """Merge the dropped copy's signals into the kept one (flags OR-ed; trust/citation if unset)."""
    k, d = kept.item.signals, dropped.item.signals
    changes: dict[str, object] = {}
    if d.deprecated and not k.deprecated:
        changes["deprecated"] = True
    if d.stale and not k.stale:
        changes["stale"] = True
    if k.trust is None and d.trust is not None:
        changes["trust"] = d.trust
    if k.citation is None and d.citation is not None:
        changes["citation"] = d.citation
    if not changes:
        return kept
    item = dataclasses.replace(kept.item, signals=dataclasses.replace(k, **changes))
    notes = tuple(f"inherited:{name}:{dropped.item.id}" for name in changes)
    return dataclasses.replace(kept, item=item, adjustments=(*kept.adjustments, *notes))


def dedupe(
    items: Sequence[ScoredItem], threshold: float = 0.8
) -> tuple[list[ScoredItem], list[tuple[ScoredItem, str]]]:
    """Drop items repeating a better one; returns (kept in input order, dropped with reason).

    Items are visited pinned first, then by ``final`` descending, ties by id, so the strongest
    copy survives and inherits the dropped copy's signals (``deprecated``/``stale`` OR-ed,
    ``trust``/``citation`` when unset). An item is dropped as ``duplicate_of:<kept id>`` when its id, its normalized
    content hash, or (containment over 5-word shingles >= ``threshold``) its content matches a kept one.

    History items (``kind == "history"``) are never deduplicated and never serve as the kept copy:
    a repeated turn is conversation, not redundant knowledge.
    """
    order = sorted(items, key=lambda s: (not s.item.pinned, -s.final, s.item.id))
    kept_ids: set[str] = set()
    kept_hashes: dict[str, str] = {}
    kept_shingles: list[tuple[str, frozenset[tuple[str, ...]]]] = []
    kept: dict[int, ScoredItem] = {}  # id(original) -> the item with inherited signals
    kept_key: dict[str, int] = {}  # kept item id -> id(original)
    dropped: dict[int, tuple[ScoredItem, str]] = {}
    for scored in order:
        item = scored.item
        if item.kind == "history":
            kept[id(scored)] = scored
            continue
        digest = content_hash(item.content)
        dup_of = item.id if item.id in kept_ids else kept_hashes.get(digest)
        sh = shingles(item.content)
        if dup_of is None:
            for kept_id, other in kept_shingles:
                if _contained(sh, other) >= threshold:
                    dup_of = kept_id
                    break
        if dup_of is not None:
            dropped[id(scored)] = (scored, f"duplicate_of:{dup_of}")
            key = kept_key[dup_of]
            kept[key] = _inherit(kept[key], scored)
            continue
        kept[id(scored)] = scored
        kept_key[item.id] = id(scored)
        kept_ids.add(item.id)
        kept_hashes[digest] = item.id
        kept_shingles.append((item.id, sh))
    return (
        [kept[id(s)] for s in items if id(s) in kept],
        [dropped[id(s)] for s in order if id(s) in dropped],
    )
