"""Render ordered items as text or chat messages."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal

from astromesh_context.types import ContextItem


def _block(item: ContextItem) -> str:
    if item.kind == "instruction":
        return item.content
    return f'<context id="{item.id}" source="{item.source}">\n{item.content}\n</context>'


def render_text(items: Sequence[ContextItem]) -> str:
    return "\n\n".join(_block(i) for i in items if i.kind != "history")


def stable_block(items: Sequence[ContextItem]) -> str:
    return "\n\n".join(_block(i) for i in items if i.kind != "history" and i.stability == "stable")


def stable_prefix_hash(items: Sequence[ContextItem]) -> str:
    return hashlib.sha256(stable_block(items).encode("utf-8")).hexdigest()


def _history_messages(items: Sequence[ContextItem]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if item.kind != "history":
            continue
        msgs = item.meta.get("messages")
        if msgs is None:
            out.append({"role": "user", "content": item.content})
        else:
            out.extend(msgs)
    return out


def render_messages(
    items: Sequence[ContextItem], query: str, style: Literal["openai", "anthropic"] = "openai"
) -> list[dict[str, Any]] | dict[str, Any]:
    system = render_text(items)
    tail = _history_messages(items)
    if query:
        tail.append({"role": "user", "content": query})
    if style == "openai":
        return [{"role": "system", "content": system}, *tail]
    if style == "anthropic":
        return {"system": system, "messages": tail}
    raise ValueError(f"unknown style: {style!r}")
