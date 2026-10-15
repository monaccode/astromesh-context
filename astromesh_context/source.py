"""Context sources: where candidate items come from."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from astromesh_context._text import content_hash
from astromesh_context.types import BuildContext, ContextItem, Kind, Stability


class Source(Protocol):
    name: str

    def candidates(
        self, query: str, ctx: BuildContext
    ) -> list[ContextItem] | Awaitable[list[ContextItem]]: ...


@dataclass
class SourceConfig:
    source: Source
    weight: float = 1.0
    max_tokens: int | None = None
    min_items: int = 0
    timeout: float | None = None
    mode: str = "blend"


class StaticSource:
    """A fixed piece of text, e.g. the system prompt."""

    def __init__(
        self,
        text: str,
        *,
        name: str = "static",
        kind: Kind = "instruction",
        stability: Stability = "stable",
        pinned: bool = False,
        score: float = 1.0,
    ) -> None:
        self.name = name
        self._item = ContextItem(
            id=f"static:{content_hash(text)[:12]}",
            content=text,
            source=name,
            kind=kind,
            stability=stability,
            score=score,
            pinned=pinned,
        )

    def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]:
        return [self._item]


class CallableSource:
    """Wraps a sync or async function returning candidate items."""

    def __init__(
        self,
        fn: Callable[[str, BuildContext], list[ContextItem] | Awaitable[list[ContextItem]]],
        *,
        name: str,
    ) -> None:
        self.name = name
        self._fn = fn

    def candidates(
        self, query: str, ctx: BuildContext
    ) -> list[ContextItem] | Awaitable[list[ContextItem]]:
        return self._fn(query, ctx)


def _text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
            if not isinstance(p, dict) or "text" in p
        ]
    else:
        parts = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        if fn:
            parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    return "\n".join(p for p in parts if p)


class HistorySource:
    """Chat history, one item per unit (a message, or a tool call with its results)."""

    def __init__(
        self,
        messages: list[dict] | None = None,
        *,
        key: str = "history",
        name: str = "history",
    ) -> None:
        self.name = name
        self._messages = messages
        self._key = key

    def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]:
        messages = self._messages if self._messages is not None else ctx.vars.get(self._key, [])
        items: list[ContextItem] = []
        i = 0
        while i < len(messages):
            start = i
            msg = messages[i]
            i += 1
            calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
            if calls:
                declared = {c.get("id") for c in calls if isinstance(c, dict)}
                while (
                    i < len(messages)
                    and messages[i].get("role") == "tool"
                    and messages[i].get("tool_call_id") in declared
                ):
                    i += 1
            unit = list(messages[start:i])
            items.append(
                ContextItem(
                    id=f"history:{start}",
                    content="\n".join(t for t in (_text(m) for m in unit) if t),
                    source=self.name,
                    kind="history",
                    stability="session",
                    score=float(start),
                    order_key=f"{start:08d}",
                    meta={"messages": unit},
                )
            )
        return items
