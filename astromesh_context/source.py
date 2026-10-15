"""Context sources: where candidate items come from."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from astromesh_context._text import content_hash
from astromesh_context.loop.messages import OpenAIAdapter, units
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


_ADAPTER = OpenAIAdapter()


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
        start = 0
        for unit in units(messages, _ADAPTER):
            items.append(
                ContextItem(
                    id=f"history:{start}",
                    content="\n".join(t for t in (_ADAPTER.text(m) for m in unit) if t),
                    source=self.name,
                    kind="history",
                    stability="session",
                    score=float(start),
                    order_key=f"{start:08d}",
                    meta={"messages": unit},
                )
            )
            start += len(unit)
        return items
