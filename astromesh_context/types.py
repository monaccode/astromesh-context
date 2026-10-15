"""Data model shared by every component."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["instruction", "knowledge", "memory", "history", "tool_result"]
Stability = Literal["stable", "session", "volatile"]


@dataclass(frozen=True)
class Signals:
    trust: str | None = None
    stale: bool = False
    deprecated: bool = False
    citation: str | None = None


@dataclass(frozen=True)
class ContextItem:
    id: str
    content: str
    source: str
    kind: Kind
    stability: Stability
    score: float = 0.0
    tokens: int | None = None
    signals: Signals = Signals()
    pinned: bool = False
    order_key: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def with_tokens(self, n: int) -> ContextItem:
        return dataclasses.replace(self, tokens=n)


@dataclass(frozen=True)
class ScoredItem:
    item: ContextItem
    calibrated: float
    final: float
    adjustments: tuple[str, ...] = ()

    @property
    def tokens(self) -> int:
        return max(1, self.item.tokens or 0)


@dataclass(frozen=True)
class Budget:
    total: int
    caps: Mapping[str, int] = field(default_factory=dict)
    min_items: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError(f"budget total must be positive, got {self.total}")


@dataclass
class Selection:
    included: list[ScoredItem]
    excluded: list[tuple[ScoredItem, str]]


@dataclass(frozen=True)
class TraceEntry:
    item_id: str
    source: str
    raw_score: float
    calibrated: float
    final: float
    tokens: int
    included: bool
    reason: str | None
    adjustments: tuple[str, ...]


@dataclass
class Trace:
    entries: list[TraceEntry]
    events: list[str]

    def explain(self) -> str:
        rows = [("id", "source", "final", "tokens", "", "reason")]
        rows.extend(
            (
                e.item_id,
                e.source,
                f"{e.final:.3f}",
                str(e.tokens),
                "in" if e.included else "out",
                e.reason or "",
            )
            for e in self.entries
        )
        widths = [max(len(r[i]) for r in rows) for i in range(6)]
        lines = [
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in rows
        ]
        if self.events:
            lines.append("")
            lines.append("events:")
            lines.extend(f"  {ev}" for ev in self.events)
        return "\n".join(lines)

    def excluded(self, reason_prefix: str) -> list[TraceEntry]:
        return [
            e
            for e in self.entries
            if not e.included and e.reason is not None and e.reason.startswith(reason_prefix)
        ]


@dataclass
class BuildContext:
    query: str
    budget: Budget
    deadline: float
    vars: Mapping[str, Any]


@dataclass
class Window:
    query: str
    items: list[ContextItem]
    tokens_used: int
    budget: int
    trace: Trace
    stable_prefix_hash: str
