# Extending astromesh-context

Four things are meant to be replaced: where candidates come from (`Source`), how they fill the
budget (`Policy`), how they are scored (`Scorer`) and how your framework's messages are read
(`MessageAdapter`). Each is a small protocol, so a plain class is enough. Every snippet on this
page is executed by `tests/test_docs_examples.py`.

## A `Source`

A source has a `name` and a `candidates(query, ctx)` method, sync or async, that returns a list
of `ContextItem`. It only proposes: it does not know the budget. Give each item a stable `id`,
a raw `score` (any scale, the engine calibrates it per source), and say how long the content
stays valid with `stability`: `stable` goes first in the prompt so the cache can reuse it,
`session` next, `volatile` last.

```python
from astromesh_context import ContextEngine, ContextItem, Signals

TICKETS = [
    ("T-1", "Customer reported a double charge on the Team plan.", 0.9),
    ("T-2", "Customer asked how to export invoices.", 0.4),
]


class TicketsSource:
    name = "tickets"

    async def candidates(self, query, ctx):
        return [
            ContextItem(
                id=f"tickets:{ticket_id}",
                content=text,
                source=self.name,
                kind="memory",
                stability="volatile",
                score=score,
                signals=Signals(citation=ticket_id),
            )
            for ticket_id, text, score in TICKETS
        ]


import asyncio

engine = ContextEngine([TicketsSource()], budget=200)
window = asyncio.run(engine.build("double charge"))
assert any(item.id == "tickets:T-1" for item in window.items)
print(window.trace.explain())
```

`Signals` is how a source tells the engine what it knows about an item: `trust`
(`"human-reviewed"` scores higher), `stale` (scores lower), `deprecated` (left out, and a copy
of a deprecated item found by another source is left out too) and `citation`.

A source that raises, or takes longer than its timeout, does not fail the build: the window is
built without it and the trace gets a `source_error:<name>:<message>` event. For a function
instead of a class, wrap it in `CallableSource(fn, name="...")`. An async `candidates` runs on the
event loop; a sync one runs in the default thread pool, and a sync source that hangs keeps its
thread after the timeout, so prefer async for anything that does I/O. A source whose items are
malformed (not a `ContextItem`, non-string `content`, a non-numeric `score`, negative
`tokens`) is dropped the same way, with `source_error:<name>:invalid item <index>: <why>`.

## A `Policy`

A policy turns scored candidates into a `Selection`: what is included, and what is excluded with
one of the reasons from the trace (`budget`, `source_cap`, `below_floor`, `deprecated`,
`share_exhausted`, `truncated`). It is a pure function, so the same candidates always give the
same window.

```python
from astromesh_context import ContextEngine, StaticSource
from astromesh_context.types import Selection


class BestScoreFirst:
    """Pinned items first, then highest final score until the budget is spent."""

    def select(self, items, budget):
        included, excluded, used = [], [], 0
        for scored in sorted(items, key=lambda s: (not s.item.pinned, -s.final)):
            if scored.item.signals.deprecated:
                excluded.append((scored, "deprecated"))
            elif used + scored.tokens <= budget.total:
                included.append(scored)
                used += scored.tokens
            else:
                excluded.append((scored, "budget"))
        return Selection(included, excluded)


engine = ContextEngine(
    [StaticSource("Always answer in Spanish.", pinned=True), TicketsSource()],
    policy=BestScoreFirst(),
    budget=30,
)
window = asyncio.run(engine.build("double charge"))
assert window.tokens_used <= 30
assert window.items[0].pinned
```

`budget.caps` and `budget.min_items` carry the per-source `max_tokens` and `min_items` from
`SourceConfig`; the built-in policies honor them, and yours may.

## A `Scorer`

A scorer runs after calibration and before deduplication. It receives every `ScoredItem` and
returns them with a new `final` score, recording what it did in `adjustments` so the trace can
show it. `LLMScorer` is one; this one boosts items that repeat a word of the query.

```python
import dataclasses
import re

from astromesh_context import ContextEngine


class KeywordBoost:
    def __init__(self, boost=1.0):
        self.boost = boost

    async def score(self, query, items):
        words = set(re.findall(r"\w+", query.lower()))
        out = []
        for scored in items:
            hits = words & set(re.findall(r"\w+", scored.item.content.lower()))
            if hits:
                scored = dataclasses.replace(
                    scored,
                    final=scored.final + self.boost,
                    adjustments=(*scored.adjustments, f"keywords={len(hits)}"),
                )
            out.append(scored)
        return out


engine = ContextEngine([TicketsSource()], scorer=KeywordBoost(), budget=200)
window = asyncio.run(engine.build("export invoices"))
best = max(window.trace.entries, key=lambda e: e.final)
assert best.item_id == "tickets:T-2"
assert any(a.startswith("keywords=") for a in best.adjustments)
```

To fall back instead of failing, raise `ScorerFallback("error" | "timeout" | "parse")`
(from `astromesh_context.scorer`): the engine builds the window on the calibrated scores and adds
a `scorer_fallback:<reason>` event to the trace.

## A `MessageAdapter`

`Compactor` and `Offloader` never touch your message type directly. They go through an adapter
with seven methods, which is how LangChain and PydanticAI messages work without being converted
to dicts. Implement it for your own type:

```python
from dataclasses import dataclass, field, replace

from astromesh_context.loop import Compactor


@dataclass(frozen=True)
class Msg:
    role: str
    text: str
    call_ids: tuple = ()
    result_id: str | None = None
    summary: bool = False


class MsgAdapter:
    def text(self, m):
        return m.text

    def role(self, m):
        return m.role

    def tool_call_ids(self, m):
        return list(m.call_ids)

    def tool_result_id(self, m):
        return m.result_id

    def is_summary(self, m):
        return m.summary

    def make_summary(self, text):
        return Msg("system", f"[summary] {text}", summary=True)

    def replace_tool_result(self, m, text):
        return replace(m, text=text)


async def summarize(old):
    return f"{len(old)} earlier messages"


history = [
    Msg("user", "first question " * 20),
    Msg("assistant", "", call_ids=("c1",)),
    Msg("tool", "lookup output " * 20, result_id="c1"),
    Msg("assistant", "first answer " * 20),
    Msg("user", "second question"),
    Msg("assistant", "second answer"),
]
compactor = Compactor(summarize, threshold_tokens=50, keep_last=2, adapter=MsgAdapter())
compacted = asyncio.run(compactor.compact(history))
assert compacted[0].summary
assert compacted[-2:] == history[-2:]
```

A tool call and its result are always kept or summarized together, so `tool_call_ids` and
`tool_result_id` are what make that possible: return the ids a message issues and the id it
answers, and the loop pieces do the rest.
