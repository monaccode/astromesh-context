"""ContextEngine: gather candidates, calibrate, dedupe, select, order, trace."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import math
import numbers
import time
from collections.abc import Mapping, Sequence
from typing import Any

from astromesh_context.calibrate import Calibrator
from astromesh_context.dedupe import dedupe
from astromesh_context.errors import ConfigError
from astromesh_context.order import order_items
from astromesh_context.policy import Policy, ScoredKnapsack
from astromesh_context.render import stable_prefix_hash
from astromesh_context.scorer import Scorer, ScorerFallback
from astromesh_context.source import HistorySource, Source, SourceConfig
from astromesh_context.tokens import CharEstimator, Tokenizer
from astromesh_context.types import Budget, BuildContext, ContextItem, Trace, TraceEntry, Window

_MODES = ("blend", "rank", "minmax", "zscore")


def _invalid(it: object) -> str | None:
    """Why a source's item cannot be used, or None when it is well formed."""
    if not isinstance(it, ContextItem):
        return f"not a ContextItem ({type(it).__name__})"
    if not isinstance(it.content, str):
        return f"content is not a str ({type(it.content).__name__})"
    if isinstance(it.score, bool) or not isinstance(it.score, numbers.Real):
        return f"score is not a number ({type(it.score).__name__})"
    if not math.isfinite(it.score):
        return f"score is not finite ({it.score})"
    if it.tokens is not None and (
        isinstance(it.tokens, bool) or not isinstance(it.tokens, int) or it.tokens < 0
    ):
        return f"tokens must be None or an int >= 0, got {it.tokens!r}"
    return None


class ContextEngine:
    def __init__(
        self,
        sources: Sequence[Source | SourceConfig],
        *,
        budget: int,
        policy: Policy | None = None,
        tokenizer: Tokenizer | None = None,
        calibrator: Calibrator | None = None,
        dedupe_threshold: float = 0.8,
        source_timeout: float = 5.0,
        scorer: Scorer | None = None,
    ) -> None:
        self.configs: list[SourceConfig] = []
        for s in sources:
            c = s if isinstance(s, SourceConfig) else SourceConfig(s)
            if c.min_items is None:
                c = dataclasses.replace(
                    c, min_items=2 if isinstance(c.source, HistorySource) else 0
                )
            if c.mode not in _MODES:
                raise ConfigError(
                    f"source {c.source.name!r}: mode must be one of {list(_MODES)}, got {c.mode!r}"
                )
            if any(other.source.name == c.source.name for other in self.configs):
                raise ConfigError(f"duplicate source name {c.source.name!r}")
            self.configs.append(c)
        self.budget_total = budget
        self.policy: Policy = policy or ScoredKnapsack()
        self.tokenizer: Tokenizer = tokenizer or CharEstimator()
        self.calibrator = calibrator or Calibrator()
        self.dedupe_threshold = dedupe_threshold
        self.source_timeout = source_timeout
        self.scorer = scorer
        self._budget = Budget(
            total=budget,
            caps={c.source.name: c.max_tokens for c in self.configs if c.max_tokens is not None},
            min_items={c.source.name: c.min_items for c in self.configs if c.min_items},
        )

    def _timeout(self, config: SourceConfig) -> float:
        return config.timeout if config.timeout is not None else self.source_timeout

    async def _gather_one(
        self, config: SourceConfig, query: str, ctx: BuildContext
    ) -> tuple[list[ContextItem], str | None]:
        name = config.source.name

        async def run() -> list[ContextItem]:
            fn = config.source.candidates
            if inspect.iscoroutinefunction(fn):
                result = await fn(query, ctx)
            else:
                result = await asyncio.to_thread(fn, query, ctx)
                if inspect.isawaitable(result):
                    result = await result
            return list(result)

        try:
            items = await asyncio.wait_for(run(), self._timeout(config))
            for index, it in enumerate(items):
                why = _invalid(it)
                if why is not None:
                    return [], f"source_error:{name}:invalid item {index}: {why}"
            return [
                it if it.source == name else dataclasses.replace(it, source=name) for it in items
            ], None
        except asyncio.TimeoutError:
            return [], f"source_error:{name}:timeout"
        except Exception as exc:  # noqa: BLE001 - any source failure degrades gracefully
            return [], f"source_error:{name}:{exc}"

    async def build(self, query: str, *, vars: Mapping[str, Any] | None = None) -> Window:
        """Build the window for ``query``.

        ``BuildContext.deadline`` handed to sources is ``time.monotonic()`` plus the largest
        effective source timeout (``SourceConfig.timeout`` or ``source_timeout``).
        """
        deadline = time.monotonic() + max(
            (self._timeout(c) for c in self.configs), default=self.source_timeout
        )
        ctx = BuildContext(query=query, budget=self._budget, deadline=deadline, vars=vars or {})
        results = await asyncio.gather(*(self._gather_one(c, query, ctx) for c in self.configs))

        events: list[str] = []
        candidates: list[ContextItem] = []
        for items, error in results:
            if error is not None:
                events.append(error)
            candidates.extend(items)

        candidates = [
            it if it.tokens is not None else it.with_tokens(self.tokenizer.count(it.content))
            for it in candidates
        ]
        scored = self.calibrator.calibrate(
            candidates,
            weights={c.source.name: c.weight for c in self.configs},
            modes={c.source.name: c.mode for c in self.configs},
        )
        if self.scorer is not None:
            try:
                scored = await self.scorer.score(query, scored)
            except ScorerFallback as e:
                events.append(f"scorer_fallback:{e.reason}")
        kept, dropped = dedupe(scored, self.dedupe_threshold)
        selection = self.policy.select(kept, self._budget)

        reasons: dict[int, str] = {id(s): r for s, r in dropped}
        merged = {s.item.id: s for s in kept}  # kept items carry the signals they inherited
        reasons.update({id(s): r for s, r in selection.excluded})
        included_ids = {id(s) for s in selection.included}
        shown = [s if id(s) in reasons else merged.get(s.item.id, s) for s in scored]
        entries = [
            TraceEntry(
                item_id=s.item.id,
                source=s.item.source,
                raw_score=s.item.score,
                calibrated=s.calibrated,
                final=s.final,
                tokens=s.tokens,
                included=id(s) in included_ids,
                reason=None if id(s) in included_ids else reasons.get(id(s), "budget"),
                adjustments=s.adjustments,
            )
            for s in shown
        ]
        items = order_items(selection.included, [c.source.name for c in self.configs])
        return Window(
            query=query,
            items=items,
            tokens_used=sum(s.tokens for s in selection.included),
            budget=self.budget_total,
            trace=Trace(entries, events),
            stable_prefix_hash=stable_prefix_hash(items),
        )
