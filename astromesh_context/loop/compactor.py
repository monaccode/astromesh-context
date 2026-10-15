"""Compactor: summarize old turns without separating a tool call from its result."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from astromesh_context.loop.messages import M, MessageAdapter, OpenAIAdapter, units
from astromesh_context.tokens import CharEstimator, Tokenizer

log = logging.getLogger("astromesh_context")

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the conversation so far for the assistant that will continue it. "
    "Keep every decision, fact, number, identifier and open question. "
    "Drop greetings and repetition. "
    "If a previous summary is included, merge it; do not summarize the summary."
)


def summary_prompt(messages: list[Any], adapter: MessageAdapter) -> str:
    lines = [f"{adapter.role(m)}: {adapter.text(m)}" for m in messages]
    return DEFAULT_SUMMARY_PROMPT + "\n\n" + "\n".join(lines)


class Compactor:
    def __init__(
        self,
        summarize: Callable[[list[M]], Awaitable[str]],
        *,
        threshold_tokens: int,
        keep_last: int = 6,
        adapter: MessageAdapter = OpenAIAdapter(),  # noqa: B008
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.summarize = summarize
        self.threshold_tokens = threshold_tokens
        self.keep_last = keep_last
        self.adapter = adapter
        self.tokenizer = tokenizer or CharEstimator()

    def _tokens(self, messages: list[M]) -> int:
        return sum(self.tokenizer.count(self.adapter.text(m)) for m in messages)

    async def compact(self, messages: list[M]) -> list[M]:
        if self._tokens(messages) <= self.threshold_tokens:
            return messages
        adapter = self.adapter
        n_lead = 0
        while (
            n_lead < len(messages)
            and adapter.role(messages[n_lead]) == "system"
            and not adapter.is_summary(messages[n_lead])
        ):
            n_lead += 1
        lead = messages[:n_lead]
        rest = messages[n_lead:]
        previous = [m for m in rest if adapter.is_summary(m)]
        body = [m for m in rest if not adapter.is_summary(m)]

        grouped = units(body, adapter)
        kept = 0
        count = 0
        while kept < len(grouped) and count < self.keep_last:
            kept += 1
            count += len(grouped[-kept])
        if kept >= len(grouped):
            return messages
        aged = [m for u in grouped[: len(grouped) - kept] for m in u]
        tail = [m for u in grouped[len(grouped) - kept :] for m in u]
        try:
            text = await self.summarize([*previous, *aged])
        except Exception as exc:  # noqa: BLE001 - a failed summary must not break the loop
            log.warning("compaction skipped, summarize failed: %s", exc)
            return messages
        return [*lead, adapter.make_summary(text), *tail]
