"""Optional model-based rescoring. Every failure degrades to ``ScorerFallback``."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol

from astromesh_context.errors import MissingExtraError
from astromesh_context.types import ScoredItem

_PROMPT = (
    "Rate how useful each numbered snippet is for answering the question. "
    "Reply with only a JSON array of numbers from 0 to 10, one per snippet, in order."
    "\n\nQuestion: {query}\n\n{snippets}"
)
_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


class Scorer(Protocol):
    async def score(self, query: str, items: Sequence[ScoredItem]) -> list[ScoredItem]: ...


class ScorerFallback(Exception):  # noqa: N818 - name fixed by the public contract
    def __init__(self, reason: Literal["error", "timeout", "parse"]) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _parse_scores(text: str, expected: int) -> list[float]:
    for match in _ARRAY.finditer(text):
        try:
            value = json.loads(match.group(0))
        except ValueError:
            continue
        if isinstance(value, list) and all(_is_number(v) for v in value):
            if len(value) != expected:
                raise ScorerFallback("parse")
            return [min(10.0, max(0.0, float(v))) for v in value]
    raise ScorerFallback("parse")


class LLMScorer:
    def __init__(
        self,
        complete: Callable[[str], Awaitable[str]],
        *,
        top_n: int = 30,
        alpha: float = 0.5,
        timeout: float = 10.0,
        max_chars: int = 800,
    ) -> None:
        self.complete = complete
        self.top_n = top_n
        self.alpha = alpha
        self.timeout = timeout
        self.max_chars = max_chars

    async def score(self, query: str, items: Sequence[ScoredItem]) -> list[ScoredItem]:
        result = list(items)
        top = sorted((s for s in result if not s.item.pinned), key=lambda s: s.final, reverse=True)[
            : self.top_n
        ]
        if not top:
            return result
        snippets = "\n\n".join(
            f"[{i}] {s.item.content[: self.max_chars]}" for i, s in enumerate(top, start=1)
        )
        prompt = _PROMPT.format(query=query, snippets=snippets)
        try:
            text = await asyncio.wait_for(self.complete(prompt), self.timeout)
        except asyncio.TimeoutError as exc:
            raise ScorerFallback("timeout") from exc
        except Exception as exc:
            raise ScorerFallback("error") from exc
        if not isinstance(text, str):
            raise ScorerFallback("parse")
        scores = _parse_scores(text, len(top))
        rescored = {
            id(s): dataclasses.replace(
                s,
                final=self.alpha * v / 10 + (1 - self.alpha) * s.calibrated,
                adjustments=(*s.adjustments, f"llm={v:g}"),
            )
            for s, v in zip(top, scores, strict=True)
        }
        return [rescored.get(id(s), s) for s in result]


def openai_compatible(
    base_url: str, api_key: str, model: str, *, timeout: float = 30.0
) -> Callable[[str], Awaitable[str]]:
    try:
        import httpx
    except ImportError as exc:
        raise MissingExtraError("install astromesh-context[llm]") from exc

    url = base_url.rstrip("/") + "/chat/completions"

    async def complete(prompt: str) -> str:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    return complete
