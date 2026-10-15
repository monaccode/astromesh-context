"""Offloader: move oversized tool results out of the conversation into a store."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from astromesh_context.loop.messages import M, MessageAdapter, OpenAIAdapter
from astromesh_context.tokens import CharEstimator, Tokenizer

_HANDLE_RE = re.compile(r"^ctx_[0-9a-f]{16}$")


class Store(Protocol):
    def put(self, text: str) -> str: ...

    def get(self, handle: str) -> str | None: ...


def _handle(text: str) -> str:
    return "ctx_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class MemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def put(self, text: str) -> str:
        handle = _handle(text)
        self._items[handle] = text
        return handle

    def get(self, handle: str) -> str | None:
        return self._items.get(handle)


class DirStore:
    """Stores each text as `<path>/<handle>.txt` (UTF-8)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def put(self, text: str) -> str:
        handle = _handle(text)
        file = self.path / f"{handle}.txt"
        if not file.exists():
            file.write_bytes(text.encode("utf-8"))
        return handle

    def get(self, handle: str) -> str | None:
        if not _HANDLE_RE.match(handle):
            return None
        file = self.path / f"{handle}.txt"
        if not file.is_file():
            return None
        return file.read_bytes().decode("utf-8")


def ctx_read(store: Store, handle: str, offset: int = 0, limit: int = 2000) -> str:
    """Return `limit` characters of the stored text starting at character `offset`."""
    text = store.get(handle)
    if text is None:
        return f"error: unknown handle {handle}"
    if offset < 0 or limit < 0:
        return "error: offset and limit must not be negative"
    if offset > len(text) or (offset == len(text) and len(text) > 0):
        return f"error: offset {offset} is past the end ({len(text)} characters)"
    return text[offset : offset + limit]


def ctx_grep(
    store: Store, handle: str, pattern: str, context: int = 2, max_matches: int = 20
) -> str:
    """Search the stored text line by line with a regular expression."""
    text = store.get(handle)
    if text is None:
        return f"error: unknown handle {handle}"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: invalid pattern: {e}"
    context = max(context, 0)
    lines = text.split("\n")
    matches = [i for i, line in enumerate(lines) if rx.search(line)][: max(max_matches, 0)]
    if not matches:
        return "no matches"
    ranges: list[list[int]] = []
    for i in matches:
        lo, hi = max(0, i - context), min(len(lines) - 1, i + context)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], hi)
        else:
            ranges.append([lo, hi])
    blocks = ["\n".join(f"{n + 1}: {lines[n]}" for n in range(lo, hi + 1)) for lo, hi in ranges]
    return "\n--\n".join(blocks)


class Offloader:
    """Replace tool results over `max_tokens` with a preview and a retrieval handle.

    Adapters that expose `tool_result_texts` / `replace_tool_result_texts` (the Anthropic one)
    are offloaded per tool_result block, so each block keeps its own id, preview and handle.
    """

    def __init__(
        self,
        store: Store | None = None,
        *,
        max_tokens: int = 1500,
        preview_tokens: int = 200,
        adapter: MessageAdapter = OpenAIAdapter(),  # noqa: B008
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.store: Store = store if store is not None else MemoryStore()
        self.max_tokens = max_tokens
        self.preview_tokens = preview_tokens
        self.adapter = adapter
        self.tokenizer = tokenizer or CharEstimator()

    def _shrink(self, text: str) -> str:
        if self.tokenizer.count(text) <= self.max_tokens:
            return text
        handle = self.store.put(text)
        preview = text[: self.preview_tokens * 4]
        return (
            f"{preview}\n\n[truncated: {len(text)} characters stored as {handle}. "
            "Use ctx_read(handle, offset, limit) or ctx_grep(handle, pattern) to see more.]"
        )

    def _offload_one(self, m: M) -> M:
        adapter: Any = self.adapter
        if adapter.tool_result_id(m) is None:
            return m
        if hasattr(adapter, "tool_result_texts"):
            texts = adapter.tool_result_texts(m)
            new = [self._shrink(t) for t in texts]
            if new == texts:
                return m
            return adapter.replace_tool_result_texts(m, new)
        text = adapter.text(m)
        new_text = self._shrink(text)
        return m if new_text == text else adapter.replace_tool_result(m, new_text)

    def offload(self, messages: list[M]) -> list[M]:
        return [self._offload_one(m) for m in messages]

    def tools(self) -> list[Callable[..., str]]:
        store = self.store

        def ctx_read_tool(handle: str, offset: int = 0, limit: int = 2000) -> str:
            """Read part of a stored tool result.

            Args:
                handle: the ctx_... handle named in the truncation notice.
                offset: first character to read (0-based).
                limit: maximum number of characters to return.
            """
            return ctx_read(store, handle, offset, limit)

        def ctx_grep_tool(
            handle: str, pattern: str, context: int = 2, max_matches: int = 20
        ) -> str:
            """Search a stored tool result line by line with a regular expression.

            Args:
                handle: the ctx_... handle named in the truncation notice.
                pattern: regular expression, case-sensitive.
                context: lines of context to show around each match.
                max_matches: stop after this many matching lines.
            """
            return ctx_grep(store, handle, pattern, context, max_matches)

        ctx_read_tool.__name__ = ctx_read_tool.__qualname__ = "ctx_read"
        ctx_grep_tool.__name__ = ctx_grep_tool.__qualname__ = "ctx_grep"
        return [ctx_read_tool, ctx_grep_tool]
