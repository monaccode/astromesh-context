"""Message adapters: let loop helpers work on OpenAI-style and Anthropic-style histories."""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar

M = TypeVar("M")

SUMMARY_NAME = "context_summary"
SUMMARY_MARKER = "[context summary]"


class MessageAdapter(Protocol[M]):
    def text(self, m: M) -> str: ...

    def role(self, m: M) -> str: ...

    def tool_call_ids(self, m: M) -> list[str]: ...

    def tool_result_id(self, m: M) -> str | None: ...

    def is_summary(self, m: M) -> bool: ...

    def make_summary(self, text: str) -> M: ...

    def replace_tool_result(self, m: M, text: str) -> M: ...


def _join_text(parts: list[str]) -> str:
    return "\n".join(p for p in parts if p)


def _content_parts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
            if not isinstance(p, dict) or "text" in p
        ]
    return []


class OpenAIAdapter:
    """Chat-completions dict messages."""

    def text(self, m: dict) -> str:
        parts = _content_parts(m.get("content"))
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            if fn:
                parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
        return _join_text(parts)

    def role(self, m: dict) -> str:
        return str(m.get("role", ""))

    def tool_call_ids(self, m: dict) -> list[str]:
        if m.get("role") != "assistant":
            return []
        return [c["id"] for c in m.get("tool_calls") or [] if isinstance(c, dict) and c.get("id")]

    def tool_result_id(self, m: dict) -> str | None:
        if m.get("role") != "tool":
            return None
        return m.get("tool_call_id") or None

    def is_summary(self, m: dict) -> bool:
        return m.get("name") == SUMMARY_NAME

    def make_summary(self, text: str) -> dict:
        return {"role": "system", "name": SUMMARY_NAME, "content": text}

    def replace_tool_result(self, m: dict, text: str) -> dict:
        return {**m, "content": text}


class AnthropicAdapter:
    """Messages-API dict messages with content blocks."""

    @staticmethod
    def _blocks(m: dict) -> list[Any]:
        content = m.get("content")
        return content if isinstance(content, list) else []

    def text(self, m: dict) -> str:
        content = m.get("content")
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for b in self._blocks(m):
            if not isinstance(b, dict):
                parts.append(str(b))
            elif b.get("type") == "tool_use":
                parts.append(f"{b.get('name', '')}({json.dumps(b.get('input', {}))})")
            elif b.get("type") == "tool_result":
                parts.extend(_content_parts(b.get("content")))
            elif "text" in b:
                parts.append(b["text"])
        return _join_text(parts)

    def role(self, m: dict) -> str:
        return str(m.get("role", ""))

    def tool_call_ids(self, m: dict) -> list[str]:
        if m.get("role") != "assistant":
            return []
        return [
            b["id"]
            for b in self._blocks(m)
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        ]

    def tool_result_id(self, m: dict) -> str | None:
        """The first tool_result's id; one user message may answer several calls."""
        if m.get("role") != "user":
            return None
        for b in self._blocks(m):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                return b.get("tool_use_id") or None
        return None

    def is_summary(self, m: dict) -> bool:
        if m.get("role") != "user":
            return False
        for b in self._blocks(m):
            if isinstance(b, dict) and b.get("type") == "text":
                return str(b.get("text", "")).startswith(SUMMARY_MARKER)
            break
        return False

    def make_summary(self, text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "text", "text": f"{SUMMARY_MARKER}\n{text}"}],
        }

    def replace_tool_result(self, m: dict, text: str) -> dict:
        blocks = [
            {**b, "content": text} if isinstance(b, dict) and b.get("type") == "tool_result" else b
            for b in self._blocks(m)
        ]
        return {**m, "content": blocks}


def units(messages: list[M], adapter: MessageAdapter[M]) -> list[list[M]]:
    """Group messages so a tool call is never separated from its results.

    A unit is one message, or an assistant message with tool calls followed by the messages
    that answer them. Orphans (a call with no result, a result with no call) stand alone.
    Calls and results without an id declare and match nothing.
    """
    out: list[list[M]] = []
    i = 0
    while i < len(messages):
        start = i
        declared = set(adapter.tool_call_ids(messages[i]))
        i += 1
        if declared:
            while i < len(messages) and adapter.tool_result_id(messages[i]) in declared:
                i += 1
        out.append(list(messages[start:i]))
    return out
