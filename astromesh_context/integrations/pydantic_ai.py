"""PydanticAI: a history processor (offload, then compact) and ``Tool`` objects for the agent.

``history_processor`` returns a ``(list[ModelMessage]) -> Awaitable[list[ModelMessage]]`` callable.
In pydantic-ai 2.x register it with ``Agent(capabilities=[ProcessHistory(processor)])``
(``pydantic_ai.capabilities.ProcessHistory``); 1.x took ``history_processors=[processor]``.
Messages are handled as pydantic-ai objects through ``PydanticAIAdapter``; tool call ids and tool
names are preserved.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from astromesh_context.integrations._base import require, with_adapter

_pai = require("pydantic_ai", "pydantic-ai")
_msgs = require("pydantic_ai.messages", "pydantic-ai")

from astromesh_context.loop.compactor import Compactor  # noqa: E402
from astromesh_context.loop.messages import SUMMARY_MARKER, _content_parts, _join_text  # noqa: E402
from astromesh_context.loop.offloader import Offloader  # noqa: E402

ModelRequest = _msgs.ModelRequest
ModelResponse = _msgs.ModelResponse
SystemPromptPart = _msgs.SystemPromptPart
UserPromptPart = _msgs.UserPromptPart
TextPart = _msgs.TextPart
ToolCallPart = _msgs.ToolCallPart
ToolReturnPart = _msgs.ToolReturnPart
RetryPromptPart = _msgs.RetryPromptPart


def _part_text(part: Any) -> str:
    if isinstance(part, (SystemPromptPart, TextPart)):
        return part.content
    if isinstance(part, UserPromptPart):
        content = part.content
        return content if isinstance(content, str) else _join_text(_content_parts(list(content)))
    if isinstance(part, ToolReturnPart):
        return part.model_response_str()
    if isinstance(part, RetryPromptPart):
        return part.model_response()
    if isinstance(part, ToolCallPart):
        return f"{part.tool_name}({part.args_as_json_str()})"
    return ""


class PydanticAIAdapter:
    """MessageAdapter over ``ModelRequest`` / ``ModelResponse`` objects."""

    def text(self, m: Any) -> str:
        return _join_text([_part_text(p) for p in m.parts])

    def role(self, m: Any) -> str:
        if isinstance(m, ModelResponse):
            return "assistant"
        if any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for p in m.parts):
            return "tool"
        if m.parts and all(isinstance(p, SystemPromptPart) for p in m.parts):
            return "system"
        return "user"

    def tool_call_ids(self, m: Any) -> list[str]:
        if not isinstance(m, ModelResponse):
            return []
        return [p.tool_call_id for p in m.parts if isinstance(p, ToolCallPart) and p.tool_call_id]

    def tool_result_id(self, m: Any) -> str | None:
        """The first tool result's id; one request may answer several calls."""
        if not isinstance(m, ModelRequest):
            return None
        for p in m.parts:
            if isinstance(p, ToolReturnPart) or (
                isinstance(p, RetryPromptPart) and p.tool_name is not None
            ):
                return p.tool_call_id or None
        return None

    def is_summary(self, m: Any) -> bool:
        return isinstance(m, ModelRequest) and bool(m.parts) and all(map(_is_summary_part, m.parts))

    def make_summary(self, text: str) -> Any:
        return ModelRequest(parts=[SystemPromptPart(content=f"{SUMMARY_MARKER}\n{text}")])

    def replace_tool_result(self, m: Any, text: str) -> Any:
        done = False
        parts = []
        for p in m.parts:
            if not done and isinstance(p, ToolReturnPart):
                parts.append(dataclasses.replace(p, content=text))
                done = True
            else:
                parts.append(p)
        return dataclasses.replace(m, parts=parts)

    def tool_result_texts(self, m: Any) -> list[str]:
        """The text of each tool return part, in order (for per-part offloading)."""
        return [_part_text(p) for p in m.parts if isinstance(p, ToolReturnPart)]

    def replace_tool_result_texts(self, m: Any, texts: list[str]) -> Any:
        """Replace each tool return part whose text changed; ids and tool names are kept."""
        it = iter(texts)
        parts = []
        for p in m.parts:
            if isinstance(p, ToolReturnPart):
                new = next(it)
                parts.append(dataclasses.replace(p, content=new) if new != _part_text(p) else p)
            else:
                parts.append(p)
        return dataclasses.replace(m, parts=parts)


def _is_summary_part(part: Any) -> bool:
    return isinstance(part, SystemPromptPart) and part.content.startswith(SUMMARY_MARKER)


def _split_leading_system(messages: list[Any]) -> list[Any]:
    """Split the leading request into runs of system prompt, summary and other parts.

    PydanticAI puts the system prompt in the same request as the first user prompt, and merges
    adjacent requests (system prompt + summary) into one on later runs. Without the split the
    compactor would age the system prompt into the summary, or fail to recognize a summary.
    """
    if not messages or not isinstance(messages[0], ModelRequest):
        return messages
    first = messages[0]
    runs: list[list[Any]] = []
    kinds: list[str] = []
    for part in first.parts:
        if _is_summary_part(part):
            kind = "summary"
        elif isinstance(part, SystemPromptPart):
            kind = "system"
        else:
            kind = "other"
        if kinds and kinds[-1] == kind:
            runs[-1].append(part)
        else:
            runs.append([part])
            kinds.append(kind)
    if len(runs) == 1:
        return messages
    return [*(dataclasses.replace(first, parts=run) for run in runs), *messages[1:]]


def history_processor(
    *, compactor: Compactor | None = None, offloader: Offloader | None = None
) -> Callable[[list[Any]], Awaitable[list[Any]]]:
    """An async history processor: offload large tool results, then compact old turns.

    The compactor and offloader are copied with a ``PydanticAIAdapter``; the offloader's store
    stays shared with ``offload_tools``.
    """
    adapter = PydanticAIAdapter()
    comp = with_adapter(compactor, adapter)
    off = with_adapter(offloader, adapter)

    async def process(messages: list[Any]) -> list[Any]:
        out = off.offload(list(messages)) if off else list(messages)
        if comp:
            out = await comp.compact(_split_leading_system(out))
        return out

    return process


def offload_tools(offloader: Offloader) -> list:
    """``ctx_read`` and ``ctx_grep`` as PydanticAI ``Tool`` objects over the offloader's store."""
    return [
        _pai.Tool(fn, name=fn.__name__, description=inspect.getdoc(fn) or None)
        for fn in offloader.tools()
    ]
