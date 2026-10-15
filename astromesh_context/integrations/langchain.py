"""LangChain 1.x: an ``AgentMiddleware`` for ``langchain.agents.create_agent``.

Hooks used (langchain 1.x): ``before_model`` / ``abefore_model`` to offload and compact the state
messages, and ``wrap_model_call`` / ``awrap_model_call`` to put the engine window in the system
message of the model request. State is replaced through the hook's return value
(``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` followed by the new messages), the supported mechanism.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

from astromesh_context.integrations._base import require, with_adapter

_middleware = require("langchain.agents.middleware", "langchain")
_messages = require("langchain_core.messages", "langchain")
_tools = require("langchain_core.tools", "langchain")
_message_graph = require("langgraph.graph.message", "langchain")

from astromesh_context.engine import ContextEngine  # noqa: E402
from astromesh_context.loop.compactor import Compactor  # noqa: E402
from astromesh_context.loop.messages import SUMMARY_NAME, _content_parts, _join_text  # noqa: E402
from astromesh_context.loop.offloader import Offloader  # noqa: E402

AgentMiddleware = _middleware.AgentMiddleware
BaseMessage = _messages.BaseMessage
SystemMessage = _messages.SystemMessage
RemoveMessage = _messages.RemoveMessage
REMOVE_ALL_MESSAGES = _message_graph.REMOVE_ALL_MESSAGES

_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


class LangChainAdapter:
    """MessageAdapter over langchain ``BaseMessage`` objects."""

    def text(self, m: BaseMessage) -> str:
        parts = _content_parts(m.content)
        for call in getattr(m, "tool_calls", None) or []:
            parts.append(f"{call.get('name', '')}({json.dumps(call.get('args', {}))})")
        return _join_text(parts)

    def role(self, m: BaseMessage) -> str:
        return _ROLES.get(m.type, m.type)

    def tool_call_ids(self, m: BaseMessage) -> list[str]:
        if m.type != "ai":
            return []
        return [c["id"] for c in getattr(m, "tool_calls", None) or [] if c.get("id")]

    def tool_result_id(self, m: BaseMessage) -> str | None:
        if m.type != "tool":
            return None
        return getattr(m, "tool_call_id", None) or None

    def is_summary(self, m: BaseMessage) -> bool:
        return m.type == "system" and m.name == SUMMARY_NAME

    def make_summary(self, text: str) -> BaseMessage:
        return SystemMessage(content=text, name=SUMMARY_NAME)

    def replace_tool_result(self, m: BaseMessage, text: str) -> BaseMessage:
        return m.model_copy(update={"content": text})


def _run_sync(coro: Awaitable[Any]) -> Any:
    """Run a coroutine from sync code, in a worker thread when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _last_human_text(messages: list[BaseMessage]) -> str | None:
    for m in reversed(messages):
        if m.type == "human":
            return _join_text(_content_parts(m.content))
    return None


class ContextMiddleware(AgentMiddleware):
    """Offload and compact the thread before each model call; optionally build the system prompt.

    ``compactor`` and ``offloader`` are used with a ``LangChainAdapter`` (copies are made, so the
    objects you pass in are not modified and an offloader keeps sharing its store with
    ``offload_tools``). When ``engine`` is set, the window built from the last human message
    is appended to the request's system message after a blank line (it is the whole system
    message when there is none); with no human message the engine is skipped.

    Use ``ainvoke`` / ``astream`` in async code. The sync hooks run the coroutines on a worker
    thread with its own event loop, so resources bound to the caller's loop will not work there.
    """

    def __init__(
        self,
        engine: ContextEngine | None = None,
        *,
        compactor: Compactor | None = None,
        offloader: Offloader | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        adapter = LangChainAdapter()
        self.compactor = with_adapter(compactor, adapter)
        self.offloader = with_adapter(offloader, adapter)

    @staticmethod
    def _update(old: list[BaseMessage], new: list[BaseMessage]) -> dict[str, Any] | None:
        if len(new) == len(old) and all(a is b for a, b in zip(new, old, strict=True)):
            return None
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new]}

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        old = list(state["messages"])
        new = self.offloader.offload(old) if self.offloader else old
        if self.compactor:
            new = _run_sync(self.compactor.compact(new))
        return self._update(old, new)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        old = list(state["messages"])
        new = self.offloader.offload(old) if self.offloader else old
        if self.compactor:
            new = await self.compactor.compact(new)
        return self._update(old, new)

    @staticmethod
    def _with_window(request: Any, window_text: str) -> Any:
        if not window_text:
            return request
        current = request.system_prompt
        text = "\n\n".join([current, window_text]) if current else window_text
        return request.override(system_message=SystemMessage(content=text))

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        query = _last_human_text(request.messages) if self.engine else None
        if query is None:
            return handler(request)
        window = _run_sync(self.engine.build(query))
        return handler(self._with_window(request, window.to_text()))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        query = _last_human_text(request.messages) if self.engine else None
        if query is None:
            return await handler(request)
        window = await self.engine.build(query)
        return await handler(self._with_window(request, window.to_text()))


def offload_tools(offloader: Offloader) -> list:
    """``ctx_read`` and ``ctx_grep`` as LangChain tools over the offloader's store."""
    return [
        _tools.StructuredTool.from_function(
            func=fn, name=fn.__name__, description=inspect.getdoc(fn) or ""
        )
        for fn in offloader.tools()
    ]
