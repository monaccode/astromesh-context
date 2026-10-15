"""Expose a ContextEngine (and an Offloader's retrieval tools) over the Model Context Protocol.

Requires mcp 2.x (``mcp.server.mcpserver.MCPServer``, the renamed FastMCP).
"""

from __future__ import annotations

from astromesh_context.engine import ContextEngine
from astromesh_context.integrations._base import require
from astromesh_context.loop.offloader import Offloader


def serve(
    engine: ContextEngine, offloader: Offloader | None = None, name: str = "astromesh-context"
):
    """Build the server. ``ctx_read`` / ``ctx_grep`` are registered when an offloader is given."""
    server = require("mcp.server.mcpserver", "mcp").MCPServer(name)

    async def ctx_build(query: str) -> str:
        """Build the context window for a query: the rendered text, then why each item was kept."""
        window = await engine.build(query)
        return window.to_text() + "\n\n" + window.trace.explain()

    server.add_tool(ctx_build, name="ctx_build")
    if offloader is not None:
        for fn in offloader.tools():
            server.add_tool(fn, name=fn.__name__)
    return server
