import pytest

pytest.importorskip("mcp")

from astromesh_context import ContextEngine, StaticSource
from astromesh_context.integrations.mcp import serve
from astromesh_context.loop import Offloader


def text(result) -> str:
    return result.content[0].text


async def test_ctx_build_returns_window_and_trace():
    engine = ContextEngine([StaticSource("PINNED-RULE-42", pinned=True)], budget=500)
    server = serve(engine, Offloader())
    names = sorted(t.name for t in await server.list_tools())
    assert names == ["ctx_build", "ctx_grep", "ctx_read"]
    out = text(await server.call_tool("ctx_build", {"query": "hello"}))
    window = await engine.build("hello")
    assert out == window.to_text() + "\n\n" + window.trace.explain()
    assert "PINNED-RULE-42" in out


async def test_ctx_read_and_grep_return_stored_content():
    off = Offloader()
    handle = off.store.put("alpha\nbeta needle\ngamma")
    server = serve(ContextEngine([], budget=100), off)
    assert text(await server.call_tool("ctx_read", {"handle": handle, "limit": 5})) == "alpha"
    assert "2: beta needle" in text(
        await server.call_tool("ctx_grep", {"handle": handle, "pattern": "needle"})
    )


async def test_without_offloader_only_ctx_build():
    server = serve(ContextEngine([], budget=100))
    assert [t.name for t in await server.list_tools()] == ["ctx_build"]
