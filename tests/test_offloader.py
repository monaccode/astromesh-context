from astromesh_context.loop import DirStore, MemoryStore, Offloader
from astromesh_context.loop.messages import AnthropicAdapter
from astromesh_context.loop.offloader import ctx_grep, ctx_read


def _tool_msg(content: str, call_id: str = "c1") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_small_results_untouched():
    msgs = [{"role": "user", "content": "hi"}, _tool_msg("short")]
    out = Offloader(MemoryStore()).offload(msgs)
    assert out == msgs
    assert out is not msgs


def test_large_result_replaced_with_preview_and_handle():
    store = MemoryStore()
    big = "x" * 10_000
    off = Offloader(store, max_tokens=1500, preview_tokens=200)
    msgs = [{"role": "user", "content": big}, _tool_msg(big)]
    out = off.offload(msgs)
    assert out[0] == msgs[0]
    handle = store.put(big)
    expected = (
        "x" * 800 + f"\n\n[truncated: 10000 characters stored as {handle}. "
        "Use ctx_read(handle, offset, limit) or ctx_grep(handle, pattern) to see more.]"
    )
    assert out[1] == {"role": "tool", "tool_call_id": "c1", "content": expected}
    assert msgs[1]["content"] == big


def test_round_trip_read_full_content():
    store = MemoryStore()
    text = "".join(f"line {i}\n" for i in range(2000))
    handle = store.put(text)
    pages, offset = [], 0
    while True:
        page = ctx_read(store, handle, offset, 2000)
        if not page:
            break
        pages.append(page)
        offset += 2000
        if offset >= len(text):
            break
    assert "".join(pages) == text
    assert ctx_read(store, handle, len(text) + 5).startswith("error: offset")
    assert ctx_read(store, handle, -1).startswith("error:")
    assert ctx_read(store, handle, 0, -1).startswith("error:")


def test_grep_context_and_limit():
    store = MemoryStore()
    lines = [f"row {i}" for i in range(1, 31)]
    lines[9] = "row 10 NEEDLE"
    lines[11] = "row 12 NEEDLE"
    lines[24] = "row 25 NEEDLE"
    handle = store.put("\n".join(lines))
    out = ctx_grep(store, handle, "NEEDLE", context=1)
    assert out == "\n--\n".join(
        [
            "9: row 9\n10: row 10 NEEDLE\n11: row 11\n12: row 12 NEEDLE\n13: row 13",
            "24: row 24\n25: row 25 NEEDLE\n26: row 26",
        ]
    )
    limited = ctx_grep(store, handle, "NEEDLE", context=0, max_matches=2)
    assert limited == "10: row 10 NEEDLE\n--\n12: row 12 NEEDLE"
    assert ctx_grep(store, handle, "needle") == "no matches"


def test_unknown_handle_and_bad_regex_return_errors():
    store = MemoryStore()
    assert ctx_read(store, "ctx_0000000000000000") == "error: unknown handle ctx_0000000000000000"
    assert ctx_grep(store, "nope", "a") == "error: unknown handle nope"
    handle = store.put("abc")
    assert ctx_grep(store, handle, "(").startswith("error: invalid pattern: ")


def test_dirstore_dedupes_by_content(tmp_path):
    d = tmp_path / "store" / "sub"
    store = DirStore(d)
    h1 = store.put("héllo")
    h2 = store.put("héllo")
    assert h1 == h2
    assert h1.startswith("ctx_")
    assert len(h1) == 20
    assert [p.name for p in d.iterdir()] == [f"{h1}.txt"]
    assert (d / f"{h1}.txt").read_bytes() == "héllo".encode()
    assert store.get(h1) == "héllo"
    assert store.get("ctx_0000000000000000") is None
    assert store.get("../../etc/passwd") is None


def test_tools_are_plain_functions_bound_to_store():
    off = Offloader(MemoryStore())
    read, grep = off.tools()
    assert (read.__name__, grep.__name__) == ("ctx_read", "ctx_grep")
    assert read.__doc__
    assert grep.__doc__
    handle = off.store.put("alpha\nbeta")
    assert read(handle) == "alpha\nbeta"
    assert grep(handle, "bet", context=0) == "2: beta"


def test_anthropic_tool_result_blocks():
    store = MemoryStore()
    off = Offloader(store, max_tokens=10, preview_tokens=2, adapter=AnthropicAdapter())
    a, b = "A" * 100, "B" * 100
    msg = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": a},
            {"type": "tool_result", "tool_use_id": "t2", "content": "ok"},
            {"type": "tool_result", "tool_use_id": "t3", "content": [{"type": "text", "text": b}]},
        ],
    }
    out = off.offload([msg])[0]
    blocks = out["content"]
    assert [x["tool_use_id"] for x in blocks] == ["t1", "t2", "t3"]
    assert blocks[0]["content"].startswith("A" * 8 + "\n\n[truncated: 100 characters stored as ")
    assert blocks[1]["content"] == "ok"
    assert blocks[2]["content"].startswith("B" * 8 + "\n\n[truncated: 100 characters")
    h1 = store.put(a)
    h3 = store.put(b)
    assert h1 in blocks[0]["content"]
    assert h3 in blocks[2]["content"]
