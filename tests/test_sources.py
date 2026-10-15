from astromesh_context.source import CallableSource, HistorySource, StaticSource
from astromesh_context.types import Budget, BuildContext, ContextItem


def make_ctx(**vars):
    return BuildContext(query="q", budget=Budget(total=100), deadline=0.0, vars=vars)


def test_static_source_stable_id():
    a = StaticSource("You are helpful.")
    b = StaticSource("You  ARE helpful.")
    (ia,) = a.candidates("q", make_ctx())
    (ib,) = b.candidates("other", make_ctx())
    assert ia.id == ib.id
    assert ia.id.startswith("static:")
    assert len(ia.id) == len("static:") + 12
    assert (ia.kind, ia.stability, ia.pinned, ia.score) == ("instruction", "stable", False, 1.0)
    assert ia.source == "static"


async def test_callable_source_sync_and_async():
    item = ContextItem("x", "hello", "c", "knowledge", "volatile")

    def sync_fn(query, ctx):
        return [item]

    async def async_fn(query, ctx):
        return [item]

    s1 = CallableSource(sync_fn, name="c")
    s2 = CallableSource(async_fn, name="c")
    assert s1.name == "c"
    assert s1.candidates("q", make_ctx()) == [item]
    assert await s2.candidates("q", make_ctx()) == [item]


def test_history_pairs_tool_call_with_results():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "result"},
        {"role": "assistant", "content": "done"},
    ]
    items = HistorySource(msgs).candidates("q", make_ctx())
    assert [i.id for i in items] == ["history:0", "history:1", "history:3"]
    assert items[1].meta["messages"] == msgs[1:3]
    assert items[1].order_key == "00000001"
    assert items[2].score == 3
    assert (items[0].kind, items[0].stability, items[0].source) == ("history", "session", "history")
    assert "result" in items[1].content


def test_history_orphan_tool_call_is_own_unit():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
        {"role": "user", "content": "next"},
    ]
    items = HistorySource(msgs).candidates("q", make_ctx())
    assert [i.id for i in items] == ["history:0", "history:1"]
    assert items[0].meta["messages"] == [msgs[0]]


def test_history_orphan_tool_result_is_own_unit():
    msgs = [
        {"role": "tool", "tool_call_id": "gone", "content": "late"},
        {"role": "assistant", "content": "ok"},
    ]
    items = HistorySource(msgs).candidates("q", make_ctx())
    assert [i.id for i in items] == ["history:0", "history:1"]
    assert items[0].meta["messages"] == [msgs[0]]


def test_history_reads_ctx_vars():
    msgs = [{"role": "user", "content": "hi"}]
    src = HistorySource(key="chat")
    assert [i.id for i in src.candidates("q", make_ctx(chat=msgs))] == ["history:0"]
    assert src.candidates("q", make_ctx()) == []


def test_history_content_handles_list_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    (item,) = HistorySource(msgs).candidates("q", make_ctx())
    assert "hello" in item.content
