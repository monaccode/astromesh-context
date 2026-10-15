from astromesh_context.loop.messages import AnthropicAdapter, OpenAIAdapter, units

OA = OpenAIAdapter()
AN = AnthropicAdapter()


def test_openai_roundtrip():
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
    }
    res = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
    assert OA.tool_call_ids(call) == ["c1"]
    assert OA.tool_call_ids(res) == []
    assert OA.tool_result_id(res) == "c1"
    assert OA.tool_result_id(call) is None
    assert OA.text(call) == "f({})"
    assert OA.replace_tool_result(res, "short") == {**res, "content": "short"}
    s = OA.make_summary("hi")
    assert s == {"role": "system", "name": "context_summary", "content": "hi"}
    assert OA.is_summary(s)
    assert not OA.is_summary(res)
    assert OA.role(res) == "tool"


def test_anthropic_roundtrip():
    call = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "looking"},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
        ],
    }
    res = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
    }
    assert AN.tool_call_ids(call) == ["t1"]
    assert AN.tool_result_id(res) == "t1"
    assert AN.tool_result_id(call) is None
    assert AN.text(call) == 'looking\nf({"a": 1})'
    assert AN.text(res) == "ok"
    out = AN.replace_tool_result(res, "short")
    assert out["content"][0] == {"type": "tool_result", "tool_use_id": "t1", "content": "short"}
    s = AN.make_summary("hi")
    assert s == {"role": "user", "content": [{"type": "text", "text": "[context summary]\nhi"}]}
    assert AN.is_summary(s)
    assert not AN.is_summary(res)


def test_units_groups_anthropic_pairs():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
        {"role": "assistant", "content": "done"},
    ]
    assert [len(u) for u in units(msgs, AN)] == [1, 2, 1]


def _call(*ids):
    return {
        "role": "assistant",
        "tool_calls": [{"id": i, "function": {"name": "f", "arguments": ""}} for i in ids],
    }


def test_units_ignores_missing_ids():
    # a call without id declares nothing; a result without id is always an orphan
    msgs = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": ""}}]},
        {"role": "tool", "content": "no id"},
        {"role": "tool", "tool_call_id": "", "content": "empty id"},
    ]
    assert [len(u) for u in units(msgs, OA)] == [1, 1, 1]


def test_units_orphans_alone():
    msgs = [
        _call("c1"),
        {"role": "user", "content": "x"},
        {"role": "tool", "tool_call_id": "zz", "content": "orphan"},
    ]
    assert [len(u) for u in units(msgs, OA)] == [1, 1, 1]
