import logging

from hypothesis import given
from hypothesis import strategies as st

from astromesh_context.loop.compactor import DEFAULT_SUMMARY_PROMPT, Compactor, summary_prompt
from astromesh_context.loop.messages import AnthropicAdapter, OpenAIAdapter, units


class PerChar:
    def count(self, text: str) -> int:
        return len(text)


def _recorder(reply="SUM"):
    seen = []

    async def summarize(msgs):
        seen.append(msgs)
        return reply

    return summarize, seen


def _u(i):
    return {"role": "user", "content": f"u{i}"}


def _call(cid, text=""):
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [{"id": cid, "function": {"name": "f", "arguments": ""}}],
    }


def _res(cid):
    return {"role": "tool", "tool_call_id": cid, "content": "r"}


async def test_under_threshold_returns_same_object():
    summarize, seen = _recorder()
    msgs = [_u(1), _u(2)]
    out = await Compactor(summarize, threshold_tokens=1000).compact(msgs)
    assert out is msgs
    assert not seen


async def test_keeps_last_and_summarizes_rest():
    summarize, seen = _recorder()
    msgs = [_u(i) for i in range(10)]
    out = await Compactor(summarize, threshold_tokens=1, keep_last=3).compact(msgs)
    assert seen == [msgs[:7]]
    assert out == [OpenAIAdapter().make_summary("SUM"), *msgs[7:]]


async def test_nothing_ages_returns_input():
    summarize, seen = _recorder()
    msgs = [_u(i) for i in range(3)]
    out = await Compactor(summarize, threshold_tokens=1, keep_last=6).compact(msgs)
    assert out is msgs
    assert not seen


async def test_never_splits_tool_call_from_results():
    summarize, seen = _recorder()
    msgs = [_u(0), _u(1), _call("c1"), _res("c1"), _u(4), _u(5)]
    # keep_last=3 would cut between the call and its result; the cut moves earlier
    out = await Compactor(summarize, threshold_tokens=1, keep_last=3).compact(msgs)
    assert seen == [msgs[:2]]
    assert out[1:] == msgs[2:]


async def test_incremental_includes_previous_summary_once():
    summarize, seen = _recorder("NEW")
    prev = OpenAIAdapter().make_summary("OLD")
    msgs = [prev] + [_u(i) for i in range(8)]
    out = await Compactor(summarize, threshold_tokens=1, keep_last=2).compact(msgs)
    assert seen[0][0] == prev
    assert seen[0][1:] == msgs[1:7]
    assert [m for m in out if OpenAIAdapter().is_summary(m)] == [
        OpenAIAdapter().make_summary("NEW")
    ]
    assert out[1:] == msgs[7:]


async def test_leading_system_preserved():
    summarize, seen = _recorder()
    sys = {"role": "system", "content": "be nice"}
    msgs = [sys] + [_u(i) for i in range(8)]
    out = await Compactor(summarize, threshold_tokens=1, keep_last=2).compact(msgs)
    assert out[0] == sys
    assert sys not in seen[0]
    assert OpenAIAdapter().is_summary(out[1])


async def test_summarize_failure_returns_input(caplog):
    async def boom(msgs):
        raise RuntimeError("down")

    msgs = [_u(i) for i in range(8)]
    with caplog.at_level(logging.WARNING, logger="astromesh_context"):
        out = await Compactor(boom, threshold_tokens=1, keep_last=2).compact(msgs)
    assert out is msgs
    assert "down" in caplog.text


async def test_orphan_tool_call_survives():
    summarize, _ = _recorder()
    msgs = [_u(0), _u(1), _call("lost"), _u(3), _u(4), _u(5)]
    out = await Compactor(summarize, threshold_tokens=1, keep_last=4).compact(msgs)
    assert msgs[2] in out  # orphan call kept as its own unit in the tail


async def test_custom_tokenizer_threshold():
    summarize, _seen = _recorder()
    msgs = [_u(i) for i in range(4)]  # 8 chars total
    comp = Compactor(summarize, threshold_tokens=8, keep_last=1, tokenizer=PerChar())
    assert await comp.compact(msgs) is msgs
    comp = Compactor(summarize, threshold_tokens=7, keep_last=1, tokenizer=PerChar())
    assert await comp.compact(msgs) is not msgs


async def test_anthropic_adapter_summary_shape():
    summarize, _ = _recorder("S")
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(6)]
    out = await Compactor(
        summarize, threshold_tokens=1, keep_last=2, adapter=AnthropicAdapter()
    ).compact(msgs)
    assert out[0] == {"role": "user", "content": [{"type": "text", "text": "[context summary]\nS"}]}


def test_summary_prompt_renders_roles():
    prompt = summary_prompt([_u(1), _res("c")], OpenAIAdapter())
    assert prompt.startswith(DEFAULT_SUMMARY_PROMPT)
    assert prompt.endswith("user: u1\ntool: r")


@st.composite
def histories(draw):
    msgs = []
    for n in range(1, draw(st.integers(0, 14)) + 1):
        kind = draw(st.sampled_from(["user", "pair", "lonecall", "loneres"]))
        cid = f"c{n}"
        if kind == "user":
            msgs.append(_u(n))
        elif kind == "pair":
            msgs += [_call(cid), _res(cid)]
        elif kind == "lonecall":
            msgs.append(_call(cid))
        else:
            msgs.append(_res(cid))
    return msgs


def _pairs(msgs):
    return [u for u in units(msgs, OpenAIAdapter()) if len(u) > 1]


@given(histories(), st.integers(1, 6))
async def test_property_pairs_never_broken(msgs, keep_last):
    summarize, _ = _recorder()
    out = await Compactor(summarize, threshold_tokens=1, keep_last=keep_last).compact(msgs)
    out_units = [tuple(map(id, u)) for u in units(out, OpenAIAdapter())]
    # every pair from the input is either wholly in the output or wholly gone
    for pair in _pairs(msgs):
        present = [any(m is x for x in out) for m in pair]
        assert all(present) or not any(present)
    # and every call/result present in the output is still adjacent in one unit
    ids_in_units = {i for u in out_units for i in u}
    for m in out:
        if m.get("role") == "tool" and any(
            c["id"] == m["tool_call_id"] for x in msgs for c in x.get("tool_calls", [])
        ):
            assert id(m) in ids_in_units
            unit = next(u for u in units(out, OpenAIAdapter()) if any(x is m for x in u))
            assert len(unit) > 1
