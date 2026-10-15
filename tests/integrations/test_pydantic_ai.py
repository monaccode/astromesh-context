import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from astromesh_context.integrations.pydantic_ai import (
    PydanticAIAdapter,
    history_processor,
    offload_tools,
)
from astromesh_context.loop import Compactor, Offloader


def tool_history(result: str):
    return [
        ModelRequest(parts=[UserPromptPart(content="look it up")]),
        ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart("lookup", result, tool_call_id="c1")]),
        ModelResponse(parts=[TextPart("found")]),
    ]


def recording_agent(processor):
    seen: list = []

    def fn(messages, info: AgentInfo):
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart("done")])

    return Agent(FunctionModel(fn), capabilities=[ProcessHistory(processor)]), seen


def test_adapter_reads_pydantic_ai_messages():
    a = PydanticAIAdapter()
    h = tool_history("out")
    assert [a.role(m) for m in h] == ["user", "assistant", "tool", "assistant"]
    assert a.role(ModelRequest(parts=[SystemPromptPart("rules")])) == "system"
    assert a.tool_call_ids(h[1]) == ["c1"]
    assert a.tool_result_id(h[2]) == "c1"
    assert a.tool_result_id(h[0]) is None
    assert "lookup" in a.text(h[1])
    summary = a.make_summary("old talk")
    assert isinstance(summary.parts[0], SystemPromptPart)
    assert summary.parts[0].content.startswith("[context summary]")
    assert a.is_summary(summary)
    assert not a.is_summary(h[0])
    shrunk = a.replace_tool_result(h[2], "short")
    part = shrunk.parts[0]
    assert (part.content, part.tool_call_id, part.tool_name) == ("short", "c1", "lookup")
    assert h[2].parts[0].content == "out"


async def test_large_tool_result_reaches_the_model_offloaded():
    off = Offloader(max_tokens=100, preview_tokens=10)
    agent, seen = recording_agent(history_processor(offloader=off))
    await agent.run("and now?", message_history=tool_history("R" * 5000))
    returns = [
        p
        for m in seen[0]
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert len(returns) == 1
    assert len(returns[0].content) < 500
    assert "[truncated: 5000 characters stored as ctx_" in returns[0].content
    assert (returns[0].tool_call_id, returns[0].tool_name) == ("c1", "lookup")
    handle = returns[0].content.split("stored as ")[1].split(".")[0]
    read, _ = offload_tools(off)
    assert read.function(handle, limit=50) == "R" * 50


async def test_long_history_reaches_the_model_compacted_with_pairs_intact():
    async def summarize(msgs):
        return "SUMMARY-OF-OLD-TURNS"

    comp = Compactor(summarize, threshold_tokens=50, keep_last=3, adapter=PydanticAIAdapter())
    history = []
    for i in range(6):
        history += [
            ModelRequest(parts=[UserPromptPart(content=f"q{i} " + "x" * 80)]),
            ModelResponse(parts=[TextPart(f"a{i} " + "y" * 80)]),
        ]
    history += tool_history("tool output " * 5)
    agent, seen = recording_agent(history_processor(compactor=comp))
    await agent.run("and now?", message_history=history)
    sent = seen[0]
    assert len(sent) < len(history)
    adapter = PydanticAIAdapter()
    assert any(adapter.is_summary(m) for m in sent)
    calls = {i for m in sent for i in adapter.tool_call_ids(m)}
    results = {adapter.tool_result_id(m) for m in sent} - {None}
    assert calls == results == {"c1"}


def test_offload_tools_are_pydantic_ai_tools():
    read, grep = offload_tools(Offloader())
    assert (read.name, grep.name) == ("ctx_read", "ctx_grep")
    assert "stored" in read.description.lower()
