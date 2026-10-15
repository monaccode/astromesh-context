import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field

from astromesh_context import ContextEngine, StaticSource
from astromesh_context.integrations.langchain import (
    ContextMiddleware,
    LangChainAdapter,
    offload_tools,
)
from astromesh_context.loop import Compactor, Offloader, units


class RecordingModel(GenericFakeChatModel):
    seen: list = Field(default_factory=list)

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    async def _agenerate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return await super()._agenerate(messages, *args, **kwargs)


def model():
    return RecordingModel(messages=iter([AIMessage(content="done")]))


def tool_history(result: str):
    return [
        HumanMessage(content="look it up"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}]),
        ToolMessage(content=result, tool_call_id="c1", name="lookup"),
        HumanMessage(content="and now?"),
    ]


def test_adapter_reads_langchain_messages():
    a = LangChainAdapter()
    call = AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": 1}, "id": "c1"}])
    result = ToolMessage(content="out", tool_call_id="c1", name="lookup")
    assert a.role(HumanMessage(content="hi")) == "user"
    assert a.role(call) == "assistant"
    assert a.role(result) == "tool"
    assert a.role(SystemMessage(content="s")) == "system"
    assert a.tool_call_ids(call) == ["c1"]
    assert a.tool_result_id(result) == "c1"
    assert "lookup" in a.text(call)
    summary = a.make_summary("old talk")
    assert isinstance(summary, SystemMessage)
    assert summary.name == "context_summary"
    assert a.is_summary(summary)
    assert not a.is_summary(SystemMessage(content="s"))
    shrunk = a.replace_tool_result(result, "short")
    assert shrunk.content == "short"
    assert (shrunk.tool_call_id, shrunk.name) == ("c1", "lookup")


async def test_engine_window_becomes_the_system_prompt():
    engine = ContextEngine([StaticSource("PINNED-RULE-42", pinned=True)], budget=500)
    m = model()
    agent = create_agent(m, tools=[], middleware=[ContextMiddleware(engine)])
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]})
    sent = m.seen[0]
    assert isinstance(sent[0], SystemMessage)
    assert "PINNED-RULE-42" in sent[0].content


async def test_no_human_message_skips_the_engine():
    engine = ContextEngine([StaticSource("PINNED-RULE-42", pinned=True)], budget=500)
    m = model()
    agent = create_agent(m, tools=[], middleware=[ContextMiddleware(engine)])
    await agent.ainvoke({"messages": [AIMessage(content="earlier")]})
    assert not any("PINNED-RULE-42" in str(x.content) for x in m.seen[0])


async def test_large_tool_result_reaches_the_model_offloaded():
    off = Offloader(max_tokens=100, preview_tokens=10)
    m = model()
    agent = create_agent(m, tools=[], middleware=[ContextMiddleware(offloader=off)])
    await agent.ainvoke({"messages": tool_history("R" * 5000)})
    tool_msg = next(x for x in m.seen[0] if isinstance(x, ToolMessage))
    assert len(tool_msg.content) < 500
    assert "[truncated: 5000 characters stored as ctx_" in tool_msg.content
    assert tool_msg.tool_call_id == "c1"
    handle = tool_msg.content.split("stored as ")[1].split(".")[0]
    read, _grep = offload_tools(off)
    assert read.invoke({"handle": handle, "limit": 50}) == "R" * 50


async def test_long_history_reaches_the_model_compacted_with_pairs_intact():
    async def summarize(msgs):
        return "SUMMARY-OF-OLD-TURNS"

    comp = Compactor(summarize, threshold_tokens=50, keep_last=3, adapter=LangChainAdapter())
    history = []
    for i in range(6):
        history += [
            HumanMessage(content=f"q{i} " + "x" * 80),
            AIMessage(content=f"a{i} " + "y" * 80),
        ]
    history += tool_history("tool output " * 5)
    m = model()
    agent = create_agent(m, tools=[], middleware=[ContextMiddleware(compactor=comp)])
    result = await agent.ainvoke({"messages": history})
    sent = m.seen[0]
    assert len(sent) < len(history)
    assert any(isinstance(x, SystemMessage) and x.name == "context_summary" for x in sent)
    assert any("SUMMARY-OF-OLD-TURNS" in str(x.content) for x in sent)
    adapter = LangChainAdapter()
    calls = {i for x in sent for i in adapter.tool_call_ids(x)}
    results = {adapter.tool_result_id(x) for x in sent} - {None}
    assert calls == results == {"c1"}
    assert all(len(u) <= 2 or not adapter.tool_call_ids(u[0]) for u in units(sent, adapter))
    # the compacted history is what the thread now holds, not just what the model saw
    assert len(result["messages"]) < len(history)


def test_sync_invoke_also_offloads():
    off = Offloader(max_tokens=100, preview_tokens=10)
    m = model()
    agent = create_agent(m, tools=[], middleware=[ContextMiddleware(offloader=off)])
    agent.invoke({"messages": tool_history("R" * 5000)})
    tool_msg = next(x for x in m.seen[0] if isinstance(x, ToolMessage))
    assert "[truncated:" in tool_msg.content


def test_offload_tools_are_langchain_tools():
    read, grep = offload_tools(Offloader())
    assert (read.name, grep.name) == ("ctx_read", "ctx_grep")
    assert "stored" in read.description.lower()
