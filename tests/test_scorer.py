import asyncio
import json

import httpx
import pytest
import respx

from astromesh_context import (
    CallableSource,
    ContextEngine,
    ContextItem,
    LLMScorer,
    StaticSource,
    openai_compatible,
)
from astromesh_context.scorer import ScorerFallback
from astromesh_context.types import ScoredItem


def item(i, pinned=False, content=None):
    return ContextItem(
        f"k{i}",
        content or f"fact number {i} unique{i}",
        "kb",
        "knowledge",
        "volatile",
        score=float(i),
        tokens=5,
        pinned=pinned,
    )


def scored(i, calibrated=0.4, pinned=False):
    return ScoredItem(item(i, pinned=pinned), calibrated=calibrated, final=calibrated)


def fake(response, prompts=None):
    async def complete(prompt):
        if prompts is not None:
            prompts.append(prompt)
        return response

    return complete


def engine_with(scorer, n=3):
    return ContextEngine(
        [CallableSource(lambda q, c: [item(i) for i in range(n)], name="kb")],
        budget=1000,
        scorer=scorer,
    )


async def test_blends_scores():
    out = await LLMScorer(fake("[10]")).score("q", [scored(0, 0.4)])
    assert out[0].final == pytest.approx(0.7)
    assert out[0].calibrated == 0.4
    assert "llm=10" in out[0].adjustments


async def test_clamps_and_keeps_order():
    items = [scored(0, 0.2), scored(1, 0.9)]
    out = await LLMScorer(fake("sure: [99, -5]")).score("q", items)
    assert [s.item.id for s in out] == ["k0", "k1"]
    # scores map to candidates by descending final: k1 gets 99 -> 10, k0 gets -5 -> 0
    assert out[1].final == pytest.approx(0.5 + 0.45)
    assert out[0].final == pytest.approx(0.1)


async def test_only_top_n_sent():
    prompts = []
    items = [scored(i, calibrated=i / 10) for i in range(5)]
    out = await LLMScorer(fake("[5, 5]", prompts), top_n=2).score("q", items)
    assert prompts[0].count("[1] ") == 1
    assert "[2] " in prompts[0]
    assert "[3] " not in prompts[0]
    assert "unique4" in prompts[0]
    assert "unique3" in prompts[0]
    assert "unique0" not in prompts[0]
    assert out[0] is items[0]
    assert "llm=5" in out[4].adjustments


async def test_pinned_not_sent():
    prompts = []
    items = [scored(0, pinned=True), scored(1)]
    out = await LLMScorer(fake("[3]", prompts)).score("q", items)
    assert "unique0" not in prompts[0]
    assert out[0] is items[0]


async def test_no_candidates_skips_call():
    prompts = []
    items = [scored(0, pinned=True)]
    out = await LLMScorer(fake("[3]", prompts)).score("q", items)
    assert out == items
    assert prompts == []


async def test_prompt_format_and_truncation():
    prompts = []
    await LLMScorer(fake("[1]", prompts), max_chars=4).score(
        "why?", [ScoredItem(item(0, content="abcdefgh"), 0.1, 0.1)]
    )
    assert prompts[0] == (
        "Rate how useful each numbered snippet is for answering the question. "
        "Reply with only a JSON array of numbers from 0 to 10, one per snippet, in order."
        "\n\nQuestion: why?\n\n[1] abcd"
    )


async def test_garbage_response_falls_back():
    w = await engine_with(LLMScorer(fake("no idea"))).build("q")
    assert "scorer_fallback:parse" in w.trace.events
    assert len(w.items) == 3


async def test_wrong_count_falls_back():
    w = await engine_with(LLMScorer(fake("[1, 2]"))).build("q")
    assert "scorer_fallback:parse" in w.trace.events


async def test_non_numbers_fall_back():
    with pytest.raises(ScorerFallback) as e:
        await LLMScorer(fake('["a"]')).score("q", [scored(0)])
    assert e.value.reason == "parse"
    with pytest.raises(ScorerFallback):
        await LLMScorer(fake("[true]")).score("q", [scored(0)])


async def test_timeout_falls_back():
    async def slow(prompt):
        await asyncio.sleep(1)
        return "[1, 1, 1]"

    w = await engine_with(LLMScorer(slow, timeout=0.05)).build("q")
    assert "scorer_fallback:timeout" in w.trace.events


async def test_exception_falls_back():
    async def boom(prompt):
        raise RuntimeError("down")

    w = await engine_with(LLMScorer(boom)).build("q")
    assert "scorer_fallback:error" in w.trace.events


async def test_success_updates_trace():
    w = await engine_with(LLMScorer(fake("[10, 0, 5]"))).build("q")
    assert not [e for e in w.trace.events if e.startswith("scorer_fallback")]
    assert all(any(a.startswith("llm=") for a in e.adjustments) for e in w.trace.entries)


async def test_engine_without_scorer_unchanged():
    w = await engine_with(None).build("q")
    assert w.trace.events == []


@respx.mock
async def test_openai_compatible_posts():
    route = respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "[7]"}}]})
    )
    complete = openai_compatible("http://llm.test/v1/", "sk-x", "m1")
    assert await complete("hello") == "[7]"
    req = route.calls[0].request
    body = json.loads(req.content)
    assert body["model"] == "m1"
    assert body["temperature"] == 0
    assert body["messages"][-1]["content"] == "hello"
    assert req.headers["authorization"] == "Bearer sk-x"


@respx.mock
async def test_http_error_becomes_error_fallback():
    respx.post("http://llm.test/v1/chat/completions").mock(return_value=httpx.Response(500))
    scorer = LLMScorer(openai_compatible("http://llm.test/v1", "k", "m"))
    w = await engine_with(scorer).build("q")
    assert "scorer_fallback:error" in w.trace.events


async def test_pinned_engine_item_survives_scorer():
    engine = ContextEngine(
        [StaticSource("rules", pinned=True), CallableSource(lambda q, c: [item(1)], name="kb")],
        budget=1000,
        scorer=LLMScorer(fake("[4]")),
    )
    w = await engine.build("q")
    assert len(w.items) == 2
