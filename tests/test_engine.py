import asyncio

import pytest

from astromesh_context import (
    BudgetExceededError,
    CallableSource,
    ContextEngine,
    ContextItem,
    HistorySource,
    ScoredKnapsack,
    SourceConfig,
    StaticSource,
)


def knowledge(n, text_len=40):
    return [
        ContextItem(f"k{i}", f"fact {i} " + "x" * text_len, "kb", "knowledge", "volatile", score=i)
        for i in range(n)
    ]


def history(n_pairs):
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"question {i} " + "y" * 30})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "z" * 30})
    return msgs


async def test_build_end_to_end():
    engine = ContextEngine(
        [
            StaticSource("system rules", pinned=True),
            CallableSource(lambda q, c: knowledge(10), name="kb"),
            HistorySource(history(6)),
        ],
        budget=200,
    )
    w = await engine.build("what?")
    ids = [i.id for i in w.items]
    assert any(i.startswith("static:") for i in ids)
    assert "history:10" in ids
    assert "history:11" in ids
    assert w.tokens_used <= 200
    assert w.budget == 200
    assert len(w.trace.entries) == 1 + 10 + 12
    assert {e.item_id for e in w.trace.entries if e.included} == set(ids)
    assert all(e.reason is None for e in w.trace.entries if e.included)
    assert all(e.reason for e in w.trace.entries if not e.included)
    assert len(w.stable_prefix_hash) == 64


async def test_failing_source_recorded_and_skipped():
    def boom(q, c):
        raise RuntimeError("kaput")

    engine = ContextEngine([StaticSource("rules"), CallableSource(boom, name="bad")], budget=100)
    w = await engine.build("q")
    assert "source_error:bad:kaput" in w.trace.events
    assert len(w.items) == 1


async def test_slow_source_times_out():
    async def slow(q, c):
        await asyncio.sleep(1)
        return []

    engine = ContextEngine(
        [StaticSource("rules"), CallableSource(slow, name="slow")],
        budget=100,
        source_timeout=0.05,
    )
    w = await engine.build("q")
    assert "source_error:slow:timeout" in w.trace.events
    assert len(w.items) == 1


async def test_source_config_timeout_overrides():
    async def slow(q, c):
        await asyncio.sleep(0.2)
        return []

    engine = ContextEngine(
        [SourceConfig(CallableSource(slow, name="slow"), timeout=0.02)], budget=100
    )
    w = await engine.build("q")
    assert "source_error:slow:timeout" in w.trace.events


async def test_all_sources_fail_returns_pinned_only():
    def boom(q, c):
        raise ValueError("no")

    engine = ContextEngine(
        [
            SourceConfig(StaticSource("must keep", pinned=True)),
            CallableSource(boom, name="a"),
            CallableSource(boom, name="b"),
        ],
        budget=100,
    )
    w = await engine.build("q")
    assert [i.pinned for i in w.items] == [True]
    assert len(w.trace.events) == 2


async def test_empty_query():
    engine = ContextEngine(
        [StaticSource("rules", pinned=True), HistorySource(history(2))], budget=500
    )
    for q in ("", "   "):
        w = await engine.build(q)
        assert len(w.items) == 5


async def test_pinned_over_budget_raises():
    engine = ContextEngine([StaticSource("x" * 400, pinned=True)], budget=10)
    with pytest.raises(BudgetExceededError):
        await engine.build("q")


async def test_source_name_overwritten_and_caps_apply():
    items = [
        ContextItem(f"a{i}", f"{i} " * 20, "wrong", "knowledge", "volatile", score=i)
        for i in range(5)
    ]
    engine = ContextEngine(
        [SourceConfig(CallableSource(lambda q, c: items, name="kb"), max_tokens=20)],
        budget=1000,
    )
    w = await engine.build("q")
    assert all(i.source == "kb" for i in w.items)
    assert len(w.items) == 2
    assert len(w.trace.excluded("source_cap")) == 3


async def test_history_min_items_default_and_explicit():
    big = [{"role": "user", "content": "a" * 1200}, {"role": "assistant", "content": "b" * 1200}]
    engine = ContextEngine(
        [CallableSource(lambda q, c: knowledge(40, 100), name="kb"), HistorySource(big)],
        budget=1000,
    )
    w = await engine.build("q")
    assert {"history:0", "history:1"} <= {i.id for i in w.items}
    explicit = ContextEngine(
        [
            CallableSource(lambda q, c: knowledge(40, 100), name="kb"),
            SourceConfig(HistorySource(big)),
        ],
        budget=1000,
        policy=ScoredKnapsack(beta=0.5),  # density order: the long history items lose
    )
    w2 = await explicit.build("q")
    assert not any(i.id.startswith("history:") for i in w2.items)


async def test_vars_reach_sources_and_deadline_set():
    seen = {}

    def grab(q, ctx):
        seen["vars"] = dict(ctx.vars)
        seen["deadline"] = ctx.deadline
        return []

    engine = ContextEngine([CallableSource(grab, name="g")], budget=10)
    await engine.build("q", vars={"a": 1})
    assert seen["vars"] == {"a": 1}
    assert seen["deadline"] > 0


async def test_dedupe_inherits_deprecation_and_trace_has_one_entry_per_candidate():
    from astromesh_context import Signals

    text = "refund window is fourteen days for the old policy " * 3
    rag = CallableSource(
        lambda q, ctx: [ContextItem("rag:p", text, "rag", "knowledge", "volatile", score=0.9)],
        name="rag",
    )
    okf = CallableSource(
        lambda q, ctx: [
            ContextItem(
                "okf:p",
                text,
                "okf",
                "knowledge",
                "volatile",
                score=0.2,
                signals=Signals(deprecated=True),
            )
        ],
        name="okf",
    )
    window = await ContextEngine([rag, SourceConfig(okf, weight=0.5)], budget=1000).build("refund")
    assert window.items == []
    reasons = {e.item_id: e.reason for e in window.trace.entries}
    assert len(window.trace.entries) == 2
    assert reasons == {"rag:p": "deprecated", "okf:p": "duplicate_of:rag:p"}
    kept = next(e for e in window.trace.entries if e.item_id == "rag:p")
    assert "inherited:deprecated:okf:p" in kept.adjustments
