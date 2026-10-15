import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from astromesh_context import CallableSource, ContextEngine, ContextItem

BUDGET = 1000
SETTINGS = settings(max_examples=50, deadline=None)


@st.composite
def item_lists(draw):
    items = []
    pinned_total = 0
    for i in range(draw(st.integers(0, 12))):
        tokens = draw(st.integers(0, 500))
        pinned = draw(st.booleans())
        if pinned and pinned_total + max(1, tokens) > BUDGET:
            pinned = False
        if pinned:
            pinned_total += max(1, tokens)
        items.append(
            ContextItem(
                f"i{i}",
                f"unique{i} " + "w" * max(0, tokens * 4 - 8),
                "src",
                "knowledge",
                draw(st.sampled_from(["stable", "session", "volatile"])),
                score=draw(st.floats(0, 10, allow_nan=False)),
                tokens=tokens,
                pinned=pinned,
            )
        )
    return items


def build(items):
    engine = ContextEngine([CallableSource(lambda q, c: items, name="src")], budget=BUDGET)
    return asyncio.run(engine.build("q"))


@SETTINGS
@given(item_lists())
def test_budget_and_pinned_hold(items):
    w = build(items)
    assert w.tokens_used <= BUDGET
    ids = {i.id for i in w.items}
    assert all(i.id in ids for i in items if i.pinned)


@SETTINGS
@given(item_lists())
def test_deterministic(items):
    assert [i.id for i in build(items).items] == [i.id for i in build(items).items]


@SETTINGS
@given(item_lists(), item_lists())
def test_stable_prefix_hash_ignores_volatile(base, extra):
    stable = [i for i in base if i.pinned and i.stability == "stable"]
    volatile = [
        ContextItem(f"v{i.id}", f"volatile {i.id}", i.source, i.kind, "volatile", i.score, i.tokens)
        for i in extra
    ]
    assert build(stable).stable_prefix_hash == build(stable + volatile).stable_prefix_hash
