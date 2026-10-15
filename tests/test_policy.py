import random

import pytest

from astromesh_context.errors import BudgetExceededError
from astromesh_context.policy import Everything, FixedSlots, ScoredKnapsack
from astromesh_context.types import Budget, ContextItem, ScoredItem, Signals


def si(id, final=0.5, tokens=100, source="s", pinned=False, deprecated=False):
    item = ContextItem(
        id=id,
        content=id,
        source=source,
        kind="knowledge",
        stability="session",
        tokens=tokens,
        pinned=pinned,
        signals=Signals(deprecated=deprecated),
    )
    return ScoredItem(item, final, final)


def ids(items):
    return [s.item.id for s in items]


def reasons(sel):
    return {s.item.id: r for s, r in sel.excluded}


def test_pinned_over_budget_raises_with_items():
    items = [si("p1", tokens=80, pinned=True), si("p2", tokens=50, pinned=True)]
    with pytest.raises(BudgetExceededError) as ei:
        ScoredKnapsack().select(items, Budget(100))
    assert ei.value.budget == 100
    assert ei.value.pinned == [("p1", 80), ("p2", 50)]
    for policy in (FixedSlots({"s": 1.0}), Everything()):
        with pytest.raises(BudgetExceededError):
            policy.select(items, Budget(100))


def test_knapsack_prefers_density():
    a = si("A", 0.9, 900)
    b = si("B", 0.6, 100)
    c = si("C", 0.6, 100)
    sel = ScoredKnapsack().select([a, b, c], Budget(250))
    assert ids(sel.included) == ["B", "C"]
    assert reasons(sel) == {"A": "budget"}


def test_knapsack_min_items_forced():
    hist = [si(f"h{i}", 0.01, 50, source="history") for i in range(3)]
    docs = [si(f"d{i}", 0.9, 50, source="rag") for i in range(3)]
    sel = ScoredKnapsack().select(docs + hist, Budget(200, min_items={"history": 2}))
    assert ids(sel.included)[:2] == ["h0", "h1"]
    assert set(ids(sel.included)) == {"h0", "h1", "d0", "d1"}
    assert reasons(sel) == {"d2": "budget", "h2": "budget"}


def test_knapsack_min_items_ignores_floor_and_caps_but_respects_budget():
    hist = [si("h0", 0.01, 50, source="history"), si("h1", 0.02, 500, source="history")]
    budget = Budget(100, caps={"history": 10}, min_items={"history": 2})
    sel = ScoredKnapsack(min_score=0.5).select(hist, budget)
    # h1 is the best by final but does not fit; h0 is forced in despite floor and cap.
    assert ids(sel.included) == ["h0"]
    assert reasons(sel) == {"h1": "below_floor"}


def test_knapsack_source_cap():
    items = [si("a", 0.9, 60, source="rag"), si("b", 0.8, 60, source="rag"), si("c", 0.1, 60)]
    sel = ScoredKnapsack().select(items, Budget(1000, caps={"rag": 100}))
    assert ids(sel.included) == ["a", "c"]
    assert reasons(sel) == {"b": "source_cap"}


def test_pinned_do_not_count_toward_cap():
    items = [si("p", 0.0, 90, source="rag", pinned=True), si("a", 0.9, 90, source="rag")]
    sel = ScoredKnapsack().select(items, Budget(1000, caps={"rag": 100}))
    assert ids(sel.included) == ["p", "a"]


def test_knapsack_below_floor():
    items = [si("hi", 0.8), si("lo", 0.2)]
    sel = ScoredKnapsack(min_score=0.5).select(items, Budget(1000))
    assert ids(sel.included) == ["hi"]
    assert reasons(sel) == {"lo": "below_floor"}


def test_knapsack_backfill():
    # b is denser than c but does not fit after a; c fits in the leftover space.
    items = [si("a", 0.9, 60), si("b", 0.7, 50), si("c", 0.2, 40)]
    sel = ScoredKnapsack().select(items, Budget(100))
    assert ids(sel.included) == ["a", "c"]
    assert reasons(sel) == {"b": "budget"}
    assert sum(s.tokens for s in sel.included) <= 100


def test_deprecated_excluded_unless_allowed():
    items = [si("ok"), si("old", deprecated=True)]
    sel = ScoredKnapsack().select(items, Budget(1000))
    assert ids(sel.included) == ["ok"]
    assert reasons(sel) == {"old": "deprecated"}
    sel = ScoredKnapsack(include_deprecated=True).select(items, Budget(1000))
    assert set(ids(sel.included)) == {"ok", "old"}
    for policy in (FixedSlots({"s": 1.0}), Everything()):
        assert reasons(policy.select(items, Budget(1000))) == {"old": "deprecated"}


def test_pinned_deprecated_is_included():
    items = [si("p", pinned=True, deprecated=True)]
    sel = ScoredKnapsack().select(items, Budget(1000))
    assert ids(sel.included) == ["p"]
    assert sel.excluded == []


def test_pinned_first_then_added_order():
    items = [si("a", 0.9), si("p", 0.0, pinned=True)]
    sel = ScoredKnapsack().select(items, Budget(1000))
    assert ids(sel.included) == ["p", "a"]


def test_zero_token_item_counts_as_one():
    sel = ScoredKnapsack().select([si("e", 0.5, tokens=0)], Budget(10))
    assert ids(sel.included) == ["e"]
    tight = [si("e1", 0.5, tokens=0), si("e2", 0.4, tokens=0)]
    sel = ScoredKnapsack().select(tight, Budget(1))
    assert ids(sel.included) == ["e1"]
    assert reasons(sel) == {"e2": "budget"}


def test_fixed_slots_does_not_redistribute():
    rag = [si("r", 0.9, 10, source="rag")]
    hist = [si("h1", 0.9, 40, source="history"), si("h2", 0.8, 40, source="history")]
    sel = FixedSlots({"rag": 0.5, "history": 0.5}).select(rag + hist, Budget(100))
    assert ids(sel.included) == ["h1", "r"]
    assert reasons(sel) == {"h2": "share_exhausted"}


def test_fixed_slots_subtracts_pinned_and_unknown_source():
    items = [
        si("p", 0.0, 20, pinned=True),
        si("a", 0.9, 40, source="rag"),
        si("b", 0.9, 41, source="rag"),
        si("z", 0.9, 1, source="other"),
    ]
    sel = FixedSlots({"rag": 0.5}).select(items, Budget(100))  # floor(0.5 * 80) = 40
    assert ids(sel.included) == ["p", "a"]
    assert reasons(sel) == {"b": "share_exhausted", "z": "share_exhausted"}


def test_fixed_slots_rejects_bad_shares():
    with pytest.raises(ValueError, match=r"at most 1.0"):
        FixedSlots({"a": 0.6, "b": 0.5})
    with pytest.raises(ValueError, match=">= 0"):
        FixedSlots({"a": -0.1})
    FixedSlots({"a": 0.5, "b": 0.5})


def test_everything_raises_when_over():
    items = [si("a", tokens=60), si("b", tokens=60)]
    with pytest.raises(BudgetExceededError) as ei:
        Everything().select(items, Budget(100))
    assert ei.value.pinned == [("a", 60), ("b", 60)]
    sel = Everything().select(items, Budget(120))
    assert ids(sel.included) == ["a", "b"]


def test_everything_truncate_marks_truncated():
    items = [si("a", 0.3, 60), si("b", 0.9, 60), si("c", 0.5, 30)]
    sel = Everything(truncate=True).select(items, Budget(100))
    assert ids(sel.included) == ["b", "c"]
    assert reasons(sel) == {"a": "truncated"}


def test_every_item_accounted_exactly_once():
    items = [si(f"i{n}", n / 10, 30 + n * 7, source=("a", "b")[n % 2]) for n in range(10)]
    items.append(si("dep", deprecated=True))
    sel = ScoredKnapsack(min_score=0.2).select(items, Budget(200, caps={"a": 80}))
    seen = ids(sel.included) + [s.item.id for s, _ in sel.excluded]
    assert sorted(seen) == sorted(ids(items))
    assert {r for _, r in sel.excluded} <= {"budget", "source_cap", "below_floor", "deprecated"}


def test_select_is_deterministic():
    items = [si(f"i{n}", (n % 4) / 4, 20 + n, source=("a", "b")[n % 2]) for n in range(20)]
    items += [si("dep1", deprecated=True), si("dep2", deprecated=True)]
    budget = Budget(250, caps={"a": 100}, min_items={"b": 2})
    for policy in (
        ScoredKnapsack(min_score=0.1),
        FixedSlots({"a": 0.5, "b": 0.5}),
        Everything(truncate=True),
    ):
        base = policy.select(items, budget)
        for seed in range(5):
            shuffled = items[:]
            random.Random(seed).shuffle(shuffled)
            assert policy.select(shuffled, budget) == base
