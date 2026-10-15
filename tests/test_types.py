import dataclasses

import pytest

from astromesh_context.errors import BudgetExceededError, ContextError, MissingExtraError, SpecError
from astromesh_context.types import Budget, ContextItem, ScoredItem, Trace, TraceEntry


def test_budget_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        Budget(0)


def test_scored_item_tokens_floor_is_one():
    it = ContextItem("a", "", "s", "knowledge", "volatile", tokens=0)
    assert ScoredItem(it, 0.5, 0.5).tokens == 1


def test_scored_item_tokens_uses_item_tokens():
    it = ContextItem("a", "x", "s", "knowledge", "volatile", tokens=42)
    assert ScoredItem(it, 0.5, 0.5).tokens == 42


def test_context_item_is_frozen_and_with_tokens_copies():
    it = ContextItem("a", "hello", "s", "knowledge", "volatile")
    assert it.with_tokens(3).tokens == 3
    assert it.tokens is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        it.content = "x"  # type: ignore[misc]


def test_trace_explain_lists_reason_and_events():
    t = Trace(
        [TraceEntry("a", "s", 1.0, 0.9, 0.9, 10, False, "budget", ())], ["source_error:x:boom"]
    )
    out = t.explain()
    assert "budget" in out
    assert "source_error:x:boom" in out
    assert "0.900" in out
    assert "out" in out


def test_trace_excluded_filters_by_prefix():
    t = Trace(
        [
            TraceEntry("a", "s", 1.0, 0.9, 0.9, 10, False, "duplicate_of:b", ()),
            TraceEntry("b", "s", 1.0, 0.9, 0.9, 10, True, None, ()),
            TraceEntry("c", "s", 1.0, 0.9, 0.9, 10, False, "budget", ()),
        ],
        [],
    )
    assert [e.item_id for e in t.excluded("duplicate_of")] == ["a"]


def test_errors_hierarchy_and_attrs():
    e = BudgetExceededError(10, [("a", 20)])
    assert isinstance(e, ContextError)
    assert e.budget == 10
    assert e.pinned == [("a", 20)]
    assert SpecError("spec.budget", "bad").path == "spec.budget"
    assert isinstance(MissingExtraError("x"), ContextError)
