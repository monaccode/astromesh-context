import pytest

from astromesh_context.eval.metrics import (
    distractor_rate,
    noise_share,
    precision_per_token,
    prefix_reuse,
    recall_at_budget,
    superseded_rate,
)
from astromesh_context.types import ContextItem


def item(id_, tokens):
    return ContextItem(id_, "x", "s", "knowledge", "volatile", tokens=tokens)


def test_recall_half():
    assert recall_at_budget({"a", "b"}, frozenset({"a", "c"})) == 0.5


def test_recall_full_and_none():
    assert recall_at_budget({"a", "c", "z"}, frozenset({"a", "c"})) == 1.0
    assert recall_at_budget(set(), frozenset({"a"})) == 0.0


def test_recall_no_relevant_is_vacuous():
    assert recall_at_budget({"a"}, frozenset()) == 1.0


def test_precision_per_token():
    docs = {"i1": "a", "i2": "b", "i3": None}
    items = [item("i1", 10), item("i2", 30), item("i3", 10)]
    assert precision_per_token(items, lambda it: docs[it.id], frozenset({"a"})) == pytest.approx(
        10 / 50
    )
    # two items from the same relevant doc both count
    docs["i2"] = "a"
    assert precision_per_token(items, lambda it: docs[it.id], frozenset({"a"})) == pytest.approx(
        40 / 50
    )


def test_precision_per_token_empty_window():
    assert precision_per_token([], lambda it: None, frozenset({"a"})) == 0.0


def test_distractor_rate():
    windows = [{"a"}, {"x"}, {"b", "x"}]
    assert distractor_rate(windows, frozenset({"x"})) == pytest.approx(2 / 3)
    assert distractor_rate([], frozenset({"x"})) == 0.0


def test_prefix_reuse():
    assert prefix_reuse(["x", "x", "y"]) == 0.5
    assert prefix_reuse(["x", "x", "x"]) == 1.0
    assert prefix_reuse(["x", "y", "z"]) == 0.0
    assert prefix_reuse(["x"]) == 1.0
    assert prefix_reuse([]) == 1.0


def test_superseded_rate():

    windows = [{"a"}, {"old"}, {"b", "old"}, set()]
    assert superseded_rate(windows, frozenset({"old"})) == 0.5
    assert superseded_rate([], frozenset({"old"})) == 0.0


def test_noise_share():

    docs = {"i1": "a", "i2": "n1", "i3": "n2", "i4": None}
    items = [item("i1", 10), item("i2", 20), item("i3", 10), item("i4", 10)]
    assert noise_share(items, lambda it: docs[it.id], frozenset({"n1", "n2"})) == pytest.approx(
        30 / 50
    )
    assert noise_share([], lambda it: None, frozenset({"n1"})) == 0.0
