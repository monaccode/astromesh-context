from astromesh_context.order import order_items
from astromesh_context.types import ContextItem, ScoredItem


def _s(id, stability="volatile", source="s", kind="knowledge", final=0.5, order_key=None):
    it = ContextItem(id, f"c-{id}", source, kind, stability, order_key=order_key)
    return ScoredItem(it, final, final)


def test_stability_groups_in_order():
    items = [_s("v", "volatile"), _s("se", "session"), _s("st", "stable")]
    assert [i.id for i in order_items(items, ["s"])] == ["st", "se", "v"]


def test_stable_sorted_by_id_regardless_of_input():
    a = [_s("b", "stable"), _s("a", "stable"), _s("c", "stable")]
    assert [i.id for i in order_items(a, [])] == ["a", "b", "c"]
    assert [i.id for i in order_items(list(reversed(a)), [])] == ["a", "b", "c"]


def test_session_by_source_order_then_order_key_then_id():
    items = [
        _s("z", "session", source="other"),
        _s("b", "session", source="two", order_key="2"),
        _s("a", "session", source="two", order_key="1"),
        _s("c", "session", source="one"),
    ]
    assert [i.id for i in order_items(items, ["one", "two"])] == ["c", "a", "b", "z"]


def test_volatile_edges_first():
    items = [_s(str(n), final=n) for n in (1, 2, 3, 4, 5)]
    assert [i.id for i in order_items(items, [])] == ["5", "3", "1", "2", "4"]


def test_history_last_and_chronological():
    items = [
        _s("h2", "stable", kind="history", order_key="0002"),
        _s("v", "volatile"),
        _s("h1", "volatile", kind="history", order_key="0001"),
        _s("hn", "session", kind="history"),
        _s("st", "stable"),
    ]
    assert [i.id for i in order_items(items, [])] == ["st", "v", "hn", "h1", "h2"]
