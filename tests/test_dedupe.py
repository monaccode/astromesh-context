from astromesh_context._text import content_hash, shingles, words
from astromesh_context.dedupe import dedupe
from astromesh_context.types import ContextItem, ScoredItem

BASE = " ".join(f"w{i}" for i in range(29))


def si(id, content, final=0.5, pinned=False):
    item = ContextItem(
        id=id, content=content, source="s", kind="knowledge", stability="session", pinned=pinned
    )
    return ScoredItem(item, final, final)


def test_text_helpers():
    assert words("La DEVOLUCIÓN, sí") == ["la", "devolución", "sí"]
    assert shingles("a b c") == frozenset({("a", "b", "c")})
    assert len(shingles("a b c d e f", 5)) == 2
    assert content_hash("A  b\n") == content_hash("a b")


def test_exact_duplicate_keeps_higher_final():
    a, b = si("a", "same text", 0.4), si("b", "same text", 0.9)
    kept, dropped = dedupe([a, b])
    assert kept == [b]
    assert dropped == [(a, "duplicate_of:b")]


def test_whitespace_and_case_insensitive_exact():
    a, b = si("a", "Hello   World\n", 0.9), si("b", "hello world", 0.1)
    kept, dropped = dedupe([a, b])
    assert kept == [a]
    assert dropped == [(b, "duplicate_of:a")]


def test_near_duplicate_over_threshold():
    a = si("a", BASE + " alpha", 0.9)
    b = si("b", BASE + " beta", 0.5)
    kept, dropped = dedupe([a, b])
    assert kept == [a]
    assert dropped == [(b, "duplicate_of:a")]


def test_below_threshold_both_kept():
    words_ = BASE.split()
    words_[14] = "changed"
    a, b = si("a", BASE + " alpha"), si("b", " ".join(words_) + " alpha")
    kept, dropped = dedupe([a, b])
    assert kept == [a, b]
    assert dropped == []


def test_pinned_always_wins():
    low = si("p", "same text", 0.1, pinned=True)
    high = si("h", "same text", 0.9)
    kept, dropped = dedupe([high, low])
    assert kept == [low]
    assert dropped == [(high, "duplicate_of:p")]


def test_pinned_duplicates_keep_best_final():
    a, b = si("a", "x y", 0.3, pinned=True), si("b", "x y", 0.6, pinned=True)
    kept, dropped = dedupe([a, b])
    assert kept == [b]
    assert dropped == [(a, "duplicate_of:b")]


def test_same_id_from_two_sources():
    a, b = si("k", "first content", 0.9), si("k", "totally different", 0.2)
    kept, dropped = dedupe([a, b])
    assert kept == [a]
    assert dropped == [(b, "duplicate_of:k")]


def test_accented_text_dedupes():
    text = "La DEVOLUCIÓN se procesa en 30 días hábiles tras recibir el producto en bodega"
    a, b = si("a", text, 0.8), si("b", text.lower(), 0.3)
    kept, dropped = dedupe([a, b])
    assert kept == [a]
    assert dropped == [(b, "duplicate_of:a")]


def test_kept_in_input_order():
    a, b, c = si("a", "one", 0.1), si("b", "two", 0.9), si("c", "three", 0.5)
    assert dedupe([a, b, c])[0] == [a, b, c]


def sig(id, content, final, pinned=False, **signals):
    from astromesh_context.types import Signals

    item = ContextItem(
        id=id,
        content=content,
        source="s",
        kind="knowledge",
        stability="session",
        pinned=pinned,
        signals=Signals(**signals),
    )
    return ScoredItem(item, final, final)


def test_dropped_deprecated_flag_is_inherited():
    kept_in = sig("rag:x", "same text", 0.9)
    dropped_in = sig("okf:x", "same text", 0.4, deprecated=True)
    kept, dropped = dedupe([kept_in, dropped_in])
    assert dropped == [(dropped_in, "duplicate_of:rag:x")]
    assert [k.item.id for k in kept] == ["rag:x"]
    assert kept[0].item.signals.deprecated is True
    assert kept[0].adjustments == ("inherited:deprecated:okf:x",)
    assert kept[0].final == 0.9


def test_stale_or_and_trust_citation_only_when_none():
    a = sig("a", "same text", 0.9, trust="human-reviewed", citation="A")
    b = sig("b", "same text", 0.4, stale=True, trust="unverified", citation="B")
    (k,), _ = dedupe([a, b])
    assert k.item.signals.stale is True
    assert k.item.signals.trust == "human-reviewed"
    assert k.item.signals.citation == "A"
    assert k.adjustments == ("inherited:stale:b",)

    c = sig("c", "same text", 0.9)
    (k2,), _ = dedupe([c, b])
    assert (k2.item.signals.trust, k2.item.signals.citation) == ("unverified", "B")
    assert set(k2.adjustments) == {"inherited:stale:b", "inherited:trust:b", "inherited:citation:b"}


def test_no_inheritance_returns_same_object():
    a, b = sig("a", "same text", 0.9), sig("b", "same text", 0.4)
    kept, _ = dedupe([a, b])
    assert kept[0] is a


def test_pinned_kept_item_inherits_but_stays_pinned_and_order_is_kept():
    first = si("first", "unrelated words here", 0.1)
    pin = sig("pin", "same text", 0.2, pinned=True)
    dup = sig("dup", "same text", 0.9, deprecated=True)
    kept, dropped = dedupe([first, pin, dup])
    assert [k.item.id for k in kept] == ["first", "pin"]
    assert kept[1].item.pinned
    assert kept[1].item.signals.deprecated
    assert dropped[0][1] == "duplicate_of:pin"
