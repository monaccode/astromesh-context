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
