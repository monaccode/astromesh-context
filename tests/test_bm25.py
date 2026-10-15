from astromesh_context._bm25 import BM25


def test_bm25_ranks_and_never_negative():
    bm = BM25()
    bm.fit([["a", "b"], ["a", "c"], ["a", "a", "d"]])
    s = bm.scores(["b"])
    assert s[0] > 0 == s[1] == s[2]
    assert all(x >= 0 for x in bm.scores(["a"]))


def test_bm25_empty_cases():
    bm = BM25()
    bm.fit([])
    assert bm.scores(["a"]) == []
    bm.fit([[], ["x"]])
    assert bm.scores([]) == [0.0, 0.0]
