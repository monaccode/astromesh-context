import pytest

pytest.importorskip("astromesh_rag")

from astromesh_rag.indexes import MemoryIndex
from astromesh_rag.patterns.hybrid import hybrid
from astromesh_rag.pipeline import IngestPipeline, Pipeline
from astromesh_rag.retrievers import BM25Index
from astromesh_rag.splitters import RecursiveSplitter
from astromesh_rag.testing import HashEmbedder
from astromesh_rag.types import Document

from astromesh_context.sources.rag import RAGSource
from astromesh_context.types import Budget, BuildContext

DOCS = [
    Document("Refunds are accepted within 30 days of purchase.", "refunds"),
    Document("Shipping takes three business days by regional carrier.", "shipping"),
    Document("Passwords can be reset from the account settings page.", "passwords"),
]


def make_ctx():
    return BuildContext(query="q", budget=Budget(total=100), deadline=0.0, vars={})


async def make_pipeline():
    embedder, index, bm25 = HashEmbedder(), MemoryIndex(), BM25Index()
    ingest = IngestPipeline(RecursiveSplitter(800, 80), embedder, index, bm25)
    pipeline = Pipeline(ingest, hybrid(embedder, index, bm25))
    await pipeline.ingest(DOCS)
    return pipeline


async def test_rag_items_carry_doc_id_and_details():
    source = RAGSource(await make_pipeline(), k=3)
    items = await source.candidates("refunds within 30 days", make_ctx())
    assert items
    top = items[0]
    assert top.id.startswith("rag:")
    assert top.source == "rag"
    assert (top.kind, top.stability, top.pinned) == ("knowledge", "volatile", False)
    assert top.meta["doc_id"] == "refunds"
    assert top.signals.citation == "refunds"
    assert "bm25" in top.meta
    assert top.score > 0
    assert "30 days" in top.content


async def test_rag_spec_path_loads_pipeline(monkeypatch, tmp_path):
    import astromesh_rag.spec as spec

    seen = {}
    pipeline = await make_pipeline()
    monkeypatch.setattr(spec, "load_spec_file", lambda p: seen.setdefault("path", p) or "spec")
    monkeypatch.setattr(spec, "build", lambda s: pipeline)
    source = RAGSource(tmp_path / "rag.yaml", name="kb")
    assert seen["path"] == tmp_path / "rag.yaml"
    items = await source.candidates("shipping carrier", make_ctx())
    assert items
    assert items[0].source == "kb"


async def test_rag_duck_typed_pipeline_needs_no_extra():
    from types import SimpleNamespace as Ns

    class Fake:
        async def query(self, q, k):
            chunk = Ns(id="c1", doc_id="d1", text="hello")
            return Ns(hits=[Ns(chunk=chunk, score=0.5, details={"vector": 0.4})])

    (item,) = await RAGSource(Fake()).candidates("q", make_ctx())
    assert item.id == "rag:c1"
    assert item.meta == {"doc_id": "d1", "vector": 0.4}
