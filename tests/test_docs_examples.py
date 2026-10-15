"""Every Python snippet in the READMEs and docs/extending.md runs, in order, per file.

A fenced ```python block runs as written. A first line ``# needs: <module>[, <module>...]``
skips the block when any of those modules is missing. A ```yaml block whose first line is ``# file: <name>`` is written
to the working directory so a later snippet can load it. The snippets assume a few names that
the surrounding prose explains (``system_prompt``, ``messages``, ``pipeline``, ``model``); they
are provided below with offline fakes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = ["README.md", "README.es.md", "docs/extending.md"]
FENCE = re.compile(r"^```(python|yaml)\n(.*?)^```", re.S | re.M)


def blocks(path: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in FENCE.finditer((ROOT / path).read_text("utf-8"))]


def has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def rag_pipeline():
    from astromesh_rag.indexes import MemoryIndex
    from astromesh_rag.patterns.hybrid import hybrid
    from astromesh_rag.pipeline import IngestPipeline, Pipeline
    from astromesh_rag.retrievers import BM25Index
    from astromesh_rag.splitters import RecursiveSplitter
    from astromesh_rag.testing import HashEmbedder
    from astromesh_rag.types import Document

    embedder, index, bm25 = HashEmbedder(), MemoryIndex(), BM25Index()
    ingest = IngestPipeline(RecursiveSplitter(800, 80), embedder, index, bm25)
    pipeline = Pipeline(ingest, hybrid(embedder, index, bm25))
    docs = [
        Document("Refunds are accepted within 30 days of purchase.", "refunds"),
        Document("Shipping takes three business days by regional carrier.", "shipping"),
    ]
    asyncio.run(pipeline.ingest(docs))
    return pipeline


def base_namespace() -> dict:
    ns: dict = {
        "system_prompt": "You are the Nimbus support agent.",
        "messages": [
            {"role": "user", "content": "Hi, I bought the Team plan."},
            {"role": "assistant", "content": "Welcome. How can I help?"},
        ],
    }
    if has("astromesh_rag"):
        ns["pipeline"] = rag_pipeline()
    return ns


def fake_model(needs: str):
    if needs == "langchain":
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        class Fake(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        return Fake(messages=iter([AIMessage(content="done")]))
    from pydantic_ai.models.test import TestModel

    return TestModel(call_tools=[])


@pytest.mark.parametrize("path", FILES)
def test_snippets_run(path, tmp_path, monkeypatch):
    shutil.copytree(ROOT / "tests" / "fixtures" / "kb", tmp_path / "kb")
    monkeypatch.chdir(tmp_path)
    found = blocks(path)
    assert any(lang == "python" for lang, _ in found), f"{path} has no python snippet"
    namespace = base_namespace()
    ran = 0
    for lang, code in found:
        first = code.splitlines()[0]
        if lang == "yaml":
            if first.startswith("# file:"):
                (tmp_path / first.split(":", 1)[1].strip()).write_text(code, encoding="utf-8")
            continue
        needs = (
            [m.strip() for m in first.split(":", 1)[1].split(",") if m.strip()]
            if first.startswith("# needs:")
            else []
        )
        for module in needs:
            pytest.importorskip(module)
            if module in ("langchain", "pydantic_ai"):
                namespace["model"] = fake_model(module)
        exec(compile(code, f"{path}[{ran}]", "exec"), namespace)
        ran += 1
    assert ran


def test_readmes_share_the_same_snippet_count():
    counts = [sum(1 for lang, _ in blocks(p) if lang == "python") for p in FILES[:2]]
    assert counts[0] == counts[1]
