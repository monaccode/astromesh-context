"""RAGSource: retrieval hits from an astromesh-rag pipeline as context candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from astromesh_context.errors import MissingExtraError
from astromesh_context.types import BuildContext, ContextItem, Signals


def _load(path: str | Path) -> Any:
    try:
        from astromesh_rag.spec import build, load_spec_file
    except ImportError:
        raise MissingExtraError("install astromesh-context[rag]") from None
    return build(load_spec_file(path))


class RAGSource:
    """`pipeline` is a spec file path, or any object with `async query(q, k)`."""

    def __init__(self, pipeline: Any, *, k: int = 20, name: str = "rag") -> None:
        self.name = name
        self._k = k
        self._pipeline = _load(pipeline) if isinstance(pipeline, (str, Path)) else pipeline

    async def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]:
        result = await self._pipeline.query(query, self._k)
        return [
            ContextItem(
                id=f"rag:{hit.chunk.id}",
                content=hit.chunk.text,
                source=self.name,
                kind="knowledge",
                stability="volatile",
                score=hit.score,
                signals=Signals(citation=hit.chunk.doc_id),
                meta={"doc_id": hit.chunk.doc_id, **hit.details},
            )
            for hit in result.hits
        ]
