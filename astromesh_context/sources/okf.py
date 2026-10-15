"""OKF sources: a bundle's index (pinned) and its concepts ranked by BM25."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from astromesh_context._bm25 import BM25
from astromesh_context._text import words
from astromesh_context.errors import MissingExtraError
from astromesh_context.types import BuildContext, ContextItem, Signals


def _okf():
    try:
        import astromesh_okf.bundle as bundle
        import astromesh_okf.concept as concept
    except ImportError:
        raise MissingExtraError("install astromesh-context[okf]") from None
    return bundle, concept


class OKFIndexSource:
    """The bundle's index as one pinned, stable instruction."""

    def __init__(self, bundle: str | Path, *, path: str = "/", name: str = "okf_index") -> None:
        self._bundle_mod, _ = _okf()
        self.name = name
        self._root = Path(bundle)
        self._path = path

    def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]:
        text = self._bundle_mod.Bundle(self._root).index(self._path)
        return [
            ContextItem(
                id=f"okf_index:{self._path}",
                content=text,
                source=self.name,
                kind="instruction",
                stability="stable",
                pinned=True,
            )
        ]


class OKFSource:
    """Concepts of a bundle ranked against the query with BM25 (header counted twice)."""

    def __init__(self, bundle: str | Path, *, limit: int = 10, name: str = "okf") -> None:
        _, self._concept_mod = _okf()
        self.name = name
        self._root = Path(bundle).resolve()
        self._limit = limit

    async def candidates(self, query: str, ctx: BuildContext) -> list[ContextItem]:
        terms = words(query)
        if not terms:
            return []
        return await asyncio.to_thread(self._search, terms)

    def _search(self, terms: list[str]) -> list[ContextItem]:
        mod = self._concept_mod
        root = self._root
        now = datetime.now(timezone.utc)
        concepts = []
        for path in sorted(root.rglob("*.md")):
            rel = path.relative_to(root).parts
            if path.name in mod.RESERVED or any(p.startswith(".") for p in rel):
                continue
            if not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            try:
                concepts.append(mod.parse_concept(root, path, now))
            except mod.InvalidConceptError:
                continue
        docs = []
        for c in concepts:
            header = words(" ".join([c.title, c.description, *c.tags]))
            docs.append(header + header + words(c.body))
        bm25 = BM25()
        bm25.fit(docs)
        ranked = sorted(
            (pair for pair in zip(bm25.scores(terms), concepts, strict=True) if pair[0] > 0),
            key=lambda pair: -pair[0],
        )[: self._limit]
        return [
            ContextItem(
                id=f"okf:{c.id}",
                content=f"# {c.title}\n\n{c.body}",
                source=self.name,
                kind="knowledge",
                stability="volatile",
                score=score,
                signals=Signals(
                    trust=c.trust,
                    stale=c.stale,
                    deprecated=c.status == "deprecated",
                    citation=c.id,
                ),
                meta={"concept_id": c.id},
            )
            for score, c in ranked
        ]
