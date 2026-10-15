"""ContextEval: run policies over a labeled dataset and report level-1 metrics."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import weakref
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astromesh_context.engine import ContextEngine
from astromesh_context.errors import MissingExtraError
from astromesh_context.eval.dataset import EvalCase, EvalDataset, nimbus_bundle
from astromesh_context.eval.metrics import (
    distractor_rate,
    noise_share,
    precision_per_token,
    prefix_reuse,
    recall_at_budget,
    superseded_rate,
)
from astromesh_context.policy import Everything, FixedSlots, ScoredKnapsack
from astromesh_context.scorer import LLMScorer
from astromesh_context.source import HistorySource, StaticSource

SYSTEM_PROMPT = (
    "You are Nimbus support. Answer only from the context provided. Cite the context id you used."
)
SESSION_SIZE = 5
ACK = "Entendido."
SPLIT_SIZE, SPLIT_OVERLAP = 800, 80
RAG_K, OKF_LIMIT = 20, 10
FIXED_SHARES = {"okf": 0.3, "rag": 0.5, "history": 0.2}

ConfigFactory = Callable[[EvalDataset, int], Awaitable[ContextEngine]]

_COLUMNS = (
    ("config", "config"),
    ("budget", "budget"),
    ("recall", "recall"),
    ("precision_per_token", "precision/token"),
    ("distractor_rate", "distractor rate"),
    ("superseded_rate", "superseded rate"),
    ("noise_share", "noise share"),
    ("tokens_per_call", "tokens/call"),
    ("candidate_tokens", "candidate tokens"),
    ("prefix_reuse", "prefix reuse"),
    ("stable_tokens", "stable tokens"),
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class Report:
    rows: list[dict]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        head = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
        rule = "|" + "|".join("---" for _ in _COLUMNS) + "|"
        lines = [head, rule]
        for row in self.rows:
            cells = []
            for key, _ in _COLUMNS:
                v = row[key]
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps({**self.meta, "rows": self.rows}, indent=2, ensure_ascii=False) + "\n"


def _sessions(cases: Sequence[EvalCase]) -> list[list[EvalCase]]:
    return [list(cases[i : i + SESSION_SIZE]) for i in range(0, len(cases), SESSION_SIZE)]


def _session_histories(session: Sequence[EvalCase]) -> list[list[dict]]:
    """Case j sees the session's small talk plus the previous cases' (query, ack) pairs."""
    base = list(session[0].history)
    out = []
    for j in range(len(session)):
        grown = list(base)
        for prev in session[:j]:
            grown.append({"role": "user", "content": prev.query})
            grown.append({"role": "assistant", "content": ACK})
        out.append(grown)
    return out


class ContextEval:
    def __init__(
        self,
        dataset: EvalDataset,
        configs: Mapping[str, ConfigFactory],
        budgets: Sequence[int],
        *,
        level: int = 1,
        answer: Callable | None = None,
        judge: Callable | None = None,
    ) -> None:
        if level != 1:
            raise ValueError(f"only level 1 is available, got level {level!r}")
        self.dataset = dataset
        self.configs = dict(configs)
        self.budgets = list(budgets)
        self.level = level
        self.answer = answer
        self.judge = judge

    async def _row(self, name: str, factory: ConfigFactory, budget: int) -> dict:
        ds = self.dataset
        engine = await factory(ds, budget)
        recalls: list[float] = []
        precisions: list[float] = []
        windows: list[set[str]] = []
        noise: list[float] = []
        pool: list[float] = []
        tokens: list[float] = []
        stable: list[float] = []
        reuse: list[float] = []
        for session in _sessions(ds.cases):
            hashes: list[str] = []
            for case, history in zip(session, _session_histories(session), strict=True):
                window = await engine.build(case.query, vars={"history": history})
                doc_ids = {d for d in map(ds.doc_id_of, window.items) if d is not None}
                recalls.append(recall_at_budget(doc_ids, case.relevant))
                precisions.append(precision_per_token(window.items, ds.doc_id_of, case.relevant))
                windows.append(doc_ids)
                noise.append(noise_share(window.items, ds.doc_id_of, ds.noise))
                pool.append(sum(e.tokens for e in window.trace.entries))
                tokens.append(window.tokens_used)
                stable.append(
                    sum(it.tokens or 0 for it in window.items if it.stability == "stable")
                )
                hashes.append(window.stable_prefix_hash)
            reuse.append(prefix_reuse(hashes))
        return {
            "config": name,
            "budget": budget,
            "recall": _mean(recalls),
            "precision_per_token": _mean(precisions),
            "distractor_rate": distractor_rate(windows, ds.distractors),
            "superseded_rate": superseded_rate(windows, ds.superseded),
            "noise_share": _mean(noise),
            "tokens_per_call": _mean(tokens),
            "candidate_tokens": _mean(pool),
            "prefix_reuse": _mean(reuse),
            "stable_tokens": _mean(stable),
        }

    async def run(self) -> Report:
        rows = [
            await self._row(name, factory, budget)
            for name, factory in self.configs.items()
            for budget in self.budgets
        ]
        meta = {
            "level": self.level,
            "corpus_sha256": self.dataset.meta.get("corpus_sha256"),
            "queries_sha256": self.dataset.meta.get("queries_sha256"),
            "cases": len(self.dataset.cases),
            "configs": list(self.configs),
            "budgets": self.budgets,
            "history_turns": self.dataset.meta.get("history_turns"),
            "config_meta": {name: getattr(f, "meta", None) for name, f in self.configs.items()},
            "command": None,  # the CLI fills it in
        }
        return Report(rows, meta)


class _Shared:
    """One ingestion and one OKF bundle, shared by every config, built on first use."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._lock = asyncio.Lock()
        self._pipeline: Any = None
        self._bundle: Path | None = None

    async def get(self, dataset: EvalDataset) -> tuple[Any, Path]:
        async with self._lock:
            if self._pipeline is None:
                try:
                    from astromesh_rag.indexes import MemoryIndex
                    from astromesh_rag.patterns.hybrid import hybrid
                    from astromesh_rag.pipeline import IngestPipeline
                    from astromesh_rag.retrievers import BM25Index
                    from astromesh_rag.splitters import RecursiveSplitter
                    from astromesh_rag.types import Document
                except ImportError:
                    raise MissingExtraError("install astromesh-context[rag]") from None

                index, bm25 = MemoryIndex(), BM25Index()
                await IngestPipeline(
                    RecursiveSplitter(SPLIT_SIZE, SPLIT_OVERLAP), self._embedder, index, bm25
                ).ingest([Document(d["text"], d["id"], d["metadata"]) for d in dataset.docs])
                self._pipeline = hybrid(self._embedder, index, bm25)
                tmp = tempfile.mkdtemp(prefix="nimbus-bundle-")
                weakref.finalize(self, shutil.rmtree, tmp, True)
                self._bundle = nimbus_bundle(Path(tmp))
            return self._pipeline, self._bundle  # type: ignore[return-value]


def default_configs(
    embedder: Any, *, complete: Callable[[str], Awaitable[str]] | None = None
) -> dict[str, ConfigFactory]:
    """everything, fixed_slots and scored_knapsack (+llm with ``complete``) over the same sources."""
    from astromesh_context.sources.okf import OKFIndexSource, OKFSource
    from astromesh_context.sources.rag import RAGSource

    shared = _Shared(embedder)
    common = {
        "rag_pattern": "hybrid",
        "rag_k": RAG_K,
        "okf_limit": OKF_LIMIT,
        "splitter": {"size": SPLIT_SIZE, "overlap": SPLIT_OVERLAP},
        "embedder": type(embedder).__name__,
    }

    def make(policy_cls: type, with_scorer: bool = False, **params: Any) -> ConfigFactory:
        async def factory(dataset: EvalDataset, budget: int) -> ContextEngine:
            pipeline, bundle = await shared.get(dataset)
            sources = [
                StaticSource(SYSTEM_PROMPT, name="system", pinned=True, stability="stable"),
                OKFIndexSource(bundle),
                OKFSource(bundle, limit=OKF_LIMIT),
                RAGSource(pipeline, k=RAG_K),
                HistorySource(key="history"),
            ]
            scorer = LLMScorer(complete) if with_scorer and complete is not None else None
            return ContextEngine(sources, budget=budget, policy=policy_cls(**params), scorer=scorer)

        factory.meta = {  # type: ignore[attr-defined]
            "policy": policy_cls.__name__,
            **params,
            **({"scorer": "LLMScorer"} if with_scorer else {}),
            **common,
        }
        return factory

    configs: dict[str, ConfigFactory] = {
        "everything": make(Everything, truncate=True),
        "fixed_slots": make(FixedSlots, shares=FIXED_SHARES),
        "scored_knapsack": make(ScoredKnapsack),
    }
    if complete is not None:
        configs["scored_knapsack+llm"] = make(ScoredKnapsack, with_scorer=True)
    return configs
