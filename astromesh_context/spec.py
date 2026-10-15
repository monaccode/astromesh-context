"""Load a ``kind: ContextPolicy`` YAML document into a ContextEngine."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from astromesh_context.engine import ContextEngine
from astromesh_context.errors import SpecError
from astromesh_context.policy import Everything, FixedSlots, Policy, ScoredKnapsack
from astromesh_context.scorer import LLMScorer, Scorer
from astromesh_context.source import (
    CallableSource,
    HistorySource,
    Source,
    SourceConfig,
    StaticSource,
)
from astromesh_context.tokens import CharEstimator, TiktokenTokenizer, Tokenizer

_API_VERSION = "astromesh/v1"
_KIND = "ContextPolicy"
_MODES = ("blend", "rank", "minmax", "zscore")
_SHARED = ("weight", "max_tokens", "min_items", "timeout", "mode")
_POLICIES: dict[str, tuple[Callable[..., Policy], tuple[str, ...]]] = {
    "scored_knapsack": (ScoredKnapsack, ("beta", "min_score", "include_deprecated")),
    "fixed_slots": (FixedSlots, ("shares", "include_deprecated")),
    "everything": (Everything, ("truncate", "include_deprecated")),
}
_SCORER_KEYS = ("top_n", "alpha", "timeout")
_STATIC_ONLY = ("pinned", "kind", "stability")
_KINDS = ("instruction", "knowledge", "memory", "history", "tool_result")
_STABILITIES = ("stable", "session", "volatile")
# type -> keys accepted besides `type`, `name` and the shared SourceConfig keys
_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "static": ("text", "text_file", *_STATIC_ONLY),
    "okf_index": ("bundle", "path", "pinned"),
    "okf": ("bundle", "limit"),
    "rag": ("pipeline", "k"),
    "history": ("key",),
    "callable": ("object", *_STATIC_ONLY),
}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(path, "must be a mapping")
    return value


def _check_keys(block: Mapping[str, Any], allowed: tuple[str, ...], path: str) -> None:
    for key in block:
        if key not in allowed:
            raise SpecError(f"{path}.{key}" if path else str(key), "unknown key")


def _string(block: Mapping[str, Any], key: str, path: str, *, required: bool = False) -> Any:
    if key not in block:
        if required:
            raise SpecError(f"{path}.{key}", "is required")
        return None
    if not isinstance(block[key], str) or not block[key]:
        raise SpecError(f"{path}.{key}", "must be a non-empty string")
    return block[key]


def _resolve(value: str, base_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def _tokenizer(value: Any) -> Tokenizer:
    if value is None or value == "chars":
        return CharEstimator()
    if isinstance(value, str) and value.startswith("tiktoken:") and len(value) > 9:
        return TiktokenTokenizer(value[9:])
    raise SpecError("spec.tokenizer", "must be 'chars' or 'tiktoken:<encoding>'")


def _policy(value: Any) -> Policy:
    if value is None:
        return ScoredKnapsack()
    block = {"type": value} if isinstance(value, str) else _mapping(value, "spec.policy")
    kind = block.get("type")
    if kind not in _POLICIES:
        raise SpecError("spec.policy", f"type must be one of {sorted(_POLICIES)}, got {kind!r}")
    factory, params = _POLICIES[kind]
    _check_keys(block, ("type", *params), "spec.policy")
    kwargs = {k: v for k, v in block.items() if k != "type"}
    if kind == "fixed_slots":
        shares = kwargs.get("shares")
        if not isinstance(shares, Mapping) or not all(_is_num(v) for v in shares.values()):
            raise SpecError("spec.policy.shares", "must be a mapping of source name to number")
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise SpecError("spec.policy", str(exc)) from exc
    except ValueError as exc:
        raise SpecError("spec.policy.shares", str(exc)) from exc


def _scorer(value: Any, complete: Callable[[str], Awaitable[str]] | None) -> Scorer | None:
    if value is None:
        return None
    block = {"type": value} if isinstance(value, str) else _mapping(value, "spec.scorer")
    if block.get("type") != "llm":
        raise SpecError("spec.scorer", f"type must be 'llm', got {block.get('type')!r}")
    _check_keys(block, ("type", *_SCORER_KEYS), "spec.scorer")
    for key in _SCORER_KEYS:
        if key in block and not _is_num(block[key]):
            raise SpecError(f"spec.scorer.{key}", "must be a number")
    if "top_n" in block and not (_is_int(block["top_n"]) and block["top_n"] > 0):
        raise SpecError("spec.scorer.top_n", "must be an integer > 0")
    if complete is None:
        raise SpecError("spec.scorer", "an llm scorer needs the `complete` callable")
    return LLMScorer(complete, **{k: v for k, v in block.items() if k != "type"})


def _shared(block: Mapping[str, Any], path: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SHARED:
        if key not in block:
            continue
        v = block[key]
        if key == "weight" and not _is_num(v):
            raise SpecError(f"{path}.weight", "must be a number")
        if key in ("max_tokens", "min_items") and not (_is_int(v) and v >= 0):
            raise SpecError(f"{path}.{key}", "must be an integer >= 0")
        if key == "timeout" and not (_is_num(v) and v > 0):
            raise SpecError(f"{path}.timeout", "must be a number > 0")
        if key == "mode" and v not in _MODES:
            raise SpecError(f"{path}.mode", f"must be one of {list(_MODES)}")
        out[key] = v
    return out


def _lookup(objects: Mapping[str, Any], key: Any, path: str) -> Any:
    if not isinstance(key, str) or key not in objects:
        raise SpecError(path, f"object {key!r} is not in `objects`")
    return objects[key]


def _static_extras(block: Mapping[str, Any], path: str) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if "pinned" in block:
        if not isinstance(block["pinned"], bool):
            raise SpecError(f"{path}.pinned", "must be a boolean")
        extra["pinned"] = block["pinned"]
    for key, allowed in (("kind", _KINDS), ("stability", _STABILITIES)):
        if key in block:
            if block[key] not in allowed:
                raise SpecError(f"{path}.{key}", f"must be one of {list(allowed)}")
            extra[key] = block[key]
    return extra


def _source(
    raw: Any, index: int, base_dir: Path, objects: Mapping[str, Any]
) -> Source | SourceConfig:
    path = f"spec.sources[{index}]"
    block = _mapping(raw, path)
    kind = block.get("type")
    if kind not in _SOURCE_KEYS:
        raise SpecError(f"{path}.type", f"must be one of {sorted(_SOURCE_KEYS)}, got {kind!r}")
    _check_keys(block, ("type", "name", *_SHARED, *_SOURCE_KEYS[kind]), path)
    shared = _shared(block, path)
    name = _string(block, "name", path)
    named: dict[str, Any] = {"name": name} if name else {}

    source: Source
    if kind == "static":
        text, text_file = block.get("text"), _string(block, "text_file", path)
        if (text is None) == (text_file is None):
            raise SpecError(f"{path}.text", "give exactly one of `text` or `text_file`")
        if text_file is not None:
            file = _resolve(text_file, base_dir)
            try:
                text = file.read_text(encoding="utf-8")
            except OSError as exc:
                raise SpecError(f"{path}.text_file", f"cannot read {file}: {exc}") from exc
        elif not isinstance(text, str):
            raise SpecError(f"{path}.text", "must be a string")
        source = StaticSource(text, **named, **_static_extras(block, path))
    elif kind == "callable":
        _string(block, "object", path, required=True)
        fn = _lookup(objects, block["object"], f"{path}.object")
        if not named:
            raise SpecError(f"{path}.name", "is required for callable sources")
        for key in _STATIC_ONLY:
            if key in block:
                raise SpecError(f"{path}.{key}", "is not supported for callable sources")
        source = CallableSource(fn, **named)
    elif kind == "history":
        key = _string(block, "key", path)
        source = HistorySource(**named, **({"key": key} if key else {}))
    elif kind == "rag":
        from astromesh_context.sources.rag import RAGSource

        if "pipeline" not in block:
            raise SpecError(f"{path}.pipeline", "is required")
        pipeline = block["pipeline"]
        if isinstance(pipeline, Mapping):
            _check_keys(pipeline, ("object",), f"{path}.pipeline")
            _string(pipeline, "object", f"{path}.pipeline", required=True)
            pipeline = _lookup(objects, pipeline["object"], f"{path}.pipeline.object")
        elif isinstance(pipeline, str) and pipeline:
            pipeline = _resolve(pipeline, base_dir)
        else:
            raise SpecError(f"{path}.pipeline", "must be a path or {object: <key>}")
        k = block.get("k")
        if k is not None and not (_is_int(k) and k > 0):
            raise SpecError(f"{path}.k", "must be an integer > 0")
        source = RAGSource(pipeline, **({"k": k} if k else {}), **named)
    else:
        bundle = _resolve(_string(block, "bundle", path, required=True), base_dir)
        if kind == "okf_index":
            from astromesh_context.sources.okf import OKFIndexSource

            if block.get("pinned", True) is not True:
                raise SpecError(f"{path}.pinned", "okf_index is always pinned")
            idx_path = _string(block, "path", path)
            source = OKFIndexSource(bundle, **({"path": idx_path} if idx_path else {}), **named)
        else:
            from astromesh_context.sources.okf import OKFSource

            limit = block.get("limit")
            if limit is not None and not (_is_int(limit) and limit > 0):
                raise SpecError(f"{path}.limit", "must be an integer > 0")
            source = OKFSource(bundle, **({"limit": limit} if limit else {}), **named)

    return SourceConfig(source, **shared) if shared else source


def load_engine_dict(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
    complete: Callable[[str], Awaitable[str]] | None = None,
    objects: Mapping[str, Any] = {},
) -> ContextEngine:
    """Build a ContextEngine from an already parsed ``kind: ContextPolicy`` document."""
    base_dir = Path(base_dir)
    doc = _mapping(raw, "")
    _check_keys(doc, ("apiVersion", "kind", "metadata", "spec"), "")
    if doc.get("apiVersion") != _API_VERSION:
        raise SpecError("apiVersion", f"must be {_API_VERSION!r}, got {doc.get('apiVersion')!r}")
    if doc.get("kind") != _KIND:
        raise SpecError("kind", f"must be {_KIND!r}, got {doc.get('kind')!r}")
    spec = _mapping(doc.get("spec"), "spec")
    _check_keys(spec, ("budget", "tokenizer", "policy", "scorer", "sources"), "spec")

    budget = spec.get("budget")
    if not (_is_int(budget) and budget > 0):
        raise SpecError("spec.budget", "is required and must be an integer > 0")
    sources_raw = spec.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise SpecError("spec.sources", "must be a non-empty list")

    tokenizer = _tokenizer(spec.get("tokenizer"))
    policy = _policy(spec.get("policy"))
    scorer = _scorer(spec.get("scorer"), complete)
    sources = [_source(s, i, base_dir, objects) for i, s in enumerate(sources_raw)]
    return ContextEngine(sources, budget=budget, policy=policy, tokenizer=tokenizer, scorer=scorer)


def load_engine(
    path: str | Path,
    *,
    complete: Callable[[str], Awaitable[str]] | None = None,
    objects: Mapping[str, Any] = {},
) -> ContextEngine:
    """Build a ContextEngine from a ``kind: ContextPolicy`` YAML file."""
    file = Path(path)
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpecError("", f"cannot read {file}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SpecError("", f"invalid YAML in {file}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SpecError("", "top level must be a mapping")
    return load_engine_dict(raw, base_dir=file.parent, complete=complete, objects=objects)
