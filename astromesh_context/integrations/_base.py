"""Shared helpers for the framework adapters."""

from __future__ import annotations

import copy
import importlib
from types import ModuleType
from typing import Any

from astromesh_context.errors import MissingExtraError


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise MissingExtraError naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(f"install astromesh-context[{extra}]") from exc


def with_adapter(obj: Any, adapter: Any) -> Any:
    """A shallow copy of a Compactor/Offloader using ``adapter``; the original is untouched."""
    if obj is None:
        return None
    clone = copy.copy(obj)
    clone.adapter = adapter
    return clone
