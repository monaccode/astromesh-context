"""Shared helpers for the framework adapters."""

from __future__ import annotations

import importlib
from types import ModuleType

from astromesh_context.errors import MissingExtraError


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or raise MissingExtraError naming the extra that provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(f"install astromesh-context[{extra}]") from exc
