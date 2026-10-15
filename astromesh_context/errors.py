"""Public exceptions. Everything raised on purpose subclasses ContextError."""

from __future__ import annotations


class ContextError(Exception):
    """Base class for all astromesh-context errors."""


class BudgetExceededError(ContextError):
    """Pinned items (or an `Everything` policy) do not fit in the budget."""

    def __init__(self, budget: int, pinned: list[tuple[str, int]]) -> None:
        self.budget = budget
        self.pinned = pinned
        detail = ", ".join(f"{item_id}={tokens}" for item_id, tokens in pinned)
        super().__init__(
            f"items need {sum(t for _, t in pinned)} tokens, budget is {budget}: {detail}"
        )


class SpecError(ContextError):
    """Invalid ContextPolicy spec; `path` is the offending field path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


class ConfigError(ContextError, ValueError):
    """Invalid engine or policy configuration, rejected at construction."""


class MissingExtraError(ContextError):
    """An optional dependency is not installed."""
