"""astromesh-context: a context-window engine for AI agents."""

__version__ = "0.1.0"

from astromesh_context.calibrate import Calibrator
from astromesh_context.engine import ContextEngine
from astromesh_context.errors import (
    BudgetExceededError,
    ContextError,
    MissingExtraError,
    SpecError,
)
from astromesh_context.policy import Everything, FixedSlots, ScoredKnapsack
from astromesh_context.scorer import LLMScorer, openai_chat, openai_compatible
from astromesh_context.source import (
    CallableSource,
    HistorySource,
    SourceConfig,
    StaticSource,
)
from astromesh_context.spec import load_engine
from astromesh_context.tokens import CharEstimator
from astromesh_context.types import Budget, ContextItem, Signals, Trace, Window

__all__ = [
    "Budget",
    "BudgetExceededError",
    "Calibrator",
    "CallableSource",
    "CharEstimator",
    "ContextEngine",
    "ContextError",
    "ContextItem",
    "Everything",
    "FixedSlots",
    "HistorySource",
    "LLMScorer",
    "MissingExtraError",
    "ScoredKnapsack",
    "Signals",
    "SourceConfig",
    "SpecError",
    "StaticSource",
    "Trace",
    "Window",
    "__version__",
    "load_engine",
    "openai_chat",
    "openai_compatible",
]
