"""ContextEval: measure context policies on a labeled corpus (public API only)."""

from astromesh_context.eval.dataset import EvalCase, EvalDataset, nimbus, nimbus_bundle
from astromesh_context.eval.metrics import (
    distractor_rate,
    precision_per_token,
    prefix_reuse,
    recall_at_budget,
)
from astromesh_context.eval.runner import ContextEval, Report, default_configs

__all__ = [
    "ContextEval",
    "EvalCase",
    "EvalDataset",
    "Report",
    "default_configs",
    "distractor_rate",
    "nimbus",
    "nimbus_bundle",
    "precision_per_token",
    "prefix_reuse",
    "recall_at_budget",
]
