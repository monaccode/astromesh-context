# ruff: noqa: RUF001
"""Per-source score calibration so scores from different sources are comparable."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from astromesh_context.types import ContextItem, ScoredItem

_MODES = ("blend", "rank", "minmax", "zscore")


class Calibrator:
    def __init__(self, stale: float = 0.7, human_reviewed: float = 1.1) -> None:
        self.stale = stale
        self.human_reviewed = human_reviewed

    def calibrate(
        self,
        items: Sequence[ContextItem],
        *,
        weights: Mapping[str, float] = {},
        modes: Mapping[str, str] = {},
    ) -> list[ScoredItem]:
        groups: dict[str, list[ContextItem]] = {}
        for it in items:
            groups.setdefault(it.source, []).append(it)

        values: dict[int, float] = {}
        for source, group in groups.items():
            mode = modes.get(source, "blend")
            if mode not in _MODES:
                raise ValueError(f"unknown calibration mode {mode!r} for source {source!r}")
            for it, v in zip(group, self._mode_values(group, mode), strict=True):
                values[id(it)] = v

        out: list[ScoredItem] = []
        for it in items:
            if it.pinned:
                out.append(ScoredItem(it, 1.0, 1.0, ()))
                continue
            cal = values[id(it)] * weights.get(it.source, 1.0)
            adjustments: list[str] = []
            if it.signals.stale:
                cal *= self.stale
                adjustments.append(f"stale×{self.stale:g}")
            if it.signals.trust == "human-reviewed":
                cal *= self.human_reviewed
                adjustments.append(f"human-reviewed×{self.human_reviewed:g}")
            if it.signals.deprecated:
                adjustments.append("deprecated")
            out.append(ScoredItem(it, cal, cal, tuple(adjustments)))
        return out

    @staticmethod
    def _mode_values(group: list[ContextItem], mode: str) -> list[float]:
        n = len(group)
        raw = [it.score for it in group]

        def rank() -> list[float]:
            order = sorted(range(n), key=lambda i: (-raw[i], group[i].id))
            res = [0.0] * n
            for r, i in enumerate(order):
                res[i] = 1 - r / n
            return res

        def minmax() -> list[float]:
            lo, hi = min(raw), max(raw)
            if hi == lo:
                return [1.0] * n
            return [(x - lo) / (hi - lo) for x in raw]

        def zscore() -> list[float]:
            mean = sum(raw) / n
            std = math.sqrt(sum((x - mean) ** 2 for x in raw) / n)
            if std == 0:
                return [0.5] * n
            return [1 / (1 + math.exp(-(x - mean) / std)) for x in raw]

        if mode == "rank":
            return rank()
        if mode == "minmax":
            return minmax()
        if mode == "zscore":
            return zscore()
        return [0.5 * a + 0.5 * b for a, b in zip(rank(), minmax(), strict=True)]
