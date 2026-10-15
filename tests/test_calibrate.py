# ruff: noqa: RUF001
import math

import pytest

from astromesh_context.calibrate import Calibrator
from astromesh_context.types import ContextItem, Signals


def item(id, score, source="s", **kw):
    return ContextItem(
        id=id, content=id, source=source, kind="knowledge", stability="volatile", score=score, **kw
    )


def cals(items, **kw):
    return {s.item.id: s for s in Calibrator().calibrate(items, **kw)}


def test_blend_two_items():
    out = cals([item("a", 0.9), item("b", 0.5)])
    assert out["a"].calibrated == pytest.approx(1.0)
    assert out["b"].calibrated == pytest.approx(0.25)
    assert out["a"].final == out["a"].calibrated


def test_equal_scores_minmax_is_one():
    out = cals([item("a", 0.5), item("b", 0.5)], modes={"s": "minmax"})
    assert out["a"].calibrated == 1.0
    assert out["b"].calibrated == 1.0


def test_sources_calibrated_independently():
    out = cals([item("a", 14.0, "bm25"), item("b", 2.0, "bm25"), item("c", 0.8, "vec")])
    assert out["a"].calibrated == pytest.approx(1.0)
    assert out["c"].calibrated == pytest.approx(1.0)


def test_weight_multiplies():
    out = cals([item("a", 0.9), item("b", 0.5)], weights={"s": 0.5})
    assert out["a"].calibrated == pytest.approx(0.5)


def test_stale_and_human_reviewed_adjust_and_record():
    out = cals(
        [
            item("a", 0.5, signals=Signals(stale=True)),
            item("b", 0.5, "t", signals=Signals(trust="human-reviewed")),
        ]
    )
    assert out["a"].adjustments == ("stale×0.7",)
    assert out["a"].calibrated == pytest.approx(0.7)
    assert out["b"].adjustments == ("human-reviewed×1.1",)
    assert out["b"].calibrated == pytest.approx(1.1)


def test_custom_multipliers_rendered():
    c = Calibrator(stale=0.5)
    out = c.calibrate([item("a", 0.5, signals=Signals(stale=True))])
    assert out[0].adjustments == ("stale×0.5",)


def test_deprecated_recorded_not_removed():
    out = cals([item("a", 0.5, signals=Signals(deprecated=True))])
    assert "a" in out
    assert out["a"].adjustments == ("deprecated",)


def test_pinned_is_one():
    out = cals([item("a", 0.1, pinned=True, signals=Signals(stale=True)), item("b", 0.9)])
    assert out["a"].calibrated == 1.0
    assert out["a"].adjustments == ()


def test_rank_ties_broken_by_id():
    out = cals([item("b", 0.5), item("a", 0.5)], modes={"s": "rank"})
    assert out["a"].calibrated == 1.0
    assert out["b"].calibrated == 0.5


def test_zscore_and_order_preserved():
    items = [item("x", 1.0), item("y", 3.0), item("z", 2.0)]
    res = Calibrator().calibrate(items, modes={"s": "zscore"})
    assert [r.item.id for r in res] == ["x", "y", "z"]
    assert res[2].calibrated == pytest.approx(0.5)
    std = math.sqrt(2 / 3)
    assert res[1].calibrated == pytest.approx(1 / (1 + math.exp(-1 / std)))


def test_zscore_zero_std_is_half():
    out = cals([item("a", 1.0), item("b", 1.0)], modes={"s": "zscore"})
    assert out["a"].calibrated == 0.5


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="bogus"):
        Calibrator().calibrate([item("a", 1.0)], modes={"s": "bogus"})
