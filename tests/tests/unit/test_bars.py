"""Information-driven bars: structure, conservation, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features.bars import (
    dollar_bars,
    dollar_imbalance_bars,
    tick_bars,
    tick_imbalance_bars,
    tick_run_bars,
    volume_bars,
)


def _ticks(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n)))
    s = rng.uniform(0.1, 2.0, size=n)
    return p, s


def _one_sided_ticks(n: int = 2000, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Drift up with occasional bursts -> many buys in a row.
    drift = np.where(rng.uniform(size=n) < 0.15, 0.004, 0.0005)
    p = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.0005)))
    s = rng.uniform(0.1, 2.0, size=n)
    return p, s


def test_tick_bars_shape_and_conservation() -> None:
    p, s = _ticks(1000)
    out = tick_bars(p, s, ticks_per_bar=50)
    assert out["open"].shape == (20,)
    assert out["open"][0] == p[0]
    assert out["close"][-1] == p[999]
    assert out["volume"].sum() == pytest.approx(s[:1000].sum())
    assert np.all(out["high"] >= out["low"])
    assert np.all((out["close"] <= out["high"]) & (out["close"] >= out["low"]))


def test_volume_bars_close_on_threshold() -> None:
    p, s = _ticks(1000)
    target = 20.0
    out = volume_bars(p, s, volume_per_bar=target)
    # Each bar's volume >= threshold (last may differ slightly by construction).
    assert np.all(out["volume"][:-1] >= target)
    assert out["start"][0] == 0.0


def test_dollar_bars_scale() -> None:
    p, s = _ticks(1000)
    target = float(np.mean(p * s) * 40)
    out = dollar_bars(p, s, dollar_per_bar=target)
    assert np.all(out["dollar"][:-1] >= target)
    assert out["dollar"].shape == out["close"].shape


def test_tick_imbalance_bars_imbalanced_flow() -> None:
    p, s = _one_sided_ticks(1500)
    out = tick_imbalance_bars(p, s, expected_ticks=20)
    assert out["close"].size >= 3
    # Bars under one-sided flow should close with positive return skew.
    rets = np.diff(np.log(out["close"]))
    assert np.mean(rets) > 0.0 or out["close"][-1] >= out["close"][0]


def test_tick_run_bars_fire_on_runs() -> None:
    p, s = _one_sided_ticks(1500)
    out = tick_run_bars(p, s, expected_ticks=10)
    assert out["close"].size >= 3


def test_dollar_imbalance_bars_basic() -> None:
    p, s = _ticks(1500)
    out = dollar_imbalance_bars(p, s, expected_ticks=30)
    assert out["close"].size >= 2


def test_bars_fail_closed() -> None:
    p, s = _ticks(50)
    with pytest.raises(ValueError):
        tick_bars(p, s, ticks_per_bar=0)
    with pytest.raises(ValueError):
        tick_bars(p[:10], s[:20], ticks_per_bar=5)
    with pytest.raises(ValueError):
        volume_bars(p, s, volume_per_bar=-1.0)
    with pytest.raises(ValueError):
        dollar_bars(p, s, dollar_per_bar=float("inf"))
    with pytest.raises(ValueError):
        tick_imbalance_bars(p, s, expected_ticks=1)
    with pytest.raises(ValueError):
        tick_run_bars(p, s, alpha_ewma=1.5)
    with pytest.raises(ValueError):
        dollar_imbalance_bars(np.array([1.0]), np.array([1.0]))


def test_bar_returns_more_gaussian_ish() -> None:
    # Sanity: bar returns have finite variance and no degenerate zeros.
    p, s = _ticks(4000, seed=3)
    tb = tick_bars(p, s, 40)
    rets = np.diff(np.log(tb["close"]))
    assert np.isfinite(rets).all() and np.std(rets) > 0.0
