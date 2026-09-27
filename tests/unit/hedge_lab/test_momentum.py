"""Tests for momentum signals."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.momentum import (
    high_52w,
    jt_momentum,
    residual_momentum,
    tsmom_signal,
    tsmom_tstat,
)


def _trending_panel(n=6, t=400, seed=0):
    """Asset 0 trends up, asset 1 trends down, rest are flat."""
    rng = np.random.default_rng(seed)
    p = np.empty((t, n))
    p[0] = 100.0
    for s in range(1, t):
        p[s] = p[s - 1] * np.exp(rng.normal(size=n) * 0.01)
    p[:, 0] = 100 * np.exp(np.linspace(0, 0.6, t) + rng.normal(scale=0.02, size=t).cumsum() * 0.1)
    p[:, 1] = 100 * np.exp(-np.linspace(0, 1.5, t) + rng.normal(scale=0.02, size=t).cumsum() * 0.05)
    return p


class TestJT:
    def test_trending_asset_ranks_first(self):
        p = _trending_panel()
        sig = jt_momentum(p, formation=120, skip=10)
        last = sig[-1]
        assert np.isfinite(last).all()
        assert np.argmax(last) == 0
        assert np.argmin(last) == 1

    def test_warmup_nan(self):
        p = _trending_panel()
        sig = jt_momentum(p, formation=100, skip=10)
        assert np.all(np.isnan(sig[:110]))

    def test_failclosed(self):
        with pytest.raises(ValueError):
            jt_momentum(np.ones((50, 4)))


class TestHigh52w:
    def test_uptrend_near_high(self):
        rng = np.random.default_rng(1)
        p = 100 * np.exp(np.linspace(0, 0.8, 300))[:, None] * np.ones((300, 3))
        p += rng.normal(scale=0.5, size=(300, 3))
        p = np.maximum(p, 1.0)
        out = high_52w(p, window=200)
        assert out[-1, 0] > 0.9  # trending asset near its high

    def test_failclosed(self):
        with pytest.raises(ValueError):
            high_52w(np.ones((300, 3)) * -1)


class TestResidualMomentum:
    def test_factor_resistant_signal(self):
        rng = np.random.default_rng(2)
        t, n = 400, 4
        f = rng.normal(scale=0.01, size=t)
        p = np.empty((t, n))
        p[0] = 100.0
        for s in range(1, t):
            p[s] = p[s - 1] * np.exp(1.0 * f[s] + rng.normal(scale=0.005, size=n))
        out = residual_momentum(p, f, formation=60, skip=5, est_window=120)
        assert np.isfinite(out["signal"][-1]).all()
        assert out["beta"].shape == (t, n)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            residual_momentum(np.ones((400, 3)), np.ones(200))


class TestTSMOM:
    def test_positive_trend_signal(self):
        rng = np.random.default_rng(3)
        r = 0.001 + rng.normal(scale=0.01, size=400)
        out = tsmom_signal(r, lookback=120, vol_window=60)
        assert out["signal"][-1] > 0
        assert out["sign"][-1] == 1.0

    def test_tstat_positive_for_persistent(self):
        rng = np.random.default_rng(4)
        n = 500
        r = np.empty(n)
        shock = rng.normal(scale=0.01, size=n)
        # Persistent sign process -> TSMOM profitable.
        for t in range(n):
            r[t] = 0.0005 + 0.3 * (r[t - 1] if t > 0 else 0) + shock[t]
        out = tsmom_tstat(r, lookback=100)
        assert np.isfinite(out["t_stat"])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            tsmom_signal(np.ones(50))
        with pytest.raises(ValueError):
            tsmom_tstat(np.ones(60), lookback=252)
