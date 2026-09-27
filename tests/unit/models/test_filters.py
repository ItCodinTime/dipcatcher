"""Tests for trend/cycle decomposition filters."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.filters import (
    baxter_king,
    beveridge_nelson,
    christiano_fitzgerald,
    corbae_ouliaris_band,
    hamilton_filter,
    hp_filter,
    ravn_uhlig_lambda,
)


def _sine_series(n: int = 400, period: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 3.0 * np.sin(2 * np.pi * t / period) + 0.02 * t + rng.normal(scale=0.1, size=n)


class TestHP:
    def test_removes_trend(self):
        v = _sine_series()
        out = hp_filter(v, lam=1600)
        assert out["trend"].shape == v.shape
        # Cycle captures the sine: correlate with true sine.
        t = np.arange(v.size)
        true = 3.0 * np.sin(2 * np.pi * t / 20.0)
        assert np.corrcoef(out["cycle"][50:-50], true[50:-50])[0, 1] > 0.9
        # Trend is smooth: tiny second differences.
        d2 = np.diff(out["trend"], 2)
        assert np.abs(d2).max() < 0.1

    def test_lambda_sensitivity(self):
        v = _sine_series()
        smooth = hp_filter(v, lam=1e6)["trend"]
        loose = hp_filter(v, lam=10.0)["trend"]
        assert np.var(np.diff(smooth, 2)) < np.var(np.diff(loose, 2))

    def test_ravn_uhlig(self):
        assert ravn_uhlig_lambda(4.0) == 1600.0
        assert ravn_uhlig_lambda(1.0) == pytest.approx(6.25)
        assert ravn_uhlig_lambda(12.0) == pytest.approx(129600.0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hp_filter(np.ones(5), lam=-1)
        with pytest.raises(ValueError):
            hp_filter(np.full(20, np.nan))


class TestBandpass:
    def test_bk_isolates_cycle(self):
        v = _sine_series(n=400, period=20.0)
        out = baxter_king(v, low=10.0, high=40.0, k=12)
        cyc = out["cycle"]
        assert np.all(np.isnan(cyc[:12])) and np.all(np.isnan(cyc[-12:]))
        t = np.arange(v.size)
        true = 3.0 * np.sin(2 * np.pi * t / 20.0)
        core = slice(24, -24)
        assert np.corrcoef(cyc[core], true[core])[0, 1] > 0.95
        # Weights sum to ~0.
        assert abs(float(out["weights"].sum())) < 1e-8

    def test_cf_no_burnin(self):
        v = _sine_series(n=300, period=15.0)
        cyc = christiano_fitzgerald(v, low=6.0, high=30.0)["cycle"]
        assert np.all(np.isfinite(cyc))
        t = np.arange(v.size)
        true = 3.0 * np.sin(2 * np.pi * t / 15.0)
        assert np.corrcoef(cyc[10:-10], true[10:-10])[0, 1] > 0.9

    def test_co_band(self):
        v = _sine_series(n=256, period=16.0)
        cyc = corbae_ouliaris_band(v, low=8.0, high=32.0)["cycle"]
        t = np.arange(v.size)
        true = 3.0 * np.sin(2 * np.pi * t / 16.0)
        assert np.corrcoef(cyc, true)[0, 1] > 0.98

    def test_failclosed(self):
        with pytest.raises(ValueError):
            baxter_king(np.ones(50), low=30.0, high=10.0)
        with pytest.raises(ValueError):
            christiano_fitzgerald(np.ones(30), low=1.0, high=4.0)


class TestHamiltonBN:
    def test_hamilton_cycle_stationary(self):
        rng = np.random.default_rng(1)
        n = 300
        rw = np.cumsum(rng.normal(size=n))
        out = hamilton_filter(rw, h=8, p=4)
        cyc = out["cycle"]
        assert np.all(np.isnan(cyc[:11])) and np.all(np.isfinite(cyc[11:]))
        # RW residuals follow MA(h-1): autocorr cuts off at lag h.
        c = cyc[11:]
        ac8 = np.corrcoef(c[:-8], c[8:])[0, 1]
        assert abs(ac8) < 0.3
        # Stationary: variance stable across halves (a trend would grow).
        assert np.var(c[c.size // 2 :]) / np.var(c[: c.size // 2]) < 2.5

    def test_bn_random_walk_cycle_zero(self):
        rng = np.random.default_rng(2)
        rw = np.cumsum(rng.normal(size=500))
        out = beveridge_nelson(rw, ar_lags=1)
        # For a pure RW, phi_sum ~ 0 and cycle ~ -fitted dy ~ -const.
        assert abs(out["phi_sum"][0]) < 0.15
        cyc = out["cycle"]
        assert np.all(np.isfinite(cyc[2:]))
        # Permanent ≈ y (deviation only via mean of dy).
        dev = out["permanent"][2:] - rw[2:]
        assert np.abs(dev).max() < 0.5

    def test_bn_ar1_differences(self):
        rng = np.random.default_rng(3)
        n = 600
        dy = np.zeros(n)
        for i in range(1, n):
            dy[i] = 0.5 * dy[i - 1] + rng.normal()
        y = np.cumsum(dy)
        out = beveridge_nelson(y, ar_lags=1)
        assert 0.3 < out["phi_sum"][0] < 0.7
        # Cycle amplitude ~ A(1)^{-1} * fitted dy: non-trivial.
        assert np.nanstd(out["cycle"]) > 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hamilton_filter(np.ones(20), h=10, p=11)
        with pytest.raises(ValueError):
            beveridge_nelson(np.ones(30), ar_lags=0)
