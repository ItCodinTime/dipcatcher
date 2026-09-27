"""Tests for hedging estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hedging import (
    basis_statistics,
    hedge_effectiveness,
    mv_hedge_ratio,
    rolling_hedge_ratio,
)


class TestMVHedge:
    def test_perfect_hedge(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=300)
        h = s + rng.normal(scale=0.05, size=300)
        out = mv_hedge_ratio(s, h)
        assert abs(out["h_star"] - 1.0) < 0.1
        assert out["rho"] > 0.95

    def test_beta_two(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=300)
        s = 2.0 * f + rng.normal(scale=0.1, size=300)
        h = f + rng.normal(scale=0.1, size=300)
        out = mv_hedge_ratio(s, h)
        assert abs(out["h_star"] - 2.0) < 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            mv_hedge_ratio(np.ones(50), np.ones(50))


class TestEffectiveness:
    def test_full_hedge_eliminates_var(self):
        rng = np.random.default_rng(2)
        s = rng.normal(size=200)
        out = hedge_effectiveness(s, s, h=1.0)
        assert out["effectiveness"] > 0.99

    def test_partial(self):
        rng = np.random.default_rng(3)
        f = rng.normal(size=300)
        s = f + rng.normal(scale=0.5, size=300)
        h = f + rng.normal(scale=0.3, size=300)
        out = hedge_effectiveness(s, h, h=1.0)
        assert 0 < out["effectiveness"] < 1.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            hedge_effectiveness(np.ones(50), np.ones(50), 1.0)


class TestRollingRatio:
    def test_converges_to_true(self):
        rng = np.random.default_rng(4)
        f = rng.normal(size=400)
        s = 1.5 * f + rng.normal(scale=0.1, size=400)
        h = f + rng.normal(scale=0.1, size=400)
        out = rolling_hedge_ratio(s, h, window=100)
        assert abs(out["ratio"][-1] - 1.5) < 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            rolling_hedge_ratio(np.ones(30), np.ones(30), window=100)


class TestBasis:
    def test_cointegrated_levels(self):
        rng = np.random.default_rng(5)
        s = np.cumsum(rng.normal(size=300))
        h = s + rng.normal(scale=0.5, size=300)
        out = basis_statistics(s, h)
        assert out["level_corr"] > 0.95
        assert out["half_life"] < 50  # basis mean-reverts quickly

    def test_failclosed(self):
        with pytest.raises(ValueError):
            basis_statistics(np.ones(50), np.ones(50))
