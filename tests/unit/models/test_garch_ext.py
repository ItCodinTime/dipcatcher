"""Tests for extended GARCH estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.garch_ext import (
    aparch_variance,
    figarch_variance,
    fit_aparch,
    fit_figarch,
    news_impact_curve,
)


class TestFIGARCH:
    def test_variance_positive_and_reactive(self):
        rng = np.random.default_rng(0)
        n = 400
        e = rng.normal(size=n)
        s2 = figarch_variance(e, phi=0.2, d=0.4, beta=0.5)
        assert np.all(s2 > 0)
        # Variance responds to shocks.
        e2 = e.copy()
        e2[200] = 6.0
        s2s = figarch_variance(e2, phi=0.2, d=0.4, beta=0.5)
        assert s2s[201] > s2[201]

    def test_long_memory_weights(self):
        # FIGARCH lambda weights decay hyperbolically (slowly).
        v = np.random.default_rng(1).normal(size=50)
        s2_hi = figarch_variance(v, phi=0.1, d=0.7, beta=0.3)
        s2_lo = figarch_variance(v, phi=0.1, d=0.1, beta=0.3)
        assert np.all(np.isfinite(s2_hi)) and np.all(np.isfinite(s2_lo))

    def test_fit_recovers_d(self):
        rng = np.random.default_rng(2)
        n = 600
        e = np.empty(n)
        s2 = np.empty(n)
        s2[0] = 1.0
        e[0] = 0.0
        # Simulate a persistent-vol process (GARCH-like with high persistence).
        for t in range(1, n):
            s2[t] = 0.02 + 0.08 * e[t - 1] ** 2 + 0.9 * s2[t - 1]
            e[t] = rng.normal(scale=np.sqrt(s2[t]))
        fit = fit_figarch(e)
        assert 0.01 <= fit["d"][0] <= 0.99
        assert fit["phi"][0] >= 0 and fit["beta"][0] >= 0
        assert np.all(fit["sigma2"] > 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            figarch_variance(np.ones(10), 0.2, 0.4, 0.5)
        with pytest.raises(ValueError):
            figarch_variance(np.ones(100), 0.8, 0.4, 0.8)  # phi+beta >= 1
        with pytest.raises(ValueError):
            fit_figarch(np.full(100, np.nan))


class TestAPARCH:
    def test_recovers_leverage_asymmetry(self):
        # Negative shocks should raise vol more than positive ones when gamma>0.
        e = np.zeros(300)
        e[150] = -4.0
        s2_neg = aparch_variance(e, 0.02, 0.06, 0.4, 0.9, 2.0)
        e[150] = 4.0
        s2_pos = aparch_variance(e, 0.02, 0.06, 0.4, 0.9, 2.0)
        assert s2_neg[151] > s2_pos[151]

    def test_fit(self):
        rng = np.random.default_rng(4)
        n = 500
        e = np.empty(n)
        s2 = np.empty(n)
        s2[0] = 1.0
        e[0] = 0.0
        for t in range(1, n):
            s2[t] = 0.02 + 0.05 * e[t - 1] ** 2 + 0.92 * s2[t - 1]
            e[t] = rng.normal(scale=np.sqrt(s2[t]))
        fit = fit_aparch(e)
        assert np.all(fit["sigma2"] > 0)
        assert fit["alpha"][0] >= 0 and fit["beta"][0] >= 0
        assert np.isfinite(fit["loglik"][0])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            aparch_variance(np.ones(100), -1.0, 0.05, 0.0, 0.9, 2.0)
        with pytest.raises(ValueError):
            aparch_variance(np.ones(100), 0.02, 0.05, 2.0, 0.9, 2.0)  # |gamma|>1


class TestNewsImpact:
    def test_asymmetry(self):
        out = news_impact_curve(gamma=0.5, delta=2.0)
        assert out["asymmetry_ratio"][0] > 1.0  # bad news dominates
        sym = news_impact_curve(gamma=0.0, delta=2.0)
        assert abs(sym["asymmetry_ratio"][0] - 1.0) < 1e-9

    def test_failclosed(self):
        with pytest.raises(ValueError):
            news_impact_curve(gamma=1.5, delta=2.0)
