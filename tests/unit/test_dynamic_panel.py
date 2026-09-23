"""Tests for dynamic panel estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dynamic_panel import anderson_hsiao, arellano_bond


def _dyn_panel(n=40, t=30, rho=0.5, seed=0):
    """AR(1) panel with fixed effects: y_it = a_i + rho y_i,t-1 + e_it."""
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=1.0, size=n)
    p = np.empty((t + 50, n))
    p[0] = a + rng.normal(size=n)
    for s in range(1, t + 50):
        p[s] = a + rho * p[s - 1] + rng.normal(size=n)
    return p[50:]  # burn-in


class TestAndersonHsiao:
    def test_recovers_rho(self):
        p = _dyn_panel(n=60, t=40, rho=0.5, seed=1)
        out = anderson_hsiao(p)
        # AH with y_{t-2} instrument is consistent but noisy — generous tol.
        assert abs(out["rho"] - 0.5) < 0.35
        assert out["se"][0] > 0

    def test_with_exogenous(self):
        rng = np.random.default_rng(2)
        n, t = 50, 30
        a = rng.normal(size=n)
        X = rng.normal(size=(t + 50, n, 1))
        p = np.empty((t + 50, n))
        p[0] = a
        for s in range(1, t + 50):
            p[s] = a + 0.4 * p[s - 1] + 0.8 * X[s, :, 0] + rng.normal(size=n)
        out = anderson_hsiao(p[50:], X[50:])
        assert out["beta"].shape == (2,)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            anderson_hsiao(np.ones((3, 10)))


class TestArellanoBond:
    def test_recovers_rho_direction(self):
        p = _dyn_panel(n=80, t=25, rho=0.5, seed=3)
        out = arellano_bond(p)
        assert 0.1 < out["rho"] < 0.9

    def test_sargan_reported(self):
        p = _dyn_panel(n=60, t=30, seed=4)
        out = arellano_bond(p, max_lag_inst=2)
        assert 0.0 <= out["sargan_p"] <= 1.0
        assert out["n_inst"] > 1

    def test_failclosed(self):
        with pytest.raises(ValueError):
            arellano_bond(np.ones((3, 3)))
        with pytest.raises(ValueError):
            arellano_bond(np.full((20, 10), np.nan))
