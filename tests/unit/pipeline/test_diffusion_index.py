"""Tests for diffusion-index forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.diffusion_index import (
    diffusion_index_forecast,
    factor_augmented_regression,
    sw_factors,
)


def _factor_panel(t=200, n=30, k=2, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(t, k))
    lam = rng.normal(size=(n, k))
    E = rng.normal(scale=0.5, size=(t, n))
    return F @ lam.T + E, F


class TestSWFactors:
    def test_recovers_factor_space(self):
        X, F_true = _factor_panel()
        out = sw_factors(X, k=2)
        assert out["factors"].shape == (200, 2)
        assert out["loadings"].shape == (30, 2)
        # Extracted factors correlate with true factors (up to rotation).
        for j in range(2):
            c = max(abs(np.corrcoef(out["factors"][:, i], F_true[:, j])[0, 1]) for i in range(2))
            assert c > 0.6
        assert out["r2"][0] > 0.5

    def test_failclosed(self):
        with pytest.raises(ValueError):
            sw_factors(np.ones((50, 10)), k=2)


class TestDiffusionIndex:
    def test_forecasts_factor_driven_y(self):
        X, F_true = _factor_panel(t=250, n=30, seed=1)
        rng = np.random.default_rng(9)
        # y_t driven by factor 1 with one-period lag.
        y = np.empty(250)
        y[0] = 0.0
        for t in range(1, 250):
            y[t] = 0.8 * F_true[t - 1, 0] + rng.normal(scale=0.3)
        out = diffusion_index_forecast(X, y, k=2, ar_lags=1, horizon=1)
        assert out["r2"] > 0.3
        assert np.isfinite(out["forecast"])
        assert out["coefs"].shape[0] == 1 + 2 + 1

    def test_failclosed(self):
        with pytest.raises(ValueError):
            diffusion_index_forecast(np.ones((50, 10)), np.ones(50), k=2)


class TestFAR:
    def test_factor_driven_observable(self):
        X, F_true = _factor_panel(t=200, n=25, seed=2)
        rng = np.random.default_rng(8)
        y = 1.5 * F_true[:, 0] + rng.normal(scale=0.2, size=200)
        out = factor_augmented_regression(X, y, k=2)
        assert out["r2"] > 0.7
        # Dropping the factor that carries y should hurt most.
        assert out["factor_sse_increase"].max() > 0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            factor_augmented_regression(np.ones((50, 10)), np.ones(40), k=2)
