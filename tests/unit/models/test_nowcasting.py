"""Tests for mixed-frequency nowcasting."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.nowcasting import (
    almon_weights,
    beta_weights,
    bridge_regression,
    fit_almon,
    fit_midas,
)


class TestBetaWeights:
    def test_normalized(self):
        w = beta_weights(12, 1.0, 5.0)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= 0)
        # theta2 > theta1 concentrates weight on early lags.
        assert w[0] > w[-1]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            beta_weights(5, -1.0, 2.0)


class TestMIDAS:
    def test_recovers_slope(self):
        rng = np.random.default_rng(0)
        n, k = 200, 6
        X = rng.normal(size=(n, k))
        w_true = beta_weights(k, 1.0, 8.0)  # weight on recent lags
        y = 0.5 + 2.0 * (X @ w_true) + rng.normal(scale=0.1, size=n)
        fit = fit_midas(y, X, k_lags=k, ar_lag=False)
        assert abs(fit["slope"] - 2.0) < 0.4
        # Weight shape should decay toward zero-lag ordering approx.
        assert fit["weights"][0] > fit["weights"][-1] * 0.5

    def test_ar_term(self):
        rng = np.random.default_rng(1)
        n, k = 250, 4
        X = rng.normal(size=(n, k))
        y = np.empty(n)
        y[0] = 0.0
        for t in range(1, n):
            y[t] = 0.4 * y[t - 1] + 1.0 * X[t, 0] + rng.normal(scale=0.2)
        fit = fit_midas(y, X, k_lags=k, ar_lag=True)
        assert abs(fit["ar_coef"] - 0.4) < 0.25

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_midas(np.ones(30), np.ones((20, 3)), k_lags=3)


class TestAlmon:
    def test_weights_shape(self):
        w = almon_weights(8, 2, np.array([0.5, 0.1, -0.02]))
        assert w.shape == (8,)
        assert w[0] == pytest.approx(0.5)

    def test_fit_recovers_lag_shape(self):
        rng = np.random.default_rng(2)
        n, k = 300, 5
        X = rng.normal(size=(n, k))
        # True weights follow a quadratic in lag.
        w_true = 1.0 + 0.3 * np.arange(k) - 0.1 * np.arange(k) ** 2
        y = X @ w_true + rng.normal(scale=0.2, size=n)
        fit = fit_almon(y, X, degree=2)
        assert fit["r2"] > 0.7
        assert np.corrcoef(fit["weights"], w_true)[0, 1] > 0.9

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_almon(np.ones(30), np.ones((30, 3)), degree=5)


class TestBridge:
    def test_recovers_coefs(self):
        rng = np.random.default_rng(3)
        n = 150
        M = rng.normal(size=(n, 3))
        G = rng.normal(size=(n, 1))
        y = 0.2 + M @ np.array([1.0, -0.5, 0.3]) + 0.7 * G[:, 0] + rng.normal(scale=0.1, size=n)
        fit = bridge_regression(y, M, G)
        assert fit["r2"] > 0.95
        assert fit["coefs"].shape == (5,)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            bridge_regression(np.ones(30), np.ones((20, 2)), np.ones((20, 1)))
