"""Tests for panel cointegration."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.panel_coint import kao_test, pedroni_panel_adf


def _coint_panels(n=10, t=150, rho_e=0.4, seed=0):
    """y and x share a common stochastic trend: x_i RW, y_i = x_i + AR(rho_e)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(t, n)), axis=0)
    e = np.empty((t, n))
    e[0] = rng.normal(size=n)
    for s in range(1, t):
        e[s] = rho_e * e[s - 1] + rng.normal(size=n)
    y = x + e
    return y, x


def _indep_panels(n=10, t=150, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=(t, n)), axis=0)
    y = np.cumsum(rng.normal(size=(t, n)), axis=0)  # independent RW
    return y, x


class TestKao:
    def test_cointegrated_rejects_more(self):
        y, x = _coint_panels()
        out = kao_test(y, x)
        yi, xi = _indep_panels()
        out_i = kao_test(yi, xi)
        # Cointegrated panel: more negative residual-ADF t.
        assert out["t_stat"] < out_i["t_stat"]
        assert out["rho"] < out_i["rho"]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kao_test(np.ones((20, 4)), np.ones((20, 4)))


class TestPedroni:
    def test_cointegrated_vs_independent(self):
        y, x = _coint_panels(rho_e=0.3, t=200)
        stat_c = pedroni_panel_adf(y, x)["group_adf"]
        yi, xi = _indep_panels(t=200)
        stat_i = pedroni_panel_adf(yi, xi)["group_adf"]
        assert stat_c < stat_i  # cointegrated residuals more stationary

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pedroni_panel_adf(np.ones((30, 2)), np.ones((30, 2)))
