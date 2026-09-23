"""Tests for VAR, cointegration, and connectedness."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.var_coint import (
    diebold_yilmaz,
    engle_granger,
    fevd,
    granger_causality_matrix,
    johansen_test,
    spread_half_life,
    var_fit,
    var_irf,
    var_select_order,
    vecm_fit,
)


def _var1(n: int = 400, rho: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal(size=2)
    return y


def _coint_pair(n: int = 500, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """x random walk; y = x + stationary spread."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    spr = np.zeros(n)
    for t in range(1, n):
        spr[t] = 0.85 * spr[t - 1] + rng.normal(scale=0.3)
    return x, x + spr


class TestVAR:
    def test_recovers_coefficients(self):
        y = _var1(rho=0.7)
        fit = var_fit(y, p=1)
        assert np.allclose(fit["A"][0], np.eye(2) * 0.7, atol=0.1)
        assert np.all(np.diag(fit["sigma"]) > 0)
        assert fit["resid"].shape == (399, 2)

    def test_order_selection(self):
        y = _var1()
        sel = var_select_order(y, p_max=6)
        assert sel["bic"][0] == 1.0
        assert 1 <= sel["aic"][0] <= 3

    def test_irf_decay(self):
        y = _var1(rho=0.7)
        fit = var_fit(y, p=1)
        irf = var_irf(fit, horizon=10)
        # Diagonal GIRF decays like rho^h.
        g = irf["girf"]
        assert abs(g[5, 0, 0]) < abs(g[1, 0, 0])
        assert g[0, 0, 0] == pytest.approx(np.sqrt(fit["sigma"][0, 0]), rel=0.05)

    def test_fevd_sums_to_one(self):
        y = _var1()
        fit = var_fit(y, p=1)
        f = fevd(fit, horizon=10)["gfevd"]
        assert np.allclose(f.sum(axis=1), 1.0, atol=1e-8)
        # Independent VAR1 -> own-shock share dominant.
        assert f[0, 0] > 0.8

    def test_failclosed(self):
        with pytest.raises(ValueError):
            var_fit(np.ones((10, 2)), p=1)
        with pytest.raises(ValueError):
            var_irf({"A": np.ones((1, 2, 2)), "sigma": np.eye(2)}, horizon=0)


class TestConnectedness:
    def test_spillover_direction(self):
        rng = np.random.default_rng(2)
        n = 500
        leader = np.zeros(n)
        y = np.zeros((n, 3))
        for t in range(1, n):
            leader[t] = 0.6 * leader[t - 1] + rng.normal()
            y[t, 0] = leader[t] + 0.3 * rng.normal()
            y[t, 1] = 0.8 * leader[t - 1] + 0.3 * rng.normal()  # follows leader
            y[t, 2] = rng.normal()
        out = diebold_yilmaz(y, p=2, horizon=10)
        assert 0.0 <= out["tsi"][0] <= 1.0
        # Leader (var 0) should be net transmitter; follower (var 1) receiver.
        assert out["net"][0] > out["net"][1]
        assert out["net"][1] < out["net"][2] + 0.5

    def test_granger_matrix(self):
        rng = np.random.default_rng(3)
        n = 400
        x = np.zeros(n)
        y = np.zeros((n, 2))
        for t in range(1, n):
            x[t] = 0.6 * x[t - 1] + rng.normal()
            y[t, 0] = x[t]
            y[t, 1] = 0.8 * x[t - 1] + rng.normal()  # y1 caused by x-history
        out = granger_causality_matrix(y, p=2)
        assert out["pvalues"][1, 0] < 0.01  # x causes y1
        assert out["pvalues"][0, 1] > 0.1  # not reverse


class TestCointegration:
    def test_engle_granger_detects(self):
        x, y = _coint_pair()
        out = engle_granger(y, x)
        assert out["tau"] < out["cv_5pct"]
        assert abs(out["beta"] - 1.0) < 0.2
        # Independent random walks: not cointegrated.
        rng = np.random.default_rng(4)
        x2 = np.cumsum(rng.normal(size=500))
        y2 = np.cumsum(rng.normal(size=500))
        out2 = engle_granger(y2, x2)
        assert out2["tau"] > out2["cv_5pct"]

    def test_johansen_rank(self):
        x, y = _coint_pair()
        m = np.column_stack([x, y])
        out = johansen_test(m, p=2, det=1)
        # Should find rank 1 (one cointegrating relation): first test
        # rejects r=0, second does not reject r<=1.
        assert out["trace"][0] > out["cv_trace5"][0]
        assert out["trace"][1] < out["cv_trace5"][1]
        # Independent walks: rank 0.
        rng = np.random.default_rng(5)
        m2 = np.column_stack([np.cumsum(rng.normal(size=500)), np.cumsum(rng.normal(size=500))])
        out2 = johansen_test(m2, p=2, det=1)
        assert out2["trace"][0] < out2["cv_trace5"][0] + 10.0

    def test_vecm_recovers_beta(self):
        x, y = _coint_pair()
        m = np.column_stack([x, y])
        out = vecm_fit(m, p=1, rank=1)
        beta = out["beta"][:, 0]
        # Cointegrating vector ~ (1, -1) up to scale.
        ratio = beta[1] / beta[0] if abs(beta[0]) > 1e-8 else np.inf
        assert abs(ratio + 1.0) < 0.3
        # Alpha adjusts negatively on the spread in y.
        assert out["alpha"].shape == (2, 1)

    def test_spread_half_life(self):
        rng = np.random.default_rng(6)
        n = 2000
        spr = np.zeros(n)
        for t in range(1, n):
            spr[t] = 0.95 * spr[t - 1] + rng.normal()
        hl = spread_half_life(spr)
        assert 10 < hl < 18  # -ln2/ln0.95 ~ 13.5
        # Random walk: rho_hat < 1 in finite samples -> long but finite hl.
        hl_rw = spread_half_life(np.cumsum(rng.normal(size=500)))
        assert np.isnan(hl_rw) or hl_rw > 30

    def test_failclosed(self):
        with pytest.raises(ValueError):
            engle_granger(np.ones(10), np.ones(10))
        with pytest.raises(ValueError):
            johansen_test(np.ones((20, 2)), p=1)
        with pytest.raises(ValueError):
            vecm_fit(np.random.default_rng(0).normal(size=(100, 2)), rank=2)
