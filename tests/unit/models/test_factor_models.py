"""Tests for cross-sectional factor models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.factor_models import (
    bai_ng_factors,
    double_sorted_factors,
    factor_residuals,
    fama_macbeth,
    gics_alpha,
    pca_factors,
    sorted_factor_returns,
)


class TestFamaMacBeth:
    def test_recovers_premia(self):
        rng = np.random.default_rng(0)
        t, n = 400, 60
        char = rng.normal(size=(t, n, 1))
        # Cross-sectional: r_i = 0.002 + 0.01*char_i + noise.
        r = 0.002 + 0.01 * char[:, :, 0] + rng.normal(scale=0.02, size=(t, n))
        out = fama_macbeth(r, char)
        assert abs(out["gamma"][0] - 0.002) < 0.005
        assert abs(out["gamma"][1] - 0.01) < 0.005
        assert out["shanken_se"][1] >= out["gamma_se"][1] - 1e-12

    def test_zero_premium(self):
        rng = np.random.default_rng(1)
        t, n = 300, 50
        char = rng.normal(size=(t, n, 1))
        r = rng.normal(scale=0.02, size=(t, n))
        out = fama_macbeth(r, char, nw_lags=3)
        assert out["gamma_t"][1] ** 2 < 9.0

    def test_static_characteristics(self):
        rng = np.random.default_rng(2)
        t, n = 200, 40
        char = rng.normal(size=(n, 1))
        r = 0.005 * char[:, 0][None, :] + rng.normal(scale=0.02, size=(t, n))
        out = fama_macbeth(r, char)
        assert abs(out["gamma"][1] - 0.005) < 0.005

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fama_macbeth(np.ones((10, 4)), np.ones((10, 4, 1)) * np.nan)
        with pytest.raises(ValueError):
            fama_macbeth(np.ones((10, 4)), np.ones((5, 4, 1)))


class TestSorts:
    def test_sorted_factor_monotone(self):
        rng = np.random.default_rng(3)
        t, n = 200, 50
        char = rng.normal(size=(t, n))
        # Return increases with characteristic.
        r = 0.01 * char + rng.normal(scale=0.02, size=(t, n))
        out = sorted_factor_returns(r, char, n_groups=5)
        assert float(out["spread"].mean()) > 0.01
        g = out["group_returns"].mean(axis=0)
        assert np.all(np.diff(g) > 0)

    def test_double_sort_legs(self):
        rng = np.random.default_rng(4)
        t, n = 150, 60
        a = rng.normal(size=(t, n))
        b = rng.normal(size=(t, n))
        r = 0.01 * a - 0.01 * b + rng.normal(scale=0.02, size=(t, n))
        out = double_sorted_factors(r, a, b, cuts=3)
        assert float(out["factor_a"].mean()) > 0.005
        assert float(out["factor_b"].mean()) < -0.005

    def test_failclosed(self):
        with pytest.raises(ValueError):
            sorted_factor_returns(np.ones((10, 4)), np.ones((10, 4)), n_groups=3)
        with pytest.raises(ValueError):
            double_sorted_factors(np.ones((10, 8)), np.ones((10, 8)), np.ones((10, 8)), cuts=4)


class TestPCA:
    def test_recovers_latent_structure(self):
        rng = np.random.default_rng(5)
        t, n, r_true = 400, 50, 3
        f = rng.normal(size=(t, r_true))
        lam = rng.normal(scale=2.0, size=(n, r_true))
        r = f @ lam.T + rng.normal(scale=0.15, size=(t, n))
        sel = bai_ng_factors(r, r_max=8)
        assert min(sel["ic1"][0], sel["ic2"][0], sel["ic3"][0]) == r_true
        assert max(sel["ic1"][0], sel["ic2"][0], sel["ic3"][0]) <= 4.0
        out = pca_factors(r, n_factors=3)
        assert out["explained"][0] > 0.8
        # Recovered factors span the true space: regress each true factor.
        for j in range(r_true):
            b = np.linalg.lstsq(out["factors"], f[:, j], rcond=None)[0]
            resid = f[:, j] - out["factors"] @ b
            assert np.var(resid) / np.var(f[:, j]) < 0.1

    def test_failclosed(self):
        with pytest.raises(ValueError):
            pca_factors(np.ones((10, 4)), n_factors=20)
        with pytest.raises(ValueError):
            bai_ng_factors(np.full((10, 4), np.nan))


class TestResiduals:
    def test_factor_residuals_orthogonal(self):
        rng = np.random.default_rng(6)
        t, n = 300, 30
        f = rng.normal(size=(t, 2))
        betas = rng.normal(size=(n, 2))
        r = f @ betas.T + rng.normal(scale=0.2, size=(t, n))
        out = factor_residuals(r, f)
        assert out["resid"].shape == (t, n)
        assert out["betas"].shape == (n, 2)
        # Residuals ~ orthogonal to factors.
        fc = f - f.mean(axis=0)
        proj = float(np.abs(out["resid"].T @ fc[:, 0]).mean())
        assert proj < 5.0
        # Residual variance ~ idiosyncratic.
        assert np.all(np.var(out["resid"], axis=0) < np.var(r, axis=0) + 1e-8)

    def test_gics_alpha(self):
        rng = np.random.default_rng(7)
        t, n = 300, 20
        m = rng.normal(scale=0.01, size=t)
        alpha_true = np.full(n, 0.001)
        r = alpha_true[None, :] + 1.2 * m[:, None] + rng.normal(scale=0.004, size=(t, n))
        out = gics_alpha(r, m)
        assert np.allclose(out["beta"], 1.2, atol=0.1)
        assert np.allclose(out["alpha"], 0.001, atol=0.002)
        assert np.mean(out["alpha_t"] > 0) > 0.5
