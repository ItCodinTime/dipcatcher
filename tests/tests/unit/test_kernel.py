"""Tests for kernel methods."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.kernel import (
    fit_gp_regression,
    fit_kernel_ridge,
    kernel_pca,
    predict_gp,
    predict_kernel_ridge,
)


class TestGP:
    def test_recovers_smooth_function(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 4, 60)[:, None]
        y = np.sin(x[:, 0] * 2.0) + rng.normal(scale=0.05, size=60)
        fit = fit_gp_regression(x, y, optimize=True)
        xt = np.linspace(0, 4, 40)[:, None]
        pred = predict_gp(fit, xt)
        true = np.sin(xt[:, 0] * 2.0)
        assert np.corrcoef(pred["mean"], true)[0, 1] > 0.9
        assert np.all(pred["std"] > 0)

    def test_uncertainty_grows_away_from_data(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 1, 40)[:, None]
        y = x[:, 0] * 2 + rng.normal(scale=0.1, size=40)
        fit = fit_gp_regression(x, y, optimize=False, noise=0.05)
        pred_in = predict_gp(fit, np.array([[0.5]]))
        pred_out = predict_gp(fit, np.array([[5.0]]))
        assert pred_out["std"][0] > pred_in["std"][0]

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_gp_regression(np.ones(5), np.ones(5))
        with pytest.raises(ValueError):
            fit_gp_regression(np.ones((30, 2)), np.ones(30))  # constant X


class TestKernelRidge:
    def test_fits_nonlinear(self):
        rng = np.random.default_rng(2)
        x = np.linspace(-3, 3, 100)[:, None]
        y = x[:, 0] ** 2 - 2 + rng.normal(scale=0.1, size=100)
        fit = fit_kernel_ridge(x, y, lam=1e-4)
        xt = np.linspace(-3, 3, 30)[:, None]
        pred = predict_kernel_ridge(fit, xt)
        assert np.corrcoef(pred, xt[:, 0] ** 2)[0, 1] > 0.95

    def test_failclosed(self):
        with pytest.raises(ValueError):
            fit_kernel_ridge(np.ones((50, 1)), np.ones(50), lam=0.0)


class TestKernelPCA:
    def test_recovers_structure(self):
        rng = np.random.default_rng(3)
        t = rng.uniform(0, 2 * np.pi, 150)
        x = np.column_stack([np.cos(t), np.sin(t)])  # ring — 1 nonlinear dim
        out = kernel_pca(x, n_components=2)
        assert out["components"].shape == (150, 2)
        assert np.all(out["eigenvalues"] > 0)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kernel_pca(np.ones((20, 2)), n_components=25)
