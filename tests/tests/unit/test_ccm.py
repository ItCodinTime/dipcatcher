"""Tests for convergent cross-mapping."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ccm import ccm, simplex_project, smap_nonlinearity, takens_embed


def _coupled_logistic(n=400, coupling=0.4, seed=0):
    """x -> y coupling: y_{t+1} = y_t (3.8 - 3.8 y_t - c x_t)."""
    x = np.empty(n)
    y = np.empty(n)
    x[0] = 0.4
    y[0] = 0.2
    for t in range(n - 1):
        x[t + 1] = x[t] * (3.6 - 3.6 * x[t])
        y[t + 1] = y[t] * (3.7 - 3.7 * y[t] - coupling * x[t])
    return np.clip(x, 1e-6, 0.999), np.clip(y, 1e-6, 0.999)


class TestTakens:
    def test_embed_shape(self):
        v = np.arange(100, dtype=float) + np.random.default_rng(0).normal(size=100)
        M = takens_embed(v, dim=3, tau=2)
        assert M.shape == (100 - 4, 3)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            takens_embed(np.ones(50), dim=1)


class TestSimplex:
    def test_predicts_identity(self):
        rng = np.random.default_rng(1)
        lib = rng.normal(size=(200, 3))
        y = lib[:, 0] * 2 + rng.normal(scale=0.05, size=200)
        pred = simplex_project(lib, y, lib[:20])
        assert np.corrcoef(pred, y[:20])[0, 1] > 0.8

    def test_failclosed(self):
        with pytest.raises(ValueError):
            simplex_project(np.ones((5, 3)), np.ones(5), np.ones((4, 3)), n_neighbors=9)


class TestCCM:
    def test_coupled_converges(self):
        x, y = _coupled_logistic(n=350, coupling=0.5)
        # x drives y => embedding of y predicts x (cross-map x|Mx(y)).
        out = ccm(
            cause=x,
            effect=y,
            dim=3,
            tau=1,
            lib_sizes=np.array([30, 60, 100, 140]),
            n_reps=10,
            seed=2,
        )
        assert out["rho"].shape == (4,)
        # Convergence: skill rises with library size.
        assert out["rho"][-1] > out["rho"][0] - 0.05

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ccm(np.ones(100), np.ones(100))


class TestSmap:
    def test_logistic_nonlinear(self):
        rng = np.random.default_rng(3)
        n = 300
        x = np.empty(n)
        x[0] = 0.3
        for t in range(n - 1):
            x[t + 1] = 3.9 * x[t] * (1 - x[t])
        x = np.clip(x + rng.normal(scale=0.01, size=n), 1e-4, 0.9999)
        out = smap_nonlinearity(x, dim=3, tau=1)
        assert np.isfinite(out["skill"]).all()

    def test_failclosed(self):
        with pytest.raises(ValueError):
            smap_nonlinearity(np.ones(100), thetas=np.array([-1.0]))
