"""Tests for models/elliptical.py — multivariate Student-t EM fit."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.elliptical import mv_t_fit, mv_t_logpdf


def _mv_t_sample(mu: np.ndarray, scatter: np.ndarray, nu: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = mu.size
    chol = np.linalg.cholesky(scatter)
    z = rng.standard_normal((n, d)) @ chol.T
    w = rng.chisquare(nu, size=n) / nu
    return mu + z / np.sqrt(w)[:, None]


def test_fit_recovers_location_and_scatter() -> None:
    mu = np.array([1.0, -2.0])
    scatter = np.array([[2.0, 0.6], [0.6, 1.0]])
    x = _mv_t_sample(mu, scatter, nu=6.0, n=8000, seed=0)
    out = mv_t_fit(x)
    est_mu = np.asarray(out["mu"])
    est_sig = np.asarray(out["sigma"])
    assert np.allclose(est_mu, mu, atol=0.15)
    rel = np.linalg.norm(est_sig - scatter) / np.linalg.norm(scatter)
    assert rel < 0.25
    assert 3.0 < float(out["nu"]) < 20.0


def test_fixed_nu_runs_and_logpdf_finite() -> None:
    mu = np.zeros(2)
    scatter = np.eye(2)
    x = _mv_t_sample(mu, scatter, nu=8.0, n=2000, seed=1)
    out = mv_t_fit(x, nu=8.0)
    lp = mv_t_logpdf(x[:5], np.asarray(out["mu"]), np.asarray(out["sigma"]), 8.0)
    assert np.isfinite(lp).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        mv_t_fit(np.ones((3, 2)))  # too few rows
    with pytest.raises(ValueError):
        mv_t_fit(np.random.default_rng(0).standard_normal((100, 2)), nu=1.5)
