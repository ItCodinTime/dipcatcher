"""Tests for models/gmm_est.py — Hansen two-step GMM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gmm_est import gmm_2step


def _ols_moments(y: np.ndarray, x: np.ndarray):
    """OLS as GMM: E[x * (y - x'b)] = 0."""

    def mom(theta: np.ndarray) -> np.ndarray:
        resid = y - x @ theta
        return x * resid[:, None]

    return mom


def test_gmm_ols_equivalent() -> None:
    rng = np.random.default_rng(0)
    n = 600
    x = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([1.0, -2.0, 0.5])
    y = x @ beta + rng.standard_normal(n) * 0.5
    out = gmm_2step(_ols_moments(y, x), np.zeros(3))
    assert np.abs(np.asarray(out["theta"]) - beta).max() < 0.3
    assert out["J"] < 30.0  # exactly identified -> J ~ 0


def test_gmm_se_reasonable() -> None:
    rng = np.random.default_rng(1)
    n = 800
    x = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = x @ np.array([0.5, 1.0]) + rng.standard_normal(n)
    out = gmm_2step(_ols_moments(y, x), np.zeros(2))
    se = np.asarray(out["se"])
    assert (se > 0).all() and se.max() < 0.5
    t = np.asarray(out["tstat"])
    assert abs(t[1]) > 3.0  # slope strongly identified


def test_overidentified_j_test() -> None:
    # IV example: one endogenous regressor, two instruments -> 1 overid rest.
    rng = np.random.default_rng(2)
    n = 1000
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    u = rng.standard_normal(n)
    x = 0.5 * z1 + 0.4 * z2 + 0.3 * u + rng.standard_normal(n)
    beta = 1.0
    y = beta * x + u

    def mom(theta: np.ndarray) -> np.ndarray:
        r = y - theta[0] * x
        return np.column_stack([r * z1, r * z2])

    out = gmm_2step(mom, np.zeros(1))
    th = float(np.asarray(out["theta"])[0])
    assert 0.7 < th < 1.5  # IV recovers toward truth vs OLS ~1.23
    assert np.isfinite(out["J_p"])
    assert out["dof"] == 1.0


def test_autocorrelated_moments_hac() -> None:
    # moment process with serial correlation -> NW weighting still converges
    rng = np.random.default_rng(3)
    n = 500
    mu_true = 2.0
    e = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        e[t] = 0.5 * e[t - 1] + eps[t]
    w = rng.choice([-1.0, 1.0], size=n)  # exogenous instrument, E[r*w]=0
    x = mu_true + e

    def mom(theta: np.ndarray) -> np.ndarray:
        r = x - theta[0]
        return np.column_stack([r, r * w])

    out = gmm_2step(mom, np.zeros(1), max_lag=5)
    assert abs(float(np.asarray(out["theta"])[0]) - mu_true) < 0.4


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        # underidentified: 1 moment, 2 params
        gmm_2step(lambda th: np.random.default_rng(0).standard_normal((100, 1)), np.zeros(2))

    def bad(th: np.ndarray) -> np.ndarray:
        return np.full((100, 3), np.nan)

    with pytest.raises(ValueError):
        gmm_2step(bad, np.zeros(2))
