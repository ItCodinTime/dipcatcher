"""Tests for portfolio/resampled.py — Michaud resampled efficiency."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.portfolio.resampled import (
    max_sharpe_weights,
    min_variance_weights,
    resampled_weights,
)


def test_weights_on_simplex() -> None:
    rng = np.random.default_rng(0)
    mu = np.array([0.05, 0.07, 0.06, 0.04])
    a = rng.standard_normal((4, 4))
    cov = a @ a.T / 4.0 + np.eye(4) * 0.02
    out = resampled_weights(mu, cov, n_obs=80, n_sims=40, rng=rng)
    w = out["weights"]
    assert abs(float(w.sum()) - 1.0) < 1e-8
    assert (w >= -1e-9).all()


def test_symmetric_assets_give_equal_weights() -> None:
    rng = np.random.default_rng(1)
    n = 4
    mu = np.full(n, 0.05)
    cov = np.full((n, n), 0.01) + np.eye(n) * 0.04  # exchangeable
    out = resampled_weights(mu, cov, n_obs=200, n_sims=80, rng=rng)
    assert np.max(np.abs(out["weights"] - 1.0 / n)) < 0.08


def test_resampling_reduces_concentration() -> None:
    rng = np.random.default_rng(2)
    mu = np.array([0.03, 0.05, 0.09, 0.06])
    a = rng.standard_normal((4, 4))
    cov = a @ a.T / 4.0 + np.eye(4) * 0.03
    single = max_sharpe_weights(mu, cov)
    resampled = resampled_weights(mu, cov, n_obs=60, n_sims=80, rng=rng)["weights"]
    assert float(resampled.max()) <= float(single.max()) + 1e-6


def test_min_variance_runs() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 3))
    cov = a @ a.T / 3.0 + np.eye(3) * 0.02
    w = min_variance_weights(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-8


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        resampled_weights(np.array([0.1]), np.array([[0.1]]))  # one asset
    with pytest.raises(ValueError):
        resampled_weights(np.array([0.1, 0.2]), np.eye(2), objective="nope")


def test_non_psd_covariance_is_refused() -> None:
    # numpy would sample this anyway (warn-and-project); the allocator layer
    # rejects the same matrix, so resampled efficiency must too.
    rng = np.random.default_rng(4)
    mu = np.array([0.05, 0.05])
    not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigvals -1, +3
    with pytest.raises(ValueError, match="semidefinite"):
        resampled_weights(mu, not_psd, n_obs=80, n_sims=5, rng=rng)
    with pytest.raises(ValueError, match="semidefinite"):
        min_variance_weights(not_psd)
    with pytest.raises(ValueError, match="semidefinite"):
        max_sharpe_weights(mu, not_psd)


def test_asymmetric_covariance_is_refused() -> None:
    mu = np.array([0.05, 0.05])
    skew = np.array([[1.0, 0.9], [0.1, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        resampled_weights(mu, skew, n_obs=80, n_sims=5, rng=np.random.default_rng(5))


def test_numerically_psd_covariance_passes() -> None:
    # Eigenvalue a hair below zero at machine noise must not be refused.
    rng = np.random.default_rng(6)
    a = rng.standard_normal((3, 3))
    cov = a @ a.T / 3.0
    cov[0, 0] -= 1e-13  # tiny negative perturbation inside the 1e-12 scale tol
    cov = (cov + cov.T) / 2.0
    w = min_variance_weights(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-8
