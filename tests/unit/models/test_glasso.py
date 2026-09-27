"""Tests for models/glasso.py — graphical lasso + partial correlation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.glasso import (
    glasso,
    glasso_from_data,
    partial_correlation,
    select_rho_bic,
)


def _sparse_precision(p: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((p, p))
    mask = rng.random((p, p)) < 0.15
    a = a * mask
    a = a + a.T
    theta = a + p * 0.6 * np.eye(p)  # diagonally dominant -> PD
    return theta


def test_glasso_recovers_structure() -> None:
    theta = _sparse_precision()
    sigma = np.linalg.inv(theta)
    rng = np.random.default_rng(1)
    x = rng.multivariate_normal(np.zeros(theta.shape[0]), sigma, size=2000)
    out = glasso(np.cov(x.T), rho=0.02)
    th = np.asarray(out["theta"])
    zero_true = np.abs(theta) < 1e-10
    np.fill_diagonal(zero_true, False)
    zero_est = np.abs(th) < 1e-6
    # most true zeros should be recovered as zeros
    assert (zero_est & zero_true).sum() > 0.6 * zero_true.sum()


def test_partial_corr_bounds() -> None:
    theta = _sparse_precision(seed=2)
    pcorr = partial_correlation(theta)
    assert np.abs(pcorr).max() <= 1.0
    assert np.allclose(np.diag(pcorr), 1.0)


def test_high_rho_diagonal() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((500, 8))
    s = np.cov(x.T)
    out = glasso(s, rho=1.0)  # huge penalty -> diagonal Theta
    th = np.asarray(out["theta"])
    off = th - np.diag(np.diag(th))
    assert np.abs(off).max() < 1e-8


def test_bic_selection() -> None:
    rng = np.random.default_rng(4)
    theta = _sparse_precision(p=8, seed=4)
    sigma = np.linalg.inv(theta)
    x = rng.multivariate_normal(np.zeros(8), sigma, size=1500)
    out = select_rho_bic(x, np.array([0.001, 0.01, 0.05, 0.2, 0.5]))
    assert 0.001 <= out["rho"] <= 0.5
    assert np.isfinite(np.asarray(out["ebic"])).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        glasso(np.eye(3), rho=-0.1)
    with pytest.raises(ValueError):
        glasso(np.full((4, 4), np.nan), rho=0.1)
    with pytest.raises(ValueError):
        glasso_from_data(np.random.default_rng(0).standard_normal((10, 5)), 0.1)
    with pytest.raises(ValueError):
        glasso(np.eye(2), rho=0.1)  # p < 3
