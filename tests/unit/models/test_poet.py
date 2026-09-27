"""Tests for models/poet.py — Fan-Liao-Mincheva POET covariance."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.poet import poet_cov, poet_select_k


def _factor_data(
    t: int = 400, p: int = 60, k: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((t, k))
    lam = rng.standard_normal((p, k)) * 0.8  # mean-zero: factors distinguishable
    u = rng.standard_normal((t, p)) * 0.3
    r = f @ lam.T + u
    sigma_true = lam @ lam.T + 0.09 * np.eye(p)
    return r, sigma_true


def test_poet_recovers_covariance() -> None:
    r, st = _factor_data()
    out = poet_cov(r, k=2)
    sig = np.asarray(out["sigma"])
    err = np.linalg.norm(sig - st) / np.linalg.norm(st)
    assert err < 0.25
    # PSD
    assert np.linalg.eigvalsh(sig).min() > -1e-8


def test_select_k_finds_two() -> None:
    r, _ = _factor_data(seed=1)
    k = poet_select_k(r, kmax=8)
    assert k == 2


def test_sparsity_positive() -> None:
    r, _ = _factor_data(seed=2)
    out = poet_cov(r, k=2)
    assert 0.0 < out["sparsity"] <= 1.0


def test_residual_shrinkage() -> None:
    # Deterministic guarantee: universal thresholding strictly shrinks
    # off-diagonal residual correlations vs the unthresholded residuals.
    rng = np.random.default_rng(5)
    t, p = 150, 40
    f = rng.standard_normal((t, 1))
    lam = rng.standard_normal((p, 1))
    r = f @ lam.T + 0.4 * rng.standard_normal((t, p))
    out = poet_cov(r, k=1)
    s_u = np.asarray(out["sigma_u"])
    d = np.sqrt(np.diag(s_u))
    corr_thr = np.abs(s_u / np.outer(d, d))
    np.fill_diagonal(corr_thr, 0.0)
    # raw residual correlations for comparison
    rc = r - r.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(r.T))
    v1 = vecs[:, -1:]
    u_raw = rc - rc @ v1 @ v1.T
    su_raw = np.corrcoef(u_raw.T)
    np.fill_diagonal(su_raw, 0.0)
    assert corr_thr.max() <= np.abs(su_raw).max() + 1e-12
    assert (corr_thr == 0.0).mean() > 0.0  # some entries zeroed


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        poet_cov(np.random.default_rng(0).standard_normal((20, 60)))  # T < 30
    with pytest.raises(ValueError):
        poet_cov(np.random.default_rng(0).standard_normal((100, 3)))  # p < 5
    r, _ = _factor_data()
    r[0, 0] = np.nan
    with pytest.raises(ValueError):
        poet_cov(r)
