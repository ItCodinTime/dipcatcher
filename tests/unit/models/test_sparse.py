"""Tests for models/sparse.py — LARS, OMP, adaptive lasso, EBIC."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sparse import (
    adaptive_lasso,
    ebic_select,
    lars_path,
    lasso_cd,
    omp,
)


def _sparse_design(n: int = 200, p: int = 30, k: int = 3, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p))
    beta_true = np.zeros(p)
    beta_true[[2, 7, 15]] = [2.0, -1.5, 1.0]
    y = x @ beta_true + 0.3 * rng.standard_normal(n)
    return x, y, beta_true


def _raw_units(beta_std: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Convert unit-norm-column coefficients back to raw units."""
    norms = np.linalg.norm(x - x.mean(axis=0), axis=0)
    return beta_std / norms


def test_lars_recovers_support() -> None:
    x, y, bt = _sparse_design()
    path = lars_path(x, y)
    bp = np.asarray(path["beta_path"])
    assert bp.shape[0] == 30
    # path should include a point with exactly the true support
    found = False
    for i in range(bp.shape[1]):
        nz = set(np.nonzero(np.abs(bp[:, i]) > 1e-8)[0].tolist())
        if nz == {2, 7, 15}:
            found = True
            raw = _raw_units(bp[:, i], x)
            assert np.abs(raw - bt).max() < 0.5
    assert found


def test_omp_exact_support() -> None:
    x, y, bt = _sparse_design(n=300)
    out = omp(x, y, k=3)
    sel = set(np.asarray(out["selected"]).tolist())
    assert sel == {2, 7, 15}
    raw = _raw_units(np.asarray(out["beta"]), x)
    assert np.abs(raw - bt).max() < 0.4


def test_lasso_cd_selects() -> None:
    x, y, bt = _sparse_design()
    out = lasso_cd(x, y, lam=5.0)
    b = np.asarray(out["beta"])
    nz = np.nonzero(np.abs(b) > 1e-6)[0]
    assert set(nz) == {2, 7, 15}
    assert out["n_nonzero"] == 3.0


def test_adaptive_lasso() -> None:
    x, y, _ = _sparse_design(n=300)
    out = adaptive_lasso(x, y, lam=3.0)
    b = np.asarray(out["beta"])
    nz = np.nonzero(np.abs(b) > 1e-6)[0]
    assert set(nz) == {2, 7, 15}


def test_ebic_picks_three() -> None:
    x, y, _ = _sparse_design(n=250)
    out = ebic_select(x, y)
    b = np.asarray(out["beta"])
    nz = np.nonzero(np.abs(b) > 1e-6)[0]
    assert set(nz) == {2, 7, 15}
    assert out["n_selected"] == 3.0


def test_lam_zero_is_ols() -> None:
    x, y, _ = _sparse_design(n=300)
    out = lasso_cd(x, y, lam=0.0)
    b = np.asarray(out["beta"])
    xc = (x - x.mean(0)) / np.linalg.norm(x - x.mean(0), axis=0)
    ols = np.linalg.lstsq(xc, y - y.mean(), rcond=None)[0]
    assert np.abs(b - ols).max() < 1e-4


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        lars_path(np.ones((5, 3)), np.ones(5))
    with pytest.raises(ValueError):
        omp(np.random.default_rng(0).standard_normal((50, 5)), np.ones(50), k=0)
    with pytest.raises(ValueError):
        lasso_cd(np.random.default_rng(0).standard_normal((50, 5)), np.ones(50), lam=-1.0)
    with pytest.raises(ValueError):
        adaptive_lasso(np.random.default_rng(0).standard_normal((20, 30)), np.ones(20), lam=1.0)
    with pytest.raises(ValueError):
        lars_path(np.column_stack([np.ones(50), np.ones(50)]), np.ones(50))
