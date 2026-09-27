"""Tests for models/functional.py — FPCA + function-on-scalar."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.functional import fos_regress, fpca, fpca_predict


def _curves(n: int = 40, g: int = 60, seed: int = 0) -> np.ndarray:
    """Two-mode curves: sin + cos mixture with random scores."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, g)
    a1 = rng.standard_normal(n) * 2.0
    a2 = rng.standard_normal(n) * 1.0
    y = a1[:, None] * np.sin(t)[None, :] + a2[:, None] * np.cos(2 * t)[None, :]
    return y + 0.05 * rng.standard_normal((n, g))


def test_fpca_fve_and_reconstruction() -> None:
    y = _curves()
    out = fpca(y, n_comp=2)
    fve = np.asarray(out["fve"])
    assert fve.sum() > 0.9  # two modes dominate
    recon = fpca_predict(out)
    err = np.abs(recon - y).mean()
    assert err < 0.2


def test_fpca_eigenfunctions_orthonormal() -> None:
    y = _curves(seed=1)
    out = fpca(y, n_comp=3)
    phi = np.asarray(out["eigfuncs"])
    w = np.asarray(out["weights"])
    gram = (phi * w[:, None]).T @ phi
    assert np.allclose(gram, np.eye(3), atol=1e-8)


def test_fpca_predict_dimension_check() -> None:
    y = _curves()
    out = fpca(y, n_comp=2)
    with pytest.raises(ValueError):
        fpca_predict(out, np.zeros((5, 3)))


def test_fos_regress_recovers_beta() -> None:
    rng = np.random.default_rng(2)
    n, g = 60, 50
    t = np.linspace(0, 1, g)
    x = rng.standard_normal(n)
    beta_true = 1.5 * np.sin(np.pi * t)  # scalar -> function coefficient
    y = np.outer(x, beta_true) + 0.05 * rng.standard_normal((n, g))
    out = fos_regress(y, x)
    beta = np.asarray(out["beta"])
    assert beta.shape == (2, g)
    assert np.abs(beta[1] - beta_true).max() < 0.2
    assert np.asarray(out["r2"]).mean() > 0.9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        fpca(np.random.default_rng(0).standard_normal((4, 60)))  # n < 5
    with pytest.raises(ValueError):
        fpca(np.random.default_rng(0).standard_normal((10, 4)))  # G < 5
    y = _curves()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        fpca(y)
    with pytest.raises(ValueError):
        fos_regress(_curves(n=10, g=20), np.ones((10, 2)))  # collinear x
