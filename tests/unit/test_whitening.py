"""Tests for models/whitening.py — PCA and ZCA whitening."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.whitening import pca_whiten, zca_whiten


def _correlated(n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([[2.0, 1.2, 0.4], [0.0, 1.5, 0.6], [0.0, 0.0, 1.0]])
    return rng.standard_normal((n, 3)) @ a.T + np.array([1.0, -2.0, 3.0])


def test_pca_whitening_identity_cov() -> None:
    x = _correlated()
    y = pca_whiten(x)["whitened"]
    cov = np.cov(y, rowvar=False)
    assert np.allclose(cov, np.eye(3), atol=0.05)


def test_zca_whitening_identity_cov_and_symmetric() -> None:
    x = _correlated(seed=1)
    out = zca_whiten(x)
    cov = np.cov(out["whitened"], rowvar=False)
    assert np.allclose(cov, np.eye(3), atol=0.05)
    w = out["matrix"]
    assert np.allclose(w, w.T, atol=1e-8)  # ZCA matrix is symmetric


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        pca_whiten(np.ones((3, 5)))  # n < d + 2
    with pytest.raises(ValueError):
        zca_whiten(np.ones((10,)))  # not 2-D
