"""Tests for models/random_projection.py — Johnson-Lindenstrauss projections."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.random_projection import (
    gaussian_random_projection,
    johnson_lindenstrauss_min_dim,
    sparse_random_projection,
)


def test_min_dim_monotone() -> None:
    assert johnson_lindenstrauss_min_dim(1000, 0.1) > johnson_lindenstrauss_min_dim(1000, 0.3)
    assert johnson_lindenstrauss_min_dim(100000, 0.2) > johnson_lindenstrauss_min_dim(100, 0.2)


def test_gaussian_preserves_pairwise_distances() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 400))
    out = gaussian_random_projection(x, k=200, rng=rng)
    y = out["projected"]
    assert y.shape == (60, 200)
    i, j = 0, 1
    d_hi = np.linalg.norm(x[i] - x[j])
    d_lo = np.linalg.norm(y[i] - y[j])
    assert abs(d_lo - d_hi) / d_hi < 0.2  # distances roughly preserved


def test_sparse_projection_shapes_and_distances() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((40, 300))
    out = sparse_random_projection(x, k=150, rng=rng)
    y = out["projected"]
    assert y.shape == (40, 150)
    # mean relative distortion over several pairs is modest
    errs = []
    for a in range(0, 10):
        for b in range(a + 1, 10):
            dh = np.linalg.norm(x[a] - x[b])
            dl = np.linalg.norm(y[a] - y[b])
            errs.append(abs(dl - dh) / dh)
    assert np.mean(errs) < 0.2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(1, 0.1)
    with pytest.raises(ValueError):
        gaussian_random_projection(np.ones((5, 4)), k=10)  # k > d
    with pytest.raises(ValueError):
        sparse_random_projection(np.ones((5, 4)), k=2, density=2.0)
