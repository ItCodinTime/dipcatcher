"""Canon tests: Sinkhorn OT, W2 barycenters (1-D and Gaussian)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.optimal_transport import (
    gaussian_bures,
    sinkhorn_plan,
    wasserstein_barycenter_1d,
)


def test_sinkhorn_marginals_exact() -> None:
    a = np.array([0.3, 0.7])
    b = np.array([0.4, 0.6])
    cost = np.abs(np.subtract.outer(np.arange(2.0), np.arange(2.0)))
    out = sinkhorn_plan(a, b, cost, eps=0.01)
    np.testing.assert_allclose(out["plan"].sum(axis=1), a, atol=1e-6)
    np.testing.assert_allclose(out["plan"].sum(axis=0), b, atol=1e-6)


def test_sinkhorn_diagonal_cheapest() -> None:
    a = np.full(4, 0.25)
    cost = np.abs(np.subtract.outer(np.arange(4.0), np.arange(4.0)))
    out = sinkhorn_plan(a, a, cost, eps=0.005)
    assert out["cost"] < 0.05  # identity coupling ~ free
    assert np.trace(out["plan"]) > 0.8


def test_sinkhorn_degenerates_to_lp_cost() -> None:
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    out = sinkhorn_plan(a, b, cost, eps=0.001)
    # OT LP cost is 0 (identity permutation)
    assert out["cost"] < 0.01


def test_wasserstein_barycenter_midpoint_gaussians() -> None:
    supports = np.stack(
        [
            np.sort(np.random.default_rng(1).normal(0, 1, 500)),
            np.sort(np.random.default_rng(2).normal(2, 1, 500)),
        ]
    )
    probs = np.full((2, 500), 1 / 500)
    out = wasserstein_barycenter_1d(supports, probs, np.array([0.5, 0.5]), n_grid=100)
    # barycenter of N(0,1) and N(2,1) under W2 is N(1,1)
    assert abs(np.mean(out["quantile"]) - 1.0) < 0.15


def test_wasserstein_barycenter_identical_members() -> None:
    sup = np.tile(np.linspace(-1, 1, 50), (3, 1))
    probs = np.full((3, 50), 1 / 50)
    out = wasserstein_barycenter_1d(sup, probs, np.full(3, 1 / 3), n_grid=50)
    np.testing.assert_allclose(out["quantile"], np.linspace(-1, 1, 50), atol=0.05)


def test_gaussian_bures_identical() -> None:
    cov = np.array([[1.0, 0.2], [0.2, 2.0]])
    out = gaussian_bures(np.zeros((2, 2)), np.stack([cov, cov]), np.array([0.5, 0.5]))
    np.testing.assert_allclose(out["cov"], cov, atol=1e-6)


def test_gaussian_bures_diagonal_closed_form() -> None:
    # W2 barycenter variance of diagonal Gaussians: sqrt(S) = mean of sqrt
    c1 = np.diag([1.0, 4.0])
    c2 = np.diag([9.0, 16.0])
    out = gaussian_bures(np.zeros((2, 2)), np.stack([c1, c2]), np.array([0.5, 0.5]))
    expected = np.diag([((1 + 3) / 2) ** 2, ((2 + 4) / 2) ** 2])
    np.testing.assert_allclose(np.diag(out["cov"]), np.diag(expected), atol=1e-3)


def test_gaussian_bures_mean() -> None:
    c = np.eye(2)
    means = np.array([[0.0, 0.0], [2.0, 4.0]])
    out = gaussian_bures(means, np.stack([c, c]), np.array([0.25, 0.75]))
    np.testing.assert_allclose(out["mean"], [1.5, 3.0])


def test_ot_validation() -> None:
    with pytest.raises(ValueError):
        sinkhorn_plan(np.array([0.5, 0.4]), np.array([1.0]), np.zeros((2, 1)), 0.1)
    with pytest.raises(ValueError):
        sinkhorn_plan(np.array([0.5, 0.5]), np.array([1.0]), np.zeros((3, 1)), 0.1)
    with pytest.raises(ValueError):
        sinkhorn_plan(np.array([0.5, 0.5]), np.array([1.0]), np.zeros((2, 1)), -1.0)
    with pytest.raises(ValueError):
        wasserstein_barycenter_1d(np.ones((2, 4)), np.ones((2, 4)), np.ones(3))
    with pytest.raises(ValueError):
        gaussian_bures(np.zeros((2, 2)), np.stack([np.eye(2), np.zeros((2, 2))]), np.ones(2) / 2)
