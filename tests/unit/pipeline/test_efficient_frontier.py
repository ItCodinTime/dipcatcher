"""Tests for models/efficient_frontier.py — Markowitz mean-variance frontier."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.efficient_frontier import (
    frontier_variance,
    frontier_weights,
    min_variance_portfolio,
    tangency_portfolio,
)


def _inputs():
    mu = np.array([0.08, 0.12, 0.10])
    a = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 0.8]])
    cov = a @ a.T / 10.0
    return mu, cov


def test_min_variance_is_minimal() -> None:
    mu, cov = _inputs()
    w = min_variance_portfolio(cov)
    assert abs(float(w.sum()) - 1.0) < 1e-9
    v_min = float(w @ cov @ w)
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.random(3)
        r /= r.sum()
        assert float(r @ cov @ r) >= v_min - 1e-9


def test_frontier_weights_hit_target_return() -> None:
    mu, cov = _inputs()
    for target in (0.09, 0.10, 0.11):
        w = frontier_weights(mu, cov, target)
        assert abs(float(w.sum()) - 1.0) < 1e-9
        assert abs(float(w @ mu) - target) < 1e-9
        # analytic variance matches the realised portfolio variance
        assert abs(float(w @ cov @ w) - frontier_variance(mu, cov, target)) < 1e-9


def test_frontier_variance_minimised_at_global_min() -> None:
    mu, cov = _inputs()
    # global min-variance return is B/A; variance there is minimal
    grid = np.linspace(0.06, 0.14, 50)
    vs = [frontier_variance(mu, cov, m) for m in grid]
    w = min_variance_portfolio(cov)
    assert min(vs) >= float(w @ cov @ w) - 1e-9


def test_tangency_sums_to_one() -> None:
    mu, cov = _inputs()
    w = tangency_portfolio(mu, cov, rf=0.02)
    assert abs(float(w.sum()) - 1.0) < 1e-9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        min_variance_portfolio(np.zeros((3, 3)))  # singular
    with pytest.raises(ValueError):
        frontier_weights(np.array([0.1]), np.array([[0.1]]), 0.1)  # one asset
