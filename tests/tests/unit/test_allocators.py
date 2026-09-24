"""Tests for portfolio allocators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.portfolio.allocators import (
    black_litterman,
    cvar_minimization,
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_volatility,
    kelly_weights,
    maximum_diversification,
    volatility_target,
)


def _cov(rng: np.random.Generator, n: int = 8) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a.T @ a / n + np.eye(n) * 0.3


class TestSimpleAllocators:
    def test_ivp_inversely_proportional(self):
        cov = np.diag([1.0, 4.0, 9.0, 16.0])
        w = inverse_volatility(cov)
        assert np.isclose(w.sum(), 1.0)
        # Weights proportional to 1/sd = 1, 1/2, 1/3, 1/4
        expected = np.array([1.0, 0.5, 1 / 3, 0.25])
        assert np.allclose(w, expected / expected.sum(), atol=1e-8)

    def test_vol_target(self):
        cov = np.diag([0.04, 0.09])
        w = np.array([0.5, 0.5])
        scaled, lev = volatility_target(w, cov, target_vol=0.30)
        vol = np.sqrt(scaled @ cov @ scaled)
        assert abs(vol - 0.30) < 1e-9
        assert lev > 1.0
        # Deleveraging direction also works.
        scaled2, lev2 = volatility_target(w, cov, target_vol=0.05)
        assert lev2 < 1.0
        assert abs(np.sqrt(scaled2 @ cov @ scaled2) - 0.05) < 1e-9

    def test_vol_target_leverage_cap(self):
        cov = np.diag([0.001, 0.001])
        _, lev = volatility_target(np.array([0.5, 0.5]), cov, 0.5, max_leverage=3.0)
        assert lev == 3.0

    def test_failclosed(self):
        with pytest.raises(ValueError):
            inverse_volatility(np.ones((3, 4)))
        with pytest.raises(ValueError):
            volatility_target(np.zeros(2), np.eye(2), 0.1)
        with pytest.raises(ValueError):
            volatility_target(np.array([1.0, 0.0]), np.eye(2), -0.1)


class TestHRP:
    def test_weights_sum_to_one(self):
        rng = np.random.default_rng(0)
        cov = _cov(rng, 10)
        w = hierarchical_risk_parity(cov)
        assert w.shape == (10,)
        assert np.isclose(w.sum(), 1.0)
        assert np.all(w >= -1e-12)

    def test_low_vol_asset_gets_weight(self):
        # One near-zero-variance asset should receive dominant weight.
        cov = np.diag([0.25, 0.25, 0.25, 0.0001])
        w = hierarchical_risk_parity(cov)
        assert w[3] > 0.3

    def test_block_correlation(self):
        rng = np.random.default_rng(1)
        f1 = rng.normal(size=(300, 1))
        f2 = rng.normal(size=(300, 1))
        x = np.hstack([f1 + 0.1 * rng.normal(size=(300, 3)), f2 + 0.1 * rng.normal(size=(300, 3))])
        cov = np.cov(x.T)
        w = hierarchical_risk_parity(cov)
        # Two clusters: total weight split roughly evenly between blocks.
        assert 0.3 < w[:3].sum() < 0.7


class TestERC:
    def test_equal_contributions(self):
        rng = np.random.default_rng(2)
        cov = _cov(rng, 6)
        w = equal_risk_contribution(cov)
        assert np.isclose(w.sum(), 1.0, atol=1e-4)
        rc = w * (cov @ w)
        total = rc.sum()
        # Risk contributions nearly equal.
        assert np.max(np.abs(rc - total / 6)) / total < 0.05

    def test_diagonal_cov(self):
        cov = np.diag([1.0, 4.0, 9.0])
        w = equal_risk_contribution(cov)
        # ERC on diagonal cov: RC_i = w_i^2 * sigma_i^2 equal -> w_i ~ 1/sigma_i.
        expected = np.array([1.0, 0.5, 1 / 3])
        expected /= expected.sum()
        assert np.allclose(w, expected, atol=0.03)


class TestMaxDiv:
    def test_beats_equal_weight_ratio(self):
        rng = np.random.default_rng(3)
        cov = _cov(rng, 6)
        sd = np.sqrt(np.diag(cov))
        w = maximum_diversification(cov)
        dr = (w @ sd) / np.sqrt(w @ cov @ w)
        we = np.full(6, 1 / 6)
        dr_eq = (we @ sd) / np.sqrt(we @ cov @ we)
        assert dr >= dr_eq - 1e-6
        assert np.isclose(w.sum(), 1.0)


class TestBlackLitterman:
    def test_no_views_returns_prior(self):
        rng = np.random.default_rng(4)
        cov = _cov(rng, 5)
        mu, post_cov = black_litterman(cov, tau=0.05)
        assert mu.shape == (5,)
        assert post_cov.shape == (5, 5)

    def test_view_moves_mu(self):
        rng = np.random.default_rng(5)
        cov = _cov(rng, 4)
        P = np.array([[1.0, -1.0, 0.0, 0.0]])
        Q = np.array([0.02])
        mu0, _ = black_litterman(cov, tau=0.05)
        mu1, _ = black_litterman(cov, views_P=P, views_Q=Q, tau=0.05)
        # Bullish asset0-vs-asset1 view raises the spread.
        assert (mu1[0] - mu1[1]) > (mu0[0] - mu0[1])

    def test_failclosed(self):
        with pytest.raises(ValueError):
            black_litterman(np.eye(3), tau=0.0)
        with pytest.raises(ValueError):
            black_litterman(np.eye(3), views_P=np.ones((2, 5)), views_Q=np.ones(2))


class TestKelly:
    def test_uncorrelated(self):
        mu = np.array([0.10, 0.20])
        cov = np.diag([0.04, 0.09])
        w = kelly_weights(mu, cov)
        # w* = Sigma^{-1} mu = mu/var
        assert np.allclose(w, [0.10 / 0.04, 0.20 / 0.09], atol=1e-8)

    def test_half_kelly(self):
        mu = np.array([0.10, 0.0])
        cov = np.diag([0.04, 0.04])
        w = kelly_weights(mu, cov, fraction=0.5)
        assert abs(w[0] - 1.25) < 1e-8
        assert abs(w[1]) < 1e-8

    def test_failclosed(self):
        with pytest.raises(ValueError):
            kelly_weights(np.array([0.1]), np.array([[0.04]]), fraction=1.5)


class TestCVaR:
    def test_weights_and_cvar(self):
        rng = np.random.default_rng(6)
        # Asset 2 has lower tail risk.
        r = np.column_stack(
            [
                rng.normal(0.001, 0.03, 500),
                rng.standard_t(3, 500) * 0.02,
                rng.normal(0.001, 0.01, 500),
            ]
        )
        w, cvar = cvar_minimization(r, alpha=0.95)
        assert np.isclose(w.sum(), 1.0)
        assert cvar > 0.0
        # Low-tail-risk asset should get meaningful weight.
        assert w[2] > 0.2

    def test_failclosed(self):
        with pytest.raises(ValueError):
            cvar_minimization(np.random.default_rng(0).normal(size=(3, 2)))
        with pytest.raises(ValueError):
            cvar_minimization(np.random.default_rng(0).normal(size=(50, 3)), alpha=1.5)
