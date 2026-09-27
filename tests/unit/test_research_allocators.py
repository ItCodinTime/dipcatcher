"""Independent numerical oracles and timing/constraint regressions."""

import numpy as np
import pytest
from scipy.optimize import brentq

from quant_fund.portfolio.research_allocators import (
    multi_period_target,
    risk_constrained_kelly,
    volatility_managed_weights,
)


def test_one_period_matches_closed_form_markowitz():
    result = multi_period_target(np.array([[0.02, 0.03]]), np.diag([0.1, 0.2]), np.zeros(2))
    np.testing.assert_allclose(result.weights[0], [0.2, 0.15], atol=1e-5)


def test_multi_period_anticipates_future_alpha_with_costs():
    # Myopic return cannot pay the entry spread; two known future periods can.
    single = multi_period_target(
        np.array([[0.01]]), np.array([[0.01]]), np.zeros(1), linear_cost=0.015
    )
    multi = multi_period_target(
        np.array([[0.01], [0.02]]), np.array([[0.01]]), np.zeros(1), linear_cost=0.015
    )
    assert single.weights[0, 0] < 1e-5
    assert multi.weights[0, 0] > 0.5


def test_mpo_turnover_and_infeasible_constraints():
    result = multi_period_target(np.array([[1.0, 0.2]]), np.eye(2), np.zeros(2), turnover_limit=0.1)
    assert result.weights.sum() <= 0.1 + 1e-7
    with pytest.raises(ValueError, match="failed"):
        multi_period_target(
            np.ones((1, 2)), np.eye(2), np.array([1.0, 0]), gross_limit=0.5, turnover_limit=0.1
        )


@pytest.mark.parametrize(
    "cov",
    [
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, 0.0], [1.0, 1.0]]),
        np.full((2, 2), np.nan),
    ],
)
def test_mpo_rejects_bad_covariance(cov):
    with pytest.raises(ValueError, match="covariance"):
        multi_period_target(np.ones((2, 2)), cov, np.zeros(2))


def test_kelly_matches_independent_scalar_constrained_solution():
    gross = np.array([[1.5], [0.6]])
    p = np.array([0.7, 0.3])
    lam = np.log(0.1) / np.log(0.7)

    def bound(w):
        return p @ (1 + w * (gross[:, 0] - 1)) ** (-lam) - 1

    root = brentq(bound, 1e-5, 1.0)
    unconstrained = (0.7 * 0.5 - 0.3 * 0.4) / (0.5 * 0.4)
    expected = min(root, unconstrained, 1.0)
    result = risk_constrained_kelly(gross, p)
    assert result.weights[0] == pytest.approx(expected, abs=2e-5)
    assert bound(result.weights[0]) <= 1e-7
    assert result.weights.sum() == pytest.approx(1)


def test_kelly_cash_for_negative_edge_and_monotone_risk():
    r = risk_constrained_kelly(np.array([[0.8], [1.1]]), np.array([0.8, 0.2]))
    assert r.weights[-1] > 0.9999
    loose = risk_constrained_kelly(
        np.array([[1.5], [0.6]]), np.array([0.7, 0.3]), drawdown_probability=0.3
    )
    tight = risk_constrained_kelly(
        np.array([[1.5], [0.6]]), np.array([0.7, 0.3]), drawdown_probability=0.01
    )
    assert tight.weights[0] < loose.weights[0]


@pytest.mark.parametrize(
    "p", [np.array([0.3, 0.3]), np.array([1.1, -0.1]), np.array([np.nan, 0.3])]
)
def test_kelly_invalid_distribution(p):
    with pytest.raises(ValueError, match="probabilities"):
        risk_constrained_kelly(np.array([[1.5], [0.6]]), p)


def test_volatility_weights_use_variance_and_strictly_past_returns():
    r = np.array([0.1, -0.1, 0.2, -0.2, 0.1, 0.5])
    w = volatility_managed_weights(r, training_scale=0.005, window=2)
    assert w[2] == pytest.approx(0.5)  # c/0.01, not c/sqrt(0.01)
    changed = r.copy()
    changed[3:] = 0.9
    other = volatility_managed_weights(changed, training_scale=0.005, window=2)
    np.testing.assert_array_equal(w[:4], other[:4])
    assert np.all(w[:2] == 0)


def test_volatility_floor_and_cap():
    w = volatility_managed_weights(np.zeros(10), training_scale=0.001, window=2, max_leverage=0.7)
    np.testing.assert_allclose(w[2:], 0.7)
    with pytest.raises(ValueError):
        volatility_managed_weights(np.zeros(10), training_scale=float("nan"))


def test_kelly_daily_return_conditioning_and_exact_cash_optimum():
    rng = np.random.default_rng(100)
    r = 0.0002 + rng.normal(size=(140, 3)) * [0.012, 0.008, 0.016]
    r[100:] *= 1.7
    result = risk_constrained_kelly(1 + r[95:135], np.full(40, 0.025))
    lam = np.log(0.1) / np.log(0.7)
    growth = np.column_stack([1 + r[95:135], np.ones(40)]) @ result.weights
    assert np.mean(growth ** (-lam)) <= 1 + 1e-7
    assert result.solver_status == "optimal"
    cash = risk_constrained_kelly(1 + r[62:102], np.full(40, 0.025))
    np.testing.assert_array_equal(cash.weights, [0.0, 0.0, 0.0, 1.0])
    assert cash.solver_status == "analytic_cash_optimum"
