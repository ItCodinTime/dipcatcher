"""Finite differences against reverse-mode gradients of the relaxed book."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.diffbacktest.jax_core import (
    _libs,
    objective_gradients,
    objective_value_jax,
    sign_ste,
)
from quant_fund.diffbacktest.numpy_core import simulate, strong_trend_prices, synthetic_prices
from quant_fund.diffbacktest.spec import GRAD_BETA, StrategyParams, active_parameters, pack, unpack

pytestmark = pytest.mark.synthetic

_BETA = GRAD_BETA


def _tsmom_params() -> StrategyParams:
    return StrategyParams(
        target_vol=0.35,
        max_gross=0.8,
        rebalance_band=0.02,
        lookback=14.0,
        skip=2.0,
        vol_lookback=10.0,
        long_only=0.7,
        one_way_cost=0.002,
        commission_bps=1.0,
        half_spread_bps=1.0,
        impact_y=0.05,
        rebalance_every=5,
        delay=1,
    )


def _rp_params() -> StrategyParams:
    return StrategyParams(
        vol_lookback=12.0,
        max_gross=0.8,
        rebalance_band=0.02,
        one_way_cost=0.002,
        commission_bps=1.0,
        half_spread_bps=1.0,
        impact_y=0.05,
        delay=1,
    )


def _fd_params(
    prices: np.ndarray,
    strategy: str,
    params: StrategyParams,
    objective: str,
    step: float,
) -> None:
    names = active_parameters(strategy)
    theta = pack(params, names)
    grad = objective_gradients(prices, strategy, params, objective, mode="smooth", beta=_BETA)
    numeric = np.empty_like(theta)
    for i in range(theta.size):
        up = theta.copy()
        dn = theta.copy()
        up[i] += step
        dn[i] -= step
        f_up = objective_value_jax(
            prices, strategy, unpack(up, names, params), objective, mode="smooth", beta=_BETA
        )
        f_dn = objective_value_jax(
            prices, strategy, unpack(dn, names, params), objective, mode="smooth", beta=_BETA
        )
        numeric[i] = (f_up - f_dn) / (2.0 * step)
    np.testing.assert_allclose(grad.parameter_gradient, numeric, rtol=5e-2, atol=5e-4)


def test_parameter_gradients_match_finite_differences() -> None:
    prices = synthetic_prices(40, 3, seed=1)
    for objective in ("pnl", "pnl_sum", "sharpe", "drawdown"):
        _fd_params(prices, "risk_parity", _rp_params(), objective, step=1e-4)
    _fd_params(strong_trend_prices(40, seed=1), "tsmom", _tsmom_params(), "sharpe", step=1e-4)


def test_price_directional_derivative() -> None:
    prices = synthetic_prices(36, 3, seed=2)
    params = _rp_params()
    grad = objective_gradients(prices, "risk_parity", params, "pnl", mode="smooth", beta=_BETA)
    rng = np.random.default_rng(0)
    direction = rng.normal(size=prices.shape)
    direction[0] = 0.0
    direction *= 0.2 * np.min(prices[1:]) / np.max(np.abs(direction[1:]))
    step = 1e-3

    def f(path: np.ndarray) -> float:
        return objective_value_jax(path, "risk_parity", params, "pnl", mode="smooth", beta=_BETA)

    numeric = (f(prices + step * direction) - f(prices - step * direction)) / (2.0 * step)
    analytic = float(np.sum(grad.price_gradient * direction))
    np.testing.assert_allclose(analytic, numeric, rtol=5e-2, atol=1e-3)


def test_hard_pnl_sum_cost_derivative_is_minus_turnover() -> None:
    prices = synthetic_prices(40, 3, seed=3)
    params = StrategyParams(
        one_way_cost=0.002, commission_bps=2.0, half_spread_bps=1.0, impact_y=0.0
    )
    sim = simulate(prices, "equal_weight", params)
    grad = objective_gradients(prices, "equal_weight", params, "pnl_sum", mode="hard", beta=_BETA)
    names = grad.parameter_names
    assert grad.parameter_gradient[names.index("one_way_cost")] == pytest.approx(
        -float(np.sum(sim.turnover)), abs=1e-8
    )
    assert grad.parameter_gradient[names.index("commission_bps")] == pytest.approx(
        -float(np.sum(sim.turnover)) / 1e4, abs=1e-8
    )
    assert grad.parameter_gradient[names.index("half_spread_bps")] == pytest.approx(
        -float(np.sum(sim.turnover)) / 1e4, abs=1e-8
    )


def test_sign_ste_forward_is_sign_and_vjp_is_tanh() -> None:
    jax, jnp = _libs()
    x = jnp.array([-1.5, -0.2, 0.0, 0.3, 2.0])
    weights = jnp.array([0.2, -0.4, 0.1, 0.5, -0.3])
    beta = 4.0
    np.testing.assert_allclose(np.asarray(sign_ste(x, beta)), np.asarray(jnp.sign(x)), atol=0.0)

    def through_ste(v: object) -> object:
        return jnp.sum(sign_ste(v, beta) * weights)

    def through_tanh(v: object) -> object:
        return jnp.sum(jnp.tanh(beta * v) * weights)

    np.testing.assert_allclose(
        np.asarray(jax.grad(through_ste)(x)),
        np.asarray(jax.grad(through_tanh)(x)),
        atol=1e-12,
    )
