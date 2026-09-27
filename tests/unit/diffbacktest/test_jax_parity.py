"""Hard JAX matches NumPy. Smooth net error shrinks as beta grows."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.diffbacktest.jax_core import _libs, simulate_jax, soft_abs, soft_threshold
from quant_fund.diffbacktest.numpy_core import (
    objective_value,
    simulate,
    strong_trend_prices,
    synthetic_prices,
)
from quant_fund.diffbacktest.spec import (
    HARD_PARITY_ATOL,
    PARITY_BETA,
    SMOOTH_NET_ATOL,
    StrategyParams,
)

pytestmark = pytest.mark.synthetic


def _cases() -> list[tuple[str, np.ndarray, StrategyParams]]:
    trend = strong_trend_prices(64, seed=2)
    syn = synthetic_prices(60, 4, seed=5)
    band = dict(one_way_cost=0.001, rebalance_band=0.0, rebalance_every=5, delay=1)
    return [
        (
            "tsmom",
            trend,
            StrategyParams(
                lookback=12,
                skip=0,
                vol_lookback=10,
                target_vol=0.4,
                max_gross=1.0,
                long_only=1,
                **band,
            ),
        ),
        ("topk", trend, StrategyParams(lookback=12, skip=0, top_k=2, require_positive=1, **band)),
        ("antonacci", trend[:, :2], StrategyParams(lookback=12, skip=2, **band)),
        (
            "equal_weight",
            trend,
            StrategyParams(gross_limit=1.0, max_name_weight=1.0, target_buffer=0.05, **band),
        ),
        ("risk_parity", syn, StrategyParams(vol_lookback=10, max_gross=1.5, **band)),
        (
            "momentum",
            syn,
            StrategyParams(
                lookback=8,
                skip=0,
                fraction=0.5,
                long_short=0,
                gross_limit=1.0,
                max_name_weight=1.0,
                target_buffer=0.05,
                **band,
            ),
        ),
        (
            "reversal",
            syn,
            StrategyParams(
                lookback=8,
                skip=0,
                fraction=0.5,
                long_short=0,
                gross_limit=1.0,
                max_name_weight=1.0,
                target_buffer=0.05,
                **band,
            ),
        ),
    ]


@pytest.mark.parametrize(
    "strategy,prices,params",
    _cases(),
    ids=[row[0] for row in _cases()],
)
def test_hard_jax_matches_numpy(strategy: str, prices: np.ndarray, params: StrategyParams) -> None:
    ref = simulate(prices, strategy, params)
    got = simulate_jax(prices, strategy, params, mode="hard")
    np.testing.assert_allclose(
        got.weights, ref.weights, atol=HARD_PARITY_ATOL, rtol=HARD_PARITY_ATOL
    )
    np.testing.assert_allclose(got.net, ref.net, atol=HARD_PARITY_ATOL, rtol=HARD_PARITY_ATOL)
    np.testing.assert_allclose(
        got.turnover, ref.turnover, atol=HARD_PARITY_ATOL, rtol=HARD_PARITY_ATOL
    )
    for name in ("pnl", "pnl_sum", "sharpe", "drawdown"):
        assert objective_value(got.net, name) == pytest.approx(
            objective_value(ref.net, name), abs=HARD_PARITY_ATOL, rel=HARD_PARITY_ATOL
        )


def test_smooth_gap_shrinks_and_meets_atol() -> None:
    for strategy, prices, params in _cases():
        ref = simulate(prices, strategy, params)
        errors = []
        for beta in (8.0, PARITY_BETA):
            got = simulate_jax(prices, strategy, params, mode="smooth", beta=beta)
            errors.append(float(np.max(np.abs(got.net - ref.net))))
        assert errors[1] < errors[0]
        assert errors[1] < SMOOTH_NET_ATOL


def test_ste_forward_weights_match_hard() -> None:
    prices = strong_trend_prices(48, seed=1)
    params = StrategyParams(lookback=12, skip=0, vol_lookback=10, long_only=1, rebalance_band=0.0)
    hard = simulate_jax(prices, "tsmom", params, mode="hard")
    ste = simulate_jax(prices, "tsmom", params, mode="ste", beta=8.0)
    np.testing.assert_allclose(ste.weights, hard.weights, atol=1e-10, rtol=1e-10)


def test_soft_maps_converge() -> None:
    _, jnp = _libs()
    import jax

    assert jax.config.jax_enable_x64
    x = jnp.linspace(-0.8, 0.8, 17)
    np.testing.assert_allclose(np.asarray(soft_threshold(x, 0.0, 8.0)), np.asarray(x), atol=1e-12)
    wide = []
    tight = []
    hard = np.sign(np.asarray(x)) * np.maximum(np.abs(np.asarray(x)) - 0.05, 0.0)
    for beta, bucket in ((8.0, wide), (PARITY_BETA, tight)):
        bucket.append(float(np.max(np.abs(np.asarray(soft_threshold(x, 0.05, beta)) - hard))))
        bucket.append(float(np.max(np.abs(np.asarray(soft_abs(x, beta)) - np.abs(np.asarray(x))))))
    assert tight[0] < wide[0]
    assert tight[1] < wide[1]
    assert tight[0] < 1e-3


def test_bad_mode_and_objective_raise() -> None:
    prices = strong_trend_prices(32, seed=0)
    with pytest.raises(ValueError, match="mode"):
        simulate_jax(prices, "equal_weight", StrategyParams(), mode="nope")
    from quant_fund.diffbacktest.jax_core import objective_gradients

    with pytest.raises(ValueError, match="objective"):
        objective_gradients(prices, "equal_weight", StrategyParams(), objective="sortino")
