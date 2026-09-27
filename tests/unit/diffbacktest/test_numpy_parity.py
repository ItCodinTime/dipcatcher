"""Hard NumPy core against the existing research books. No JAX."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from quant_fund.diffbacktest.numpy_core import (
    apply_costs,
    apply_rebalance_band,
    drawdown,
    objective_value,
    sharpe,
    simple_returns,
    simulate,
    strong_trend_prices,
    synthetic_prices,
    terminal_pnl,
    trailing_sigma,
)
from quant_fund.diffbacktest.spec import StrategyParams, limitations
from quant_fund.hedge_lab.directional import (
    antonacci_returns,
    risk_parity_blend,
    topk_long_returns,
    tsmom_book_returns,
)
from quant_fund.hedge_lab.directional import (
    simple_returns as directional_returns,
)
from quant_fund.metrics.returns import max_drawdown, sharpe_ratio
from quant_fund.research.net_replay import MarketPanel, ReplayConfig, Strategy, _weights

pytestmark = pytest.mark.synthetic

_ROOT = Path("src/quant_fund/diffbacktest")


def _tsmom_params(**overrides: object) -> StrategyParams:
    base = dict(
        target_vol=0.40,
        max_gross=1.0,
        rebalance_band=0.0,
        lookback=16.0,
        skip=2.0,
        vol_lookback=8.0,
        long_only=1.0,
        rebalance_every=5,
        delay=1,
        one_way_cost=0.001,
        commission_bps=0.0,
        half_spread_bps=0.0,
        impact_y=0.0,
        periods_per_year=252.0,
    )
    base.update(overrides)
    return StrategyParams(**base)  # type: ignore[arg-type]


def test_reference_sources_do_not_import_jax() -> None:
    for name in ("spec.py", "numpy_core.py", "__init__.py"):
        tree = ast.parse((_ROOT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("jax") for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("jax")


def test_simple_returns_match_directional() -> None:
    px = strong_trend_prices(40, seed=1)
    np.testing.assert_allclose(simple_returns(px), directional_returns(px))


def test_costs_match_hand_calculation() -> None:
    weights = np.array([[0.0], [0.0], [0.2], [0.2], [0.5]], dtype=float)
    returns = np.zeros_like(weights)
    params = _tsmom_params(one_way_cost=0.01, impact_y=0.1, vol_lookback=8.0)
    sigma = np.full_like(weights, 0.2)
    # Bypass simulate(); the row count is too short for a full book.
    net, l1, cost = apply_costs(weights, returns, params, sigma)
    expected_l1 = np.array([0.0, 0.0, 0.2, 0.0, 0.3])
    np.testing.assert_allclose(l1, expected_l1)
    impact = 0.1 * 0.2 * np.power(expected_l1, 1.5)
    linear = 0.01 * expected_l1
    np.testing.assert_allclose(cost, linear + impact)
    np.testing.assert_allclose(net, -(linear + impact))


def test_terminal_pnl_compounds() -> None:
    net = np.array([0.0, 0.01, -0.02])
    assert terminal_pnl(net, 1.0) == pytest.approx(1.0 * 1.01 * 0.98 - 1.0)


def test_sharpe_and_drawdown_delegate() -> None:
    rng = np.random.default_rng(0)
    net = rng.normal(0.0003, 0.01, size=80)
    assert sharpe(net) == pytest.approx(float(sharpe_ratio(net)["sharpe"]))
    assert drawdown(net) == pytest.approx(float(max_drawdown(net)))
    assert objective_value(net, "sharpe") == pytest.approx(sharpe(net))
    assert objective_value(net, "drawdown") == pytest.approx(drawdown(net))


def test_band_zero_is_identity_and_positive_band_cuts_turnover() -> None:
    w = np.array([[0.0, 0.0], [0.5, -0.2], [0.5, -0.2], [0.1, 0.4]], dtype=float)
    np.testing.assert_array_equal(apply_rebalance_band(w, 0.0), w)
    banded = apply_rebalance_band(w, 0.15)
    assert np.sum(np.abs(np.diff(banded, axis=0))) < np.sum(np.abs(np.diff(w, axis=0)))


def test_tsmom_matches_directional_book() -> None:
    px = strong_trend_prices(90, seed=2)
    params = _tsmom_params()
    sim = simulate(px, "tsmom", params)
    ref = tsmom_book_returns(
        px,
        lookback=16,
        skip=2,
        vol_lookback=8,
        asset_vol=0.40,
        long_only=True,
        delay=1,
        rebalance_every=5,
        one_way_cost=0.001,
        max_gross=1.0,
    )
    np.testing.assert_allclose(sim.net, ref, atol=1e-10, rtol=1e-10)


def test_topk_matches_directional_book() -> None:
    px = strong_trend_prices(90, seed=3)
    params = _tsmom_params(top_k=2.0, require_positive=1.0, one_way_cost=0.001)
    sim = simulate(px, "topk", params)
    ref = topk_long_returns(
        px,
        lookback=16,
        skip=2,
        top_k=2,
        delay=1,
        rebalance_every=5,
        one_way_cost=0.001,
        require_positive=True,
        sma=0,
        crash_lookback=0,
    )
    np.testing.assert_allclose(sim.net, ref, atol=1e-10, rtol=1e-10)


def test_antonacci_matches_directional_book() -> None:
    px = strong_trend_prices(90, seed=4)[:, :2]
    params = _tsmom_params()
    sim = simulate(px, "antonacci", params)
    ref = antonacci_returns(
        px[:, 0],
        px[:, 1],
        lookback=16,
        skip=2,
        delay=1,
        rebalance_every=5,
        one_way_cost=0.001,
    )
    np.testing.assert_allclose(sim.net, ref, atol=1e-10, rtol=1e-10)


def test_risk_parity_matches_directional_blend() -> None:
    px = synthetic_prices(80, 4, seed=5, mu=0.0002, sigma=0.01)
    params = _tsmom_params(
        vol_lookback=10.0,
        max_gross=1.0,
        rebalance_band=0.0,
        one_way_cost=0.001,
        delay=1,
    )
    sim = simulate(px, "risk_parity", params)
    streams = {str(i): sim.asset_returns[:, i] for i in range(px.shape[1])}
    ref = risk_parity_blend(streams, lookback=10, delay=1, one_way_cost=0.001)
    np.testing.assert_allclose(sim.net, ref, atol=1e-10, rtol=1e-10)


def _panel(prices: np.ndarray) -> MarketPanel:
    t_len, n_names = prices.shape
    dates = [datetime(2020, 1, 1, tzinfo=UTC)] * t_len
    names = [f"N{i}" for i in range(n_names)]
    known = np.ones((t_len, n_names), dtype=bool)
    return MarketPanel(dates, names, prices, prices, np.ones_like(prices), known)


def test_rank_weights_match_net_replay_constructor() -> None:
    px = synthetic_prices(40, 4, seed=6)
    params = _tsmom_params(
        lookback=8.0,
        fraction=0.5,
        long_short=0.0,
        gross_limit=1.0,
        max_name_weight=0.5,
        target_buffer=0.05,
        rebalance_every=1,
        delay=1,
        rebalance_band=0.0,
    )
    config = ReplayConfig(gross_limit=1.0, max_name_weight=0.5, target_buffer=0.05)
    panel = _panel(px)
    eligible = np.ones(px.shape[1], dtype=bool)
    for family, name in (("momentum", "momentum"), ("reversal", "reversal")):
        sim = simulate(px, name, params)
        strategy = Strategy(name="s", family=family, lookback=8, fraction=0.5, long_short=False)
        for t in range(9, px.shape[0]):
            ref = _weights(panel, t - 1, eligible, strategy, config)
            np.testing.assert_allclose(sim.weights[t], ref, atol=1e-12)


def test_equal_weight_matches_net_replay_constructor() -> None:
    px = synthetic_prices(20, 4, seed=7)
    params = _tsmom_params(
        gross_limit=1.0,
        max_name_weight=0.5,
        target_buffer=0.05,
        delay=1,
        rebalance_band=0.0,
    )
    sim = simulate(px, "equal_weight", params)
    config = ReplayConfig(gross_limit=1.0, max_name_weight=0.5, target_buffer=0.05)
    strategy = Strategy(name="ew", family="equal_weight", lookback=8, fraction=0.5)
    ref = _weights(_panel(px), 10, np.ones(4, dtype=bool), strategy, config)
    np.testing.assert_allclose(sim.weights[-1], ref, atol=1e-12)


def test_weights_do_not_depend_on_future_prices() -> None:
    px = synthetic_prices(60, 3, seed=8)
    params = _tsmom_params(lookback=12.0, skip=2.0, vol_lookback=8.0, rebalance_every=4)
    t0 = 30
    for strategy in ("tsmom", "topk", "risk_parity", "momentum", "reversal", "equal_weight"):
        book = params if strategy != "antonacci" else params
        prices = px if strategy != "antonacci" else px[:, :2]
        shocked = np.array(prices, copy=True)
        shocked[t0:] *= 1.25
        base = simulate(prices, strategy, book).weights
        alt = simulate(shocked, strategy, book).weights
        np.testing.assert_allclose(
            base[: t0 + 1],
            alt[: t0 + 1],
            atol=1e-12,
            err_msg=strategy,
        )


def test_rejects_non_causal_delay_and_bad_prices() -> None:
    px = synthetic_prices(30, 2, seed=9)
    with pytest.raises(ValueError, match="delay"):
        simulate(px, "tsmom", _tsmom_params(delay=0))
    with pytest.raises(ValueError, match="positive"):
        simulate(np.vstack([px, np.zeros((1, 2))]), "equal_weight", _tsmom_params())
    with pytest.raises(ValueError, match="N=2"):
        simulate(synthetic_prices(30, 3, seed=1), "antonacci", _tsmom_params())


def test_limitations_are_honest() -> None:
    text = " ".join(limitations())
    assert "SYNTHETIC" in text
    assert "not a live" in text.lower() or "Not a live" in text
    assert "upper bound" in text
    assert "backtest.engine" in text


def test_trailing_sigma_is_causal() -> None:
    px = synthetic_prices(40, 2, seed=10)
    returns = simple_returns(px)
    sig = trailing_sigma(returns, 8, delay=1)
    assert np.all(sig[:8] == 0.0)
    window = returns[10 - 8 : 10]
    np.testing.assert_allclose(sig[10], np.std(window, axis=0, ddof=1))
