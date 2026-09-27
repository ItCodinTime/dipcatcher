"""NAV accounting, cost monotonicity, and scale invariance of the daily engine."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from quant_fund.backtest.engine import Book, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from tests.property._books import T0, daily_bars, research_config, weights_frame
from tests.property._profiles import adversarial_settings

_step = st.floats(min_value=-0.03, max_value=0.03, allow_nan=False, allow_infinity=False)


def _path(n_days: int, n_names: int, steps: list[float]) -> list[list[float]]:
    prices = [[100.0 * (1.0 + i) for i in range(n_names)]]
    cursor = 0
    for _ in range(1, n_days):
        row = []
        for name in range(n_names):
            prev = prices[-1][name]
            row.append(prev * (1.0 + steps[cursor]))
            cursor += 1
        prices.append(row)
    return prices


def _replay_nav(prices: list[list[float]], result, initial: float, borrow_bps: float) -> None:
    """Cash + marked positions equals each recorded NAV, including borrow."""
    cash = initial
    shares: dict[str, float] = {}
    fills = result.fills
    by_time: dict[object, list[dict[str, object]]] = {}
    if fills.height:
        for row in fills.iter_rows(named=True):
            by_time.setdefault(row["fill_time"], []).append(row)
    for row in result.equity.iter_rows(named=True):
        when = row["event_time"]
        for fill in by_time.get(when, []):
            sid = str(fill["security_id"])
            qty = float(fill["quantity"])
            px = float(fill["price"])
            explicit = (
                float(fill["fee"])
                + float(fill["spread_cost"])
                + float(fill["impact_cost"])
                + float(fill["turnover_cost"])
            )
            cash -= qty * px + explicit
            shares[sid] = shares.get(sid, 0.0) + qty
        day = (when - T0).days
        marks = {f"S{i}": prices[day][i] for i in range(len(prices[day]))}
        short_notional = sum(
            abs(min(qty, 0.0)) * marks.get(sid, 0.0) for sid, qty in shares.items()
        )
        borrow = short_notional * (borrow_bps / 1e4) / 252.0
        cash_marked = cash - borrow
        assert Book(cash=cash_marked, shares=dict(shares)).nav(marks) == pytest.approx(
            float(row["nav"]), rel=1e-9, abs=1e-6
        )


@given(
    n_days=st.integers(min_value=3, max_value=5),
    steps=st.lists(_step, min_size=8, max_size=8),
    weight=st.floats(min_value=-0.2, max_value=0.4, allow_nan=False, allow_infinity=False),
    commission=st.floats(min_value=0.0, max_value=15.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_nav_equals_cash_plus_positions(
    n_days: int,
    steps: list[float],
    weight: float,
    commission: float,
) -> None:
    prices = _path(n_days, 2, steps)
    bars = daily_bars(prices)
    grid = weights_frame(n_days, 2, weight)
    cfg = research_config(commission_bps=commission, half_spread_bps=0.0, borrow_bps_per_year=0.0)
    result = run_backtest(bars, grid, cfg, initial_nav=1_000_000.0)
    _replay_nav(prices, result, 1_000_000.0, 0.0)
    if result.equity.height:
        end = float(result.equity["nav"][-1])
        assert float(result.metrics["total_return"]) == pytest.approx(end / 1_000_000.0 - 1.0)
    assert result.metrics["research_only"] is True
    assert result.metrics["live_pnl_claim"] is False


@given(steps=st.lists(_step, min_size=6, max_size=6))
@adversarial_settings()
def test_zero_target_stays_in_cash(steps: list[float]) -> None:
    prices = _path(4, 1, steps)
    result = run_backtest(
        daily_bars(prices),
        weights_frame(4, 1, 0.0),
        research_config(frictionless=True),
        initial_nav=250_000.0,
    )
    assert result.fills.height == 0
    assert result.equity["nav"].to_list() == pytest.approx([250_000.0] * result.equity.height)


@given(steps=st.lists(_step, min_size=2, max_size=2))
@adversarial_settings()
def test_frictionless_full_investment_tracks_the_total_return_close(steps: list[float]) -> None:
    prices = _path(3, 1, steps)
    initial = 1_000_000.0
    result = run_backtest(
        daily_bars(prices),
        weights_frame(1, 1, 1.0),
        research_config(frictionless=True),
        initial_nav=initial,
    )
    # Decision on day 0 fills at day 1's open, then marks at that day's close.
    assert result.equity.height == 2
    assert float(result.equity["nav"][0]) == pytest.approx(initial * prices[1][0] / prices[1][0])
    assert float(result.equity["nav"][0]) == pytest.approx(initial)
    # Day-2 mark is the total-return close while the share count stays put.
    assert float(result.equity["nav"][1]) == pytest.approx(initial * prices[2][0] / prices[1][0])


@given(
    low=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    high=st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_higher_commission_never_raises_flat_market_nav(low: float, high: float) -> None:
    assume(high >= low)
    # Two dates => one next-open fill. A later rebalance would pay commission
    # on the dust trade, so the cash gap would not equal the first notional.
    prices = [[100.0], [100.0]]
    bars = daily_bars(prices)
    grid = weights_frame(1, 1, 0.4)
    initial = 1_000_000.0
    cheap = run_backtest(
        bars, grid, research_config(commission_bps=low, half_spread_bps=0.0), initial_nav=initial
    )
    rich = run_backtest(
        bars, grid, research_config(commission_bps=high, half_spread_bps=0.0), initial_nav=initial
    )
    assert float(rich.equity["nav"][-1]) <= float(cheap.equity["nav"][-1]) + 1e-6
    assert float(cheap.fills["quantity"][0]) == pytest.approx(float(rich.fills["quantity"][0]))
    notional = abs(float(cheap.fills["quantity"][0]) * float(cheap.fills["price"][0]))
    extra = notional * (high - low) / 1e4
    assert float(cheap.equity["nav"][-1]) - float(rich.equity["nav"][-1]) == pytest.approx(extra)


@given(scale=st.floats(min_value=0.5, max_value=6.0, allow_nan=False, allow_infinity=False))
@adversarial_settings()
def test_nav_return_is_invariant_to_currency_scale(scale: float) -> None:
    prices = [[100.0, 50.0], [101.0, 49.0], [103.0, 51.0], [100.0, 52.0]]
    bars = daily_bars(prices, adv=1e12)
    scaled = bars.with_columns(
        (pl.col("open") * scale).alias("open"),
        (pl.col("close") * scale).alias("close"),
        (pl.col("close_total_return") * scale).alias("close_total_return"),
        (pl.col("adv") * scale).alias("adv"),
    )
    grid = weights_frame(4, 2, 0.25)
    cfg = research_config(commission_bps=5.0, half_spread_bps=2.0, bps_per_turnover=1.0)
    base = run_backtest(bars, grid, cfg, initial_nav=1_000_000.0)
    other = run_backtest(scaled, grid, cfg, initial_nav=1_000_000.0 * scale)
    assert np.asarray(other.equity["nav"].to_list()) / scale == pytest.approx(
        np.asarray(base.equity["nav"].to_list())
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
@adversarial_settings()
def test_backtest_is_deterministic_and_matches_fast_replay(seed: int) -> None:
    rng = np.random.default_rng(seed)
    steps = rng.uniform(-0.02, 0.02, size=6).tolist()
    prices = _path(4, 2, steps)
    bars = daily_bars(prices)
    grid = weights_frame(4, 2, 0.2)
    cfg = research_config(commission_bps=3.0, half_spread_bps=1.0, bps_per_turnover=0.5)
    first = run_backtest(bars, grid, cfg, initial_nav=500_000.0)
    second = run_backtest(bars, grid, cfg, initial_nav=500_000.0)
    fast = run_backtest_fast(bars, grid, cfg, initial_nav=500_000.0)
    assert first.equity["nav"].to_list() == second.equity["nav"].to_list()
    assert fast.equity["nav"].to_list() == pytest.approx(first.equity["nav"].to_list())
    if first.fills.height:
        assert fast.fills["turnover_cost"].to_list() == pytest.approx(
            first.fills["turnover_cost"].to_list()
        )
