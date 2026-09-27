"""Delta-neutral carry engine: paired spot+perp, funding harvest, wick liq."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.carry_engine import run_carry_backtest
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _pair_bars(
    perp_prices: list[float],
    spot_prices: list[float] | None = None,
    sid: str = "A",
    step: timedelta = timedelta(hours=1),
    perp_high: list[float] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    spot_prices = spot_prices if spot_prices is not None else perp_prices
    n = len(perp_prices)
    times = [T0 + i * step for i in range(n)]
    perp = pl.DataFrame(
        {
            "security_id": [sid] * n,
            "event_time": times,
            "open": perp_prices,
            "high": perp_high if perp_high is not None else [p * 1.001 for p in perp_prices],
            "low": [p * 0.999 for p in perp_prices],
            "close": perp_prices,
            "volume": [1_000_000.0] * n,
            "source": ["synthetic"] * n,
        }
    )
    spot = pl.DataFrame(
        {
            "security_id": [sid] * n,
            "event_time": times,
            "open": spot_prices,
            "high": [p * 1.001 for p in spot_prices],
            "low": [p * 0.999 for p in spot_prices],
            "close": spot_prices,
            "volume": [1_000_000.0] * n,
            "source": ["synthetic"] * n,
        }
    )
    return perp, spot


def _weights(w: float, sid: str = "A", at: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + timedelta(hours=at)],
            "security_id": [sid],
            "target_weight": [w],
        }
    )


def _funding(rate: float, at: int, sid: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [sid],
            "event_time": [T0 + timedelta(hours=at)],
            "value": [rate],
        }
    )


def _cfg(tmp_path) -> object:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.stale_price_bars = 3
    return cfg


def test_rejects_negative_weights(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    perp, spot = _pair_bars([100.0] * 10)
    with pytest.raises(ValueError, match=">= 0"):
        run_carry_backtest(perp, spot, None, _weights(-0.5), cfg, initial_nav=1e5)


def test_delta_neutral_price_move_washes(tmp_path) -> None:
    """A 20% price crash should barely move equity: spot loss ≈ perp gain."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 5 + [80.0] * 15
    perp, spot = _pair_bars(prices)
    res = run_carry_backtest(perp, spot, None, _weights(0.5), cfg, initial_nav=1e5)
    nav = res.equity["nav"].to_list()
    # fill at bar1 open → pair 0.5*1e5/100 = 500 units; crash to 80 nets ≈ 0
    assert abs(nav[-1] / 1e5 - 1.0) < 0.02


def test_funding_received_on_short_perp(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    perp, spot = _pair_bars([100.0] * 20)
    res = run_carry_backtest(perp, spot, _funding(0.01, at=5), _weights(0.5), cfg, initial_nav=1e5)
    # 500 units short perp at mark 100, rate 1% → receive ~500
    assert res.metrics["funding_received_total"] == pytest.approx(500.0, rel=0.05)
    assert res.metrics["funding_paid_total"] == pytest.approx(0.0)
    assert res.metrics["total_return"] == pytest.approx(0.005, abs=1e-3)


def test_spot_leg_paid_from_cash(tmp_path) -> None:
    """Target weight 2.0x exceeds cash: spot leg must clamp, no free leverage."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    perp, spot = _pair_bars([100.0] * 10)
    res = run_carry_backtest(perp, spot, None, _weights(2.0), cfg, initial_nav=1e5)
    # cash can't go below the buffer → spot notional < equity
    nav = res.equity["nav"].to_list()
    assert nav[-1] > 0
    gross = res.equity["gross"].max()
    assert gross <= 1.0 + 1e-6


def test_wick_liquidation_unwinds_pair(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.maint_margin_ratio = 0.5
    cfg.perp.liquidation_fee_bps = 50.0
    prices = [100.0] * 10
    highs = [100.0] * 3 + [400.0] + [100.0] * 6  # wick spike on bar 3
    perp, spot = _pair_bars(prices, perp_high=highs)
    res = run_carry_backtest(perp, spot, None, _weights(0.9), cfg, initial_nav=1e5)
    assert res.metrics["liquidation_count"] >= 1


def test_liquidation_unwinds_spot_at_close_not_low(tmp_path) -> None:
    """Force-closed perp leg pays the wick; the hedge leg is unwound at the
    bar's close — never at the opposite extreme of the same bar (that would
    fabricate a cross-venue spread that never traded simultaneously)."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.maint_margin_ratio = 0.5
    prices = [100.0] * 10
    highs = [100.0] * 3 + [400.0] + [100.0] * 6  # perp wick on bar 3
    perp, spot = _pair_bars(prices, perp_high=highs)
    spot = spot.with_columns(
        pl.when(pl.col("event_time") == T0 + timedelta(hours=3))
        .then(50.0)  # spot prints a deep low that never coincides with the perp high
        .otherwise(pl.col("low"))
        .alias("low")
    )
    res = run_carry_backtest(perp, spot, None, _weights(0.9), cfg, initial_nav=1e5)
    pa = res.metrics["pnl_attribution"]
    assert pa["liquidation_events"]
    ev = pa["liquidation_events"][0]
    assert ev["liq_perp_price"] == pytest.approx(400.0)  # short pays the wick
    assert ev["liq_spot_price"] == pytest.approx(100.0)  # hedge unwinds at close
    # attribution closes the books: equity change == sum of components
    assert abs(pa["conservation_error"]) < 1e-6


def test_missing_spot_bar_blocks_entry(tmp_path) -> None:
    """No spot print at exec bar → pair cannot open (no stale-hedge fills)."""
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    perp, _ = _pair_bars([100.0] * 10)
    # spot panel starts 5 bars late → exec at bar1 has no spot mark
    _, spot_late = _pair_bars([100.0] * 5)
    spot_late = spot_late.with_columns(
        (pl.col("event_time") + pl.duration(hours=5)).alias("event_time")
    )
    res = run_carry_backtest(perp, spot_late, None, _weights(0.5, at=0), cfg, initial_nav=1e5)
    assert res.fills.height == 0
