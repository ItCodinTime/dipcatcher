"""USDT-M perp backtester: funding cashflows, margin, liquidation, annualization."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.engine import StaleValuationError
from quant_fund.backtest.perp_engine import infer_periods_per_year, run_perp_backtest
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _bars(
    prices: list[float], sid: str = "A", step: timedelta = timedelta(hours=1)
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [sid] * len(prices),
            "event_time": [T0 + i * step for i in range(len(prices))],
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1_000_000.0] * len(prices),
            "source": ["synthetic"] * len(prices),
        }
    )


def _weights(
    w: float, sid: str = "A", at: int = 0, step: timedelta = timedelta(hours=1)
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + at * step],
            "security_id": [sid],
            "target_weight": [w],
        }
    )


def _funding(
    rate: float, at: int, sid: str = "A", step: timedelta = timedelta(hours=1)
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [sid],
            "event_time": [T0 + at * step],
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


def test_infer_periods_per_year_hourly() -> None:
    times = [T0 + i * timedelta(hours=1) for i in range(10)]
    ppy = infer_periods_per_year(times)
    assert abs(ppy - 8766.0) < 1.0


def test_infer_periods_per_year_4h() -> None:
    times = [T0 + i * timedelta(hours=4) for i in range(10)]
    assert abs(infer_periods_per_year(times) - 2191.5) < 1.0


def test_funding_debits_longs_and_credits_shorts(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    prices = [100.0] * 20
    # Long book: funding at bar 5 costs qty*mark*rate ≈ (0.5*1e5/100)*100*0.01 = 500
    res_long = run_perp_backtest(
        _bars(prices), _funding(0.01, at=5), _weights(0.5), cfg, initial_nav=1e5
    )
    # Short book earns the same funding
    res_short = run_perp_backtest(
        _bars(prices), _funding(0.01, at=5), _weights(-0.5), cfg, initial_nav=1e5
    )
    assert res_long.metrics["funding_paid_total"] == pytest.approx(500.0, rel=0.05)
    assert res_long.metrics["total_return"] == pytest.approx(-0.005, abs=1e-3)
    assert res_short.metrics["total_return"] == pytest.approx(0.005, abs=1e-3)
    assert res_long.metrics["funding_events_applied"] == 1


def test_funding_disabled_by_config(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.funding_enabled = False
    res = run_perp_backtest(
        _bars([100.0] * 10), _funding(0.01, at=5), _weights(0.5), cfg, initial_nav=1e5
    )
    assert res.metrics["funding_events_applied"] == 0
    assert res.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_leverage_cap_scales_orders(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 2.0
    res = run_perp_backtest(_bars([100.0] * 10), None, _weights(5.0), cfg, initial_nav=1e5)
    gross = res.equity["gross"].max()
    assert gross <= 2.0 + 1e-6


def test_wick_liquidation_on_crash(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.max_leverage = 3.0
    cfg.perp.maint_margin_ratio = 0.05
    # enter at 100, price collapses to 60 → wick check liquidates
    prices = [100.0, 100.0, 60.0, 60.0, 60.0, 60.0]
    res = run_perp_backtest(_bars(prices), None, _weights(2.9), cfg, initial_nav=1e4)
    assert res.metrics["liquidation_count"] >= 1
    assert res.metrics["liquidation_cost"] > 0.0
    assert res.metrics["ruined"] or res.metrics["total_return"] < -0.5


def test_fill_delay_bars_shifts_execution(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.fill_delay_bars = 2
    res = run_perp_backtest(_bars([100.0] * 10), None, _weights(0.5, at=0), cfg, initial_nav=1e5)
    assert res.fills.height == 1
    # signal at bar 0 → exec at bar 3 (0 + 1 + delay 2)
    assert res.fills["fill_time"][0] == T0 + timedelta(hours=3)


def test_stale_mark_fails_closed(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.risk_gate.stale_price_bars = 2
    a = _bars([100.0] * 10, sid="A")
    b = _bars([50.0] * 10, sid="B").filter(
        pl.col("event_time") < T0 + timedelta(hours=3)
    )  # B goes dark after bar 2
    bars = pl.concat([a, b])
    w = pl.DataFrame(
        {
            "event_time": [T0, T0],
            "security_id": ["A", "B"],
            "target_weight": [0.4, 0.4],
        }
    )
    with pytest.raises(StaleValuationError):
        run_perp_backtest(bars, None, w, cfg, initial_nav=1e5)


def test_metrics_carry_research_labels_and_crypto_annualization(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    res = run_perp_backtest(
        _bars([100.0, 100.5, 101.0, 100.5, 101.5] * 4),
        None,
        _weights(0.5),
        cfg,
        initial_nav=1e5,
    )
    m = res.metrics
    assert m["research_only"] is True
    assert m["live_pnl_claim"] is False
    assert m["book_type"] == "usdtm_perp"
    assert m["periods_per_year"] == pytest.approx(8766.0, rel=0.01)
    assert "cagr" in m and "funding_paid_total" in m


def test_funding_spike_multiplier_scales_cashflow(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.frictionless = True
    cfg.perp.funding_spike_multiplier = 3.0
    res = run_perp_backtest(
        _bars([100.0] * 10), _funding(0.01, at=5), _weights(0.5), cfg, initial_nav=1e5
    )
    assert res.metrics["funding_paid_total"] == pytest.approx(1500.0, rel=0.05)
