"""Regression tests for the event_sim deep audit (look-ahead, fail-open, parity)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.event_sim.costs import spread_crossing_cost
from quant_fund.backtest.event_sim.sensitivity import execution_sensitivity
from quant_fund.backtest.event_sim.simulator import EventSimSpec, run_event_backtest
from quant_fund.config.loader import load_config
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


def _cfg(tmp_path, *, frictionless: bool = True):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = frictionless
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    return cfg


def _bars(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _day(sid: str, day: int, price: float, *, source: str = "file", hour: int = 0) -> dict:
    when = datetime(2024, 1, day, hour, tzinfo=UTC)
    return {
        "security_id": sid,
        "event_time": when,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "close_total_return": price,
        "volume": 1_000_000.0,
        "adv": 100_000_000.0,
        "vol_20": 0.02,
        "source": source,
    }


def _weights(pairs: list[tuple[int, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, day, tzinfo=UTC) for day, _sid, _w in pairs],
            "security_id": [sid for _day_, sid, _w in pairs],
            "target_weight": [w for _day_, _sid, w in pairs],
        }
    )


def test_held_name_with_bad_open_is_marked_at_pre_exec_price(tmp_path) -> None:
    """A held name whose exec-bar open is invalid must keep its prior mark.

    Using that bar's own close in the open-time NAV leaks future prices into
    sizing; the engine's pre_exec_marks contract applies to event_sim too.
    """
    cfg = _cfg(tmp_path, frictionless=True)
    rows = [_day("A", day, 100.0) for day in range(1, 4)]
    for day, price in ((1, 50.0), (2, 50.0)):
        rows.append(_day("B", day, price))
    b3 = _day("B", 3, 500.0)
    b3["open"] = 0.0  # invalid open: no exec print, but a valid close prints
    rows.append(b3)
    bars = _bars(rows)
    weights = _weights([(1, "B", 0.4), (2, "A", 0.5), (2, "B", 0.4)])
    ref = run_backtest(bars, weights, cfg, initial_nav=100_000.0)
    sim = run_event_backtest(bars, weights, cfg, EventSimSpec(), initial_nav=100_000.0)
    np.testing.assert_allclose(
        sim.result.equity["nav"].to_numpy(), ref.equity["nav"].to_numpy(), atol=1e-9
    )
    assert sim.result.fills.height == ref.fills.height
    a_fills = sim.result.fills.filter(pl.col("security_id") == "A")
    assert a_fills.height == 1  # the leaked close would overstate NAV and cash-reject this buy
    assert float(a_fills["quantity"][0]) == pytest.approx(500.0, abs=1e-6)


def test_l2_resting_expires_after_starved_book(tmp_path) -> None:
    """A resting order whose bars run out must leave the book, not freeze it."""
    cfg = _cfg(tmp_path, frictionless=True)
    bars = _bars([_day("A", day, 100.0) for day in range(1, 6)])
    weights = _weights([(1, "A", 1.0), (3, "A", 1.0), (4, "A", 1.0)])
    books = {
        ("A", datetime(2024, 1, 2, tzinfo=UTC)): OrderBookSnapshot(
            security_id="A",
            symbol="A",
            event_time=datetime(2024, 1, 2, tzinfo=UTC),
            available_time=datetime(2024, 1, 2, tzinfo=UTC),
            source="test",
            bids=[BookLevel(price=99.0, size=50.0)],
            asks=[BookLevel(price=101.0, size=5.0)],
            depth=1,
        ),
        ("A", datetime(2024, 1, 5, tzinfo=UTC)): OrderBookSnapshot(
            security_id="A",
            symbol="A",
            event_time=datetime(2024, 1, 5, tzinfo=UTC),
            available_time=datetime(2024, 1, 5, tzinfo=UTC),
            source="test",
            bids=[BookLevel(price=99.0, size=50.0)],
            asks=[BookLevel(price=101.0, size=7.0)],
            depth=1,
        ),
    }
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fill_model="l2_queue", l2_rest_bars=1),
        initial_nav=100_000.0,
        books=books,
    )
    # day4 starves the resting order (no book); day5 must re-walk the book.
    assert sim.result.fills.height == 2
    assert float(sim.result.fills["quantity"][0]) == pytest.approx(5.0)
    assert float(sim.result.fills["quantity"][1]) == pytest.approx(7.0)


def test_duplicate_bar_panel_rejected(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    bars = _bars([_day("A", 1, 100.0), _day("A", 1, 100.0), _day("A", 2, 100.0)])
    weights = _weights([(1, "A", 1.0)])
    with pytest.raises(ValueError, match="duplicate bars"):
        run_event_backtest(bars, weights, cfg, EventSimSpec(), initial_nav=10_000.0)


def test_borrow_charge_not_clamped_by_cash(tmp_path) -> None:
    """The borrow debit is unconditional — engine parity even when it sinks cash."""
    cfg = _cfg(tmp_path, frictionless=False)
    cfg.costs.commission_bps = 0.0
    cfg.costs.half_spread_bps = 0.0
    cfg.costs.impact_y = 0.0
    cfg.costs.borrow_bps_per_year = 10_000_000.0
    bars = _bars([_day("A", day, 100.0) for day in range(1, 5)])
    weights = _weights([(1, "A", -1.0)])
    ref = run_backtest(bars, weights, cfg, initial_nav=1_000.0)
    sim = run_event_backtest(bars, weights, cfg, EventSimSpec(), initial_nav=1_000.0)
    np.testing.assert_allclose(
        sim.result.equity["nav"].to_numpy(), ref.equity["nav"].to_numpy(), atol=1e-6
    )
    # borrow = 1000 * (1e7/1e4)/252 ≈ 3968.25 > proceeds+cash → cash goes negative.
    assert sim.min_cash == pytest.approx(1_000.0 + 1_000.0 - 100_000_000.0 / 25_200.0, abs=1e-6)


def test_sensitivity_deltas_are_baseline_minus_scenario(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=False)
    cfg.costs.impact_y = 0.2
    cfg.costs.half_spread_bps = 5.0
    rows = [_day("A", day, 100.0 + day) for day in range(1, 7)]
    bars = _bars(rows)
    weights = _weights([(1, "A", 0.5), (4, "A", 0.0)])
    report = execution_sensitivity(
        bars,
        weights,
        cfg,
        signal_latencies=(0, 1),
        exchange_latencies=(0,),
        impact_multipliers=(1.0, 2.0),
        initial_nav=100_000.0,
        seed=3,
    )
    rows_by_key = {(r["signal_to_order_bars"], r["impact_multiplier"]): r for r in report["rows"]}
    degraded = rows_by_key[(0, 2.0)]
    base = rows_by_key[(0, 1.0)]
    # Positive delta means the scenario made less than the baseline cell.
    assert degraded["net_pnl_delta"] == pytest.approx(
        float(base["net_pnl"]) - float(degraded["net_pnl"])
    )
    assert float(degraded["net_pnl_delta"]) >= 0.0
    assert degraded["signal_to_order"] == "0 bars"
    assert rows_by_key[(1, 2.0)]["signal_to_order"] == "1 bars"


def test_vwap_nan_volume_does_not_overfill(tmp_path) -> None:
    """A bar with non-finite volume is no tape — the carry must not fill there."""
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day, volume in ((1, 1e6), (2, 10.0), (3, float("nan")), (4, 5_000.0), (5, 1e6)):
        row = _day("A", day, 100.0)
        row["volume"] = volume
        rows.append(row)
    bars = _bars(rows)
    weights = _weights([(1, "A", 1.0), (2, "A", 1.0), (3, "A", 1.0)])
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fill_model="vwap", vwap_window_bars=3),
        initial_nav=100_000.0,
    )
    # Placement window sees 10 and 5000 → 10 fills on day 2, 990 rests. The
    # NaN-volume bar must skip the carry; day 4 completes the order.
    day3_fills = sim.result.fills.filter(pl.col("fill_time") == datetime(2024, 1, 3, tzinfo=UTC))
    assert day3_fills.height == 0
    assert float(sim.result.fills["quantity"].abs().sum()) == pytest.approx(1_000.0, abs=1e-6)


def test_colliding_fills_accumulate_turnover(tmp_path) -> None:
    """Two fills landing on one bar must accumulate turnover, not reset it."""
    cfg = _cfg(tmp_path, frictionless=True)
    dates = [datetime(2024, 1, d, tzinfo=UTC) for d in (1, 2, 8, 9)]
    rows = []
    for when in dates:
        for sid, price in (("A", 100.0), ("B", 100.0)):
            row = _day(sid, when.day, price)
            row["event_time"] = when
            rows.append(row)
    bars = _bars(rows)
    weights = pl.DataFrame(
        {
            "event_time": [dates[0], dates[1]],
            "security_id": ["A", "B"],
            "target_weight": [0.5, 0.5],
        }
    )
    # signal_to_order=6d collapses both signals' ORDER onto Jan 8, so both
    # FILL on Jan 9: A +0.5 turn, then A -0.5 and B +0.5 → 1.5 total.
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(signal_to_order=timedelta(days=6)),
        initial_nav=1_000.0,
    )
    assert sim.result.fills.height == 3
    assert sim.result.equity["turnover"][-1] == pytest.approx(1.5, abs=1e-9)


def test_expired_signal_stages_log_cancel_events(tmp_path) -> None:
    cfg = _cfg(tmp_path, frictionless=True)
    bars = _bars([_day("A", day, 100.0) for day in range(1, 4)])
    weights = _weights([(2, "A", 1.0)])  # signals fire only for indices < n-1
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(signal_to_order_bars=2),
        initial_nav=10_000.0,
    )
    cancels = [e for e in sim.events if e["kind"] == "CANCEL"]
    assert any(e["reason"] == "expired_after_sample" for e in cancels)
    sim2 = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(order_to_exchange_bars=2),
        initial_nav=10_000.0,
    )
    cancels2 = [e for e in sim2.events if e["kind"] == "CANCEL"]
    assert any(e["reason"] == "expired_after_sample" for e in cancels2)


def test_ambiguous_latency_spec_rejected() -> None:
    with pytest.raises(ValueError, match="not both"):
        EventSimSpec(signal_to_order=timedelta(days=1), signal_to_order_bars=1)
    with pytest.raises(ValueError, match="not both"):
        EventSimSpec(order_to_exchange=timedelta(0), order_to_exchange_bars=2)


def test_degraded_book_uses_fallback_half_spread(tmp_path) -> None:
    """Bars carrying book columns but null quotes still pay a nonzero spread."""
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day in range(1, 4):
        row = _day("A", day, 100.0)
        row["best_bid"] = None
        row["best_ask"] = None
        row["top_bid_size"] = None
        row["top_ask_size"] = None
        rows.append(row)
    bars = _bars(rows)
    weights = _weights([(1, "A", 0.5)])
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(fill_model="l2_queue", l2_rest_bars=0),
        initial_nav=100_000.0,
    )
    assert sim.result.fills.height >= 1
    # Fallback half-spread of 4bps over $100 → ask 100.02, not ~100.0001.
    assert float(sim.result.fills["price"][0]) == pytest.approx(100.02, abs=1e-9)


def test_spread_crossing_requires_quantity_for_quote_override() -> None:
    """A quote half-spread without a quantity must not silently use the bps path."""
    with pytest.raises(ValueError, match="requires quantity"):
        spread_crossing_cost(1_000.0, 5.0, half_spread=0.5)
    # quantity alone is the normal bps path — quote_execution_cost passes it always.
    assert spread_crossing_cost(1_000.0, 5.0, quantity=10.0) == pytest.approx(0.5)
    assert spread_crossing_cost(1_000.0, 5.0, half_spread=0.5, quantity=10.0) == pytest.approx(5.0)


def test_min_notional_rechecked_after_participation_clamp(tmp_path) -> None:
    """A clamped order that lands below min_notional must be skipped, not filled."""
    cfg = _cfg(tmp_path, frictionless=True)
    rows = []
    for day in range(1, 4):
        row = _day("A", day, 100.0)
        row["adv"] = 3_000.0
        rows.append(row)
    bars = _bars(rows)
    weights = _weights([(1, "A", 1.0)])
    # Non-legacy spec: min_notional != 1.0. Participation clamps the desired
    # $100k order to $3k notional — below the $5k min, so nothing fills.
    sim = run_event_backtest(
        bars,
        weights,
        cfg,
        EventSimSpec(min_notional=5_000.0),
        initial_nav=100_000.0,
    )
    assert sim.result.fills.height == 0
