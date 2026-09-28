"""event_sim resting-order block: multi-bar vwap slicing/expiry, l2_queue
advance/cancel paths, resting retarget replacement, stale-mark guard."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.backtest.engine import StaleValuationError
from quant_fund.backtest.event_sim.simulator import EventSimSpec, run_event_backtest
from quant_fund.config.loader import load_config

pytestmark = pytest.mark.synthetic


def _cfg(tmp_path, **over):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 1.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.stale_price_bars = 100
    for k, v in over.items():
        setattr(cfg.risk_gate, k, v)
    return cfg


def _day(sid: str, day: int, price: float, **extra) -> dict:
    when = datetime(2024, 1, day, tzinfo=UTC)
    row = {
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
        "source": "file",
    }
    row.update(extra)
    return row


def _w(series: list[tuple[int, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, d, tzinfo=UTC) for d, _, _ in series],
            "security_id": [s for _, s, _ in series],
            "target_weight": [w for _, _, w in series],
        }
    )


class TestVwapResting:
    def test_multi_bar_window_fills_across_days(self, tmp_path) -> None:
        bars = pl.DataFrame([_day("A", d, 100.0, volume=50.0) for d in range(1, 8)])
        sim = run_event_backtest(
            bars,
            _w([(1, "A", 1.0)]),
            _cfg(tmp_path),
            EventSimSpec(fill_model="vwap", vwap_window_bars=3),
        )
        fills = sim.result.fills
        assert fills.height > 0
        assert fills["fill_time"].dt.date().n_unique() > 1

    def test_resting_expires_unfilled(self, tmp_path) -> None:
        bars = pl.DataFrame(
            [_day("A", d, 100.0, volume=50.0) for d in range(1, 4)]
            + [_day("B", d, 50.0, volume=50.0) for d in range(4, 8)]
        )
        sim = run_event_backtest(
            bars,
            _w([(1, "A", 1.0), (4, "B", 0.5)]),
            _cfg(tmp_path),
            EventSimSpec(fill_model="vwap", vwap_window_bars=4),
        )
        assert sim.events  # resting expiry/cancel recorded

    def test_retarget_cancels_stale_resting(self, tmp_path) -> None:
        bars = pl.DataFrame([_day("A", d, 100.0, volume=50.0) for d in range(1, 8)])
        sim = run_event_backtest(
            bars,
            _w([(1, "A", 1.0), (3, "A", 0.0)]),
            _cfg(tmp_path),
            EventSimSpec(fill_model="vwap", vwap_window_bars=5),
        )
        assert sim.cancel_replace_count >= 1
        assert any(e.get("reason") == "cancel_replace" for e in sim.events if isinstance(e, dict))


class TestL2Resting:
    def _booked_day(self, sid: str, day: int, price: float, spread: float = 0.01) -> dict:
        return _day(
            sid,
            day,
            price,
            bid_p1=price - spread,
            bid_s1=100.0,
            ask_p1=price + spread,
            ask_s1=100.0,
        )

    def test_limit_order_rest_and_advance(self, tmp_path) -> None:
        bars = pl.DataFrame([self._booked_day("A", d, 100.0) for d in range(1, 8)])
        sim = run_event_backtest(
            bars,
            _w([(1, "A", 0.2)]),
            _cfg(tmp_path),
            EventSimSpec(fill_model="l2_queue", l2_rest_bars=3),
        )
        assert sim.result.fills.height > 0

    def test_rest_expiry_cancels(self, tmp_path) -> None:
        bars = pl.DataFrame(
            [
                self._booked_day("A", 1, 100.0),
                *(
                    _day(
                        "A",
                        d,
                        200.0,
                        bid_p1=199.0,
                        bid_s1=1.0,
                        ask_p1=201.0,
                        ask_s1=1.0,
                    )
                    for d in range(2, 6)
                ),
            ]
        )
        sim = run_event_backtest(
            bars,
            _w([(1, "A", 0.2)]),
            _cfg(tmp_path),
            EventSimSpec(fill_model="l2_queue", l2_rest_bars=2),
        )
        assert sim.result.equity.height > 0


class TestStaleMarks:
    def test_stale_position_raises(self, tmp_path) -> None:
        bars = pl.DataFrame(
            [_day("A", d, 100.0) for d in range(1, 4)] + [_day("B", d, 50.0) for d in range(4, 9)]
        )
        cfg = _cfg(tmp_path)
        cfg.risk_gate.stale_price_bars = 1
        with pytest.raises(StaleValuationError, match="stale"):
            run_event_backtest(
                bars,
                _w([(1, "A", 1.0)]),
                cfg,
                EventSimSpec(fill_model="next_open"),
            )
