"""event_sim simulator edge paths: EventSimSpec validation matrix, book-row
snapshot guards, book indexing fallbacks, expiry/halts/PDT branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.event_sim import simulator
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
    for k, v in over.items():
        setattr(cfg.kill_switch, k, v) if k == "state" else setattr(cfg, k, v)
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


def _weights() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
            "security_id": ["A", "A"],
            "target_weight": [1.0, 0.0],
        }
    )


class TestSpecValidation:
    def test_fill_model(self) -> None:
        with pytest.raises(ValueError, match="fill_model"):
            EventSimSpec(fill_model="dark_pool")  # type: ignore[arg-type]

    def test_negative_latency_bars(self) -> None:
        with pytest.raises(ValueError, match="latency"):
            EventSimSpec(signal_to_order_bars=-1)
        with pytest.raises(ValueError, match="latency"):
            EventSimSpec(order_to_exchange_bars=-1)

    def test_negative_latency_timedelta(self) -> None:
        with pytest.raises(ValueError, match="signal_to_order"):
            EventSimSpec(signal_to_order=timedelta(seconds=-1))
        with pytest.raises(ValueError, match="order_to_exchange"):
            EventSimSpec(order_to_exchange=timedelta(seconds=-1))

    def test_vwap_window_and_l2_rest(self) -> None:
        with pytest.raises(ValueError, match="vwap window"):
            EventSimSpec(vwap_window_bars=0)
        with pytest.raises(ValueError, match="vwap window"):
            EventSimSpec(l2_rest_bars=-1)

    def test_min_notional_and_settlement(self) -> None:
        with pytest.raises(ValueError, match="min_notional"):
            EventSimSpec(min_notional=-1.0)
        with pytest.raises(ValueError, match="min_notional"):
            EventSimSpec(min_notional=float("nan"))
        with pytest.raises(ValueError, match="settlement_bars"):
            EventSimSpec(settlement_bars=-1)

    def test_pdt_gfv_modes(self) -> None:
        with pytest.raises(ValueError, match="pdt_mode"):
            EventSimSpec(pdt_mode="yolo")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="pdt_mode"):
            EventSimSpec(gfv_mode="yolo")  # type: ignore[arg-type]

    def test_fee_schedule_and_cost_knobs(self) -> None:
        with pytest.raises(ValueError, match="fee_schedule"):
            EventSimSpec(fee_schedule="free")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ac_eta"):
            EventSimSpec(ac_eta=-1.0)
        with pytest.raises(ValueError, match="taf_per_share"):
            EventSimSpec(taf_per_share=-1.0)
        with pytest.raises(ValueError, match="ac_tau"):
            EventSimSpec(ac_tau=0.0)
        with pytest.raises(ValueError, match="taf_min"):
            EventSimSpec(taf_min=10.0, taf_max=1.0)


class TestRunValidation:
    def test_initial_nav(self, tmp_path) -> None:
        bars = pl.DataFrame([_day("A", 1, 100.0)])
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError, match="initial_nav"):
                run_event_backtest(
                    bars, _weights(), _cfg(tmp_path), EventSimSpec(), initial_nav=bad
                )

    def test_close_auction_rejected(self, tmp_path) -> None:
        cfg = _cfg(tmp_path)
        cfg.execution.allow_close_auction = True
        bars = pl.DataFrame([_day("A", 1, 100.0)])
        with pytest.raises(ValueError, match="next_open"):
            run_event_backtest(bars, _weights(), cfg, EventSimSpec())


class TestSnapshotFromRow:
    def test_non_datetime_and_naive_time(self) -> None:
        assert simulator._snapshot_from_row({"event_time": "x", "security_id": "A"}, 4.0) is None
        row = _day("A", 1, 100.0)
        row["event_time"] = row["event_time"].replace(tzinfo=None)
        assert simulator._snapshot_from_row(row, 4.0) is None

    def test_book_columns_valid(self) -> None:
        row = _day(
            "A", 1, 100.0, best_bid=99.0, best_ask=101.0, top_bid_size=10.0, top_ask_size=5.0
        )
        snap = simulator._snapshot_from_row(row, 0.0)
        assert snap is not None and snap.depth == 1
        assert snap.bids[0].price == 99.0 and snap.asks[0].price == 101.0

    def test_book_columns_invalid_falls_back(self) -> None:
        row = _day(
            "A", 1, 100.0, best_bid=101.0, best_ask=99.0, top_bid_size=10.0, top_ask_size=5.0
        )
        snap = simulator._snapshot_from_row(row, 4.0)
        assert snap is not None  # crossed book falls back to close +/- half-spread
        assert snap.bids[0].price < snap.asks[0].price

    def test_book_columns_unparseable(self) -> None:
        row = _day("A", 1, 100.0, best_bid="x", best_ask=101.0, top_bid_size=10.0, top_ask_size=5.0)
        assert simulator._snapshot_from_row(row, 0.0) is None

    def test_missing_close_returns_none(self) -> None:
        row = {"security_id": "A", "event_time": datetime(2024, 1, 1, tzinfo=UTC)}
        assert simulator._snapshot_from_row(row, 4.0) is None


class TestLoadBooks:
    def test_indexed_from_book_columns(self) -> None:
        bars = pl.DataFrame(
            [
                _day(
                    "A",
                    1,
                    100.0,
                    best_bid=99.0,
                    best_ask=101.0,
                    top_bid_size=10.0,
                    top_ask_size=5.0,
                ),
            ]
        )
        indexed = simulator._load_books(
            bars,
            EventSimSpec(fill_model="l2_queue"),
            None,
            {
                datetime(2024, 1, 1, tzinfo=UTC): [
                    _day(
                        "A",
                        1,
                        100.0,
                        best_bid=99.0,
                        best_ask=101.0,
                        top_bid_size=10.0,
                        top_ask_size=5.0,
                    )
                ]
            },
        )
        when = datetime(2024, 1, 1, tzinfo=UTC)
        assert ("A", when) in indexed

    def test_non_l2_returns_empty(self) -> None:
        bars = pl.DataFrame([_day("A", 1, 100.0)])
        assert simulator._load_books(bars, EventSimSpec(fill_model="vwap"), None, {}) == {}

    def test_high_low_synthesizes(self) -> None:
        bars = pl.DataFrame([_day("A", 1, 100.0, high=101.0, low=99.0)])
        day_rows = {datetime(2024, 1, 1, tzinfo=UTC): [dict(r) for r in bars.iter_rows(named=True)]}
        indexed = simulator._load_books(
            bars, EventSimSpec(fill_model="l2_queue", seed=1), None, day_rows
        )
        assert indexed  # synthetic LOB from high/low envelope

    def test_close_only_fallback(self) -> None:
        row = _day("A", 1, 100.0)
        row.pop("high")
        row.pop("low")
        bars = pl.DataFrame([row])
        day_rows = {datetime(2024, 1, 1, tzinfo=UTC): [row]}
        indexed = simulator._load_books(bars, EventSimSpec(fill_model="l2_queue"), None, day_rows)
        assert ("A", datetime(2024, 1, 1, tzinfo=UTC)) in indexed

    def test_explicit_books_override(self) -> None:
        bars = pl.DataFrame([_day("A", 1, 100.0)])
        indexed = simulator._load_books(bars, EventSimSpec(), {}, {})
        assert indexed == {}


class TestRunEdges:
    def test_expired_after_sample_cancel(self, tmp_path) -> None:
        # Weight lands on the last bar: exchange scheduled but no fill bar.
        bars = pl.DataFrame([_day("A", d, 100.0) for d in range(1, 4)])
        weights = pl.DataFrame(
            {
                "event_time": [datetime(2024, 1, 3, tzinfo=UTC)],
                "security_id": ["A"],
                "target_weight": [1.0],
            }
        )
        sim = run_event_backtest(
            bars, weights, _cfg(tmp_path), EventSimSpec(), initial_nav=50_000.0
        )
        assert sim.result.metrics["live_pnl_claim"] is False

    def test_kill_switch_halts_orders(self, tmp_path) -> None:
        cfg = _cfg(tmp_path)
        cfg.kill_switch.state = "HALT"
        bars = pl.DataFrame([_day("A", d, 100.0) for d in range(1, 6)])
        sim = run_event_backtest(bars, _weights(), cfg, EventSimSpec(), initial_nav=50_000.0)
        # HALT state blocks every order attempt; the NAV stays flat at initial.
        assert sim.result.fills.height == 0
        navs = sim.result.equity["nav"].to_numpy()
        assert (navs == 50_000.0).all()

    def test_vwap_path_with_generated_books(self, tmp_path) -> None:
        bars = pl.DataFrame([_day("A", d, 100.0 + d) for d in range(1, 8)])
        sim = run_event_backtest(
            bars,
            _weights(),
            _cfg(tmp_path),
            EventSimSpec(fill_model="vwap", vwap_window_bars=2),
            initial_nav=100_000.0,
        )
        assert sim.result.metrics["execution_sim"] is True

    def test_l2_queue_with_book_columns_in_bars(self, tmp_path) -> None:
        rows = [
            _day(
                "A",
                d,
                100.0,
                best_bid=99.5,
                best_ask=100.5,
                top_bid_size=500.0,
                top_ask_size=500.0,
            )
            for d in range(1, 8)
        ]
        bars = pl.DataFrame(rows)
        sim = run_event_backtest(
            bars,
            _weights(),
            _cfg(tmp_path),
            EventSimSpec(fill_model="l2_queue", l2_rest_bars=2),
            initial_nav=100_000.0,
        )
        assert sim.result.fills.height >= 1

    def test_pdt_block_warning(self, tmp_path) -> None:
        bars = pl.DataFrame([_day("A", d, 100.0) for d in range(1, 8)])
        weights = pl.DataFrame(
            {
                "event_time": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 2, tzinfo=UTC),
                    datetime(2024, 1, 3, tzinfo=UTC),
                    datetime(2024, 1, 4, tzinfo=UTC),
                ],
                "security_id": ["A", "A", "A", "A"],
                "target_weight": [1.0, 0.0, 1.0, 0.0],
            }
        )
        sim = run_event_backtest(
            bars,
            weights,
            _cfg(tmp_path),
            EventSimSpec(pdt_mode="block", pdt_max_day_trades=0),
            initial_nav=20_000.0,
        )
        assert "execution_warnings" in sim.result.metrics
