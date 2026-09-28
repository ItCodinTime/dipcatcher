"""Coverage: event_sim simulator helpers — spec validation, delays, row
indexing, book snapshots, book loading branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest.event_sim.simulator import (
    EventSimSpec,
    _delay_bars,
    _index_rows,
    _legacy_costs,
    _load_books,
    _schedule_for,
    _snapshot_from_row,
)
from quant_fund.config.models import AppConfig


class TestSpecValidation:
    def test_defaults_ok(self) -> None:
        EventSimSpec()

    def test_bad_fill_model(self) -> None:
        with pytest.raises(ValueError, match="fill_model"):
            EventSimSpec(fill_model="magic")  # type: ignore[arg-type]

    def test_negative_latency_bars(self) -> None:
        with pytest.raises(ValueError, match="latency"):
            EventSimSpec(signal_to_order_bars=-1)
        with pytest.raises(ValueError, match="latency"):
            EventSimSpec(order_to_exchange_bars=-2)

    def test_negative_timedelta(self) -> None:
        with pytest.raises(ValueError, match="signal_to_order"):
            EventSimSpec(signal_to_order=timedelta(seconds=-1))
        with pytest.raises(ValueError, match="order_to_exchange"):
            EventSimSpec(order_to_exchange=timedelta(minutes=-5))

    def test_vwap_and_l2_bounds(self) -> None:
        with pytest.raises(ValueError, match="vwap window"):
            EventSimSpec(vwap_window_bars=0)
        with pytest.raises(ValueError, match="vwap window"):
            EventSimSpec(l2_rest_bars=-1)

    def test_min_notional(self) -> None:
        with pytest.raises(ValueError, match="min_notional"):
            EventSimSpec(min_notional=-0.5)
        with pytest.raises(ValueError, match="min_notional"):
            EventSimSpec(min_notional=float("inf"))

    def test_settlement_bars(self) -> None:
        with pytest.raises(ValueError, match="settlement"):
            EventSimSpec(settlement_bars=-1)

    def test_gate_modes(self) -> None:
        with pytest.raises(ValueError, match="pdt_mode and gfv_mode"):
            EventSimSpec(pdt_mode="lax")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="pdt_mode and gfv_mode"):
            EventSimSpec(gfv_mode="lax")  # type: ignore[arg-type]
        # valid alternates pass
        EventSimSpec(pdt_mode="block", gfv_mode="off")

    def test_fee_schedule(self) -> None:
        with pytest.raises(ValueError, match="fee_schedule"):
            EventSimSpec(fee_schedule="nyse")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "field",
        ["ac_eta", "ac_gamma", "sec_per_million", "taf_per_share", "taf_min", "taf_max"],
    )
    def test_negative_fee_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            EventSimSpec(**{field: -1.0})  # type: ignore[arg-type]

    def test_tau_positive(self) -> None:
        with pytest.raises(ValueError, match="ac_tau"):
            EventSimSpec(ac_tau=0.0)
        with pytest.raises(ValueError, match="ac_tau"):
            EventSimSpec(ac_tau=float("nan"))

    def test_taf_min_max_order(self) -> None:
        with pytest.raises(ValueError, match="taf_min cannot exceed"):
            EventSimSpec(taf_min=10.0, taf_max=1.0)


class TestDelayBars:
    def test_no_delay_returns_bars(self) -> None:
        dates = [datetime(2024, 1, i, tzinfo=UTC) for i in range(1, 10)]
        assert _delay_bars(dates, 0, None, 3) == 3

    def test_timedelta_counts_bars(self) -> None:
        dates = [datetime(2024, 1, 1, 10, 0, tzinfo=UTC) + timedelta(hours=i) for i in range(10)]
        # +3h from dates[0] -> bars while dates[0+step] < dates[0]+3h -> step=3
        assert _delay_bars(dates, 0, timedelta(hours=3), 0) == 3

    def test_timedelta_past_end(self) -> None:
        # delay never elapses within the calendar -> all remaining bars
        dates = [datetime(2024, 1, 1, tzinfo=UTC)]
        assert _delay_bars(dates, 0, timedelta(days=5), 0) == 1


def _bars(with_l2: bool = False, with_hl: bool = True) -> pl.DataFrame:
    times = [datetime(2024, 3, 1, 16, 0, tzinfo=UTC), datetime(2024, 3, 2, 16, 0, tzinfo=UTC)]
    data = {
        "security_id": ["AAA", "AAA", "BBB", "BBB"],
        "event_time": times * 2,
        "open": [100.0] * 4,
        "close": [101.0, 102.0, 50.0, 51.0],
        "close_total_return": [101.0, 102.0, 50.0, 51.0],
        "volume": [1e6] * 4,
        "source": ["synthetic"] * 4,
    }
    if with_hl:
        data["high"] = [103.0] * 4
        data["low"] = [99.0] * 4
    if with_l2:
        data["best_bid"] = [100.5, 101.5, 49.8, 50.8]
        data["best_ask"] = [100.7, 101.7, 50.0, 51.0]
        data["top_bid_size"] = [500.0] * 4
        data["top_ask_size"] = [400.0] * 4
    return pl.DataFrame(data)


class TestIndexRows:
    def test_groups_by_day(self) -> None:
        rows, dates, synthetic = _index_rows(_bars())
        assert len(dates) == 2
        assert synthetic is True
        assert all(len(rows[d]) == 2 for d in dates)
        # adv synthesized from close*volume when absent
        assert rows[dates[0]][0]["adv"] == pytest.approx(101.0 * 1e6)
        assert rows[dates[0]][0]["vol_20"] == pytest.approx(0.02)

    def test_uses_supplied_adv(self) -> None:
        bars = _bars().with_columns(pl.lit(123.0).alias("adv"))
        rows, dates, _ = _index_rows(bars)
        assert rows[dates[0]][0]["adv"] == pytest.approx(123.0)


class TestSnapshotFromRow:
    def test_naive_time_returns_none(self) -> None:
        row = {"event_time": datetime(2024, 1, 1), "security_id": "AAA", "close": 100.0}
        assert _snapshot_from_row(row, 4.0) is None

    def test_l2_fields_build_snapshot(self) -> None:
        rows, dates, _ = _index_rows(_bars(with_l2=True))
        snap = _snapshot_from_row(rows[dates[0]][0], 4.0)
        assert snap is not None
        assert snap.source == "synthetic"
        assert snap.bids[0].price == pytest.approx(100.5)
        assert snap.asks[0].price == pytest.approx(100.7)

    def test_invalid_l2_falls_back_to_close(self) -> None:
        row = {
            "event_time": datetime(2024, 1, 1, 16, 0, tzinfo=UTC),
            "security_id": "AAA",
            "close": 100.0,
            "volume": 1000.0,
            "best_bid": "bad",
            "best_ask": 101.0,
            "top_bid_size": 1.0,
            "top_ask_size": 1.0,
            "source": "synthetic",
        }
        # unfloatable L2 field -> the l2 branch itself is rejected
        assert _snapshot_from_row(row, 4.0) is None

    def test_crossed_l2_falls_back(self) -> None:
        row = {
            "event_time": datetime(2024, 1, 1, 16, 0, tzinfo=UTC),
            "security_id": "AAA",
            "close": 100.0,
            "volume": 1000.0,
            "best_bid": 101.0,
            "best_ask": 100.0,
            "top_bid_size": 1.0,
            "top_ask_size": 1.0,
        }
        snap = _snapshot_from_row(row, 4.0)
        assert snap is not None
        assert snap.bids[0].price < snap.asks[0].price

    def test_no_close_returns_none(self) -> None:
        row = {
            "event_time": datetime(2024, 1, 1, 16, 0, tzinfo=UTC),
            "security_id": "AAA",
            "close": None,
        }
        assert _snapshot_from_row(row, 4.0) is None


class TestLoadBooks:
    def test_supplied_books_passthrough(self) -> None:
        rows, dates, _ = _index_rows(_bars())
        snap = _snapshot_from_row(rows[dates[0]][0], 4.0)
        assert snap is not None
        books = {("AAA", snap.event_time): snap}
        out = _load_books(_bars(), EventSimSpec(), books, rows)
        assert out == books

    def test_non_l2_returns_empty(self) -> None:
        rows, _, _ = _index_rows(_bars(with_l2=True))
        out = _load_books(_bars(with_l2=True), EventSimSpec(fill_model="next_open"), None, rows)
        assert out == {}

    def test_l2_uses_feature_cols(self) -> None:
        bars = _bars(with_l2=True)
        rows, _, _ = _index_rows(bars)
        out = _load_books(bars, EventSimSpec(fill_model="l2_queue"), None, rows)
        assert len(out) == 4
        assert {k[0] for k in out} == {"AAA", "BBB"}

    def test_l2_synthesizes_from_hl(self) -> None:
        bars = _bars(with_hl=True)
        rows, _, _ = _index_rows(bars)
        out = _load_books(bars, EventSimSpec(fill_model="l2_queue", seed=3), None, rows)
        assert len(out) == 4
        for snap in out.values():
            assert snap.bids[0].price < snap.asks[0].price

    def test_l2_falls_back_to_close_books(self) -> None:
        bars = _bars(with_hl=False)
        rows, _, _ = _index_rows(bars)
        out = _load_books(bars, EventSimSpec(fill_model="l2_queue"), None, rows)
        assert len(out) == 4
        for snap in out.values():
            assert snap.source == "synthetic"


class TestSchedules:
    def test_legacy_costs(self) -> None:
        assert _legacy_costs(EventSimSpec())
        assert not _legacy_costs(EventSimSpec(fee_schedule="alpaca"))
        assert not _legacy_costs(EventSimSpec(ac_eta=1.0))
        assert not _legacy_costs(EventSimSpec(spread_from_book=True))

    def test_schedule_for_bps(self) -> None:
        sched = _schedule_for(EventSimSpec(), AppConfig())
        assert sched.name == "bps"
        assert sched.commission_bps == pytest.approx(1.0)

    def test_schedule_for_alpaca(self) -> None:
        sched = _schedule_for(EventSimSpec(fee_schedule="alpaca"), AppConfig())
        assert sched.name.startswith("alpaca") or "alpaca" in sched.name.lower()
