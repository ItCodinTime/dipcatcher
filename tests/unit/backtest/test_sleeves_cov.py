"""Coverage tests for backtest/sleeves.py: hysteresis membership options
(prefix entry bars, rate/vol sizing, contested slots, drift edges), the
fail-closed validation guards, and blend_weights multiplier edges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.sleeves import (
    basis_carry_hysteresis_weights,
    basis_carry_weights,
    blend_weights,
    cross_sectional_momentum_weights,
    funding_carry_weights,
    funding_spike_fade_weights,
    slow_trend_weights,
    sweep_reclaim_weights,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _panel(prices: dict[str, list[float]], step: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    rows = []
    for sid, px in prices.items():
        for i, p in enumerate(px):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + i * step,
                    "open": p,
                    "high": p * 1.001,
                    "low": p * 0.999,
                    "close": p,
                    "volume": 1e5,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _funding(rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    """(sid, hour_offset, rate) tuples → a funding frame."""
    return pl.DataFrame(
        {
            "security_id": [r[0] for r in rows],
            "event_time": [T0 + timedelta(hours=r[1]) for r in rows],
            "value": [r[2] for r in rows],
        }
    )


def _hyst(bars: pl.DataFrame, fund: pl.DataFrame, **kw: object) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "enter_rate": 0.0003,
        "exit_rate": 0.0,
        "lookback_events": 1,
        "name_weight": 0.08,
    }
    return basis_carry_hysteresis_weights(bars, fund, **(defaults | kw))


# --- enter_rate_by_prefix ---------------------------------------------------


def test_prefix_bar_blocks_entry_below_venue_threshold() -> None:
    """A prefixed sid must clear its own bar; unprefixed sids use the global."""
    bars = _panel({"DYDX:BTC": [100.0] * 20, "BIN:ETH": [100.0] * 20})
    fund = _funding([("DYDX:BTC", 2, 0.00045), ("BIN:ETH", 2, 0.00045)])
    w = _hyst(bars, fund, enter_rate_by_prefix={"DYDX:": 0.0006})
    # 0.00045 clears the global 0.0003 bar but not the venue's 0.0006
    bin_rows = w.filter(pl.col("security_id") == "BIN:ETH")
    assert bin_rows.height == 1
    assert bin_rows["event_time"][0] == T0 + timedelta(hours=2)
    assert bin_rows["target_weight"][0] == pytest.approx(0.08)
    assert w.filter(pl.col("security_id") == "DYDX:BTC").height == 0


def test_prefix_first_match_in_dict_order_wins() -> None:
    """The first matching prefix sets the bar — later (even narrower) ones are dead."""
    bars = _panel({"DYDX:BTC": [100.0] * 20})
    fund = _funding([("DYDX:BTC", 2, 0.0005)])
    broad_first = _hyst(bars, fund, enter_rate_by_prefix={"DYDX:": 0.0006, "DYDX:BTC": 0.0004})
    assert broad_first.height == 0  # "DYDX:" matched first → bar is 0.0006
    specific_first = _hyst(bars, fund, enter_rate_by_prefix={"DYDX:BTC": 0.0004, "DYDX:": 0.0006})
    assert specific_first.height == 1  # "DYDX:BTC" matched first → bar is 0.0004


def test_prefix_bar_can_lower_the_entry_threshold() -> None:
    """Prefix bars override in both directions — a lower bar admits sooner."""
    bars = _panel({"DYDX:BTC": [100.0] * 20, "BIN:ETH": [100.0] * 20})
    fund = _funding([("DYDX:BTC", 2, 0.0002), ("BIN:ETH", 2, 0.0002)])
    w = _hyst(bars, fund, enter_rate_by_prefix={"DYDX:": 0.0001})
    assert w.filter(pl.col("security_id") == "DYDX:BTC").height == 1
    assert w.filter(pl.col("security_id") == "BIN:ETH").height == 0  # below global bar


def test_nonmatching_prefixes_leave_global_bar() -> None:
    bars = _panel({"DYDX:BTC": [100.0] * 20, "BIN:ETH": [100.0] * 20})
    fund = _funding([("DYDX:BTC", 2, 0.0005), ("BIN:ETH", 2, 0.0005)])
    w = _hyst(bars, fund, enter_rate_by_prefix={"ZZZ:": 0.9})
    assert w.height == 2


def test_prefixed_sid_still_exits_on_global_exit_rate() -> None:
    """Entry is venue-scoped, exit is global: after entering, a dip under the
    prefixed bar (but above exit) holds silently; below exit it leaves."""
    bars = _panel({"DYDX:BTC": [100.0] * 20})
    fund = _funding(
        [
            ("DYDX:BTC", 2, 0.0007),  # enters (clears 0.0006)
            ("DYDX:BTC", 10, 0.0004),  # under its bar, above exit → held, no row
            ("DYDX:BTC", 15, -0.01),  # below global exit → exit row
        ]
    )
    w = _hyst(bars, fund, enter_rate_by_prefix={"DYDX:": 0.0006})
    rows = w.sort("event_time").rows(named=True)
    assert len(rows) == 2
    assert rows[0]["event_time"] == T0 + timedelta(hours=2)
    assert rows[0]["target_weight"] == pytest.approx(0.08)
    assert rows[1]["event_time"] == T0 + timedelta(hours=15)
    assert rows[1]["target_weight"] == 0.0


# --- basis_carry_hysteresis_weights membership semantics --------------------


def test_hysteresis_band_holds_then_reentry_needs_enter() -> None:
    """A rate in [exit, enter) holds a member but won't admit a new one."""
    bars = _panel({"A": [100.0] * 30})
    fund = _funding(
        [
            ("A", 2, 0.01),  # enters
            ("A", 10, 0.0002),  # inside the band → held, no row
            ("A", 15, -0.01),  # exits
            ("A", 20, 0.0002),  # back inside the band → NOT re-entered
            ("A", 25, 0.01),  # clears enter again → re-enters
        ]
    )
    w = _hyst(bars, fund)
    rows = w.sort("event_time").rows(named=True)
    assert [r["event_time"] for r in rows] == [
        T0 + timedelta(hours=2),
        T0 + timedelta(hours=15),
        T0 + timedelta(hours=25),
    ]
    assert [r["target_weight"] for r in rows] == pytest.approx([0.08, 0.0, 0.08])


def test_member_exits_when_its_bars_leave_the_panel() -> None:
    """A held sid whose bars stop (delisted) exits at the first missing stamp."""
    bars = _panel({"A": [100.0] * 11, "B": [100.0] * 20})
    fund = _funding([("A", 2, 0.01)])  # B has no funding → never a candidate
    w = _hyst(bars, fund)
    a_rows = w.filter(pl.col("security_id") == "A").sort("event_time")
    assert a_rows["target_weight"].to_list() == pytest.approx([0.08, 0.0])
    assert a_rows["event_time"][1] == T0 + timedelta(hours=11)


def test_max_names_contested_slots_go_to_highest_rates() -> None:
    bars = _panel({s: [100.0] * 10 for s in ("A", "B", "C")})
    fund = _funding([("A", 2, 0.005), ("B", 2, 0.003), ("C", 2, 0.001)])
    w = _hyst(bars, fund, max_names=2)
    assert set(w["security_id"]) == {"A", "B"}  # C's lower rate loses the slot


def test_rate_scale_ref_sizes_the_book_off_mean_rate() -> None:
    """book mean / rate_scale_ref scales every emitted row, floored and capped."""
    bars = _panel({"A": [100.0] * 10, "B": [100.0] * 10})
    fund = _funding([("A", 2, 0.002), ("B", 2, 0.001)])
    rich = _hyst(bars, fund, rate_scale_ref=0.001)
    # mean rate 0.0015 / ref 0.001 → scale 1.5 → 0.08 * 1.5
    assert rich["target_weight"].to_list() == pytest.approx([0.12, 0.12])
    thin = _hyst(bars, fund, rate_scale_ref=0.02)
    # 0.0015 / 0.02 = 0.075 → floored at 0.3 → 0.08 * 0.3
    assert thin["target_weight"].to_list() == pytest.approx([0.024, 0.024])


def test_rate_exponent_tilts_weights_to_higher_payers() -> None:
    """rate^exponent tilts share toward payers while gross stays 2×name_weight."""
    bars = _panel({"A": [100.0] * 10, "B": [100.0] * 10})
    fund = _funding([("A", 2, 0.001), ("B", 2, 0.002)])
    w = _hyst(bars, fund, rate_exponent=1.0).sort("security_id")
    wa, wb = w["target_weight"].to_list()
    assert wb == pytest.approx(2 * wa)  # 2× the rate → 2× the share
    assert wa + wb == pytest.approx(0.16)


def test_vol_lookback_dilutes_wild_names() -> None:
    """min(1, vol_ref / vol_i) shrinks the wild name, leaves the calm one whole."""
    wild = [100.0 if i % 2 == 0 else 105.0 for i in range(30)]
    calm = [100.0 + 0.05 * i for i in range(30)]
    bars = _panel({"WILD": wild, "CALM": calm})
    fund = _funding([("WILD", 15, 0.01), ("CALM", 15, 0.01)])
    w = _hyst(bars, fund, vol_lookback=10, vol_ref=0.01)
    w_calm = w.filter(pl.col("security_id") == "CALM")["target_weight"][0]
    w_wild = w.filter(pl.col("security_id") == "WILD")["target_weight"][0]
    assert w_calm == pytest.approx(0.08)  # ~flat series → vol ≈ 0 → full size
    assert w_wild < 0.5 * w_calm  # ±5% alternation → vol ≈ 0.05 ≫ vol_ref


def test_entry_without_price_skips_drift_resize() -> None:
    """close <= 0 at the entry bar leaves entry_px unset → drift resizes skipped."""
    px = [100.0] * 20
    px[2] = 0.0  # zero close on the entry bar
    px[10:] = [200.0] * 10  # later doubles — would breach band=1.5 if priced in
    bars = _panel({"A": px})
    fund = _funding([("A", 2, 0.01)])
    w = _hyst(bars, fund, rebalance_band=1.5)
    assert w.height == 1  # entry only; no resize rows without an entry price


# --- fail-closed guards -----------------------------------------------------


def test_bars_missing_columns_fail_closed() -> None:
    bad = pl.DataFrame({"security_id": ["A"], "event_time": [T0]})
    fund = _funding([("A", 2, 0.01)])
    with pytest.raises(ValueError, match="missing columns"):
        funding_carry_weights(bad, fund)


@pytest.mark.parametrize(
    "fn",
    [
        funding_carry_weights,
        funding_spike_fade_weights,
        basis_carry_weights,
        basis_carry_hysteresis_weights,
    ],
)
def test_empty_funding_fails_closed(fn) -> None:
    bars = _panel({"A": [100.0] * 5})
    empty = pl.DataFrame({"security_id": [], "event_time": [], "value": []})
    with pytest.raises(ValueError, match="non-empty"):
        fn(bars, empty)


@pytest.mark.parametrize("fn", [funding_carry_weights, basis_carry_weights])
def test_funding_without_value_column_fails_closed(fn) -> None:
    bars = _panel({"A": [100.0] * 5})
    bad = _funding([("A", 2, 0.01)]).rename({"value": "rate"})
    with pytest.raises(ValueError, match="value"):
        fn(bars, bad)


def test_momentum_requires_lookback_beyond_skip() -> None:
    bars = _panel({"A": [100.0] * 10})
    with pytest.raises(ValueError, match="lookback_bars"):
        cross_sectional_momentum_weights(bars, lookback_bars=4, skip_bars=4)


def test_sweep_reclaim_param_guards() -> None:
    bars = _panel({"A": [100.0] * 10})
    with pytest.raises(ValueError, match="hold_bars"):
        sweep_reclaim_weights(bars, hold_bars=0)
    with pytest.raises(ValueError, match="decay"):
        sweep_reclaim_weights(bars, decay=0.0)
    with pytest.raises(ValueError, match="decay"):
        sweep_reclaim_weights(bars, decay=1.5)


def test_slow_trend_requires_fast_below_slow() -> None:
    bars = _panel({"A": [100.0] * 10})
    with pytest.raises(ValueError, match="fast_bars"):
        slow_trend_weights(bars, fast_bars=30, slow_bars=30)


@pytest.mark.parametrize("band", [1.0, 0.9, 0.0, np.nan, np.inf])
def test_hysteresis_rebalance_band_must_exceed_one(band: float) -> None:
    bars = _panel({"A": [100.0] * 5})
    fund = _funding([("A", 2, 0.01)])
    with pytest.raises(ValueError, match="rebalance_band"):
        _hyst(bars, fund, rebalance_band=band)


# --- blend_weights edges ----------------------------------------------------


def _sleeve_frame(w: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + timedelta(hours=1)],
            "security_id": ["A"],
            "target_weight": [w],
        }
    )


@pytest.mark.parametrize("mult", [np.inf, -np.inf, np.nan])
def test_blend_rejects_nonfinite_multipliers(mult: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        blend_weights({"x": _sleeve_frame(0.05)}, {"x": mult})


def test_blend_zero_and_missing_multipliers_emit_empty_typed_frame() -> None:
    # x skipped by explicit zero; y missing → defaults to 0.0 → also skipped
    out = blend_weights({"x": _sleeve_frame(0.05), "y": _sleeve_frame(0.05)}, {"x": 0.0})
    assert out.height == 0
    assert dict(out.schema) == {
        "event_time": pl.Datetime,
        "security_id": pl.String,
        "target_weight": pl.Float64,
    }
    assert blend_weights({}, {}).height == 0


def test_blend_applies_named_multipliers() -> None:
    out = blend_weights({"x": _sleeve_frame(0.05)}, {"x": 2.0})
    assert out["target_weight"][0] == pytest.approx(0.10)
