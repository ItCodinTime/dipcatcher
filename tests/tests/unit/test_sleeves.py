"""Signal sleeve tests: causality, sign conventions, netting, caps."""

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


def _wiggled(n: int, base: float = 100.0) -> list[float]:
    return [base + 0.3 * np.sin(i) for i in range(n)]  # nonzero vol, no trend


def test_funding_carry_longs_negative_funding() -> None:
    bars = _panel({"A": _wiggled(30), "B": _wiggled(30)})
    fund = pl.DataFrame(
        {
            "security_id": ["A", "B", "A", "B"],
            "event_time": [
                T0 + timedelta(hours=2),
                T0 + timedelta(hours=2),
                T0 + timedelta(hours=10),
                T0 + timedelta(hours=10),
            ],
            "value": [-0.01, 0.01, -0.01, 0.01],
        }
    )
    w = funding_carry_weights(bars, fund, lookback_events=2, vol_window=4)
    assert w.height > 0
    late = w.filter(pl.col("event_time") > T0 + timedelta(hours=10))
    a_w = late.filter(pl.col("security_id") == "A")["target_weight"]
    b_w = late.filter(pl.col("security_id") == "B")["target_weight"]
    assert (a_w > 0).all()  # negative funding → long
    assert (b_w < 0).all()  # positive funding → short


def test_funding_carry_ignores_future_events() -> None:
    bars = _panel({"A": _wiggled(10)})
    # funding event only at bar 8 → no weights before bar 8+1
    fund = pl.DataFrame(
        {"security_id": ["A"], "event_time": [T0 + timedelta(hours=8)], "value": [-0.01]}
    )
    w = funding_carry_weights(bars, fund, vol_window=3)
    if w.height:
        assert w["event_time"].min() >= T0 + timedelta(hours=8)


def test_momentum_longs_winners_shorts_losers() -> None:
    n = 60
    up = [100.0 * (1.002**i) for i in range(n)]
    down = [100.0 * (0.998**i) for i in range(n)]
    flat = [100.0] * n
    bars = _panel({"UP": up, "DOWN": down, "FLAT": flat})
    w = cross_sectional_momentum_weights(bars, lookback_bars=20, skip_bars=2, vol_window=10)
    last_t = w["event_time"].max()
    last = w.filter(pl.col("event_time") == last_t)
    assert last.filter(pl.col("security_id") == "UP")["target_weight"][0] > 0
    assert last.filter(pl.col("security_id") == "DOWN")["target_weight"][0] < 0


def test_sweep_reclaim_emits_after_reclaim_bar() -> None:
    # Flat-ish bars with prior_low = 99.95; bar 25 wicks to 99.0 (sweep) and
    # closes back at 100.0 (reclaim) → long pulse starting strictly after.
    rows = [
        {
            "security_id": "A",
            "event_time": T0 + timedelta(hours=i),
            "open": 100.0,
            "high": 100.1,
            "low": 99.95,
            "close": 100.0,
            "volume": 1e5,
            "source": "synthetic",
        }
        for i in range(30)
    ]
    rows[25] = {**rows[25], "low": 99.0, "close": 100.0, "high": 100.05}
    bars3 = pl.DataFrame(rows)
    w3 = sweep_reclaim_weights(bars3, lookback=20, hold_bars=4, decay=1.0)
    assert w3.height > 0
    assert (w3["target_weight"] > 0).all()  # low-sweep reclaim → long
    # weight stamped at the signal bar; the engine fills it at the NEXT open
    assert w3["event_time"].min() >= T0 + timedelta(hours=25)

    # Follow-through (close stays below prior_low) emits nothing.
    rows[25]["close"] = 99.5
    w_follow = sweep_reclaim_weights(pl.DataFrame(rows), lookback=20, hold_bars=4)
    assert w_follow.height == 0


def test_slow_trend_sign() -> None:
    up = [100.0 + 0.5 * i for i in range(120)]
    down = [150.0 - 0.5 * i for i in range(120)]
    bars = _panel({"UP": up, "DOWN": down})
    w = slow_trend_weights(bars, fast_bars=10, slow_bars=30, vol_window=10)
    last_t = w["event_time"].max()
    last = w.filter(pl.col("event_time") == last_t)
    assert last.filter(pl.col("security_id") == "UP")["target_weight"][0] > 0
    assert last.filter(pl.col("security_id") == "DOWN")["target_weight"][0] < 0


def test_blend_nets_duplicate_weights() -> None:
    t = T0 + timedelta(hours=1)
    s1 = pl.DataFrame({"event_time": [t], "security_id": ["A"], "target_weight": [0.05]})
    s2 = pl.DataFrame({"event_time": [t], "security_id": ["A"], "target_weight": [-0.03]})
    out = blend_weights({"x": s1, "y": s2}, {"x": 1.0, "y": 1.0})
    assert out.height == 1
    assert out["target_weight"][0] == pytest.approx(0.02)


def test_funding_spike_fade_only_fires_on_extremes() -> None:
    bars = _panel({"A": _wiggled(80), "B": _wiggled(80)})
    # A: flat funding history then one extreme positive spike; B: flat
    frows = []
    for i in range(0, 60, 4):
        frows.append({"security_id": "A", "event_time": T0 + timedelta(hours=i), "value": 0.0001})
        frows.append({"security_id": "B", "event_time": T0 + timedelta(hours=i), "value": 0.0001})
    frows.append({"security_id": "A", "event_time": T0 + timedelta(hours=60), "value": 0.02})
    fund = pl.DataFrame(frows)
    w = funding_spike_fade_weights(bars, fund, lookback_events=15, z_threshold=2.0, vol_window=8)
    late = w.filter(pl.col("event_time") > T0 + timedelta(hours=60))
    assert late.height > 0
    a_rows = late.filter(pl.col("security_id") == "A")
    assert (a_rows["target_weight"] < 0).all()  # extreme positive funding → short
    assert late.filter(pl.col("security_id") == "B").height == 0  # no spike → no trade


def test_gross_scale_caps_book() -> None:
    bars = _panel({s: _wiggled(30) for s in ("A", "B", "C", "D")})
    fund = pl.DataFrame(
        {
            "security_id": ["A", "B", "C", "D"],
            "event_time": [T0 + timedelta(hours=2)] * 4,
            "value": [-0.01, -0.01, -0.01, -0.01],
        }
    )
    w = funding_carry_weights(bars, fund, gross_scale=0.1, vol_window=4)
    gross = w.group_by("event_time").agg(pl.col("target_weight").abs().sum())
    assert (gross["target_weight"] <= 0.1 + 1e-9).all()


def test_basis_carry_positive_rates_only() -> None:
    bars = _panel({"A": _wiggled(30), "B": _wiggled(30), "C": _wiggled(30)})
    fund = pl.DataFrame(
        {
            "security_id": ["A", "B", "C", "A", "B", "C"],
            "event_time": [T0 + timedelta(hours=2)] * 3 + [T0 + timedelta(hours=10)] * 3,
            "value": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],  # C negative → ineligible
        }
    )
    w = basis_carry_weights(bars, fund, lookback_events=2, gross_scale=0.9, max_name=1.0)
    assert w.height > 0
    late = w.filter(pl.col("event_time") > T0 + timedelta(hours=10))
    assert late.height > 0
    assert (late["target_weight"] >= 0).all()  # pair book is long-only weights
    assert late.filter(pl.col("security_id") == "C").height == 0
    # A's rate (0.02) is twice B's (0.01) → A gets ~2x B's weight
    wa = late.filter(pl.col("security_id") == "A")["target_weight"][0]
    wb = late.filter(pl.col("security_id") == "B")["target_weight"][0]
    assert wa == pytest.approx(2 * wb, rel=0.05)
    gross = late.group_by("event_time").agg(pl.col("target_weight").sum())
    assert (gross["target_weight"] <= 0.9 + 1e-9).all()


def test_basis_carry_empty_when_no_positive_funding() -> None:
    bars = _panel({"A": _wiggled(30)})
    fund = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [T0 + timedelta(hours=2)],
            "value": [-0.02],
        }
    )
    w = basis_carry_weights(bars, fund)
    assert w.height == 0


def test_basis_carry_hysteresis_event_driven() -> None:
    bars = _panel({"A": _wiggled(40), "B": _wiggled(40)})
    fund = pl.DataFrame(
        {
            "security_id": ["A", "B", "A", "B", "A", "B"],
            "event_time": [
                T0 + timedelta(hours=2),
                T0 + timedelta(hours=2),
                T0 + timedelta(hours=8),
                T0 + timedelta(hours=8),
                T0 + timedelta(hours=30),
                T0 + timedelta(hours=30),
            ],
            "value": [0.01, 0.0004, 0.01, 0.0004, -0.02, 0.0004],
        }
    )
    w = basis_carry_hysteresis_weights(
        bars, fund, enter_rate=0.0003, exit_rate=0.0, lookback_events=2, name_weight=0.1
    )
    # A enters early (rate 0.01 ≥ enter), B qualifies too (0.0004 ≥ enter)
    assert w.height > 0
    entries = w.filter(pl.col("target_weight") > 0)
    exits = w.filter(pl.col("target_weight") == 0)
    assert entries.height >= 2
    a_exit = exits.filter(pl.col("security_id") == "A")
    assert a_exit.height >= 1  # A's rate goes negative at t=30 → exit row
    # sparse: no rows emitted while membership is static
    times_with_rows = w["event_time"].n_unique()
    assert times_with_rows < 10


def test_basis_carry_hysteresis_rebalance_band_resizes() -> None:
    """A held pair whose mark drifts past the band re-emits name_weight."""
    bars = _panel({"A": [100.0] * 10 + [160.0] * 10, "B": _wiggled(20)})
    fund = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [T0 + timedelta(hours=2), T0 + timedelta(hours=6)],
            "value": [0.01, 0.01],
        }
    )
    w = basis_carry_hysteresis_weights(
        bars,
        fund,
        enter_rate=0.0003,
        exit_rate=0.0,
        lookback_events=1,
        name_weight=0.08,
        rebalance_band=1.5,
    )
    a_rows = w.filter(pl.col("security_id") == "A").sort("event_time")
    # entry at first qualifying bar + one resize once drift hits 1.6 > 1.5
    assert a_rows.height == 2
    assert (a_rows["target_weight"] == 0.08).all()
    resize_time = a_rows["event_time"][1]
    assert resize_time == T0 + timedelta(hours=10)  # first 160 close
    # without a band, only the entry exists
    w_flat = basis_carry_hysteresis_weights(
        bars, fund, enter_rate=0.0003, exit_rate=0.0, lookback_events=1, name_weight=0.08
    )
    assert w_flat.filter(pl.col("security_id") == "A").height == 1
