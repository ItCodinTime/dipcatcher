"""Known-answer tests for the five strategy signal builders.

Each test builds a tiny synthetic panel where the signal is computable by
hand, plus a no-lookahead "shift test": truncating the panel must not change
any signal value at or before the cut date.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.research.replication.panel import make_panel, market_returns
from quant_fund.research.replication.signals import (
    bab_signal,
    low_volatility_signal,
    overnight_signal,
    reversal_signal,
    tsmom_signal,
    tug_of_war_signal,
    turn_of_month_mask,
    turn_of_month_signal,
)


def _panel_from_returns(returns: dict[str, list[float]], start: str = "2024-01-02") -> pl.DataFrame:
    """Build a panel where each security realizes a given daily ret_cc path.

    Overnight leg is set to zero (open == prev close) so ret_cc == ret_id.
    """
    frames = []
    import pandas as pd

    for sec, rets in returns.items():
        n = len(rets)
        days = pd.bdate_range(start=start, periods=n)
        close = 100.0 * np.cumprod(1.0 + np.asarray(rets))
        open_ = np.concatenate([[close[0] / (1 + rets[0])], close[:-1]])
        frames.append(
            pl.DataFrame(
                {
                    "security_id": sec,
                    "symbol": sec,
                    "event_time": days.to_numpy(),
                    "open": open_,
                    "close": close,
                }
            )
        )
    return make_panel(pl.concat(frames))


def _signals_by_day(sig: pl.DataFrame, sec: str) -> np.ndarray:
    return sig.filter(pl.col("security_id") == sec).sort("event_time")["signal"].to_numpy()


# ---------- known answers ----------


def test_tsmom_sign_and_vol_scaling() -> None:
    # Security A: +1% every day for 6 days (up), B: -1% every day (down).
    panel = _panel_from_returns({"A": [0.01] * 6, "B": [-0.01] * 6})
    sig = tsmom_signal(panel, lookback=3, vol_com=2)
    a, b = _signals_by_day(sig, "A"), _signals_by_day(sig, "B")
    # ret_cc[0] is null, so lookback-3 windows stay null through idx=2
    assert np.isnan(a[:3]).all() and np.isnan(b[:3]).all()  # warmup nulls
    assert (a[3:] > 0).all() and (b[3:] < 0).all()
    # A/B have identical constant vol -> symmetric signal magnitude
    assert a[3] == pytest.approx(-b[3], rel=1e-6)


def test_reversal_negates_past_return() -> None:
    panel = _panel_from_returns({"A": [0.02, -0.01, 0.03, -0.02, 0.01]})
    sig = reversal_signal(panel, lookback=2)
    s = _signals_by_day(sig, "A")
    # ret_cc[0] is null so windows covering row 0 stay null: first valid
    # signal is idx=lookback covering returns rows 1..lookback.
    assert np.isnan(s[0]) and np.isnan(s[1])
    # day idx2: -(log(0.99)+log(1.03))
    assert s[2] == pytest.approx(-(np.log(0.99) + np.log(1.03)))
    assert s[4] == pytest.approx(-(np.log(0.98) + np.log(1.01)))


def test_lowvol_prefers_low_vol() -> None:
    rng_a = [0.03, -0.03, 0.03, -0.03, 0.03, -0.03]  # high vol
    rng_b = [0.001, -0.001, 0.001, -0.001, 0.001, -0.001]  # low vol
    panel = _panel_from_returns({"A": rng_a, "B": rng_b})
    sig = low_volatility_signal(panel, lookback=4)
    a, b = _signals_by_day(sig, "A"), _signals_by_day(sig, "B")
    # ret_cc[0] null -> first valid window at idx=lookback covering rows 1..4
    # low-vol asset B must have strictly higher (less negative) signal
    assert (b[4:] > a[4:]).all()
    assert a[4] == pytest.approx(-np.std(rng_a[1:5], ddof=1), rel=1e-6)


def test_bab_signal_low_beta_wins() -> None:
    n = 40
    rng = np.random.default_rng(5)
    mkt = rng.normal(0.0, 0.01, n)
    hi = 0.005 + 1.6 * mkt + rng.normal(0.0, 0.005, n)  # beta ~1.6
    lo = 0.006 + 0.4 * mkt + rng.normal(0.0, 0.005, n)  # beta ~0.4
    panel = _panel_from_returns({"HI": hi.tolist(), "LO": lo.tolist()})
    market = market_returns(panel)
    sig = bab_signal(panel, market, lookback=20)
    hi_s = _signals_by_day(sig, "HI")[-1]
    lo_s = _signals_by_day(sig, "LO")[-1]
    # signal = -beta_ts -> low-beta asset must rank higher
    assert lo_s > hi_s
    # shrunk beta_ts lives between 0 and 2 -> signal in (-2, 0)
    assert -2.0 < hi_s < 0.0


def test_overnight_and_tug_of_war() -> None:
    # open gaps: craft returns where overnight leg is nonzero.
    frames = []
    import pandas as pd

    days = pd.bdate_range(start="2024-01-02", periods=5).to_numpy()
    # A: opens +2% each day then closes flat (pure overnight return)
    open_ = np.array([100.0, 102.0, 104.04, 106.1208, 108.243216])
    close = open_ * np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    frames.append(
        pl.DataFrame(
            {"security_id": "A", "symbol": "A", "event_time": days, "open": open_, "close": close}
        )
    )
    panel = make_panel(pl.concat(frames))
    sig = overnight_signal(panel, lookback=2)
    s = _signals_by_day(sig, "A")
    # day2 signal = log(1.02) + log(1.02) (days1-2 overnight legs)
    assert s[2] == pytest.approx(2 * np.log(1.02), rel=1e-6)
    tow = tug_of_war_signal(panel, lookback=2)
    t = _signals_by_day(tow, "A")
    # intraday legs are ~0 -> tug == overnight cum
    assert t[2] == pytest.approx(2 * np.log(1.02), rel=1e-6)


def test_turn_of_month_mask_known_days() -> None:
    import pandas as pd

    days = pd.bdate_range("2024-01-29", "2024-02-06").to_numpy()
    # 2024-01-31 = last trading day of Jan; Feb 1,2,5 = first 3 of Feb
    mask = turn_of_month_mask(days)
    idx = pd.DatetimeIndex(days)
    flagged = set(idx[mask].strftime("%Y-%m-%d"))
    assert flagged == {"2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05"}
    assert mask.sum() == 4


def test_turn_of_month_signal_broadcast() -> None:
    panel = _panel_from_returns({"A": [0.01] * 8, "B": [0.005] * 8}, start="2024-01-29")
    sig = turn_of_month_signal(panel)
    # same flag for both assets on each date
    piv = sig.sort("event_time").group_by("event_time").agg(pl.col("signal").n_unique())
    assert (piv["signal"] == 1).all()


# ---------- no-lookahead shift tests ----------


@pytest.mark.parametrize(
    "builder",
    [
        lambda f: tsmom_signal(f, lookback=4, vol_com=3),
        lambda f: reversal_signal(f, lookback=3),
        lambda f: low_volatility_signal(f, lookback=4),
        lambda f: overnight_signal(f, lookback=2),
        lambda f: tug_of_war_signal(f, lookback=2),
    ],
)
def test_signal_no_lookahead_shift(builder) -> None:
    rng = np.random.default_rng(42)
    returns = {s: rng.normal(0.0, 0.02, 30).tolist() for s in ("A", "B", "C")}
    full = _panel_from_returns(returns)
    # prefix truncation at a fixed date — nothing after 2024-01-26 survives
    truncated = full.filter(pl.col("event_time") <= pl.datetime(2024, 1, 26))
    sig_full = builder(full)
    sig_trunc = builder(truncated)
    joined = sig_full.join(sig_trunc, on=["security_id", "event_time"], how="inner", suffix="_t")
    assert joined.height == sig_trunc.height
    a = joined["signal"].to_numpy()
    b = joined["signal_t"].to_numpy()
    both_nan = np.isnan(a) & np.isnan(b)
    np.testing.assert_allclose(np.where(both_nan, 0.0, a), np.where(both_nan, 0.0, b), atol=1e-12)


def test_tom_no_lookahead_month_boundary() -> None:
    """ToM uses only the trading calendar (public at t) — truncate at a month
    boundary and all earlier flags must be identical."""
    returns = {"A": [0.001] * 45, "B": [0.002] * 45}
    full = _panel_from_returns(returns, start="2024-01-02")
    truncated = full.filter(pl.col("event_time") <= pl.datetime(2024, 1, 31))
    sig_full = turn_of_month_signal(full)
    sig_trunc = turn_of_month_signal(truncated)
    joined = sig_full.join(sig_trunc, on=["security_id", "event_time"], how="inner", suffix="_t")
    np.testing.assert_array_equal(joined["signal"].to_numpy(), joined["signal_t"].to_numpy())


def test_bab_no_lookahead_shift() -> None:
    rng = np.random.default_rng(1)
    returns = {s: rng.normal(0.0, 0.02, 40).tolist() for s in ("A", "B", "C")}
    full = _panel_from_returns(returns)
    market_full = market_returns(full)
    truncated = full.filter(pl.col("event_time") <= pl.datetime(2024, 1, 26))
    market_trunc = market_returns(truncated)
    a = bab_signal(full, market_full, lookback=10)
    b = bab_signal(truncated, market_trunc, lookback=10)
    joined = a.join(b, on=["security_id", "event_time"], how="inner", suffix="_t")
    x = joined["signal"].to_numpy()
    y = joined["signal_t"].to_numpy()
    both_nan = np.isnan(x) & np.isnan(y)
    np.testing.assert_allclose(np.where(both_nan, 0.0, x), np.where(both_nan, 0.0, y), atol=1e-12)
