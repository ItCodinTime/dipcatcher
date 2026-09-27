"""Hypothesis property tests for the replication harness.

Invariants checked on random synthetic panels:
- boundedness: sign-restricted / decile outputs stay in range;
- no-lookahead: truncating the panel cannot change earlier signals;
- determinism: identical inputs give identical signals;
- score bounds: IC in [-1, 1], Brier/ECE in [0, 1];
- ToM flags depend only on the calendar, not on prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.research.replication.evaluation import (
    daily_ic,
    directional_scores,
    forward_returns,
)
from quant_fund.research.replication.panel import make_panel, market_returns
from quant_fund.research.replication.signals import (
    bab_signal,
    low_volatility_signal,
    overnight_signal,
    reversal_signal,
    tsmom_signal,
    tug_of_war_signal,
    turn_of_month_signal,
)


def _panel(returns: np.ndarray) -> pl.DataFrame:
    """n_assets x n_days return matrix -> canonical panel (zero overnight leg)."""
    n_assets, n_days = returns.shape
    days = pd.bdate_range("2024-01-02", periods=n_days).to_numpy()
    frames = []
    for i in range(n_assets):
        close = 100.0 * np.cumprod(1.0 + returns[i])
        open_ = np.concatenate([[close[0] / (1 + returns[i, 0])], close[:-1]])
        frames.append(
            pl.DataFrame(
                {
                    "security_id": f"S{i}",
                    "symbol": f"S{i}",
                    "event_time": days,
                    "open": open_,
                    "close": close,
                }
            )
        )
    return make_panel(pl.concat(frames))


RETURNS = st.builds(
    lambda seed, n_a, n_d: (
        np.random.default_rng(seed).normal(0.0, 0.02, (n_a, n_d)).clip(-0.2, 0.2)
    ),
    seed=st.integers(0, 10_000),
    n_a=st.integers(3, 6),
    n_d=st.integers(30, 60),
)


@given(RETURNS)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_signals_finite_or_null_and_deterministic(returns: np.ndarray) -> None:
    panel = _panel(returns)
    for builder in (
        lambda f: reversal_signal(f, lookback=5),
        lambda f: low_volatility_signal(f, lookback=10),
        lambda f: overnight_signal(f, lookback=5),
        lambda f: tug_of_war_signal(f, lookback=5),
    ):
        sig = builder(panel)
        assert set(sig.columns) == {"security_id", "event_time", "signal"}
        vals = sig["signal"].drop_nulls().to_numpy()
        assert np.isfinite(vals).all()
        # determinism
        assert builder(panel).equals(sig)


@given(RETURNS)
@settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_reversal_signal_bounded_by_cum_log_returns(returns: np.ndarray) -> None:
    panel = _panel(returns)
    lookback = 5
    sig = reversal_signal(panel, lookback=lookback)
    # |signal| <= sum of |log(1+r)| over lookback days (max |r| <= 0.2 by clip)
    bound = lookback * np.log(1.25)
    vals = sig["signal"].drop_nulls().to_numpy()
    assert (np.abs(vals) <= bound + 1e-12).all()


@given(RETURNS)
@settings(max_examples=12, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_no_lookahead_truncation_invariant(returns: np.ndarray) -> None:
    panel = _panel(returns)
    days = sorted(panel["event_time"].unique().to_list())
    cut = days[int(len(days) * 0.6)]
    trunc = panel.filter(pl.col("event_time") <= cut)
    builders = (
        lambda f: tsmom_signal(f, lookback=10, vol_com=5),
        lambda f: reversal_signal(f, lookback=5),
        lambda f: low_volatility_signal(f, lookback=10),
        lambda f: overnight_signal(f, lookback=5),
        lambda f: tug_of_war_signal(f, lookback=5),
        lambda f: bab_signal(f, market_returns(f), lookback=12),
    )
    for i, builder in enumerate(builders):
        full_sig = builder(panel)
        trunc_sig = builder(trunc)
        joined = full_sig.join(
            trunc_sig, on=["security_id", "event_time"], how="inner", suffix="_t"
        )
        a = joined["signal"].to_numpy()
        b = joined["signal_t"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        np.testing.assert_allclose(
            np.where(both_nan, 0.0, a),
            np.where(both_nan, 0.0, b),
            atol=1e-10,
            err_msg=f"builder {i} leaked future information",
        )


@given(RETURNS)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_ic_and_directional_score_bounds(returns: np.ndarray) -> None:
    panel = _panel(returns)
    sig = reversal_signal(panel, lookback=5)
    fwd = forward_returns(panel, 1, "ret_cc")
    ic = daily_ic(sig, fwd, min_assets=3)
    if ic.height:
        assert (ic["ic"].abs() <= 1.0 + 1e-9).all()
    up = fwd.with_columns((pl.col("fwd_ret") > 0.0).cast(pl.Float64).alias("up")).select(
        "security_id", "event_time", "up"
    )
    scores = directional_scores(sig, up, n_buckets=3, min_bucket_obs=10)
    if scores["n_scored"] > 0:
        assert 0.0 <= scores["brier"] <= 1.0
        assert 0.0 <= scores["ece"] <= 1.0
        assert scores["log_loss"] >= 0.0


@given(RETURNS)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_tom_flags_price_independent(returns: np.ndarray) -> None:
    """ToM flags must be a pure function of the trading calendar."""
    panel = _panel(returns)
    shocked = panel.with_columns((pl.col("close") * 3.0).alias("close"))
    flags_a = turn_of_month_signal(panel)
    flags_b = turn_of_month_signal(shocked)
    np.testing.assert_array_equal(
        flags_a.sort(["security_id", "event_time"])["signal"].to_numpy(),
        flags_b.sort(["security_id", "event_time"])["signal"].to_numpy(),
    )
    vals = flags_a["signal"].unique().to_numpy()
    assert set(np.asarray(vals)).issubset({0.0, 1.0})
