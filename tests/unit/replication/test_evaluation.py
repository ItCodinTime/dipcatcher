"""Known-answer tests for the scoring harness (IC, deciles, directional)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from quant_fund.research.replication.evaluation import (
    daily_ic,
    decile_spread,
    directional_scores,
    forward_returns,
    ic_summary,
    market_timing_eval,
)


def _frames(signal_map: dict[str, list[float | None]], ret_map: dict[str, list[float]]):
    """Build signal + ret_cc frames with shared dates (business days)."""
    import pandas as pd

    secs = list(signal_map)
    n = len(ret_map[secs[0]])
    days = pd.bdate_range("2024-01-02", periods=n).to_numpy()
    flat_signal = [float("nan") if v is None else float(v) for s in secs for v in signal_map[s]]
    sig_df = pl.DataFrame(
        {
            "security_id": np.repeat(secs, n),
            "event_time": np.tile(days, len(secs)),
            "signal": np.asarray(flat_signal, dtype=np.float64),
        }
    )
    panel = pl.DataFrame(
        {
            "security_id": np.repeat(secs, n),
            "event_time": np.tile(days, len(secs)),
            "ret_cc": np.concatenate([ret_map[s] for s in secs]),
        }
    )
    return sig_df, panel


def test_forward_returns_compounds_next_h_bars() -> None:
    import pandas as pd

    days = pd.bdate_range("2024-01-02", periods=4).to_numpy()
    panel = pl.DataFrame(
        {
            "security_id": ["A"] * 4,
            "event_time": days,
            "ret_cc": [0.01, -0.02, 0.03, 0.05],
        }
    )
    fwd = forward_returns(panel, 2, "ret_cc").sort("event_time")
    assert fwd["fwd_ret"][0] == pytest.approx((1 - 0.02) * (1.03) - 1.0)
    assert fwd["fwd_ret"][1] == pytest.approx(1.03 * 1.05 - 1.0)
    assert fwd["fwd_ret"][2] is None and fwd["fwd_ret"][3] is None


def test_daily_ic_perfect_and_inverse() -> None:
    # One date, 6 assets: signal perfectly monotone with fwd return.
    rets = {s: [0.0, (i + 1) * 0.001] for i, s in enumerate("ABCDEF")}
    sig_map = {s: [None, (i + 1) * 0.1] for i, s in enumerate("ABCDEF")}
    sig_df, panel = _frames(sig_map, rets)
    fwd = panel.rename({"ret_cc": "fwd_ret"})
    ic = daily_ic(sig_df, fwd, min_assets=4)
    assert ic.height == 1
    assert ic["ic"][0] == pytest.approx(1.0)
    ic_neg = daily_ic(sig_df.with_columns((-pl.col("signal")).alias("signal")), fwd, min_assets=4)
    assert ic_neg["ic"][0] == pytest.approx(-1.0)


def test_daily_ic_min_assets_filters() -> None:
    rets = {s: [0.0, 0.001] for s in "AB"}
    sig_map = {s: [None, 0.1] for s in "AB"}
    sig_df, panel = _frames(sig_map, rets)
    ic = daily_ic(sig_df, panel.rename({"ret_cc": "fwd_ret"}), min_assets=4)
    assert ic.height == 0


def test_ic_summary_bounds() -> None:
    frame = pl.DataFrame(
        {"event_time": [1, 2, 3, 4, 5], "ic": [0.1, 0.2, -0.1, 0.05, 0.15], "n_assets": 5}
    )
    s = ic_summary(frame, 1)
    assert -1.0 <= s["mean_ic"] <= 1.0
    assert s["n_dates"] == 5.0
    assert math.isfinite(s["ic_tstat_nw"])
    assert 0.0 <= s["ic_p_value_nw"] <= 1.0


def test_decile_spread_known_answer() -> None:
    # 10 assets, one date; signal ranks align with fwd returns by +0.01 steps.
    n = 10
    rets = {f"S{i}": [0.0, i * 0.01 - 0.045] for i in range(n)}
    sig_map = {f"S{i}": [None, float(i)] for i in range(n)}
    sig_df, panel = _frames(sig_map, rets)
    res = decile_spread(sig_df, panel.rename({"ret_cc": "fwd_ret"}), n_buckets=10, min_assets=10)
    # top bucket asset S9 (fwd 0.045) minus bottom S0 (-0.045) = 0.09
    assert res["descriptive_bucket_spread_mean"] == pytest.approx(0.09)
    assert res["descriptive_spread_n_dates"] == 1.0


def test_directional_scores_no_lookahead_and_bounds() -> None:
    # Signal perfectly predicts up-down by construction: bucket 4 -> up,
    # bucket 0 -> down. After warmup the forecaster must approach 0/1 probs.
    n_days, n_assets = 40, 10
    rng = np.random.default_rng(3)
    import pandas as pd

    days = pd.bdate_range("2024-01-02", periods=n_days).to_numpy()
    sec_col, day_col, sig_col, up_col = [], [], [], []
    for d in days:
        perm = rng.permutation(n_assets)
        for rank, ai in enumerate(perm):
            sec_col.append(f"S{ai}")
            day_col.append(d)
            sig_col.append(float(rank))
            # top half of ranks always up, bottom half always down
            up_col.append(1.0 if rank >= n_assets // 2 else 0.0)
    sig = pl.DataFrame(
        {
            "security_id": sec_col,
            "event_time": np.asarray(day_col),
            "signal": np.asarray(sig_col),
        }
    )
    up = pl.DataFrame(
        {
            "security_id": sec_col,
            "event_time": np.asarray(day_col),
            "up": np.asarray(up_col),
        }
    )
    res = directional_scores(sig, up, n_buckets=2, min_bucket_obs=20)
    assert res["n_scored"] > 0
    assert 0.0 <= res["brier"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0
    # a perfectly separable predictor should beat climatology meaningfully
    assert res["brier"] < res["brier_climatology"]
    assert res["brier_skill_vs_climatology"] > 0.0


def test_directional_scores_empty_input() -> None:
    empty_sig = pl.DataFrame(
        schema={"security_id": pl.String, "event_time": pl.Datetime, "signal": pl.Float64}
    )
    empty_up = pl.DataFrame(
        schema={"security_id": pl.String, "event_time": pl.Datetime, "up": pl.Float64}
    )
    assert directional_scores(empty_sig, empty_up)["n_scored"] == 0.0


def test_market_timing_eval_known_answer() -> None:
    import pandas as pd

    days = pd.bdate_range("2024-01-02", periods=30).to_numpy()
    flags = np.zeros(30)
    flags[[9, 10, 11]] = 1.0  # three in-window days
    mkt = pl.DataFrame(
        {
            "event_time": days,
            "mkt_ret_cc": np.where(flags == 1.0, 0.05, -0.001),
        }
    )
    flag_frame = pl.DataFrame({"event_time": days, "signal": flags})
    res = market_timing_eval(mkt, flag_frame)
    assert res["n_in_window"] == 3.0
    assert res["descriptive_in_window_mean"] == pytest.approx(0.05)
    assert res["descriptive_in_minus_out"] > 0.0
    assert res["descriptive_welch_p"] < 0.05
    assert 0.0 <= res["direction_brier"] <= 1.0
