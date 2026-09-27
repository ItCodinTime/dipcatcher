"""Scoring for replicated signals — proper scores headline, descriptive second.

Honesty contract: the headline metrics are proper scores (Brier, log-loss of a
walk-forward directional probability forecast vs a climatology baseline) and
cross-sectional rank IC statistics (per-date Spearman, Newey-West t-stat via
``metrics.inference.mean_tstat`` with overlap-aware lags). Portfolio-style
statistics (decile spreads, in/out-of-window mean returns) are emitted only
under the ``descriptive_*`` key namespace — they are not headline evidence.

No-lookahead convention: signal rows at date ``t`` are joined to forward
returns that begin at ``t + 1``; the directional forecaster estimates
``P(up | signal bucket)`` from pairs strictly before ``t`` (expanding window).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from quant_fund.metrics.inference import mean_tstat, overlap_aware_hac_lags
from quant_fund.metrics.probability import brier_score, expected_calibration_error, log_loss


def forward_returns(frame: pl.DataFrame, horizon: int, col: str = "ret_cc") -> pl.DataFrame:
    """Per-security compounded forward return over the next ``horizon`` bars.

    ``fwd[t] = prod(1 + r_{t+1..t+h}) - 1``; the last ``horizon`` rows per
    security are null and never scored.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    frame = frame.sort(["security_id", "event_time"])
    rolled = pl.col(col).log1p().rolling_sum(window_size=horizon, min_samples=horizon)
    fwd = (rolled.shift(-horizon).over("security_id")).exp() - 1.0
    return frame.with_columns(fwd.alias("fwd_ret")).select("security_id", "event_time", "fwd_ret")


def daily_ic(signal: pl.DataFrame, fwd: pl.DataFrame, min_assets: int = 4) -> pl.DataFrame:
    """Per-date cross-sectional Spearman IC between signal and forward return."""
    joined = signal.join(fwd, on=["security_id", "event_time"], how="inner").drop_nulls()
    joined = joined.filter(pl.col("signal").is_finite() & pl.col("fwd_ret").is_finite())
    rows: list[dict[str, Any]] = []
    for (event_time,), part in joined.group_by("event_time", maintain_order=True):
        if part.height < min_assets:
            continue
        rho = stats.spearmanr(part["signal"].to_numpy(), part["fwd_ret"].to_numpy()).statistic
        if rho is not None and math.isfinite(float(rho)):
            rows.append(
                {"event_time": event_time, "ic": float(rho), "n_assets": float(part.height)}
            )
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={"event_time": pl.Datetime, "ic": pl.Float64, "n_assets": pl.Float64}
        )
    )


def ic_summary(ic_frame: pl.DataFrame, horizon: int) -> dict[str, Any]:
    """Mean IC + Newey-West t-stat with overlap-aware (Hansen-Hodrick) lags."""
    ics = ic_frame["ic"].to_numpy() if ic_frame.height else np.empty(0)
    n_dates = int(ics.size)
    lags = overlap_aware_hac_lags(n_dates, horizon)
    mean_ic, t_stat, p_value = mean_tstat(ics, lags=lags)
    return {
        "mean_ic": mean_ic,
        "ic_tstat_nw": t_stat,
        "ic_p_value_nw": p_value,
        "ic_nw_lags": float(lags),
        "n_dates": float(n_dates),
        "ic_positive_fraction": float(np.mean(ics > 0.0)) if n_dates else float("nan"),
    }


def decile_spread(
    signal: pl.DataFrame, fwd: pl.DataFrame, n_buckets: int = 10, min_assets: int = 10
) -> dict[str, float]:
    """Descriptive top-minus-bottom-bucket mean forward return (not headline).

    Returns mean spread across dates plus its Newey-West t-stat — descriptive
    performance context only; the receipt keeps it under ``descriptive_``.
    """
    joined = signal.join(fwd, on=["security_id", "event_time"], how="inner").drop_nulls()
    joined = joined.filter(pl.col("signal").is_finite() & pl.col("fwd_ret").is_finite())
    spreads: list[float] = []
    for (_event_time,), part in joined.group_by("event_time", maintain_order=True):
        if part.height < min_assets:
            continue
        s = part["signal"].to_numpy()
        r = part["fwd_ret"].to_numpy()
        ranks01 = (stats.rankdata(s) - 1.0) / s.size  # [0, 1)
        bucket = np.clip((ranks01 * n_buckets).astype(int), 0, n_buckets - 1)
        top, bot = bucket == n_buckets - 1, bucket == 0
        if top.any() and bot.any():
            spreads.append(float(np.mean(r[top]) - np.mean(r[bot])))
    arr = np.asarray(spreads)
    # mean is always reportable; the t-stat needs n>=3 and stays NaN below it
    mean_spread = float(np.mean(arr)) if arr.size else float("nan")
    _mu, t_stat, _p = mean_tstat(arr)
    return {
        "descriptive_bucket_spread_mean": mean_spread,
        "descriptive_bucket_spread_tstat": t_stat,
        "descriptive_spread_n_dates": float(arr.size),
    }


def _bucket_assign(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Within-date quantile bucket ids 0..n_buckets-1 for a cross-section."""
    ranks01 = (stats.rankdata(values) - 1.0) / values.size  # [0, 1)
    return np.clip((ranks01 * n_buckets).astype(int), 0, n_buckets - 1)


def directional_scores(
    signal: pl.DataFrame,
    outcome_up: pl.DataFrame,
    *,
    n_buckets: int = 5,
    min_bucket_obs: int = 40,
    fixed_bucket_col: str | None = None,
) -> dict[str, Any]:
    """Walk-forward directional probability forecast scored with proper rules.

    For each date ``t`` the signal's within-date quantile bucket is known at
    ``t``; the forecast ``P(up)`` is the empirical up-rate of that bucket over
    all observations strictly before ``t`` (expanding window — no lookahead).
    The climatology baseline is the expanding unconditional up-rate. Buckets
    with fewer than ``min_bucket_obs`` past observations are unscored (NaN,
    dropped). ``fixed_bucket_col`` skips ranking and uses a provided integer
    column (e.g. the ToM 0/1 flag) — for market-timing signals.
    """
    joined = signal.join(outcome_up, on=["security_id", "event_time"], how="inner")
    joined = joined.drop_nulls(subset=["signal", "up"]).sort("event_time")
    joined = joined.filter(pl.col("signal").is_finite())
    if joined.height == 0:
        return {"n_scored": 0.0}
    n_bucket_slots = max(64, int(n_buckets) * 2)
    probs: list[float] = []
    clim: list[float] = []
    outs: list[float] = []
    bucket_sum = np.zeros(n_bucket_slots, dtype=float)
    bucket_cnt = np.zeros(n_bucket_slots, dtype=float)
    total_sum = 0.0
    total_cnt = 0.0
    for (_event_time,), part in joined.group_by("event_time", maintain_order=True):
        sig = part["signal"].to_numpy()
        up = part["up"].to_numpy()
        if fixed_bucket_col is None:
            buckets = _bucket_assign(sig, n_buckets)
        else:
            buckets = part[fixed_bucket_col].to_numpy().astype(int)
        for bucket, y in zip(buckets, up, strict=True):
            if total_cnt > 0:
                clim.append(total_sum / total_cnt)
                outs.append(float(y))
                if 0 <= bucket < n_bucket_slots and bucket_cnt[bucket] >= min_bucket_obs:
                    probs.append(bucket_sum[bucket] / bucket_cnt[bucket])
                else:
                    probs.append(float("nan"))
        for bucket, y in zip(buckets, up, strict=True):
            if 0 <= bucket < n_bucket_slots:
                bucket_sum[bucket] += float(y)
                bucket_cnt[bucket] += 1.0
        total_sum += float(np.sum(up))
        total_cnt += float(up.size)
    outs_arr = np.asarray(outs)
    clim_arr = np.asarray(clim)
    prob_arr = np.asarray(probs)
    scored = np.isfinite(prob_arr)
    n_scored = int(scored.sum())
    result: dict[str, Any] = {
        "n_scored": float(n_scored),
        "n_eligible": float(outs_arr.size),
    }
    if n_scored == 0:
        return result
    y = outs_arr[scored]
    p = prob_arr[scored]
    c = clim_arr[scored]
    brier = brier_score(p, y)
    brier_clim = brier_score(c, y)
    result.update(
        {
            "brier": brier,
            "log_loss": log_loss(p, y),
            "ece": expected_calibration_error(p, y, n_bins=min(10, n_buckets * 2)),
            "brier_climatology": brier_clim,
            "log_loss_climatology": log_loss(c, y),
            "brier_skill_vs_climatology": brier_clim - brier,
            "base_rate_up": float(np.mean(y)),
        }
    )
    return result


def market_timing_eval(
    market: pl.DataFrame,
    flags: pl.DataFrame,
    ret_col: str = "mkt_ret_cc",
) -> dict[str, Any]:
    """McConnell-Xu (2008) style in-window vs out-of-window market test.

    ``flags`` must carry ``event_time`` + ``signal`` (the ToM 0/1 flag).
    Descriptive statistics are prefixed ``descriptive_``; the proper-score leg
    is the directional forecaster with the flag as the (fixed) bucket.
    """
    flag_dates = flags.select("event_time", "signal").unique(
        subset="event_time", keep="first", maintain_order=True
    )
    merged = (
        market.select("event_time", ret_col)
        .join(flag_dates, on="event_time", how="inner")
        .drop_nulls()
        .sort("event_time")
    )
    if merged.height < 10:
        return {"n_obs": float(merged.height)}
    flag = merged["signal"].to_numpy() > 0.5
    ret = merged[ret_col].to_numpy()
    in_w, out_w = ret[flag], ret[~flag]
    # Welch t-test, descriptive only (calendar windows are serially lumpy).
    t_res = stats.ttest_ind(in_w, out_w, equal_var=False)
    up = merged.select("event_time", (pl.col(ret_col) > 0.0).cast(pl.Float64).alias("up"))
    pseudo_signal = merged.select(
        pl.lit("MKT").alias("security_id"),
        "event_time",
        pl.col("signal").alias("bucket_flag"),
    ).with_columns(pl.col("bucket_flag").cast(pl.Float64).alias("signal"))
    pseudo_outcome = up.with_columns(pl.lit("MKT").alias("security_id")).select(
        "security_id", "event_time", "up"
    )
    scores = directional_scores(
        pseudo_signal,
        pseudo_outcome,
        fixed_bucket_col="bucket_flag",
        n_buckets=2,
        min_bucket_obs=5,
    )
    return {
        "n_obs": float(merged.height),
        "n_in_window": float(flag.sum()),
        "descriptive_in_window_mean": float(np.mean(in_w)),
        "descriptive_out_window_mean": float(np.mean(out_w)),
        "descriptive_in_minus_out": float(np.mean(in_w) - np.mean(out_w)),
        "descriptive_welch_t": float(t_res.statistic),
        "descriptive_welch_p": float(t_res.pvalue),
        **{f"direction_{k}": v for k, v in scores.items()},
    }
