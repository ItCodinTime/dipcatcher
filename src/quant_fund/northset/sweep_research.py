"""Institutional evidence battery for Northset liquidity sweeps.

The detector lives in ``sweeps.py``. This module tests whether reclaim and
follow-through events survive controls: executable next-open timing,
cross-sectional market demeaning, horizon-aware HAC, block bootstrap,
within-date permutation placebos, BH-FDR, chronological fold stability,
volatility regimes, explicit round-trip cost hurdles, matched and
liquidity-quartile controls, name-clustered t, two-way clustered t,
wild-cluster bootstrap, overnight untradeable-gap disclosure,
out-of-time holdout, a
predeclared primary test, and event ADV-participation disclosure.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    mean_tstat,
    onesided_from_twosided,
    two_way_clustered_mean_tstat,
    wild_cluster_bootstrap_two_way_p,
)

_SIGNALS = ("sweep_reject_signed", "sweep_follow_signed")

# Predeclared before looking at the sample. H45 is the primary executable
# test; event-study FDR, lookback cells, and other H-rows are secondaries.
PRIMARY_EXECUTABLE_TEST: dict[str, Any] = {
    "id": "H45_northset_follow_control",
    "signal": "sweep_follow_signed",
    "horizon": 1,
    "entry": "next_open",
    "control": "same_date_eligible_non_swept",
    "role": "primary_executable",
    "declared_before_look": True,
}


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    a, b = x[mask], y[mask]
    if float(np.std(a)) <= 1e-18 or float(np.std(b)) <= 1e-18:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _deduplicate_sweep_episodes(frame: pl.DataFrame, cooldown: int) -> pl.DataFrame:
    """Keep only the first same-direction event within a per-security cooldown."""
    if cooldown <= 0:
        return frame
    out = frame.sort(["security_id", "event_time"])
    for signal in _SIGNALS:
        pos = f"_{signal}_recent_pos"
        neg = f"_{signal}_recent_neg"
        out = out.with_columns(
            (pl.col(signal) > 0.0)
            .cast(pl.Int8)
            .shift(1)
            .rolling_max(window_size=cooldown, min_samples=1)
            .over("security_id")
            .fill_null(0)
            .alias(pos),
            (pl.col(signal) < 0.0)
            .cast(pl.Int8)
            .shift(1)
            .rolling_max(window_size=cooldown, min_samples=1)
            .over("security_id")
            .fill_null(0)
            .alias(neg),
        )
        out = out.with_columns(
            pl.when(
                ((pl.col(signal) > 0.0) & (pl.col(pos) == 0))
                | ((pl.col(signal) < 0.0) & (pl.col(neg) == 0))
            )
            .then(pl.col(signal))
            .otherwise(0.0)
            .alias(signal)
        ).drop([pos, neg])
    return out


def sweep_forward_frame(
    sweep_frame: pl.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 20),
    vol_lookback: int = 20,
) -> pl.DataFrame:
    """Attach next-open-to-future-close excess returns and PIT volatility regime.

    An event is known only after the event bar closes. Every horizon therefore
    enters at the next bar's open and exits at close[t+h]. The cross-sectional
    date mean is removed before testing. The volatility regime uses returns
    strictly before the event via a shifted rolling standard deviation.
    Invalid OHLC prints are excluded from lagged vol and from entry/exit prices.
    """
    required = ("security_id", "event_time", "open", "close", *_SIGNALS)
    missing = [c for c in required if c not in sweep_frame.columns]
    if missing:
        raise ValueError(f"sweep frame missing required columns: {missing}")
    hs = tuple(sorted(set(int(h) for h in horizons)))
    if not hs or any(h < 1 for h in hs):
        raise ValueError("horizons must contain positive integers")
    if vol_lookback < 3:
        raise ValueError("vol_lookback must be >= 3")
    frame = sweep_frame.sort(["security_id", "event_time"])
    ok = pl.col("ohlc_ok") if "ohlc_ok" in frame.columns else pl.lit(True)
    prev_ok = ok.shift(1).over("security_id")
    frame = frame.with_columns(
        pl.when(ok & prev_ok.fill_null(False))
        .then((pl.col("close") / pl.col("close").shift(1).over("security_id")).log())
        .otherwise(None)
        .alias("_log_ret")
    )
    frame = frame.with_columns(
        pl.col("_log_ret")
        .shift(1)
        .rolling_std(window_size=vol_lookback, min_samples=vol_lookback)
        .over("security_id")
        .alias("sweep_lagged_vol")
    )
    frame = frame.with_columns(
        pl.col("sweep_lagged_vol").median().over("event_time").alias("_date_vol_median")
    ).with_columns(
        pl.when(pl.col("sweep_lagged_vol").is_null())
        .then(None)
        .when(pl.col("sweep_lagged_vol") >= pl.col("_date_vol_median"))
        .then(pl.lit("high"))
        .otherwise(pl.lit("low"))
        .alias("sweep_vol_regime")
    )
    exprs: list[pl.Expr] = []
    entry_column = "return_open" if "return_open" in frame.columns else "open"
    exit_column = "return_close" if "return_close" in frame.columns else "close"
    for horizon in hs:
        next_open = pl.col(entry_column).shift(-1).over("security_id")
        exit_close = pl.col(exit_column).shift(-horizon).over("security_id")
        entry_ok = ok.shift(-1).over("security_id")
        exit_ok = ok.shift(-horizon).over("security_id")
        exprs.append(
            pl.when(entry_ok.fill_null(False) & exit_ok.fill_null(False))
            .then(exit_close / next_open - 1.0)
            .otherwise(None)
            .alias(f"sweep_exec_ret_{horizon}")
        )
    frame = frame.with_columns(*exprs)
    next_open = pl.col(entry_column).shift(-1).over("security_id")
    event_close = pl.col(exit_column)
    entry_ok = ok.shift(-1).over("security_id")
    frame = frame.with_columns(
        pl.when(ok.fill_null(False) & entry_ok.fill_null(False))
        .then(next_open / event_close - 1.0)
        .otherwise(None)
        .alias("sweep_overnight_ret")
    )
    frame = frame.with_columns(
        (
            pl.col("sweep_overnight_ret") - pl.col("sweep_overnight_ret").mean().over("event_time")
        ).alias("sweep_overnight_excess")
    )
    for horizon in hs:
        raw = f"sweep_exec_ret_{horizon}"
        frame = frame.with_columns(
            (pl.col(raw) - pl.col(raw).mean().over("event_time")).alias(
                f"sweep_excess_ret_{horizon}"
            )
        )
    return frame.drop(["_log_ret", "_date_vol_median"])


def estimated_round_trip_cost_bps(frame: pl.DataFrame, config: AppConfig) -> float:
    """Median PIT two-sided spread, commission, turnover, and impact hurdle."""
    costed = _attach_event_costs(frame, config)
    values = costed["sweep_round_trip_cost_bps"].to_numpy().astype(float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _attach_event_costs(frame: pl.DataFrame, config: AppConfig) -> pl.DataFrame:
    """Attach per-row costs using volatility available strictly before the event."""
    costs = config.costs
    base = 2.0 * (
        float(costs.commission_bps) + float(costs.half_spread_bps) + float(costs.bps_per_turnover)
    )
    out = frame
    if "sweep_lagged_vol" not in out.columns:
        out = out.sort(["security_id", "event_time"]).with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("security_id"))
            .log()
            .alias("_cost_ret")
        )
        out = out.with_columns(
            pl.col("_cost_ret")
            .shift(1)
            .rolling_std(window_size=20, min_samples=20)
            .over("security_id")
            .alias("sweep_lagged_vol")
        ).drop("_cost_ret")
    participation = float(config.execution.participation_rate)
    return out.with_columns(
        (
            pl.lit(base)
            + 2.0
            * float(costs.impact_y)
            * pl.col("sweep_lagged_vol")
            * math.sqrt(participation)
            * 1e4
        ).alias("sweep_round_trip_cost_bps")
    )


def within_date_permutation_test(
    frame: pl.DataFrame,
    *,
    score: str,
    target: str,
    min_names: int,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int]:
    """Permutation placebo preserving each date's score and target marginals."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _date, group in (
        frame.select(["event_time", score, target])
        .drop_nulls()
        .group_by("event_time", maintain_order=True)
    ):
        x = group[score].to_numpy().astype(float)
        y = group[target].to_numpy().astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) >= min_names:
            groups.append((x[mask], y[mask]))
    observed_by_date = np.asarray([_finite_corr(x, y) for x, y in groups], dtype=float)
    observed_by_date = observed_by_date[np.isfinite(observed_by_date)]
    if observed_by_date.size < 3:
        return {
            "observed_mean_ic": float("nan"),
            "placebo_p_value": float("nan"),
            "n_dates": int(observed_by_date.size),
            "n_permutations": int(n_permutations),
        }
    observed = float(np.mean(observed_by_date))
    rng = np.random.default_rng(seed)
    placebo = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        per_date = [_finite_corr(rng.permutation(x), y) for x, y in groups]
        vals = np.asarray(per_date, dtype=float)
        placebo[i] = float(np.nanmean(vals))
    p_value = (1.0 + float(np.sum(np.abs(placebo) >= abs(observed)))) / (
        float(n_permutations) + 1.0
    )
    return {
        "observed_mean_ic": observed,
        "placebo_p_value": float(p_value),
        "n_dates": int(observed_by_date.size),
        "n_permutations": int(n_permutations),
    }


def _event_daily_frame(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    regime: str | None = None,
) -> tuple[pl.DataFrame, int]:
    subset = frame.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
    if regime is not None:
        subset = subset.filter(pl.col("sweep_vol_regime") == regime)
    subset = subset.select(
        "event_time",
        (pl.col(signal).sign() * pl.col(target)).alias("_daily"),
        "sweep_round_trip_cost_bps",
    ).drop_nulls()
    if subset.height == 0:
        empty = pl.DataFrame(
            schema={
                "event_time": frame.schema.get("event_time", pl.Datetime("us", "UTC")),
                "_daily": pl.Float64,
                "_daily_cost_bps": pl.Float64,
            }
        )
        return empty, 0
    daily = subset.group_by("event_time", maintain_order=True).agg(
        pl.col("_daily").mean(),
        pl.col("sweep_round_trip_cost_bps").mean().alias("_daily_cost_bps"),
    )
    return daily, int(subset.height)


def _panel_calendar(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select("event_time").unique().sort("event_time")


def _embed_value_on_calendar(
    daily: pl.DataFrame,
    calendar: pl.DataFrame,
    value_col: str,
) -> np.ndarray:
    """Map a date-level series onto the panel calendar; idle dates are 0."""
    n_cal = int(calendar.height)
    if daily.height == 0 or n_cal == 0 or value_col not in daily.columns:
        return np.zeros(n_cal, dtype=float)
    aligned = calendar.join(
        daily.select("event_time", value_col),
        on="event_time",
        how="left",
    ).sort("event_time")
    return aligned[value_col].fill_null(0.0).to_numpy().astype(float)


def _embed_on_calendar(
    daily: pl.DataFrame,
    calendar: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Map event-date means onto the panel calendar; idle dates are 0.

    HAC / block bootstrap then see calendar time, not compressed event dates.
    """
    return (
        _embed_value_on_calendar(daily, calendar, "_daily"),
        _embed_value_on_calendar(daily, calendar, "_daily_cost_bps"),
    )


def _event_daily_series(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    regime: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Event-date series (no idle zeros). Prefer calendar embed for inference."""
    daily, n_events = _event_daily_frame(frame, signal=signal, target=target, regime=regime)
    if daily.height == 0:
        return np.array([], dtype=float), np.array([], dtype=float), 0
    return (
        daily["_daily"].to_numpy().astype(float),
        daily["_daily_cost_bps"].to_numpy().astype(float),
        n_events,
    )


def matched_control_difference(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    min_events: int,
    min_dates: int,
) -> dict[str, float | int | str]:
    """Direction-matched event-vs-control difference of same-date excess returns.

    Controls are sweep-eligible names on the same date that swept **neither**
    side. Per date and event direction ``s``: ``s * (mean event excess −
    mean control excess)``; date cells are averaged, then HAC t across dates.
    This controls for date-level effects *and* for eligibility selection.
    """
    required = ("event_time", "sweep_eligible", "sweep_high", "sweep_low", signal, target)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for matched controls: {missing}")
    base = frame.filter(
        pl.col("sweep_eligible") & pl.col(target).is_not_null() & pl.col(target).is_finite()
    )
    controls = (
        base.filter(
            (pl.col("sweep_high").fill_null(0.0) == 0.0)
            & (pl.col("sweep_low").fill_null(0.0) == 0.0)
        )
        .group_by("event_time")
        .agg(
            pl.col(target).mean().alias("_control_mean"),
            pl.len().alias("_n_controls"),
        )
    )
    events = (
        base.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .with_columns(pl.col(signal).sign().alias("_direction"))
        .group_by(["event_time", "_direction"])
        .agg(
            pl.col(target).mean().alias("_event_mean"),
            pl.len().alias("_n_events"),
        )
    )
    cells = events.join(controls, on="event_time", how="inner").with_columns(
        (pl.col("_direction") * (pl.col("_event_mean") - pl.col("_control_mean"))).alias("_diff")
    )
    if cells.height == 0:
        return _empty_control_diff("same_date_eligible_non_swept")
    daily = cells.group_by("event_time", maintain_order=True).agg(pl.col("_diff").mean())
    return _hac_date_diffs(
        daily["_diff"].to_numpy().astype(float),
        n_events=int(cells["_n_events"].sum()),
        n_controls=int(controls["_n_controls"].sum()),
        min_events=min_events,
        min_dates=min_dates,
        design="same_date_eligible_non_swept",
        calendar_diffs=_embed_value_on_calendar(daily, _panel_calendar(frame), "_diff"),
    )


def _empty_control_diff(design: str) -> dict[str, float | int | str]:
    return {
        "control_design": design,
        "n_events": 0,
        "n_control_rows": 0,
        "n_dates": 0,
        "n_calendar_dates": 0,
        "sample_adequate": False,
        "mean_diff_bps": float("nan"),
        "hac_t": float("nan"),
        "p_value": float("nan"),
        "inference_index": "calendar_including_idle_zeros",
    }


def _hac_date_diffs(
    diffs: np.ndarray,
    *,
    n_events: int,
    n_controls: int,
    min_events: int,
    min_dates: int,
    design: str,
    calendar_diffs: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    finite = diffs[np.isfinite(diffs)]
    sample_adequate = bool(n_events >= min_events and finite.size >= min_dates)
    mean = float(np.mean(finite)) if finite.size else float("nan")
    infer = calendar_diffs if calendar_diffs is not None else finite
    if sample_adequate:
        _mu, t_stat, p_value = mean_tstat(infer, lags=1)
    else:
        t_stat = p_value = float("nan")
    return {
        "control_design": design,
        "n_events": int(n_events),
        "n_control_rows": int(n_controls),
        "n_dates": int(finite.size),
        "n_calendar_dates": int(infer.size),
        "sample_adequate": sample_adequate,
        "mean_diff_bps": float(mean * 1e4) if math.isfinite(mean) else float("nan"),
        "hac_t": float(t_stat),
        "p_value": float(p_value),
        "inference_index": "calendar_including_idle_zeros",
    }


def liquidity_matched_control_difference(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    min_events: int,
    min_dates: int,
    n_quartiles: int = 4,
) -> dict[str, float | int | str]:
    """Event-vs-control difference matched on date *and* lagged dollar-volume quartile.

    Lagged dollar volume is ``close * volume`` shifted one bar per name (PIT).
    Quartiles are formed among sweep-eligible names on the same date. Controls
    are eligible non-swept names in the event's quartile. This is stricter than
    ``matched_control_difference``: a thin-name event is not compared to the
    date's liquid names.
    """
    required = (
        "event_time",
        "security_id",
        "sweep_eligible",
        "sweep_high",
        "sweep_low",
        "close",
        "volume",
        signal,
        target,
    )
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for liquidity-matched controls: {missing}")
    if n_quartiles < 2:
        raise ValueError("n_quartiles must be >= 2")
    base = (
        frame.sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close") * pl.col("volume")).shift(1).over("security_id").alias("_lagged_dvol")
        )
        .filter(
            pl.col("sweep_eligible")
            & pl.col(target).is_not_null()
            & pl.col(target).is_finite()
            & pl.col("_lagged_dvol").is_finite()
            & (pl.col("_lagged_dvol") > 0.0)
        )
        .with_columns(
            (
                (
                    (pl.col("_lagged_dvol").rank("average").over("event_time") - 1.0)
                    * float(n_quartiles)
                    / pl.len().over("event_time")
                )
                .floor()
                .clip(0, n_quartiles - 1)
                .cast(pl.Int8)
            ).alias("_liq_q")
        )
    )
    if base.height == 0:
        return _empty_control_diff("same_date_eligible_non_swept_lagged_dvol_quartile")
    controls = (
        base.filter(
            (pl.col("sweep_high").fill_null(0.0) == 0.0)
            & (pl.col("sweep_low").fill_null(0.0) == 0.0)
        )
        .group_by(["event_time", "_liq_q"])
        .agg(
            pl.col(target).mean().alias("_control_mean"),
            pl.len().alias("_n_controls"),
        )
    )
    events = (
        base.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .with_columns(pl.col(signal).sign().alias("_direction"))
        .group_by(["event_time", "_direction", "_liq_q"])
        .agg(
            pl.col(target).mean().alias("_event_mean"),
            pl.len().alias("_n_events"),
        )
    )
    cells = events.join(controls, on=["event_time", "_liq_q"], how="inner").with_columns(
        (pl.col("_direction") * (pl.col("_event_mean") - pl.col("_control_mean"))).alias("_diff")
    )
    if cells.height == 0:
        return _empty_control_diff("same_date_eligible_non_swept_lagged_dvol_quartile")
    daily = cells.group_by("event_time", maintain_order=True).agg(pl.col("_diff").mean())
    return _hac_date_diffs(
        daily["_diff"].to_numpy().astype(float),
        n_events=int(cells["_n_events"].sum()),
        n_controls=int(controls["_n_controls"].sum()),
        min_events=min_events,
        min_dates=min_dates,
        design="same_date_eligible_non_swept_lagged_dvol_quartile",
        calendar_diffs=_embed_value_on_calendar(daily, _panel_calendar(frame), "_diff"),
    )


def name_clustered_event_mean(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    min_names: int,
    min_events: int,
) -> dict[str, float | int | str]:
    """t-test of per-security mean signed excess (names, not stacked dates).

    Date-level HAC already clusters by calendar. This robustness check collapses
    each name to one mean, then uses an iid t across names (``lags=0``). Names
    are not a time series; Newey–West lag-1 would be a fiction.
    """
    required = ("security_id", "event_time", signal, target)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for name-clustered t: {missing}")
    subset = (
        frame.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .select(
            "security_id",
            (pl.col(signal).sign() * pl.col(target)).alias("_signed_excess"),
        )
        .drop_nulls()
    )
    if subset.height == 0:
        return {
            "cluster": "security_id",
            "se_kernel": "iid_across_names",
            "n_events": 0,
            "n_names": 0,
            "sample_adequate": False,
            "mean_excess_bps": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
        }
    per_name = subset.group_by("security_id").agg(pl.col("_signed_excess").mean())
    means = per_name["_signed_excess"].to_numpy().astype(float)
    means = means[np.isfinite(means)]
    n_events = int(subset.height)
    sample_adequate = bool(n_events >= min_events and means.size >= min_names)
    if sample_adequate:
        mean, t_stat, p_value = mean_tstat(means, lags=0)
    else:
        mean = float(np.mean(means)) if means.size else float("nan")
        t_stat = p_value = float("nan")
    return {
        "cluster": "security_id",
        "se_kernel": "iid_across_names",
        "n_events": n_events,
        "n_names": int(means.size),
        "sample_adequate": sample_adequate,
        "mean_excess_bps": float(mean * 1e4),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def two_way_clustered_event_mean(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    min_names: int,
    min_events: int,
    n_boot: int = 199,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Cameron–Gelbach–Miller two-way clustered t on event-level signed excess.

    Date HAC compresses names; name-clustered t compresses dates. This uses
    every event observation and clusters by both ``event_time`` and
    ``security_id``. Idle calendar zeros are not events and are not included.
    """
    required = ("security_id", "event_time", signal, target)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for two-way clustered t: {missing}")
    subset = (
        frame.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .select(
            "security_id",
            "event_time",
            (pl.col(signal).sign() * pl.col(target)).alias("_signed_excess"),
        )
        .drop_nulls()
    )
    if subset.height == 0:
        return {
            "cluster": "event_time_and_security_id",
            "se_kernel": "cameron_gelbach_miller",
            "inference_index": "event_rows_not_calendar_zeros",
            "n_events": 0,
            "n_names": 0,
            "n_dates": 0,
            "sample_adequate": False,
            "mean_excess_bps": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "wild_bootstrap_p": float("nan"),
        }
    y = subset["_signed_excess"].to_numpy().astype(float)
    finite = np.isfinite(y)
    y = y[finite]
    names = subset["security_id"].to_numpy()[finite]
    dates = subset["event_time"].to_numpy()[finite]
    n_events = int(y.size)
    n_names = int(np.unique(names).size)
    n_dates = int(np.unique(dates).size)
    sample_adequate = bool(n_events >= min_events and n_names >= min_names and n_dates >= 2)
    mean = float(np.mean(y)) if n_events else float("nan")
    wild_p = float("nan")
    if sample_adequate:
        mean, t_stat, p_value, _n, _na, _nb = two_way_clustered_mean_tstat(y, dates, names)
        wild_p = wild_cluster_bootstrap_two_way_p(
            y,
            dates,
            names,
            observed_t=float(t_stat),
            n_boot=int(n_boot),
            seed=int(seed),
        )
    else:
        t_stat = p_value = float("nan")
    return {
        "cluster": "event_time_and_security_id",
        "se_kernel": "cameron_gelbach_miller",
        "inference_index": "event_rows_not_calendar_zeros",
        "n_events": n_events,
        "n_names": n_names,
        "n_dates": n_dates,
        "sample_adequate": sample_adequate,
        "mean_excess_bps": float(mean * 1e4) if math.isfinite(mean) else float("nan"),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "wild_bootstrap_p": float(wild_p),
    }


def overnight_gap_event_mean(
    frame: pl.DataFrame,
    *,
    signal: str,
    min_events: int,
    min_dates: int,
) -> dict[str, float | int | str]:
    """Calendar HAC on signed close-to-next-open excess.

    The executable study enters at ``O_{t+1}``. Close-to-close ICs still load
    the overnight jump ``O_{t+1}/C_t - 1``. This is a disclosure of that
    untradeable piece, not an edge claim.
    """
    required = ("security_id", "event_time", signal, "sweep_overnight_excess")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for overnight gap: {missing}")
    subset = (
        frame.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .select(
            "event_time",
            (pl.col(signal).sign() * pl.col("sweep_overnight_excess")).alias("_signed"),
        )
        .drop_nulls()
    )
    if subset.height == 0:
        return {
            "method": "event_close_to_next_open",
            "n_events": 0,
            "n_dates": 0,
            "n_calendar_dates": 0,
            "sample_adequate": False,
            "mean_gap_bps": float("nan"),
            "hac_t": float("nan"),
            "p_value": float("nan"),
            "inference_index": "calendar_including_idle_zeros",
        }
    daily = subset.group_by("event_time", maintain_order=True).agg(pl.col("_signed").mean())
    event_series = daily["_signed"].to_numpy().astype(float)
    event_series = event_series[np.isfinite(event_series)]
    cal = _embed_value_on_calendar(daily, _panel_calendar(frame), "_signed")
    n_events = int(subset.height)
    sample_adequate = bool(n_events >= min_events and event_series.size >= min_dates)
    mean = float(np.mean(event_series)) if event_series.size else float("nan")
    if sample_adequate:
        _mu, t_stat, p_value = mean_tstat(cal, lags=1)
    else:
        t_stat = p_value = float("nan")
    return {
        "method": "event_close_to_next_open",
        "n_events": n_events,
        "n_dates": int(event_series.size),
        "n_calendar_dates": int(cal.size),
        "sample_adequate": sample_adequate,
        "mean_gap_bps": float(mean * 1e4) if math.isfinite(mean) else float("nan"),
        "hac_t": float(t_stat),
        "p_value": float(p_value),
        "inference_index": "calendar_including_idle_zeros",
    }


def out_of_time_holdout(
    frame: pl.DataFrame,
    *,
    signal: str,
    target: str,
    n_folds: int,
    min_events: int,
    min_dates: int,
) -> dict[str, float | int | str | bool]:
    """Last chronological ``1/n_folds`` of event dates as a frozen holdout.

    In-sample is the earlier complement. The bound is same-sign of the two
    means when both slices clear the event/date floors — not a p-hacked
    re-estimation of lookback on the holdout.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    subset = (
        frame.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .select(
            "event_time",
            (pl.col(signal).sign() * pl.col(target)).alias("_signed_excess"),
        )
        .drop_nulls()
    )
    empty: dict[str, float | int | str | bool] = {
        "split": "last_fold_holdout",
        "n_folds": int(n_folds),
        "n_insample_dates": 0,
        "n_holdout_dates": 0,
        "n_insample_events": 0,
        "n_holdout_events": 0,
        "sample_adequate": False,
        "insample_mean_bps": float("nan"),
        "holdout_mean_bps": float("nan"),
        "same_sign": float("nan"),
    }
    if subset.height == 0:
        return empty
    daily = (
        subset.group_by("event_time", maintain_order=True)
        .agg(
            pl.col("_signed_excess").mean().alias("_daily"),
            pl.len().alias("_n_events"),
        )
        .sort("event_time")
    )
    n_dates = int(daily.height)
    holdout_n = max(1, n_dates // n_folds)
    split_at = n_dates - holdout_n
    if split_at < 1:
        return empty
    insample = daily[:split_at]
    holdout = daily[split_at:]
    is_series = insample["_daily"].to_numpy().astype(float)
    oot_series = holdout["_daily"].to_numpy().astype(float)
    is_series = is_series[np.isfinite(is_series)]
    oot_series = oot_series[np.isfinite(oot_series)]
    n_is_events = int(insample["_n_events"].sum())
    n_oot_events = int(holdout["_n_events"].sum())
    is_adequate = bool(n_is_events >= min_events and is_series.size >= min_dates)
    oot_adequate = bool(n_oot_events >= min_events and oot_series.size >= min_dates)
    sample_adequate = bool(is_adequate and oot_adequate)
    is_mean = float(np.mean(is_series)) if is_series.size else float("nan")
    oot_mean = float(np.mean(oot_series)) if oot_series.size else float("nan")
    if sample_adequate and math.isfinite(is_mean) and math.isfinite(oot_mean):
        same_sign = 1.0 if (is_mean * oot_mean) > 0.0 else 0.0
    else:
        same_sign = float("nan")
        is_mean = is_mean if is_adequate else float("nan")
        oot_mean = oot_mean if oot_adequate else float("nan")
    return {
        "split": "last_fold_holdout",
        "n_folds": int(n_folds),
        "n_insample_dates": int(is_series.size),
        "n_holdout_dates": int(oot_series.size),
        "n_insample_events": n_is_events,
        "n_holdout_events": n_oot_events,
        "sample_adequate": sample_adequate,
        "insample_mean_bps": float(is_mean * 1e4) if math.isfinite(is_mean) else float("nan"),
        "holdout_mean_bps": float(oot_mean * 1e4) if math.isfinite(oot_mean) else float("nan"),
        "same_sign": same_sign,
    }


def event_adv_participation(
    frame: pl.DataFrame,
    config: AppConfig,
) -> dict[str, float | int | str]:
    """How large a modeled fill is versus lagged ADV on sweep-event rows.

    Trade notional is ``participation_rate * close * volume`` on the event bar;
    lagged ADV is the PIT rolling mean of dollar volume. This is a crowding
    disclosure, not a live capacity claim.
    """
    required = ("security_id", "event_time", "close", "volume", "sweep_high", "sweep_low")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing columns for event ADV participation: {missing}")
    lookback = max(3, int(config.northset.sweep_vol_lookback))
    rate = float(config.execution.participation_rate)
    tagged = (
        frame.sort(["security_id", "event_time"])
        .with_columns((pl.col("close") * pl.col("volume")).alias("_dvol"))
        .with_columns(
            pl.col("_dvol")
            .shift(1)
            .rolling_mean(window_size=lookback, min_samples=lookback)
            .over("security_id")
            .alias("_lagged_adv")
        )
    )
    events = tagged.filter(
        (pl.col("sweep_high").fill_null(0.0) == 1.0) | (pl.col("sweep_low").fill_null(0.0) == 1.0)
    ).with_columns((pl.lit(rate) * pl.col("_dvol") / pl.col("_lagged_adv")).alias("_participation"))
    values = events["_participation"].to_numpy().astype(float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return {
            "method": "participation_rate_times_event_dvol_over_lagged_adv",
            "n_events": 0,
            "median_participation": float("nan"),
            "p95_participation": float("nan"),
            "share_above_ten_pct_adv": float("nan"),
        }
    return {
        "method": "participation_rate_times_event_dvol_over_lagged_adv",
        "n_events": int(values.size),
        "median_participation": float(np.median(values)),
        "p95_participation": float(np.quantile(values, 0.95)),
        "share_above_ten_pct_adv": float(np.mean(values > 0.10)),
    }


def lead_lag_diagnostics(frame: pl.DataFrame, *, signal: str) -> dict[str, float | int | str]:
    """Negative-control lead test: today's signal vs *yesterday's* excess return.

    A leak-free pipeline may still show loading here (sweeps are momentum-built
    events); the number characterizes pre-trend exposure and catches gross
    lookahead wiring bugs. It is a disclosure, not an edge claim.
    """
    lagged = frame.sort(["security_id", "event_time"]).with_columns(
        (
            pl.col("close").shift(1).over("security_id")
            / pl.col("close").shift(2).over("security_id")
            - 1.0
        ).alias("_prev_ret")
    )
    lagged = lagged.with_columns(
        (pl.col("_prev_ret") - pl.col("_prev_ret").mean().over("event_time")).alias("_prev_excess")
    )
    subset = (
        lagged.filter(pl.col(signal).is_not_null() & (pl.col(signal) != 0.0))
        .select(
            "event_time",
            (pl.col(signal).sign() * pl.col("_prev_excess")).alias("_signed_lead"),
        )
        .drop_nulls()
    )
    if subset.height == 0:
        return {
            "diagnostic": "signal_vs_prior_bar_excess_return",
            "n_events": 0,
            "n_dates": 0,
            "mean_lead_bps": float("nan"),
            "hac_t": float("nan"),
            "p_value": float("nan"),
        }
    daily = subset.group_by("event_time", maintain_order=True).agg(pl.col("_signed_lead").mean())
    series = daily["_signed_lead"].to_numpy().astype(float)
    series = series[np.isfinite(series)]
    cal = _embed_value_on_calendar(daily, _panel_calendar(frame), "_signed_lead")
    mean = float(np.mean(series)) if series.size else float("nan")
    _mu, t_stat, p_value = mean_tstat(cal, lags=1)
    return {
        "diagnostic": "signal_vs_prior_bar_excess_return",
        "n_events": int(subset.height),
        "n_dates": int(series.size),
        "n_calendar_dates": int(cal.size),
        "mean_lead_bps": float(mean * 1e4),
        "hac_t": float(t_stat),
        "p_value": float(p_value),
        "inference_index": "calendar_including_idle_zeros",
    }


def event_coverage_stats(frame: pl.DataFrame) -> dict[str, Any]:
    """Event counts, breadth, concentration, and calendar coverage disclosure."""
    out: dict[str, Any] = {
        "n_high_events": 0,
        "n_low_events": 0,
        "n_both_ambiguous": 0,
        "n_unique_event_securities": 0,
        "top_security_event_share": float("nan"),
        "n_event_dates": 0,
        "n_calendar_dates": int(frame["event_time"].n_unique()),
        "n_ohlc_quarantined": int((~frame["ohlc_ok"]).sum()) if "ohlc_ok" in frame.columns else 0,
    }
    events = frame.filter(
        (pl.col("sweep_high").fill_null(0.0) == 1.0) | (pl.col("sweep_low").fill_null(0.0) == 1.0)
    )
    out["n_high_events"] = int((frame["sweep_high"].fill_null(0.0) == 1.0).sum())
    out["n_low_events"] = int((frame["sweep_low"].fill_null(0.0) == 1.0).sum())
    if "sweep_both" in frame.columns:
        out["n_both_ambiguous"] = int((frame["sweep_both"].fill_null(0.0) == 1.0).sum())
    if events.height == 0:
        return out
    per_security = events.group_by("security_id").len().sort("len", descending=True)
    out["n_unique_event_securities"] = int(per_security.height)
    out["top_security_event_share"] = float(per_security["len"][0] / events.height)
    out["n_event_dates"] = int(events["event_time"].n_unique())
    return out


def _fold_summary(values: np.ndarray, n_folds: int, purge_bars: int) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size < n_folds or n_folds < 2:
        return {
            "n_folds": 0,
            "positive_fraction": float("nan"),
            "worst_mean_bps": float("nan"),
            "fold_means_bps": [],
        }
    means: list[float] = []
    for index, part in enumerate(np.array_split(finite, n_folds)):
        purged = part[purge_bars:] if index > 0 and purge_bars > 0 else part
        if purged.size:
            means.append(float(np.mean(purged) * 1e4))
    if len(means) != n_folds:
        return {
            "n_folds": 0,
            "positive_fraction": float("nan"),
            "worst_mean_bps": float("nan"),
            "fold_means_bps": [],
            "purge_bars": int(purge_bars),
        }
    return {
        "n_folds": int(len(means)),
        "positive_fraction": float(np.mean(np.asarray(means) > 0.0)),
        "worst_mean_bps": float(min(means)),
        "fold_means_bps": means,
        "purge_bars": int(purge_bars),
    }


def parameter_sensitivity_grid(
    sweep_frame: pl.DataFrame,
    config: AppConfig,
) -> dict[str, Any]:
    """Re-run the horizon-1 event study across detector lookbacks.

    Every cell is a *trial* and is disclosed as such; the grid exists to show
    parameter fragility, not to let the best cell become the headline.
    """
    from quant_fund.northset.sweeps import liquidity_sweep_frame

    ns = config.northset
    lookbacks = tuple(int(v) for v in ns.sweep_sensitivity_lookbacks)
    rows: list[dict[str, Any]] = []
    base_cols = [
        c
        for c in (
            "security_id",
            "event_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_open",
            "return_close",
        )
        if c in sweep_frame.columns
    ]
    base = sweep_frame.select(base_cols)
    for lookback in lookbacks:
        variant = sweep_forward_frame(
            liquidity_sweep_frame(base, lookback=lookback),
            horizons=(1,),
            vol_lookback=int(ns.sweep_vol_lookback),
        )
        variant = _deduplicate_sweep_episodes(variant, int(ns.sweep_cooldown_bars))
        variant = _attach_event_costs(variant, config)
        for signal in _SIGNALS:
            daily, _costs, n_events = _event_daily_series(
                variant, signal=signal, target="sweep_excess_ret_1"
            )
            event_series = daily[np.isfinite(daily)]
            calendar = _panel_calendar(variant)
            daily_frame, _n = _event_daily_frame(
                variant, signal=signal, target="sweep_excess_ret_1"
            )
            cal_series, _cal_costs = _embed_on_calendar(daily_frame, calendar)
            adequate = bool(
                n_events >= int(ns.sweep_min_events)
                and event_series.size >= int(ns.sweep_min_dates)
            )
            if adequate:
                mean = float(np.mean(event_series))
                _mu, t_stat, p_value = mean_tstat(cal_series, lags=1)
            else:
                mean = float(np.mean(event_series)) if event_series.size else float("nan")
                t_stat = p_value = float("nan")
            rows.append(
                {
                    "lookback": int(lookback),
                    "signal": signal,
                    "n_events": int(n_events),
                    "n_dates": int(event_series.size),
                    "n_calendar_dates": int(cal_series.size),
                    "sample_adequate": adequate,
                    "mean_excess_bps": float(mean * 1e4),
                    "hac_t": float(t_stat),
                    "p_value": float(p_value),
                    "inference_index": "calendar_including_idle_zeros",
                }
            )
    signs: dict[str, set[float]] = {s: set() for s in _SIGNALS}
    for row in rows:
        value = float(row["mean_excess_bps"])
        if math.isfinite(value) and bool(row["sample_adequate"]):
            signs[str(row["signal"])].add(math.copysign(1.0, value) if value != 0.0 else 0.0)
    return {
        "lookbacks": list(lookbacks),
        "cells": rows,
        "n_parameter_trials": len(rows),
        "sign_stable_by_signal": {
            signal: (len(observed) <= 1) for signal, observed in signs.items()
        },
        "disclosure": "every_cell_counted_as_a_trial",
    }


def sweep_evidence_battery(
    sweep_frame: pl.DataFrame,
    config: AppConfig,
) -> dict[str, Any]:
    """Run the complete institutional sweep evidence battery."""
    ns = config.northset
    horizons = tuple(int(h) for h in ns.sweep_horizons)
    frame = sweep_forward_frame(
        sweep_frame,
        horizons=horizons,
        vol_lookback=int(ns.sweep_vol_lookback),
    )
    frame = _deduplicate_sweep_episodes(frame, int(ns.sweep_cooldown_bars))
    frame = _attach_event_costs(frame, config)
    cost_bps = estimated_round_trip_cost_bps(frame, config)
    calendar = _panel_calendar(frame)
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    p_indices: list[int] = []
    for signal in _SIGNALS:
        for horizon in horizons:
            target = f"sweep_excess_ret_{horizon}"
            daily, n_events = _event_daily_frame(frame, signal=signal, target=target)
            event_vals = (
                daily["_daily"].to_numpy().astype(float)
                if daily.height
                else np.array([], dtype=float)
            )
            event_finite = event_vals[np.isfinite(event_vals)]
            cal_vals, cal_cost_bps = _embed_on_calendar(daily, calendar)
            sample_adequate = bool(
                n_events >= int(ns.sweep_min_events)
                and event_finite.size >= int(ns.sweep_min_dates)
            )
            mean = float(np.mean(event_finite)) if event_finite.size else float("nan")
            if sample_adequate:
                _mean, t_stat, p_value = mean_tstat(cal_vals, lags=max(1, horizon))
                lo, hi, _point = bootstrap_mean_ci(
                    cal_vals,
                    n_boot=int(ns.sweep_n_boot),
                    block=max(2, horizon),
                    seed=int(config.validation.seed) + horizon + len(rows) * 17,
                )
            else:
                t_stat = p_value = lo = hi = float("nan")
            cost_adjusted = cal_vals - cal_cost_bps / 1e4
            ca_mean_cal = float(np.mean(cost_adjusted)) if cost_adjusted.size else float("nan")
            # Cost-adjusted *event* mean (conditional on trading).
            if daily.height:
                event_cost = daily["_daily_cost_bps"].to_numpy().astype(float)
                mask = np.isfinite(event_vals) & np.isfinite(event_cost)
                ca_event = event_vals[mask] - event_cost[mask] / 1e4
                ca_mean = float(np.mean(ca_event)) if ca_event.size else float("nan")
            else:
                ca_mean = float("nan")
            if sample_adequate:
                _ca_mean, ca_t, ca_p_two = mean_tstat(cost_adjusted, lags=max(1, horizon))
                ca_p_greater = onesided_from_twosided(ca_t, ca_p_two, greater=True)
            else:
                ca_t = ca_p_greater = float("nan")
            row: dict[str, Any] = {
                "signal": signal,
                "horizon": int(horizon),
                "entry": "next_open",
                "exit": f"close_t_plus_{horizon}",
                "control": "same_date_cross_sectional_mean",
                "inference_index": "calendar_including_idle_zeros",
                "n_events": int(n_events),
                "n_dates": int(event_finite.size),
                "n_calendar_dates": int(cal_vals.size),
                "sample_adequate": sample_adequate,
                "min_events": int(ns.sweep_min_events),
                "min_dates": int(ns.sweep_min_dates),
                "mean_excess_bps": float(mean * 1e4),
                "calendar_mean_excess_bps": float(float(np.mean(cal_vals)) * 1e4)
                if cal_vals.size
                else float("nan"),
                "hac_t": float(t_stat),
                "p_value": float(p_value),
                "bootstrap_lo_bps": float(lo * 1e4),
                "bootstrap_hi_bps": float(hi * 1e4),
                "hit_rate": float(np.mean(event_finite > 0.0))
                if event_finite.size
                else float("nan"),
                "round_trip_cost_bps": float(cost_bps),
                "cost_adjusted_mean_bps": float(ca_mean * 1e4),
                "cost_adjusted_hac_t": float(ca_t),
                "cost_adjusted_p_greater": float(ca_p_greater),
                "calendar_cost_adjusted_mean_bps": float(ca_mean_cal * 1e4)
                if math.isfinite(ca_mean_cal)
                else float("nan"),
                **(
                    _fold_summary(cal_vals, int(ns.sweep_n_folds), horizon)
                    if sample_adequate
                    else {
                        "n_folds": 0,
                        "positive_fraction": float("nan"),
                        "worst_mean_bps": float("nan"),
                        "fold_means_bps": [],
                    }
                ),
                "reject_fdr": False,
            }
            rows.append(row)
            if math.isfinite(p_value):
                p_indices.append(len(rows) - 1)
                p_values.append(float(p_value))
    fdr_cutoff = 0.0
    if p_values:
        rejected, fdr_cutoff = benjamini_hochberg(np.asarray(p_values), alpha=0.05)
        for idx, reject in zip(p_indices, rejected, strict=True):
            rows[idx]["reject_fdr"] = bool(reject)

    regime_rows: list[dict[str, Any]] = []
    for signal in _SIGNALS:
        for regime in ("low", "high"):
            daily, n_events = _event_daily_frame(
                frame,
                signal=signal,
                target="sweep_excess_ret_1",
                regime=regime,
            )
            event_vals = (
                daily["_daily"].to_numpy().astype(float)
                if daily.height
                else np.array([], dtype=float)
            )
            event_finite = event_vals[np.isfinite(event_vals)]
            cal_vals, _costs = _embed_on_calendar(daily, calendar)
            event_mean = float(np.mean(event_finite)) if event_finite.size else float("nan")
            sample_adequate = bool(
                n_events >= int(ns.sweep_min_events)
                and event_finite.size >= int(ns.sweep_min_dates)
            )
            if sample_adequate:
                _mu, t_stat, p_value = mean_tstat(cal_vals, lags=1)
            else:
                t_stat = p_value = float("nan")
            regime_rows.append(
                {
                    "signal": signal,
                    "regime": regime,
                    "n_events": int(n_events),
                    "n_dates": int(event_finite.size),
                    "n_calendar_dates": int(cal_vals.size),
                    "sample_adequate": sample_adequate,
                    "mean_excess_bps": float(event_mean * 1e4),
                    "hac_t": float(t_stat),
                    "p_value": float(p_value),
                    "inference_index": "calendar_including_idle_zeros",
                }
            )

    placebos: dict[str, dict[str, float | int]] = {}
    for i, signal in enumerate(_SIGNALS):
        placebos[signal] = within_date_permutation_test(
            frame,
            score=signal,
            target="sweep_excess_ret_1",
            min_names=int(ns.min_names),
            n_permutations=int(ns.sweep_n_permutations),
            seed=int(config.validation.seed) + 1009 * (i + 1),
        )

    matched_controls = {
        signal: matched_control_difference(
            frame,
            signal=signal,
            target="sweep_excess_ret_1",
            min_events=int(ns.sweep_min_events),
            min_dates=int(ns.sweep_min_dates),
        )
        for signal in _SIGNALS
    }
    liquidity_matched_controls = {
        signal: liquidity_matched_control_difference(
            frame,
            signal=signal,
            target="sweep_excess_ret_1",
            min_events=int(ns.sweep_min_events),
            min_dates=int(ns.sweep_min_dates),
        )
        for signal in _SIGNALS
    }
    name_clustered = {
        signal: name_clustered_event_mean(
            frame,
            signal=signal,
            target="sweep_excess_ret_1",
            min_names=int(ns.min_names),
            min_events=int(ns.sweep_min_events),
        )
        for signal in _SIGNALS
    }
    two_way_clustered = {
        signal: two_way_clustered_event_mean(
            frame,
            signal=signal,
            target="sweep_excess_ret_1",
            min_names=int(ns.min_names),
            min_events=int(ns.sweep_min_events),
            n_boot=min(199, max(20, int(ns.sweep_n_boot))),
            seed=int(config.validation.seed) + 50,
        )
        for signal in _SIGNALS
    }
    overnight_gaps = {
        signal: overnight_gap_event_mean(
            frame,
            signal=signal,
            min_events=int(ns.sweep_min_events),
            min_dates=int(ns.sweep_min_dates),
        )
        for signal in _SIGNALS
    }
    oot_holdouts = {
        signal: out_of_time_holdout(
            frame,
            signal=signal,
            target="sweep_excess_ret_1",
            n_folds=int(ns.sweep_n_folds),
            min_events=int(ns.sweep_min_events),
            min_dates=max(5, int(ns.sweep_min_dates) // 2),
        )
        for signal in _SIGNALS
    }
    lead_diagnostics = {signal: lead_lag_diagnostics(frame, signal=signal) for signal in _SIGNALS}
    sensitivity = parameter_sensitivity_grid(sweep_frame, config)
    adv_participation = event_adv_participation(frame, config)
    n_parameter_trials = int(sensitivity["n_parameter_trials"])
    n_counted_trials = (
        len(rows)
        + len(regime_rows)
        + len(placebos)
        + len(matched_controls)
        + len(liquidity_matched_controls)
        + len(name_clustered)
        + len(two_way_clustered)
        + len(overnight_gaps)
        + len(oot_holdouts)
        + len(lead_diagnostics)
        + n_parameter_trials
    )
    trial_ledger = {
        "n_event_study_cells": len(rows),
        "n_regime_cells": len(regime_rows),
        "n_placebos": len(placebos),
        "n_matched_controls": len(matched_controls),
        "n_liquidity_matched_controls": len(liquidity_matched_controls),
        "n_name_clustered": len(name_clustered),
        "n_two_way_clustered": len(two_way_clustered),
        "n_overnight_gaps": len(overnight_gaps),
        "n_oot_holdouts": len(oot_holdouts),
        "n_lead_diagnostics": len(lead_diagnostics),
        "n_parameter_trials": n_parameter_trials,
        "n_counted_trials": int(n_counted_trials),
        "fdr_family": "northset_sweep_event_studies_only",
        "disclosure": "primary_predeclared_secondaries_counted",
    }

    return {
        "timing_contract": "event_close_then_next_open",
        "control_contract": "same_date_cross_sectional_mean",
        "inference_index": "calendar_including_idle_zeros",
        "horizons": list(horizons),
        "episode_cooldown_bars": int(ns.sweep_cooldown_bars),
        "round_trip_cost_bps": float(cost_bps),
        "event_studies": rows,
        "volatility_regimes": regime_rows,
        "permutation_placebos": placebos,
        "matched_controls": matched_controls,
        "liquidity_matched_controls": liquidity_matched_controls,
        "name_clustered": name_clustered,
        "two_way_clustered": two_way_clustered,
        "overnight_gaps": overnight_gaps,
        "oot_holdouts": oot_holdouts,
        "lead_diagnostics": lead_diagnostics,
        "parameter_sensitivity": sensitivity,
        "coverage": event_coverage_stats(frame),
        "adv_participation": adv_participation,
        "primary_test": dict(PRIMARY_EXECUTABLE_TEST),
        "trial_ledger": trial_ledger,
        "fdr_cutoff": float(fdr_cutoff),
        "research_only": True,
    }
