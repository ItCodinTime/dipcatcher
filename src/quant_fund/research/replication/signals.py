"""Point-in-time signal builders for the replicated strategies.

Every function takes the canonical panel (``panel.PANEL_COLUMNS``) and returns a
long frame ``[security_id, event_time, signal]`` where ``signal`` at date ``t``
uses only information available by the close of ``t`` — it is scored against
returns that start at ``t + 1``. ``NaN`` signals mean "not enough history" and
are never scored.

Signal definitions follow the original papers:

- ``tsmom_signal`` — Moskowitz, Ooi & Pedersen (2012): ``sign(R_{t-J,t})`` of
  the past-``lookback`` cumulative return, scaled by ex-ante vol
  ``1/sigma_t`` (EWMA of squared daily returns, center of mass ``vol_com``).
- ``reversal_signal`` — Jegadeesh (1990, ``lookback=21``) and Lehmann (1990,
  ``lookback=5``): negated past-``lookback`` cumulative log return.
- ``low_volatility_signal`` — Baker, Bradley & Wurgler (2011): negated
  trailing-``lookback`` daily-return standard deviation (long low-vol).
- ``bab_signal`` — Frazzini & Pedersen (2014): negated shrunk beta
  ``-(w * beta_hat + (1-w) * 1)`` with ``beta_hat = corr_i,M * sigma_i/sigma_M``
  on daily returns vs the equal-weight market (long low-beta).
- ``overnight_signal`` / ``tug_of_war_signal`` — Lou, Polk & Skouras (2019):
  cumulative overnight log return over ``lookback`` days, resp. the
  overnight-minus-intraday difference ("tug of war").
- ``turn_of_month_mask`` / ``turn_of_month_signal`` — McConnell & Xu (2008):
  market-level flag for the last trading day of the month through the first
  three trading days of the next month (Lakonishok & Smidt 1988 -1..+3
  convention; same across assets, evaluated as a market-timing signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

SIGNAL_COLUMNS: tuple[str, ...] = ("security_id", "event_time", "signal")


def _log_cum(col: str, lookback: int) -> pl.Expr:
    """Rolling sum of log1p over ``lookback`` bars (null until full window)."""
    return pl.col(col).log1p().rolling_sum(window_size=lookback, min_samples=lookback)


def _require_lookback(lookback: int) -> int:
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    return int(lookback)


def _sorted_panel(frame: pl.DataFrame) -> pl.DataFrame:
    if "security_id" not in frame.columns or "event_time" not in frame.columns:
        raise ValueError("panel needs security_id and event_time")
    return frame.sort(["security_id", "event_time"])


def _signal_frame(frame: pl.DataFrame, signal_expr: pl.Expr) -> pl.DataFrame:
    return frame.with_columns(signal_expr.alias("signal")).select(SIGNAL_COLUMNS)


def tsmom_signal(frame: pl.DataFrame, lookback: int = 252, vol_com: int = 60) -> pl.DataFrame:
    """Moskowitz-Ooi-Pedersen (2012) TSMOM: ``sign(R_J) / sigma_ex_ante``.

    The paper uses a 12-month lookback and scales each position by an
    exponentially weighted estimate of ex-ante volatility (center of mass 60
    days). ``vol_com`` is the pandas ``com`` convention, so span = 2*com+1.
    """
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    mom = _log_cum("ret_cc", lookback).over("security_id")
    var = (
        pl.col("ret_cc").pow(2).ewm_mean(com=float(vol_com), ignore_nulls=True).over("security_id")
    )
    signal = pl.when(mom.is_not_null() & (var > 0.0)).then(mom.sign() / var.sqrt()).otherwise(None)
    return _signal_frame(frame, signal)


def reversal_signal(frame: pl.DataFrame, lookback: int = 21) -> pl.DataFrame:
    """Jegadeesh (1990) / Lehmann (1990): negated past-``lookback`` return."""
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    return _signal_frame(frame, (-_log_cum("ret_cc", lookback)).over("security_id"))


def low_volatility_signal(frame: pl.DataFrame, lookback: int = 252) -> pl.DataFrame:
    """Baker-Bradley-Wurgler (2011): negated trailing daily-return volatility."""
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    vol = pl.col("ret_cc").rolling_std(window_size=lookback, min_samples=lookback)
    return _signal_frame(frame, (-vol).over("security_id"))


def bab_signal(
    frame: pl.DataFrame,
    market: pl.DataFrame,
    lookback: int = 252,
    shrink_weight: float = 0.6,
    shrink_target: float = 1.0,
) -> pl.DataFrame:
    """Frazzini-Pedersen (2014): negated shrunk beta ``beta_ts``.

    ``beta_hat = corr_hat * sigma_i / sigma_M`` from daily returns over the
    trailing ``lookback`` window (FP use ~1 year of daily data), shrunk toward
    1: ``beta_ts = shrink_weight * beta_hat + (1 - shrink_weight) * shrink_target``
    (FP use 0.6/0.4 toward 1.0). ``market`` needs ``event_time, mkt_ret_cc``.
    """
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    merged = (
        frame.select("security_id", "event_time", "ret_cc")
        .join(market.select("event_time", "mkt_ret_cc"), on="event_time", how="left")
        .sort(["security_id", "event_time"])
    )
    # Rolling corr/beta is built in pandas for clarity and correctness.
    out_parts: list[pl.DataFrame] = []
    for _key, part in merged.group_by("security_id", maintain_order=True):
        pdf = part.to_pandas()
        r = pdf["ret_cc"]
        m = pdf["mkt_ret_cc"]
        rolling_cov = r.rolling(lookback, min_periods=lookback).cov(m)
        rolling_var_m = m.rolling(lookback, min_periods=lookback).var()
        sigma_i = r.rolling(lookback, min_periods=lookback).std()
        sigma_m = np.sqrt(rolling_var_m)
        corr = rolling_cov / (sigma_i * sigma_m)
        beta_hat = corr * (sigma_i / sigma_m)
        beta_ts = float(shrink_weight) * beta_hat + (1.0 - float(shrink_weight)) * float(
            shrink_target
        )
        out_parts.append(
            part.select("security_id", "event_time").with_columns(
                pl.Series("signal", (-beta_ts).to_numpy())
            )
        )
    if not out_parts:
        return pl.DataFrame(
            schema={"security_id": pl.String, "event_time": pl.Datetime, "signal": pl.Float64}
        )
    return pl.concat(out_parts)


def overnight_signal(frame: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """Lou-Polk-Skouras (2019): cumulative overnight log return, ``lookback`` days."""
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    return _signal_frame(frame, _log_cum("ret_on", lookback).over("security_id"))


def tug_of_war_signal(frame: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
    """LPS (2019) tug-of-war: cum overnight minus cum intraday log return."""
    lookback = _require_lookback(lookback)
    frame = _sorted_panel(frame)
    on = _log_cum("ret_on", lookback).over("security_id")
    intra = _log_cum("ret_id", lookback).over("security_id")
    return _signal_frame(frame, (on - intra))


def turn_of_month_mask(dates: np.ndarray) -> np.ndarray:
    """Boolean mask of McConnell-Xu (2008) turn-of-month days on a calendar.

    Window = the last trading day of each month plus the first three trading
    days of the following month (the ``-1..+3`` convention). Computed purely
    from the observed trading calendar: ``last`` = a day whose successor sits
    in a different month; ``first3`` = position <= 2 within a month group,
    guarded so a panel that starts mid-month does not flag late-start days.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).sort_values()
    if idx.empty:
        return np.zeros(0, dtype=bool)
    month = idx.to_period("M").to_numpy()
    # last observed trading day of each month: next bar is a different month
    is_last = np.empty(len(idx), dtype=bool)
    is_last[:-1] = month[1:] != month[:-1]
    # The panel's final day is flagged as month-end only if it is plausibly a
    # true month-end (last trading day is always the 28th-31st); otherwise a
    # panel that simply ends mid-month would mint a false ToM day.
    is_last[-1] = idx[-1].day >= 20
    # position within each month group
    pos = pd.Series(np.arange(len(idx))).groupby(month).cumcount().to_numpy()
    first_day_dom = pd.Series(idx.day).groupby(month).transform("min").to_numpy()
    # Only trust first-3 when the group plausibly contains the month start.
    first3 = (pos <= 2) & (first_day_dom <= 7)
    return np.asarray(is_last | first3, dtype=bool)


def turn_of_month_signal(frame: pl.DataFrame) -> pl.DataFrame:
    """Broadcast the ToM flag across assets: same ``0/1`` signal for all names."""
    frame = _sorted_panel(frame)
    dates = frame.select("event_time").unique(maintain_order=True).sort("event_time")
    mask = turn_of_month_mask(dates["event_time"].to_numpy())
    flags = dates.with_columns(pl.Series("signal", mask.astype(np.float64)))
    return frame.select("security_id", "event_time").join(flags, on="event_time", how="left")
