"""Opt-in OHLC spread estimators for transaction-cost calibration.

Default backtest costs use a flat ``half_spread_bps``. When a caller opts into
``CostConfig.spread_estimator`` ∈ {corwin_schultz, abdi_ranaldo, roll}, the
reference event loop estimates a relative full spread from OHLC history known
at the signal close, converts it to half-spread bps, and charges

    max(flat_half_spread_bps, calibrated_half_spread_bps)

so the configured flat half-spread is always a floor. Insufficient / undefined
estimates fall back to the floor (never invent a cheaper cost).

``run_backtest_fast`` refuses the calibrated path fail-closed — the vectorized
kernel only reproduces the flat half-spread surface.

References:
- Roll (1984) implied spread from first-order return autocovariance.
- Corwin & Schultz (2012) high-low two-day spread.
- Abdi & Ranaldo (2017) close-high-low spread.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

Array = NDArray[np.float64]

SpreadEstimatorName = Literal["flat", "corwin_schultz", "abdi_ranaldo", "roll"]
SPREAD_ESTIMATORS: tuple[str, ...] = ("flat", "corwin_schultz", "abdi_ranaldo", "roll")
CALIBRATED_SPREAD_ESTIMATORS: tuple[str, ...] = ("corwin_schultz", "abdi_ranaldo", "roll")

_CS_DEN = 3.0 - 2.0 * math.sqrt(2.0)
_FLOOR = 1e-12


def is_calibrated_spread_estimator(name: str) -> bool:
    return str(name) in CALIBRATED_SPREAD_ESTIMATORS


def relative_to_half_spread_bps(relative_full_spread: float) -> float:
    """Convert a relative full spread (fraction of price) to half-spread bps."""
    if not np.isfinite(relative_full_spread) or relative_full_spread < 0.0:
        return float("nan")
    return float(0.5 * relative_full_spread * 1e4)


def floored_half_spread_bps(floor_bps: float, relative_full_spread: float) -> float:
    """Effective half-spread bps: flat floor, never below it.

    Non-finite / negative calibrated estimates fall back to ``floor_bps``.
    """
    if not np.isfinite(floor_bps) or floor_bps < 0.0:
        raise ValueError("floor_bps must be finite and non-negative")
    calibrated = relative_to_half_spread_bps(relative_full_spread)
    if not np.isfinite(calibrated) or calibrated < 0.0:
        return float(floor_bps)
    return float(max(floor_bps, calibrated))


def roll_relative_spread(close: Array) -> float:
    """Roll (1984) relative spread ``2√(-γ₁) / mid`` from close levels.

    Positive lag-1 autocovariance of price changes → undefined → NaN.
    """
    px = np.asarray(close, dtype=float).reshape(-1)
    px = px[np.isfinite(px) & (px > 0.0)]
    if px.size < 12:
        return float("nan")
    dp = np.diff(px)
    a = dp[1:]
    b = dp[:-1]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 10:
        return float("nan")
    gamma = float(np.cov(a[mask], b[mask], ddof=0)[0, 1])
    if not math.isfinite(gamma) or gamma >= 0.0:
        return float("nan")
    abs_spread = 2.0 * math.sqrt(-gamma)
    mid = float(np.mean(px))
    if mid <= _FLOOR:
        return float("nan")
    return float(abs_spread / mid)


def corwin_schultz_relative_spread(high: Array, low: Array) -> float:
    """Corwin–Schultz (2012) mean relative spread over PIT pairs ``(t-1, t)``.

    Negative pair estimates are clipped to zero before averaging (paper
    convention); dropping them would upward-bias the mean.
    """
    h = np.asarray(high, dtype=float).reshape(-1)
    lo = np.asarray(low, dtype=float).reshape(-1)
    if h.size != lo.size or h.size < 2:
        return float("nan")
    ok = np.isfinite(h) & np.isfinite(lo) & (h > 0.0) & (lo > 0.0) & (h >= lo)
    if int(ok.sum()) < 2:
        return float("nan")
    h = h[ok]
    lo = lo[ok]
    if h.size < 2:
        return float("nan")
    # PIT pairs (t-1, t): use consecutive valid bars after the filter.
    h0, h1 = h[:-1], h[1:]
    l0, l1 = lo[:-1], lo[1:]
    beta = (np.log(h1 / l1) ** 2) + (np.log(h0 / l0) ** 2)
    h2 = np.maximum(h1, h0)
    l2 = np.minimum(l1, l0)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DEN - np.sqrt(
        np.clip(gamma, 0.0, None) / _CS_DEN
    )
    exp_a = np.exp(alpha)
    spread = 2.0 * (exp_a - 1.0) / (1.0 + exp_a)
    finite = np.isfinite(spread)
    if int(finite.sum()) < 1:
        return float("nan")
    return float(np.mean(np.clip(spread[finite], 0.0, None)))


def abdi_ranaldo_relative_spread(high: Array, low: Array, close: Array) -> float:
    """Abdi–Ranaldo (2017) mean relative close-high-low spread."""
    h = np.asarray(high, dtype=float).reshape(-1)
    lo = np.asarray(low, dtype=float).reshape(-1)
    c = np.asarray(close, dtype=float).reshape(-1)
    if h.size != lo.size or h.size != c.size or h.size < 2:
        return float("nan")
    m = 0.5 * (h + lo)
    ok = (
        np.isfinite(h)
        & np.isfinite(lo)
        & np.isfinite(c)
        & np.isfinite(m)
        & (h > 0.0)
        & (lo > 0.0)
        & (c > 0.0)
        & (h >= lo)
    )
    if int(ok.sum()) < 2:
        return float("nan")
    h = h[ok]
    lo = lo[ok]
    c = c[ok]
    m = m[ok]
    if c.size < 2:
        return float("nan")
    eta = (c[1:] - m[1:]) * (c[:-1] - m[:-1])
    spread = 2.0 * np.sqrt(np.clip(eta, 0.0, None))
    rel = spread / c[1:]
    finite = np.isfinite(rel) & (rel >= 0.0) & (rel < 1.0)
    if int(finite.sum()) < 1:
        return float("nan")
    return float(np.mean(rel[finite]))


def estimate_relative_spread(
    estimator: str,
    *,
    high: Array,
    low: Array,
    close: Array,
) -> float:
    """Dispatch a relative full-spread estimate for one OHLC window."""
    name = str(estimator)
    if name == "corwin_schultz":
        return corwin_schultz_relative_spread(high, low)
    if name == "abdi_ranaldo":
        return abdi_ranaldo_relative_spread(high, low, close)
    if name == "roll":
        return roll_relative_spread(close)
    raise ValueError(
        f"unknown spread estimator {name!r}; expected one of {CALIBRATED_SPREAD_ESTIMATORS}"
    )


def pit_calibrated_half_spread_bps(
    bars: pl.DataFrame,
    *,
    estimator: str,
    lookback: int,
    floor_bps: float,
) -> dict[tuple[str, datetime], float]:
    """PIT map ``(security_id, event_time) → floored half-spread bps``.

    The estimate at date ``t`` uses only bars of that name with
    ``event_time <= t``, truncated to the trailing ``lookback`` rows.
    Missing OHLC or an undefined estimator falls back to ``floor_bps``.
    """
    if str(estimator) not in CALIBRATED_SPREAD_ESTIMATORS:
        raise ValueError(f"spread estimator {estimator!r} is not a calibrated OHLC estimator")
    if type(lookback) is not int or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    if not np.isfinite(floor_bps) or floor_bps < 0.0:
        raise ValueError("floor_bps must be finite and non-negative")
    required = ("security_id", "event_time", "high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing columns for spread calibration: {missing}")

    frame = bars.sort(["security_id", "event_time"])
    out: dict[tuple[str, datetime], float] = {}
    for key, group in frame.group_by("security_id", maintain_order=True):
        security = str(key[0] if isinstance(key, tuple) else key)
        times = group["event_time"].to_list()
        high = group["high"].to_numpy().astype(float)
        low = group["low"].to_numpy().astype(float)
        close = group["close"].to_numpy().astype(float)
        for i, et in enumerate(times):
            start = max(0, i + 1 - lookback)
            rel = estimate_relative_spread(
                estimator,
                high=high[start : i + 1],
                low=low[start : i + 1],
                close=close[start : i + 1],
            )
            out[(security, et)] = floored_half_spread_bps(floor_bps, rel)
    return out


def require_ohlc_for_calibration(bars: pl.DataFrame, estimator: str) -> None:
    """Fail closed when calibrated costs are requested without OHLC columns."""
    if not is_calibrated_spread_estimator(estimator):
        return
    required = ("high", "low", "close")
    missing = [c for c in required if c not in bars.columns]
    if missing:
        raise ValueError(
            f"spread calibration requires OHLC columns {list(required)}; missing {missing}"
        )
