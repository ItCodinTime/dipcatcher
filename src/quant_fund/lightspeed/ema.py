"""Causal moving averages and realized vol used by the frozen Lightspeed books."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-12


def sma_seeded_ema(closes: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Causal EMA with SMA seed; NaN until the seed bar. Matches Lightspeed lab."""
    values = np.asarray(closes, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    window = int(period)
    if len(values) < window or window <= 0:
        return out
    out[window - 1] = float(np.mean(values[:window]))
    alpha = 2.0 / (window + 1.0)
    for i in range(window, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Causal rolling mean, NaN-padded; window ends at t inclusive."""
    series = np.asarray(values, dtype=float)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    width = int(window)
    if n < width or width <= 0:
        return out
    csum = np.cumsum(np.insert(series, 0, 0.0))
    out[width - 1 :] = (csum[width:] - csum[:-width]) / width
    return out


def rolling_std(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Population rolling std, NaN-padded. Causal: window ends at t inclusive."""
    series = np.asarray(values, dtype=float)
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    width = int(window)
    if n < width or width <= 1:
        return out
    csum = np.cumsum(np.insert(series, 0, 0.0))
    csum2 = np.cumsum(np.insert(series * series, 0, 0.0))
    sums = csum[width:] - csum[:-width]
    sums2 = csum2[width:] - csum2[:-width]
    means = sums / width
    var = np.maximum(sums2 / width - means * means, 0.0)
    out[width - 1 :] = np.sqrt(var)
    return out


def realized_ann_vol_from_returns(
    returns: NDArray[np.float64],
    window: int,
    periods_per_year: float = 252.0,
) -> NDArray[np.float64]:
    return rolling_std(returns, window) * math.sqrt(float(periods_per_year))


def realized_ann_vol_from_closes(
    closes: NDArray[np.float64],
    window: int,
    periods_per_year: float = 252.0,
) -> NDArray[np.float64]:
    """Causal annualized vol of simple daily returns, NaN-padded."""
    prices = np.asarray(closes, dtype=float)
    n = len(prices)
    rets = np.zeros(n, dtype=float)
    rets[1:] = prices[1:] / np.maximum(prices[:-1], _EPS) - 1.0
    out = np.full(n, np.nan, dtype=float)
    width = int(window)
    if n < width + 1 or width <= 1:
        return out
    csum = np.cumsum(np.insert(rets, 0, 0.0))
    csum2 = np.cumsum(np.insert(rets * rets, 0, 0.0))
    sums = csum[width:] - csum[:-width]
    sums2 = csum2[width:] - csum2[:-width]
    means = sums / width
    var = np.maximum(sums2 / width - means * means, 0.0)
    out[width - 1 :] = np.sqrt(var) * math.sqrt(float(periods_per_year))
    return out


def quantize(weight: float, quantum: float) -> float:
    if quantum <= 0:
        return float(weight)
    return float(round(weight / quantum) * quantum)


def clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def delay_weights(weights: NDArray[np.float64], sessions: int) -> NDArray[np.float64]:
    series = np.asarray(weights, dtype=float)
    if int(sessions) <= 0:
        return series
    out = np.zeros_like(series)
    out[int(sessions) :] = series[: -int(sessions)]
    return out
