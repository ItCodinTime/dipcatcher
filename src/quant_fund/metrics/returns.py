"""Return and risk-adjusted performance. See docs/MATH_SPEC.md."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def wealth_index(returns: Array) -> Array:
    r = np.asarray(returns, dtype=float)
    return np.cumprod(1.0 + r)


def drawdown_series(returns: Array) -> Array:
    w = wealth_index(returns)
    peak = np.maximum.accumulate(w)
    return w / peak - 1.0


def max_drawdown(returns: Array) -> float:
    dd = drawdown_series(returns)
    if dd.size == 0:
        return 0.0
    return float(np.min(dd))


def cagr(returns: Array, periods_per_year: float = 252.0) -> float:
    w = wealth_index(returns)
    if w.size == 0:
        return float("nan")
    years = w.size / periods_per_year
    if years <= 0 or w[-1] <= 0:
        return float("nan")
    return float(w[-1] ** (1.0 / years) - 1.0)


def annualized_vol(returns: Array, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(returns: Array, mar: float = 0.0, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    d = np.minimum(r - mar, 0.0)
    if r.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean(d**2)) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Array,
    rf_per_period: float = 0.0,
    periods_per_year: float = 252.0,
    *,
    irregular: bool = False,
) -> dict[str, float | int | bool]:
    r = np.asarray(returns, dtype=float)
    excess = r - rf_per_period
    n = int(excess.size)
    result: dict[str, float | int | bool] = {
        "n": n,
        "periods_per_year": periods_per_year,
        "annualized": not irregular,
        "flag_high_sharpe": False,
    }
    if n < 2 or np.std(excess, ddof=1) == 0:
        result["sharpe"] = float("nan")
        return result
    sr_periodic = float(np.mean(excess) / np.std(excess, ddof=1))
    sr = sr_periodic if irregular else sr_periodic * np.sqrt(periods_per_year)
    result["sharpe"] = sr
    result["flag_high_sharpe"] = bool(np.isfinite(sr) and abs(sr) > 5.0)
    return result


def sortino_ratio(returns: Array, mar: float = 0.0, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    dd = downside_deviation(r, mar=mar, periods_per_year=1.0)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float((np.mean(r) - mar) / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns: Array, periods_per_year: float = 252.0) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def turnover(weights: Array, prev_weights: Array) -> float:
    return float(np.sum(np.abs(np.asarray(weights) - np.asarray(prev_weights))))
