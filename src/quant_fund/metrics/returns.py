"""Return and risk-adjusted performance. See docs/MATH_SPEC.md."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _validate_periods(periods_per_year: float) -> None:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")


def wealth_index(returns: Array) -> Array:
    """Cumulative wealth path starting at 1.0.

    Empty input → empty array. Non-finite returns propagate as honest NaN
    (no silent zero-fill). Not a live P&L claim.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return np.asarray([], dtype=float)
    return np.cumprod(1.0 + r)


def drawdown_series(returns: Array) -> Array:
    w = wealth_index(returns)
    if w.size == 0:
        return np.asarray([], dtype=float)
    # Guard divide-by-zero / non-finite peaks → honest NaN path
    peak = np.maximum.accumulate(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = w / peak - 1.0
    return dd


def max_drawdown(returns: Array) -> float:
    dd = drawdown_series(returns)
    if dd.size == 0:
        return 0.0
    # A partially invalid path must not be summarized from its finite prefix.
    if not np.all(np.isfinite(dd)):
        return float("nan")
    return float(np.min(dd))


def cagr(returns: Array, periods_per_year: float = 252.0) -> float:
    _validate_periods(periods_per_year)
    w = wealth_index(returns)
    if w.size == 0:
        return float("nan")
    if not np.isfinite(w[-1]):
        return float("nan")
    years = w.size / periods_per_year
    if years <= 0 or w[-1] <= 0:
        return float("nan")
    return float(w[-1] ** (1.0 / years) - 1.0)


def annualized_vol(returns: Array, periods_per_year: float = 252.0) -> float:
    _validate_periods(periods_per_year)
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size < 2:
        return float("nan")
    if not np.all(np.isfinite(r)):
        # Honest NaN when any return is non-finite (no silent drop).
        return float("nan")
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def downside_deviation(returns: Array, mar: float = 0.0, periods_per_year: float = 252.0) -> float:
    """Annualized downside deviation; ``mar`` is a per-period return."""
    _validate_periods(periods_per_year)
    if not np.isfinite(mar):
        raise ValueError("mar must be finite")
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
    """Research-diagnostic Sharpe only — never a live P&L / promotion claim.

    Empty, constant (zero excess vol), or non-finite returns → honest NaN.
    ``flag_high_sharpe`` is a lab sanity flag, not an edge claim.
    """
    _validate_periods(periods_per_year)
    if not np.isfinite(rf_per_period):
        raise ValueError("rf_per_period must be finite")
    r = np.asarray(returns, dtype=float).reshape(-1)
    excess = r - rf_per_period
    n = int(excess.size)
    result: dict[str, float | int | bool] = {
        "n": n,
        "periods_per_year": periods_per_year,
        "annualized": not irregular,
        "flag_high_sharpe": False,
    }
    # Fail-closed: empty / short / constant / any non-finite → NaN (no silent drop).
    if n < 2 or not np.all(np.isfinite(excess)):
        result["sharpe"] = float("nan")
        return result
    vol = float(np.std(excess, ddof=1))
    if vol == 0.0:
        result["sharpe"] = float("nan")
        return result
    sr_periodic = float(np.mean(excess) / vol)
    sr = sr_periodic if irregular else sr_periodic * np.sqrt(periods_per_year)
    result["sharpe"] = sr
    result["flag_high_sharpe"] = bool(np.isfinite(sr) and abs(sr) > 5.0)
    return result


def sortino_ratio(returns: Array, mar: float = 0.0, periods_per_year: float = 252.0) -> float:
    """Annualized Sortino (research diagnostic only — not a live edge).

    Empty/short, zero downside deviation, or non-finite returns → honest NaN.
    """
    _validate_periods(periods_per_year)
    if not np.isfinite(mar):
        raise ValueError("mar must be finite")
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size < 2 or not np.all(np.isfinite(r)):
        return float("nan")
    dd = downside_deviation(r, mar=mar, periods_per_year=1.0)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    return float((np.mean(r) - mar) / dd * np.sqrt(periods_per_year))


def calmar_ratio(returns: Array, periods_per_year: float = 252.0) -> float:
    """CAGR / |max drawdown| — research diagnostic only (never a live claim).

    Empty path (mdd==0), zero drawdown, non-finite mdd/CAGR → honest NaN.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0 or not np.all(np.isfinite(r)):
        return float("nan")
    mdd = max_drawdown(r)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    growth = cagr(r, periods_per_year)
    if not np.isfinite(growth):
        return float("nan")
    return float(growth / abs(mdd))


def turnover(weights: Array, prev_weights: Array) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    p = np.asarray(prev_weights, dtype=float).reshape(-1)
    if w.shape != p.shape:
        raise ValueError(f"turnover weight length mismatch: {w.shape[0]} vs {p.shape[0]}")
    if w.size == 0:
        return 0.0
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(p))):
        return float("nan")
    return float(np.sum(np.abs(w - p)))
