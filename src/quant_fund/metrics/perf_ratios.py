"""Performance ratios that extend the classical Sharpe/Sortino battery.

- Adjusted (skew/kurtosis-aware) Sharpe ratio, Pezier & White (2006).
- Modigliani-Modigliani M^2 risk-adjusted performance, Modigliani &
  Modigliani (1997).
- Rachev (R-) ratio of expected tail gain to expected tail loss, Biglova,
  Ortobelli, Rachev & Stoyanov (2004).
- Gain-to-pain ratio, Schwager (*Hedge Fund Market Wizards*, 2012).
- Upside potential ratio, Sortino, van der Meer & Plantinga (1999).

Returns use the per-period, positive-is-gain convention.  Fail-closed on
non-finite input or degenerate dispersion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_returns(r: Array, min_obs: int = 20) -> Array:
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"returns must be finite with >= {min_obs} observations")
    return arr


def adjusted_sharpe_ratio(returns: Array, rf: float = 0.0) -> float:
    """Pezier-White (2006) skew/kurtosis-adjusted Sharpe (per-period)."""
    arr = _as_returns(returns)
    excess = arr - rf
    sd = float(excess.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("zero dispersion")
    sr = float(excess.mean()) / sd
    z = (excess - excess.mean()) / sd
    skew = float((z**3).mean())
    exkurt = float((z**4).mean() - 3.0)
    return float(sr * (1.0 + (skew / 6.0) * sr - (exkurt / 24.0) * sr**2))


def m_squared(
    returns: Array, benchmark: Array, rf: float = 0.0, periods_per_year: float = 252.0
) -> dict[str, float]:
    """Modigliani-Modigliani M^2 (annualised), reported vs the benchmark risk."""
    port = _as_returns(returns)
    bench = _as_returns(benchmark)
    sd = float(port.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("zero portfolio dispersion")
    sr = float((port - rf).mean()) / sd
    sd_bench = float(bench.std(ddof=1))
    m2_period = rf + sr * sd_bench
    return {
        "m2_period": float(m2_period),
        "m2_annual": float(m2_period * periods_per_year),
        "sharpe": float(sr),
    }


def rachev_ratio(returns: Array, alpha: float = 0.05, beta: float = 0.05, rf: float = 0.0) -> float:
    """Rachev ratio: expected top-``beta`` gain over expected worst-``alpha`` loss."""
    if not (0.0 < alpha < 0.5 and 0.0 < beta < 0.5):
        raise ValueError("alpha and beta must lie in (0, 0.5)")
    x = _as_returns(returns) - rf
    q_low = float(np.quantile(x, alpha))
    q_high = float(np.quantile(x, 1.0 - beta))
    left = x[x <= q_low]
    right = x[x >= q_high]
    etl = -float(left.mean()) if left.size else 0.0  # expected tail loss (positive)
    etg = float(right.mean()) if right.size else 0.0  # expected tail gain
    if etl <= 0.0:
        raise ValueError("non-positive expected tail loss; ratio undefined")
    return float(etg / etl)


def gain_to_pain_ratio(returns: Array, rf: float = 0.0) -> float:
    """Schwager gain-to-pain: sum of returns over the sum of absolute losses."""
    x = _as_returns(returns) - rf
    pain = float(np.sum(np.maximum(-x, 0.0)))
    if pain <= 0.0:
        raise ValueError("no losses; gain-to-pain undefined")
    return float(np.sum(x) / pain)


def upside_potential_ratio(returns: Array, mar: float = 0.0) -> float:
    """Sortino-van der Meer-Plantinga (1999) upside potential ratio."""
    x = _as_returns(returns)
    upside = float(np.mean(np.maximum(x - mar, 0.0)))
    downside = float(np.sqrt(np.mean(np.minimum(x - mar, 0.0) ** 2)))
    if downside <= 0.0:
        raise ValueError("no downside risk; ratio undefined")
    return float(upside / downside)
