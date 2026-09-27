"""Additional liquidity and drawdown-adjusted performance ratios.

- **Hui-Heubel** (1984) liquidity ratio: the price range over a window relative
  to turnover, ``LR = ((P_max - P_min)/P_min) / (V / (S * P_bar))``; higher
  values mean less liquidity (price moves more per unit of turnover).
- **Martin ratio / Ulcer Performance Index** (Martin & McCann 1989): excess
  return divided by the Ulcer Index (the RMS drawdown), rewarding smooth
  equity curves.

Fail-closed on non-finite input or degenerate denominators.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def hui_heubel_ratio(
    prices: Array, volume_shares: Array, shares_outstanding: float
) -> float:
    """Hui-Heubel liquidity ratio over the supplied window (higher = less liquid)."""
    p = np.asarray(prices, dtype=float).ravel()
    v = np.asarray(volume_shares, dtype=float).ravel()
    if (
        p.size < 2
        or p.size != v.size
        or not (np.isfinite(p).all() and np.isfinite(v).all())
    ):
        raise ValueError("prices and volume must be finite, aligned, length >= 2")
    if (
        not np.isfinite(shares_outstanding)
        or shares_outstanding <= 0.0
        or (p <= 0).any()
        or (v < 0).any()
    ):
        raise ValueError(
            "require positive prices, shares outstanding, non-negative volume"
        )
    p_min = float(p.min())
    price_range = (float(p.max()) - p_min) / p_min
    p_bar = float(p.mean())
    dollar_volume = float(np.sum(v * p))
    turnover = dollar_volume / (shares_outstanding * p_bar)
    if not np.isfinite(turnover) or turnover <= 0.0:
        raise ValueError("zero turnover; ratio undefined")
    ratio = float(price_range / turnover)
    if not np.isfinite(ratio):
        raise ValueError("non-finite Hui-Heubel ratio")
    return ratio


def ulcer_index(returns: Array) -> float:
    """Ulcer Index: RMS percentage drawdown of the equity curve."""
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError("returns must be finite with >= 2 observations")
    if (r <= -1.0).any():
        raise ValueError("returns must exceed -1 to preserve positive wealth")
    equity = np.cumprod(1.0 + r)
    if not np.isfinite(equity).all() or (equity <= 0.0).any():
        raise ValueError("degenerate equity curve")
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    drawdown = 100.0 * (equity / peak - 1.0)
    return float(np.sqrt(np.mean(drawdown**2)))


def martin_ratio(
    returns: Array, rf: float = 0.0, periods_per_year: float = 252.0
) -> float:
    """Martin ratio: annualised excess return in percent divided by percent Ulcer Index.

    ``rf`` is the per-period risk-free return in fractional units.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 2 or not np.isfinite(r).all():
        raise ValueError("returns must be finite with >= 2 observations")
    if (
        not np.isfinite(rf)
        or not np.isfinite(periods_per_year)
        or periods_per_year <= 0.0
    ):
        raise ValueError("rf must be finite and periods_per_year positive and finite")
    ui = ulcer_index(r)
    if ui <= 0.0:
        raise ValueError("zero Ulcer Index (no drawdown); ratio undefined")
    ann_excess_pct = 100.0 * float(r.mean() - rf) * periods_per_year
    ratio = float(ann_excess_pct / ui)
    if not np.isfinite(ratio):
        raise ValueError("non-finite Martin ratio")
    return ratio
