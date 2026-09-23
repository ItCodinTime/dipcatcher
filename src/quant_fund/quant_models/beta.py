"""Market beta, Blume adjustment, CAPM cost of equity.

Counterpart of davidalmeida90/finance-agent-kit ``market_beta``. No
Damodaran scrape; ERP is an argument. Research cost-of-capital helper.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def market_beta(asset_excess: ArrayLike, market_excess: ArrayLike) -> float:
    """OLS slope of asset excess on market excess. Fail-closed on short samples."""
    y = np.asarray(asset_excess, dtype=float).reshape(-1)
    x = np.asarray(market_excess, dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        raise ValueError("need at least 3 finite overlapping observations")
    xm = x[ok] - x[ok].mean()
    ym = y[ok] - y[ok].mean()
    den = float(np.dot(xm, xm))
    if den <= 0:
        raise ValueError("market excess has zero variance")
    return float(np.dot(xm, ym) / den)


def blume_beta(raw_beta: float, *, toward: float = 1.0, weight: float = 2.0 / 3.0) -> float:
    """Blume (1975): ``weight * raw + (1-weight) * toward``."""
    if not np.isfinite(raw_beta) or not np.isfinite(toward) or not np.isfinite(weight):
        raise ValueError("inputs must be finite")
    return float(weight * raw_beta + (1.0 - weight) * toward)


def cost_of_equity(rf: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM: ``rf + beta * ERP``. ERP is supplied; nothing is scraped."""
    if not all(np.isfinite(v) for v in (rf, beta, equity_risk_premium)):
        raise ValueError("inputs must be finite")
    return float(rf + beta * equity_risk_premium)
