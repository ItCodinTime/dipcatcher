"""Approximate P&L attribution. Labeled as approximation where not rigorous."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def factor_selection_timing(
    portfolio_ret: NDArray[np.float64],
    market_ret: NDArray[np.float64],
    residual_ret: NDArray[np.float64],
    cost_ret: NDArray[np.float64],
) -> dict[str, float]:
    """Decompose average return into market, residual (selection), and cost.

    Timing is the covariance between weight-implied residual and market, reported separately
    only if `residual_ret` is already residualized. This is an approximation.
    """
    return {
        "market": float(np.nanmean(market_ret)),
        "selection_residual": float(np.nanmean(residual_ret)),
        "cost": float(np.nanmean(cost_ret)),
        "total": float(np.nanmean(portfolio_ret)),
        "convention": 1.0,  # 1 = approximation
    }
