"""Gu–Kelly–Xiu (2020) style predictive R² vs a zero forecast.

Catalog already has PCR / PLS / Huber GBRT (``pcr``, ``pls``, ``gbrt``).
This module is the metric from NBER w25398, not another model family.
Neural nets remain behind ADR-007.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def r2_oos(y_true: ArrayLike, y_hat: ArrayLike) -> float:
    """``1 - Σ(y − ŷ)² / Σ y²``. Denominator is the zero forecast, not a mean."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    f = np.asarray(y_hat, dtype=float).reshape(-1)
    if y.size != f.size:
        raise ValueError("y_true and y_hat length mismatch")
    ok = np.isfinite(y) & np.isfinite(f)
    if not ok.any():
        return float("nan")
    sse = float(np.sum((y[ok] - f[ok]) ** 2))
    sst = float(np.sum(y[ok] ** 2))
    if sst <= 0:
        return float("nan")
    return 1.0 - sse / sst
