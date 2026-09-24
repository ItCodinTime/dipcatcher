"""Scale-free forecast-accuracy statistics.

- Theil's U1 (Theil 1966): ``sqrt(MSE) / (sqrt(mean a^2) + sqrt(mean f^2))``,
  bounded in ``[0, 1]`` (0 = perfect).
- Theil's U2: RMSE of the forecast relative to the naive no-change forecast;
  ``< 1`` beats the random walk.
- Mean absolute scaled error (Hyndman & Koehler 2006): MAE scaled by the
  in-sample naive (seasonal) MAE; ``< 1`` beats the naive benchmark.

Fail-closed on non-finite input or mismatched lengths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pair(actual: Array, forecast: Array, n: int = 3) -> tuple[Array, Array]:
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if a.size != f.size or a.size < n or not (np.isfinite(a).all() and np.isfinite(f).all()):
        raise ValueError(f"actual and forecast must be finite, aligned, length >= {n}")
    return a, f


def theil_u1(actual: Array, forecast: Array) -> float:
    """Theil's U1 inequality coefficient in [0, 1]."""
    a, f = _pair(actual, forecast)
    rmse = float(np.sqrt(np.mean((f - a) ** 2)))
    denom = float(np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(f**2)))
    if denom <= 0.0:
        raise ValueError("degenerate series")
    return rmse / denom


def theil_u2(actual: Array, forecast: Array) -> float:
    """Theil's U2 relative to the naive no-change forecast (< 1 beats naive)."""
    a, f = _pair(actual, forecast, n=2)
    num = float(np.sqrt(np.sum((f[1:] - a[1:]) ** 2)))
    denom = float(np.sqrt(np.sum((a[1:] - a[:-1]) ** 2)))
    if denom <= 0.0:
        raise ValueError("naive benchmark has zero error")
    return num / denom


def mase(actual: Array, forecast: Array, seasonality: int = 1) -> float:
    """Mean absolute scaled error (Hyndman-Koehler 2006)."""
    a, f = _pair(actual, forecast, n=seasonality + 2)
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1")
    scale = float(np.mean(np.abs(a[seasonality:] - a[:-seasonality])))
    if scale <= 0.0:
        raise ValueError("naive seasonal MAE is zero")
    return float(np.mean(np.abs(a - f)) / scale)
