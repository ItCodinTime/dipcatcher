"""Exponential smoothing state-space forecasters.

Implements the classical smoothing recursions:

- Simple exponential smoothing (SES; Brown 1959) — flat forecasts.
- Holt (1957) linear trend, with the Gardner & McKenzie (1985) damped-trend
  extension.
- Holt-Winters (Winters 1960) additive and multiplicative seasonality.

Smoothing parameters are estimated by minimising the in-sample one-step
sum of squared errors with bounded L-BFGS-B; initial states use the standard
heterogeneous heuristics from Hyndman, Koehler, Ord & Snyder (2008,
*Forecasting with Exponential Smoothing*). Fail-closed on non-finite input,
too little history, or (for multiplicative seasonality) non-positive data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ETSFit:
    """Fitted exponential-smoothing model.

    ``kind`` is one of ``"ses"``, ``"holt"``, ``"holt_winters"``.  ``season``
    holds the final ``period`` seasonal states (empty for non-seasonal models);
    ``seasonal`` is ``"add"`` or ``"mul"``.
    """

    kind: str
    seasonal: str
    period: int
    n_obs: int
    params: dict[str, float]
    level: float
    trend: float
    season: Array
    fitted: Array
    resid: Array
    sse: float
    aic: float


def _check(y: Array, *, positive: bool = False, min_obs: int = 5) -> Array:
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < min_obs or not np.isfinite(arr).all():
        raise ValueError(f"series must be finite with >= {min_obs} observations")
    if positive and (arr <= 0.0).any():
        raise ValueError("multiplicative seasonality requires strictly positive data")
    return arr


def _aic(sse: float, n: int, k: int) -> float:
    if sse <= 0.0 or n <= 0:
        return float("-inf")
    return float(n * np.log(sse / n) + 2 * k)


def _ses_recursion(y: Array, alpha: float, level0: float) -> tuple[Array, float]:
    n = y.size
    fitted = np.empty(n)
    level = level0
    for t in range(n):
        fitted[t] = level
        level = alpha * y[t] + (1.0 - alpha) * level
    return fitted, level


def ses_fit(y: Array, alpha: float | None = None) -> ETSFit:
    """Simple exponential smoothing.  ``alpha`` is estimated when ``None``."""
    arr = _check(y)
    level0 = float(arr[0])

    def sse_of(a: float) -> float:
        fitted, _ = _ses_recursion(arr, a, level0)
        resid = arr - fitted
        return float(resid[1:] @ resid[1:])

    if alpha is None:
        res = minimize(lambda p: sse_of(p[0]), x0=[0.3], bounds=[(1e-4, 1.0 - 1e-4)])
        a = float(res.x[0])
    else:
        a = float(alpha)
    fitted, level = _ses_recursion(arr, a, level0)
    resid = arr - fitted
    sse = float(resid[1:] @ resid[1:])
    return ETSFit(
        kind="ses",
        seasonal="none",
        period=0,
        n_obs=arr.size,
        params={"alpha": a},
        level=float(level),
        trend=0.0,
        season=np.empty(0),
        fitted=fitted,
        resid=resid,
        sse=sse,
        aic=_aic(sse, arr.size - 1, 2),
    )


def _holt_recursion(
    y: Array, alpha: float, beta: float, phi: float, level0: float, trend0: float
) -> tuple[Array, float, float]:
    n = y.size
    fitted = np.empty(n)
    level, trend = level0, trend0
    for t in range(n):
        fitted[t] = level + phi * trend
        prev_level = level
        level = alpha * y[t] + (1.0 - alpha) * (prev_level + phi * trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend
    return fitted, level, trend


def holt_fit(y: Array, damped: bool = False) -> ETSFit:
    """Holt (1957) linear trend, optionally Gardner-McKenzie (1985) damped."""
    arr = _check(y, min_obs=6)
    level0 = float(arr[0])
    trend0 = float(np.mean(np.diff(arr[: min(arr.size, 6)])))

    def sse_of(params: Array) -> float:
        a, b, phi = params
        fitted, _, _ = _holt_recursion(arr, a, b, phi, level0, trend0)
        resid = arr - fitted
        return float(resid[1:] @ resid[1:])

    bounds = [(1e-4, 1.0 - 1e-4), (1e-4, 1.0 - 1e-4), (0.8, 1.0) if damped else (1.0, 1.0)]
    x0 = np.array([0.3, 0.1, 0.98 if damped else 1.0])
    res = minimize(sse_of, x0=x0, bounds=bounds)
    a, b, phi = (float(v) for v in res.x)
    fitted, level, trend = _holt_recursion(arr, a, b, phi, level0, trend0)
    resid = arr - fitted
    sse = float(resid[1:] @ resid[1:])
    return ETSFit(
        kind="holt",
        seasonal="none",
        period=0,
        n_obs=arr.size,
        params={"alpha": a, "beta": b, "phi": phi},
        level=float(level),
        trend=float(trend),
        season=np.empty(0),
        fitted=fitted,
        resid=resid,
        sse=sse,
        aic=_aic(sse, arr.size - 1, 4 if damped else 3),
    )


def _hw_recursion(
    y: Array,
    m: int,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    level0: float,
    trend0: float,
    season0: Array,
    mul: bool,
) -> tuple[Array, float, float, Array]:
    n = y.size
    fitted = np.empty(n)
    level, trend = level0, trend0
    season = season0.copy()
    for t in range(n):
        s_prev = season[t % m]
        if mul:
            fitted[t] = (level + phi * trend) * s_prev
        else:
            fitted[t] = level + phi * trend + s_prev
        prev_level = level
        if mul:
            level = alpha * (y[t] / s_prev) + (1.0 - alpha) * (prev_level + phi * trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend
            season[t % m] = gamma * (y[t] / level) + (1.0 - gamma) * s_prev
        else:
            level = alpha * (y[t] - s_prev) + (1.0 - alpha) * (prev_level + phi * trend)
            trend = beta * (level - prev_level) + (1.0 - beta) * phi * trend
            season[t % m] = gamma * (y[t] - prev_level - phi * trend) + (1.0 - gamma) * s_prev
    return fitted, level, trend, season


def holt_winters_fit(y: Array, period: int, seasonal: str = "add", damped: bool = False) -> ETSFit:
    """Holt-Winters seasonal smoothing.  ``seasonal`` is ``"add"`` or ``"mul"``."""
    if period < 2:
        raise ValueError("period must be >= 2")
    mul = seasonal == "mul"
    if seasonal not in {"add", "mul"}:
        raise ValueError("seasonal must be 'add' or 'mul'")
    arr = _check(y, positive=mul, min_obs=2 * period + 2)
    m = period
    first, second = arr[:m], arr[m : 2 * m]
    level0 = float(first.mean())
    trend0 = float((second.mean() - first.mean()) / m)
    if mul:
        season0 = np.array([arr[i] / level0 for i in range(m)], dtype=float)
    else:
        season0 = np.array([arr[i] - level0 for i in range(m)], dtype=float)

    def sse_of(params: Array) -> float:
        a, b, g, phi = params
        fitted, *_ = _hw_recursion(arr, m, a, b, g, phi, level0, trend0, season0, mul)
        resid = arr - fitted
        return float(resid[m:] @ resid[m:])

    bounds = [
        (1e-4, 1.0 - 1e-4),
        (1e-4, 1.0 - 1e-4),
        (1e-4, 1.0 - 1e-4),
        (0.8, 1.0) if damped else (1.0, 1.0),
    ]
    x0 = np.array([0.3, 0.05, 0.1, 0.98 if damped else 1.0])
    res = minimize(sse_of, x0=x0, bounds=bounds)
    a, b, g, phi = (float(v) for v in res.x)
    fitted, level, trend, season = _hw_recursion(arr, m, a, b, g, phi, level0, trend0, season0, mul)
    resid = arr - fitted
    sse = float(resid[m:] @ resid[m:])
    return ETSFit(
        kind="holt_winters",
        seasonal=seasonal,
        period=m,
        n_obs=arr.size,
        params={"alpha": a, "beta": b, "gamma": g, "phi": phi},
        level=float(level),
        trend=float(trend),
        season=season,
        fitted=fitted,
        resid=resid,
        sse=sse,
        aic=_aic(sse, arr.size - m, 5 if damped else 4),
    )


def ets_forecast(fit: ETSFit, h: int) -> Array:
    """Multi-step point forecasts from a fitted model."""
    if h < 1:
        raise ValueError("h must be >= 1")
    if fit.kind == "ses":
        return np.full(h, fit.level)
    phi = fit.params.get("phi", 1.0)
    steps = np.arange(1, h + 1)
    # Damped trend cumulative factor sum_{i=1..k} phi^i; equals k when phi == 1.
    if abs(phi - 1.0) < 1e-12:
        damp = steps.astype(float)
    else:
        damp = phi * (1.0 - phi**steps) / (1.0 - phi)
    base = fit.level + damp * fit.trend
    if fit.kind == "holt":
        return base
    m = fit.period
    # Seasonal state is indexed by phase ``t % m``; continue from the last obs.
    seas = np.array([fit.season[(fit.n_obs - 1 + int(k)) % m] for k in steps], dtype=float)
    return base * seas if fit.seasonal == "mul" else base + seas
