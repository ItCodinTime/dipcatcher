"""The Theta forecasting method (Assimakopoulos & Nikolopoulos 2000).

The classic Theta method decomposes a series into two *theta lines*.  For a
coefficient ``theta`` the line is

    Z_t(theta) = theta * y_t + (1 - theta) * (a + b t),

where ``a + b t`` is the ordinary-least-squares trend of ``y`` on time.  The
standard method uses ``theta = 0`` (the linear-regression line, extrapolated
linearly) and ``theta = 2`` (which doubles the local curvature and is
extrapolated with simple exponential smoothing).  The two extrapolations are
averaged with equal weights.

Hyndman & Billah (2003) showed that this construction is equivalent to simple
exponential smoothing with a drift equal to half the OLS slope; the recursion
here follows the explicit two-line form.  Fail-closed on non-finite input or
too little history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ThetaFit:
    """Fitted classic (two-line) Theta model."""

    alpha: float
    intercept: float
    slope: float
    ses_level: float
    n_obs: int
    theta: float
    fitted: Array
    resid: Array


def _ses_level_path(z: Array, alpha: float) -> tuple[Array, float]:
    n = z.size
    onestep = np.empty(n)
    level = float(z[0])
    for t in range(n):
        onestep[t] = level
        level = alpha * z[t] + (1.0 - alpha) * level
    return onestep, level


def theta_fit(y: Array, theta: float = 2.0) -> ThetaFit:
    """Fit the classic Theta method (``theta`` controls the curvature line)."""
    arr = np.asarray(y, dtype=float).ravel()
    if arr.size < 6 or not np.isfinite(arr).all():
        raise ValueError("series must be finite with >= 6 observations")
    if theta <= 1.0:
        raise ValueError("theta must be > 1 for the curvature line")
    n = arr.size
    t = np.arange(n, dtype=float)
    # OLS trend (the theta = 0 line).
    design = np.column_stack([np.ones(n), t])
    coef, *_ = np.linalg.lstsq(design, arr, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])
    trend_line = intercept + slope * t
    z2 = theta * arr + (1.0 - theta) * trend_line

    def sse_of(alpha: float) -> float:
        onestep, _ = _ses_level_path(z2, alpha)
        resid = z2[1:] - onestep[1:]
        return float(resid @ resid)

    opt = minimize_scalar(sse_of, bounds=(1e-4, 1.0 - 1e-4), method="bounded")
    alpha = float(opt.x)
    ses_onestep, ses_level = _ses_level_path(z2, alpha)
    # In-sample one-step fit combines the two lines with equal weight, rescaled
    # so the curvature line recovers the level (theta=2 -> weight 1/theta).
    w = 1.0 / theta
    fitted = w * ses_onestep + (1.0 - w) * trend_line
    resid = arr - fitted
    return ThetaFit(
        alpha=alpha,
        intercept=intercept,
        slope=slope,
        ses_level=float(ses_level),
        n_obs=n,
        theta=float(theta),
        fitted=fitted,
        resid=resid,
    )


def theta_forecast(fit: ThetaFit, h: int) -> Array:
    """Multi-step Theta forecasts.

    The regression line is extrapolated linearly while the theta-2 line is
    extrapolated flat (its final SES level); the two are combined with the
    ``1/theta`` weighting used in fitting.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    steps = np.arange(fit.n_obs, fit.n_obs + h, dtype=float)
    trend_extrap = fit.intercept + fit.slope * steps
    ses_extrap = np.full(h, fit.ses_level)
    w = 1.0 / fit.theta
    return w * ses_extrap + (1.0 - w) * trend_extrap
