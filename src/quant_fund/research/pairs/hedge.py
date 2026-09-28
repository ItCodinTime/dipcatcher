"""Hedge-ratio estimation for a candidate pair: static OLS + scalar Kalman.

Every estimator here consumes exactly the arrays it is given — callers that
pass a trailing window get a point-in-time-correct estimate. Research /
statistics only: nothing here sizes, routes, or values a trade.

References:
- Engle, Granger (1987): static cointegrating regression.
- Elliott, van der Hoek, Malcolm (2005): Kalman-filter pairs state space.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _pair(y: Array, x: Array) -> tuple[Array, Array]:
    yv = np.asarray(y, dtype=float).reshape(-1)
    xv = np.asarray(x, dtype=float).reshape(-1)
    if yv.size != xv.size or yv.size < 10:
        raise ValueError("y and x must be equal-length arrays with >= 10 obs")
    if not np.all(np.isfinite(yv)) or not np.all(np.isfinite(xv)):
        raise ValueError("y and x must be finite")
    return yv, xv


def ols_hedge_ratio(y: Array, x: Array) -> tuple[float, float]:
    """Static cointegrating regression ``y = alpha + beta*x + eps``.

    Returns ``(alpha, beta)`` — intercept and hedge ratio.
    """
    yv, xv = _pair(y, x)
    design = np.column_stack([np.ones(yv.size), xv])
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    return float(beta[0]), float(beta[1])


def kalman_hedge_ratio(
    y: Array,
    x: Array,
    *,
    q: float = 1e-4,
    r: float | None = None,
    burn: int = 30,
    alpha: float | None = None,
) -> Array:
    """Scalar Kalman-filter hedge ratio; ``beta_t`` follows a random walk.

    Observation equation ``y_t = alpha + beta_t * x_t + eps_t`` with
    ``eps_t ~ N(0, r)``; state equation ``beta_t = beta_{t-1} + w_t`` with
    ``w_t ~ N(0, q)`` (Elliott et al. 2005). The intercept and the
    measurement-noise variance are treated as fixed nuisance parameters
    calibrated by OLS on the ``burn``-observation prefix only — so within a
    single call the filter is strictly causal: ``beta[t]`` depends on
    observations ``<= t`` plus the frozen burn-in calibration. Calling it on
    a trailing window ending at ``t`` is point-in-time safe, and mutating
    inputs after index ``t*`` leaves the ``<= t*`` path bit-identical.

    Returns the filtered ``beta`` path, shape ``(t,)`` — ``out[t]`` is the
    post-update estimate after observing ``(x_t, y_t)``.
    """
    yv, xv = _pair(y, x)
    n = yv.size
    if not np.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and > 0")
    burn_i = min(max(int(burn), 5), n)
    a0, b0 = ols_hedge_ratio(yv[:burn_i], xv[:burn_i])
    if alpha is None:
        alpha_ = a0  # burn-in intercept — causal within the call
    else:
        alpha_ = float(alpha)
        if not np.isfinite(alpha_):
            raise ValueError("alpha must be finite")
    if r is None:
        resid = yv[:burn_i] - alpha_ - b0 * xv[:burn_i]
        r_ = float(np.var(resid))
        if not np.isfinite(r_) or r_ <= 0:
            r_ = 1e-6
    else:
        r_ = float(r)
        if not np.isfinite(r_) or r_ <= 0:
            raise ValueError("r must be finite and > 0")

    beta = float(b0)
    var = 1.0  # diffuse prior variance on the initial slope
    out = np.empty(n)
    for t in range(n):
        # predict: beta_t|t-1 = beta_{t-1}; variance grows by q
        var += q
        # update with observation t
        innov = (yv[t] - alpha_) - beta * xv[t]
        s = xv[t] * xv[t] * var + r_
        gain = xv[t] * var / s
        beta += gain * innov
        var = max((1.0 - gain * xv[t]) * var, 1e-12)
        out[t] = beta
    return out
