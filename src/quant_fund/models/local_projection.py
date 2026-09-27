"""Jorda (2005) local projections for impulse-response estimation.

Instead of iterating a fitted VAR, local projections estimate the impulse
response at each horizon ``h`` with a separate regression

    y_{t+h} = alpha_h + beta_h x_t + controls_t + e_{t+h},

so ``beta_h`` is the response of ``y`` to the impulse ``x`` at horizon ``h``.
Controls are lagged values of ``y`` and ``x``.  Because the residuals at horizon
``h`` are serially correlated by construction, standard errors use a
Newey-West HAC covariance with a horizon-dependent bandwidth.

Reference: O. Jorda (2005), "Estimation and inference of impulse responses by
local projections", American Economic Review.  Fail-closed on non-finite input
or too little history.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _nw_var(x_design: Array, resid: Array, col: int, lag: int) -> float:
    """Newey-West variance of a single OLS coefficient at index ``col``."""
    n, k = x_design.shape
    xtx_inv = np.linalg.inv(x_design.T @ x_design)
    u = resid[:, None] * x_design
    meat = u.T @ u
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        g = u[j:].T @ u[:-j]
        meat += w * (g + g.T)
    cov = xtx_inv @ meat @ xtx_inv
    return float(cov[col, col])


def local_projection(
    y: Array, x: Array, horizons: int = 12, control_lags: int = 2
) -> dict[str, Array]:
    """Estimate local-projection impulse responses of ``y`` to impulse ``x``."""
    ya = np.asarray(y, dtype=float).ravel()
    xa = np.asarray(x, dtype=float).ravel()
    if ya.size != xa.size or ya.size < horizons + control_lags + 20:
        raise ValueError("y and x must be aligned with enough history for the horizons")
    if not (np.isfinite(ya).all() and np.isfinite(xa).all()):
        raise ValueError("y and x must be finite")
    if horizons < 1 or control_lags < 0:
        raise ValueError("horizons >= 1 and control_lags >= 0 required")
    n = ya.size
    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    for h in range(horizons + 1):
        rows = []
        target = []
        for t in range(control_lags, n - h):
            controls = [1.0, xa[t]]
            for lag in range(1, control_lags + 1):
                controls.append(ya[t - lag])
                controls.append(xa[t - lag])
            rows.append(controls)
            target.append(ya[t + h])
        design = np.asarray(rows)
        yv = np.asarray(target)
        beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
        resid = yv - design @ beta
        irf[h] = beta[1]  # coefficient on x_t
        se[h] = np.sqrt(max(_nw_var(design, resid, 1, h + 1), 0.0))
    return {
        "horizon": np.arange(horizons + 1, dtype=float),
        "irf": irf,
        "se": se,
        "ci_low": irf - 1.96 * se,
        "ci_high": irf + 1.96 * se,
    }
