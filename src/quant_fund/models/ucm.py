"""Harvey (1989) unobserved-components structural time series.

The model decomposes ``y_t`` into a random-walk-with-drift level, a
stochastic cycle, and observation noise:

    y_t   = mu_t + c_t + e_t,           e_t ~ N(0, s_e)
    mu_t  = mu_{t-1} + b_{t-1} + n_t,   n_t ~ N(0, s_n)
    b_t   = b_{t-1} + z_t,              z_t ~ N(0, s_z)
    [c_t; c*_t] = rho * R(lam) [c_{t-1}; c*_{t-1}] + k_t,  k_t ~ N(0, s_k I)

with ``R(lam)`` the rotation matrix at frequency ``lam = 2*pi/period`` and
damping ``rho`` in (0, 1).  Estimation is exact Gaussian MLE on the Kalman
innovations (Durbin-Koopman 2012); smoothing uses Rauch-Tung-Striebel via
:func:`quant_fund.models.state_space.kalman_filter`.

Reference: A. C. Harvey (1989), "Forecasting, Structural Time Series Models
and the Kalman Filter".  Fail-closed on non-finite input or T < 12.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from quant_fund.models.state_space import kalman_filter, kalman_smoother

Array = NDArray[np.float64]


def _as_vector(y: Array, min_obs: int = 12) -> Array:
    v = np.asarray(y, dtype=float).ravel()
    if v.size < min_obs:
        raise ValueError(f"need at least {min_obs} observations")
    if not np.all(np.isfinite(v)):
        raise ValueError("y must contain only finite values")
    return v


def _ssm(theta: Array, cycle_period: float | None) -> tuple[Array, Array, Array, Array]:
    """Build (F, H, Q, R) from log-variance params (plus logit rho for cycle)."""
    if cycle_period is None:
        se, sn, sz = np.exp(theta[:3])
        f = np.array([[1.0, 1.0], [0.0, 1.0]])
        h = np.array([[1.0, 0.0]])
        q = np.diag([sn, sz])
        r = np.array([[se]])
        return f, h, q, r
    se, sn, sz, sk = np.exp(theta[:4])
    rho = 1.0 / (1.0 + math.exp(-float(theta[4])))
    lam = 2.0 * math.pi / cycle_period
    rot = rho * np.array([[math.cos(lam), math.sin(lam)], [-math.sin(lam), math.cos(lam)]])
    f = np.block(
        [
            [np.array([[1.0, 1.0]]), np.zeros((1, 2))],
            [np.zeros((1, 2)), np.zeros((1, 2))],
            [np.zeros((2, 2)), rot],
        ]
    )
    f[1, 1] = 1.0
    h = np.array([[1.0, 0.0, 1.0, 0.0]])
    q = np.diag([sn, sz, sk, sk])
    r = np.array([[se]])
    return f, h, q, r


def _nll(theta: Array, y2: Array, cycle_period: float | None) -> float:
    f, h, q, r = _ssm(theta, cycle_period)
    try:
        out = kalman_filter(y2, f, h, q, r)
    except (ValueError, np.linalg.LinAlgError):
        return 1e10
    return -float(out["loglik"][0])


def ucm_fit(
    y: Array, cycle_period: float | None = None, *, max_iter: int = 200
) -> dict[str, Array]:
    """MLE fit of the level + slope (+ cycle) structural model.

    ``cycle_period`` in bars (e.g. 12 for monthly seasonality); ``None``
    fits trend + noise only.  Returns fitted variances, damping, filtered
    state paths, and the log-likelihood.
    """
    v = _as_vector(y)
    if cycle_period is not None and (not np.isfinite(cycle_period) or cycle_period < 2):
        raise ValueError("cycle_period must be >= 2 or None")
    y2 = v.reshape(-1, 1)
    var = float(np.var(v)) or 1.0
    theta0 = np.log([var * 0.5, var * 0.05, var * 0.01])
    if cycle_period is not None:
        theta0 = np.concatenate([theta0, [math.log(var * 0.05), 0.0]])
    res = optimize.minimize(
        _nll,
        theta0,
        args=(y2, cycle_period),
        method="BFGS",
        options={"maxiter": max_iter},
    )
    f, h, q, r = _ssm(res.x, cycle_period)
    filt = kalman_filter(y2, f, h, q, r)
    sm = kalman_smoother(filt, f)
    se, sn, sz = np.exp(res.x[:3])
    out: dict[str, Array] = {
        "sigma_obs": np.array([math.sqrt(se)]),
        "sigma_level": np.array([math.sqrt(sn)]),
        "sigma_slope": np.array([math.sqrt(sz)]),
        "loglik": np.array([-float(res.fun)]),
        "converged": np.array([float(bool(res.success))]),
        "x_smooth": sm["x_smooth"],
        "x_filt": filt["x_filt"],
        "F": f,
        "has_cycle": np.array([float(cycle_period is not None)]),
        "level": sm["x_smooth"][:, 0],
        "slope": sm["x_smooth"][:, 1],
    }
    if cycle_period is not None:
        rho = 1.0 / (1.0 + math.exp(-float(res.x[4])))
        out["sigma_cycle"] = np.array([math.sqrt(float(np.exp(res.x[3])))])
        out["rho"] = np.array([rho])
        out["cycle"] = sm["x_smooth"][:, 2]
    else:
        out["cycle"] = np.zeros(v.size)
    return out


def ucm_forecast(fit: dict[str, Array], steps: int = 12) -> dict[str, Array]:
    """Iterate the fitted transition forward from the last filtered state.

    The cycle is damped by rho each step; the level inherits the last slope.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    f = np.asarray(fit["F"], dtype=float)
    x = np.asarray(fit["x_filt"], dtype=float)[-1]
    path = np.empty((steps, x.size))
    level = np.empty(steps)
    for i in range(steps):
        x = f @ x
        path[i] = x
        level[i] = x[0] + (x[2] if fit["has_cycle"][0] > 0.5 else 0.0)
    return {"y_hat": level, "state_path": path}
