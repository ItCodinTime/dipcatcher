"""Nonlinear state-space filters: EKF, UKF, and bootstrap particle filter.

``models/state_space.py`` provides the linear Kalman filter; this module
covers the nonlinear canon.

References:
- Julier & Uhlmann (1997): unscented transform / UKF sigma points.
- Gordon, Salmond & Smith (1993): bootstrap particle filter (SIR).
- Anderson & Moore (1979) / Jazwinski (1970): extended Kalman filter.
- Doucet, de Freitas & Gordon (2001): sequential Monte Carlo methods.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _check_series(y: Array, n: int = 10) -> Array:
    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"observations must be finite with length >= {n}")
    return v


def extended_kalman(
    y: Array,
    f,
    F_jac,
    h,
    H_jac,
    Q: float,
    R: float,
    x0: float,
    P0: float,
) -> dict[str, Array]:
    """Scalar extended Kalman filter (Jazwinski 1970).

    ``f``/``F_jac``: nonlinear transition and its Jacobian.
    ``h``/``H_jac``: nonlinear observation and its Jacobian.
    Q, R: process and observation noise variances (positive).
    """
    v = _check_series(y)
    if Q <= 0 or R <= 0 or P0 <= 0:
        raise ValueError("Q, R, P0 must be positive")
    n = v.size
    xs = np.empty(n)
    Ps = np.empty(n)
    x, P = float(x0), float(P0)
    for t in range(n):
        xp = float(f(x))
        F = float(F_jac(x))
        Pp = F * F * P + Q
        H = float(H_jac(xp))
        S = H * H * Pp + R
        if S <= 0:
            raise ValueError("degenerate innovation variance")
        K = Pp * H / S
        innov = v[t] - float(h(xp))
        x = xp + K * innov
        P = max((1.0 - K * H) * Pp, 1e-14)
        xs[t] = x
        Ps[t] = P
    return {"state": xs, "variance": Ps}


def _sigma_points(x: float, P: float, kappa: float = 2.0) -> tuple[Array, Array, Array]:
    """Julier–Uhlmann sigma points for a scalar state: 3 points."""
    s = math.sqrt(max((1.0 + kappa) * P, 1e-14))
    pts = np.array([x, x + s, x - s])
    wm = np.array([kappa / (1.0 + kappa), 1.0 / (2 * (1 + kappa)), 1.0 / (2 * (1 + kappa))])
    wc = wm.copy()
    # Julier-Uhlmann: W0c = W0m + (1 - alpha^2 + beta); alpha=1, beta=2.
    wc[0] = wm[0] + 2.0
    return pts, wm, wc


def unscented_kalman(
    y: Array,
    f,
    h,
    Q: float,
    R: float,
    x0: float,
    P0: float,
    kappa: float = 2.0,
) -> dict[str, Array]:
    """Scalar unscented Kalman filter (Julier–Uhlmann 1997)."""
    v = _check_series(y)
    if Q <= 0 or R <= 0 or P0 <= 0:
        raise ValueError("Q, R, P0 must be positive")
    n = v.size
    xs = np.empty(n)
    Ps = np.empty(n)
    x, P = float(x0), float(P0)
    for t in range(n):
        pts, wm, wc = _sigma_points(x, P, kappa)
        fp = np.array([float(f(pt)) for pt in pts])
        xp = float(wm @ fp)
        Pp = float(wc @ ((fp - xp) ** 2)) + Q
        pts2, wm2, wc2 = _sigma_points(xp, Pp, kappa)
        hp = np.array([float(h(pt)) for pt in pts2])
        yp = float(wm2 @ hp)
        Pyy = float(wc2 @ ((hp - yp) ** 2)) + R
        Pxy = float(wc2 @ ((pts2 - xp) * (hp - yp)))
        if Pyy <= 0:
            raise ValueError("degenerate innovation variance")
        K = Pxy / Pyy
        x = xp + K * (v[t] - yp)
        P = max(Pp - K * Pyy * K, 1e-14)
        xs[t] = x
        Ps[t] = P
    return {"state": xs, "variance": Ps}


def particle_filter(
    y: Array,
    f,
    obs_loglik,
    q_std: float,
    n_particles: int = 500,
    x0: float = 0.0,
    p0_std: float = 1.0,
    seed: int = 0,
    resample_frac: float = 0.5,
) -> dict[str, Array]:
    """Gordon–Salmond–Smith (1993) bootstrap/SIR particle filter.

    ``f(x, rng)`` draws the next state; ``obs_loglik(x, y_t)`` returns
    log p(y_t | x). Systematic resampling when ESS < frac*N.
    Returns filtered means, variances, ESS path, and log-likelihood.
    """
    v = _check_series(y, n=5)
    if q_std <= 0 or p0_std <= 0:
        raise ValueError("noise scales must be positive")
    if not (10 <= n_particles <= 100000):
        raise ValueError("n_particles must be in [10, 100000]")
    rng = np.random.default_rng(seed)
    n = v.size
    parts = rng.normal(x0, p0_std, n_particles)
    means = np.empty(n)
    vars_ = np.empty(n)
    ess = np.empty(n)
    loglik = 0.0

    def systematic(weights: Array) -> Array:
        c = np.cumsum(weights)
        u0 = rng.random() / n_particles
        u = u0 + np.arange(n_particles) / n_particles
        idx = np.searchsorted(c, u)
        return np.clip(idx, 0, n_particles - 1)

    w = np.full(n_particles, 1.0 / n_particles)
    for t in range(n):
        parts = np.array([float(f(pt, rng)) for pt in parts]) + rng.normal(0.0, q_std, n_particles)
        ll = np.array([float(obs_loglik(pt, v[t])) for pt in parts])
        mx = ll.max()
        if not np.isfinite(mx):
            raise ValueError("observation likelihood returned all -inf")
        lw = ll - mx
        w = np.exp(lw)
        tot = w.sum()
        if tot <= 0 or not np.isfinite(tot):
            raise ValueError("particle degeneracy: zero total weight")
        w /= tot
        loglik += mx + math.log(tot / n_particles)
        ess[t] = 1.0 / float(w @ w)
        if ess[t] < resample_frac * n_particles:
            parts = parts[systematic(w)]
            w = np.full(n_particles, 1.0 / n_particles)
        means[t] = float(w @ parts)
        vars_[t] = float(w @ ((parts - means[t]) ** 2))
    return {
        "mean": means,
        "variance": vars_,
        "ess": ess,
        "loglik": np.array([loglik]),
    }
