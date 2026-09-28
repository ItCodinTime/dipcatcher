"""Quantile VAR (QVAR): VAR estimated at a conditional quantile.

Chavleishvili & Manganelli; standard quantile-regression VAR: each equation
is fit by minimizing the pinball loss at quantile tau over its own lag
block, then the system is propagated for quantile impulse responses and
quantile forecasts (VaR-style conditional tails).

- ``qvar_fit``: per-equation quantile regression at ``tau`` via iterative
  reweighted least squares on the pinball gradient (BFGS on a smoothed
  check-loss) — robust, no LP dependency.
- ``qvar_irf``: generalized IRF at quantile tau (shock added to the
  state, conditional quantile paths vs the unconditional baseline).
- ``qvar_forecast``: iterate the fitted companion form for conditional
  tau-quantile paths.

Fail-closed: tau in (0,1), shapes/finite checks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def _check_panel(y: Array, p: int, tau: float) -> tuple[Array, int, int]:
    yy = np.asarray(y, dtype=float)
    if yy.ndim != 2:
        raise ValueError("y must be (T, N)")
    t, n = yy.shape
    if not np.isfinite(yy).all():
        raise ValueError("non-finite input")
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    if p < 1 or t <= p * n + p + 2:
        raise ValueError("need p >= 1 and enough observations")
    return yy, t, n


def _design(yy: Array, p: int) -> tuple[Array, Array]:
    t, n = yy.shape
    x = np.column_stack(
        [np.ones(t - p)] + [yy[p - lag_i - 1 : t - lag_i - 1] for lag_i in range(p)]
    )
    return x, yy[p:]


def _pinball_loss(b: Array, x: Array, y: Array, tau: float, smooth: float) -> float:
    r = y - x @ b
    # Huber-smoothed pinball keeps BFGS well-conditioned at the kink
    quad = np.where(np.abs(r) <= smooth, r * r / (2 * smooth), np.abs(r) - smooth / 2)
    return float(np.sum(np.where(r >= 0, tau * quad, (1 - tau) * quad)))


def qvar_fit(
    y: Array,
    p: int = 1,
    tau: float = 0.5,
    smooth: float = 1e-4,
) -> dict[str, Array | float]:
    """Fit quantile VAR. B rows: [intercept, A_1' .. A_p']; shape (1+Np, N)."""
    yy, t, n = _check_panel(y, p, tau)
    if not np.isfinite(smooth) or smooth <= 0:
        raise ValueError("smooth must be positive")
    x, dep = _design(yy, p)
    kx = x.shape[1]
    b = np.zeros((kx, n))
    for i in range(n):
        b0 = np.linalg.lstsq(x, dep[:, i], rcond=None)[0]
        res = minimize(
            _pinball_loss,
            b0,
            args=(x, dep[:, i], tau, smooth),
            method="BFGS",
            options={"maxiter": 500},
        )
        b[:, i] = res.x
    fitted = x @ b
    resid = dep - fitted
    return {
        "B": b,
        "resid": resid,
        "tau": float(tau),
        "n_obs": t - p,
        "loss": float((np.maximum(resid, 0) * tau + np.maximum(-resid, 0) * (1 - tau)).mean()),
    }


def qvar_forecast(fit_B: Array, history: Array, p: int, n_steps: int) -> Array:
    """Iterate fitted quantile regression: deterministic tau-path."""
    b = np.asarray(fit_B, dtype=float)
    hist = np.asarray(history, dtype=float)
    n = b.shape[1]
    k = b.shape[0] - 1
    if k % n or hist.shape[0] < p or hist.shape[1] != n:
        raise ValueError("history must be (>=p, N) consistent with B")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    f = np.zeros((n * p, n * p))
    for lag in range(p):
        f[:n, lag * n : (lag + 1) * n] = b[1 + lag * n : 1 + (lag + 1) * n].T
    f[n:, :-n] = np.eye(n * (p - 1))
    const = np.zeros(n * p)
    const[:n] = b[0]
    state = np.concatenate([hist[-1 - lag_i] for lag_i in range(p)])
    out = np.zeros((n_steps, n))
    for h in range(n_steps):
        state = const + f @ state
        out[h] = state[:n]
    return out


def qvar_irf(
    y: Array,
    p: int = 1,
    tau: float = 0.1,
    shock: int = 0,
    size: float = 1.0,
    n_steps: int = 20,
) -> Array:
    """Quantile IRF: forecast path with variable ``shock`` shifted by
    ``size`` sigma at the first horizon vs the base path."""
    yy, _, n = _check_panel(y, p, tau)
    if not 0 <= shock < n:
        raise ValueError("shock index out of range")
    fit = qvar_fit(yy, p, tau)
    bmat = np.asarray(fit["B"], dtype=float)
    hist = yy[-p:].copy()
    sd = np.std(yy[:, shock])
    hist_up = hist.copy()
    hist_up[-1, shock] += size * sd
    base = qvar_forecast(bmat, hist, p, n_steps)
    up = qvar_forecast(bmat, hist_up, p, n_steps)
    return (up - base) / max(sd, 1e-12)
