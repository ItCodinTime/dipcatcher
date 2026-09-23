"""Asymmetric GARCH: Nelson (1991) EGARCH and Glosten-Jagannathan-Runkle
(1993) GJR-GARCH, both Gaussian QMLE.

EGARCH(1,1):  ln sig2_t = w + b ln sig2_{t-1} + a z_{t-1} + g (|z_{t-1}|
              - E|z|),  z = r/sig,  E|z| = sqrt(2/pi).
  Leverage: negative returns move vol via g > 0 convention here uses
  a z term (a < 0 = leverage) plus the symmetric |z| innovation g.

GJR-GARCH(1,1): sig2_t = w + (a + g 1{r_{t-1} < 0}) r^2_{t-1}
              + b sig2_{t-1}.

Fail-closed: non-finite input, insufficient obs, stationarity
violations rejected in the fitted-parameter diagnostics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]

_EZ = np.sqrt(2.0 / np.pi)


def _check(y: Array, n_min: int = 100) -> Array:
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < n_min or not np.isfinite(yy).all():
        raise ValueError("y must be finite with >= 100 obs")
    return yy


def _egarch_path(y: Array, w: float, a: float, g: float, b: float, mu: float) -> Array | None:
    n = y.size
    ln_sig2 = np.empty(n)
    ln_sig2[0] = np.log(max(float(np.var(y)), 1e-8))
    for t in range(1, n):
        z = (y[t - 1] - mu) / np.sqrt(np.exp(ln_sig2[t - 1]))
        ln_sig2[t] = w + b * ln_sig2[t - 1] + a * z + g * (abs(z) - _EZ)
        if not np.isfinite(ln_sig2[t]) or abs(ln_sig2[t]) > 40:
            return None
    return np.exp(ln_sig2)


def egarch_fit(y: Array) -> dict[str, Array | float]:
    """EGARCH(1,1) Gaussian QMLE."""
    yy = _check(y)
    v0 = float(yy.var())

    def nll(th: Array) -> float:
        w, a, g, b, mu = (float(v) for v in th)
        if b < 0 or b >= 0.9999:
            return 1e12
        s2 = _egarch_path(yy, w, a, g, b, mu)
        if s2 is None:
            return 1e12
        return float(0.5 * np.sum(np.log(2 * np.pi * s2) + (yy - mu) ** 2 / s2))

    th0 = np.array([0.02 * np.log(v0 + 1e-8) - 0.05, -0.1, 0.1, 0.95, float(yy.mean())])
    res = optimize.minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 6000})
    w, a, g, b, mu = (float(v) for v in res.x)
    s2 = _egarch_path(yy, w, a, g, b, mu)
    if s2 is None:
        raise ValueError("EGARCH fit diverged")
    return {
        "omega": w,
        "alpha": a,
        "gamma": g,
        "beta": b,
        "mu": mu,
        "sig2": s2,
        "vol": np.sqrt(s2),
        "loglik": float(-res.fun),
        "persistence": b,
    }


def _gjr_path(y: Array, w: float, a: float, g: float, b: float, mu: float) -> Array | None:
    n = y.size
    s2 = np.empty(n)
    s2[0] = max(float(np.var(y)), 1e-8)
    for t in range(1, n):
        e = y[t - 1] - mu
        s2[t] = w + (a + g * float(e < 0.0)) * e * e + b * s2[t - 1]
        if not np.isfinite(s2[t]) or s2[t] <= 0:
            return None
    return s2


def gjr_garch_fit(y: Array) -> dict[str, Array | float]:
    """GJR-GARCH(1,1) Gaussian QMLE (leverage: gamma > 0)."""
    yy = _check(y)
    v0 = float(yy.var())

    def nll(th: Array) -> float:
        w, a, g, b, mu = (float(v) for v in th)
        # stationarity: a + g/2 + b < 1
        if w <= 0 or a < 0 or g < 0 or b < 0 or a + 0.5 * g + b >= 0.9999:
            return 1e12
        s2 = _gjr_path(yy, w, a, g, b, mu)
        if s2 is None:
            return 1e12
        return float(0.5 * np.sum(np.log(2 * np.pi * s2) + (yy - mu) ** 2 / s2))

    th0 = np.array([0.05 * v0, 0.02, 0.1, 0.9, float(yy.mean())])
    res = optimize.minimize(nll, th0, method="Nelder-Mead", options={"maxiter": 6000})
    w, a, g, b, mu = (float(v) for v in res.x)
    s2 = _gjr_path(yy, w, a, g, b, mu)
    if s2 is None:
        raise ValueError("GJR fit diverged")
    return {
        "omega": w,
        "alpha": a,
        "gamma": g,
        "beta": b,
        "mu": mu,
        "sig2": s2,
        "vol": np.sqrt(s2),
        "loglik": float(-res.fun),
        "persistence": a + 0.5 * g + b,
    }


def news_impact_curve(fit: dict[str, Array | float], model: str, shocks: Array) -> Array:
    """News impact curve: sig2 (or ln sig2) response to a shock e.

    For GJR: sig2 = w + (a + g 1{e<0}) e^2 + b sig2_bar (sig2_bar =
    unconditional). For EGARCH the argument is the standardized shock
    z = e/sig and the response is exp(ln sig2) evaluated at z.
    """
    eps = np.asarray(shocks, dtype=float).ravel()
    if eps.size == 0 or not np.isfinite(eps).all():
        raise ValueError("shocks must be finite")
    if model == "gjr":
        w = float(fit["omega"])
        a, g = float(fit["alpha"]), float(fit["gamma"])
        s2bar = float(np.asarray(fit["sig2"]).mean())
        return w + (a + g * (eps < 0)) * eps**2 + float(fit["beta"]) * s2bar
    if model == "egarch":
        w, a, g, b = (float(fit[k]) for k in ("omega", "alpha", "gamma", "beta"))
        lnbar = float(np.mean(np.log(np.asarray(fit["sig2"]))))
        return np.exp(w + b * lnbar + a * eps + g * (np.abs(eps) - _EZ))
    raise ValueError("model must be gjr|egarch")
