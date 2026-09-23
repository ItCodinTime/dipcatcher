"""GAS / score-driven volatility (Creal, Koopman & Lucas 2013).

GAS(1,1): f_{t+1} = omega + A u_t + B f_t, where u_t is the scaled
score of the observation density w.r.t. f_t. Implemented scalings:

- ``inv_sqrt``: u_t = (dlogL/df_t) / sqrt(I_t), Fisher info at f_t.
  For Gaussian data u_t = (y^2/f - 1)/sqrt(2).
- ``unit``: u_t = 2 f_t (dlogL/df_t), the CKL scaled-score form.
  Gaussian: u_t = y^2/f - 1.  Student-t:
  u_t = (nu+1) y^2 / ((nu-2) f + y^2) - 1.

Densities: Gaussian (f = variance) and Student-t (f = variance of the
standardized t, nu estimated). Fail-closed: non-finite input, nu too
small, degenerate paths.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.special import gammaln

Array = NDArray[np.float64]


def _score_gauss(y: float, f: float) -> tuple[float, float]:
    """(score dlogL/df, Fisher info I) for y ~ N(0, f)."""
    s = (y * y - f) / (2.0 * f * f)
    info = 1.0 / (2.0 * f * f)
    return s, info


def _ll_gauss(y: Array, f: Array) -> float:
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * f) + y * y / f))


def _score_t(y: float, f: float, nu: float) -> tuple[float, float]:
    """Score and a numerical info proxy for standardized t with var f.

    Density: y = sqrt(f) z, z ~ t_nu / sqrt(nu/(nu-2)).
    """
    z2 = y * y / f
    a = nu - 2.0
    s = -1.0 / (2.0 * f) + (nu + 1.0) * z2 / (2.0 * f * (a + z2))
    info = nu / (2.0 * f * f * (nu + 3.0)) + 1.0 / (4.0 * f * f * max(nu - 4.0, 1.0))
    return s, max(info, 1e-12)


def _ll_t(y: Array, f: Array, nu: float) -> float:
    z2 = y * y / f
    a = nu - 2.0
    c = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(np.pi * a)
    ll = c - 0.5 * np.log(f) - 0.5 * (nu + 1.0) * np.log1p(z2 / a)
    return float(np.sum(ll))


def gas_vol_fit(
    y: Array,
    dist: str = "gauss",
    scaling: str = "inv_sqrt",
) -> dict[str, Array | float]:
    """Fit GAS(1,1) volatility by MLE.

    Returns omega, A, B, nu (t only), f path, scaled scores, loglik.
    """
    yy = np.asarray(y, dtype=float).ravel()
    if yy.size < 100 or not np.isfinite(yy).all():
        raise ValueError("y must be finite, >= 100 obs")
    if dist not in ("gauss", "t"):
        raise ValueError("dist must be gauss|t")
    if scaling not in ("unit", "inv_sqrt"):
        raise ValueError("scaling must be unit|inv_sqrt")
    n = yy.size
    v0 = float(yy.var())

    def path(theta: Array) -> tuple[Array, Array] | None:
        omega, A, B = float(theta[0]), float(theta[1]), float(theta[2])
        nu = float(theta[3]) if dist == "t" else None
        if B < 0 or B >= 0.999 or omega <= 0:
            return None
        # unit-scaled scores are bounded below by -1: omega > A keeps f > 0
        if scaling == "unit" and omega <= A:
            return None
        if dist == "t" and (nu is None or nu <= 4.05 or nu > 100.0):
            return None
        f = np.empty(n)
        u = np.empty(n)
        f[0] = max(v0, 1e-8)
        u[0] = 0.0
        for t in range(1, n):
            fp = f[t - 1]
            s, info = _score_t(yy[t - 1], fp, nu) if nu is not None else _score_gauss(yy[t - 1], fp)
            u[t - 1] = s / np.sqrt(info) if scaling == "inv_sqrt" else s * 2.0 * fp
            f[t] = omega + A * u[t - 1] + B * f[t - 1]
            if not np.isfinite(f[t]) or f[t] <= 0:
                return None
        return f, u

    def nll(theta: Array) -> float:
        out = path(theta)
        if out is None:
            return 1e12
        f, _ = out
        if dist == "t":
            return -_ll_t(yy, f, float(theta[3]))
        return -_ll_gauss(yy, f)

    theta0 = np.array(
        [0.05 * v0, 0.02 * v0 if scaling == "unit" else 0.1, 0.9] + ([8.0] if dist == "t" else [])
    )
    bounds = [(1e-10, None), (0.0, 1.0), (0.0, 0.999)] + ([(4.05, 100.0)] if dist == "t" else [])
    res = optimize.minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
    out = path(res.x)
    if out is None or not np.isfinite(res.fun):
        raise ValueError("GAS fit failed")
    f, u = out
    ret: dict[str, Array | float] = {
        "omega": float(res.x[0]),
        "A": float(res.x[1]),
        "B": float(res.x[2]),
        "f": f,
        "u": u,
        "vol": np.sqrt(f),
        "loglik": float(-res.fun),
        "converged": float(res.success),
    }
    if dist == "t":
        ret["nu"] = float(res.x[3])
    return ret


def gas_vol_forecast(fit: dict[str, Array | float], y_last: float, dist: str = "gauss") -> float:
    """One-step-ahead variance forecast f_{T+1}."""
    f = np.asarray(fit["f"], dtype=float)
    omega, A, B = float(fit["omega"]), float(fit["A"]), float(fit["B"])
    nu = float(fit.get("nu", 8.0))
    y = float(y_last)
    s, info = _score_t(y, f[-1], nu) if dist == "t" else _score_gauss(y, f[-1])
    u_last = s / np.sqrt(info)
    return float(omega + A * u_last + B * f[-1])
