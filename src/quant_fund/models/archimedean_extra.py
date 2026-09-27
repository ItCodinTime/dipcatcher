"""Frank and Joe Archimedean copulas (complementing Clayton/Gumbel).

- **Frank** (Frank 1979) is the only Archimedean copula that is radially
  symmetric and admits negative as well as positive dependence.  Its Kendall's
  tau is ``tau = 1 - 4/theta + 4 D_1(theta)/theta`` with the order-1 Debye
  function ``D_1``; parameters are fitted by inverting tau.
- **Joe** (Joe 1990) has upper-tail dependence and ``theta >= 1``; it is fitted
  by maximum likelihood on its analytic density.

Both provide the CDF and a conditional-inversion sampler.  Fail-closed on
invalid parameters or non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import kendalltau

Array = NDArray[np.float64]

_EPS = 1e-9


def _as_uv(u: Array) -> Array:
    arr = np.asarray(u, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 10 or not np.isfinite(arr).all():
        raise ValueError("u must be a finite (n, 2) array of pseudo-observations")
    return np.asarray(np.clip(arr, _EPS, 1.0 - _EPS), dtype=float)


# ------------------------------------------------------------------------ Frank
def frank_tau(theta: float) -> float:
    """Kendall's tau for the Frank copula with parameter ``theta``."""
    if abs(theta) < 1e-8:
        return 0.0
    integ, _ = quad(lambda t: t / np.expm1(t), 0.0, theta)
    d1 = integ / theta
    return float(1.0 - 4.0 / theta + 4.0 * d1 / theta)


def frank_cdf(u: Array, v: Array, theta: float) -> Array:
    """Frank copula CDF."""
    if abs(theta) < 1e-8:
        return np.asarray(np.asarray(u, dtype=float) * np.asarray(v, dtype=float), dtype=float)
    a = np.expm1(-theta)  # e^{-theta} - 1
    gu = np.expm1(-theta * np.asarray(u, dtype=float))
    gv = np.expm1(-theta * np.asarray(v, dtype=float))
    return np.asarray(-1.0 / theta * np.log1p(gu * gv / a), dtype=float)


def frank_fit(u: Array) -> dict[str, float]:
    """Fit Frank ``theta`` by inverting sample Kendall's tau."""
    uv = _as_uv(u)
    tau = float(kendalltau(uv[:, 0], uv[:, 1])[0])
    if abs(tau) < 1e-4:
        return {"theta": 0.0, "tau": tau}
    theta = float(brentq(lambda th: frank_tau(th) - tau, -50.0, 50.0, xtol=1e-8))
    return {"theta": theta, "tau": tau}


def frank_sim(theta: float, n: int, rng: np.random.Generator | None = None) -> Array:
    """Sample the Frank copula by conditional inversion."""
    gen = np.random.default_rng() if rng is None else rng
    u = gen.random(n)
    w = gen.random(n)
    if abs(theta) < 1e-8:
        return np.column_stack([u, w])
    out_v = np.empty(n)
    a = np.exp(-theta) - 1.0
    for i in range(n):
        eu = np.exp(-theta * u[i])

        def cond(v: float, eu: float = eu, wi: float = w[i]) -> float:
            gv = np.exp(-theta * v) - 1.0
            return float(eu * gv / (a + (eu - 1.0) * gv) - wi)

        out_v[i] = brentq(cond, _EPS, 1.0 - _EPS, xtol=1e-9)
    return np.column_stack([u, out_v])


# -------------------------------------------------------------------------- Joe
def joe_cdf(u: Array, v: Array, theta: float) -> Array:
    """Joe copula CDF (``theta >= 1``)."""
    if theta < 1.0:
        raise ValueError("Joe copula requires theta >= 1")
    a = (1.0 - np.asarray(u, dtype=float)) ** theta
    b = (1.0 - np.asarray(v, dtype=float)) ** theta
    return 1.0 - (a + b - a * b) ** (1.0 / theta)


def joe_pdf(u: Array, v: Array, theta: float) -> Array:
    """Joe copula density."""
    if theta < 1.0:
        raise ValueError("Joe copula requires theta >= 1")
    uu = np.clip(np.asarray(u, dtype=float), _EPS, 1.0 - _EPS)
    vv = np.clip(np.asarray(v, dtype=float), _EPS, 1.0 - _EPS)
    a = (1.0 - uu) ** theta
    b = (1.0 - vv) ** theta
    s = a + b - a * b
    return np.asarray(
        s ** (1.0 / theta - 2.0) * ((1.0 - uu) * (1.0 - vv)) ** (theta - 1.0) * (theta - 1.0 + s),
        dtype=float,
    )


def joe_fit(u: Array) -> dict[str, float]:
    """Fit Joe ``theta`` by maximum likelihood."""
    uv = _as_uv(u)

    def nll(theta: float) -> float:
        val = joe_pdf(uv[:, 0], uv[:, 1], theta)
        if not np.all(val > 0):
            return 1e12
        return -float(np.log(val).sum())

    res = minimize_scalar(nll, bounds=(1.0001, 20.0), method="bounded")
    return {"theta": float(res.x), "loglik": float(-res.fun)}


def joe_sim(theta: float, n: int, rng: np.random.Generator | None = None) -> Array:
    """Sample the Joe copula by conditional inversion."""
    if theta < 1.0:
        raise ValueError("Joe copula requires theta >= 1")
    gen = np.random.default_rng() if rng is None else rng
    u = gen.random(n)
    w = gen.random(n)
    out_v = np.empty(n)
    for i in range(n):
        au = (1.0 - u[i]) ** (theta - 1.0)

        def cond(v: float, ui: float = u[i], au: float = au, wi: float = w[i]) -> float:
            a = (1.0 - ui) ** theta
            b = (1.0 - v) ** theta
            s = a + b - a * b
            return float(au * (1.0 - b) * s ** (1.0 / theta - 1.0) - wi)

        out_v[i] = brentq(cond, _EPS, 1.0 - _EPS, xtol=1e-9)
    return np.column_stack([u, out_v])
