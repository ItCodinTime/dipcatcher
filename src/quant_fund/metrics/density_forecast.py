"""Density-forecast evaluation via the probability integral transform.

References:
- Rosenblatt (1952); Dawid (1984): PIT — correct density forecasts
  produce iid Uniform(0,1) transforms.
- Berkowitz (2001): censored-normal transform + LR test for iid N(0,1)
  of z = Phi^{-1}(PIT).
- Diebold, Gunther & Tay (1998): PIT histogram diagnostics.
- Knuppel (2015): Berkowitz test cautions for OOS evaluation.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _pit(pits: Array, n: int = 50) -> Array:
    p = np.asarray(pits, dtype=float).reshape(-1)
    if p.size < n or not np.all(np.isfinite(p)):
        raise ValueError(f"PIT series must be finite with length >= {n}")
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("PIT values must lie in the open interval (0, 1)")
    return p


def pit_histogram(pits: Array, bins: int = 10) -> dict[str, Array | float]:
    """Diebold–Gunther–Tay (1998) PIT histogram + chi-square uniformity.

    Returns bin shares, the chi-square statistic, and its p-value."""
    p = _pit(pits)
    if not (5 <= bins <= 50):
        raise ValueError("bins must be in [5, 50]")
    counts, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    exp = p.size / bins
    chi2 = float(np.sum((counts - exp) ** 2 / exp))
    return {
        "shares": counts / p.size,
        "edges": edges,
        "chi2": chi2,
        "pvalue": float(1.0 - stats.chi2.cdf(chi2, bins - 1)),
    }


def berkowitz_test(pits: Array, tail_censor: float | None = None) -> dict[str, float]:
    """Berkowitz (2001) LR test: z = Phi^{-1}(PIT) is iid N(0,1) iff the
    density forecast is correct.

    Fits (mu, sigma, rho) of an AR(1) on z and compares to (0, 1, 0);
    LR ~ chi2(3). With ``tail_censor`` = c in (0, 0.5), z values beyond
    Phi^{-1}(1-c) are censored at the boundary (Berkowitz's censoring
    for tail-focused evaluation)."""
    p = _pit(pits)
    z = stats.norm.ppf(np.clip(p, 1e-12, 1 - 1e-12))
    n = z.size
    cens = tail_censor is not None
    tc = float(tail_censor) if tail_censor is not None else 0.0
    if cens:
        if not (0.0 < tc < 0.5):
            raise ValueError("tail_censor must be in (0, 0.5)")
        zmax = float(stats.norm.ppf(1.0 - tc))
        zmin = float(stats.norm.ppf(tc))
        z = np.clip(z, zmin, zmax)

    def ll(mu: float, sig: float, rho: float) -> float:
        sig = max(abs(sig), 1e-6)
        # AR(1): z_t = mu + rho (z_{t-1} - mu_last) ... use conditional form
        # z_t | z_{t-1} ~ N(mu + rho(z_{t-1} - mu), sig^2) for uncensored.
        out = 0.0
        prev = z[0]
        for i in range(1, n):
            zi = z[i]
            m = mu + rho * (prev - mu)
            if cens and (zi >= zmax or zi <= zmin):
                # censored contribution: log survival of the tail
                u = (zmax - m) / sig
                out += math.log(max(stats.norm.sf(u), 1e-300))
            else:
                d = (zi - m) / sig
                out += -0.5 * d * d - math.log(sig) - 0.5 * math.log(2 * math.pi)
            prev = zi
        return out

    # Unrestricted fit (mu, sigma, rho).
    from scipy import optimize as opt

    res = opt.minimize(
        lambda th: -ll(th[0], abs(th[1]) + 1e-4, np.clip(th[2], -0.99, 0.99)),
        np.array([z.mean(), z.std(), 0.0]),
        method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-5},
    )
    ll1 = -res.fun
    ll0 = ll(0.0, 1.0, 0.0)
    lr = float(max(-2.0 * (ll0 - ll1), 0.0))
    return {
        "statistic": lr,
        "pvalue": float(1.0 - stats.chi2.cdf(lr, 3)),
        "mu_hat": float(res.x[0]),
        "sigma_hat": float(abs(res.x[1]) + 1e-4),
        "rho_hat": float(np.clip(res.x[2], -0.99, 0.99)),
        "n": float(n),
    }


def pit_autocorrelation(pits: Array, lags: int = 5) -> dict[str, Array | float]:
    """Ljung–Box on z = Phi^{-1}(PIT) — detects residual dependence that
    uniform marginal tests miss (Diebold et al. 1998)."""
    p = _pit(pits)
    z = stats.norm.ppf(np.clip(p, 1e-12, 1 - 1e-12))
    n = z.size
    if not (1 <= lags <= n // 4):
        raise ValueError("lags must be in [1, n/4]")
    zc = z - z.mean()
    denom = float(zc @ zc)
    if denom <= 0:
        raise ValueError("degenerate transformed PIT")
    acf = np.array([float(np.dot(zc[k:], zc[:-k]) / denom) for k in range(1, lags + 1)])
    lb = float(n * (n + 2) * np.sum(acf * acf / (n - np.arange(1, lags + 1))))
    return {
        "acf": acf,
        "ljung_box": lb,
        "pvalue": float(1.0 - stats.chi2.cdf(lb, lags)),
        "lags": float(lags),
    }
