"""Yield-curve fitting: Nelson-Siegel, Svensson, Diebold-Li dynamics.

Nelson-Siegel (1987):
  y(m) = b0 + b1 f1(m) + b2 f2(m),
  f1 = (1 - e^{-m/lam}) / (m/lam),   f2 = f1 - e^{-m/lam}.

Svensson (1994) adds a second hump term with (b3, lam2).

Diebold-Li (2006): fix lam (they use lam=0.0609 at monthly maturity),
fit betas per date by OLS, then model each beta_t as AR(1) and rebuild
the forecast curve.

Fail-closed: fewer maturities than factors, non-finite yields, lam<=0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def ns_loadings(maturities: Array, lam: float) -> Array:
    """(n,3) Nelson-Siegel loading matrix [1, f1, f2]."""
    m = np.asarray(maturities, dtype=float).ravel()
    if m.size < 3 or not np.isfinite(m).all() or (m <= 0).any():
        raise ValueError("maturities must be finite, > 0, >= 3 points")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be > 0")
    x = m / lam
    e = np.exp(-x)
    f1 = np.where(x > 1e-10, (1.0 - e) / x, 1.0)
    f2 = f1 - e
    return np.column_stack([np.ones_like(m), f1, f2])


def svensson_loadings(maturities: Array, lam1: float, lam2: float) -> Array:
    """(n,4) Svensson loading matrix [1, f1(lam1), f2(lam1), f2(lam2)]."""
    base = ns_loadings(maturities, lam1)
    m = np.asarray(maturities, dtype=float).ravel()
    if not np.isfinite(lam2) or lam2 <= 0.0:
        raise ValueError("lam2 must be > 0")
    x = m / lam2
    e = np.exp(-x)
    f2b = np.where(x > 1e-10, (1.0 - e) / x - e, 0.0)
    return np.column_stack([base, f2b])


def ns_fit(maturities: Array, yields: Array, lam: float | None = None) -> dict[str, Array | float]:
    """Fit Nelson-Siegel to a single curve. If ``lam`` is None, profile-
    least-squares over a lam grid + local refinement."""
    m = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if m.shape != y.shape or m.size < 3:
        raise ValueError("maturities and yields must match, >= 3")
    if not np.isfinite(y).all():
        raise ValueError("yields must be finite")

    def solve(lam_v: float) -> tuple[Array, float]:
        L = ns_loadings(m, lam_v)
        b, *_ = np.linalg.lstsq(L, y, rcond=None)
        r = y - L @ b
        return b, float(r @ r)

    if lam is None:
        grid = np.logspace(-1.5, 1.5, 60)
        best = min(grid, key=lambda lv: solve(lv)[1])
        res = optimize.minimize_scalar(
            lambda lv: solve(lv)[1], bounds=(best / 3.0, best * 3.0), method="bounded"
        )
        lam_hat = float(res.x)
    else:
        lam_hat = float(lam)
    betas, ssr = solve(lam_hat)
    return {
        "beta": betas,
        "lam": lam_hat,
        "ssr": ssr,
        "fitted": ns_loadings(m, lam_hat) @ betas,
        "residuals": y - ns_loadings(m, lam_hat) @ betas,
    }


def svensson_fit(maturities: Array, yields: Array) -> dict[str, Array | float]:
    """Fit Svensson curve; (lam1, lam2) by coarse grid + Nelder-Mead."""
    m = np.asarray(maturities, dtype=float).ravel()
    y = np.asarray(yields, dtype=float).ravel()
    if m.shape != y.shape or m.size < 4:
        raise ValueError("maturities and yields must match, >= 4")
    if not np.isfinite(y).all():
        raise ValueError("yields must be finite")

    def solve(l1: float, l2: float) -> tuple[Array, float]:
        L = svensson_loadings(m, l1, l2)
        b, *_ = np.linalg.lstsq(L, y, rcond=None)
        r = y - L @ b
        return b, float(r @ r)

    def ssr_of(lams: Array) -> float:
        l1, l2 = float(lams[0]), float(lams[1])
        if l1 <= 0 or l2 <= 0 or abs(l1 - l2) < 0.05:
            return 1e12
        return solve(l1, l2)[1]

    best = (0.5, 2.0)
    best_ssr = np.inf
    for l1 in np.linspace(0.1, 1.5, 8):
        for l2 in np.linspace(1.0, 8.0, 8):
            s = ssr_of(np.array([l1, l2]))
            if s < best_ssr:
                best_ssr, best = s, (float(l1), float(l2))
    res = optimize.minimize(ssr_of, np.array(best), method="Nelder-Mead")
    lam1_hat, lam2_hat = float(res.x[0]), float(res.x[1])
    betas, ssr = solve(lam1_hat, lam2_hat)
    L = svensson_loadings(m, lam1_hat, lam2_hat)
    return {
        "beta": betas,
        "lam1": lam1_hat,
        "lam2": lam2_hat,
        "ssr": ssr,
        "fitted": L @ betas,
        "residuals": y - L @ betas,
    }


def diebold_li(
    yield_panel: Array,
    maturities: Array,
    lam: float = 0.0609,
    ar_lags: int = 1,
) -> dict[str, Array | float]:
    """Diebold-Li two-step: per-date OLS betas + AR(1) factor dynamics.

    ``yield_panel`` is (n_dates, n_mats). Returns betas (n_dates,3),
    AR(1) coefficients per factor, and the loading matrix.
    """
    y = np.asarray(yield_panel, dtype=float)
    m = np.asarray(maturities, dtype=float).ravel()
    if y.ndim != 2 or y.shape[1] != m.size or y.shape[0] < 10:
        raise ValueError("yield_panel must be (n_dates>=10, n_mats)")
    if not np.isfinite(y).all():
        raise ValueError("panel must be finite")
    L = ns_loadings(m, lam)
    betas = np.empty((y.shape[0], 3))
    for t in range(y.shape[0]):
        b, *_ = np.linalg.lstsq(L, y[t], rcond=None)
        betas[t] = b
    ar = np.empty((3, 2))
    for j in range(3):
        b_j = betas[:, j]
        z = np.column_stack([np.ones(b_j.size - ar_lags), b_j[:-ar_lags]])
        c, *_ = np.linalg.lstsq(z, b_j[ar_lags:], rcond=None)
        ar[j] = c
    return {
        "betas": betas,
        "ar_coef": ar,
        "loadings": L,
        "lam": float(lam),
        "resid_std": np.asarray([float(np.std(y[:, i] - (L @ betas.T)[i])) for i in range(m.size)]),
    }


def diebold_li_forecast(fit: dict[str, Array | float], horizon: int) -> Array:
    """h-step-ahead beta forecasts -> implied forward curve per horizon."""
    if horizon < 1:
        raise ValueError("horizon >= 1")
    betas = np.asarray(fit["betas"], dtype=float)
    ar = np.asarray(fit["ar_coef"], dtype=float)
    L = np.asarray(fit["loadings"], dtype=float)
    cur = betas[-1].copy()
    outs = []
    for _ in range(horizon):
        cur = ar[:, 0] + ar[:, 1] * cur
        outs.append(cur.copy())
    beta_path = np.asarray(outs)
    return np.asarray(beta_path @ L.T, dtype=float)
