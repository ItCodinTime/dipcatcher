"""Threshold autoregression: Tong SETAR(2;p;d) and the Hansen (1999)
sup-Wald threshold test.

SETAR: y_t = phi_1' z_{t-1} if y_{t-d} <= gamma, phi_2' z_{t-1}
otherwise, z_{t-1} = (1, y_{t-1}, ..., y_{t-p})'. Estimated by
conditional least squares over a trimmed grid of candidate thresholds.

Hansen (1999): H0 is linear AR(p); the sup-Wald (equivalently sup-F)
statistic has a nonstandard null — the bootstrap p-value resamples
residuals from the fitted H0 model and recurses the series.

Fail-closed: insufficient effective sample, empty regimes after trim,
non-finite inputs all raise ``ValueError``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _design(y: Array, p: int, delay: int) -> tuple[Array, Array, Array]:
    """Return (y_dep, Z, thresh_var) aligned so all rows share indices."""
    n = y.size
    m = max(p, delay)
    n_eff = n - m
    if n_eff <= p + 5:
        raise ValueError("effective sample too small")
    y_dep = y[m:]
    z = np.ones((n_eff, p + 1))
    for lag in range(1, p + 1):
        z[:, lag] = y[m - lag : n - lag]
    v = y[m - delay : n - delay]
    return y_dep, z, v


def _ols(y: Array, z: Array) -> tuple[Array, float]:
    beta, *_ = np.linalg.lstsq(z, y, rcond=None)
    resid = y - z @ beta
    return beta, float(resid @ resid)


def _regime_ssr(y: Array, z: Array, v: Array, gamma: float) -> tuple[float, Array, Array] | None:
    lo = v <= gamma
    hi = ~lo
    if lo.sum() < z.shape[1] + 1 or hi.sum() < z.shape[1] + 1:
        return None
    b1, s1 = _ols(y[lo], z[lo])
    b2, s2 = _ols(y[hi], z[hi])
    return s1 + s2, b1, b2


def setar_fit(
    y: Array,
    p: int = 1,
    delay: int = 1,
    gamma_grid: Array | None = None,
    trim: float = 0.15,
) -> dict[str, Array | float]:
    """Conditional-least-squares SETAR(2; p; d) fit over a gamma grid.

    ``trim`` keeps each regime at >= max(trim share, p+2) observations.
    Returns dict: gamma, coef_regime1/2, ssr, ssr_linear, f_stat,
    n_regime1/2, grid, grid_ssr.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 30 or not np.isfinite(y).all():
        raise ValueError("y must be finite, >= 30 obs")
    if p < 1 or delay < 1 or not 0.0 < trim < 0.5:
        raise ValueError("p, delay >= 1 and trim in (0, 0.5)")
    y_dep, z, v = _design(y, p, delay)
    n_eff = y_dep.size
    min_regime = max(int(np.ceil(trim * n_eff)), p + 2)

    _, ssr0 = _ols(y_dep, z)

    if gamma_grid is None:
        lo, hi = np.quantile(v, [trim, 1.0 - trim])
        grid = np.unique(v[(v >= lo) & (v <= hi)])
    else:
        grid = np.unique(np.asarray(gamma_grid, dtype=float).ravel())
    if grid.size < 2:
        raise ValueError("empty candidate threshold grid")

    ssr_g = np.full(grid.size, np.inf)
    best: tuple[float, Array, Array] | None = None
    best_g = float("nan")
    for i, g in enumerate(grid):
        r = _regime_ssr(y_dep, z, v, float(g))
        if r is None or min((v <= g).sum(), (v > g).sum()) < min_regime:
            continue
        ssr_g[i] = r[0]
        if best is None or r[0] < best[0]:
            best = r
            best_g = float(g)
    if best is None:
        raise ValueError("no feasible threshold split")

    ssr_best, b1, b2 = best
    f_stat = float(n_eff * (ssr0 - ssr_best) / max(ssr_best, 1e-18))
    return {
        "gamma": best_g,
        "coef_regime1": b1,
        "coef_regime2": b2,
        "ssr": float(ssr_best),
        "ssr_linear": float(ssr0),
        "f_stat": f_stat,
        "n_regime1": float((v <= best_g).sum()),
        "n_regime2": float((v > best_g).sum()),
        "grid": grid,
        "grid_ssr": ssr_g,
        "thresh_var": v,
        "p": float(p),
        "delay": float(delay),
    }


def hansen_threshold_test(
    y: Array,
    p: int = 1,
    delay: int = 1,
    n_boot: int = 200,
    trim: float = 0.15,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Hansen (1999) bootstrap test of H0: linear AR(p) vs SETAR(2;p;d).

    Statistic: F = sup_gamma n (S0 - S(g))/S(g). Bootstrap: resample
    residuals of the fitted linear AR(p), recurse, recompute F.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 30 or not np.isfinite(y).all():
        raise ValueError("y must be finite, >= 30 obs")
    if n_boot < 20:
        raise ValueError("n_boot >= 20 for a meaningful bootstrap")
    gen = rng if rng is not None else np.random.default_rng(0)

    fit = setar_fit(y, p=p, delay=delay, trim=trim)
    f_obs = float(fit["f_stat"])

    y_dep, z, _ = _design(y, p, delay)
    b0, ssr0 = _ols(y_dep, z)
    resid = y_dep - z @ b0
    resid = resid - resid.mean()

    m = max(p, delay)
    count = 0
    for _ in range(n_boot):
        e = gen.choice(resid, size=y_dep.size, replace=True)
        # recurse AR(p) under H0 on the original scale
        y_sim = np.empty(y.size)
        y_sim[:m] = y[:m]
        e_full = np.concatenate([np.zeros(m), e])
        for t in range(m, y.size):
            zt = np.concatenate([[1.0], y_sim[t - np.arange(1, p + 1)]])
            y_sim[t] = float(zt @ b0) + e_full[t]
        try:
            f_sim = float(setar_fit(y_sim, p=p, delay=delay, trim=trim)["f_stat"])
        except ValueError:
            continue
        count += int(f_sim >= f_obs)
    pvalue = float((count + 1.0) / (n_boot + 1.0))
    return {
        "f_stat": f_obs,
        "pvalue": pvalue,
        "gamma": float(fit["gamma"]),
        "ssr_linear": float(fit["ssr_linear"]),
        "ssr_threshold": float(fit["ssr"]),
        "n_boot": float(n_boot),
    }


def setar_predict(fit: dict[str, Array | float], y_history: Array, steps: int = 1) -> Array:
    """Recursive h-step SETAR forecast from a fitted model dict."""
    y = np.asarray(y_history, dtype=float).ravel()
    p = int(fit["p"])
    delay = int(fit["delay"])
    gamma = float(fit["gamma"])
    b1 = np.asarray(fit["coef_regime1"], dtype=float)
    b2 = np.asarray(fit["coef_regime2"], dtype=float)
    if y.size < max(p, delay) or not np.isfinite(y).all():
        raise ValueError("y_history too short or non-finite")
    if steps < 1:
        raise ValueError("steps >= 1")
    hist = np.concatenate([y, np.empty(steps)])
    for h in range(steps):
        t = y.size + h
        zt = np.concatenate([[1.0], hist[t - np.arange(1, p + 1)]])
        v_t = hist[t - delay]
        hist[t] = float(zt @ (b1 if v_t <= gamma else b2))
    return np.asarray(hist[y.size :], dtype=float)
