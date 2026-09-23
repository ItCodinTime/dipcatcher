"""Extreme-value theory: tail-index estimators, POT/GPD, GEV, dependence.

All estimators operate on the *upper tail* of the supplied series — pass
losses (``-returns``) or positive exceedances.  Functions fail closed on
degenerate input; estimators that are well-posed but ill-conditioned on the
observed tail return ``nan`` rather than raising.

References:
- Hill (1975). A simple general approach to inference about the tail of a
  distribution. *Ann. Statist.* 3(5).
- Pickands (1975). Statistical inference using extreme order statistics.
  *Ann. Statist.* 3(1).
- Dekkers, Einmahl, de Haan (1989). A moment estimator for the index of an
  extreme-value distribution. *Ann. Statist.* 17(4).
- Balkema, de Haan (1974) / Pickands (1975): GPD limit for excesses (POT).
- Davison, Smith (1990). Models for exceedances over high thresholds. *JRSS-B*.
- Embrechts, Klüppelberg, Mikosch (1997). *Modelling Extremal Events*.
- Davis, Mikosch (2009). The extremogram. *Bernoulli* 15(4).
- Poon, Rockinger, Tawn (2004). Extreme value dependence in financial markets.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_vector(x: Array, name: str = "x", *, min_obs: int = 16) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < min_obs:
        raise ValueError(f"{name} must contain at least {min_obs} finite observations")
    return v


def _top_order_stats(x: Array, k: int) -> Array:
    """Descending top-k order statistics, fail-closed on k range."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer")
    n = x.size
    if k < 2 or k >= n - 1:
        raise ValueError(f"k must be in [2, {n - 2}]")
    return np.sort(x)[::-1]


def hill_estimator(x: Array, k: int = 20) -> float:
    """Hill (1975) tail-index estimate ``gamma`` over the top ``k`` order stats.

    Requires strictly positive tail values (log-spaced); pass ``-returns`` or
    positive exceedances.  ``gamma > 0`` indicates a regularly varying tail.
    """
    v = _as_vector(x)
    order = _top_order_stats(v, k)
    top = order[:k]
    threshold = order[k]
    if np.any(top <= 0.0) or threshold <= 0.0:
        raise ValueError("hill_estimator requires positive tail observations")
    logs = np.log(top) - np.log(threshold)
    return float(np.mean(logs))


def pickands_estimator(x: Array, k: int = 20) -> float:
    """Pickands (1975) tail-index estimate from order stats k, 2k, 4k."""
    v = _as_vector(x)
    n = v.size
    if isinstance(k, bool) or not isinstance(k, int) or k < 1 or 4 * k >= n:
        raise ValueError(f"k must be a positive integer with 4k < n={n}")
    order = np.sort(v)[::-1]
    a, b, c = order[k - 1], order[2 * k - 1], order[4 * k - 1]
    denom = b - c
    if denom <= 0.0:
        return float("nan")
    num = a - b
    if num <= 0.0:
        return float("nan")
    return float(np.log(num / denom) / np.log(2.0))


def dekkers_moments(x: Array, k: int = 20) -> float:
    """Dekkers–Einmahl–de Haan (1989) moments estimator of the EV index.

    Defined for both signs of ``gamma`` (unlike Hill) so it handles thin tails.
    Requires positive tail observations for the log transforms.
    """
    v = _as_vector(x)
    order = _top_order_stats(v, k)
    top = order[:k]
    threshold = order[k]
    if np.any(top <= 0.0) or threshold <= 0.0:
        raise ValueError("dekkers_moments requires positive tail observations")
    logs = np.log(top) - np.log(threshold)
    m1 = float(np.mean(logs))
    m2 = float(np.mean(logs**2))
    if m2 <= 0.0:
        return float("nan")
    correction = 1.0 - m1**2 / m2
    if correction == 0.0:
        return float("nan")
    return m1 + 1.0 - 0.5 / correction


def gpd_fit(x: Array, threshold: float | None = None) -> dict[str, float]:
    """Peaks-over-threshold GPD MLE (Balkema–de Haan / Davison–Smith).

    Excesses over ``threshold`` (default: 90th percentile) are fit with
    ``scipy.stats.genpareto`` (loc fixed at 0).  Returns shape ``xi``, scale
    ``sigma``, threshold ``u``, exceedance count and the empirical exceedance
    probability ``phi_u = N_u / n`` needed for tail VaR/ES.
    """
    v = _as_vector(x)
    u = float(np.quantile(v, 0.9)) if threshold is None else float(threshold)
    excess = v[v > u] - u
    if excess.size < 10:
        raise ValueError("gpd_fit requires at least 10 exceedances")
    xi, _, sigma = sstats.genpareto.fit(excess, floc=0.0)
    return {
        "xi": float(xi),
        "sigma": float(sigma),
        "u": u,
        "n_exceedances": float(excess.size),
        "phi_u": float(excess.size / v.size),
        "mean_excess": float(excess.mean()),
    }


def gpd_var_es(
    xi: float, sigma: float, u: float, phi_u: float, alpha: float = 0.99
) -> dict[str, float]:
    """Tail VaR/ES under the fitted GPD tail (Embrechts et al. 1997, ch. 6.5).

    ``VaR_p = u + sigma/xi * ((alpha/phi_u)^xi - 1)`` for ``alpha`` above the
    threshold coverage ``1 - phi_u``.  ES follows from the GPD mean-excess
    identity ``ES = (VaR + sigma - xi*u) / (1 - xi)``, valid for ``xi < 1``.
    """
    for name, val in (("xi", xi), ("sigma", sigma), ("u", u), ("phi_u", phi_u), ("alpha", alpha)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite")
    if sigma <= 0.0 or not (0.0 < phi_u <= 1.0) or not (0.0 < alpha < 1.0):
        raise ValueError("sigma>0, 0<phi_u<=1 and 0<alpha<1 required")
    if 1.0 - alpha >= phi_u:
        # alpha below threshold: GPD quantile not applicable.
        raise ValueError("alpha must exceed the threshold coverage 1 - phi_u")
    ratio = (1.0 - alpha) / phi_u
    var = u + (sigma / xi) * (ratio ** (-xi) - 1.0) if xi != 0.0 else u - sigma * np.log(ratio)
    es = (var + sigma - xi * u) / (1.0 - xi) if xi < 1.0 else float("nan")
    return {"var": float(var), "es": float(es)}


def gev_fit(x: Array, block: int = 21) -> dict[str, float]:
    """Block-maxima GEV fit (Fisher–Tippett/Gnedenko; scipy ``genextreme``).

    ``scipy.stats.genextreme`` uses the *negated* shape convention
    (``c = -xi``); the returned ``xi`` is restored to the standard EV
    convention where ``xi > 0`` is Fréchet (heavy tail).
    """
    v = _as_vector(x)
    if isinstance(block, bool) or not isinstance(block, int) or block < 2:
        raise ValueError("block must be an integer >= 2")
    n_blocks = v.size // block
    if n_blocks < 8:
        raise ValueError("gev_fit requires at least 8 complete blocks")
    maxima = v[: n_blocks * block].reshape(n_blocks, block).max(axis=1)
    c, loc, scale = sstats.genextreme.fit(maxima)
    return {
        "xi": float(-c),
        "mu": float(loc),
        "sigma": float(scale),
        "n_blocks": float(n_blocks),
        "block": float(block),
    }


def gev_return_level(xi: float, mu: float, sigma: float, period_blocks: float) -> float:
    """GEV T-block return level ``mu + sigma/xi * ((-ln(1-1/T))^-xi - 1)``."""
    for name, val in (("xi", xi), ("mu", mu), ("sigma", sigma), ("T", period_blocks)):
        if not np.isfinite(val):
            raise ValueError(f"{name} must be finite")
    if sigma <= 0.0 or period_blocks <= 1.0:
        raise ValueError("sigma>0 and T>1 required")
    y = -np.log(1.0 - 1.0 / period_blocks)
    if xi == 0.0:
        return float(mu - sigma * np.log(y))
    return float(mu + (sigma / xi) * (y ** (-xi) - 1.0))


def empirical_tail_dependence(x: Array, y: Array, q: float = 0.95) -> dict[str, float]:
    """Empirical upper/lower tail dependence (Poon–Rockinger–Tawn 2004).

    ``lambda_u = P(y > Q_y(q) | x > Q_x(q))`` and symmetrically for the lower
    tail; estimates converge to the copula tail-dependence coefficients as
    ``q -> 1`` for a bivariate regularly varying pair.
    """
    a = _as_vector(x, "x", min_obs=32)
    b = _as_vector(y, "y", min_obs=32)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if not (0.5 < q < 1.0):
        raise ValueError("q must be in (0.5, 1)")
    qx_u, qy_u = np.quantile(a, q), np.quantile(b, q)
    qx_l, qy_l = np.quantile(a, 1.0 - q), np.quantile(b, 1.0 - q)
    hi_x, hi_y = a > qx_u, b > qy_u
    lo_x, lo_y = a < qx_l, b < qy_l
    lam_u = float(np.mean(hi_y[hi_x])) if hi_x.any() else float("nan")
    lam_l = float(np.mean(lo_y[lo_x])) if lo_x.any() else float("nan")
    return {
        "lambda_upper": lam_u,
        "lambda_lower": lam_l,
        "n_upper": float(hi_x.sum()),
        "n_lower": float(lo_x.sum()),
    }


def extremogram(x: Array, threshold: float | None = None, max_lag: int = 10) -> Array:
    """Davis–Mikosch (2009) extremogram ``P(x_{t+h} > a | x_t > a)``, h=1..max_lag.

    ``a`` defaults to the 95th percentile.  Returns an array of length
    ``max_lag``; lags without joint exceedances give ``nan``.
    """
    v = _as_vector(x, min_obs=max_lag + 32)
    a = float(np.quantile(v, 0.95)) if threshold is None else float(threshold)
    if not np.isfinite(a):
        raise ValueError("threshold must be finite")
    exceed = v > a
    if exceed.sum() < 2:
        raise ValueError("too few exceedances for extremogram")
    out = np.full(max_lag, np.nan)
    base = exceed.mean()
    for h in range(1, max_lag + 1):
        joint = float(np.mean(exceed[:-h] & exceed[h:]))
        out[h - 1] = joint / base if base > 0.0 else np.nan
    return out


def mean_excess(x: Array, n_thresholds: int = 20) -> tuple[Array, Array]:
    """Mean-excess function over a grid of thresholds (Davison–Smith diagnostic).

    Returns ``(thresholds, mean_excess_values)``; a linear-in-u mean excess
    supports the GPD tail assumption.  Thresholds span the 50th–97th percentiles.
    """
    v = _as_vector(x)
    if isinstance(n_thresholds, bool) or not isinstance(n_thresholds, int) or n_thresholds < 3:
        raise ValueError("n_thresholds must be an integer >= 3")
    grid = np.quantile(v, np.linspace(0.5, 0.97, n_thresholds))
    me = np.array([float(np.mean(v[v > u] - u)) if (v > u).any() else np.nan for u in grid])
    return grid, me


def hill_plot(x: Array, ks: Array | None = None) -> tuple[Array, Array]:
    """Hill estimates over a grid of ``k`` (Hill-plot diagnostic)."""
    v = _as_vector(x)
    if ks is None:
        ks = np.arange(5, max(6, v.size // 4), dtype=float)
    ks = np.asarray(ks, dtype=int).reshape(-1)
    ks = ks[(ks >= 2) & (ks < v.size - 1)]
    if ks.size == 0:
        raise ValueError("no admissible k for hill_plot")
    return ks, np.array([hill_estimator(v, int(k)) for k in ks])


def moment_ratio_plot_stat(x: Array, k: int = 20) -> float:
    """Dekkers–Einmahl–de Haan moment ratio ``M2 / (2 * M1)`` — >1 signals
    a heavy (xi > 0) tail under the second-order expansion."""
    v = _as_vector(x)
    order = _top_order_stats(v, k)
    top = order[:k]
    threshold = order[k]
    if np.any(top <= 0.0) or threshold <= 0.0:
        raise ValueError("moment ratio requires positive tail observations")
    logs = np.log(top) - np.log(threshold)
    m1 = float(np.mean(logs))
    m2 = float(np.mean(logs**2))
    return m2 / (2.0 * m1) if m1 > 0.0 else float("nan")
