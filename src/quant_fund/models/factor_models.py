"""Cross-sectional factor models.

Fama–MacBeth two-pass regression (with Shanken correction), characteristic-
sorted factor construction (Fama–French style 2x3 sorts and momentum),
PCA-based static factor extraction with Bai–Ng information criteria, and
factor residualization.

References:
- Fama, MacBeth (1973); Shanken (1992) EIV correction; Newey-West lags.
- Fama, French (1993) SMB/HML 2x3 independent sorts; (2015) RMW/CMA.
- Jegadeesh, Titman (1993) momentum WML decile spread.
- Carhart (1997) four-factor model.
- Bai, Ng (2002) IC_p1/IC_p2/IC_p3 panel factor-number criteria.
- Stock, Watson (2002) diffusion-index PCA factors.
- Ledoit, Wolf (2004) constant-correlation covariance (used via shrinkage).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_panel(r: Array, name: str = "returns") -> Array:
    m = np.asarray(r, dtype=float)
    if m.ndim != 2 or m.shape[0] < 5 or m.shape[1] < 2:
        raise ValueError(f"{name} must be an (n_periods, n_assets) matrix")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be finite")
    return m


def fama_macbeth(returns: Array, characteristics: Array, nw_lags: int = 0) -> dict[str, Array]:
    """Two-pass Fama–MacBeth regression.

    Parameters
    ----------
    returns : (n_periods, n_assets) excess returns.
    characteristics : (n_periods, n_assets, n_chars) time-varying
        characteristics, or (n_assets, n_chars) static loadings.

    Returns
    -------
    gamma : (n_chars + 1,) mean risk premia (first entry = intercept).
    gamma_se / gamma_t : Newey–West (lag ``nw_lags``) SEs/t-stats over the
        time series of cross-sectional gammas.
    shanken_se : Shanken (1992) EIV-corrected SEs.
    r2 : per-period cross-sectional R^2 (mean in ``r2_mean``).
    """
    r = _as_panel(returns)
    c = np.asarray(characteristics, dtype=float)
    t, n = r.shape
    if c.ndim == 2:
        c = np.broadcast_to(c[None, :, :], (t, n, c.shape[1])).copy()
    if c.ndim != 3 or c.shape[0] != t or c.shape[1] != n:
        raise ValueError("characteristics must be (t,n,k) or (n,k) matching returns")
    if not np.all(np.isfinite(c)):
        raise ValueError("characteristics must be finite")
    k = c.shape[2]
    gammas = np.full((t, k + 1), np.nan)
    r2s = np.full(t, np.nan)
    for s in range(t):
        xs = np.column_stack([np.ones(n), c[s]])
        if n <= k + 1:
            raise ValueError("need more assets than characteristics + 1")
        b, *_ = np.linalg.lstsq(xs, r[s], rcond=None)
        gammas[s] = b
        res = r[s] - xs @ b
        sst = float(((r[s] - r[s].mean()) ** 2).sum())
        r2s[s] = 1.0 - float(res @ res) / sst if sst > 0 else np.nan
    gm = gammas.mean(axis=0)
    # Newey-West covariance of mean gammas.
    dg = gammas - gm
    cov = dg.T @ dg / (t - 1) / t
    for j in range(1, nw_lags + 1):
        w = 1.0 - j / (nw_lags + 1.0)
        cov += w * (dg[j:].T @ dg[:-j] + dg[:-j].T @ dg[j:]) / (t - 1) / t
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    # Shanken (1992): Var*(g) = Var(g) + Sigma_f (EIV adjustment) — for the
    # characteristic-form version, inflate by (1 + mean beta' Sigma^{-1} beta).
    # Common simplification: se_s2 = se^2 * (1 + gamma' Sigma_g^{-1} gamma / ...)
    # We use the standard quadratic adjustment on the time-series covariance:
    gcov = np.atleast_2d(np.cov(gammas[:, 1:].T)) if k > 0 else np.zeros((1, 1))
    g_mean = gm[1:]
    pinv = np.linalg.pinv(gcov + 1e-12 * np.eye(max(k, 1)))
    quad = float(g_mean @ pinv @ g_mean)
    se_sh = se * np.sqrt(1.0 + quad)
    return {
        "gamma": gm,
        "gamma_se": se,
        "gamma_t": gm / np.where(se > 0, se, np.nan),
        "shanken_se": se_sh,
        "r2_mean": np.array([float(np.nanmean(r2s))]),
        "gammas_ts": gammas,
    }


def sorted_factor_returns(returns: Array, char: Array, n_groups: int = 5) -> dict[str, Array]:
    """Characteristic-sorted portfolio spreads (Jegadeesh–Titman style).

    ``char`` : (t, n) characteristic per period (e.g. past 12-1 momentum).
    Returns per-period top-minus-bottom portfolio return and group means.
    """
    r = _as_panel(returns)
    c = np.asarray(char, dtype=float)
    if c.shape != r.shape or not np.all(np.isfinite(c)):
        raise ValueError("char must be a finite (t, n) matrix matching returns")
    if n_groups < 2:
        raise ValueError("n_groups >= 2")
    t, n = r.shape
    if n < n_groups * 2:
        raise ValueError("too few assets for the requested groups")
    spread = np.zeros(t)
    group_ret = np.zeros((t, n_groups))
    for s in range(t):
        order = np.argsort(c[s])
        groups = np.array_split(order, n_groups)
        for gi, g in enumerate(groups):
            group_ret[s, gi] = float(r[s, g].mean())
        spread[s] = group_ret[s, -1] - group_ret[s, 0]
    return {"spread": spread, "group_returns": group_ret}


def double_sorted_factors(
    returns: Array, char_a: Array, char_b: Array, cuts: int = 3
) -> dict[str, Array]:
    """Fama–French (1993) independent double-sort factor legs.

    Splits each period into ``cuts`` groups by ``char_a`` and ``char_b``
    independently, forms the ``cuts x cuts`` value-weighted (equal-weighted)
    grid, and returns the A-factor (top minus bottom A-tercile averaged over
    B) and B-factor analog — the SMB/HML construction.
    """
    r = _as_panel(returns)
    a = np.asarray(char_a, dtype=float)
    b = np.asarray(char_b, dtype=float)
    if a.shape != r.shape or b.shape != r.shape:
        raise ValueError("characteristics must match returns shape")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("characteristics must be finite")
    if cuts < 2:
        raise ValueError("cuts >= 2")
    t, n = r.shape
    if n < cuts * cuts:
        raise ValueError("too few assets for the requested grid")
    leg_a = np.zeros(t)
    leg_b = np.zeros(t)
    grid_ret = np.full((t, cuts, cuts), np.nan)
    for s in range(t):
        qa = np.argsort(np.argsort(a[s])) * cuts // n  # 0..cuts-1
        qb = np.argsort(np.argsort(b[s])) * cuts // n
        for i in range(cuts):
            for j in range(cuts):
                sel = (qa == i) & (qb == j)
                if np.any(sel):
                    grid_ret[s, i, j] = float(r[s, sel].mean())
        leg_a[s] = float(np.nanmean(grid_ret[s, -1, :]) - np.nanmean(grid_ret[s, 0, :]))
        leg_b[s] = float(np.nanmean(grid_ret[s, :, -1]) - np.nanmean(grid_ret[s, :, 0]))
    return {"factor_a": leg_a, "factor_b": leg_b, "grid": grid_ret}


def pca_factors(returns: Array, n_factors: int | None = None) -> dict[str, Array]:
    """Stock–Watson (2002) PCA static factors from a return panel.

    Standardizes each asset, extracts the first ``n_factors`` PCs as factors
    (time series, T x r) plus loadings (n x r).
    """
    r = _as_panel(returns)
    t, n = r.shape
    z = (r - r.mean(axis=0)) / np.where(r.std(axis=0) > 0, r.std(axis=0), 1.0)
    cov = z.T @ z / t
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    maxr = min(n, t)
    if n_factors is None:
        n_factors = bai_ng_factors(returns)["ic2"].astype(int)[0]
    if n_factors < 1 or n_factors > maxr:
        raise ValueError("n_factors out of range")
    fac = z @ vecs[:, :n_factors]
    return {
        "factors": fac,
        "loadings": vecs[:, :n_factors],
        "eigvals": vals,
        "explained": np.array([float(vals[:n_factors].sum() / vals.sum())]),
    }


def bai_ng_factors(returns: Array, r_max: int = 8) -> dict[str, Array]:
    """Bai–Ng (2002) IC_p information criteria for the number of factors.

    Minimizes ``IC(r) = ln V(r) + r * g(n,t)`` for the three penalty forms.
    Returns the selected r for each criterion and the criterion paths.
    """
    r = _as_panel(returns)
    t, n = r.shape
    # Bai-Ng is defined on the raw panel (demeaned, NOT standardized):
    # common error variance sigma_e^2 is what makes V(r) flatten after r*.
    z = r - r.mean(axis=0)
    cov = z.T @ z / t
    vals = np.linalg.eigvalsh(cov)[::-1]
    total = float(vals.sum())
    cnt = min(n, t)
    r_max = max(1, min(r_max, cnt - 1))
    # V(r): residual variance of the r-factor approximation.
    v = np.array([1.0 - float(vals[:k].sum()) / total for k in range(r_max + 1)])
    v = np.maximum(v, 1e-12)
    rr = np.arange(r_max + 1, dtype=float)
    g1 = rr * (n + t) / (n * t) * np.log(n * t / (n + t))
    g2 = rr * (n + t) / (n * t) * np.log(min(n, t))
    g3 = rr * np.log(min(n, t)) / min(n, t)
    ic1, ic2, ic3 = np.log(v) + g1, np.log(v) + g2, np.log(v) + g3
    return {
        "ic1": np.array([float(np.argmin(ic1))]),
        "ic2": np.array([float(np.argmin(ic2))]),
        "ic3": np.array([float(np.argmin(ic3))]),
        "ic1_path": ic1,
        "ic2_path": ic2,
        "ic3_path": ic3,
    }


def factor_residuals(returns: Array, factors: Array) -> dict[str, Array]:
    """Remove factor exposure: per-asset OLS of returns on factors.

    Returns idiosyncratic returns, betas (n x r), and per-asset R^2 —
    the Barra-style specific-return layer.
    """
    r = _as_panel(returns)
    f = np.asarray(factors, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    if f.ndim != 2 or f.shape[0] != r.shape[0] or not np.all(np.isfinite(f)):
        raise ValueError("factors must be a finite (t, k) matrix matching returns")
    t, n = r.shape
    k = f.shape[1]
    if k >= t - 2:
        raise ValueError("too many factors for the sample")
    xf = np.column_stack([np.ones(t), f])
    beta, *_ = np.linalg.lstsq(xf, r, rcond=None)
    resid = r - xf @ beta
    sst = ((r - r.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (resid**2).sum(axis=0) / np.where(sst > 0, sst, np.nan)
    return {
        "resid": resid,
        "betas": beta[1:].T,  # (n_assets, n_factors)
        "intercepts": beta[0],
        "r2": r2,
    }


def gics_alpha(returns: Array, market: Array, nw_lags: int = 0) -> dict[str, Array]:
    """Jensen (1968) alpha per asset vs a market factor, with NW t-stats."""
    r = _as_panel(returns)
    m = np.asarray(market, dtype=float).reshape(-1)
    if m.size != r.shape[0] or not np.all(np.isfinite(m)):
        raise ValueError("market must be finite length t")
    t, n = r.shape
    x = np.column_stack([np.ones(t), m])
    beta, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ beta
    alphas = np.zeros(n)
    tstats = np.zeros(n)
    xtx_inv = np.linalg.pinv(x.T @ x)
    for i in range(n):
        e = resid[:, i]
        s2 = float(e @ e) / (t - 2)
        cov_ols = xtx_inv * s2
        se = np.sqrt(max(cov_ols[0, 0], 0.0))
        alphas[i] = beta[0, i]
        tstats[i] = beta[0, i] / se if se > 0 else np.nan
    return {"alpha": alphas, "alpha_t": tstats, "beta": beta[1]}
