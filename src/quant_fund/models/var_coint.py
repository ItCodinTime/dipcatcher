"""Vector autoregression, cointegration, and connectedness.

VAR(p) estimation with information criteria, impulse responses and FEVD,
the Diebold–Yilmaz connectedness index, Engle–Granger/Johansen cointegration,
VECM estimation, and spread half-life.

References:
- Sims (1980) VAR; Lutkepohl (2005) companion form / IRF / FEVD.
- Pesaran, Shin (1998) generalized IRF/FEVD (ordering-invariant).
- Diebold, Yilmaz (2012) connectedness index from generalized FEVD.
- Engle, Granger (1987) two-step cointegration test.
- Johansen (1991/1995) trace and max-eigenvalue tests; MacKinnon-Haug-
    Michelis (1999) response-surface critical values (approximate).
- VECM: beta via reduced-rank regression (Johansen MLE).
- Half-life: -ln(2)/ln(rho) from AR(1) on the spread.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_panel(y: Array, name: str = "y") -> Array:
    m = np.asarray(y, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2 or m.shape[0] < 10 or not np.all(np.isfinite(m)):
        raise ValueError(f"{name} must be a finite (t, n) matrix, t >= 10")
    if np.any(np.std(m, axis=0) <= 0.0):
        raise ValueError(f"{name} has a constant column")
    return m


def var_fit(y: Array, p: int = 1) -> dict[str, Array]:
    """VAR(p) via OLS: ``y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + u_t``.

    Returns coefficient matrices (p, n, n), intercept (n,), residual
    covariance (n, n), residuals (t-p, n), and log-likelihood-based
    information criteria (aic/bic/hq) for THIS p.
    """
    m = _as_panel(y)
    t, n = m.shape
    if p < 1 or p >= t // 4:
        raise ValueError("p must be >= 1 and << t")
    rows = t - p
    x = np.ones((rows, 1 + n * p))
    for lag in range(1, p + 1):
        x[:, 1 + (lag - 1) * n : 1 + lag * n] = m[p - lag : t - lag]
    tgt = m[p:]
    b, *_ = np.linalg.lstsq(x, tgt, rcond=None)
    resid = tgt - x @ b
    sigma = resid.T @ resid / rows
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("residual covariance not positive definite")
    k = n * n * p + n
    ic = {
        "aic": logdet + 2.0 * k / rows,
        "bic": logdet + k * math.log(rows) / rows,
        "hq": logdet + 2.0 * k * math.log(math.log(rows)) / rows,
    }
    a = b[1:].T.reshape(n, p, n).transpose(1, 0, 2)  # (p, n, n): A_lag @ y
    return {
        "A": a,
        "const": b[0],
        "sigma": sigma,
        "resid": resid,
        "ic": np.array([ic["aic"], ic["bic"], ic["hq"]]),
        "p": np.array([float(p)]),
    }


def var_select_order(y: Array, p_max: int = 8) -> dict[str, Array]:
    """Select VAR lag order minimizing AIC/BIC/HQ over 1..p_max."""
    m = _as_panel(y)
    if p_max < 1 or p_max >= m.shape[0] // 4:
        raise ValueError("p_max out of range")
    ics = np.array([var_fit(m, p)["ic"] for p in range(1, p_max + 1)])
    return {
        "aic": np.array([float(np.argmin(ics[:, 0]) + 1)]),
        "bic": np.array([float(np.argmin(ics[:, 1]) + 1)]),
        "hq": np.array([float(np.argmin(ics[:, 2]) + 1)]),
        "ic_paths": ics,
    }


def var_irf(fit: dict[str, Array], horizon: int = 20) -> dict[str, Array]:
    """Impulse responses: generalized (Pesaran–Shin) and Cholesky.

    Returns arrays (horizon+1, n, n): psi[h, i, j] = response of variable i
    to a unit shock in j. GIRF normalizes by sigma_jj; OIRF uses Cholesky.
    """
    a = np.asarray(fit["A"], dtype=float)
    sigma = np.asarray(fit["sigma"], dtype=float)
    p, n = a.shape[0], a.shape[1]
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    psi = np.zeros((horizon + 1, n, n))
    psi[0] = np.eye(n)
    for h in range(1, horizon + 1):
        for lag in range(1, min(p, h) + 1):
            psi[h] += a[lag - 1] @ psi[h - lag]
    chol = np.linalg.cholesky(sigma)
    sd = np.sqrt(np.diag(sigma))
    oirf = np.einsum("hij,jk->hik", psi, chol)
    girf = np.einsum("hij,jk->hik", psi, sigma) / sd[None, None, :]
    return {"girf": girf, "oirf": oirf, "psi": psi}


def fevd(fit: dict[str, Array], horizon: int = 10) -> dict[str, Array]:
    """Generalized FEVD (Pesaran–Shin): share of variable i's h-step
    forecast error variance attributable to shock j. Rows sum to 1 after
    normalization.
    """
    irf = var_irf(fit, horizon)
    girf, psi = irf["girf"], irf["psi"]
    sigma = np.asarray(fit["sigma"], dtype=float)
    # theta_ij(h) = sigma_jj^{-1} sum_l (e_i' Psi_l Sigma e_j)^2 / mse_i;
    # girf[h,i,j] = (Psi_l Sigma)_ij / sd_j already carries the sigma_jj^{-1}.
    num = np.sum(girf**2, axis=0)  # (n, n)
    mse = np.zeros(sigma.shape[0])
    for h in range(horizon + 1):
        mse += np.diag(psi[h] @ sigma @ psi[h].T)
    gf = num / mse[:, None]
    gf /= gf.sum(axis=1, keepdims=True)
    return {"gfevd": gf, "mse": mse}


def diebold_yilmaz(y: Array, p: int = 2, horizon: int = 10) -> dict[str, Array]:
    """Diebold–Yilmaz (2012) connectedness from a VAR(p).

    Returns the generalized-FEVD connectedness table, total spillover
    index, net directional connectedness per variable, and to/from sums.
    """
    m = _as_panel(y)
    fit = var_fit(m, p)
    gf = fevd(fit, horizon)["gfevd"]
    n = m.shape[1]
    off = gf - np.diag(np.diag(gf))
    to = off.sum(axis=0)  # column sums: shock j's contribution to others
    frm = off.sum(axis=1)  # row sums: i's variance explained by others
    net = to - frm
    tsi = float(np.mean(frm) / 1.0)  # mean share from others, in [0,1] scale
    # DY total spillover = sum of off-diagonals / n (each row sums to 1).
    tsi = float(off.sum() / n)
    return {
        "table": gf,
        "tsi": np.array([tsi]),
        "to": to,
        "from": frm,
        "net": net,
    }


def engle_granger(y: Array, x: Array, max_lag: int | None = None) -> dict[str, float]:
    """Engle–Granger (1987) two-step cointegration test.

    Regresses y on x (levels), then runs an ADF test (no deterministic
    terms) on the residual. p-value via MacKinnon (2010) EG response
    surface for the 2-variable case (tau coefficients, T>=50 approx).
    """
    yv = np.asarray(y, dtype=float).reshape(-1)
    xv = np.asarray(x, dtype=float).reshape(-1)
    if yv.size != xv.size or yv.size < 30 or not np.all(np.isfinite(yv)):
        raise ValueError("y and x must be finite equal-length >= 30")
    t = yv.size
    xm = np.column_stack([np.ones(t), xv])
    beta, *_ = np.linalg.lstsq(xm, yv, rcond=None)
    res = yv - xm @ beta
    dy = np.diff(res)
    rows = t - 1
    if max_lag is None:
        max_lag = int(12.0 * (rows / 100.0) ** 0.25)
    lags = min(max_lag, rows // 4)
    # ADF: dy_i on [res_i, dy_{i-1}, ..., dy_{i-lags}], i = lags..T-2.
    tgt = dy[lags:]
    xr = np.column_stack([res[lags:-1]] + [dy[lags - j : dy.size - j] for j in range(1, lags + 1)])
    b, *_ = np.linalg.lstsq(xr, tgt, rcond=None)
    e = tgt - xr @ b
    s2 = float(e @ e) / (tgt.size - xr.shape[1])
    cov = np.linalg.pinv(xr.T @ xr) * s2
    tau = float(b[0] / math.sqrt(max(cov[0, 0], 1e-16)))
    # MacKinnon EG-2 criticals (asymptotic): -3.34 (1%), -2.76 (5%), -2.45 (10%)
    # finite-T downward adjustment ~ -(1/T)*(c1 + c2/T): use -6.5/T shift approx.
    adj = -6.5 / t
    return {
        "tau": tau,
        "cv_5pct": -2.76 + adj,
        "beta": beta[1],
        "resid": res,
        "lags": float(lags),
    }


def johansen_test(y: Array, p: int = 2, det: int = 0) -> dict[str, Array]:
    """Johansen (1991) trace and max-eigenvalue cointegration tests.

    ``det``: 0 = no deterministic terms in levels; 1 = constant.
    Critical values: MacKinnon–Haug–Michelis (1999) asymptotic tables
    (5% and 1% for det=0, r<=10). For det=1 a constant-shift approximation
    is applied.
    """
    m = _as_panel(y)
    t, n = m.shape
    if p < 1 or p >= t // 4:
        raise ValueError("p out of range")
    if det not in (0, 1):
        raise ValueError("det must be 0 or 1")
    dy = np.diff(m, axis=0)
    y_lag = m[:-1]
    rows = t - 1
    # Residualize dy and y_lag on [Δy_{-1..-(p-1)}, (const)] — regressors are
    # aligned to the last rows - (p-1) observations.
    lag_rows = rows - (p - 1)
    xr = np.ones((lag_rows, 1)) if det == 1 else np.empty((lag_rows, 0))
    for lag in range(1, p):
        xr = np.column_stack([xr, dy[p - 1 - lag : p - 1 - lag + lag_rows]])
    dyy = dy[p - 1 :]
    yll = y_lag[p - 1 :]
    if xr.shape[1] > 0:
        bx, *_ = np.linalg.lstsq(xr, dyy, rcond=None)
        r0 = dyy - xr @ bx
        by, *_ = np.linalg.lstsq(xr, yll, rcond=None)
        r1 = yll - xr @ by
    else:
        r0, r1 = dyy, yll
    s00 = r0.T @ r0 / lag_rows
    s01 = r0.T @ r1 / lag_rows
    s11 = r1.T @ r1 / lag_rows
    # Johansen solves |lam*S11 - S10 S00^{-1} S01| = 0 — a generalized
    # eigenproblem (symmetric A vs spd B), NOT eigvalsh on the product.
    a_mat = s01.T @ np.linalg.pinv(s00) @ s01
    lam = np.sort(sla.eigh(a_mat, s11)[0])[::-1]
    lam = np.clip(lam, 0.0, 1.0 - 1e-12)
    trace = np.array([-lag_rows * np.sum(np.log(1.0 - lam[r:])) for r in range(n)])
    maxe = np.array([-lag_rows * math.log(1.0 - lam[r]) for r in range(n)])
    # MHM (1999) asymptotic 5% trace CVs (det=0): r0..r4 for n-r in 1..5+.
    cv_trace5 = np.array([3.76, 9.24, 15.41, 22.85, 31.52, 41.30, 52.20, 64.28, 77.48, 91.78])
    cv_max5 = np.array([3.76, 11.22, 17.66, 24.90, 32.88, 41.50, 50.94, 61.05, 72.08, 84.09])
    shift = 3.0 if det == 1 else 0.0  # constant shifts the distribution right
    dim = np.arange(1, n + 1)
    cv_t = np.array([cv_trace5[min(d - 1, 9)] + shift for d in dim])
    cv_m = np.array([cv_max5[min(d - 1, 9)] + shift for d in dim])
    return {
        "trace": trace,
        "max_eig": maxe,
        "eigvals": lam,
        "cv_trace5": cv_t,
        "cv_max5": cv_m,
        "rank_trace": np.array([float(np.sum(trace < cv_t))], dtype=float),
        "rank_max": np.array([float(np.sum(maxe < cv_m))], dtype=float),
    }


def vecm_fit(y: Array, p: int = 2, rank: int = 1) -> dict[str, Array]:
    """VECM via Johansen reduced-rank regression.

    ``dy_t = alpha beta' y_{t-1} + sum Gamma_i dy_{t-i} + u_t`` estimated by
    concentrating out short-run dynamics, then reduced-rank regression.
    Returns alpha (n x r), beta (n x r), Gammas, and adjustment strength.
    """
    m = _as_panel(y)
    t, n = m.shape
    if rank < 1 or rank >= n:
        raise ValueError("rank must be in [1, n)")
    if p < 1 or p >= t // 4:
        raise ValueError("p out of range")
    dy = np.diff(m, axis=0)
    rows = t - 1 - (p - 1)
    xr = np.empty((rows, 0))
    for lag in range(1, p):
        xr = np.column_stack([xr, dy[p - 1 - lag : p - 1 - lag + rows]])
    dyy = dy[p - 1 :]
    yll = m[p - 1 : t - 1]
    if xr.shape[1] > 0:
        bx, *_ = np.linalg.lstsq(xr, dyy, rcond=None)
        r0 = dyy - xr @ bx
        by, *_ = np.linalg.lstsq(xr, yll, rcond=None)
        r1 = yll - xr @ by
    else:
        r0, r1 = dyy, yll
    s00 = r0.T @ r0 / rows
    s01 = r0.T @ r1 / rows
    s11 = r1.T @ r1 / rows
    a_mat = s01.T @ np.linalg.pinv(s00) @ s01
    vals, vecs = sla.eigh(a_mat, s11)
    order = np.argsort(vals)[::-1]
    vecs_r = vecs[:, order[:rank]]
    # Normalize beta (Johansen normalization beta' S11 beta = I).
    norm = np.linalg.cholesky(vecs_r.T @ s11 @ vecs_r)
    beta = vecs_r @ np.linalg.inv(norm)
    alpha = s01 @ beta
    ect = r1 @ beta  # error-correction term
    # Estimate short-run gammas given the EC term.
    xfull = np.column_stack([ect, xr]) if xr.shape[1] else ect
    bf, *_ = np.linalg.lstsq(xfull, dyy, rcond=None)
    resid = dyy - xfull @ bf
    gamma = bf[rank:].reshape(n, p - 1, n).transpose(1, 0, 2) if p > 1 else np.zeros((0, n, n))
    return {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "ect": ect,
        "resid": resid,
        "eigvals": vals[order],
    }


def spread_half_life(spread: Array) -> float:
    """Half-life of mean reversion: ``-ln(2)/ln(rho)`` from AR(1) on spread."""
    v = np.asarray(spread, dtype=float).reshape(-1)
    if v.size < 10 or not np.all(np.isfinite(v)):
        raise ValueError("spread must be finite length >= 10")
    x = np.column_stack([np.ones(v.size - 1), v[:-1]])
    b, *_ = np.linalg.lstsq(x, v[1:], rcond=None)
    rho = float(b[1])
    if rho <= 0.0:
        return np.inf
    if rho >= 1.0:
        return np.inf
    return -math.log(2.0) / math.log(rho)


def granger_causality_matrix(y: Array, p: int = 2) -> dict[str, Array]:
    """Pairwise Granger F-tests: entry [i, j] is the p-value for
    'variable j Granger-causes variable i'."""
    m = _as_panel(y)
    t, n = m.shape
    if p < 1 or p >= t // 4:
        raise ValueError("p out of range")
    pmat = np.full((n, n), np.nan)
    fmat = np.full((n, n), np.nan)
    rows = t - p
    for i in range(n):
        xr = np.column_stack([np.ones(rows)] + [m[p - lag : t - lag, i] for lag in range(1, p + 1)])
        tgt = m[p:, i]
        b0, *_ = np.linalg.lstsq(xr, tgt, rcond=None)
        rss0 = float(((tgt - xr @ b0) ** 2).sum())
        for j in range(n):
            if i == j:
                continue
            xu = np.column_stack([xr] + [m[p - lag : t - lag, j] for lag in range(1, p + 1)])
            b1, *_ = np.linalg.lstsq(xu, tgt, rcond=None)
            rss1 = float(((tgt - xu @ b1) ** 2).sum())
            f = ((rss0 - rss1) / p) / (rss1 / (rows - xr.shape[1] - p))
            fmat[i, j] = f
            pmat[i, j] = float(sstats.f.sf(f, p, rows - xr.shape[1] - p))
    return {"pvalues": pmat, "fstats": fmat}
