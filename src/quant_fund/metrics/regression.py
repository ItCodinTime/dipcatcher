"""Regression estimation and diagnostics battery.

Pure NumPy/SciPy OLS with heteroskedasticity-consistent and HAC standard
errors, the classic mis-specification diagnostics, robust estimators,
instrumental variables, and one-step GMM.

References:
- Gauss–Markov OLS; White (1980) HC0–HC3; Cribari-Neto (2004) HC4.
- Newey, West (1987) HAC — delegates to ``metrics.hac``.
- Durbin, Watson (1950/1951) serial-correlation statistic.
- Breusch, Godfrey (1978) LM serial-correlation test.
- Breusch, Pagan (1979) / White (1980) heteroskedasticity tests.
- Ramsey (1969) RESET specification test.
- Goldfeld, Quandt (1965) groupwise heteroskedasticity test.
- Brown, Durbin, Evans (1975) CUSUM recursive-residual stability.
- Cook (1977) distance; Belsley, Kuh, Welsch (1980) DFBETAS/VIF.
- Theil (1950)/Sen (1968) slope estimator; Huber (1964) M-estimation.
- Koenker, Bassett (1978) quantile regression (median via LP).
- Sargan (1958)/Hansen (1982) J overidentification test; 2SLS.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_xy(y: Array, x: Array) -> tuple[Array, Array]:
    v = np.asarray(y, dtype=float).reshape(-1)
    m = np.asarray(x, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    if m.ndim != 2 or m.shape[0] != v.size or not np.all(np.isfinite(m)):
        raise ValueError("x must be a finite n x k matrix matching y")
    if v.size < m.shape[1] + 4 or not np.all(np.isfinite(v)):
        raise ValueError("insufficient or non-finite y")
    return v, m


def _with_const(x: Array) -> Array:
    return np.column_stack([np.ones(x.shape[0]), x])


def ols(y: Array, x: Array, add_const: bool = True) -> dict[str, Array]:
    """OLS fit: coefficients, residuals, fitted, R^2, XtX inverse."""
    v, m = _as_xy(y, x)
    if add_const:
        m = _with_const(m)
    beta, *_ = np.linalg.lstsq(m, v, rcond=None)
    resid = v - m @ beta
    fitted = m @ beta
    xtx_inv = np.linalg.pinv(m.T @ m)
    sst = float(((v - v.mean()) ** 2).sum())
    sse = float((resid**2).sum())
    return {
        "beta": beta,
        "resid": resid,
        "fitted": fitted,
        "xtx_inv": xtx_inv,
        "r2": np.array([1.0 - sse / sst if sst > 0 else np.nan]),
        "design": m,
    }


def hc_covariance(
    resid: Array, design: Array, kind: str = "HC3"
) -> Array:
    """White (1980) heteroskedasticity-consistent covariance of OLS betas.

    ``kind`` in {HC0, HC1, HC2, HC3, HC4 (Cribari-Neto 2004)}.
    """
    e = np.asarray(resid, dtype=float).reshape(-1)
    m = np.asarray(design, dtype=float)
    if m.ndim != 2 or m.shape[0] != e.size or not np.all(np.isfinite(m)):
        raise ValueError("design must be finite and match resid")
    n, k = m.shape
    if n <= k:
        raise ValueError("need more rows than parameters")
    xtx_inv = np.linalg.pinv(m.T @ m)
    h = np.einsum("ij,jk,ik->i", m, xtx_inv, m)  # leverage
    kind_l = kind.upper()
    if kind_l == "HC0":
        scale = e**2
    elif kind_l == "HC1":
        scale = e**2 * n / (n - k)
    elif kind_l == "HC2":
        scale = e**2 / np.maximum(1.0 - h, 1e-8)
    elif kind_l == "HC3":
        scale = e**2 / np.maximum(1.0 - h, 1e-8) ** 2
    elif kind_l == "HC4":
        hbar = h.mean()
        delta = np.minimum(4.0, h / max(hbar, 1e-12))
        scale = e**2 / np.maximum(1.0 - h, 1e-8) ** delta
    else:
        raise ValueError(f"unknown HC kind: {kind!r}")
    meat = (m * scale[:, None]).T @ m
    return xtx_inv @ meat @ xtx_inv


def hac_covariance(
    resid: Array, design: Array, lag: int | None = None
) -> Array:
    """Newey–West (1987) HAC covariance of OLS betas (Bartlett kernel)."""
    e = np.asarray(resid, dtype=float).reshape(-1)
    m = np.asarray(design, dtype=float)
    if m.ndim != 2 or m.shape[0] != e.size or not np.all(np.isfinite(m)):
        raise ValueError("design must be finite and match resid")
    n, k = m.shape
    if n <= k:
        raise ValueError("need more rows than parameters")
    if lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    if lag < 0 or lag >= n:
        raise ValueError("lag must be in [0, n)")
    xtx_inv = np.linalg.pinv(m.T @ m)
    meat = np.zeros((k, k))
    for j in range(lag + 1):
        w = 1.0 - j / (lag + 1.0)
        block = (m[j:] * e[j:, None]).T @ (m[: n - j] * e[: n - j, None])
        meat += w * (block + block.T) if j > 0 else block
    return xtx_inv @ meat @ xtx_inv


def ols_summary(
    y: Array, x: Array, se: str = "HC3", hac_lag: int | None = None
) -> dict[str, Array]:
    """Full OLS summary with chosen covariance (OLS|HC0..HC4|HAC)."""
    fit = ols(y, x)
    e, m, b = fit["resid"], fit["design"], fit["beta"]
    n, k = m.shape
    se_l = se.upper()
    if se_l == "OLS":
        s2 = float(e @ e) / (n - k)
        cov = fit["xtx_inv"] * s2
    elif se_l.startswith("HC"):
        cov = hc_covariance(e, m, se_l)
    elif se_l == "HAC":
        cov = hac_covariance(e, m, hac_lag)
    else:
        raise ValueError(f"unknown se kind: {se!r}")
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    t = b / np.where(sd > 0, sd, np.nan)
    p = 2.0 * sstats.t.sf(np.abs(t), df=n - k)
    return {
        "beta": b,
        "se": sd,
        "t": t,
        "p": p,
        "resid": e,
        "r2": fit["r2"],
        "cov": cov,
    }


def durbin_watson(resid: Array) -> float:
    """Durbin–Watson statistic ``sum((e_t - e_{t-1})^2) / sum(e_t^2)``."""
    e = np.asarray(resid, dtype=float).reshape(-1)
    if e.size < 5 or not np.all(np.isfinite(e)):
        raise ValueError("resid must have >= 5 finite values")
    denom = float(e @ e)
    if denom <= 0.0:
        raise ValueError("resid has zero variance")
    return float(np.sum(np.diff(e) ** 2) / denom)


def breusch_godfrey(
    y: Array, x: Array, lags: int = 1
) -> dict[str, float]:
    """Breusch–Godfrey LM test: aux regression of resid on [X, lagged resid]."""
    v, m = _as_xy(y, x)
    m = _with_const(m)
    n, k = m.shape
    if lags < 1 or n <= k + lags + 2:
        raise ValueError("lags >= 1 with sufficient data required")
    e = ols(v, m[:, 1:])["resid"]
    e_lag = np.column_stack(
        [np.concatenate([np.zeros(j), e[:-j]]) for j in range(1, lags + 1)]
    )
    aux = np.column_stack([m, e_lag])
    r2 = float(ols(e, aux[:, 1:], add_const=True)["r2"][0])
    lm = (n - lags) * r2
    return {"lm": lm, "pvalue": float(sstats.chi2.sf(lm, lags)), "lags": float(lags)}


def breusch_pagan(y: Array, x: Array) -> dict[str, float]:
    """Breusch–Pagan (1979) LM: scaled SSR of squared residuals on X."""
    v, m = _as_xy(y, x)
    m = _with_const(m)
    e = ols(v, m[:, 1:])["resid"]
    g = e**2 / float(np.mean(e**2))
    aux = ols(g, m[:, 1:])
    ssr = float(np.sum((aux["fitted"] - g.mean()) ** 2))
    lm = ssr / 2.0
    k = m.shape[1] - 1
    return {"lm": lm, "pvalue": float(sstats.chi2.sf(lm, k))}


def white_test(y: Array, x: Array) -> dict[str, float]:
    """White (1980) general heteroskedasticity test (all squares + cross)."""
    v, m = _as_xy(y, x)
    m = _with_const(m)
    e = ols(v, m[:, 1:])["resid"]
    zcols = []
    for i in range(1, m.shape[1]):
        for j in range(i, m.shape[1]):
            zcols.append(m[:, i] * m[:, j])
    z = np.column_stack(zcols)
    aux = ols(e**2, z)
    n = v.size
    r2 = float(aux["r2"][0])
    lm = n * r2
    df = z.shape[1]
    return {"lm": lm, "pvalue": float(sstats.chi2.sf(lm, df)), "df": float(df)}


def ramsey_reset(
    y: Array, x: Array, powers: tuple[int, ...] = (2, 3)
) -> dict[str, float]:
    """Ramsey (1969) RESET: F-test on powers of fitted values."""
    v, m = _as_xy(y, x)
    m = _with_const(m)
    base = ols(v, m[:, 1:])
    e0 = base["resid"]
    n, k = m.shape
    q = len(powers)
    if q < 1 or n <= k + q:
        raise ValueError("insufficient data for RESET powers")
    z = np.column_stack([base["fitted"] ** p for p in powers])
    aug = np.column_stack([m, z])
    e1 = ols(v, aug[:, 1:])["resid"]
    rss0, rss1 = float(e0 @ e0), float(e1 @ e1)
    if rss1 <= 0.0:
        raise ValueError("augmented model degenerates")
    f = ((rss0 - rss1) / q) / (rss1 / (n - k - q))
    return {"f": f, "pvalue": float(sstats.f.sf(f, q, n - k - q))}


def goldfeld_quandt(y: Array, x: Array, drop: int | None = None) -> dict[str, float]:
    """Goldfeld–Quandt (1965): F = RSS2/RSS1 on first/last thirds."""
    v, m = _as_xy(y, x)
    n = v.size
    d = n // 3 if drop is None else int(drop)
    if d < 0 or 2 * (n - d) // 2 <= m.shape[1] + 1:
        raise ValueError("drop too large for the sample")
    # Standard: split sample into first/last thirds after ordering by X1.
    order = np.argsort(m[:, 0])
    third = (n - d) // 2
    lo_i, hi_i = order[:third], order[-third:]
    e1 = ols(v[lo_i], m[lo_i])["resid"]
    e2 = ols(v[hi_i], m[hi_i])["resid"]
    rss1, rss2 = float(e1 @ e1), float(e2 @ e2)
    if rss1 <= 0.0:
        raise ValueError("first-group RSS is zero")
    f = rss2 / rss1
    df = third - m.shape[1] - 1
    return {"f": f, "pvalue": float(sstats.f.sf(f, df, df)), "df": float(df)}


def cusum_recursive(y: Array, x: Array) -> dict[str, Array]:
    """Brown–Durbin–Evans (1975) CUSUM of standardized recursive residuals.

    Returns the CUSUM path and ±5% significance lines (Anselmo-style linear
    bounds used in practice).
    """
    v, m = _as_xy(y, x)
    m = _with_const(m)
    n, k = m.shape
    rec: list[float] = []
    for t in range(k + 1, n + 1):
        fit = np.linalg.lstsq(m[: t - 1], v[: t - 1], rcond=None)[0]
        pred = float(m[t - 1] @ fit)
        h = float(m[t - 1] @ np.linalg.pinv(m[: t - 1].T @ m[: t - 1]) @ m[t - 1])
        rec.append((v[t - 1] - pred) / math.sqrt(max(1.0 + h, 1e-12)))
    w = np.asarray(rec)
    s = float(w.std(ddof=1))
    if s <= 0.0 or not np.isfinite(s):
        raise ValueError("recursive residuals degenerate")
    cus = np.cumsum(w / s)
    t_axis = np.arange(1, w.size + 1, dtype=float)
    a = 0.948 * math.sqrt(w.size)  # 5% critical lines (BDE constants)
    bound = a * (1.0 + 2.0 * t_axis / w.size)
    return {"cusum": cus, "upper": bound, "lower": -bound, "t": t_axis}


def cusum_sq(y: Array, x: Array) -> dict[str, Array]:
    """Brown–Durbin–Evans CUSUM of squares: ``W_r = sum_{j<=r} w_j^2 / sum w_j^2``.

    Detects parameter instability that signed CUSUM cannot (e.g. variance
    breaks, coefficient sign flips). Boundary: asymptotic Kolmogorov
    approximation ``c0 = 1.36 * sqrt(2/m)`` (BDE 1975 large-m 5% bound).
    """
    v, m = _as_xy(y, x)
    m = _with_const(m)
    n, k = m.shape
    rec: list[float] = []
    for t in range(k + 1, n + 1):
        fit = np.linalg.lstsq(m[: t - 1], v[: t - 1], rcond=None)[0]
        pred = float(m[t - 1] @ fit)
        h = float(m[t - 1] @ np.linalg.pinv(m[: t - 1].T @ m[: t - 1]) @ m[t - 1])
        rec.append((v[t - 1] - pred) / math.sqrt(max(1.0 + h, 1e-12)))
    w = np.asarray(rec)
    denom = float(w @ w)
    if denom <= 0.0:
        raise ValueError("recursive residuals degenerate")
    wsq = np.cumsum(w * w) / denom
    n_m = w.size
    frac = np.arange(1, n_m + 1) / n_m
    c0 = 1.36 * math.sqrt(2.0 / n_m)
    return {
        "w": wsq,
        "frac": frac,
        "upper": frac + c0,
        "lower": frac - c0,
    }


def vif(x: Array) -> Array:
    """Variance-inflation factors for each column of X (no constant col)."""
    m = np.asarray(x, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    n, k = m.shape
    if k < 2:
        return np.ones(k)
    out = np.zeros(k)
    for j in range(k):
        others = np.delete(m, j, axis=1)
        r2 = float(ols(m[:, j], others)["r2"][0])
        out[j] = 1.0 / max(1.0 - r2, 1e-12)
    return out


def cooks_distance(y: Array, x: Array) -> Array:
    """Cook (1977) distance for each observation."""
    v, m = _as_xy(y, x)
    m = _with_const(m)
    fit = ols(v, m[:, 1:])
    e = fit["resid"]
    n, k = m.shape
    xtx_inv = fit["xtx_inv"]
    h = np.einsum("ij,jk,ik->i", m, xtx_inv, m)
    s2 = float(e @ e) / (n - k)
    if s2 <= 0.0:
        raise ValueError("zero residual variance")
    return (e**2 / (k * s2)) * (h / np.maximum(1.0 - h, 1e-12) ** 2)


def theil_sen(x: Array, y: Array) -> dict[str, float]:
    """Theil–Sen median slope for simple regression (with median intercept)."""
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.size != yv.size or xv.size < 4 or not np.all(np.isfinite(xv)):
        raise ValueError("x and y must be finite equal-length >= 4")
    slopes = []
    n = xv.size
    for i in range(n - 1):
        dx = xv[i + 1 :] - xv[i]
        ok = np.abs(dx) > 1e-12
        slopes.extend(((yv[i + 1 :] - yv[i])[ok] / dx[ok]).tolist())
    if not slopes:
        raise ValueError("all x values identical")
    slope = float(np.median(slopes))
    intercept = float(np.median(yv - slope * xv))
    return {"slope": slope, "intercept": intercept}


def huber_regression(
    x: Array, y: Array, c: float = 1.345, max_iter: int = 100
) -> dict[str, Array]:
    """Huber (1964) M-estimation via IRLS (psi clipped at c*sigma)."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    xv = _with_const(xv)
    if xv.shape[0] != yv.size or not np.all(np.isfinite(xv)) or not np.all(np.isfinite(yv)):
        raise ValueError("x/y must be finite and equal-length")
    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        r = yv - xv @ beta
        s = float(np.median(np.abs(r - np.median(r))) * 1.4826)
        if s <= 1e-12:
            break
        u = r / s
        w = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        xw = xv * np.sqrt(w)[:, None]
        yw = yv * np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        if float(np.max(np.abs(beta_new - beta))) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    resid = yv - xv @ beta
    return {"beta": beta, "resid": resid, "iterations": np.array([float(iters + 1)])}


def quantile_regression(
    x: Array, y: Array, tau: float = 0.5
) -> dict[str, Array]:
    """Koenker–Bassett (1978) quantile regression via LP (HiGHS)."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    xv = _with_const(xv)
    n, k = xv.shape
    if xv.shape[0] != yv.size or not np.all(np.isfinite(xv)):
        raise ValueError("x/y must be finite and equal-length")
    if not np.isfinite(tau) or not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1)")
    # vars: [beta (k), u+ (n), u- (n)]; min tau*1'u+ + (1-tau)*1'u-
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    # y = Xb + u+ - u-  ->  Xb + u+ - u- = y
    a_eq = np.column_stack([xv, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * k + [(0.0, None)] * (2 * n)
    res = opt.linprog(
        c, A_eq=a_eq, b_eq=yv, bounds=bounds, method="highs"
    )
    if not res.success:
        raise ValueError(f"quantile regression LP failed: {res.message}")
    beta = res.x[:k]
    resid = yv - xv @ beta
    return {"beta": beta, "resid": resid, "tau": np.array([tau])}


def two_sls(
    y: Array, x_endog: Array, x_exog: Array, instruments: Array
) -> dict[str, Array]:
    """Two-stage least squares.

    ``x_exog`` are included exogenous regressors (constant added);
    ``instruments`` are excluded instruments z (must satisfy order cond.).
    Returns betas for [const, x_endog, x_exog], plus first-stage R^2.
    """
    v = np.asarray(y, dtype=float).reshape(-1)
    xe = np.asarray(x_endog, dtype=float)
    xx = np.asarray(x_exog, dtype=float)
    zi = np.asarray(instruments, dtype=float)
    for arr, name in ((xe, "x_endog"), (xx, "x_exog"), (zi, "instruments")):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] != v.size or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite n x k matrix matching y")
        if name == "x_endog":
            xe = arr
        elif name == "x_exog":
            xx = arr
        else:
            zi = arr
    n = v.size
    k_end, k_ex, k_z = xe.shape[1], xx.shape[1], zi.shape[1]
    if k_z < k_end:
        raise ValueError("under-identified: need at least as many instruments as endogenous regressors")
    if n < k_end + k_ex + k_z + 2:
        raise ValueError("insufficient data for 2SLS")
    z_full = np.column_stack([np.ones(n), xx, zi])
    # First stage: x_endog ~ z_full
    xhat = np.zeros_like(xe)
    r2s = np.zeros(k_end)
    for j in range(k_end):
        f = ols(xe[:, j], z_full[:, 1:])
        xhat[:, j] = f["fitted"]
        r2s[j] = float(f["r2"][0])
    x_full = np.column_stack([np.ones(n), xhat, xx])
    beta, *_ = np.linalg.lstsq(x_full, v, rcond=None)
    # Residuals use ORIGINAL regressors, not fitted.
    x_orig = np.column_stack([np.ones(n), xe, xx])
    resid = v - x_orig @ beta
    return {
        "beta": beta,
        "resid": resid,
        "first_stage_r2": r2s,
        "design": x_orig,
    }


def sargan_hansen_j(
    y: Array, x_endog: Array, x_exog: Array, instruments: Array
) -> dict[str, float]:
    """Sargan–Hansen J test of overidentifying restrictions."""
    fit = two_sls(y, x_endog, x_exog, instruments)
    e = fit["resid"]
    zi = np.asarray(instruments, dtype=float)
    if zi.ndim == 1:
        zi = zi.reshape(-1, 1)
    xx = np.asarray(x_exog, dtype=float)
    if xx.ndim == 1:
        xx = xx.reshape(-1, 1)
    n = e.size
    z_full = np.column_stack([np.ones(n), xx, zi])
    aux = ols(e, z_full[:, 1:])
    j = n * float(aux["r2"][0])
    df = zi.shape[1] - (x_endog.shape[1] if np.asarray(x_endog).ndim > 1 else 1)
    if df < 1:
        raise ValueError("model is exactly identified — J undefined")
    return {"j": j, "pvalue": float(sstats.chi2.sf(j, df)), "df": float(df)}
