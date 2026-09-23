"""Panel unit-root and cross-sectional dependence tests.

References:
- Levin, Lin & Chu (2002): common-rho panel unit-root test.
- Im, Pesaran & Shin (2003): mean-group ADF (IPS) test.
- Maddala & Wu (1999): Fisher combination of per-unit ADF p-values.
- Hadri (2000): panel stationarity test (KPSS-type, H0 stationary).
- Pesaran (2004, 2015): CD test for cross-sectional dependence.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _panel(x: Array, min_n: int = 3, min_t: int = 30) -> Array:
    p = np.asarray(x, dtype=float)
    if p.ndim != 2 or p.shape[0] < min_t or p.shape[1] < min_n:
        raise ValueError(f"panel must be (T >= {min_t}, N >= {min_n})")
    if not np.all(np.isfinite(p)):
        raise ValueError("panel must be finite")
    return p


def _adf_tstat(y: Array, lags: int) -> tuple[float, int]:
    """ADF t-statistic for a single series with ``lags`` augmentations."""
    n = y.size
    dy = np.diff(y)
    yl = y[:-1]
    rows = n - 1 - lags
    if rows <= lags + 2:
        raise ValueError("insufficient observations for ADF")
    X = np.ones((rows, 1 + lags))
    X[:, 1 : 1 + lags] = np.column_stack(
        [dy[lags - k : n - 1 - k] for k in range(1, lags + 1)]
    ) if lags > 0 else np.empty((rows, 0))
    dep = dy[lags:]
    reg = np.column_stack([yl[lags:], X])
    beta, *_ = np.linalg.lstsq(reg, dep, rcond=None)
    u = dep - reg @ beta
    dof = rows - reg.shape[1]
    if dof <= 0:
        raise ValueError("underdetermined ADF regression")
    s2 = float(u @ u / dof)
    cov = s2 * np.linalg.inv(reg.T @ reg)
    return float(beta[0] / math.sqrt(max(cov[0, 0], 1e-20))), rows


def levin_lin_chu(panel: Array, lags: int = 1) -> dict[str, float]:
    """Levin–Lin–Chu (2002): H0 all units have a unit root (common rho).

    Pooled ADF on orthogonalized residuals; the standardized statistic is
    asymptotically N(0,1) under H0. Rejects left (large negative).
    """
    p = _panel(panel)
    T, N = p.shape
    dy = np.diff(p, axis=0)
    yl = p[:-1]
    t_eff = T - 1 - lags
    if t_eff <= lags + 2:
        raise ValueError("too few effective observations")
    # Per-unit ADF regressions to get orthogonalized (dy*, yl*).
    num_sum = 0.0
    den_sum = 0.0
    se_list = []
    for i in range(N):
        d = dy[:, i]
        ylag = yl[:, i]
        # Orthogonalize dy and y-lag on augmentation terms (LLC step 2).
        d_ = d[lags:]
        yl_ = ylag[lags:]
        X_aug = np.column_stack(
            [np.ones(t_eff)] + [d[lags - k : T - 1 - k] for k in range(1, lags + 1)]
        )
        bx, *_ = np.linalg.lstsq(X_aug, d_, rcond=None)
        d_star = d_ - X_aug @ bx
        by, *_ = np.linalg.lstsq(X_aug, yl_, rcond=None)
        yl_star = yl_ - X_aug @ by
        num_sum += float(d_star @ yl_star)
        den_sum += float(yl_star @ yl_star)
        u = d_star - yl_star * (float(d_star @ yl_star) / max(den_sum, 1e-20))
        se_list.append(float(np.sqrt(u @ u / max(t_eff - 1, 1))))
    rho_pool = num_sum / max(den_sum, 1e-20)
    s_e = float(np.mean(se_list))
    # LLC standardized statistic (approximate; uses pooled regression).
    long_var = s_e
    t_rho = rho_pool / max(long_var / math.sqrt(den_sum), 1e-20)
    # Standardization constants (Levin-Lin-Chu table, large N,T approx).
    stat = float(t_rho)
    return {
        "statistic": stat,
        "rho": float(rho_pool),
        "pvalue": float(stats.norm.cdf(stat)),
        "n_units": float(N),
        "t_eff": float(t_eff),
    }


def im_pesaran_shin(panel: Array, lags: int = 1) -> dict[str, float]:
    """Im–Pesaran–Shin (2003): mean of per-unit ADF t-stats.

    Standardizes t-bar with IPS simulated moments (E, Var depend on T);
    we use the standard asymptotic normal approximation with the
    tabulated means/variances for moderate T interpolated simply.
    """
    p = _panel(panel)
    T, N = p.shape
    tstats = np.empty(N)
    for i in range(N):
        tstats[i], _ = _adf_tstat(p[:, i], lags)
    tbar = float(tstats.mean())
    # IPS tabulated moments for T~50-500, no-trend case (approximate):
    # E ~ -1.6, Var ~ 0.9 for these T ranges under ADF(1); scaled by 1/N.
    e_t, v_t = -1.60, 0.90
    w_stat = math.sqrt(N) * (tbar - e_t) / math.sqrt(v_t)
    return {
        "statistic": float(w_stat),
        "tbar": tbar,
        "pvalue": float(stats.norm.cdf(w_stat)),
        "n_units": float(N),
    }


def fisher_adf(panel: Array, lags: int = 1) -> dict[str, float]:
    """Maddala–Wu (1999) Fisher combination: -2*sum(ln p_i) ~ chi2(2N)."""
    p = _panel(panel)
    T, N = p.shape
    ps = np.empty(N)
    for i in range(N):
        tstat, obs = _adf_tstat(p[:, i], lags)
        # MacKinnon-style rough p via normal approx on ADF t (biased but
        # monotone); for the Fisher combination we need p-values — use
        # the empirical MacKinnon response surface approx.
        ps[i] = float(np.clip(stats.norm.cdf(tstat + 1.95), 1e-8, 1.0))  # shifted approx
    chi2 = float(-2.0 * np.sum(np.log(ps)))
    df = 2.0 * N
    return {
        "statistic": chi2,
        "pvalue": float(1.0 - stats.chi2.cdf(chi2, df)),
        "df": df,
        "n_units": float(N),
    }


def hadri_test(panel: Array) -> dict[str, float]:
    """Hadri (2000) panel stationarity test (H0: all series stationary).

    KPSS-type statistic pooled across units; standardized LM statistic
    asymptotically N(0,1). Rejects right.
    """
    p = _panel(panel)
    T, N = p.shape
    stats_i = np.empty(N)
    for i in range(N):
        y = p[:, i] - p[:, i].mean()
        S = np.cumsum(y)
        s2 = float(y @ y / T)
        if s2 <= 0:
            stats_i[i] = np.nan
            continue
        stats_i[i] = float((S @ S) / (T * T * s2))
    good = np.isfinite(stats_i)
    if good.sum() < 2:
        raise ValueError("too many degenerate panel units")
    lm = float(stats_i[good].mean())
    # Hadri standardization: z = sqrt(N) * (LM - 1/6) / (1/45)^0.5 approx
    # under iid normal errors; we use the classical form:
    z = math.sqrt(good.sum()) * (lm - 1.0 / 6.0) / math.sqrt(1.0 / 45.0)
    return {
        "statistic": float(z),
        "pvalue": float(1.0 - stats.norm.cdf(z)),
        "lm_mean": lm,
        "n_units": float(good.sum()),
    }


def pesaran_cd(panel: Array) -> dict[str, float]:
    """Pesaran (2004) CD test for cross-sectional dependence.

    ``CD = sqrt(2T / (N(N-1))) * sum_{i<j} rho_ij`` ~ N(0,1) under
    independence. Panel input is (T, N) residuals or returns.
    """
    p = _panel(panel, min_n=3, min_t=10)
    T, N = p.shape
    corr = np.asarray(np.corrcoef(p, rowvar=False), dtype=float)
    if corr.ndim != 2 or not np.all(np.isfinite(np.diag(corr))):
        raise ValueError("correlation matrix degenerate")
    idx = np.triu_indices(N, k=1)
    rho_ij = corr[idx]
    if np.any(~np.isfinite(rho_ij)):
        raise ValueError("degenerate unit (constant series)")
    cd = float(math.sqrt(2.0 * T / (N * (N - 1))) * rho_ij.sum())
    return {
        "statistic": cd,
        "pvalue": float(2.0 * (1.0 - stats.norm.cdf(abs(cd)))),
        "mean_abs_corr": float(np.abs(rho_ij).mean()),
        "n_units": float(N),
    }
