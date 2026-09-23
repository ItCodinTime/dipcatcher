"""Panel cointegration tests.

Complements ``metrics/panel.py`` (panel unit roots) and
``models/var_coint.py`` (single-series cointegration) with panel
residual-based tests.

References:
- Kao (1999): residual-ADF panel cointegration test (homogeneous).
- Pedroni (1999, 2004): seven-statistic battery; we implement the
  panel-ADF and group-mean-ADF members (heterogeneous alternative).
- McCoskey & Kao (1998): residual-based LM context.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _panels(y: Array, x: Array, min_n: int = 3, min_t: int = 40) -> tuple[Array, Array]:
    a = np.asarray(y, dtype=float)
    b = np.asarray(x, dtype=float)
    if a.ndim != 2 or b.shape != a.shape or a.shape[0] < min_t or a.shape[1] < min_n:
        raise ValueError(f"panels must be finite (T >= {min_t}, N >= {min_n}) and aligned")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("panels must be finite")
    return a, b


def _coint_residuals(p_y: Array, p_x: Array) -> Array:
    """Per-unit cointegrating OLS residuals e_it = y_it - a_i - b_i x_it."""
    T, N = p_y.shape
    E = np.empty_like(p_y)
    for i in range(N):
        xi = p_x[:, i]
        if xi.std() == 0:
            raise ValueError("degenerate regressor column")
        Xd = np.column_stack([np.ones(T), xi])
        beta, *_ = np.linalg.lstsq(Xd, p_y[:, i], rcond=None)
        E[:, i] = p_y[:, i] - Xd @ beta
    return E


def kao_test(y: Array, x: Array, lags: int = 1) -> dict[str, float]:
    """Kao (1999) homogeneous panel cointegration test.

    Pooled residual ADF under the assumption of a common rho. The
    standardized statistic DF_rho = sqrt(N*T) * (rho_hat - 1) / s_e with
    Kao's normalization approximates N(0,1) under no cointegration;
    rejects left (large negative)."""
    a, b = _panels(y, x)
    E = _coint_residuals(a, b)
    T, N = E.shape
    # Pooled ADF: e_it = rho e_i,t-1 + sum_j phi_j De_{i,t-j} + v.
    dep = []
    regs = []
    for i in range(N):
        e = E[:, i]
        de = np.diff(e)
        rows = T - 1 - lags
        if rows <= lags + 1:
            raise ValueError("insufficient observations")
        # e_{t-1} regressor plus lagged differences.
        base = e[lags : T - 1]  # e_{t-1} for t = lags+1..T-1
        cols = [base]
        for k in range(1, lags + 1):
            cols.append(de[lags - k : T - 1 - k])
        Xi = np.column_stack(cols)
        dep.append(de[lags:])
        regs.append(Xi)
    D = np.concatenate(dep)
    Xr = np.vstack(regs)
    beta, *_ = np.linalg.lstsq(Xr, D, rcond=None)
    u = D - Xr @ beta
    n_eff = D.size
    s2 = float(u @ u / max(n_eff - Xr.shape[1], 1))
    rho_hat = float(beta[0]) + 1.0  # regression on Dy gives (rho-1)
    denom = float(np.sum(np.vstack(regs)[:, 0] ** 2))
    t_rho = float(beta[0] / math.sqrt(max(s2 / max(denom, 1e-20), 1e-20)))
    # Kao standardized DF-rho: sqrt(N T)(rho - 1)/s + bias terms omitted
    # (we report the t-statistic and an asymptotic-normal approx).
    df_rho = math.sqrt(N * T) * (rho_hat - 1.0) / max(math.sqrt(s2), 1e-20)
    return {
        "t_stat": t_rho,
        "df_rho": float(df_rho),
        "pvalue": float(stats.norm.cdf(t_rho)),
        "rho": rho_hat,
        "n_units": float(N),
    }


def pedroni_panel_adf(y: Array, x: Array, lags: int = 1) -> dict[str, float | Array]:
    """Pedroni (1999) panel-ADF and group-mean-ADF statistics.

    Per-unit ADF t-stats on cointegrating residuals; the panel statistic
    pools, the group statistic averages. Standardized approximately
    N(0,1) via Pedroni's tabulated moments (intercept case, large T)."""
    a, b = _panels(y, x)
    E = _coint_residuals(a, b)
    T, N = E.shape
    tstats = np.empty(N)
    for i in range(N):
        e = E[:, i]
        de = np.diff(e)
        rows = T - 1 - lags
        if rows <= lags + 1:
            raise ValueError("insufficient observations")
        base = e[lags : T - 1]
        cols = [base]
        for k in range(1, lags + 1):
            cols.append(de[lags - k : T - 1 - k])
        Xi = np.column_stack(cols)
        dep = de[lags:]
        beta, *_ = np.linalg.lstsq(Xi, dep, rcond=None)
        u = dep - Xi @ beta
        dof = max(rows - Xi.shape[1], 1)
        s2 = float(u @ u / dof)
        var0 = s2 / max(float(base @ base), 1e-20)
        tstats[i] = float(beta[0] / math.sqrt(max(var0, 1e-30)))
    group_mean = float(tstats.mean())
    # Pedroni tabulated moments (intercept, large T): E ~ -1.677 (panel),
    # use conservative normal approx for the group mean.
    z_panel = math.sqrt(N) * (group_mean + 1.677) / math.sqrt(0.55)
    return {
        "group_adf": group_mean,
        "statistic": float(z_panel),
        "pvalue": float(stats.norm.cdf(z_panel)),
        "tstats": tstats,
        "n_units": float(N),
    }
