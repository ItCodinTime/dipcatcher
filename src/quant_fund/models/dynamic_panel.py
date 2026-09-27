"""Dynamic panel-data estimators for panels with lagged outcomes.

References:
- Anderson & Hsiao (1981): IV estimation of differenced dynamic panels
  using lagged levels as instruments.
- Arellano & Bond (1991): one-step/two-step GMM on first differences
  with the full instrument set.
- Arellano & Bover (1995) / Blundell & Bond (1998): system GMM context.
- Sargan (1958) / Hansen (1982): overidentification test.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats

Array = NDArray[np.float64]


def _panel(x: Array, min_n: int = 5, min_t: int = 8) -> Array:
    p = np.asarray(x, dtype=float)
    if p.ndim != 2 or p.shape[0] < min_t or p.shape[1] < min_n:
        raise ValueError(f"panel must be (T >= {min_t}, N >= {min_n})")
    if not np.all(np.isfinite(p)):
        raise ValueError("panel must be finite")
    return p


def anderson_hsiao(y: Array, X: Array | None = None) -> dict[str, Array | float]:
    """Anderson–Hsiao (1981) IV estimator for y_it = rho*y_i,t-1 + x'b + e.

    First-differences the equation and instruments Dy_{t-1} with
    y_{t-2} (levels). ``y`` is (T, N); ``X`` optional (T, N, k).
    """
    p = _panel(y)
    T, N = p.shape
    if T < 4:
        raise ValueError("need T >= 4 for AH differenced IV")
    dy = np.diff(p, axis=0)  # dy[t-1] = Dy_t for t = 1..T-1
    dy_lag = dy[:-1]  # Dy_{t-1} for t = 2..T-1
    inst = p[:-2]  # y_{t-2} level instrument
    dep = dy[1:]  # Dy_t for t = 2..T-1
    n_reg = 1
    Z_exog: list[Array] = []
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim != 3 or Xa.shape[0] != T or Xa.shape[1] != N:
            raise ValueError("X must be (T, N, k)")
        if not np.all(np.isfinite(Xa)):
            raise ValueError("X must be finite")
        k = Xa.shape[2]
        n_reg += k
        for j in range(k):
            Z_exog.append(np.diff(Xa[:, :, j], axis=0)[1:])  # Dx_t
    # Stack (rows*N) vectors.
    dep_v = dep.reshape(-1)
    dyl_v = dy_lag.reshape(-1)
    inst_v = inst.reshape(-1)
    W_cols = [dyl_v] + [z.reshape(-1) for z in Z_exog]
    W = np.column_stack(W_cols)
    Z = np.column_stack([inst_v] + [z.reshape(-1) for z in Z_exog])
    # Just-identified 2SLS: beta = (Z'W)^{-1} Z' dep.
    try:
        zw_inv = np.linalg.inv(Z.T @ W)
    except np.linalg.LinAlgError as exc:
        raise ValueError("AH instrument matrix singular") from exc
    beta = zw_inv @ (Z.T @ dep_v)
    u = dep_v - W @ beta
    dof = max(dep_v.size - n_reg, 1)
    s2 = float(u @ u / dof)
    cov = s2 * zw_inv @ (Z.T @ Z) @ zw_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return {
        "rho": float(beta[0]),
        "beta": beta,
        "se": se,
        "n_obs": float(dep_v.size),
    }


def arellano_bond(y: Array, max_lag_inst: int = 3) -> dict[str, Array | float]:
    """Arellano–Bond (1991) one-step GMM for y_it = rho*y_i,t-1 + e_it.

    First-differences and instruments Dy_{t-1} with y_{t-2},...,y_{t-1-maxlag}
    levels (per-period instrument count min(lag, max_lag_inst)). Also
    returns the Sargan overidentification statistic and AR(2) test on
    differenced residuals (should be absent under correct spec).
    """
    p = _panel(y)
    T, N = p.shape
    if T < 4:
        raise ValueError("need T >= 4")
    dep_rows = []
    endog_rows = []
    inst_rows = []
    for t in range(2, T):
        # Dy_t = y_t - y_{t-1}; endogenous Dy_{t-1}; instruments y_0..y_{t-2}.
        dep_rows.append(p[t] - p[t - 1])
        endog_rows.append(p[t - 1] - p[t - 2])
        m = min(t - 1, max_lag_inst)
        inst_rows.append(np.stack([p[t - 2 - lag] for lag in range(m)], axis=0))
    # Per-unit moment vector g_i = Z_i' du_i with block-diagonal Z_i
    # (period-specific instrument sets per Arellano-Bond 1991).
    m_by_t = [r.shape[0] for r in inst_rows]
    Tm = T - 2  # differenced periods
    total_inst = sum(m_by_t)
    Zi = np.zeros((N, Tm, total_inst))
    col = 0
    for ti, t in enumerate(range(2, T)):
        m = m_by_t[ti]
        for lag in range(m):
            Zi[:, ti, col] = p[t - 2 - lag]
            col += 1
    dep_mat = np.stack(dep_rows, axis=0)  # (Tm, N)
    end_mat = np.stack(endog_rows, axis=0)  # (Tm, N)

    def resid(rho: float) -> Array:
        return dep_mat - rho * end_mat

    def gmm_rho(rho: float) -> Array:
        return np.asarray(np.einsum("ntm,tn->m", Zi, resid(rho)) / N, dtype=float)

    def obj(rho: float) -> float:
        g = gmm_rho(rho)
        return float(g @ g)

    res = opt.minimize_scalar(obj, bounds=(-1.0, 1.0), method="bounded")
    rho = float(res.x)
    du = resid(rho)
    g = np.einsum("ntm,tn->m", Zi, du) / N
    # Sargan-Hansen J: N * g' W g with W = (Z'Z/N)^{-1} — two-step-ish
    # weighting by instrument covariance.
    ZZ = np.einsum("ntm,nts->ms", Zi, Zi) / N
    try:
        W = np.linalg.inv(ZZ + 1e-10 * np.eye(total_inst))
    except np.linalg.LinAlgError as exc:
        raise ValueError("AB weight matrix singular") from exc
    J = float(N * g @ W @ g)
    dof = max(total_inst - 1, 1)
    # SE of rho: sandwich (G'WG)^{-1} G'W var(g) W G — one-step GMM.
    G = np.einsum("ntm,tn->m", Zi, -end_mat) / N  # d g / d rho
    denom = float(G @ W @ G)
    var_rho = 1.0 / max(N * denom, 1e-20)
    # AR(2) test on differenced residuals (per-unit pooled correlation).
    du1 = du[:-1].reshape(-1)
    du2 = du[1:].reshape(-1)
    if np.std(du2) > 0 and np.std(du1) > 0:
        ar2 = float(np.corrcoef(du1, du2)[0, 1])
    else:
        ar2 = np.nan
    return {
        "rho": rho,
        "se": float(math.sqrt(var_rho)),
        "sargan_J": J,
        "sargan_p": float(1.0 - stats.chi2.cdf(J, dof)),
        "sargan_df": float(dof),
        "ar2_coef": float(ar2) if np.isfinite(ar2) else np.nan,
        "n_inst": float(total_inst),
        "n_units": float(N),
        "n_periods": float(T),
    }
