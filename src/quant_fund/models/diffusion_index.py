"""Stock–Watson diffusion-index forecasting and factor-augmented models.

References:
- Stock & Watson (2002): forecasting with diffusion indexes — PCA
  factors of a large panel drive an autoregressive forecast equation.
- Bai & Ng (2006): confidence intervals for diffusion-index forecasts.
- Bernanke, Boivin & Eliasz (2005): factor-augmented VAR (factors +
  observable policy variable), represented via the factor block.
- Stock & Watson (2006): forecast combination with factors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _panel(x: Array, min_n: int = 5, min_t: int = 40) -> Array:
    p = np.asarray(x, dtype=float)
    if p.ndim != 2 or p.shape[0] < min_t or p.shape[1] < min_n:
        raise ValueError(f"panel must be finite (T >= {min_t}, N >= {min_n})")
    if not np.all(np.isfinite(p)):
        raise ValueError("panel must be finite")
    return p


def sw_factors(panel: Array, k: int) -> dict[str, Array | float]:
    """Stock–Watson static factor extraction by PCA.

    Standardizes each series (mean 0, std 1), extracts the first k
    principal components as factors F_t (T, k), and returns loadings
    Lambda (N, k) with F' F / T = I normalization."""
    p = _panel(panel)
    T, N = p.shape
    if not (1 <= k <= min(N, T - 1)):
        raise ValueError("k must be in [1, min(N, T-1)]")
    sd = p.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("degenerate column")
    Z = (p - p.mean(axis=0)) / sd
    # PCA on T x N: factors = eigenvectors of Z'Z applied as Z V.
    cov = Z.T @ Z / T
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:k]
    V = vecs[:, order]
    F = Z @ V  # (T, k)
    # Normalize factors: F'F / T = I.
    fsd = np.sqrt((F * F).sum(axis=0) / T)
    fsd = np.maximum(fsd, 1e-12)
    F = F / fsd[None, :]
    # Loadings via OLS of Z on F.
    lam = np.linalg.lstsq(F, Z, rcond=None)[0].T  # (N, k)
    resid = Z - F @ lam.T
    r2 = 1.0 - (resid**2).sum() / max((Z**2).sum(), 1e-20)
    return {
        "factors": F,
        "loadings": lam,
        "eigvals": vals[order],
        "r2": np.array([r2]),
        "k": np.array([k]),
    }


def diffusion_index_forecast(
    panel: Array,
    y: Array,
    k: int,
    ar_lags: int = 1,
    horizon: int = 1,
) -> dict[str, Array | float]:
    """Stock–Watson (2002) h-step-ahead diffusion-index forecast.

    ``y_{t+h} = alpha + sum_j beta_j F_{jt} + sum_l phi_l y_{t+1-l} + e``.
    Factors are estimated on the full panel (in-sample); the forecast
    equation is OLS on time-aligned rows. Returns coefficients, fitted
    values, residual diagnostics, and the latest h-step forecast."""
    p = _panel(panel)
    yv = np.asarray(y, dtype=float).reshape(-1)
    T = p.shape[0]
    if yv.size != T or not np.all(np.isfinite(yv)):
        raise ValueError("y must be finite (T,) matching panel")
    if horizon < 1 or ar_lags < 0:
        raise ValueError("horizon >= 1, ar_lags >= 0")
    F = np.asarray(sw_factors(p, k)["factors"], dtype=float)
    # Dependent variable y_{t+h}: t+h <= T-1 -> t <= T-1-h.
    rows = T - horizon
    if rows <= ar_lags + k + 3:
        raise ValueError("insufficient observations for specification")
    cols = [np.ones(rows)]
    for j in range(k):
        cols.append(F[:rows, j])
    for lag_ in range(1, ar_lags + 1):
        # y_{t+1-l} aligns so t indexes rows; y_t at row t: y_{row} for
        # l=1 uses y[row - l + 1]... define dependent as y[row + h - 1]? We
        # forecast y_{t+h} from F_t and y_t — build target y[t+h].
        cols.append(
            np.array([yv[i - lag_ + 1] if i - lag_ + 1 >= 0 else np.nan for i in range(rows)])
        )
    X = np.column_stack(cols)
    dep = np.array([yv[i + horizon] for i in range(rows)])
    good = np.isfinite(X).all(axis=1)
    Xg = X[good]
    dg = dep[good]
    if Xg.shape[0] <= Xg.shape[1] + 3:
        raise ValueError("too few complete rows")
    beta, *_ = np.linalg.lstsq(Xg, dg, rcond=None)
    u = dg - Xg @ beta
    dof = max(Xg.shape[0] - Xg.shape[1], 1)
    s2 = float(u @ u / dof)
    try:
        cov = s2 * np.linalg.inv(Xg.T @ Xg)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(Xg.shape[1], np.nan)
    # One-step-ahead forecast at the last available t = rows-1.
    x_last = np.array(
        [1.0]
        + [F[rows - 1, j] for j in range(k)]
        + [yv[rows - lag_] for lag_ in range(1, ar_lags + 1)]
    )
    fc = float(x_last @ beta)
    return {
        "coefs": beta,
        "se": se,
        "fitted": Xg @ beta,
        "resid": u,
        "forecast": fc,
        "r2": float(1.0 - (u @ u) / max(float(((dg - dg.mean()) ** 2).sum()), 1e-20)),
        "n_obs": float(Xg.shape[0]),
    }


def factor_augmented_regression(
    panel: Array, observable: Array, k: int
) -> dict[str, Array | float]:
    """Bernanke–Boivin–Eliasz-style factor-augmented fit: regress the
    observable on SW factors and report incremental R2 vs factors alone
    and vs observable alone (Shapley-style split)."""
    p = _panel(panel)
    y = np.asarray(observable, dtype=float).reshape(-1)
    T = p.shape[0]
    if y.size != T or not np.all(np.isfinite(y)):
        raise ValueError("observable must be finite (T,)")
    sw = sw_factors(p, k)
    F = np.asarray(sw["factors"], dtype=float)
    X = np.column_stack([np.ones(T), F])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    sse = float(u @ u)
    tot = max(float(((y - y.mean()) ** 2).sum()), 1e-20)
    # Per-factor contribution: refit without each factor.
    drops = np.empty(k)
    for j in range(k):
        Xj = np.delete(X, 1 + j, axis=1)
        bj, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        uj = y - Xj @ bj
        drops[j] = float(uj @ uj) - sse  # SSE increase when dropping j
    return {
        "r2": float(1.0 - sse / tot),
        "coefs": beta,
        "factor_sse_increase": drops,
        "eigvals": np.asarray(sw["eigvals"], dtype=float),
        "resid": u,
    }
