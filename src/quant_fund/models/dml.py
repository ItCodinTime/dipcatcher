"""Double/debiased machine learning — partially linear regression.

Chernozhukov et al. (2018) PLR: with ``Y = theta D + g(X) + U`` and
``D = m(X) + V``, the Neyman-orthogonal score uses residuals on both
sides: estimate ``g`` and ``m`` by cross-fitted ridge regression,
form ``res_y = Y - g_hat(X)`` and ``res_d = D - m_hat(X)``, then

    theta = <res_d, res_y> / <res_d, res_d>

with an influence-function standard error from
``psi_i = res_d_i * (res_y_i - theta * res_d_i)`` and
``se = sqrt(var(psi)) / (sqrt(n) * E[res_d^2])``.

References: V. Chernozhukov et al. (2018), Econometrics Journal 21(1).
Cross-fitting K folds prevents own-observation overfit bias.  Fail-closed
on non-finite input, n < 2*folds, or degenerate residual treatment.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _as_inputs(y: Array, d: Array, x: Array) -> tuple[Array, Array, Array]:
    ya = np.asarray(y, dtype=float).ravel()
    da = np.asarray(d, dtype=float).ravel()
    xa = np.asarray(x, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    n = ya.shape[0]
    if da.shape[0] != n or xa.shape[0] != n:
        raise ValueError("y, d, x must share the same row count")
    if n < 20:
        raise ValueError("need at least 20 observations")
    if not np.all(np.isfinite(ya)) or not np.all(np.isfinite(da)) or not np.all(np.isfinite(xa)):
        raise ValueError("inputs must be finite")
    return ya, da, xa


def _ridge_coef(x: Array, y: Array, lam: float) -> Array:
    """Closed-form ridge with an unpenalized intercept."""
    n = x.shape[0]
    xd = np.hstack([np.ones((n, 1)), x])
    pen = np.eye(xd.shape[1]) * lam
    pen[0, 0] = 0.0
    return np.linalg.solve(xd.T @ xd + pen, xd.T @ y)


def _crossfit_resid(target: Array, x: Array, folds: list[Array], lam: float) -> Array:
    """Out-of-fold ridge residuals."""
    resid = np.empty(target.shape[0])
    for te in folds:
        tr = np.setdiff1d(np.arange(target.shape[0]), te)
        b = _ridge_coef(x[tr], target[tr], lam)
        resid[te] = target[te] - np.hstack([np.ones((te.size, 1)), x[te]]) @ b
    return resid


def _folds(n: int, k: int, seed: int) -> list[Array]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(idx[i::k]) for i in range(k)]


def dml_plr(
    y: Array,
    d: Array,
    x: Array,
    *,
    n_folds: int = 5,
    lam: float = 1.0,
    seed: int = 0,
) -> dict[str, Array]:
    """PLR treatment effect with cross-fitted ridge nuisances.

    Returns ``theta``, its influence-function standard error, a 95% CI,
    residual diagnostics, and the fold count actually used.
    """
    ya, da, xa = _as_inputs(y, d, x)
    n = ya.shape[0]
    if not 2 <= n_folds <= n // 10:
        raise ValueError("n_folds must be >= 2 and leave >= 10 obs per fold")
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and positive")
    folds = _folds(n, n_folds, seed)
    res_y = _crossfit_resid(ya, xa, folds, lam)
    res_d = _crossfit_resid(da, xa, folds, lam)
    dd = float(res_d @ res_d)
    if dd <= 1e-12:
        raise ValueError("treatment residual variance collapsed (m_hat = d)")
    theta = float(res_d @ res_y) / dd
    psi = res_d * (res_y - theta * res_d)
    var_theta = float(psi @ psi) / (dd * dd / n) / n
    se = float(np.sqrt(max(var_theta, 0.0)))
    z = 1.959964  # normal 97.5%
    r2_d = 1.0 - dd / float(np.sum((da - da.mean()) ** 2) + 1e-12)
    return {
        "theta": np.array([theta]),
        "se": np.array([se]),
        "ci_lo": np.array([theta - z * se]),
        "ci_hi": np.array([theta + z * se]),
        "r2_treatment": np.array([r2_d]),
        "n": np.array([n]),
        "n_folds": np.array([n_folds]),
        "res_d_norm": np.array([float(np.sqrt(dd / n))]),
    }


def dml_irm(
    y: Array,
    d: Array,
    x: Array,
    *,
    n_folds: int = 5,
    lam: float = 1.0,
    seed: int = 0,
) -> dict[str, Array]:
    """Interactive regression model (binary treatment, orthogonal ATE).

    ``theta = E[ g1(X) - g0(X) + D*(Y - g1(X))/pi + (1-D)*(g0 - Y)/(1-pi) ]``
    with ridge outcome nuisances per arm and a logistic-lite propensity
    (cross-fitted ridge on the linear probability scale, clipped to
    [0.05, 0.95] for overlap).
    """
    ya, da, xa = _as_inputs(y, d, x)
    n = ya.shape[0]
    uniq = np.unique(da)
    if uniq.size != 2 or not np.isin(uniq, [0.0, 1.0]).all():
        raise ValueError("d must be binary 0/1")
    if not 2 <= n_folds <= n // 10:
        raise ValueError("n_folds must be >= 2 and leave >= 10 obs per fold")
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("lam must be finite and positive")
    folds = _folds(n, n_folds, seed)
    # propensity: cross-fitted ridge on linear probability, clipped
    e_d = da - _crossfit_resid(da, xa, folds, lam)
    e_d = np.clip(e_d, 0.05, 0.95)
    # outcome nuisances per arm, cross-fitted
    g1 = np.empty(n)
    g0 = np.empty(n)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        tr1 = tr[da[tr] == 1.0]
        tr0 = tr[da[tr] == 0.0]
        if tr1.size < 5 or tr0.size < 5:
            raise ValueError("treatment arm too thin inside a fold")
        xd_te = np.hstack([np.ones((te.size, 1)), xa[te]])
        g1[te] = xd_te @ _ridge_coef(xa[tr1], ya[tr1], lam)
        g0[te] = xd_te @ _ridge_coef(xa[tr0], ya[tr0], lam)
    psi = (g1 - g0) + da * (ya - g1) / e_d - (1.0 - da) * (ya - g0) / (1.0 - e_d)
    theta = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    z = 1.959964
    return {
        "theta": np.array([theta]),
        "se": np.array([se]),
        "ci_lo": np.array([theta - z * se]),
        "ci_hi": np.array([theta + z * se]),
        "n": np.array([n]),
        "n_folds": np.array([n_folds]),
        "propensity_range": np.array([float(e_d.min()), float(e_d.max())]),
    }
