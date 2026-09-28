"""Panel causal-inference primitives: diff-in-diff and interrupted series.

Complements ``event_study`` (market-model event studies) with the panel
estimators standard in program evaluation:

- ``diff_in_diff``: canonical 2x2 DiD — ATT = (E[y|D=1,t=1] - E[y|D=1,t=0])
  minus the same contrast for controls — plus the general two-way fixed
  effects regression (unit + time dummies, interactions absorbed by FE).
- ``interrupted_time_series``: segmented regression of a single series,
  y ~ b0 + b1 t + b2 1{t >= tau} + b3 (t - tau)+, with Newey-West HAC
  standard errors from ``quant_fund.metrics.hac``.

Returns point estimates, standard errors, and fitted paths. Fail-closed:
empty cells, degenerate variance, tau out of range, non-finite input.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from quant_fund.metrics.hac import andrews_bandwidth

Array = NDArray[np.float64]


def _cross_lrv(u: Array, v: Array, lag: int) -> float:
    """Cross-series long-run variance Gamma^uv_0 + sum_k w_k (G^uv_k + G^vu_k).

    Bartlett kernel; this is the (i,j) entry of the HAC meat for score
    series u_i, u_j.
    """
    t_len = u.size
    lag = max(0, min(lag, t_len - 2))
    g0 = float(u @ v / t_len)
    out = g0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        g_uv = float(u[: t_len - k] @ v[k:] / t_len)
        g_vu = float(v[: t_len - k] @ u[k:] / t_len)
        out += w * (g_uv + g_vu)
    return out


def _ols(x: Array, y: Array) -> tuple[Array, Array, Array]:
    """OLS via least squares; returns (beta, residuals, XtX^-1)."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    xtx_inv = np.linalg.inv(x.T @ x)
    return beta, resid, xtx_inv


def diff_in_diff(
    y: Array,
    treated: Array,
    post: Array,
) -> dict[str, float]:
    """Canonical 2x2 difference-in-differences.

    ``y`` outcomes, ``treated`` in {0,1} group flag, ``post`` in {0,1}
    period flag. ATT = DiD of cell means; SE from the four cell variances
    (independence across cells).
    """
    yy = np.asarray(y, dtype=float).ravel()
    d = np.asarray(treated, dtype=float).ravel()
    p = np.asarray(post, dtype=float).ravel()
    if yy.size != d.size or yy.size != p.size or yy.size < 4:
        raise ValueError("y/treated/post must be equal-length with n >= 4")
    if not np.isfinite(yy).all() or not np.isfinite(d).all() or not np.isfinite(p).all():
        raise ValueError("non-finite input")
    if set(np.unique(d)) - {0.0, 1.0} or set(np.unique(p)) - {0.0, 1.0}:
        raise ValueError("treated and post must be binary {0, 1}")
    means = np.empty(4)
    var = 0.0
    for k, (dv, pv) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
        cell = yy[(d == dv) & (p == pv)]
        if cell.size == 0:
            raise ValueError(f"empty cell treated={dv} post={pv}")
        means[k] = float(np.mean(cell))
        var += float(np.var(cell, ddof=1) / cell.size) if cell.size > 1 else 0.0
    att = (means[0] - means[1]) - (means[2] - means[3])
    se = float(np.sqrt(var))
    return {
        "att": float(att),
        "se": se,
        "t_stat": float(att / se) if se > 0 else float("inf"),
        "p_value": float(2.0 * stats.t.sf(abs(att / se), max(int(yy.size) - 4, 1)))
        if se > 0
        else 0.0,
    }


def twfe_did(
    y: Array,
    unit: Array,
    time: Array,
    treat_post: Array,
) -> dict[str, Array | float]:
    """Two-way fixed-effects DiD regression (staggered-capable).

    y ~ unit_FE + time_FE + beta * treat_post via dummy OLS; returns beta,
    its OLS standard error, and the regression residuals (for downstream
    clustered-SE refinements).
    """
    yy = np.asarray(y, dtype=float).ravel()
    unit = np.asarray(unit).ravel()
    time = np.asarray(time).ravel()
    tp = np.asarray(treat_post, dtype=float).ravel()
    n = yy.size
    if unit.size != n or time.size != n or tp.size != n:
        raise ValueError("unit/time/treat_post must match y length")
    if not np.isfinite(yy).all() or not np.isfinite(tp).all():
        raise ValueError("non-finite input")
    units = np.unique(unit)
    times = np.unique(time)
    if units.size >= n or times.size >= n:
        raise ValueError("degenerate panel (FEs consume all df)")
    u_d = np.zeros((n, units.size - 1))
    t_d = np.zeros((n, times.size - 1))
    for j, u in enumerate(units[1:]):
        u_d[:, j] = unit == u
    for j, t in enumerate(times[1:]):
        t_d[:, j] = time == t
    x = np.column_stack([np.ones(n), u_d, t_d, tp])
    beta, resid, xtx_inv = _ols(x, yy)
    dof = n - x.shape[1]
    if dof < 1:
        raise ValueError("no residual degrees of freedom")
    sigma2 = float(resid @ resid / dof)
    se_beta = float(np.sqrt(sigma2 * xtx_inv[-1, -1]))
    return {
        "att": float(beta[-1]),
        "se": se_beta,
        "t_stat": float(beta[-1] / se_beta) if se_beta > 0 else float("inf"),
        "resid": resid,
        "dof": float(dof),
    }


def interrupted_time_series(
    y: Array,
    tau: int,
    hac_lag: int | None = None,
) -> dict[str, float | Array]:
    """Segmented-regression interruption analysis.

    Fits y ~ b0 + b1 t + b2 post + b3 (t - tau)+ and reports level/slope
    changes with HAC standard errors (Newey-West via ``kernel_lrv``).
    """
    yy = np.asarray(y, dtype=float).ravel()
    t_len = yy.size
    if t_len < 12 or not np.isfinite(yy).all():
        raise ValueError("y must be a finite array with T >= 12")
    if not (3 <= tau <= t_len - 3):
        raise ValueError("tau must satisfy 3 <= tau <= T - 3")
    t = np.arange(t_len, dtype=float)
    post = (t >= tau).astype(float)
    x = np.column_stack([np.ones(t_len), t, post, np.maximum(t - tau + 1.0, 0.0)])
    beta, resid, xtx_inv = _ols(x, yy)
    # HAC cov(beta) = XtX^-1 B XtX^-1, B_ij = cross-LRV of score series
    scores = x * resid[:, None]
    if hac_lag is None:
        hac_lag = max(1, int(andrews_bandwidth(resid)))
    cov_meat = np.empty((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            cov_meat[i, j] = cov_meat[j, i] = _cross_lrv(scores[:, i], scores[:, j], hac_lag)
    cov_beta = xtx_inv @ cov_meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov_beta), 0.0, None))
    return {
        "beta": beta,
        "se": se,
        "level_change": float(beta[2]),
        "slope_change": float(beta[3]),
        "level_change_t": float(beta[2] / se[2]) if se[2] > 0 else float("inf"),
        "slope_change_t": float(beta[3] / se[3]) if se[3] > 0 else float("inf"),
        "fitted": x @ beta,
        "resid": resid,
    }
