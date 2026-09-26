"""Two-step GMM estimation with HAC weighting (Hansen 1982).

Given a moment function ``moments(theta) -> (T, q)`` matrix whose row
means should be zero at the true parameter, estimate theta (p <= q) by
minimizing ``g' W g`` where g is the (q,) sample mean moment. Two-step:
identity weight matrix first, then W = S^{-1} with S the Bartlett
Newey-West long-run variance of the moment process.

Standard errors use the sandwich (G'WG)^{-1}/T with G the numerical
Jacobian of the mean moment. Overidentification is assessed by Hansen's
J statistic T * g' W g ~ chi2(q - p).

Fail-closed: q < p (underidentified), non-finite moments, singular
weight/covariance matrices.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

Array = NDArray[np.float64]
MomentsFn = Callable[[Array], Array]


def _nw_cov(m: Array, max_lag: int | None = None) -> Array:
    """Bartlett Newey-West long-run variance of a (T, q) process."""
    t, q = m.shape
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)))
    max_lag = max(0, min(max_lag, t - 2))
    mc = m - m.mean(axis=0)
    s = mc.T @ mc / t
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        g_l = mc[lag:].T @ mc[:-lag] / t
        s = s + w * (g_l + g_l.T)
    return np.asarray(s, dtype=np.float64)


def _mean_moment(moments: MomentsFn, theta: Array) -> Array:
    m = np.asarray(moments(theta), dtype=np.float64)
    return np.asarray(m.mean(axis=0), dtype=np.float64)


def gmm_2step(
    moments: MomentsFn,
    theta0: Array,
    max_lag: int | None = None,
) -> dict[str, Array | float]:
    """Hansen two-step GMM.

    ``moments`` must be callable: theta -> (T, q) finite array.
    Returns theta, se, t-stats, J, J p-value, weight matrix, objective.
    """
    fn = moments
    t0 = np.asarray(theta0, dtype=float).ravel()
    m0 = np.asarray(fn(t0), dtype=np.float64)
    if m0.ndim != 2 or not np.isfinite(m0).all():
        raise ValueError("moments(theta0) must return a finite (T, q) matrix")
    t_n, q = m0.shape
    p = t0.size
    if q < p:
        raise ValueError("underidentified: need q >= p moments")
    if t_n < 2 * q + 10:
        raise ValueError("insufficient observations for GMM")

    def obj(theta: Array, w: Array) -> float:
        m = np.asarray(fn(theta), dtype=np.float64)
        if m.shape != (t_n, q) or not np.isfinite(m).all():
            return 1e14
        g = m.mean(axis=0)
        return float(g @ w @ g)

    # step 1
    r1 = optimize.minimize(obj, t0, args=(np.eye(q),), method="BFGS")
    m1 = np.asarray(fn(r1.x), dtype=np.float64)
    s = _nw_cov(m1, max_lag)
    try:
        w2 = np.linalg.inv(s + 1e-10 * np.eye(q))
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular HAC weight matrix") from exc

    # step 2
    r2 = optimize.minimize(obj, r1.x, args=(w2,), method="BFGS")
    theta = r2.x
    m2 = np.asarray(fn(theta), dtype=np.float64)
    g2 = m2.mean(axis=0)

    # numerical Jacobian of mean moment
    jac = np.zeros((q, p))
    for j in range(p):
        h = 1e-6 * max(1.0, abs(theta[j]))
        e = np.zeros(p)
        e[j] = h
        jac[:, j] = (_mean_moment(fn, theta + e) - _mean_moment(fn, theta - e)) / (2.0 * h)
    try:
        cov = np.linalg.inv(jac.T @ w2 @ jac) / t_n
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular GMM covariance") from exc
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    jstat = t_n * float(g2 @ w2 @ g2)
    dof = q - p
    jp = float(1.0 - stats.chi2.cdf(jstat, dof)) if dof > 0 else np.nan
    return {
        "theta": theta,
        "se": se,
        "tstat": theta / np.maximum(se, 1e-12),
        "J": jstat,
        "J_p": jp,
        "dof": float(dof),
        "W": w2,
        "S": _nw_cov(m2, max_lag),
        "objective": float(g2 @ w2 @ g2),
        "g": g2,
    }
