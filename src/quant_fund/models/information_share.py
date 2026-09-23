"""Price discovery: Hasbrouck (1995) information shares and
Gonzalo-Granger (1995) permanent-transitory decomposition.

Two cointegrated price series share a common trend. The VECM
Delta p_t = c + alpha (beta' p_{t-1}) + Gamma Delta p_{t-1} + eps_t
is estimated by Engle-Granger beta + OLS on the error-correction term
(sufficient for the 2-series case the formulas target).

- Information share of series j: contribution of its innovation variance
  to the variance of the common-trend innovation. With correlated
  innovations the exact share is ordering-dependent; we report the
  lower/upper bounds from both Cholesky orderings (Hasbrouck 1995).
- Gonzalo-Granger: permanent component weight w = alpha_perp normalized
  to sum to one.

Fail-closed: fewer than ~30 obs, non-finite input, degenerate alpha.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _vecm_2(p1: Array, p2: Array, k_diff: int) -> tuple[Array, Array, Array]:
    """Estimate VECM alpha, residuals, ect for 2-series case.

    beta from Engle-Granger OLS (p1 on p2); then
    Delta p_t = c + alpha ect_{t-1} + sum_i Gamma_i Delta p_{t-i} + eps.
    """
    n = p1.size
    x = np.column_stack([np.ones(n), p2])
    b, *_ = np.linalg.lstsq(x, p1, rcond=None)
    beta = np.array([1.0, -float(b[1])])
    p = np.column_stack([p1, p2])
    ect = p @ beta
    dp = np.diff(p, axis=0)
    rows = []
    dep = []
    m = k_diff + 1
    for t in range(m, dp.shape[0]):
        row = [1.0, ect[t]]
        for lag in range(1, k_diff + 1):
            row.extend(dp[t - lag])
        rows.append(row)
        dep.append(dp[t])
    X = np.asarray(rows)
    Y = np.asarray(dep)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    alpha = coef[1]  # (2,) adjustment speeds on ect
    return alpha, resid, beta


def hasbrouck_is(
    price_a: Array,
    price_b: Array,
    k_diff: int = 1,
) -> dict[str, Array | float]:
    """Hasbrouck (1995) information-share bounds for two cointegrated series.

    Returns per-series [lower, upper] bounds (Cholesky ordering bounds)
    plus the psi vector and residual covariance. Shares sum to 1.
    """
    a = np.asarray(price_a, dtype=float).ravel()
    b = np.asarray(price_b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 30:
        raise ValueError("two series of equal length >= 30 required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("series must be finite")
    alpha, resid, beta = _vecm_2(a, b, k_diff)
    if np.abs(alpha).max() < 1e-10:
        raise ValueError("degenerate adjustment: alpha ~ 0 (no error correction)")
    psi = np.array([alpha[1], -alpha[0]], dtype=float)
    s = psi.sum()
    if abs(s) < 1e-12:
        raise ValueError("psi sums to ~0; cannot normalize")
    psi = psi / s
    omega = np.cov(resid.T)

    def shares(order: tuple[int, int]) -> Array:
        # Cholesky of reordered Omega; IS_j = (psi' F)_j^2 / (psi' Omega psi)
        idx = np.asarray(order)
        om = omega[np.ix_(idx, idx)]
        f = np.linalg.cholesky(om)
        psi_ord = psi[idx]
        contrib = psi_ord @ f  # row vector psi'F
        denom = float(psi @ omega @ psi)
        sh = contrib**2 / denom
        out = np.empty(2)
        out[idx] = sh
        return out

    s_ab = shares((0, 1))
    s_ba = shares((1, 0))
    lo = np.minimum(s_ab, s_ba)
    hi = np.maximum(s_ab, s_ba)
    return {
        "is_a": np.array([lo[0], hi[0]]),
        "is_b": np.array([lo[1], hi[1]]),
        "psi": psi,
        "alpha": alpha,
        "beta": beta,
        "omega_diag": np.diag(omega),
        "resid_corr": float(
            resid[:, 0] @ resid[:, 1] / (resid[:, 0].std() * resid[:, 1].std() * resid.shape[0])
        ),
    }


def gonzalo_granger(
    price_a: Array,
    price_b: Array,
    k_diff: int = 1,
) -> dict[str, Array | float]:
    """Gonzalo-Granger permanent component weights w = alpha_perp / sum."""
    a = np.asarray(price_a, dtype=float).ravel()
    b = np.asarray(price_b, dtype=float).ravel()
    if a.shape != b.shape or a.size < 30:
        raise ValueError("two series of equal length >= 30 required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("series must be finite")
    alpha, _, beta = _vecm_2(a, b, k_diff)
    perp = np.array([alpha[1], -alpha[0]], dtype=float)
    s = perp.sum()
    if abs(s) < 1e-12:
        raise ValueError("alpha_perp sums to ~0; weights undefined")
    w = perp / s
    return {
        "w_a": float(w[0]),
        "w_b": float(w[1]),
        "alpha": alpha,
        "beta": beta,
    }
