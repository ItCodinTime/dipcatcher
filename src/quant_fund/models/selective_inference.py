"""Post-selection (polyhedral) confidence intervals.

Lee, Sun, Sun & Taylor (2016, "Exact post-selection inference, with
application to the lasso"): when the model/variable report was chosen by a
data-dependent selection event expressible as the polyhedron {A y <= b}
conditional on sigma, then conditional on that event the pivot

    F_{eta^T mu, sigma^2 (I-P) eta}^{[V-, V+]}(eta^T y)

is Uniform(0, 1), where [V-, V+] is the truncation interval induced by the
selection on the line through y in direction eta. Inverting the truncated
Gaussian CDF yields valid post-selection intervals.

This module provides the general machinery plus the canonical case of
conditioning on the sign of an OLS/screening coefficient
(``sign_conditioned_interval``), the workhorse for "we report the direction
the estimator pointed in" workflows.

Fail-closed: non-finite input, inconsistent polyhedron, sigma <= 0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]


def _truncation_bounds(
    a: Array, b: Array, y: Array, eta: Array, sigma2: float
) -> tuple[float, float, float]:
    """V-, V+ for the polyhedron {A y <= b} along direction eta.

    With y = z + t*eta where z = y - t*eta, t = eta^T y / ||eta||^2, the
    constraint A(z + t eta) <= b splits into t-bounds by sign of A eta.
    Returns (v_minus, v_plus, t_hat) in eta^T y units.
    """
    a_eta = a @ eta
    eta_sq = float(eta @ eta)
    t_hat = float(eta @ y)
    z = y - (t_hat / eta_sq) * eta  # orthogonal complement
    resid = b - a @ z
    v_minus, v_plus = -np.inf, np.inf
    for j in range(a.shape[0]):
        bound = resid[j] / a_eta[j] if abs(a_eta[j]) > 1e-300 else np.inf
        if a_eta[j] > 0:
            v_plus = min(v_plus, bound)
        elif a_eta[j] < 0:
            v_minus = max(v_minus, bound)
        elif resid[j] < 0:
            return np.inf, -np.inf, t_hat  # y violates polyhedron
    return v_minus * eta_sq, v_plus * eta_sq, t_hat


def _log_cdf_diff(hi: float, lo: float) -> float:
    """log(Phi(hi) - Phi(lo)) for hi > lo; stable in the far tails."""
    l_hi = stats.norm.logcdf(hi)
    l_lo = stats.norm.logcdf(lo)
    if l_lo >= l_hi:
        return -np.inf  # Phi difference numerically zero
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(l_hi + np.log1p(-np.exp(l_lo - l_hi)))


def _tg_cdf(x: float, mu: float, sigma: float, lo: float, hi: float) -> float:
    """CDF of N(mu, sigma^2) truncated to [lo, hi] evaluated at x."""
    if hi <= lo:
        return np.nan
    z_lo = (lo - mu) / sigma
    z_hi = (hi - mu) / sigma
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    z = (x - mu) / sigma
    log_denom = _log_cdf_diff(z_hi, z_lo)
    log_num = _log_cdf_diff(z, z_lo)
    if log_denom <= -745.0 or not np.isfinite(log_denom):
        return np.nan  # truncation interval carries ~zero mass at this mu
    return float(np.exp(np.clip(log_num - log_denom, -700.0, 0.0)))


def polyhedral_interval(
    y: Array,
    a: Array,
    b: Array,
    eta: Array,
    sigma: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Exact post-selection CI for eta^T mu given selection {A y <= b}.

    Assumes y ~ N(mu, sigma^2 I). Inverts the truncated-Gaussian pivot by
    bisection. Returns the interval plus the observed statistic and the
    truncation bounds in eta^T y units.
    """
    yy = np.asarray(y, dtype=float).ravel()
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float).ravel()
    ee = np.asarray(eta, dtype=float).ravel()
    if aa.ndim != 2 or aa.shape[1] != yy.size or bb.size != aa.shape[0]:
        raise ValueError("A must be (k, n), b length k, matching len(y)")
    if ee.size != yy.size or not np.isfinite(ee).all() or float(ee @ ee) == 0.0:
        raise ValueError("eta must be a non-zero vector matching len(y)")
    if not np.isfinite(yy).all() or not np.isfinite(aa).all() or not np.isfinite(bb).all():
        raise ValueError("non-finite input")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    v_lo, v_hi, t_hat = _truncation_bounds(aa, bb, yy, ee, sigma * sigma)
    if not (v_lo <= t_hat <= v_hi):
        raise ValueError("y does not satisfy the stated selection event")
    sd = float(sigma * np.sqrt(ee @ ee))
    stat = float(ee @ yy)

    def _cdf_at(mu: float) -> float:
        return _tg_cdf(stat, mu, sd, v_lo, v_hi)

    def _solve_mu(target: float) -> float:
        # F_mu(stat) is decreasing in mu; bisection on the pivot.
        lo, hi = stat - 60.0 * sd, stat + 60.0 * sd
        # shrink endpoints toward stat until the truncated CDF is finite
        f_lo = _cdf_at(lo)
        for _ in range(60):
            if np.isfinite(f_lo):
                break
            lo = 0.5 * (lo + stat)
            f_lo = _cdf_at(lo)
        f_hi = _cdf_at(hi)
        for _ in range(60):
            if np.isfinite(f_hi):
                break
            hi = 0.5 * (hi + stat)
            f_hi = _cdf_at(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return float("nan")
        if f_lo <= target:
            return float(lo)
        if f_hi >= target:
            return float(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = _cdf_at(mid)
            if not np.isfinite(f_mid):
                return float(mid)
            if f_mid > target:  # mu too small -> move right
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    lower = _solve_mu(1.0 - alpha / 2.0)
    upper = _solve_mu(alpha / 2.0)
    return {
        "estimate": stat,
        "ci_lo": float(lower),
        "ci_hi": float(upper),
        "sd": sd,
        "v_lo": float(v_lo),
        "v_hi": float(v_hi),
        "t_hat": float(t_hat),
    }


def sign_conditioned_interval(
    y: Array,
    eta: Array,
    sigma: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Post-selection CI conditioned on eta^T y >= 0 (one-sided report)."""
    ee = np.asarray(eta, dtype=float).ravel()
    if ee.size == 0 or not np.isfinite(ee).all():
        raise ValueError("eta must be a finite non-empty vector")
    # selection event {eta^T y >= 0} <-> {-eta^T y <= 0}
    a = -ee[None, :]
    b = np.zeros(1)
    return polyhedral_interval(y, a, b, ee, sigma, alpha)
