"""Causal weight engines for research allocation.

Two layers:

1. Pure weight functions on estimators
   (:func:`inverse_volatility_weights`, :func:`risk_parity_weights`,
   :func:`kelly_weights`, :func:`volatility_target_scale`). These take a
   validated covariance matrix (and mean vector for Kelly) and return raw
   weights with no constraint projection applied.
2. The causal driver :func:`fit_weights`, which estimates covariance (and
   the mean for Kelly) from a *trailing returns window only* and then
   applies :func:`apply_constraints`. Feeding it a window ending at ``t-1``
   is the only way to obtain weights for decision ``t`` — the layer-2 API
   cannot see past the window it is given.

All engines fail closed on degenerate covariance via
:func:`validate_covariance`. Risk parity uses a Newton iteration on the
convex log-barrier ERC formulation (Roncalli): with ``f(y) = 0.5 y'Σy -
c·Σ log y_i`` strictly convex for ``c > 0`` (the ``diag(c/y²)`` Hessian
term keeps it positive definite even on rank-deficient PSD inputs), the
minimizer satisfies ``y_i (Σy)_i = c`` for every asset — exactly equal risk
contribution after normalization. No published market performance is
claimed; these are textbook constructions for controlled evaluation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from quant_fund.research.allocation.constraints import (
    AllocationConstraints,
    Array,
    DegenerateCovarianceError,
    apply_constraints,
    validate_covariance,
)

EngineName = Literal["inverse_volatility", "risk_parity", "kelly", "vol_target"]

#: Registered engine identifiers.
ENGINE_NAMES: tuple[str, ...] = (
    "inverse_volatility",
    "risk_parity",
    "kelly",
    "vol_target",
)

#: Relative smallest-eigenvalue floor below which Kelly refuses to invert.
#: ERC and inverse-volatility tolerate rank-deficient PSD input (the
#: log-barrier Hessian stays positive definite); Kelly does not.
KELLY_EIGEN_FLOOR = 1e-12

#: Minimum rows of a trailing window before estimators are fitted.
MIN_WINDOW = 5


def estimate_covariance(returns: Array) -> Array:
    """Sample covariance (ddof=1) of a trailing window, validated."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 or r.shape[1] < 1:
        raise ValueError("returns must be a T x N matrix with T >= 2")
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    cov = np.cov(r, rowvar=False, ddof=1)
    if r.shape[1] == 1:
        cov = cov.reshape(1, 1)
    # Symmetrize within machine noise before the strict checks.
    cov = 0.5 * (cov + cov.T)
    return validate_covariance(np.asarray(cov, dtype=float))


def estimate_mean(returns: Array) -> Array:
    """Per-period sample mean of a trailing window."""
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be a finite T x N matrix")
    return np.asarray(r.mean(axis=0), dtype=float)


def inverse_volatility_weights(cov: Array) -> Array:
    """``w_i = (1/σ_i) / Σ_j (1/σ_j)`` — equalizes marginal vol contribution."""
    m = validate_covariance(cov)
    sd = np.sqrt(np.diag(m))
    iv = 1.0 / sd
    return np.asarray(iv / iv.sum(), dtype=float)


def risk_parity_weights(
    cov: Array,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    barrier: float = 1.0,
) -> Array:
    """Equal-risk-contribution weights via Newton on the log-barrier problem.

    Minimizes ``f(y) = 0.5 y'Σy - c Σ_i log y_i`` (``c = barrier``) with a
    damped Newton iteration. The minimizer satisfies ``y_i (Σy)_i = c`` for
    all ``i``, so ``w = y / Σ y`` has exactly equal risk contributions
    ``w_i (Σw)_i = σ_p² / n``. Convergence is measured on the relative risk
    contribution spread itself. Near the optimum the iteration can enter a
    floating-point limit cycle (spread stalls at ~1e-10 while ``f`` no
    longer decreases measurably); the best iterate seen is then accepted
    provided its spread beats a relaxed bound ``max(100 * tol, 1e-7)`` —
    the true residual is still reported by the evaluation layer. Otherwise
    the solver raises rather than returning an unconverged book.
    """
    m = validate_covariance(cov)
    n = m.shape[0]
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be positive and finite")
    if not np.isfinite(barrier) or barrier <= 0.0:
        raise ValueError("barrier must be positive and finite")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    c = float(barrier)
    relaxed_tol = max(100.0 * tol, 1e-7)
    # Inverse-volatility start: feasible, scale-free, deterministic.
    sd = np.sqrt(np.diag(m))
    y = c / sd * n

    def _rc_spread(v: Array) -> float:
        rc = v * (m @ v)
        return float(np.abs(rc - c).max()) / c

    best_spread = math.inf
    best_y = y.copy()
    for _ in range(max_iter):
        spread = _rc_spread(y)
        if spread < best_spread:
            best_spread, best_y = spread, y.copy()
        if spread <= tol:
            w = y / y.sum()
            return np.asarray(w, dtype=float)
        grad = m @ y - c / y
        hess = m + np.diag(c / (y * y))
        direction = np.linalg.solve(hess, grad)
        # Damped Newton: keep y > 0 and decrease the objective.
        step = 1.0
        f_now = float(0.5 * y @ (m @ y) - c * np.log(y).sum())
        grad_dot = float(direction @ grad)
        while step > 1e-12:
            candidate = y - step * direction
            if np.all(candidate > 0.0):
                f_new = float(0.5 * candidate @ (m @ candidate) - c * np.log(candidate).sum())
                if f_new <= f_now - 1e-4 * step * grad_dot or f_new < f_now:
                    y = candidate
                    break
            step *= 0.5
        else:
            break  # line search exhausted; fall through to best-iterate check
    if best_spread <= relaxed_tol:
        w = best_y / best_y.sum()
        return np.asarray(w, dtype=float)
    raise ValueError(
        f"risk parity solver did not converge in {max_iter} iterations "
        f"(best RC spread {best_spread:.3e})"
    )


def kelly_weights(
    mu: Array,
    cov: Array,
    *,
    fraction: float = 1.0,
) -> Array:
    """Continuous-time Kelly book ``w* = f · Σ⁻¹ μ`` (Thorp 2006).

    ``fraction`` in ``(0, 1]`` — 1.0 is full Kelly, 0.5 half Kelly. The
    matrix must be invertible in the above sense: a near-singular
    covariance fails closed with :class:`DegenerateCovarianceError` rather
    than silently using a pseudoinverse. Output is the raw growth-optimal
    book; long-only clipping and the gross cap belong to
    :func:`apply_constraints`.
    """
    m = validate_covariance(cov)
    v = np.asarray(mu, dtype=float).reshape(-1)
    if v.size != m.shape[0] or not np.all(np.isfinite(v)):
        raise ValueError("mu must be a finite vector matching covariance")
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    min_eig = float(np.linalg.eigvalsh(m).min())
    scale = max(float(np.diag(m).max()), 1e-30)
    if min_eig <= KELLY_EIGEN_FLOOR * scale:
        raise DegenerateCovarianceError("covariance is near-singular; Kelly inversion refused")
    w = fraction * np.linalg.solve(m, v)
    return np.asarray(w, dtype=float)


def volatility_target_scale(
    weights: Array,
    cov: Array,
    target_vol: float,
) -> tuple[Array, float]:
    """Scale ``weights`` so ex-ante portfolio vol equals ``target_vol``.

    Returns ``(scaled_weights, leverage)`` with ``leverage =
    target_vol / sqrt(w'Σw)`` — uncapped here; the gross cap lives in
    :func:`apply_constraints`. Fails closed on a zero-variance book.
    Per-period units: ``target_vol`` and ``cov`` must share the same return
    frequency.
    """
    m = validate_covariance(cov)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != m.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching covariance")
    if not np.isfinite(target_vol) or target_vol <= 0.0:
        raise ValueError("target_vol must be positive and finite")
    var = float(w @ m @ w)
    if var <= 0.0:
        raise DegenerateCovarianceError("portfolio variance is zero; cannot target vol")
    leverage = target_vol / math.sqrt(var)
    return np.asarray(w * leverage, dtype=float), float(leverage)


def portfolio_vol(weights: Array, cov: Array) -> float:
    """Ex-ante per-period volatility ``sqrt(w'Σw)`` of a book."""
    m = validate_covariance(cov)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != m.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching covariance")
    var = float(w @ m @ w)
    if var < 0.0:
        raise DegenerateCovarianceError("negative portfolio variance")
    return math.sqrt(max(var, 0.0))


def risk_contributions(weights: Array, cov: Array) -> Array:
    """Absolute risk contributions ``w_i (Σw)_i``; shares sum to 1."""
    m = validate_covariance(cov)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != m.shape[0] or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a finite vector matching covariance")
    return np.asarray(w * (m @ w), dtype=float)


def fit_weights(
    engine: EngineName,
    returns_window: Array,
    constraints: AllocationConstraints | None = None,
    *,
    kelly_fraction: float = 0.5,
    target_vol: float | None = None,
    base_engine: EngineName = "inverse_volatility",
    erc_tol: float = 1e-10,
    erc_max_iter: int = 200,
) -> Array:
    """Fit constrained weights on a trailing returns window (strictly causal).

    ``returns_window`` must contain only observations strictly before the
    decision point — the walk-forward driver in
    :mod:`~quant_fund.research.allocation.evaluation` slices
    ``returns[t-window:t]`` and holds this function to that contract.

    ``"vol_target"`` computes ``base_engine`` weights and rescales them to
    ex-ante ``target_vol`` per period (leverage is then bounded by
    ``constraints.leverage_cap``). ``"kelly"`` uses the window sample mean
    and ``kelly_fraction``.
    """
    if engine not in ENGINE_NAMES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
    if engine == "vol_target":
        if base_engine == "vol_target":
            raise ValueError("vol_target cannot be its own base engine")
        if base_engine not in ENGINE_NAMES:
            raise ValueError(f"unknown base engine {base_engine!r}")
        if target_vol is None or not np.isfinite(target_vol) or target_vol <= 0.0:
            raise ValueError("vol_target requires a positive finite target_vol")
    cons = constraints if constraints is not None else AllocationConstraints()
    r = np.asarray(returns_window, dtype=float)
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError("returns_window must be a finite T x N matrix")
    if r.shape[0] < MIN_WINDOW or r.shape[1] < 1:
        raise ValueError(f"returns_window needs >= {MIN_WINDOW} rows and >= 1 asset")

    cov = estimate_covariance(r)
    raw: Array
    if engine == "inverse_volatility" or (
        engine == "vol_target" and base_engine == "inverse_volatility"
    ):
        raw = inverse_volatility_weights(cov)
    elif engine == "risk_parity" or (engine == "vol_target" and base_engine == "risk_parity"):
        raw = risk_parity_weights(cov, tol=erc_tol, max_iter=erc_max_iter)
    else:
        mu = estimate_mean(r)
        raw = kelly_weights(mu, cov, fraction=kelly_fraction)

    if engine == "vol_target":
        assert target_vol is not None  # validated above
        raw, _ = volatility_target_scale(raw, cov, target_vol)
    return apply_constraints(raw, cons)
