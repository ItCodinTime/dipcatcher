"""Research allocators with explicit information and numerical contracts.

Boyd et al. (2017): finite-horizon convex trading (constant wealth approximation).
Busseti, Ryu & Boyd (2016): risk-constrained Kelly on a discrete scenario law.
Moreira & Muir (2017): inverse *variance* exposure, with training-only scaling.
These are reduced research models, not reproductions of published market results.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

Array = NDArray[np.float64]


@dataclass(frozen=True)
class AllocationResult:
    weights: Array
    objective: float
    solver_status: str
    constraint_error: float


def _positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def multi_period_target(
    forecasts: Array,
    covariance: Array,
    previous: Array,
    *,
    risk_aversion: float = 1.0,
    linear_cost: float = 0.0,
    quadratic_cost: float = 0.0,
    gross_limit: float = 1.0,
    turnover_limit: float = 2.0,
) -> AllocationResult:
    """Plan long-only risky weights, leaving residual cash at zero interest.

    Maximize sum_h [mu_h'w_h - gamma/2 w_h'Sigma_h w_h
    - c ||w_h-w_(h-1)||_1 - eta/2 ||w_h-w_(h-1)||_2^2].
    All forecasts must be available at the current decision; execute only row 0
    and replan. Covariance and returns use the same per-period units. No terminal
    liquidation is assumed. Wealth and intervening weight drift are approximated
    as constant during the planning horizon, as in convex multi-period research.
    """
    mu = np.asarray(forecasts, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    prev = np.asarray(previous, dtype=float)
    if mu.ndim != 2 or min(mu.shape) < 1 or not np.isfinite(mu).all():
        raise ValueError("forecasts must be a nonempty finite H x N matrix")
    h, n = mu.shape
    if prev.shape != (n,) or not np.isfinite(prev).all() or np.any(prev < 0):
        raise ValueError("previous must be a nonnegative finite N vector")
    _positive(risk_aversion, "risk_aversion")
    _positive(gross_limit, "gross_limit")
    if gross_limit > 1 or prev.sum() > 1 + 1e-10:
        raise ValueError("long-only cash-funded portfolios require exposure <= 1")
    for name, value in [
        ("linear_cost", linear_cost),
        ("quadratic_cost", quadratic_cost),
        ("turnover_limit", turnover_limit),
    ]:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if cov.shape == (n, n):
        cov = np.broadcast_to(cov, (h, n, n)).copy()
    if cov.shape != (h, n, n) or not np.isfinite(cov).all():
        raise ValueError("covariance must be finite N x N or H x N x N")
    for matrix in cov:
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(matrix).min() < 0:
            raise ValueError("covariance must be positive semidefinite")
    weights = cp.Variable((h, n))
    constraints = [weights >= 0, cp.sum(weights, axis=1) <= gross_limit]
    terms = []
    for step in range(h):
        delta = weights[step] - (prev if step == 0 else weights[step - 1])
        constraints.append(cp.norm1(delta) <= turnover_limit)
        terms.append(
            mu[step] @ weights[step]
            - risk_aversion / 2 * cp.quad_form(weights[step], cp.psd_wrap(cov[step]))
            - linear_cost * cp.norm1(delta)
            - quadratic_cost / 2 * cp.sum_squares(delta)
        )
    problem = cp.Problem(cp.Maximize(sum(terms)), constraints)
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError(f"multi-period optimization failed: {problem.status}")
    plan = np.asarray(weights.value, dtype=float)
    delta = np.diff(np.vstack([prev, plan]), axis=0)
    error = max(
        0.0,
        float(-plan.min()),
        float(plan.sum(axis=1).max() - gross_limit),
        float(np.abs(delta).sum(axis=1).max() - turnover_limit),
    )
    if not np.isfinite(plan).all() or error > 1e-7:
        raise ValueError("multi-period solution violates feasibility tolerance")
    return AllocationResult(plan, float(problem.value), str(problem.status), error)


def risk_constrained_kelly(
    gross_returns: Array,
    probabilities: Array,
    *,
    wealth_floor: float = 0.7,
    drawdown_probability: float = 0.1,
) -> AllocationResult:
    """Scenario Kelly with E[growth**(-lambda)] <= 1, lambda=log(beta)/log(alpha).

    Appends a unit-return cash asset, so the last returned weight is cash.
    alpha is the wealth floor relative to INITIAL wealth, not a trailing peak.
    The bound assumes repeated IID outcomes from the specified scenario law;
    fitting that law to history does not establish a real-market guarantee.
    """
    gross = np.asarray(gross_returns, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if (
        gross.ndim != 2
        or min(gross.shape) < 1
        or not np.isfinite(gross).all()
        or np.any(gross <= 0)
    ):
        raise ValueError("gross_returns must be a strictly positive finite S x N matrix")
    if p.shape != (gross.shape[0],) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("probabilities must be strictly positive and match scenarios")
    if not np.isclose(p.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    for name, value in [
        ("wealth_floor", wealth_floor),
        ("drawdown_probability", drawdown_probability),
    ]:
        if not np.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must lie in (0, 1)")
    lam = float(np.log(drawdown_probability) / np.log(wealth_floor))
    outcomes = np.column_stack([gross, np.ones(gross.shape[0])])
    # Jensen: E[log(g)] <= log(E[g]) <= 0 when every risky mean <= 1.
    # Cash is then an exact optimum; the cone has no strictly feasible point.
    if np.all(p @ gross <= 1.0):
        cash = np.zeros(outcomes.shape[1])
        cash[-1] = 1.0
        return AllocationResult(cash, 0.0, "analytic_cash_optimum", 0.0)
    weights = cp.Variable(outcomes.shape[1])
    growth = outcomes @ weights
    # Share one log-growth hypograph between objective and risk constraint.
    # This avoids duplicate exponential-cone representations near unit growth.
    log_growth = cp.Variable(gross.shape[0])
    log_moment = cp.log_sum_exp(np.log(p) - lam * log_growth)
    # Keep the log-growth objective unscaled: large multipliers can produce
    # an inaccurate conic termination at daily return magnitudes.
    problem = cp.Problem(
        cp.Maximize(p @ log_growth),
        [weights >= 0, cp.sum(weights) == 1, log_growth <= cp.log(growth), log_moment <= 0],
    )
    problem.solve(solver="CLARABEL", max_iter=300)
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise ValueError(f"Kelly optimization failed: {problem.status}")
    w = np.asarray(weights.value, dtype=float)
    if not np.isfinite(w).all() or np.any(outcomes @ w <= 0):
        raise ValueError("Kelly returned invalid wealth")
    moment = float(logsumexp(np.log(p) - lam * np.log(outcomes @ w)))
    error = max(0.0, float(-w.min()), abs(float(w.sum()) - 1), moment)
    if error > 1e-7:
        raise ValueError("Kelly solution violates feasibility tolerance")
    return AllocationResult(w, float(p @ np.log(outcomes @ w)), str(problem.status), error)


def volatility_managed_weights(
    returns: Array,
    *,
    training_scale: float,
    window: int = 21,
    variance_floor: float = 1e-8,
    max_leverage: float = 1.0,
) -> Array:
    """Causal capped c / variance positions: output[t] uses returns strictly before t.

    Returns are equally spaced simple strategy/factor returns. ``training_scale``
    must be frozen using a separate earlier training sample. Rolling population
    variance substitutes for the paper's previous-month realized variance. Zero
    exposure during warmup; floors/caps are explicit departures from the paper.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all() or np.any(r < -1):
        raise ValueError("returns must be a finite vector of simple returns >= -1")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer >= 2")
    for name, value in [
        ("training_scale", training_scale),
        ("variance_floor", variance_floor),
        ("max_leverage", max_leverage),
    ]:
        _positive(value, name)
    w = np.zeros(r.size)
    for t in range(window, r.size):
        variance = float(np.var(r[t - window : t]))
        w[t] = min(max_leverage, training_scale / max(variance_floor, variance))
    return w
