"""Single-session convex allocation in NAV fractions; no silent risk relaxation.

See docs/COST_AWARE_CONSTRUCTION.md for units, assumptions and research basis.
This module accepts forecasts; it does not establish that they predict returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class AllocationConfig:
    risk_window: int = 60
    risk_aversion: float = 20.0
    uncertainty_aversion: float = 1.0
    covariance_shrinkage: float = 0.25
    alpha_scale: float = 1.0
    turnover_limit: float = 0.25

    def validate(self) -> None:
        if type(self.risk_window) is not int or not 20 <= self.risk_window <= 252:
            raise ValueError("risk_window must be an integer in [20, 252]")
        for key in ("risk_aversion", "uncertainty_aversion", "alpha_scale", "turnover_limit"):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.risk_aversion <= 0 or not 0 < self.turnover_limit <= 4:
            raise ValueError("risk_aversion > 0 and turnover_limit in (0, 4] required")
        if (
            isinstance(self.covariance_shrinkage, bool)
            or not np.isfinite(self.covariance_shrinkage)
            or not 0 <= self.covariance_shrinkage <= 1
        ):
            raise ValueError("covariance_shrinkage must be in [0, 1]")


class AllocationFailure(ValueError):
    """A rejected solve with diagnostics; it never provides tradable weights."""

    def __init__(self, message: str, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


def _failed_solve_diagnostic(problem: cp.Problem, status: str) -> dict[str, Any]:
    stats = problem.solver_stats
    iterations = getattr(stats, "num_iters", None) if stats is not None else None
    elapsed = getattr(stats, "solve_time", None) if stats is not None else None
    return {
        "solver": "CLARABEL",
        "status": status,
        "iterations": iterations if type(iterations) is int and iterations >= 0 else None,
        "solve_time_seconds": float(elapsed)
        if isinstance(elapsed, (int, float)) and np.isfinite(elapsed) and elapsed >= 0
        else None,
        "weights_accepted": False,
    }


def allocate(
    alpha: np.ndarray,
    covariance: np.ndarray,
    uncertainty: np.ndarray,
    previous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: np.ndarray,
    impact: np.ndarray,
    *,
    config: AllocationConfig,
    linear_cost: float,
    gross_limit: float,
    name_limit: float,
    cash_buffer: float,
    borrow_cost: float = 0.0,
    funding_cost: float = 0.0,
    cash_return: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize from actual holdings, with costs and capacity normalized by NAV.

    impact_i * |delta_weight_i|**1.5 is the estimated impact/NAV. Holding
    rates and forecasts are for one planning session. All inputs are as-of.
    """
    config.validate()
    a, cov = np.asarray(alpha, dtype=float), np.asarray(covariance, dtype=float)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError("alpha must be a nonempty finite vector")
    vectors = [
        np.asarray(v, dtype=float) for v in (uncertainty, previous, lower, upper, capacity, impact)
    ]
    if any(v.shape != a.shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError("allocation vectors must be finite and match alpha")
    u, prev, lo, hi, cap, imp = vectors
    if np.any(u < 0) or np.any(cap < 0) or np.any(imp < 0) or np.any(lo > hi):
        raise ValueError("invalid uncertainty, capacity, impact or bounds")
    if (
        cov.shape != (a.size, a.size)
        or not np.isfinite(cov).all()
        or not np.allclose(cov, cov.T, atol=1e-12, rtol=0)
    ):
        raise ValueError("covariance must be finite, square and symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    cov = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    rates = [linear_cost, borrow_cost, funding_cost, cash_return]
    if not np.isfinite(rates).all() or min(rates) < 0 or funding_cost < cash_return:
        raise ValueError("nonnegative costs and funding_cost >= cash_return required")
    if (
        not np.isfinite([gross_limit, name_limit, cash_buffer]).all()
        or not 0 < gross_limit <= 2
        or not 0 < name_limit <= 1
        or not 0 <= cash_buffer < 1
    ):
        raise ValueError("invalid exposure or cash limits")

    w = cp.Variable(a.size)
    delta = cp.Variable(a.size, nonneg=True)
    trade_cost = cp.Variable(nonneg=True)
    estimated_cost = linear_cost * cp.sum(delta) + (
        imp[imp > 0] @ cp.power(delta[imp > 0], 1.5) if np.any(imp > 0) else cp.Constant(0)
    )
    # Cash opportunity cost + incremental debt spread is convex even with leverage.
    holding_cost = (
        borrow_cost * cp.sum(cp.pos(-w))
        + cash_return * cp.sum(w)
        + (funding_cost - cash_return) * cp.pos(cp.sum(w) - 1)
    )
    objective = (
        a @ w
        - config.risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
        - config.uncertainty_aversion * (u @ cp.abs(w))
        - trade_cost
        - holding_cost
    )
    constraints = [
        delta >= w - prev,
        delta >= prev - w,
        trade_cost >= estimated_cost,
        w >= lo,
        w <= hi,
        delta <= cap,
        cp.sum(delta) <= config.turnover_limit,
        cp.norm1(w) + gross_limit * trade_cost <= gross_limit,
        cp.abs(w) + name_limit * trade_cost <= name_limit,
    ]
    if gross_limit <= 1:
        constraints.append(cp.sum(w) + trade_cost <= 1 - cash_buffer)
    # Positive objective scaling improves conditioning without changing the optimum.
    problem = cp.Problem(cp.Maximize(100 * objective), constraints)
    try:
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-8, tol_feas=1e-8, tol_gap_rel=1e-8)
    except cp.error.SolverError as exc:
        raise AllocationFailure(
            f"allocation solver failed: {exc}",
            _failed_solve_diagnostic(problem, "solver_error"),
        ) from exc
    if problem.status != cp.OPTIMAL or w.value is None:
        raise AllocationFailure(
            f"allocation failed: {problem.status}",
            _failed_solve_diagnostic(problem, str(problem.status)),
        )
    result = np.asarray(w.value, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("allocation solver returned nonfinite weights")
    # Suppress numerical dust around no-trade, then recheck every constraint.
    result[np.abs(result - prev) < 1e-8] = prev[np.abs(result - prev) < 1e-8]
    result[np.abs(result) < 1e-10] = 0.0
    w.value = result
    delta.value = np.abs(result - prev)
    trade_cost.value = float(linear_cost * delta.value.sum() + imp @ delta.value**1.5)
    violation = max(float(np.max(c.violation())) for c in constraints)
    if not np.isfinite(violation) or violation > 1e-7:
        raise ValueError(f"allocation constraint residual {violation} exceeds tolerance")
    d = np.abs(result - prev)
    return result, {
        "status": problem.status,
        "solver": "CLARABEL",
        "objective": float(objective.value),
        "expected_return_proxy": float(a @ result),
        "predicted_variance": float(result @ cov @ result),
        "uncertainty_penalty": float(config.uncertainty_aversion * (u @ np.abs(result))),
        "predicted_linear_cost": float(linear_cost * d.sum()),
        "predicted_impact_cost": float(imp @ d**1.5),
        "predicted_holding_cost": float(holding_cost.value),
        "turnover": float(d.sum()),
        "max_constraint_violation": violation,
        "previous_weights": prev.tolist(),
        "target_weights": result.tolist(),
        "alpha": a.tolist(),
        "uncertainty": u.tolist(),
        "capacity": cap.tolist(),
    }
