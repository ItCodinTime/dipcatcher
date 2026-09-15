"""CVXPY portfolio optimizer. Infeasible → diagnostics, no silent relaxation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.models.covariance import repair_psd
from quant_fund.schemas.errors import OptimizationInfeasible
from quant_fund.schemas.portfolio import OptimizationDiagnostics

Array = NDArray[np.float64]


def optimize_mean_variance(
    alpha: Array,
    sigma: Array,
    w_prev: Array,
    config: AppConfig,
    *,
    sector_loadings: Array | None = None,
    factor_loadings: Array | None = None,
    beta_loadings: Array | None = None,
    tc_linear: Array | None = None,
    scenarios: Array | None = None,
) -> tuple[Array, OptimizationDiagnostics]:
    import cvxpy as cp

    a = np.asarray(alpha, dtype=float)
    n = a.size
    sig, diag = repair_psd(np.asarray(sigma, dtype=float), config.train.psd_eigen_tol)
    w = cp.Variable(n)
    wp = np.asarray(w_prev, dtype=float)
    if wp.size != n:
        wp = np.zeros(n)
    tc = np.ones(n) if tc_linear is None else np.asarray(tc_linear, dtype=float)
    risk = cp.quad_form(w, sig)
    turnover = cp.norm1(w - wp)
    tcost = config.optimizer.lambda_tc * (tc @ cp.abs(w - wp))
    cons = config.constraints
    constraints = [
        cp.norm1(w) <= cons.gross_leverage,
        cp.abs(cp.sum(w)) <= cons.net_exposure,
        # Cash is the residual portfolio weight and must not fall below the buffer.
        cp.sum(w) <= 1.0 - cons.cash_buffer,
        cp.sum(cp.pos(w)) <= cons.long_max,
        cp.sum(cp.pos(-w)) <= cons.short_max,
        w <= cons.name_max,
        w >= cons.name_min,
        turnover <= cons.turnover_limit,
    ]
    if cons.predicted_vol_max is not None:
        if cons.predicted_vol_max < 0:
            raise ValueError("predicted_vol_max must be non-negative")
        constraints.append(risk <= cons.predicted_vol_max**2)
    if beta_loadings is not None:
        beta = np.asarray(beta_loadings, dtype=float).reshape(-1)
        if beta.size != n:
            raise ValueError("beta_loadings must have one value per asset")
        constraints.append(cp.abs(beta @ w) <= cons.beta_abs_max)
    if cons.max_positions is not None and not 0 <= cons.max_positions <= n:
        raise ValueError("max_positions must be between zero and the number of assets")
    if sector_loadings is not None:
        s = np.asarray(sector_loadings, dtype=float)
        constraints.append(cp.abs(s.T @ w) <= cons.sector_abs_max)
    if factor_loadings is not None:
        f = np.asarray(factor_loadings, dtype=float)
        constraints.append(cp.abs(f.T @ w) <= cons.factor_abs_max)

    obj: cp.Maximize | cp.Minimize
    if config.optimizer.mode == "cvar" and scenarios is not None:
        sc = np.asarray(scenarios, dtype=float)
        s_n = sc.shape[0]
        zeta = cp.Variable()
        u = cp.Variable(s_n)
        losses = -sc @ w
        alpha_c = cons.cvar_alpha
        constraints += [u >= losses - zeta, u >= 0]
        cvar = zeta + 1.0 / ((1.0 - alpha_c) * s_n) * cp.sum(u)
        if cons.cvar_limit is not None:
            constraints.append(cvar <= cons.cvar_limit)
            obj = cp.Maximize(a @ w - tcost)
        else:
            obj = cp.Minimize(cvar + tcost - a @ w)
    else:
        obj = cp.Maximize(a @ w - config.optimizer.lambda_risk * risk - tcost)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=config.optimizer.solver, verbose=False)
    except Exception as exc:  # noqa: BLE001
        return wp, OptimizationDiagnostics(
            status="solver_error",
            feasible=False,
            message=str(exc),
            conflicting_constraints=["solver_error"],
        )
    status = str(prob.status)
    if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
        slack = _constraint_names(constraints)
        diag_out = OptimizationDiagnostics(
            status=status,
            feasible=False,
            message="infeasible_or_unbounded",
            conflicting_constraints=slack,
        )
        if status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
            raise OptimizationInfeasible(diag_out.model_dump_json())
        return wp, diag_out
    wv = np.asarray(w.value, dtype=float).reshape(-1)
    if cons.max_positions is not None:
        # CVXPY's default conic solvers do not support mixed-integer quadratic
        # programs. Solve the convex relaxation, freeze the strongest names, and
        # resolve the original objective on that active set. Zeroing weights can
        # only reduce PSD risk and exposure, while the second solve preserves all
        # turnover and transaction-cost constraints.
        keep_count = cons.max_positions
        keep = np.argsort(np.abs(wv))[-keep_count:] if keep_count else np.array([], dtype=int)
        constraints.extend(w[i] == 0 for i in range(n) if i not in set(keep.tolist()))
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=config.optimizer.solver, verbose=False)
        except Exception as exc:  # noqa: BLE001
            return wp, OptimizationDiagnostics(
                status="solver_error",
                feasible=False,
                message=str(exc),
                conflicting_constraints=["max_positions"],
            )
        status = str(prob.status)
        if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
            diag_out = OptimizationDiagnostics(
                status=status,
                feasible=False,
                message="infeasible_or_unbounded",
                conflicting_constraints=["max_positions", *_constraint_names(constraints)],
            )
            if status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
                raise OptimizationInfeasible(diag_out.model_dump_json())
            return wp, diag_out
        wv = np.asarray(w.value, dtype=float).reshape(-1)
    pred_vol = float(np.sqrt(max(wv @ sig @ wv, 0.0)))
    return wv, OptimizationDiagnostics(
        status=status,
        feasible=True,
        objective=float(prob.value) if prob.value is not None else None,
        expected_alpha=float(a @ wv),
        predicted_volatility=pred_vol,
        predicted_cost=float(tc @ np.abs(wv - wp)),
        turnover=float(np.sum(np.abs(wv - wp))),
        gross=float(np.sum(np.abs(wv))),
        net=float(np.sum(wv)),
    )


def _constraint_names(constraints: list) -> list[str]:
    names = []
    for i, c in enumerate(constraints):
        names.append(f"c{i}:{c}")
    return names[:12]


def component_risk(weights: Array, sigma: Array) -> tuple[Array, Array, float]:
    w = np.asarray(weights, dtype=float)
    s = np.asarray(sigma, dtype=float)
    v = float(w @ s @ w)
    sig_p = float(np.sqrt(max(v, 0.0)))
    if sig_p == 0:
        z = np.zeros_like(w)
        return z, z, 0.0
    sw = s @ w
    mcr = sw / sig_p
    cr = w * sw / sig_p
    return mcr, cr, sig_p
