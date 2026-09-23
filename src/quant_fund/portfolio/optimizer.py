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
    adv_dollars: Array | None = None,
    nav: float = 1.0,
) -> tuple[Array, OptimizationDiagnostics]:
    import cvxpy as cp

    a = np.asarray(alpha, dtype=float)
    n = a.size
    if a.ndim != 1 or n == 0 or not np.isfinite(a).all():
        raise ValueError("alpha must be a non-empty finite 1-D array")
    sig, diag = repair_psd(np.asarray(sigma, dtype=float), config.train.psd_eigen_tol)
    if sig.shape != (n, n):
        raise ValueError("sigma must have one row and column per alpha")
    w = cp.Variable(n)
    wp = np.asarray(w_prev, dtype=float)
    if wp.size != n:
        raise ValueError("w_prev must have one value per alpha")
    if wp.ndim != 1 or not np.isfinite(wp).all():
        raise ValueError("w_prev must be a finite 1-D array")
    tc = np.ones(n) if tc_linear is None else np.asarray(tc_linear, dtype=float)
    if tc.shape != (n,) or not np.isfinite(tc).all() or np.any(tc < 0):
        raise ValueError("tc_linear must be a finite non-negative vector matching alpha")
    risk = cp.quad_form(w, sig)
    turnover = cp.norm1(w - wp)
    tcost = config.optimizer.lambda_tc * (tc @ cp.abs(w - wp))
    # Explicit soft turnover penalty (distinct from linear TC and hard limit).
    turn_pen = float(getattr(config.optimizer, "lambda_turnover", 0.0) or 0.0) * turnover
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
    # HARD capacity: |Δw_i| * nav / ADV_i <= max_adv_participation (when ADV provided)
    if adv_dollars is not None and cons.max_adv_participation is not None:
        adv = np.asarray(adv_dollars, dtype=float).reshape(-1)
        if adv.size != n:
            raise ValueError("adv_dollars must have one value per asset")
        if not np.isfinite(adv).all() or np.any(adv <= 0):
            raise ValueError("adv_dollars must be finite and positive")
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("nav must be finite and positive when ADV capacity is enabled")
        adv_safe = adv
        # |w - wp| * nav / adv <= cap  ⇒  |w - wp| <= cap * adv / nav
        cap = float(cons.max_adv_participation) * adv_safe / float(nav)
        constraints.append(cp.abs(w - wp) <= cap)
    if cons.predicted_vol_max is not None:
        if cons.predicted_vol_max < 0:
            raise ValueError("predicted_vol_max must be non-negative")
        constraints.append(risk <= cons.predicted_vol_max**2)
    if beta_loadings is not None:
        beta = np.asarray(beta_loadings, dtype=float).reshape(-1)
        if beta.size != n or not np.isfinite(beta).all():
            raise ValueError("beta_loadings must be a finite vector with one value per asset")
        constraints.append(cp.abs(beta @ w) <= cons.beta_abs_max)
    if cons.max_positions is not None and not 0 <= cons.max_positions <= n:
        raise ValueError("max_positions must be between zero and the number of assets")
    if sector_loadings is not None:
        s = np.asarray(sector_loadings, dtype=float)
        if s.ndim != 2 or s.shape[0] != n or not np.isfinite(s).all():
            raise ValueError("sector_loadings must be a finite matrix with one row per asset")
        constraints.append(cp.abs(s.T @ w) <= cons.sector_abs_max)
    if factor_loadings is not None:
        f = np.asarray(factor_loadings, dtype=float)
        if f.ndim != 2 or f.shape[0] != n or not np.isfinite(f).all():
            raise ValueError("factor_loadings must be a finite matrix with one row per asset")
        constraints.append(cp.abs(f.T @ w) <= cons.factor_abs_max)

    obj: cp.Maximize | cp.Minimize
    if config.optimizer.mode == "cvar":
        if scenarios is None:
            raise ValueError(
                "scenarios are required when optimizer.mode is cvar; refusing mean-variance fallback"
            )
        sc = np.asarray(scenarios, dtype=float)
        if sc.ndim != 2 or sc.shape[1] != n or sc.shape[0] == 0 or not np.isfinite(sc).all():
            raise ValueError(
                "scenarios must be a non-empty finite matrix with one column per asset"
            )
        s_n = sc.shape[0]
        zeta = cp.Variable()
        u = cp.Variable(s_n)
        losses = -sc @ w
        alpha_c = cons.cvar_alpha
        constraints += [u >= losses - zeta, u >= 0]
        cvar = zeta + 1.0 / ((1.0 - alpha_c) * s_n) * cp.sum(u)
        if cons.cvar_limit is not None:
            constraints.append(cvar <= cons.cvar_limit)
            obj = cp.Maximize(a @ w - tcost - turn_pen)
        else:
            obj = cp.Minimize(cvar + tcost + turn_pen - a @ w)
    else:
        obj = cp.Maximize(a @ w - config.optimizer.lambda_risk * risk - tcost - turn_pen)

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
    if wv.shape != (n,) or not np.isfinite(wv).all():
        return wp, OptimizationDiagnostics(
            status="invalid_solver_output",
            feasible=False,
            message="solver returned non-finite or incorrectly shaped weights",
            conflicting_constraints=["solver_output"],
        )
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
        if wv.shape != (n,) or not np.isfinite(wv).all():
            return wp, OptimizationDiagnostics(
                status="invalid_solver_output",
                feasible=False,
                message="solver returned non-finite or incorrectly shaped weights",
                conflicting_constraints=["max_positions", "solver_output"],
            )
    variance = float(wv @ sig @ wv)
    if not np.isfinite(variance) or variance < -1e-10:
        return wp, OptimizationDiagnostics(
            status="invalid_solver_output",
            feasible=False,
            message="solver returned invalid portfolio variance",
            conflicting_constraints=["solver_output"],
        )
    pred_vol = float(np.sqrt(max(variance, 0.0)))
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
    if w.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if s.ndim != 2 or s.shape != (w.size, w.size):
        raise ValueError("sigma must be square with one row per weight")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(s)):
        raise ValueError("weights and sigma must contain only finite values")
    v = float(w @ s @ w)
    if not np.isfinite(v) or v < -1e-10:
        raise ValueError("sigma produces an invalid portfolio variance")
    sig_p = float(np.sqrt(max(v, 0.0)))
    if sig_p == 0:
        z = np.zeros_like(w)
        return z, z, 0.0
    sw = s @ w
    mcr = sw / sig_p
    cr = w * sw / sig_p
    return mcr, cr, sig_p
