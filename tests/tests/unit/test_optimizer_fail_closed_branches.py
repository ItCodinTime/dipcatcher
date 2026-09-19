"""Optimizer fail-closed branches: validation, CVaR mode, solver failure paths.

Complements test_optimizer / test_optimizer_edges / test_optimizer_adv_capacity by
exercising the guard and diagnostics branches that previously had no coverage.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.portfolio.optimizer import optimize_mean_variance

N = 3
ALPHA = np.array([0.05, -0.04, 0.03])
SIG = np.eye(N) * 0.02**2


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.optimizer.lambda_tc = 0.0
    cfg.constraints.turnover_limit = 1.0
    return cfg


def test_sigma_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="sigma must have one row and column per alpha"):
        optimize_mean_variance(ALPHA, np.eye(N + 1) * 0.04, np.zeros(N), _cfg())


def test_w_prev_two_dimensional_raises() -> None:
    # size matches n but ndim != 1 → fail closed instead of flattening silently.
    with pytest.raises(ValueError, match="w_prev must be a finite 1-D array"):
        optimize_mean_variance(ALPHA, SIG, np.zeros((N, 1)), _cfg())


@pytest.mark.parametrize(
    "tc",
    [
        np.ones(N + 1),
        np.array([1.0, -1.0, 1.0]),
        np.array([1.0, np.nan, 1.0]),
    ],
)
def test_tc_linear_validation_fails_closed(tc: np.ndarray) -> None:
    with pytest.raises(ValueError, match="tc_linear must be a finite non-negative vector"):
        optimize_mean_variance(ALPHA, SIG, np.zeros(N), _cfg(), tc_linear=tc)


def test_predicted_vol_max_negative_raises() -> None:
    cfg = _cfg()
    cfg.constraints.predicted_vol_max = -0.1
    with pytest.raises(ValueError, match="predicted_vol_max must be non-negative"):
        optimize_mean_variance(ALPHA, SIG, np.zeros(N), cfg)


def test_max_positions_out_of_range_raises() -> None:
    cfg = _cfg()
    cfg.constraints.max_positions = N + 1
    with pytest.raises(ValueError, match="max_positions must be between zero and"):
        optimize_mean_variance(ALPHA, SIG, np.zeros(N), cfg)


def test_cvar_mode_runs_with_and_without_limit() -> None:
    rng = np.random.default_rng(7)
    scenarios = rng.normal(0.0, 0.02, size=(64, N))
    cfg = _cfg()
    cfg.optimizer.mode = "cvar"
    cfg.optimizer.lambda_risk = 1.0
    w, diag = optimize_mean_variance(ALPHA, SIG, np.zeros(N), cfg, scenarios=scenarios)
    assert diag.feasible
    assert np.all(np.isfinite(w))

    cfg_lim = _cfg()
    cfg_lim.optimizer.mode = "cvar"
    cfg_lim.optimizer.lambda_risk = 1.0
    cfg_lim.constraints.cvar_limit = 0.05
    w_lim, diag_lim = optimize_mean_variance(ALPHA, SIG, np.zeros(N), cfg_lim, scenarios=scenarios)
    assert diag_lim.feasible
    assert np.all(np.isfinite(w_lim))


def test_solver_exception_returns_prior_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(cp.Problem, "solve", boom)
    wp = np.array([0.01, -0.01, 0.0])
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, _cfg())
    assert diag.feasible is False
    assert diag.status == "solver_error"
    assert "solver_error" in diag.conflicting_constraints
    np.testing.assert_allclose(w, wp)


def test_unsolved_problem_returns_prior_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-op solve leaves a non-optimal status → diagnostics, never silent zeros."""
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    wp = np.array([0.02, -0.02, 0.01])
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, _cfg())
    assert diag.feasible is False
    assert diag.message == "infeasible_or_unbounded"
    np.testing.assert_allclose(w, wp)


def test_max_positions_second_solve_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cp.Problem.solve
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("second solve crashed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", flaky)
    cfg = _cfg()
    cfg.constraints.max_positions = 2
    wp = np.zeros(N)
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, cfg)
    assert calls["n"] >= 2
    assert diag.feasible is False
    assert diag.status == "solver_error"
    assert "max_positions" in diag.conflicting_constraints
    np.testing.assert_allclose(w, wp)


def test_valid_sector_factor_beta_loadings_run() -> None:
    """Exposures-when-supplied path: valid loadings tighten but stay feasible."""
    cfg = _cfg()
    cfg.constraints.sector_abs_max = 1.0
    cfg.constraints.factor_abs_max = 1.0
    cfg.constraints.beta_abs_max = 1.0
    sectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    factors = np.array([[0.1], [0.2], [-0.1]])
    beta = np.array([0.5, -0.5, 0.0])
    w, diag = optimize_mean_variance(
        ALPHA,
        SIG,
        np.zeros(N),
        cfg,
        sector_loadings=sectors,
        factor_loadings=factors,
        beta_loadings=beta,
    )
    assert diag.feasible
    assert np.all(np.isfinite(w))


def _fake_solve(status: str, value: float, wval: np.ndarray | None = None):  # noqa: ANN202
    def solve(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self._status = status
        self._value = value
        if wval is not None:
            # Bypass the public setter's finiteness check to simulate a solver
            # that reports OPTIMAL but hands back non-finite weights.
            self.variables()[0]._value = wval

    return solve


def test_invalid_solver_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cp.Problem, "solve", _fake_solve(cp.OPTIMAL, 0.0, np.full(N, np.nan)))
    wp = np.array([0.01, -0.01, 0.0])
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, _cfg())
    assert diag.feasible is False
    assert diag.status == "invalid_solver_output"
    np.testing.assert_allclose(w, wp)


def _first_real_then(fake_solve):  # noqa: ANN001, ANN202
    original = cp.Problem.solve
    calls = {"n": 0}

    def solve(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        calls["n"] += 1
        if calls["n"] >= 2:
            return fake_solve(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    solve.calls = calls  # type: ignore[attr-defined]
    return solve


def test_max_positions_retry_nonoptimal_returns_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve = _first_real_then(_fake_solve(cp.SOLVER_ERROR, float("nan")))
    monkeypatch.setattr(cp.Problem, "solve", solve)
    cfg = _cfg()
    cfg.constraints.max_positions = 2
    wp = np.zeros(N)
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, cfg)
    assert solve.calls["n"] >= 2  # type: ignore[attr-defined]
    assert diag.feasible is False
    assert diag.message == "infeasible_or_unbounded"
    assert "max_positions" in diag.conflicting_constraints
    np.testing.assert_allclose(w, wp)


def test_max_positions_retry_infeasible_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fund.schemas.errors import OptimizationInfeasible

    solve = _first_real_then(_fake_solve(cp.INFEASIBLE, float("nan")))
    monkeypatch.setattr(cp.Problem, "solve", solve)
    cfg = _cfg()
    cfg.constraints.max_positions = 2
    with pytest.raises(OptimizationInfeasible):
        optimize_mean_variance(ALPHA, SIG, np.zeros(N), cfg)


def test_max_positions_retry_invalid_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve = _first_real_then(_fake_solve(cp.OPTIMAL, 0.0, np.full(N, np.nan)))
    monkeypatch.setattr(cp.Problem, "solve", solve)
    cfg = _cfg()
    cfg.constraints.max_positions = 2
    wp = np.array([0.01, -0.01, 0.0])
    w, diag = optimize_mean_variance(ALPHA, SIG, wp, cfg)
    assert diag.feasible is False
    assert diag.status == "invalid_solver_output"
    assert "max_positions" in diag.conflicting_constraints
    np.testing.assert_allclose(w, wp)
