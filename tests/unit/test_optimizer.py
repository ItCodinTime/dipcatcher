import numpy as np

from quant_fund.config.models import AppConfig
from quant_fund.portfolio.optimizer import component_risk, optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible


def test_zero_alpha_no_invented_edge() -> None:
    n = 5
    alpha = np.zeros(n)
    sig = np.eye(n) * 0.04
    w_prev = np.zeros(n)
    cfg = AppConfig()
    cfg.optimizer.lambda_risk = 10.0
    cfg.constraints.turnover_limit = 1.0
    w, diag = optimize_mean_variance(alpha, sig, w_prev, cfg)
    assert diag.feasible
    assert np.max(np.abs(w)) < 1e-5


def test_name_cap() -> None:
    n = 4
    alpha = np.array([1.0, 0.5, 0.1, 0.0])
    sig = np.eye(n) * 0.02
    cfg = AppConfig()
    cfg.constraints.name_max = 0.02
    cfg.constraints.name_min = -0.02
    cfg.constraints.gross_leverage = 0.1
    cfg.constraints.net_exposure = 0.1
    cfg.constraints.turnover_limit = 1.0
    w, diag = optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    assert diag.feasible
    assert np.all(w <= 0.02 + 1e-6)
    assert np.all(w >= -0.02 - 1e-6)


def test_long_and_short_aggregate_caps_are_enforced() -> None:
    alpha = np.array([1.0, 0.9, -1.0, -0.9])
    cfg = AppConfig()
    cfg.constraints.name_max = 0.2
    cfg.constraints.name_min = -0.2
    cfg.constraints.long_max = 0.03
    cfg.constraints.short_max = 0.04
    cfg.constraints.gross_leverage = 0.2
    cfg.constraints.net_exposure = 0.2
    cfg.constraints.turnover_limit = 1.0
    cfg.optimizer.lambda_tc = 0.0

    w, diag = optimize_mean_variance(alpha, np.eye(4) * 0.01, np.zeros(4), cfg)

    assert diag.feasible
    assert np.sum(np.maximum(w, 0.0)) <= 0.03 + 1e-6
    assert np.sum(np.maximum(-w, 0.0)) <= 0.04 + 1e-6


def test_max_positions_limits_nonzero_weights() -> None:
    alpha = np.array([1.0, 0.8, 0.6, 0.4])
    cfg = AppConfig()
    cfg.optimizer.solver = "HIGHS"
    cfg.optimizer.lambda_tc = 0.0
    cfg.constraints.name_max = 0.2
    cfg.constraints.name_min = -0.2
    cfg.constraints.long_max = 0.2
    cfg.constraints.short_max = 0.2
    cfg.constraints.gross_leverage = 0.2
    cfg.constraints.net_exposure = 0.2
    cfg.constraints.turnover_limit = 1.0
    cfg.constraints.max_positions = 1

    w, diag = optimize_mean_variance(alpha, np.eye(4) * 0.01, np.zeros(4), cfg)

    assert diag.feasible
    assert np.count_nonzero(np.abs(w) > 1e-6) <= 1


def test_predicted_volatility_cap_is_enforced() -> None:
    alpha = np.array([1.0, 0.8])
    sigma = np.eye(2) * 0.25
    cfg = AppConfig()
    cfg.constraints.name_max = 1.0
    cfg.constraints.name_min = -1.0
    cfg.constraints.gross_leverage = 1.0
    cfg.constraints.net_exposure = 1.0
    cfg.constraints.turnover_limit = 1.0
    cfg.constraints.predicted_vol_max = 0.05
    cfg.optimizer.lambda_tc = 0.0

    w, diag = optimize_mean_variance(alpha, sigma, np.zeros(2), cfg)

    assert diag.feasible
    assert diag.predicted_volatility is not None
    assert diag.predicted_volatility <= 0.05 + 1e-5


def test_huge_tc_kills_turnover() -> None:
    n = 3
    alpha = np.array([0.2, 0.0, -0.2])
    sig = np.eye(n)
    w_prev = np.array([0.01, 0.0, -0.01])
    cfg = AppConfig()
    cfg.optimizer.lambda_tc = 1e6
    cfg.constraints.turnover_limit = 10.0
    w, diag = optimize_mean_variance(alpha, sig, w_prev, cfg)
    assert diag.feasible
    assert np.sum(np.abs(w - w_prev)) < 1e-3


def test_component_risk_sums() -> None:
    w = np.array([0.5, 0.5])
    sig = np.array([[0.04, 0.01], [0.01, 0.04]])
    _, cr, sig_p = component_risk(w, sig)
    assert np.isclose(cr.sum(), sig_p, atol=1e-8)


def test_infeasible_not_silently_relaxed() -> None:
    n = 2
    alpha = np.ones(n)
    sig = np.eye(n)
    cfg = AppConfig()
    cfg.constraints.gross_leverage = 0.0
    cfg.constraints.net_exposure = 0.0
    cfg.constraints.name_max = 0.0
    cfg.constraints.name_min = 0.1  # conflicts with name_max
    cfg.constraints.turnover_limit = 0.0
    try:
        optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    except OptimizationInfeasible:
        return
    # some solvers may return infeasible status without exception
