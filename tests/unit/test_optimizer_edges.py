"""Wave 20: optimizer / component_risk edge fixtures.

Singular cov, empty/mismatched weights×sigma, gross/net diagnostics.
Does not replace ADV/turnover suites (test_optimizer_adv_capacity,
test_turnover_soft_and_stress).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.portfolio.optimizer import component_risk, optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible


def test_component_risk_empty_weights() -> None:
    w = np.array([])
    sig = np.zeros((0, 0))
    mcr, cr, sig_p = component_risk(w, sig)
    assert mcr.shape == (0,)
    assert cr.shape == (0,)
    assert sig_p == 0.0


def test_component_risk_zero_weights() -> None:
    w = np.zeros(3)
    sig = np.eye(3) * 0.04
    mcr, cr, sig_p = component_risk(w, sig)
    assert sig_p == 0.0
    assert np.allclose(mcr, 0.0)
    assert np.allclose(cr, 0.0)


def test_component_risk_mismatched_raises() -> None:
    w = np.array([0.5, 0.5])
    sig = np.eye(3) * 0.04
    with pytest.raises((ValueError, TypeError)):
        component_risk(w, sig)


@pytest.mark.parametrize(
    ("weights", "sigma", "message"),
    [
        (np.ones((1, 2)), np.eye(2), "weights must be one-dimensional"),
        (np.ones(2), np.eye(3), "sigma must be square"),
        (np.array([np.nan, 0.0]), np.eye(2), "only finite values"),
        (np.array([0.0, 1.0]), np.array([[1.0, 0.0], [0.0, -1.0]]), "invalid portfolio variance"),
    ],
)
def test_component_risk_rejects_invalid_inputs(
    weights: np.ndarray, sigma: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        component_risk(weights, sigma)


def test_component_risk_near_singular_finite() -> None:
    # Rank-1-ish cov: nearly singular but PSD after float noise
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 1))
    sig = a @ a.T + 1e-12 * np.eye(4)
    w = np.array([0.4, -0.2, 0.3, -0.1])
    mcr, cr, sig_p = component_risk(w, sig)
    assert np.isfinite(sig_p) and sig_p >= 0.0
    assert np.all(np.isfinite(mcr))
    assert np.all(np.isfinite(cr))
    assert np.isclose(cr.sum(), sig_p, atol=1e-6)


def test_optimize_w_prev_size_mismatch_fails_closed() -> None:
    """w_prev wrong length cannot silently change turnover economics."""
    n = 3
    alpha = np.array([0.05, -0.04, 0.03])
    sig = np.eye(n) * 0.02**2
    cfg = AppConfig()
    cfg.optimizer.lambda_tc = 0.0
    cfg.constraints.turnover_limit = 1.0
    with pytest.raises(ValueError, match="w_prev must have one value per alpha"):
        optimize_mean_variance(alpha, sig, np.zeros(1), cfg)


def test_optimize_near_singular_cov_repairs_and_runs() -> None:
    n = 3
    # Singular: identical rows → indefinite/near-singular without repair
    base = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, 0.99], [0.99, 0.99, 1.0]]) * 0.04
    # Make slightly indefinite
    eigvals, eigvecs = np.linalg.eigh(base)
    eigvals[0] = -1e-6
    sig = eigvecs @ np.diag(eigvals) @ eigvecs.T
    alpha = np.array([0.02, -0.01, 0.015])
    cfg = AppConfig()
    cfg.optimizer.lambda_risk = 1.0
    cfg.optimizer.lambda_tc = 0.0
    cfg.constraints.turnover_limit = 1.0
    cfg.constraints.gross_leverage = 1.0
    cfg.constraints.net_exposure = 1.0
    w, diag = optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    assert diag.feasible
    assert np.all(np.isfinite(w))
    assert diag.predicted_volatility is not None
    assert np.isfinite(diag.predicted_volatility)


def test_optimizer_rejects_malformed_loadings_and_scenarios() -> None:
    n = 3
    alpha = np.array([0.05, -0.04, 0.03])
    sig = np.eye(n) * 0.02**2
    cfg = AppConfig()
    with pytest.raises(ValueError, match="sector_loadings"):
        optimize_mean_variance(alpha, sig, np.zeros(n), cfg, sector_loadings=np.ones((2, 1)))
    with pytest.raises(ValueError, match="factor_loadings"):
        optimize_mean_variance(
            alpha, sig, np.zeros(n), cfg, factor_loadings=np.ones((n, 1)) * np.nan
        )
    with pytest.raises(ValueError, match="beta_loadings"):
        optimize_mean_variance(alpha, sig, np.zeros(n), cfg, beta_loadings=np.ones(2))
    cfg.optimizer.mode = "cvar"
    with pytest.raises(ValueError, match="scenarios"):
        optimize_mean_variance(alpha, sig, np.zeros(n), cfg, scenarios=np.ones((4, 2)))


def test_optimize_gross_net_diagnostics_populated() -> None:
    n = 4
    alpha = np.array([0.05, -0.04, 0.03, -0.02])
    sig = np.eye(n) * 0.02**2
    cfg = AppConfig()
    cfg.optimizer.lambda_tc = 0.0
    cfg.constraints.gross_leverage = 0.5
    cfg.constraints.net_exposure = 0.2
    cfg.constraints.turnover_limit = 1.0
    cfg.constraints.name_max = 0.2
    cfg.constraints.name_min = -0.2
    w, diag = optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    assert diag.feasible
    assert diag.gross is not None and diag.net is not None
    assert diag.gross <= 0.5 + 1e-5
    assert abs(diag.net) <= 0.2 + 1e-5
    assert float(np.sum(np.abs(w))) == pytest.approx(diag.gross, abs=1e-8)
    assert float(np.sum(w)) == pytest.approx(diag.net, abs=1e-8)


def test_optimize_empty_alpha_edge() -> None:
    """n=0 → empty problem; solver may error or return empty weights."""
    alpha = np.array([])
    sig = np.zeros((0, 0))
    cfg = AppConfig()
    cfg.constraints.turnover_limit = 1.0
    try:
        w, diag = optimize_mean_variance(alpha, sig, np.array([]), cfg)
    except (OptimizationInfeasible, ValueError, Exception):
        return
    assert w.size == 0 or not diag.feasible


def test_optimize_tight_gross_zero_feasible() -> None:
    """gross_leverage=0 with zero w_prev is feasible (all-zero book)."""
    n = 2
    alpha = np.ones(n)
    sig = np.eye(n) * 0.01
    cfg = AppConfig()
    cfg.constraints.gross_leverage = 0.0
    cfg.constraints.net_exposure = 0.0
    cfg.constraints.name_max = 0.0
    cfg.constraints.name_min = 0.0
    cfg.constraints.turnover_limit = 0.0
    cfg.optimizer.lambda_tc = 0.0
    w, diag = optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    assert diag.feasible
    assert np.allclose(w, 0.0, atol=1e-6)
    assert diag.gross is not None and diag.gross <= 1e-6
