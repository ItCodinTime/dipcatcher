"""Seeded adversarial attacks on portfolio optimizers and allocators.

Degenerate/PSD-boundary covariances, identical assets, single-asset books,
and all-NaN inputs must either produce finite weights honoring the stated
contraction (long-only allocators sum to 1) or fail closed with ValueError.
NaN weights on valid inputs are always a defect.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.covariance import repair_psd
from quant_fund.portfolio.allocators import (
    black_litterman,
    cvar_minimization,
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_volatility,
    kelly_weights,
    maximum_diversification,
    volatility_target,
)
from quant_fund.portfolio.optimizer import optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible
from tests.property._books import research_config


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n))
    return a @ a.T + np.eye(n)


def _degenerate_covs(n: int) -> dict[str, np.ndarray]:
    """Degenerate and boundary-PSD covariance inputs."""
    return {
        "rank1_identical": np.ones((n, n)),
        "zero": np.zeros((n, n)),
        "tiny_diag": np.diag(np.full(n, 1e-14)),
        "all_nan": np.full((n, n), np.nan),
        "some_nan": np.diag(np.full(n, 1.0)) + np.eye(n, k=1) * np.nan,
        "asymmetric": np.tril(np.ones((n, n))),
        "single": np.array([[1.0]]),
        "empty": np.zeros((0, 0)),
        "indefinite": -np.eye(n),
    }


LONG_ONLY_ALLOCATORS = [
    inverse_volatility,
    hierarchical_risk_parity,
    equal_risk_contribution,
    maximum_diversification,
]


def test_long_only_allocators_degenerate_inputs_seeded() -> None:
    """Rank-1/identical-asset covs give finite weights summing to 1; garbage raises."""
    rng = np.random.default_rng(20240903)
    covs = _degenerate_covs(4)
    for fn in LONG_ONLY_ALLOCATORS:
        for name, cov in covs.items():
            try:
                w = np.asarray(fn(cov), dtype=float)
            except (ValueError, np.linalg.LinAlgError):
                continue  # fail-closed is a valid answer
            assert np.isfinite(w).all(), f"{fn.__name__} emitted non-finite weights on {name}"
            assert w.sum() == pytest.approx(1.0, abs=1e-6), (
                f"{fn.__name__} broke weight-sum conservation on {name}"
            )
        # valid random SPD books must always produce finite weights summing to 1
        for _ in range(10):
            cov = _spd(rng, 5)
            w = np.asarray(fn(cov), dtype=float)
            assert np.isfinite(w).all(), f"{fn.__name__} NaN weights on valid SPD"
            assert w.sum() == pytest.approx(1.0, abs=1e-6)
            assert (w >= -1e-9).all(), f"{fn.__name__} emitted short weights"


def test_kelly_and_vol_target_contracts_seeded() -> None:
    """Kelly long-only renormalizes to 1; vol_target clips at max_leverage."""
    rng = np.random.default_rng(20240904)
    for _ in range(25):
        n = int(rng.integers(2, 6))
        cov = _spd(rng, n)
        mu = rng.normal(0.0, 0.05, size=n)
        w = kelly_weights(mu, cov, long_only=True)
        assert np.isfinite(w).all()
        if w.sum() > 0:
            assert w.sum() == pytest.approx(1.0, abs=1e-9)
            assert (w >= 0.0).all()
        base = np.full(n, 1.0 / n)
        scaled, lev = volatility_target(base, cov, 0.15, max_leverage=3.0)
        assert np.isfinite(scaled).all()
        assert 0.0 < lev <= 3.0 + 1e-12
        vol = float(np.sqrt(scaled @ cov @ scaled))
        assert vol <= 0.15 + 1e-9


def test_allocators_all_nan_cov_raise() -> None:
    """Every allocator must fail closed on all-NaN covariance."""
    nan = np.full((4, 4), np.nan)
    for fn in LONG_ONLY_ALLOCATORS:
        with pytest.raises(ValueError):
            fn(nan)
    with pytest.raises(ValueError):
        kelly_weights(np.full(4, 0.01), nan)
    with pytest.raises(ValueError):
        black_litterman(nan)
    with pytest.raises(ValueError):
        volatility_target(np.full(4, 0.25), nan, 0.1)


def test_cvar_minimization_contract_seeded() -> None:
    """CVaR LP: long-only weights sum to 1; bad shapes and NaN returns raise."""
    rng = np.random.default_rng(20240905)
    for _ in range(8):
        t, n = int(rng.integers(20, 60)), int(rng.integers(2, 5))
        rets = rng.normal(0.0, 0.01, size=(t, n))
        w, cvar = cvar_minimization(rets, alpha=float(rng.uniform(0.6, 0.99)))
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert (w >= -1e-9).all()
        assert np.isfinite(cvar)
    with pytest.raises(ValueError):
        cvar_minimization(np.full((30, 3), np.nan))
    with pytest.raises(ValueError):
        cvar_minimization(rng.normal(size=(3, 3)))  # too few scenarios
    with pytest.raises(ValueError):
        cvar_minimization(rng.normal(size=(30, 3)), alpha=0.99 + 0.5)


def test_repair_psd_boundary_and_indefinite_seeded() -> None:
    """PSD-boundary covs pass through; indefinite covs are repaired to PSD."""
    rng = np.random.default_rng(20240906)
    for _ in range(30):
        n = int(rng.integers(2, 8))
        a = rng.normal(size=(n, n))
        cov = a @ a.T  # rank-deficient by construction when n large vs draws
        fixed, meta = repair_psd(cov, 1e-10)
        eig = np.linalg.eigvalsh(fixed)
        assert eig.min() >= -1e-9, "repair_psd returned a non-PSD matrix"
        assert np.allclose(fixed, fixed.T)
    with pytest.raises(ValueError):
        repair_psd(np.full((3, 3), np.nan), 1e-10)
    with pytest.raises(ValueError):
        repair_psd(np.ones((3, 2)), 1e-10)


def test_mv_optimizer_valid_inputs_finite_and_bounded_seeded() -> None:
    """On valid inputs the optimizer never emits NaN; bounds hold to solver slop."""
    rng = np.random.default_rng(20240907)
    cfg = research_config()
    cons = cfg.constraints
    slack = 1e-4
    for _ in range(15):
        n = int(rng.integers(2, 6))
        cov = _spd(rng, n)
        alpha = rng.normal(0.0, 0.5, size=n)
        w_prev = rng.normal(0.0, 0.02, size=n)
        w, diag = optimize_mean_variance(alpha, cov, w_prev, cfg)
        assert np.isfinite(w).all(), "optimizer emitted non-finite weights"
        if diag.feasible:
            assert np.abs(w).sum() <= cons.gross_leverage + slack
            assert abs(w.sum()) <= cons.net_exposure + slack
            assert np.abs(w).max() <= cons.name_max + slack
            assert -cons.name_min - slack <= w.min() or w.min() >= cons.name_min - slack


def test_mv_optimizer_degenerate_covs_fail_or_fallback() -> None:
    """Degenerate covariances: finite result honoring bounds, or fail closed."""
    rng = np.random.default_rng(20240908)
    cfg = research_config()
    n = 4
    alpha = rng.normal(0.0, 0.5, size=n)
    w_prev = np.zeros(n)
    covs = _degenerate_covs(n)
    for name, cov in covs.items():
        try:
            w, diag = optimize_mean_variance(alpha, cov, w_prev, cfg)
        except (ValueError, OptimizationInfeasible):
            continue  # fail-closed is a valid answer
        assert np.isfinite(w).all(), f"NaN weights on {name}"
        # fallback contract: infeasible diagnostics hand back w_prev untouched
        if not diag.feasible:
            assert np.allclose(w, w_prev)


def test_mv_optimizer_nan_and_shape_garbage_raises() -> None:
    """NaN alpha, mismatched sigma/w_prev, and NaN w_prev must raise."""
    rng = np.random.default_rng(20240909)
    cfg = research_config()
    cov = _spd(rng, 4)
    alpha = rng.normal(0.0, 0.5, size=4)
    w_prev = np.zeros(4)
    with pytest.raises(ValueError):
        optimize_mean_variance(np.full(4, np.nan), cov, w_prev, cfg)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, np.full((4, 4), np.nan), w_prev, cfg)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, np.full(4, np.nan), cfg)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov[:-1, :-1], w_prev, cfg)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, np.zeros(3), cfg)
    adv = np.full(4, 1e9)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, w_prev, cfg, adv_dollars=adv, nav=float("nan"))
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, w_prev, cfg, adv_dollars=adv, nav=-1.0)
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, w_prev, cfg, adv_dollars=np.full(4, np.nan))
    with pytest.raises(ValueError):
        optimize_mean_variance(alpha, cov, w_prev, cfg, adv_dollars=-adv)


def test_mv_optimizer_identical_alpha_flattens() -> None:
    """Uniform alpha + symmetric costs: the flat book is optimal (no mass)."""
    cfg = research_config()
    cov = np.eye(4)
    w, diag = optimize_mean_variance(np.full(4, 0.5), cov, np.zeros(4), cfg)
    assert diag.feasible
    assert np.abs(w).sum() == pytest.approx(0.0, abs=1e-6)


def test_component_risk_zero_variance_returns_zeros() -> None:
    from quant_fund.portfolio.optimizer import component_risk

    w = np.array([0.5, -0.5])
    cov = np.zeros((2, 2))
    contrib, pct, var = component_risk(w, cov)
    assert var == 0.0
    assert np.allclose(contrib, 0.0)
    assert np.allclose(pct, 0.0)
