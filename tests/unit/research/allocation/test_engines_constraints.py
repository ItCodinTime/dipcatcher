"""Known-answer and fail-closed tests for allocation engines/constraints.

All inputs are SYNTHETIC — correctness fixtures, not market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.research.allocation import (
    AllocationConstraints,
    DegenerateCovarianceError,
    InfeasibleConstraintsError,
    apply_constraints,
    estimate_covariance,
    fit_weights,
    inverse_volatility_weights,
    kelly_weights,
    portfolio_vol,
    risk_contributions,
    risk_parity_weights,
    validate_covariance,
    volatility_target_scale,
    weights_satisfy,
)

Array = NDArray[np.float64]


def _rng_returns(seed: int = 0, t: int = 60, n: int = 3) -> Array:
    """Deterministic synthetic return window with a well-conditioned cov."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = a @ a.T * 1e-4 + np.diag(rng.uniform(1e-4, 4e-4, n))
    return np.asarray(rng.multivariate_normal(np.zeros(n), cov, t), dtype=float)


# --- known answers -------------------------------------------------------


def test_inverse_volatility_two_asset_by_hand() -> None:
    # sigma = (0.02, 0.01) -> inverse vols (50, 100) -> w = (1/3, 2/3).
    cov = np.diag([0.0004, 0.0001])
    w = inverse_volatility_weights(cov)
    assert np.allclose(w, [1.0 / 3.0, 2.0 / 3.0], atol=1e-12)


def test_risk_parity_equals_inverse_vol_on_diagonal_cov() -> None:
    # Uncorrelated assets: ERC weights == inverse-volatility weights.
    cov = np.diag([0.0004, 0.0001, 0.0009])
    w_erc = risk_parity_weights(cov)
    w_iv = inverse_volatility_weights(cov)
    assert np.allclose(w_erc, w_iv, atol=1e-10)


def test_risk_parity_equal_contributions_on_spd() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4))
    cov = a @ a.T * 1e-3 + np.diag(rng.uniform(1e-3, 4e-3, 4))
    w = risk_parity_weights(cov)
    shares = risk_contributions(w, cov)
    shares = shares / shares.sum()
    assert np.allclose(shares, 0.25, atol=1e-8)


def test_risk_parity_raises_when_not_converged() -> None:
    cov = np.diag([0.0004, 0.0001])
    with pytest.raises(ValueError, match="did not converge"):
        risk_parity_weights(cov, max_iter=1, tol=1e-14)


def test_kelly_one_asset_by_hand() -> None:
    # w* = f * mu / sigma^2 = 0.5 * 0.001 / 0.0004 = 1.25.
    w = kelly_weights(np.array([0.001]), np.array([[0.0004]]), fraction=0.5)
    assert np.allclose(w, [1.25], atol=1e-12)


def test_kelly_two_asset_diagonal_by_hand() -> None:
    # w_i = f * mu_i / sigma_i^2 = 0.5 * (0.001/0.0004, 0.0005/0.0001).
    w = kelly_weights(np.array([0.001, 0.0005]), np.diag([0.0004, 0.0001]), fraction=0.5)
    assert np.allclose(w, [1.25, 2.5], atol=1e-12)


def test_volatility_target_scale_by_hand() -> None:
    # sigma_p^2 = 0.25*4e-4 + 0.25*1e-4 = 1.25e-4; lev = 0.01/sqrt(1.25e-4).
    cov = np.diag([0.0004, 0.0001])
    w, lev = volatility_target_scale(np.array([0.5, 0.5]), cov, 0.01)
    expected_lev = 0.01 / np.sqrt(1.25e-4)
    assert np.isclose(lev, expected_lev, atol=1e-12)
    assert np.isclose(portfolio_vol(w, cov), 0.01, atol=1e-12)


# --- fail-closed covariance ----------------------------------------------


def test_zero_variance_asset_fails_closed() -> None:
    with pytest.raises(DegenerateCovarianceError, match="zero-variance"):
        validate_covariance(np.diag([0.0004, 0.0]))


def test_non_psd_covariance_fails_closed() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
    with pytest.raises(DegenerateCovarianceError, match="semidefinite"):
        inverse_volatility_weights(bad)


def test_asymmetric_covariance_fails_closed() -> None:
    bad = np.array([[0.0004, 0.0002], [0.0, 0.0001]])
    with pytest.raises(DegenerateCovarianceError, match="symmetric"):
        risk_parity_weights(bad)


def test_nonfinite_covariance_fails_closed() -> None:
    bad = np.array([[0.0004, np.nan], [np.nan, 0.0001]])
    with pytest.raises(DegenerateCovarianceError, match="finite"):
        validate_covariance(bad)


def test_kelly_refuses_singular_covariance() -> None:
    singular = np.array([[1e-4, 1e-4], [1e-4, 1e-4]])  # PSD, rank 1
    with pytest.raises(DegenerateCovarianceError, match="near-singular"):
        kelly_weights(np.array([0.001, 0.001]), singular)


def test_vol_target_refuses_zero_variance_book() -> None:
    # w = (1, -1) on a perfectly-correlated equal-variance book -> var 0.
    cov = np.array([[1e-4, 1e-4], [1e-4, 1e-4]])
    with pytest.raises(DegenerateCovarianceError, match="zero"):
        volatility_target_scale(np.array([1.0, -1.0]), cov, 0.01)


def test_constant_window_fails_closed() -> None:
    returns = np.full((20, 2), 0.001)
    with pytest.raises(DegenerateCovarianceError):
        estimate_covariance(returns)


# --- constraints ----------------------------------------------------------


def test_apply_constraints_long_only_clips() -> None:
    w = apply_constraints(np.array([-0.5, 1.0]), AllocationConstraints())
    assert np.allclose(w, [0.0, 1.0])


def test_apply_constraints_max_weight() -> None:
    w = apply_constraints(np.array([0.9, 0.1]), AllocationConstraints(max_weight=0.6))
    assert np.allclose(w, [0.6, 0.1])


def test_apply_constraints_leverage_cap_scales_down() -> None:
    w = apply_constraints(np.array([0.6, 0.6]), AllocationConstraints(leverage_cap=0.5))
    assert np.allclose(w, [0.25, 0.25])
    assert np.abs(w).sum() <= 0.5 + 1e-12


def test_apply_constraints_min_weight_drops_small_positions() -> None:
    w = apply_constraints(
        np.array([0.02, 0.48, 0.5]),
        AllocationConstraints(min_weight=0.05),
    )
    assert np.allclose(w, [0.0, 0.48, 0.5])


def test_apply_constraints_short_book_capped() -> None:
    # gross = 3.5 > cap 3.0 -> uniform rescale to 3/3.5.
    w = apply_constraints(
        np.array([2.0, -1.5]),
        AllocationConstraints(long_only=False, leverage_cap=3.0, max_weight=5.0),
    )
    assert np.abs(w).sum() <= 3.0 + 1e-12
    assert np.allclose(w, [2.0 * 3.0 / 3.5, -1.5 * 3.0 / 3.5], atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leverage_cap": 0.0},
        {"max_weight": -1.0},
        {"min_weight": -0.1},
        {"min_weight": 0.9, "max_weight": 0.5},
        {"min_weight": 2.0, "leverage_cap": 1.0},
    ],
)
def test_infeasible_constraints_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(InfeasibleConstraintsError):
        AllocationConstraints(**kwargs)  # type: ignore[arg-type]


def test_weights_satisfy_reports_violations() -> None:
    cons = AllocationConstraints(leverage_cap=1.0, max_weight=0.7, min_weight=0.05)
    # gross = 1.12 > cap; 0.9 > max_weight; 0.02 < min_weight.
    violations = weights_satisfy(np.array([0.9, 0.02, 0.2]), cons)
    assert "leverage_cap" in violations
    assert "max_weight" in violations
    assert "min_weight" in violations


# --- causal driver --------------------------------------------------------


def test_fit_weights_respects_constraints() -> None:
    returns = _rng_returns()
    cons = AllocationConstraints(leverage_cap=1.0, max_weight=0.9)
    for engine in ("inverse_volatility", "risk_parity", "kelly"):
        w = fit_weights(engine, returns, cons)  # type: ignore[arg-type]
        assert weights_satisfy(w, cons) == []


def test_fit_weights_vol_target_reaches_ex_ante_target() -> None:
    returns = _rng_returns()
    cov = estimate_covariance(returns)
    cons = AllocationConstraints(leverage_cap=10.0)
    w = fit_weights("vol_target", returns, cons, target_vol=0.02, base_engine="risk_parity")
    assert np.isclose(portfolio_vol(w, cov), 0.02, atol=1e-10)


def test_fit_weights_vol_target_bounded_by_cap() -> None:
    returns = _rng_returns()
    cons = AllocationConstraints(leverage_cap=1.0)
    w = fit_weights("vol_target", returns, cons, target_vol=0.5)
    assert np.abs(w).sum() <= 1.0 + 1e-12


def test_fit_weights_unknown_engine_rejected() -> None:
    with pytest.raises(ValueError, match="unknown engine"):
        fit_weights("moonshot", _rng_returns())  # type: ignore[arg-type]


def test_fit_weights_short_window_rejected() -> None:
    with pytest.raises(ValueError, match="rows"):
        fit_weights("inverse_volatility", _rng_returns(t=3))


def test_fit_weights_deterministic() -> None:
    returns = _rng_returns()
    a = fit_weights("risk_parity", returns)
    b = fit_weights("risk_parity", returns)
    assert np.array_equal(a, b)
