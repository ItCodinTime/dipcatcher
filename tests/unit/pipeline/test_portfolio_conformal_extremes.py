"""Wave 9: portfolio_conformal edge extremes (empty, n=2, misaligned, coverage)."""

import numpy as np
import pytest

from quant_fund.portfolio.portfolio_conformal import (
    SplitPortfolioCQR,
    gaussian_band,
    historical_port_vol,
    sets_by_date,
)


def test_sets_by_date_empty_mapping_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        sets_by_date({}, {}, alpha=0.10)


def test_sets_by_date_n2_chrono_split_runs() -> None:
    """n=2 uses tiny chrono split (max(n//3,1)); must not crash."""
    weights = np.full((2, 3), 1.0 / 3.0)
    returns = np.array([[0.01, -0.01, 0.0], [0.02, 0.0, -0.01]])
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.r_p.size == 2
    assert sets.lower.size == 2
    assert sets.upper.size == 2
    assert sets.n_cal >= 0
    # Coverage on tiny test slice is not a live claim — just finite bands.
    assert np.isfinite(sets.lower).all() and np.isfinite(sets.upper).all()
    assert sets.upper[0] >= sets.lower[0]


def test_sets_by_date_misaligned_mapping_dates_intersect() -> None:
    """Common-date intersection; missing dates dropped (not a crash)."""
    weights = {1: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5]), 3: np.array([0.0, 1.0])}
    returns = {2: np.array([0.01, -0.01]), 3: np.array([0.0, 0.02]), 9: np.array([0.1, 0.1])}
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.dates == ["2", "3"]
    assert sets.r_p.size == 2


def test_sets_by_date_length_mismatch_arrays_raise() -> None:
    with pytest.raises(ValueError):
        sets_by_date(
            np.full((3, 2), 0.5),
            np.full((4, 2), 0.01),
            alpha=0.10,
        )


def test_historical_port_vol_tiny_n_floor() -> None:
    assert historical_port_vol(np.array([0.01])) == pytest.approx(1e-8)
    assert historical_port_vol(np.array([np.nan, np.nan])) == pytest.approx(1e-8)
    assert historical_port_vol(np.array([0.0, 0.0])) == pytest.approx(0.0)


def test_gaussian_band_rejects_alpha_bounds() -> None:
    with pytest.raises(ValueError):
        gaussian_band(0.0, 0.01, 0.0)
    with pytest.raises(ValueError):
        gaussian_band(0.0, 0.01, 1.0)


@pytest.mark.parametrize("mu, sigma", [(0.0, -0.01), (float("nan"), 0.01), (0.0, float("inf"))])
def test_gaussian_band_rejects_invalid_location_scale(mu: float, sigma: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        gaussian_band(mu, sigma, 0.05)


def test_coverage_bound_honesty_small_sample_not_guaranteed() -> None:
    """Finite-sample coverage can undershoot nominal — do not claim 1-α."""
    rng = np.random.default_rng(42)
    n_dates, n_names = 40, 4  # small → chrono test slice tiny
    weights = np.full((n_dates, n_names), 1.0 / n_names)
    returns = rng.normal(0.0, 0.02, size=(n_dates, n_names))
    sets = sets_by_date(weights, returns, alpha=0.10)
    y = sets.r_p[sets.test_mask]
    lo = sets.lower[sets.test_mask]
    hi = sets.upper[sets.test_mask]
    if y.size == 0:
        pytest.skip("no test fold on this split")
    cov = float(np.mean((y >= lo) & (y <= hi)))
    # Honesty: we record coverage; we do NOT assert cov >= 1-alpha on tiny n.
    assert 0.0 <= cov <= 1.0
    assert sets.alpha == pytest.approx(0.10)
    # qhat finite; bands defined
    assert np.isfinite(sets.qhat)


def test_split_portfolio_cqr_calibrate_empty_scores() -> None:
    cqr = SplitPortfolioCQR(0.10)
    # All-nan r_p → scores non-finite; quantile path must still return something finite or nan
    r = np.array([np.nan, np.nan, np.nan])
    lo = np.full(3, -0.01)
    hi = np.full(3, 0.01)
    cqr.calibrate(r, lo, hi)
    assert cqr.scores_ is not None
    assert cqr.scores_.size == 3
