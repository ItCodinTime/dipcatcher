"""Coverage gaps in portfolio_conformal: key normalization, panel stacking,
vol alignment, mixed/misaligned panels, and bench guards."""

import datetime as dt

import numpy as np
import pytest

from quant_fund.portfolio.portfolio_conformal import (
    SplitPortfolioCQR,
    bench_portfolio_cqr,
    sets_by_date,
)


def test_ordered_keys_falls_back_to_str_sort_for_mixed_types() -> None:
    """int/str keys cannot be compared → fallback orders by str(key): "1" < "b"."""
    weights = {1: np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    returns = {1: np.array([0.02, 0.0]), "b": np.array([0.0, 0.03])}
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.dates == ["1", "b"]
    assert np.isclose(sets.r_p[0], 0.02)
    assert np.isclose(sets.r_p[1], 0.03)


def test_as_date_keys_normalizes_mixed_date_types() -> None:
    """np.datetime64 → str(), date-like → isoformat(), everything else → str()."""
    weights = np.full((4, 2), 0.5)
    returns = np.full((4, 2), 0.01)
    dates = [np.datetime64("2024-01-01"), dt.date(2024, 1, 2), "2024-01-03", 4]
    sets = sets_by_date(weights, returns, alpha=0.10, dates=dates)
    assert sets.dates == ["2024-01-01", "2024-01-02", "2024-01-03", "4"]


def test_mapping_panel_dates_filter_preserves_given_order() -> None:
    """Explicit dates on a mapping panel subset AND reorder the walked dates."""
    weights = {1: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5]), 3: np.array([0.0, 1.0])}
    returns = {1: np.array([0.02, 0.0]), 2: np.array([0.01, -0.01]), 3: np.array([0.0, 0.03])}
    sets = sets_by_date(weights, returns, alpha=0.10, dates=[3, 1])
    assert sets.dates == ["3", "1"]
    assert np.isclose(sets.r_p[0], 0.03)
    assert np.isclose(sets.r_p[1], 0.02)


def test_one_dimensional_panel_treated_as_single_date() -> None:
    """1-d input is one date's cross-section; empty cal fold is not a crash."""
    weights = np.array([0.5, 0.5])
    returns = np.array([0.02, -0.01])
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.dates == ["0"]
    assert np.isclose(sets.r_p[0], 0.005)
    assert sets.n_cal == 0
    assert np.isfinite(sets.lower).all() and np.isfinite(sets.upper).all()
    assert sets.upper[0] >= sets.lower[0]


def test_panel_rejects_non_matrix_array() -> None:
    with pytest.raises(ValueError, match="date-map"):
        sets_by_date(np.zeros((2, 2, 2)), np.zeros((2, 2)), alpha=0.10)


def test_array_panel_rejects_misaligned_dates() -> None:
    weights = np.full((3, 2), 0.5)
    returns = np.full((3, 2), 0.01)
    with pytest.raises(ValueError, match="align"):
        sets_by_date(weights, returns, alpha=0.10, dates=["a", "b"])


def test_vol_scalar_broadcasts_to_all_dates() -> None:
    rng = np.random.default_rng(5)
    n = 60
    weights = np.full((n, 4), 0.25)
    returns = rng.normal(0.0, 0.02, size=(n, 4))
    sets = sets_by_date(weights, returns, alpha=0.10, vol=0.02)
    assert sets.scale is not None
    assert np.allclose(sets.scale, 0.02)


def test_vol_mapping_looks_up_each_date_and_nan_for_missing() -> None:
    """Missing dates are NaN (fail-closed), never backfilled."""
    n = 40
    weights = np.full((n, 2), 0.5)
    returns = np.full((n, 2), 0.01)
    vol = {0: 0.01, 2: 0.03}
    sets = sets_by_date(weights, returns, alpha=0.10, vol=vol)
    assert sets.scale is not None
    assert sets.scale[0] == pytest.approx(0.01)
    assert np.isnan(sets.scale[1])
    assert sets.scale[2] == pytest.approx(0.03)


def test_vol_single_element_array_broadcasts() -> None:
    n = 40
    weights = np.full((n, 2), 0.5)
    returns = np.full((n, 2), 0.01)
    sets = sets_by_date(weights, returns, alpha=0.10, vol=np.array([0.015]))
    assert sets.scale is not None
    assert np.allclose(sets.scale, 0.015)


def test_vol_length_mismatch_raises() -> None:
    weights = np.full((4, 2), 0.5)
    returns = np.full((4, 2), 0.01)
    with pytest.raises(ValueError, match="scalar or align"):
        sets_by_date(weights, returns, alpha=0.10, vol=np.array([0.01, 0.02]))


def test_mixed_panel_types_walk_the_array_date_axis() -> None:
    """Mapping keys [1,2,3] vs array axis [0,1,2]: same length → array axis wins."""
    weights = {1: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5]), 3: np.array([0.0, 1.0])}
    returns = np.array([[0.02, 0.0], [0.01, -0.01], [0.0, 0.03]])
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert sets.dates == ["0", "1", "2"]
    assert np.isclose(sets.r_p[0], 0.02)


def test_mixed_panel_length_mismatch_raises() -> None:
    weights = {1: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5])}
    returns = np.zeros((3, 2))
    with pytest.raises(ValueError, match="same dates"):
        sets_by_date(weights, returns, alpha=0.10)


def test_panel_width_mismatch_pads_with_nan_not_renormalize() -> None:
    """Ragged cross-sections pad to the widest row; NaN names are dropped."""
    weights = {1: np.array([0.5, 0.5]), 2: np.array([1.0])}
    returns = {1: np.array([0.02, 0.04]), 2: np.array([0.03, 0.05, 0.07])}
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert np.isclose(sets.r_p[0], 0.5 * 0.02 + 0.5 * 0.04)
    # Date 2: w padded to [1.0, NaN, NaN] → only the first name contributes.
    assert np.isclose(sets.r_p[1], 0.03)


def test_split_cqr_metadata_reports_family_alpha_and_qhat() -> None:
    r_p = np.array([0.01, -0.02, 0.03, 0.00])
    lo = np.full(4, -0.01)
    hi = np.full(4, 0.01)
    cqr = SplitPortfolioCQR(0.25).calibrate(r_p, lo, hi)
    meta = cqr.metadata()
    assert meta.family == "conformal"
    assert meta.name == "split_portfolio_cqr"
    assert meta.version == "v1"
    assert meta.extra["alpha"] == pytest.approx(0.25)
    assert meta.extra["qhat"] == pytest.approx(cqr.qhat)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_sets_by_date_rejects_alpha_out_of_bounds(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        sets_by_date(np.full((4, 2), 0.5), np.full((4, 2), 0.01), alpha=alpha)


def test_sets_by_date_disjoint_mappings_raise_no_dates() -> None:
    """Zero common dates → fail-closed before any conformal math runs."""
    weights = {1: np.array([1.0]), 2: np.array([0.5])}
    returns = {3: np.array([0.01]), 4: np.array([0.02])}
    with pytest.raises(ValueError, match="no dates"):
        sets_by_date(weights, returns, alpha=0.10)


def test_bench_rejects_partial_panel_arguments() -> None:
    weights = np.full((4, 2), 0.5)
    returns = np.full((4, 2), 0.01)
    with pytest.raises(ValueError, match="both"):
        bench_portfolio_cqr(weights, None)
    with pytest.raises(ValueError, match="both"):
        bench_portfolio_cqr(None, returns)


def test_bench_small_test_fold_reports_nan_kupiec() -> None:
    """Test fold < 10 hits → Kupiec stats reported as NaN, not silently dropped."""
    row = bench_portfolio_cqr(n_dates=12, n_names=4, seed=7)
    assert np.isnan(row["miss_rate"])
    assert np.isnan(row["kupiec_lr"])
    assert np.isnan(row["kupiec_p"])
    assert row["dgp"] == "fixture"
    assert row["seed"] == pytest.approx(7.0)


def test_single_point_calibration_takes_that_score() -> None:
    """k = ceil(2 * (1 - alpha)) > n = 1 → clipped to the single sample score."""
    cqr = SplitPortfolioCQR(0.10).calibrate(np.array([0.05]), np.array([-0.01]), np.array([0.01]))
    assert cqr.qhat == pytest.approx(0.04)
    lo, hi = cqr.predict_sets(np.array([-0.01]), np.array([0.01]))
    assert lo[0] == pytest.approx(-0.05)
    assert hi[0] == pytest.approx(0.05)


def test_predict_sets_zero_scale_collapses_to_base_interval() -> None:
    """Zero scale clamps to 1e-12 → expansion ≈ 0, the base band survives."""
    cqr = SplitPortfolioCQR(0.10).calibrate(
        np.array([0.05, -0.05]), np.array([-0.01, -0.01]), np.array([0.01, 0.01])
    )
    lo, hi = cqr.predict_sets(np.array([-0.01]), np.array([0.01]), np.array([0.0]))
    assert lo[0] == pytest.approx(-0.01, abs=1e-9)
    assert hi[0] == pytest.approx(0.01, abs=1e-9)


def test_predict_sets_never_returns_inverted_interval() -> None:
    """A malformed base band (lower > upper) is normalized to a valid set."""
    cqr = SplitPortfolioCQR(0.10).calibrate(
        np.array([0.0, 0.0]), np.array([-0.01, -0.01]), np.array([0.01, 0.01])
    )
    lo, hi = cqr.predict_sets(np.array([0.05]), np.array([-0.05]))
    assert lo[0] <= hi[0]


def test_all_nan_train_window_falls_back_to_zero_mean() -> None:
    """All-NaN train fold → mu = 0.0 with the vol floor; finite bands, no crash."""
    n = 40
    weights = np.full((n, 2), 0.5)
    returns = np.full((n, 2), 0.01)
    returns[:20] = np.nan
    sets = sets_by_date(weights, returns, alpha=0.10)
    assert np.isnan(sets.r_p[:20]).all()
    assert np.isfinite(sets.qhat)
    assert np.isfinite(sets.lower).all() and np.isfinite(sets.upper).all()
