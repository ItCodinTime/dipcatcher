import numpy as np
import pytest

from quant_fund.metrics.conformal import set_metrics
from quant_fund.models.cv_plus import CVPlus, assign_cv_folds, cv_plus_coverage_level

SEED = 11
ALPHA = 0.10
# Barber et al. (2021) minmax / Theorem 3: coverage ≥ 1-α, plus 0.05 slack.
COVERAGE_FLOOR = 1.0 - ALPHA - 0.05


def test_exchangeable_gaussian_coverage_meets_1_minus_alpha() -> None:
    rng = np.random.default_rng(SEED)
    y_tr = rng.normal(size=150)
    y_te = rng.normal(size=600)
    cv = CVPlus(ALPHA, n_folds=5).fit(y_tr, np.zeros_like(y_tr))
    lo, hi = cv.predict_interval(np.zeros_like(y_te), np.ones_like(y_te))
    metrics = set_metrics(y_te, lo, hi)
    assert metrics.n == y_te.size
    assert metrics.mean_width > 0.0
    assert metrics.coverage >= COVERAGE_FLOOR
    assert "sharpe" not in cv.metadata().extra
    assert cv.metadata().extra["coverage_identity"] == "1-alpha"
    assert cv_plus_coverage_level(ALPHA) == 1.0 - ALPHA


def test_sets_nest_when_alpha_decreases() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    pred = np.zeros_like(y)
    lo_in = np.full_like(y, -0.2)
    hi_in = np.full_like(y, 0.2)
    wide = CVPlus(0.05, n_folds=5).fit(y, lower=lo_in, upper=hi_in)
    tight = CVPlus(0.20, n_folds=5).fit(y, lower=lo_in, upper=hi_in)
    wlo, whi = wide.predict_sets(lo_in, hi_in)
    tlo, thi = tight.predict_sets(lo_in, hi_in)
    assert np.all(wlo <= tlo + 1e-12)
    assert np.all(whi >= thi - 1e-12)
    mid = np.zeros(40)
    scale = np.ones(40)
    wlo_i, whi_i = CVPlus(0.05, n_folds=5).fit(y, pred).predict_interval(mid, scale)
    tlo_i, thi_i = CVPlus(0.20, n_folds=5).fit(y, pred).predict_interval(mid, scale)
    assert np.all(wlo_i <= tlo_i + 1e-12)
    assert np.all(whi_i >= thi_i - 1e-12)


def test_dates_group_folds_by_time_not_stacked_rows() -> None:
    dates = np.repeat(np.arange(10), 5)
    y = np.linspace(-1.0, 1.0, dates.size)
    cv = CVPlus(0.10, n_folds=5).fit(y, np.zeros_like(y), dates=dates)
    assert cv.fold_id_ is not None
    for d in range(10):
        folds = np.unique(cv.fold_id_[dates == d])
        assert folds.size == 1
    # Contiguous time blocks: dates 0-1, 2-3, 4-5, 6-7, 8-9.
    assert assign_cv_folds(10, 5, np.arange(10)).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError, match="unique dates"):
        CVPlus(0.10, n_folds=5).fit(y[:5], np.zeros(5), dates=np.zeros(5))


def test_n1_and_empty_raise_or_nan() -> None:
    with pytest.raises(ValueError):
        CVPlus(0.10, n_folds=5).fit(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        CVPlus(0.10, n_folds=5).fit(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        CVPlus(0.10, n_folds=5).fit(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.zeros(4),
        )
    empty_lo, empty_hi = CVPlus(0.10).predict_sets(np.array([]), np.array([]))
    assert empty_lo.size == 0 and empty_hi.size == 0
    empty_m, empty_s = CVPlus(0.10).predict_interval(np.array([]), np.array([]))
    assert empty_m.size == 0 and empty_s.size == 0
    nan_lo, nan_hi = CVPlus(0.10).predict_sets(np.array([0.0]), np.array([1.0]))
    assert np.isnan(nan_lo).all() and np.isnan(nan_hi).all()
    nan_a, nan_b = CVPlus(0.10).predict_interval(np.array([0.0]), 1.0)
    assert np.isnan(nan_a).all() and np.isnan(nan_b).all()


def test_alpha_and_folds_must_be_valid() -> None:
    with pytest.raises(ValueError):
        CVPlus(0.0)
    with pytest.raises(ValueError):
        CVPlus(1.0)
    with pytest.raises(ValueError):
        CVPlus(0.10, n_folds=1)
    with pytest.raises(ValueError):
        CVPlus(0.10, aggregation="naive")
    assert cv_plus_coverage_level(0.10, "plus") == 0.80
    assert cv_plus_coverage_level(0.10, "jaw") == 0.80
