import numpy as np
import pytest

from quant_fund.metrics.conformal import set_metrics
from quant_fund.models.jackknife_plus import JackknifePlus

SEED = 11
ALPHA = 0.10
# Barber, Candès, Ramdas, Tibshirani (2021): coverage ≥ 1-2α, plus 0.05 slack.
COVERAGE_FLOOR = 1.0 - 2.0 * ALPHA - 0.05


def test_exchangeable_gaussian_coverage_meets_1_minus_2alpha() -> None:
    rng = np.random.default_rng(SEED)
    y_tr = rng.normal(size=150)
    y_te = rng.normal(size=600)
    jp = JackknifePlus(ALPHA).fit(y_tr, np.zeros_like(y_tr))
    lo, hi = jp.predict_interval(np.zeros_like(y_te), np.ones_like(y_te))
    metrics = set_metrics(y_te, lo, hi)
    assert metrics.n == y_te.size
    assert metrics.mean_width > 0.0
    assert metrics.coverage >= COVERAGE_FLOOR


def test_sets_nest_when_alpha_decreases() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    pred = np.zeros_like(y)
    lo_in = np.full_like(y, -0.2)
    hi_in = np.full_like(y, 0.2)
    wide = JackknifePlus(0.05).fit_residuals(y, lo_in, hi_in)
    tight = JackknifePlus(0.20).fit_residuals(y, lo_in, hi_in)
    wlo, whi = wide.predict_sets(lo_in, hi_in)
    tlo, thi = tight.predict_sets(lo_in, hi_in)
    assert np.all(wlo <= tlo + 1e-12)
    assert np.all(whi >= thi - 1e-12)
    mid = np.zeros(40)
    scale = np.ones(40)
    wlo_i, whi_i = JackknifePlus(0.05).fit(y, pred).predict_interval(mid, scale)
    tlo_i, thi_i = JackknifePlus(0.20).fit(y, pred).predict_interval(mid, scale)
    assert np.all(wlo_i <= tlo_i + 1e-12)
    assert np.all(whi_i >= thi_i - 1e-12)


def test_n1_and_empty_raise_or_nan() -> None:
    jp = JackknifePlus(0.10)
    with pytest.raises(ValueError):
        jp.fit(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        JackknifePlus(0.10).fit(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        JackknifePlus(0.10).fit_residuals(np.array([1.0]), np.array([0.0]), np.array([2.0]))
    empty_lo, empty_hi = JackknifePlus(0.10).predict_sets(np.array([]), np.array([]))
    assert empty_lo.size == 0 and empty_hi.size == 0
    empty_m, empty_s = JackknifePlus(0.10).predict_interval(np.array([]), np.array([]))
    assert empty_m.size == 0 and empty_s.size == 0
    nan_lo, nan_hi = JackknifePlus(0.10).predict_sets(np.array([0.0]), np.array([1.0]))
    assert np.isnan(nan_lo).all() and np.isnan(nan_hi).all()
    nan_a, nan_b = JackknifePlus(0.10).predict_interval(np.array([0.0]), 1.0)
    assert np.isnan(nan_a).all() and np.isnan(nan_b).all()


def test_alpha_must_be_open_unit_interval() -> None:
    with pytest.raises(ValueError):
        JackknifePlus(0.0)
    with pytest.raises(ValueError):
        JackknifePlus(1.0)
