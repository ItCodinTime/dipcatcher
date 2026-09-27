"""CV+ fail-closed guards and remaining edge branches.

Complements test_cv_plus / test_cv_plus_aggregations: dates passed with explicit
bands, the default zero-pred and unit-scale broadcasts, JAW weight masking, and
the defensive RuntimeError guards on corrupted fitted state.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cv_plus import CVPlus


def _fit(aggregation: str = "minmax", n: int = 60) -> CVPlus:
    rng = np.random.default_rng(7)
    y = rng.normal(size=n)
    return CVPlus(alpha=0.10, n_folds=5, aggregation=aggregation).fit(y, np.zeros_like(y))


def test_jaw_rows_with_no_positive_weights_return_nan() -> None:
    model = _fit("jaw")
    mid = np.linspace(-0.5, 0.5, 4)
    lo, hi = model.predict_interval(mid, scale=1.0, weights=np.zeros(int(model.n_)))
    assert np.isnan(lo).all() and np.isnan(hi).all()


def test_jaw_drops_zero_negative_and_nan_weights_but_stays_finite() -> None:
    model = _fit("jaw")
    weights = np.ones(int(model.n_))
    weights[:10] = 0.0  # dropped by w > 0
    weights[10:15] = np.nan  # dropped by isfinite
    weights[15:20] = -1.0  # dropped by w > 0
    lo, hi = model.predict_interval(np.linspace(-0.5, 0.5, 4), scale=1.0, weights=weights)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo <= hi)


def test_fit_bands_with_dates_groups_surviving_dates() -> None:
    y = np.linspace(-1.0, 1.0, 20)
    dates = np.arange(20.0)
    lower = np.full(20, -0.5)
    upper = np.full(20, 0.5)
    lower[3] = np.nan  # masks the row and its date out of fold assignment
    model = CVPlus(0.10, n_folds=5).fit(y, lower=lower, upper=upper, dates=dates)
    assert model.n_ == 19
    assert model.fold_id_ is not None and model.fold_id_.size == 19
    # Contiguous time blocks: fold ids are non-decreasing in date order.
    assert np.all(np.diff(model.fold_id_) >= 0)
    assert np.unique(model.fold_id_).size == 5


def test_fit_bands_dates_length_mismatch_raises() -> None:
    y = np.linspace(-1.0, 1.0, 20)
    with pytest.raises(ValueError, match="y and dates must have the same length"):
        CVPlus(0.10, n_folds=2).fit(
            y,
            lower=np.full(20, -0.5),
            upper=np.full(20, 0.5),
            dates=np.arange(10),
        )


def test_fit_without_pred_treats_y_as_residual() -> None:
    y = np.arange(20.0)
    model = CVPlus(0.10, n_folds=5).fit(y)
    assert model.fold_loc_ is not None
    # residual = y - 0 = y: fold 0 (rows 0-3) leaves out mean(y[4:]) = 11.5.
    assert model.fold_loc_[:4] == pytest.approx(11.5)
    lo, hi = model.predict_interval(np.zeros(3))
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))


def test_predict_interval_default_scale_matches_unit_scale() -> None:
    model = _fit()
    mid = np.linspace(-1.0, 1.0, 5)
    lo_default, hi_default = model.predict_interval(mid)
    lo_unit, hi_unit = model.predict_interval(mid, scale=np.ones(mid.size))
    assert np.allclose(lo_default, lo_unit)
    assert np.allclose(hi_default, hi_unit)


def test_missing_fold_scale_fails_closed() -> None:
    model = _fit()
    model.fold_scale_ = None
    with pytest.raises(RuntimeError, match="no fitted fold scale"):
        model.predict_interval(np.zeros(3))


def test_missing_scores_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fit()
    model.scores_ = None
    monkeypatch.setattr(model, "_ready", lambda: True)
    with pytest.raises(RuntimeError, match="no fitted scores"):
        model.predict_sets(np.zeros(3), np.ones(3))


def test_missing_fold_locations_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _fit()
    model.fold_loc_ = None
    monkeypatch.setattr(model, "_ready", lambda: True)
    with pytest.raises(RuntimeError, match="no fitted fold locations"):
        model.predict_sets(np.zeros(3), np.ones(3))
