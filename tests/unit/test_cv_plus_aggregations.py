"""CV+ aggregation branches: plus/JAW quantiles, degeneracies, broadcasts.

Complements test_cv_plus (default minmax) by exercising the weighted JAW and
plus aggregation paths, complement-size-one scales, and scalar broadcasts.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cv_plus import (
    CVPlus,
    cv_plus_coverage_level,
    kfold_mean_and_scale,
)


def _fit_model(aggregation: str, *, n: int = 60, alpha: float = 0.10) -> CVPlus:
    rng = np.random.default_rng(11)
    y = rng.normal(0.0, 1.0, size=n)
    pred = rng.normal(0.0, 0.2, size=n)
    model = CVPlus(alpha=alpha, n_folds=5, aggregation=aggregation)
    return model.fit(y, pred)


def test_plus_aggregation_intervals_are_finite_and_ordered() -> None:
    model = _fit_model("plus")
    mid = np.linspace(-1.0, 1.0, 8)
    lo, hi = model.predict_interval(mid, scale=1.0)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(lo <= hi)
    meta = model.metadata()
    assert meta.extra["coverage_identity"] == "1-2*alpha"
    assert model.fold_loc_ is not None and model.scores_ is not None


def test_jaw_aggregation_uniform_matches_plus_shape() -> None:
    model = _fit_model("jaw")
    n = int(model.n_)
    mid = np.linspace(-1.0, 1.0, 6)
    lo_uniform, hi_uniform = model.predict_interval(mid, scale=1.0)
    assert np.all(np.isfinite(lo_uniform)) and np.all(lo_uniform <= hi_uniform)

    weights = np.ones(n, dtype=float)
    lo_w, hi_w = model.predict_interval(mid, scale=1.0, weights=weights)
    assert np.allclose(lo_uniform, lo_w)
    assert np.allclose(hi_uniform, hi_w)

    skewed = np.linspace(0.1, 1.0, n)
    lo_s, hi_s = model.predict_interval(mid, scale=1.0, weights=skewed)
    assert np.all(np.isfinite(lo_s)) and np.all(lo_s <= hi_s)


def test_jaw_weights_length_mismatch_fails_closed() -> None:
    model = _fit_model("jaw")
    with pytest.raises(ValueError, match="weights must have one entry per training residual"):
        model.predict_interval(np.array([0.0]), scale=1.0, weights=np.ones(3))


def test_kfold_complement_single_point_sets_zero_scale() -> None:
    residual = np.array([1.0, 2.0, 3.0, 4.0])
    fold_id = np.array([0, 0, 0, 1])  # fold 0's complement is a single point
    loc, scale = kfold_mean_and_scale(residual, fold_id)
    assert np.allclose(scale[:3], 0.0)
    assert np.isfinite(scale[3])
    assert np.allclose(loc[:3], 4.0)


def test_kfold_single_fold_covering_all_points_fails_closed() -> None:
    with pytest.raises(ValueError, match="each fold must leave at least one complement point"):
        kfold_mean_and_scale(np.array([1.0, 2.0]), np.array([0, 0]))


def test_fit_with_below_above_broadcasts_scalars() -> None:
    y = np.array([0.8, 1.2, 1.1, 0.9, 1.0])
    model = CVPlus(alpha=0.2, n_folds=2, aggregation="minmax")
    model.fit(y, lower=np.array([0.0]), upper=np.array([2.0]))
    lo, hi = model.predict_interval(np.array([1.0]), scale=1.0)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert model.n_ == y.size


def test_fit_pred_scalar_broadcasts() -> None:
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    model = CVPlus(alpha=0.2, n_folds=2)
    model.fit(y, pred=np.array([0.0]))
    assert model.n_ == y.size


def test_predict_sets_empty_mismatch_and_unfitted() -> None:
    unfitted = CVPlus(alpha=0.1, n_folds=3)
    empty_lo, empty_hi = unfitted.predict_sets(np.array([]), np.array([]))
    assert empty_lo.size == 0 and empty_hi.size == 0

    nan_lo, nan_hi = unfitted.predict_sets(np.array([0.0]), np.array([1.0]))
    assert np.isnan(nan_lo).all() and np.isnan(nan_hi).all()

    fitted = _fit_model("minmax")
    with pytest.raises(ValueError, match="lower and upper must have the same length"):
        fitted.predict_sets(np.array([0.0, 1.0]), np.array([1.0]))


def test_predict_interval_scale_edges_and_unfitted() -> None:
    unfitted = CVPlus(alpha=0.1, n_folds=3)
    lo, hi = unfitted.predict_interval(np.array([0.0, 1.0]))
    assert np.isnan(lo).all() and np.isnan(hi).all()

    empty_lo, empty_hi = unfitted.predict_interval(np.array([]))
    assert empty_lo.size == 0 and empty_hi.size == 0

    fitted = _fit_model("minmax")
    with pytest.raises(ValueError, match="x_mid and scale must have the same length"):
        fitted.predict_interval(np.array([0.0, 1.0]), scale=np.array([1.0, 2.0, 3.0]))

    lo2, hi2 = fitted.predict_interval(np.array([0.0, 1.0]), scale=np.array([1.0]))
    assert np.all(np.isfinite(lo2)) and np.all(np.isfinite(hi2))


def test_plus_nan_rows_return_nan_not_garbage() -> None:
    model = _fit_model("plus")
    lo, hi = model.predict_interval(np.array([np.nan, 0.0]), scale=1.0)
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    assert np.isfinite(lo[1]) and np.isfinite(hi[1])


def test_coverage_level_aggregation_dependent() -> None:
    assert cv_plus_coverage_level(0.1, "minmax") == pytest.approx(0.9)
    assert cv_plus_coverage_level(0.1, "plus") == pytest.approx(0.8)
    assert cv_plus_coverage_level(0.1, "jaw") == pytest.approx(0.8)
