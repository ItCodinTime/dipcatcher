"""Wave 17: weighted_conformal edge extremes (empty/short/mismatch/bins/alpha; no Sharpe)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.weighted_conformal import (
    WeightedSplitCQR,
    _synthetic_vol_shift,
    bench_weighted_cqr,
    likelihood_ratio_weights,
    weighted_conformal_quantile,
)


def test_empty_scores_return_zero_quantile() -> None:
    assert weighted_conformal_quantile(np.asarray([]), np.asarray([]), 0.10) == 0.0
    assert weighted_conformal_quantile(np.array([np.nan, np.nan]), np.ones(2), 0.10) == 0.0
    assert weighted_conformal_quantile(np.array([1.0, 2.0]), np.array([-1.0, 0.0]), 0.10) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(3), 0.10)


def test_bad_alpha_raises() -> None:
    s = np.array([0.1, 0.2, 0.3])
    w = np.ones(3)
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            weighted_conformal_quantile(s, w, alpha)
        with pytest.raises(ValueError, match="alpha"):
            WeightedSplitCQR(alpha)


def test_bad_bins_raises() -> None:
    with pytest.raises(ValueError, match="bins"):
        WeightedSplitCQR(0.10, bins=0)
    with pytest.raises(ValueError, match="bins"):
        likelihood_ratio_weights(np.ones(5), np.ones(3), bins=0)
    with pytest.raises(ValueError, match="bins"):
        likelihood_ratio_weights(np.ones(5), np.ones(3), bins=-2)


def test_calibrate_x_cal_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        WeightedSplitCQR(0.10).calibrate(np.ones(4), np.zeros(4), np.ones(4), np.ones(3))


def test_predict_x_query_mismatch_raises() -> None:
    y, lo, hi, x, *_ = _synthetic_vol_shift(40, 20, 3)
    model = WeightedSplitCQR(0.10).calibrate(y, lo, hi, x)
    with pytest.raises(ValueError, match="align"):
        model.predict_sets(lo[:10], hi[:10], x[:5])


def test_empty_calibrate_predict_zero_expansion() -> None:
    model = WeightedSplitCQR(0.10).calibrate(
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
    )
    plo, phi = model.predict_sets(np.array([-0.2]), np.array([0.2]), np.array([1.0]))
    assert plo.shape == (1,)
    assert phi.shape == (1,)
    assert float(plo[0]) == pytest.approx(-0.2, abs=1e-12)
    assert float(phi[0]) == pytest.approx(0.2, abs=1e-12)


def test_short_n1_calibrate_runs() -> None:
    model = WeightedSplitCQR(0.10).calibrate(
        np.array([0.0]),
        np.array([-0.5]),
        np.array([0.5]),
        np.array([1.0]),
    )
    plo, phi = model.predict_sets(
        np.array([-0.5, -0.5]), np.array([0.5, 0.5]), np.array([1.0, 2.0])
    )
    assert plo.shape == (2,)
    assert phi.shape == (2,)
    assert np.all(np.isfinite(plo))
    assert np.all(np.isfinite(phi))


def test_likelihood_ratio_degenerate_covariate_finite() -> None:
    """Constant / empty-finite covariates fall back to clipped ones."""
    flat = likelihood_ratio_weights(np.ones(30), np.ones(10), bins=4)
    assert flat.shape == (30,)
    assert np.all(np.isfinite(flat))
    assert np.all(flat > 0)
    nan_cal = likelihood_ratio_weights(np.full(8, np.nan), np.ones(4), bins=4)
    assert nan_cal.shape == (8,)
    assert np.all(np.isfinite(nan_cal))
    assert np.all(nan_cal > 0)


def test_bench_forbids_sharpe_pnl_keys() -> None:
    row = bench_weighted_cqr(n_cal=200, n_test=100, seed=11)
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "return")
    for key in row:
        low = key.lower()
        assert all(tok not in low for tok in forbidden), key
    assert "coverage" in row
    assert "mean_width" in row
    assert row["n"] > 0
