"""Wave 17: localized_conformal edge extremes (empty/short/bandwidth/ESS; no Sharpe)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores
from quant_fund.models.localized_conformal import (
    LocalizedCQR,
    _synthetic_het_vol,
    bench_localized_cqr,
    calibration_bandwidth,
    clip_weights,
    effective_sample_size,
    localized_conformal_quantile,
    rbf_weights,
)


def test_empty_scores_return_zero_quantile() -> None:
    assert localized_conformal_quantile(np.asarray([]), np.asarray([]), 0.10) == 0.0
    assert localized_conformal_quantile(np.array([np.nan]), np.array([1.0]), 0.10) == 0.0
    assert localized_conformal_quantile(np.array([1.0, 2.0]), np.array([0.0, -1.0]), 0.10) == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        localized_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.ones(2), 0.10)


def test_bad_alpha_raises() -> None:
    s = np.array([0.1, 0.2, 0.3])
    w = np.ones(3)
    for alpha in (0.0, 1.0, -0.05, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            localized_conformal_quantile(s, w, alpha)
        with pytest.raises(ValueError, match="alpha"):
            LocalizedCQR(alpha)


def test_bad_bandwidth_and_min_ess_raise() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        LocalizedCQR(0.10, bandwidth=0.0)
    with pytest.raises(ValueError, match="bandwidth"):
        LocalizedCQR(0.10, bandwidth=-1.0)
    with pytest.raises(ValueError, match="bandwidth"):
        rbf_weights(np.array([0.0, 1.0]), 0.5, bandwidth=0.0)
    with pytest.raises(ValueError, match="min_ess"):
        LocalizedCQR(0.10, min_ess=0.0)
    with pytest.raises(ValueError, match="min_ess"):
        LocalizedCQR(0.10, min_ess=-5.0)


def test_bad_weight_clip_raises() -> None:
    with pytest.raises(ValueError, match="weight clip"):
        clip_weights(np.ones(3), clip=(2.0, 1.0))
    with pytest.raises(ValueError, match="weight clip"):
        clip_weights(np.ones(3), clip=(-0.1, 1.0))


def test_calibrate_x_cal_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="align"):
        LocalizedCQR(0.10).calibrate(np.ones(5), np.zeros(5), np.ones(5), np.ones(4))


def test_predict_x_query_mismatch_raises() -> None:
    y, lo, hi, x, *_ = _synthetic_het_vol(40, 20, 5)
    model = LocalizedCQR(0.10).calibrate(y, lo, hi, x)
    with pytest.raises(ValueError, match="align"):
        model.predict_sets(lo[:8], hi[:8], x[:3])


def test_empty_calibrate_predict_zero_expansion() -> None:
    model = LocalizedCQR(0.10).calibrate(
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
        np.asarray([], dtype=float),
    )
    plo, phi = model.predict_sets(np.array([-0.25]), np.array([0.25]), np.array([0.5]))
    assert float(plo[0]) == pytest.approx(-0.25, abs=1e-12)
    assert float(phi[0]) == pytest.approx(0.25, abs=1e-12)
    assert model.global_qhat == 0.0


def test_short_n1_and_bandwidth_fallback() -> None:
    assert calibration_bandwidth(np.asarray([])) == pytest.approx(1.0, abs=1e-12)
    assert calibration_bandwidth(np.array([3.0])) == pytest.approx(1.0, abs=1e-12)
    model = LocalizedCQR(0.10).calibrate(
        np.array([0.0]),
        np.array([-0.4]),
        np.array([0.4]),
        np.array([1.0]),
    )
    assert model.bandwidth_ == pytest.approx(1.0, abs=1e-12)
    plo, phi = model.predict_sets(np.array([-0.4]), np.array([0.4]), np.array([1.0]))
    assert np.isfinite(plo[0]) and np.isfinite(phi[0])


def test_ess_edges() -> None:
    assert effective_sample_size(np.asarray([])) == 0.0
    assert effective_sample_size(np.zeros(4)) == 0.0
    assert effective_sample_size(np.array([np.nan, -1.0, 0.0])) == 0.0
    assert effective_sample_size(np.array([7.0])) == pytest.approx(1.0, abs=1e-12)
    # One dominant weight → ESS ≈ 1
    spike = effective_sample_size(np.array([1.0, 1e-12, 1e-12]))
    assert spike == pytest.approx(1.0, abs=1e-6)
    # Uniform → ESS = n
    assert effective_sample_size(np.full(17, 2.5)) == pytest.approx(17.0, abs=1e-12)


def test_low_ess_falls_back_to_global_qhat() -> None:
    """Far query with tiny bandwidth → ESS < min_ess → global conformal quantile."""
    x_cal = np.concatenate([np.zeros(40), np.array([20.0, 20.1])])
    y = np.concatenate([np.full(40, 0.05), np.array([3.0, 3.2])])
    lo = np.zeros_like(y)
    hi = np.zeros_like(y)
    model = LocalizedCQR(0.10, bandwidth=0.01, min_ess=12.0).calibrate(y, lo, hi, x_cal)
    q_global = conformal_quantile(cqr_scores(y, lo, hi), 0.10)
    far_lo, far_hi = model.predict_sets(np.array([0.0]), np.array([0.0]), np.array([20.0]))
    assert float(far_hi[0] - far_lo[0]) == pytest.approx(2.0 * q_global, abs=1e-12)


def test_nonfinite_query_uses_global() -> None:
    y, lo, hi, x, *_ = _synthetic_het_vol(80, 40, 9)
    model = LocalizedCQR(0.10).calibrate(y, lo, hi, x)
    plo, phi = model.predict_sets(np.array([0.0]), np.array([0.0]), np.array([np.nan]))
    assert float(phi[0] - plo[0]) == pytest.approx(2.0 * model.global_qhat, abs=1e-12)


def test_bench_forbids_sharpe_pnl_keys() -> None:
    row = bench_localized_cqr(n_cal=180, n_test=90, seed=13)
    forbidden = ("sharpe", "sortino", "calmar", "pnl", "nav", "return")
    for key in row:
        low = key.lower()
        assert all(tok not in low for tok in forbidden), key
    assert "coverage" in row
    assert "mean_width" in row
    assert row["n"] > 0
