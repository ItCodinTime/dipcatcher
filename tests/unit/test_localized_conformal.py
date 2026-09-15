import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
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


def test_rbf_weights_formula() -> None:
    x_cal = np.array([0.0, 1.0, 2.0])
    w = rbf_weights(x_cal, 1.0, bandwidth=1.0)
    expected = np.exp(-np.square((x_cal - 1.0) / 1.0))
    assert w == pytest.approx(expected, abs=1e-12)
    assert w[1] == pytest.approx(1.0, abs=1e-12)


def test_bandwidth_is_median_positive_pairwise_diff() -> None:
    x = np.array([0.0, 0.0, 2.0, 2.0])
    assert calibration_bandwidth(x) == pytest.approx(2.0, abs=1e-12)
    assert calibration_bandwidth(np.ones(8)) == pytest.approx(1.0, abs=1e-12)


def test_uniform_weights_match_global_quantile() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(size=150)
    alpha = 0.10
    q_u = conformal_quantile(scores, alpha)
    q_w = localized_conformal_quantile(scores, np.ones_like(scores), alpha)
    q_c = localized_conformal_quantile(scores, np.full_like(scores, 2.5), alpha)
    assert q_w == pytest.approx(q_u, abs=1e-12)
    assert q_c == pytest.approx(q_u, abs=1e-12)


def test_homoskedastic_width_similar_to_global() -> None:
    rng = np.random.default_rng(11)
    n = 800
    x = np.ones(n)
    y = rng.normal(0.0, 1.0, size=n)
    lo = np.full(n, -0.40)
    hi = np.full(n, 0.40)
    cal, te = slice(0, 400), slice(400, None)
    model = LocalizedCQR(0.10).calibrate(y[cal], lo[cal], hi[cal], x[cal])
    llo, lhi = model.predict_sets(lo[te], hi[te], x[te])
    q_g = conformal_quantile(cqr_scores(y[cal], lo[cal], hi[cal]), 0.10)
    glo, ghi = expand_interval(lo[te], hi[te], q_g)
    loc_w = float(np.mean(lhi - llo))
    glo_w = float(np.mean(ghi - glo))
    assert loc_w == pytest.approx(glo_w, rel=0.05, abs=1e-12)


def test_heteroskedastic_high_vol_wider_than_low_vol() -> None:
    y_c, lo_c, hi_c, x_c, _y_t, lo_t, hi_t, x_t = _synthetic_het_vol(600, 400, 21)
    model = LocalizedCQR(0.10).calibrate(y_c, lo_c, hi_c, x_c)
    plo, phi = model.predict_sets(lo_t, hi_t, x_t)
    width = phi - plo
    high = x_t > 1.0
    low = ~high
    high_w = float(np.mean(width[high]))
    low_w = float(np.mean(width[low]))
    assert high_w > low_w


def test_exchangeable_coverage_at_least_0_85() -> None:
    y_c, lo_c, hi_c, x_c, y_t, lo_t, hi_t, x_t = _synthetic_het_vol(600, 400, 21)
    model = LocalizedCQR(0.10).calibrate(y_c, lo_c, hi_c, x_c)
    plo, phi = model.predict_sets(lo_t, hi_t, x_t)
    cov = float(np.mean((y_t >= plo) & (y_t <= phi)))
    assert cov >= 0.85


def test_sparse_kernel_falls_back_to_global() -> None:
    x_cal = np.concatenate([np.zeros(80), np.array([10.0, 10.02, 10.04])])
    y = np.concatenate([np.full(80, 0.1), np.array([4.0, 4.2, 4.4])])
    lo = np.zeros_like(y)
    hi = np.zeros_like(y)
    model = LocalizedCQR(0.10, bandwidth=0.05, min_ess=12.0).calibrate(y, lo, hi, x_cal)
    q_global = conformal_quantile(cqr_scores(y, lo, hi), 0.10)
    far_lo, far_hi = model.predict_sets(np.array([0.0]), np.array([0.0]), np.array([10.0]))
    assert float(far_hi[0] - far_lo[0]) == pytest.approx(2.0 * q_global, abs=1e-12)


def test_clip_weights_finite_positive() -> None:
    w = clip_weights(np.array([0.0, 0.5, np.inf, np.nan, 1e9]))
    assert np.all(np.isfinite(w))
    assert np.all(w >= 1e-3 - 1e-12)
    assert np.all(w <= 1e3 + 1e-12)


def test_ess_is_n_for_uniform_weights() -> None:
    w = np.ones(25)
    assert effective_sample_size(w) == pytest.approx(25.0, abs=1e-12)
    assert effective_sample_size(np.zeros(5)) == 0.0


def test_bench_coverage_width_no_sharpe() -> None:
    row = bench_localized_cqr(seed=21)
    assert "coverage" in row
    assert "mean_width" in row
    assert "high_vol_mean_width" in row
    assert "low_vol_mean_width" in row
    assert all("sharpe" not in key.lower() for key in row)
    assert row["n"] > 0
    assert row["high_vol_mean_width"] > row["low_vol_mean_width"]
    assert row["coverage"] >= 0.85


def test_localized_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        LocalizedCQR(0.0)
    with pytest.raises(ValueError):
        localized_conformal_quantile(np.array([1.0, 2.0]), np.ones(2), 0.0)
