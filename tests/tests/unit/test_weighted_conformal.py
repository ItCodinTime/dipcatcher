import numpy as np
import pytest

from quant_fund.metrics.conformal import conformal_quantile, cqr_scores, expand_interval
from quant_fund.models.weighted_conformal import (
    WeightedSplitCQR,
    _synthetic_vol_shift,
    bench_weighted_cqr,
    likelihood_ratio_weights,
    weighted_conformal_quantile,
)


def test_uniform_weights_match_unweighted_quantile() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(size=150)
    alpha = 0.10
    q_u = conformal_quantile(scores, alpha)
    q_w = weighted_conformal_quantile(scores, np.ones_like(scores), alpha)
    q_c = weighted_conformal_quantile(scores, np.full_like(scores, 2.5), alpha)
    assert q_w == pytest.approx(q_u, abs=1e-12)
    assert q_c == pytest.approx(q_u, abs=1e-12)


def test_vol_shift_weighted_sets_wider_than_unweighted() -> None:
    y_cal, lo_c, hi_c, x_cal, _y_te, lo_t, hi_t, x_te = _synthetic_vol_shift(800, 400, 7)
    model = WeightedSplitCQR(0.10).calibrate(y_cal, lo_c, hi_c, x_cal)
    wlo, whi = model.predict_sets(lo_t, hi_t, x_te)
    q_u = conformal_quantile(cqr_scores(y_cal, lo_c, hi_c), 0.10)
    ulo, uhi = expand_interval(lo_t, hi_t, q_u)
    assert float(np.mean(whi - wlo)) > float(np.mean(uhi - ulo))


def test_likelihood_ratio_weights_positive_finite() -> None:
    rng = np.random.default_rng(0)
    x_cal = rng.normal(size=200)
    x_te = rng.normal(loc=1.5, size=80)
    w = likelihood_ratio_weights(x_cal, x_te, bins=8)
    assert w.shape == x_cal.shape
    assert np.all(np.isfinite(w))
    assert np.all(w > 0)
    assert float(w.max()) <= 1e3 + 1e-12
    assert float(w.min()) >= 1e-3 - 1e-12
    flat = likelihood_ratio_weights(np.ones(40), np.ones(10), bins=8)
    assert np.all(np.isfinite(flat)) and np.all(flat > 0)


def test_bench_coverage_width_no_sharpe() -> None:
    row = bench_weighted_cqr(seed=7)
    assert "coverage" in row
    assert "mean_width" in row
    assert "median_width" in row
    assert all("sharpe" not in key.lower() for key in row)
    assert row["n"] > 0
    assert row["mean_width"] > row["unweighted_mean_width"]


def test_weighted_quantile_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        weighted_conformal_quantile(np.array([1.0, 2.0]), np.ones(2), 0.0)
