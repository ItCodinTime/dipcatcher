import numpy as np
import pytest

from quant_fund.metrics.conformal import (
    conformal_quantile,
    cqr_scores,
    expand_interval,
    set_metrics,
)
from quant_fund.models.conformal import (
    AdaptiveConformal,
    MondrianACI,
    MondrianCQR,
    SplitCQR,
    SplitOneSided,
)


def test_conformal_quantile_finite_sample_level() -> None:
    s = np.arange(1.0, 11.0)
    q = conformal_quantile(s, 0.10)
    # ((10+1)*0.9)/10 = 0.99 → higher quantile near the max
    assert q >= 9.0


def test_cqr_coverage_on_exchangeable_residuals() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=800)
    lo = np.full_like(y, -0.5)
    hi = np.full_like(y, 0.5)
    cal, te = y[:400], y[400:]
    lo_c, hi_c = lo[:400], hi[:400]
    cqr = SplitCQR(0.10).calibrate(cal, lo_c, hi_c)
    a, b = cqr.predict_sets(lo[400:], hi[400:])
    cov = float(np.mean((te >= a) & (te <= b)))
    assert cov >= 0.85
    assert cqr.qhat > 0


def test_sets_nest_when_alpha_decreases() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    lo = np.full_like(y, -0.2)
    hi = np.full_like(y, 0.2)
    wide = SplitCQR(0.05).calibrate(y, lo, hi)
    tight = SplitCQR(0.20).calibrate(y, lo, hi)
    wlo, whi = wide.predict_sets(lo, hi)
    tlo, thi = tight.predict_sets(lo, hi)
    assert np.all(wlo <= tlo + 1e-12)
    assert np.all(whi >= thi - 1e-12)


def test_aci_alpha_falls_after_miss_streak() -> None:
    aci = AdaptiveConformal(alpha=0.10, gamma=0.05)
    aci.initialize(np.array([0.0, 0.1]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    start = aci.alpha_t
    for _ in range(8):
        aci.update(1.0)
    assert aci.alpha_t < start


def test_aci_path_covers_exchangeable() -> None:
    rng = np.random.default_rng(2)
    n, names = 60, 8
    y = rng.normal(scale=1.0, size=n * names)
    lo = np.full_like(y, -0.4)
    hi = np.full_like(y, 0.4)
    dates = np.repeat(np.arange(n), names)
    aci = AdaptiveConformal(alpha=0.10, gamma=0.05, score_window=200)
    aci.initialize(y[: 20 * names], lo[: 20 * names], hi[: 20 * names])
    path = aci.run(y[20 * names :], lo[20 * names :], hi[20 * names :], dates[20 * names :])
    cov = float(np.nanmean(path.covered))
    assert cov >= 0.80
    assert path.alpha_t.size == n - 20


def test_expand_interval_symmetric() -> None:
    lo, hi = expand_interval(np.array([0.0]), np.array([1.0]), 0.25)
    assert lo[0] == -0.25
    assert hi[0] == 1.25
    assert np.isclose(cqr_scores(np.array([1.2]), np.array([0.0]), np.array([1.0]))[0], 0.2)


def test_onesided_bound_expands() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    bound = np.full_like(y, 2.5)
    one = SplitOneSided(0.25).calibrate(y, bound)
    out = one.predict_bound(bound)
    assert one.qhat >= 0.0
    assert np.all(out >= bound)


def test_set_metrics_empty() -> None:
    m = set_metrics(np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
    assert m.n == 0
    assert np.isnan(m.coverage)


def test_conformal_quantile_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        conformal_quantile(np.array([1.0, 2.0]), 0.0)


def test_mondrian_covers_heteroskedastic_high_vol() -> None:
    rng = np.random.default_rng(3)
    n = 1200
    vol = rng.choice(np.array([0.4, 2.0]), size=n)
    labels = np.where(vol > 1.0, "high_vol", "low_vol")
    y = rng.normal(0.0, vol)
    lo = np.full(n, -0.25)
    hi = np.full(n, 0.25)
    cal, te = slice(0, 600), slice(600, None)
    glo, ghi = SplitCQR(0.10).calibrate(y[cal], lo[cal], hi[cal]).predict_sets(lo[te], hi[te])
    high = labels[te] == "high_vol"
    gcov = float(np.mean((y[te][high] >= glo[high]) & (y[te][high] <= ghi[high])))
    mon = MondrianCQR(0.10).calibrate(y[cal], lo[cal], hi[cal], labels[cal], vol[cal])
    mlo, mhi = mon.predict_sets(lo[te], hi[te], labels[te], vol[te])
    mcov = float(np.mean((y[te][high] >= mlo[high]) & (y[te][high] <= mhi[high])))
    assert mcov >= 0.85
    assert mcov >= gcov - 1e-9
    assert mon.qhat["high_vol"] >= mon.qhat["low_vol"]


def test_mondrian_aci_tracks_groups() -> None:
    rng = np.random.default_rng(4)
    n, names = 80, 6
    vol = np.where(np.arange(n * names) % 2 == 0, 0.5, 2.0)
    labels = np.where(vol > 1.0, "high_vol", "low_vol")
    y = rng.normal(0.0, vol)
    lo = np.full(y.shape, -0.3)
    hi = np.full(y.shape, 0.3)
    dates = np.repeat(np.arange(n), names)
    warm = 20 * names
    aci = MondrianACI(alpha=0.10, gamma=0.05, score_window=200)
    aci.initialize(y[:warm], lo[:warm], hi[:warm], labels[:warm], vol[:warm])
    path = aci.run(y[warm:], lo[warm:], hi[warm:], dates[warm:], labels[warm:], vol[warm:])
    assert float(np.nanmean(path.covered)) >= 0.80
    assert "high_vol" in aci.groups
    assert "low_vol" in aci.groups
