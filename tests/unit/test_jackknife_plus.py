from pathlib import Path

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


def test_length_mismatch_and_coverage_identity() -> None:
    from quant_fund.models.jackknife_plus import jackknife_plus_coverage_level

    y = np.linspace(-1.0, 1.0, 30)
    with pytest.raises(ValueError, match="same length"):
        JackknifePlus(0.10).fit(y, np.zeros(29))
    with pytest.raises(ValueError, match="same length"):
        JackknifePlus(0.10).fit_residuals(y, np.full(10, -0.1), np.full(30, 0.1))
    fitted = JackknifePlus(0.10).fit(y, np.zeros_like(y))
    with pytest.raises(ValueError, match="same length"):
        fitted.predict_sets(np.zeros(2), np.ones(3))
    with pytest.raises(ValueError, match="same length"):
        fitted.predict_interval(np.zeros(2), np.ones(4))
    assert jackknife_plus_coverage_level(0.10) == pytest.approx(0.80)
    meta = fitted.metadata()
    assert meta.extra["coverage_identity"] == "1-2*alpha"
    assert meta.extra["coverage_guarantee_scope"] == "marginal_exchangeable"
    assert "not training-conditional" in str(meta.extra["coverage_guarantee_claim"])
    assert meta.extra["research_only"] is True
    assert "sharpe" not in meta.extra


def test_vol_scaled_location_is_in_return_units() -> None:
    """Heteroskedastic fit (y/vol) must convert LOO loc through vol at test time.

    Scaling only the residual half-width leaves the band centered at a
    dimensionless LOO mean while y is a daily return — coverage collapses.
    """
    rng = np.random.default_rng(21)
    vol = rng.uniform(0.01, 0.04, 500)
    y = rng.normal(0.15, 1.0, 500) * vol
    jp = JackknifePlus(ALPHA).fit(y[:200] / vol[:200])
    lo, hi = jp.predict_interval(np.zeros(300), vol[200:])
    metrics = set_metrics(y[200:], lo, hi)
    assert metrics.n == 300
    assert metrics.coverage >= COVERAGE_FLOOR


@pytest.mark.synthetic
def test_jackknife_plus_synthetic_panel_meets_floor(tmp_path: Path) -> None:
    from quant_fund.config import load_config
    from quant_fund.pipeline.dataset import build_gold, panel
    from quant_fund.research.benches import bench_jackknife_plus

    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 16
    cfg.data.synthetic_n_days = 160
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    build_gold(cfg)
    blob = bench_jackknife_plus(panel(cfg), cfg)
    assert blob, "jackknife+ bench returned an empty receipt"
    assert blob.get("meets_coverage_floor") is True
    assert float(blob["coverage"]) >= 0.78
    assert "sharpe" not in str(blob).lower()
