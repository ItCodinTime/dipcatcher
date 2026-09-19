"""Wave 3 extreme MATH_SPEC fixtures: Kupiec, Acerbi–Tasche ES, corr-spike, Ville.

All fixtures are research/math correctness only — not live P&L evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quant_fund.metrics.analytics import stress_scenarios
from quant_fund.metrics.evalues import (
    e_process,
    e_process_threshold,
    e_value_bernoulli,
)
from quant_fund.metrics.probability import kupiec_pof
from quant_fund.metrics.risk import gaussian_es, gaussian_var, historical_es, historical_var
from quant_fund.models.jackknife_plus import JackknifePlus, loo_mean_and_scale


def _kupiec_lr_closed_form(n: int, x: int, p: float) -> float:
    """Closed-form Kupiec LR_uc for exact hit counts (MATH_SPEC)."""
    rate = x / n
    return -2.0 * (
        (n - x) * math.log(1.0 - p)
        + x * math.log(p)
        - (n - x) * math.log(1.0 - rate)
        - x * math.log(rate)
    )


def test_kupiec_known_lr_exact_hit_counts() -> None:
    # n=100, x=10, p=0.05 → rate=0.10; known LR from closed form
    n, x, p = 100, 10, 0.05
    hits = np.array([1.0] * x + [0.0] * (n - x))
    rate, lr, pv = kupiec_pof(hits, p)
    expected = _kupiec_lr_closed_form(n, x, p)
    assert rate == pytest.approx(0.1)
    assert lr == pytest.approx(expected, rel=1e-12)
    assert pv == pytest.approx(float(stats.chi2.sf(expected, df=1)), rel=1e-12)


def test_kupiec_rejects_severe_undercount() -> None:
    # Expected 5% hits; only 2/200 → undercount should reject at 5%
    n, x, p = 200, 2, 0.05
    hits = np.array([1.0] * x + [0.0] * (n - x))
    rate, lr, pv = kupiec_pof(hits, p)
    assert rate == pytest.approx(0.01)
    assert lr == pytest.approx(_kupiec_lr_closed_form(n, x, p), rel=1e-12)
    assert pv < 0.01


def test_kupiec_accepts_exact_calibration() -> None:
    # Exactly 10/200 = 5% → LR ≈ 0, do not reject
    n, x, p = 200, 10, 0.05
    hits = np.array([1.0] * x + [0.0] * (n - x))
    rate, lr, pv = kupiec_pof(hits, p)
    assert rate == pytest.approx(0.05)
    assert abs(lr) < 1e-9
    assert pv > 0.99


def test_kupiec_boundary_all_or_none_hits_nan() -> None:
    # Implementation returns nan LR when x=0 or x=n (log(0) avoided)
    none_hits = np.zeros(50)
    all_hits = np.ones(50)
    r0, lr0, p0 = kupiec_pof(none_hits, 0.05)
    r1, lr1, p1 = kupiec_pof(all_hits, 0.05)
    assert r0 == 0.0 and math.isnan(lr0) and math.isnan(p0)
    assert r1 == 1.0 and math.isnan(lr1) and math.isnan(p1)


def test_acerbi_tasche_ties_at_var() -> None:
    # All losses equal → ES equals that loss regardless of alpha
    losses = np.full(8, 1.25)
    for alpha in (0.5, 0.75, 0.9, 0.95):
        assert historical_es(losses, alpha) == pytest.approx(1.25)
        assert historical_var(losses, alpha) == pytest.approx(1.25)


def test_acerbi_tasche_tiny_n_and_ordered_losses() -> None:
    # n=1: entire mass is the single observation
    assert historical_es(np.array([5.0]), 0.5) == pytest.approx(5.0)
    assert historical_es(np.array([5.0]), 0.9) == pytest.approx(5.0)
    # n=2, alpha=0.5 → tail_mass=1.0 → mean of largest one
    losses2 = np.array([1.0, 3.0])
    assert historical_es(losses2, 0.5) == pytest.approx(3.0)
    # Ordered distinct, integer tail: n=5, alpha=0.6 → tail_mass=2 → (10+8)/2
    ordered = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert historical_es(ordered, 0.6) == pytest.approx(9.0)
    # Fractional boundary on ordered: alpha=0.7 → tail_mass=1.5 → (10 + 0.5*8)/1.5
    assert historical_es(ordered, 0.7) == pytest.approx((10.0 + 0.5 * 8.0) / 1.5)


def test_acerbi_tasche_es_ge_var_with_ties_on_boundary() -> None:
    # Ties sitting on VaR: losses with plateau; ES must still be >= VaR
    losses = np.array([0.1, 0.2, 0.5, 0.5, 0.5, 0.8, 0.9, 1.0])
    for alpha in (0.75, 0.875, 0.9):
        var = historical_var(losses, alpha)
        es = historical_es(losses, alpha)
        assert es >= var - 1e-12
        assert np.isfinite(es)


def test_corr_spike_stress_more_adverse_when_corr_high() -> None:
    # Long-only equal book: high corr spike ⇒ larger 1σ adverse P&L vs low corr
    w = np.array([0.25, 0.25, 0.25, 0.25])
    vols = np.full(4, 0.02)
    # Diagonal cov → baseline vols; stress rebuilds corr matrix
    cov = np.diag(vols**2)
    high = stress_scenarios(w, cov=cov, corr_spike=0.95, asset_vols=vols)
    low = stress_scenarios(w, cov=cov, corr_spike=0.0, asset_vols=vols)
    assert np.isfinite(high["corr_spike_1sigma_pnl"])
    assert np.isfinite(low["corr_spike_1sigma_pnl"])
    # More negative = more adverse under spiked correlation
    assert float(high["corr_spike_1sigma_pnl"]) < float(low["corr_spike_1sigma_pnl"])
    # Without cov, corr pnl stays nan
    bare = stress_scenarios(w, asset_vols=vols)
    assert math.isnan(float(bare["corr_spike_1sigma_pnl"]))


def test_ville_threshold_exact_boundary() -> None:
    # Craft a path that lands exactly on threshold 20
    path = np.array([1.0, 5.0, 19.999, 20.0, 25.0])
    out = e_process_threshold(path, level=0.05)
    assert out["threshold"] == 20.0
    assert out["reject"] is True
    assert out["first_cross"] == 3
    # Just below threshold: no reject
    below = e_process_threshold(np.array([1.0, 10.0, 19.999]), level=0.05)
    assert below["reject"] is False
    assert below["first_cross"] is None


def test_evalue_alt_lambda_edge_when_2alpha_ge_1() -> None:
    # alpha=0.6 ⇒ 2α≥1 ⇒ midpoint alt; one-step still unit-mean under null
    alpha = 0.6
    e_miss = e_value_bernoulli(1.0, alpha)
    e_hit = e_value_bernoulli(0.0, alpha)
    assert e_miss > 1.0
    assert 0.0 < e_hit < 1.0
    assert np.isclose(alpha * e_miss + (1.0 - alpha) * e_hit, 1.0)


def test_e_process_empty_and_single_miss() -> None:
    empty = e_process(np.asarray([], dtype=float), 0.10)
    assert empty.shape == (1,) and empty[0] == 1.0
    one = e_process(np.array([1.0]), 0.10)
    assert one.shape == (1,)
    assert one[0] == pytest.approx(e_value_bernoulli(1.0, 0.10))


def test_jackknife_loo_mean_scale_closed_form() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    loc, scale = loo_mean_and_scale(y)
    # Manual LOO means
    assert loc[0] == pytest.approx((2 + 3 + 4) / 3)
    assert loc[3] == pytest.approx((1 + 2 + 3) / 3)
    assert np.all(scale > 0)
    # n=2 → scale undefined → zeros
    loc2, scale2 = loo_mean_and_scale(np.array([1.0, 3.0]))
    assert loc2.tolist() == pytest.approx([3.0, 1.0])
    assert np.allclose(scale2, 0.0)


def test_jackknife_plus_n2_boundary_produces_finite_or_raises() -> None:
    # n=2 is minimal for fit; intervals may be degenerate but must not crash
    jp = JackknifePlus(0.10).fit(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    lo, hi = jp.predict_interval(np.array([0.5]), np.array([1.0]))
    assert lo.shape == (1,) and hi.shape == (1,)
    assert np.isfinite(lo).all() or np.isnan(lo).all()


def test_gaussian_tail_metrics_filter_invalid_samples() -> None:
    losses = np.array([0.01, np.nan, 0.02, np.inf])
    assert np.isfinite(gaussian_var(losses))
    assert np.isfinite(gaussian_es(losses))
    assert np.isnan(gaussian_var(np.array([0.01])))
    with pytest.raises(ValueError):
        gaussian_es(losses, alpha=1.0)
