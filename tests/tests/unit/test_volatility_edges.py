"""Wave 14: EWMA/GARCH/HAR fail-closed + ewma_variance closed-form edges."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.volatility import EWMAVol, GARCHVol, HARVol, RollingVol, ewma_variance


def test_ewma_variance_closed_form_two_steps() -> None:
    # var[0] = r0^2; var[1] = lam*var0 + (1-lam)*r0^2 = r0^2 (same)
    # var[2] = lam*var1 + (1-lam)*r1^2
    r = np.array([0.1, -0.2, 0.05])
    lam = 0.94
    v = ewma_variance(r, lam=lam)
    assert v[0] == pytest.approx(0.01)
    assert v[1] == pytest.approx(lam * 0.01 + (1 - lam) * 0.01)
    assert v[2] == pytest.approx(lam * v[1] + (1 - lam) * 0.04)


def test_ewma_variance_invalid_lambda_fail_closed() -> None:
    r = np.ones(5) * 0.01
    with pytest.raises(ValueError, match="lam"):
        ewma_variance(r, lam=1.1)
    with pytest.raises(ValueError, match="lam"):
        ewma_variance(r, lam=-0.01)
    with pytest.raises(ValueError, match="lam"):
        ewma_variance(r, lam=float("nan"))


def test_ewma_variance_empty_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ewma_variance(np.array([]), lam=0.94)


def test_ewmavol_init_invalid_lambda() -> None:
    with pytest.raises(ValueError, match="lam"):
        EWMAVol(lam=1.5)
    with pytest.raises(ValueError, match="lam"):
        EWMAVol(lam=-0.1)


def test_ewmavol_predict_from_returns_matches_sqrt_variance() -> None:
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.01])
    lam = 0.9
    vol = EWMAVol(lam=lam).predict_from_returns(r)
    assert np.allclose(vol, np.sqrt(ewma_variance(r, lam)))


def test_garch_short_series_fallback_no_arch_fit() -> None:
    # size < 50 → skip arch_model; last_sigma from sample std
    y = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005])
    g = GARCHVol().fit(np.zeros((y.size, 1)), y, returns=y)
    assert g.result is None
    assert g.last_sigma == pytest.approx(float(np.std(y * 100.0, ddof=1) / 100.0))
    pred = g.predict(np.zeros((3, 1)))
    assert pred.shape == (3,)
    assert np.allclose(pred, g.last_sigma)


def test_garch_empty_finite_uses_default_sigma() -> None:
    empty = np.array([])
    g = GARCHVol().fit(np.zeros((0, 1)), empty, returns=empty)
    assert g.result is None
    assert g.last_sigma == pytest.approx(0.01)
    assert np.allclose(g.predict(np.zeros((2, 1))), 0.01)


def test_har_unfitted_predict_nan() -> None:
    h = HARVol()
    assert h.fitted is False
    out = h.predict(np.ones((5, 4)))
    assert out.shape == (5,)
    assert np.all(np.isnan(out))


def test_har_short_series_stays_unfitted() -> None:
    # mask.sum() < 10 → no fit
    rv = np.arange(1.0, 8.0)
    h = HARVol().fit(np.zeros((rv.size, 1)), rv)
    assert h.fitted is False


def test_har_design_past_only_row0_nan_lags() -> None:
    rv = np.arange(1.0, 30.0)
    design = HARVol.har_design(rv)
    assert np.isnan(design[0, 1])  # lagged rv
    # first usable daily lag at t=1
    assert design[1, 1] == rv[0]
    # 5-day trailing mean needs window ending at t-1 with 5 lags → t>=5
    assert np.isnan(design[4, 2])
    assert np.isfinite(design[5, 2])


def test_rolling_vol_short_window_prefix_nan() -> None:
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    out = RollingVol(window=3).predict_from_returns(r)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.isfinite(out[2:]).all()
    assert out[2] == pytest.approx(float(np.std(r[0:3], ddof=1)))
