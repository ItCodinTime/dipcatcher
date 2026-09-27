"""Tests for models/theta.py — the classic Theta method."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.theta import theta_fit, theta_forecast


def test_theta_fits_and_forecasts_trend() -> None:
    t = np.arange(100, dtype=float)
    rng = np.random.default_rng(0)
    y = 3.0 + 0.4 * t + 0.5 * rng.standard_normal(t.size)
    fit = theta_fit(y)
    assert 0.0 < fit.alpha < 1.0
    assert fit.slope > 0.3
    fc = theta_forecast(fit, 12)
    assert fc.shape == (12,)
    assert np.all(np.diff(fc) > 0)  # positive drift on an up-trend
    assert np.isfinite(fc).all()


def test_theta_in_sample_fit_reasonable() -> None:
    t = np.arange(150, dtype=float)
    rng = np.random.default_rng(1)
    y = 10.0 + 0.2 * t + 2.0 * np.sin(t / 6.0) + 0.3 * rng.standard_normal(t.size)
    fit = theta_fit(y)
    ss_res = float(fit.resid @ fit.resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert 1.0 - ss_res / ss_tot > 0.5


def test_theta_equals_half_drift_property() -> None:
    # Hyndman & Billah (2003): theta forecast drift is ~ half the OLS slope.
    t = np.arange(80, dtype=float)
    y = 1.0 + 1.0 * t  # exact linear, slope 1
    fit = theta_fit(y)
    fc = theta_forecast(fit, 20)
    drift = float(np.mean(np.diff(fc)))
    assert abs(drift - 0.5) < 0.1


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        theta_fit(np.arange(3.0))
    with pytest.raises(ValueError):
        theta_fit(np.arange(20.0), theta=1.0)
    with pytest.raises(ValueError):
        theta_forecast(theta_fit(np.arange(20.0) + 1.0), 0)
