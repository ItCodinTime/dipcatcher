"""Tests for models/ets.py — SES / Holt / Holt-Winters exponential smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ets import ets_forecast, holt_fit, holt_winters_fit, ses_fit


def test_ses_recovers_constant_level() -> None:
    rng = np.random.default_rng(0)
    y = 5.0 + 0.2 * rng.standard_normal(200)
    fit = ses_fit(y)
    assert 0.0 < fit.params["alpha"] < 1.0
    fc = ets_forecast(fit, 5)
    assert fc.shape == (5,)
    assert np.allclose(fc, fc[0])  # flat forecast
    assert abs(fc[0] - 5.0) < 0.3


def test_holt_tracks_linear_trend() -> None:
    t = np.arange(120, dtype=float)
    rng = np.random.default_rng(1)
    y = 2.0 + 0.5 * t + 0.3 * rng.standard_normal(t.size)
    fit = holt_fit(y)
    fc = ets_forecast(fit, 10)
    assert np.all(np.diff(fc) > 0)  # keeps rising
    # one-step-ahead forecast slope is close to the true trend
    assert abs((fc[-1] - fc[0]) / 9.0 - 0.5) < 0.2


def test_damped_trend_flattens() -> None:
    t = np.arange(120, dtype=float)
    y = 1.0 + 0.4 * t
    plain = ets_forecast(holt_fit(y, damped=False), 40)
    damped = ets_forecast(holt_fit(y, damped=True), 40)
    # damped long-horizon forecast should not exceed the linear one
    assert damped[-1] <= plain[-1] + 1e-6


def _seasonal_series(n: int = 240, period: int = 12, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    season = 3.0 * np.sin(2 * np.pi * (t % period) / period)
    return 10.0 + 0.05 * t + season + 0.4 * rng.standard_normal(n)


def test_holt_winters_additive_captures_season() -> None:
    period = 12
    y = _seasonal_series(period=period)
    fit = holt_winters_fit(y, period=period, seasonal="add")
    assert fit.season.shape == (period,)
    fc = ets_forecast(fit, period)
    assert np.isfinite(fc).all()
    # forecast should retain seasonal variation, not collapse to a line
    assert fc.std() > 1.0
    # in-sample fit explains most variance
    ss_res = float(fit.resid[period:] @ fit.resid[period:])
    ss_tot = float(((y[period:] - y[period:].mean()) ** 2).sum())
    assert 1.0 - ss_res / ss_tot > 0.7


def test_holt_winters_multiplicative_runs() -> None:
    period = 12
    y = _seasonal_series(period=period) + 20.0  # keep strictly positive
    fit = holt_winters_fit(y, period=period, seasonal="mul")
    fc = ets_forecast(fit, period)
    assert np.isfinite(fc).all()
    assert (fc > 0).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        ses_fit(np.array([1.0, np.nan, 3.0, 4.0, 5.0]))
    with pytest.raises(ValueError):
        holt_winters_fit(np.arange(10.0), period=1)
    with pytest.raises(ValueError):
        holt_winters_fit(-np.abs(np.arange(60.0)) - 1.0, period=12, seasonal="mul")
    with pytest.raises(ValueError):
        ets_forecast(ses_fit(np.ones(10) + np.arange(10) * 0.01), 0)
