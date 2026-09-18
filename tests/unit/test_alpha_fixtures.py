"""Wave 14: rolling_beta / residualize / HistoricalMeanAlpha closed-form + boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.alpha import HistoricalMeanAlpha, residualize, rolling_beta


def test_historical_mean_alpha_closed_form() -> None:
    y = np.array([0.01, 0.02, -0.01, 0.04])
    x = np.zeros((4, 2))
    m = HistoricalMeanAlpha().fit(x, y)
    assert m.mean_ == pytest.approx(float(np.mean(y)))
    pred = m.predict(np.zeros((3, 2)))
    assert pred.shape == (3,)
    assert np.allclose(pred, m.mean_)


def test_historical_mean_alpha_empty_or_all_nan_is_zero() -> None:
    m = HistoricalMeanAlpha().fit(np.zeros((0, 1)), np.array([]))
    assert m.mean_ == 0.0
    assert np.allclose(m.predict(np.zeros((2, 1))), 0.0)
    m2 = HistoricalMeanAlpha().fit(np.zeros((3, 1)), np.array([np.nan, np.nan, np.nan]))
    assert m2.mean_ == 0.0


def test_historical_mean_alpha_ignores_nan_in_y() -> None:
    y = np.array([0.1, np.nan, 0.3])
    m = HistoricalMeanAlpha().fit(np.zeros((3, 1)), y)
    assert m.mean_ == pytest.approx(0.2)


def test_rolling_beta_known_multiple() -> None:
    # asset = 2 * mkt → beta ≈ 2 after window fills
    mkt = np.array([0.01, -0.02, 0.015, 0.0, 0.01, -0.005, 0.02, 0.01])
    asset = 2.0 * mkt
    out = rolling_beta(asset, mkt, window=4)
    assert np.all(np.isnan(out[:3]))  # first window-1 are nan
    assert np.isfinite(out[3:]).all()
    assert np.allclose(out[3:], 2.0, atol=1e-10)


def test_rolling_beta_constant_market_stays_nan() -> None:
    asset = np.linspace(-0.01, 0.01, 10)
    mkt = np.zeros(10)
    out = rolling_beta(asset, mkt, window=5)
    assert np.all(np.isnan(out))


def test_rolling_beta_short_series_all_nan() -> None:
    asset = np.array([0.01, 0.02])
    mkt = np.array([0.01, -0.01])
    out = rolling_beta(asset, mkt, window=5)
    assert out.shape == (2,)
    assert np.all(np.isnan(out))


def test_residualize_closed_form_single_factor() -> None:
    # y = 0.5 + 1.5 * f + noise0
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 0.5 + 1.5 * f
    resid, betas, intercept = residualize(y, f.reshape(-1, 1))
    assert betas.shape == (1,)
    assert betas[0] == pytest.approx(1.5)
    assert intercept == pytest.approx(0.5)
    assert np.allclose(resid, 0.0, atol=1e-12)


def test_residualize_preserves_nan_rows() -> None:
    y = np.array([1.0, np.nan, 3.0, 4.0])
    f = np.array([[1.0], [2.0], [3.0], [4.0]])
    resid, betas, intercept = residualize(y, f)
    assert np.isnan(resid[1])
    assert np.isfinite(resid[0]) and np.isfinite(resid[2]) and np.isfinite(resid[3])
    # Fit on finite rows only: y=[1,3,4], f=[1,3,4]
    assert betas.shape == (1,)


def test_residualize_two_factor_r2_identity() -> None:
    rng = np.random.default_rng(7)
    f = rng.normal(size=(40, 2))
    true_b = np.array([0.3, -0.7])
    y = 0.1 + f @ true_b
    resid, betas, intercept = residualize(y, f)
    assert np.allclose(betas, true_b, atol=1e-10)
    assert intercept == pytest.approx(0.1)
    assert np.allclose(resid, 0.0, atol=1e-10)


def test_metadata_historical_mean() -> None:
    meta = HistoricalMeanAlpha().metadata()
    assert meta.family == "alpha"
    assert meta.name == "historical_mean"
