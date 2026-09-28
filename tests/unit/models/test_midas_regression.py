"""Canon tests: MIDAS regression."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.midas_regression import (
    almon_weights,
    midas_fit,
    midas_forecast,
)


def _midas_panel(t: int = 200, k: int = 8, seed: int = 4):
    rng = np.random.default_rng(seed)
    # true weights decay smoothly (recent lags matter more)
    true_w = np.exp(-0.3 * np.arange(k) / (k - 1))
    true_w /= true_w.sum()
    x = rng.normal(0, 1, (t, k))
    y = 0.5 + 2.0 * x @ true_w + rng.normal(0, 0.1, t)
    return y, x, true_w


def test_almon_weights_normalized() -> None:
    w = almon_weights(10, np.array([0.0, -2.0]))
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()
    # negative theta_2 -> humped/decaying shape, not uniform
    assert not np.allclose(w, 0.1)


def test_almon_weights_monotone() -> None:
    w = almon_weights(10, np.array([2.0, 0.0]))
    assert np.all(np.diff(w) > 0)  # increasing weights for theta_1 > 0


def test_exp_almon_recovers_weights() -> None:
    y, x, true_w = _midas_panel()
    out = midas_fit(y, x, scheme="exp_almon")
    assert out["r2"] > 0.9
    # fitted kernel shape should track the true decay
    corr = np.corrcoef(out["weights"], true_w)[0, 1]
    assert corr > 0.7
    assert out["beta"] == pytest.approx(2.0, abs=0.6)


def test_umidas_ols_path() -> None:
    y, x, _ = _midas_panel()
    out = midas_fit(y, x, scheme="umidas")
    assert out["r2"] > 0.9
    assert out["lag_coefs"].shape == (8,)
    assert abs(out["intercept"] - 0.5) < 0.3


def test_midas_forecast() -> None:
    y, x, _ = _midas_panel()
    fit_a = midas_fit(y, x, scheme="exp_almon")
    fit_u = midas_fit(y, x, scheme="umidas")
    pa = midas_forecast(fit_a, x[:5])
    pu = midas_forecast(fit_u, x[:5])
    assert pa.shape == (5,) and pu.shape == (5,)
    np.testing.assert_allclose(pa, fit_a["fitted"][:5], atol=1e-8)
    np.testing.assert_allclose(pu, fit_u["fitted"][:5], atol=1e-8)


def test_midas_forecast_1d_input() -> None:
    y, x, _ = _midas_panel()
    fit = midas_fit(y, x, scheme="exp_almon")
    out = midas_forecast(fit, x[0])
    assert out.shape == (1,)


def test_midas_validation() -> None:
    y, x, _ = _midas_panel()
    with pytest.raises(ValueError):
        midas_fit(y[:5], x[:5], scheme="exp_almon")  # T < K+3
    with pytest.raises(ValueError):
        midas_fit(y, x[:, :1], scheme="exp_almon")  # K < 2
    with pytest.raises(ValueError):
        midas_fit(np.full(200, np.nan), x)
    with pytest.raises(ValueError):
        midas_fit(y, x, scheme="bogus")
    with pytest.raises(ValueError):
        almon_weights(1, np.zeros(2))
    with pytest.raises(ValueError):
        almon_weights(5, np.zeros(3))
    fit = midas_fit(y, x, scheme="umidas")
    with pytest.raises(ValueError):
        midas_forecast(fit, x[:, :4])
    with pytest.raises(ValueError):
        midas_forecast(fit, np.full((3, 8), np.nan))
