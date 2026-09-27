"""Tests for models/gas.py — Creal-Koopman-Lucas score-driven vol."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.gas import gas_vol_fit, gas_vol_forecast


def _garch_returns(n: int = 1500, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    r = np.empty(n)
    sig2[0] = 0.5
    r[0] = np.sqrt(0.5) * rng.standard_normal()
    for t in range(1, n):
        sig2[t] = 0.03 + 0.12 * r[t - 1] ** 2 + 0.85 * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    return r


def test_gas_gauss_fits_garch_data() -> None:
    r = _garch_returns()
    out = gas_vol_fit(r, dist="gauss")
    assert out["converged"] == 1.0 or np.isfinite(out["loglik"])
    f = np.asarray(out["f"])
    assert (f > 0).all()
    # fitted vol should correlate with rolling realized vol
    win = 20
    roll = np.array([np.std(r[i - win : i]) for i in range(win, r.size)])
    corr = np.corrcoef(np.sqrt(f[win:]), roll)[0, 1]
    assert corr > 0.5


def test_gas_t_estimates_df() -> None:
    rng = np.random.default_rng(3)
    nu_true = 7.0
    n = 1500
    z = rng.standard_t(nu_true, size=n) / np.sqrt(nu_true / (nu_true - 2.0))
    # mild vol clustering on top
    sig = 1.0 + 0.3 * np.abs(np.convolve(z, np.ones(10) / 10, mode="same"))
    y = z * sig
    out = gas_vol_fit(y, dist="t")
    assert 4.05 <= out["nu"] <= 100.0
    assert np.isfinite(np.asarray(out["f"])).all()


def test_forecast_one_step() -> None:
    r = _garch_returns(seed=5)
    out = gas_vol_fit(r)
    f_next = gas_vol_forecast(out, r[-1])
    assert f_next > 0.0
    assert np.isfinite(f_next)


def test_unit_scaling_runs() -> None:
    r = _garch_returns(seed=9)
    out = gas_vol_fit(r, dist="gauss", scaling="unit")
    assert np.isfinite(np.asarray(out["f"])).all()
    assert (np.asarray(out["f"]) > 0).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gas_vol_fit(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        gas_vol_fit(np.full(200, np.nan))
    with pytest.raises(ValueError):
        gas_vol_fit(np.random.default_rng(0).standard_normal(200), dist="bogus")
    with pytest.raises(ValueError):
        gas_vol_fit(np.random.default_rng(0).standard_normal(200), scaling="bogus")
