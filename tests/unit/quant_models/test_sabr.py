"""Tests for models/sabr.py — Hagan (2002) SABR implied vol + calibration."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sabr import (
    sabr_alpha_from_atm,
    sabr_atm_vol,
    sabr_fit,
    sabr_implied_vol,
    sabr_rho_nu_from_smile,
)


def test_atm_limit_matches_atm_vol() -> None:
    f, t = 100.0, 1.0
    a, b, r, n = 0.3, 0.5, -0.3, 0.4
    atm_func = sabr_atm_vol(f, t, a, b, r, n)
    atm_series = float(sabr_implied_vol(f, np.array([f]), t, a, b, r, n)[0])
    assert atm_func == pytest.approx(atm_series, rel=1e-10)
    assert atm_func > 0.0


def test_negative_rho_downward_skew() -> None:
    f, t = 1.0, 0.5
    k = np.linspace(0.7, 1.4, 15)
    vols = sabr_implied_vol(f, k, t, 0.25, 0.5, -0.5, 0.8)
    # negative rho -> low strikes have higher vol than high strikes
    assert vols[0] > vols[-1]


def test_alpha_recovers_atm() -> None:
    f, t = 80.0, 2.0
    a_true, b, r, n = 0.4, 0.7, 0.1, 0.6
    atm = sabr_atm_vol(f, t, a_true, b, r, n)
    a_hat = sabr_alpha_from_atm(f, t, atm, b, r, n)
    assert a_hat == pytest.approx(a_true, rel=1e-6)


def test_fit_recovers_smile() -> None:
    rng = np.random.default_rng(7)
    f, t = 100.0, 1.0
    a, b, r, n = 0.35, 0.5, -0.4, 0.9
    k = np.linspace(80, 120, 11)
    v = sabr_implied_vol(f, k, t, a, b, r, n)
    v_noisy = v * (1.0 + 0.002 * rng.standard_normal(k.size))
    fit = sabr_fit(f, t, k, v_noisy, beta=b)
    assert fit["converged"] == 1.0
    assert fit["rmse"] < 1e-3
    assert fit["rho"] == pytest.approx(r, abs=0.15)


def test_two_step_rho_nu() -> None:
    f, t = 100.0, 1.0
    k = np.linspace(85, 115, 9)
    v = sabr_implied_vol(f, k, t, 0.3, 0.5, -0.35, 0.7)
    out = sabr_rho_nu_from_smile(f, t, k, v, beta=0.5)
    assert out["rmse"] < 1e-6
    assert out["rho"] == pytest.approx(-0.35, abs=0.05)


def test_shifted_sabr() -> None:
    vols = sabr_implied_vol(
        -0.005, np.array([-0.01, 0.0, 0.01]), 1.0, 0.2, 0.5, 0.0, 0.5, shift=0.02
    )
    assert np.isfinite(vols).all()
    assert (vols > 0).all()


def test_fail_closed() -> None:
    k = np.array([90.0, 100.0, 110.0])
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, k, 1.0, -0.1, 0.5, 0.0, 0.5)
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, k, 1.0, 0.3, 0.5, 1.0, 0.5)
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, k, -1.0, 0.3, 0.5, 0.0, 0.5)
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, np.array([-5.0]), 1.0, 0.3, 0.5, 0.0, 0.5)
    with pytest.raises(ValueError):
        sabr_alpha_from_atm(100.0, 1.0, -0.2, 0.5, 0.0, 0.5)
    with pytest.raises(ValueError):
        sabr_fit(100.0, 1.0, k[:2], np.array([0.2, 0.2]))


def test_vol_positive_across_smile() -> None:
    f, t = 50.0, 0.25
    k = np.linspace(30, 80, 25)
    v = sabr_implied_vol(f, k, t, 0.4, 0.0, 0.2, 1.2)
    assert (v > 0.0).all()
    assert np.isfinite(v).all()
