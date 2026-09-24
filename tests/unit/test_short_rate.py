"""Tests for models/short_rate.py — Vasicek and CIR short-rate models."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.short_rate import (
    cir_bond_price,
    cir_calibrate,
    cir_simulate,
    vasicek_bond_price,
    vasicek_calibrate,
    vasicek_simulate,
)


def test_vasicek_bond_price_bounds() -> None:
    p1 = vasicek_bond_price(0.03, 1.0, kappa=0.5, theta=0.04, sigma=0.01)
    p5 = vasicek_bond_price(0.03, 5.0, kappa=0.5, theta=0.04, sigma=0.01)
    assert 0.0 < p5 < p1 < 1.0  # longer maturity -> lower price
    assert abs(vasicek_bond_price(0.03, 0.0, 0.5, 0.04, 0.01) - 1.0) < 1e-12


def test_vasicek_calibration_recovers_params() -> None:
    rng = np.random.default_rng(0)
    dt = 1.0 / 252.0
    r = vasicek_simulate(0.03, kappa=1.2, theta=0.05, sigma=0.02, n=60000, dt=dt, rng=rng)
    out = vasicek_calibrate(r, dt=dt)
    assert abs(out["theta"] - 0.05) < 0.01
    assert abs(out["kappa"] - 1.2) < 0.4
    assert abs(out["sigma"] - 0.02) < 0.004


def test_cir_bond_price_bounds_and_nonneg_paths() -> None:
    p = cir_bond_price(0.03, 3.0, kappa=0.6, theta=0.04, sigma=0.05)
    assert 0.0 < p < 1.0
    rng = np.random.default_rng(1)
    path = cir_simulate(0.03, kappa=0.6, theta=0.04, sigma=0.05, n=5000, rng=rng)
    assert (path >= 0).all()


def test_cir_calibration_runs() -> None:
    rng = np.random.default_rng(2)
    dt = 1.0 / 252.0
    r = cir_simulate(0.04, kappa=1.0, theta=0.05, sigma=0.06, n=60000, dt=dt, rng=rng)
    out = cir_calibrate(r, dt=dt)
    assert abs(out["theta"] - 0.05) < 0.02
    assert out["kappa"] > 0 and out["sigma"] > 0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        vasicek_bond_price(0.03, 1.0, kappa=0.0, theta=0.04, sigma=0.01)
    with pytest.raises(ValueError):
        cir_bond_price(0.03, 1.0, kappa=0.5, theta=-0.01, sigma=0.05)
    with pytest.raises(ValueError):
        vasicek_calibrate(np.ones(10))
