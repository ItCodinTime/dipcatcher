"""Tests for models/johnson_sb.py — Johnson SB and SL distributions."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.johnson_sb import (
    johnson_sb_cdf,
    johnson_sb_fit,
    johnson_sb_pdf,
    johnson_sb_ppf,
    johnson_sl_cdf,
    johnson_sl_fit,
    johnson_sl_pdf,
    johnson_sl_ppf,
)


def test_sb_pdf_integrates_to_one() -> None:
    xi, lam = 0.0, 1.0
    x = np.linspace(1e-4, 1.0 - 1e-4, 20000)
    area = float(np.trapezoid(johnson_sb_pdf(x, gamma=0.5, delta=1.5, xi=xi, lam=lam), x))
    assert abs(area - 1.0) < 5e-3


def test_sb_ppf_inverts_cdf() -> None:
    for p in (0.05, 0.3, 0.5, 0.8, 0.95):
        x = johnson_sb_ppf(p, gamma=-0.3, delta=1.2, xi=0.0, lam=2.0)
        assert abs(float(johnson_sb_cdf(np.array([x]), -0.3, 1.2, 0.0, 2.0)[0]) - p) < 1e-8


def test_sb_fit_recovers_shape() -> None:
    rng = np.random.default_rng(0)
    gamma, delta, xi, lam = 0.4, 1.3, 0.0, 1.0
    z = rng.standard_normal(30000)
    u = 1.0 / (1.0 + np.exp(-(z - gamma) / delta))
    x = xi + lam * u
    out = johnson_sb_fit(x)
    assert abs(out["gamma"] - gamma) < 0.15
    assert abs(out["delta"] - delta) < 0.15


def test_sl_roundtrip_and_fit() -> None:
    rng = np.random.default_rng(1)
    gamma, delta, xi = 0.0, 1.5, 0.0
    z = rng.standard_normal(30000)
    x = xi + np.exp((z - gamma) / delta)
    out = johnson_sl_fit(x)
    assert abs(out["delta"] - delta) < 0.2
    p = 0.6
    q = johnson_sl_ppf(p, out["gamma"], out["delta"], out["xi"])
    assert (
        abs(float(johnson_sl_cdf(np.array([q]), out["gamma"], out["delta"], out["xi"])[0]) - p)
        < 1e-6
    )
    assert (johnson_sl_pdf(np.array([1.0, 2.0]), 0.0, 1.5, 0.0) > 0).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        johnson_sb_pdf(np.array([0.5]), gamma=0.0, delta=-1.0, xi=0.0, lam=1.0)
    with pytest.raises(ValueError):
        johnson_sb_ppf(1.5, 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        johnson_sl_fit(np.arange(3.0))
