"""Tests for models/nig_vg.py — NIG and Variance-Gamma distributions."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.nig_vg import (
    nig_fit_moments,
    nig_pdf,
    nig_rvs,
    vg_fit_moments,
    vg_pdf,
    vg_rvs,
)


def test_nig_pdf_integrates_to_one() -> None:
    x = np.linspace(-30, 30, 8001)
    area = float(np.trapezoid(nig_pdf(x, alpha=2.0, beta=-0.5, delta=1.0, mu=0.2), x))
    assert abs(area - 1.0) < 1e-3


def test_nig_symmetric_when_beta_zero() -> None:
    x = np.linspace(-6, 6, 121)
    pdf = nig_pdf(x, alpha=1.5, beta=0.0, delta=1.0, mu=0.0)
    assert np.allclose(pdf, pdf[::-1], atol=1e-10)


def test_nig_moment_inversion_is_consistent() -> None:
    rng = np.random.default_rng(0)
    data = nig_rvs(alpha=2.0, beta=-0.6, delta=1.2, mu=0.3, size=200000, rng=rng)
    out = nig_fit_moments(data)
    a, b, d, mu = out["alpha"], out["beta"], out["delta"], out["mu"]
    g = np.sqrt(a**2 - b**2)
    mean_th = mu + d * b / g
    var_th = d * a**2 / g**3
    skew_th = 3.0 * b / (a * np.sqrt(d * g))
    exk_th = 3.0 * (1.0 + 4.0 * (b / a) ** 2) / (d * g)
    m = float(data.mean())
    v = float(data.var(ddof=1))
    z = (data - m) / np.sqrt(v)
    assert abs(mean_th - m) < 1e-6
    assert abs(var_th - v) < 1e-6
    assert abs(skew_th - float((z**3).mean())) < 1e-6
    assert abs(exk_th - (float((z**4).mean()) - 3.0)) < 1e-6


def test_vg_pdf_integrates_to_one() -> None:
    x = np.linspace(-20, 20, 8001)
    area = float(np.trapezoid(vg_pdf(x, sigma=1.0, nu=0.5, theta=0.0, mu=0.0), x))
    assert abs(area - 1.0) < 2e-3


def test_vg_fit_recovers_nu() -> None:
    rng = np.random.default_rng(1)
    data = vg_rvs(sigma=1.0, nu=0.4, theta=0.0, mu=0.0, size=200000, rng=rng)
    out = vg_fit_moments(data)
    assert abs(out["nu"] - 0.4) < 0.15
    assert abs(out["sigma"] - 1.0) < 0.1


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        nig_pdf(np.array([0.0]), alpha=1.0, beta=1.5, delta=1.0)
    with pytest.raises(ValueError):
        vg_pdf(np.array([0.0]), sigma=0.0, nu=1.0)
    with pytest.raises(ValueError):
        vg_rvs(sigma=1.0, nu=-1.0)
