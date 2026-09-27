"""Tests for models/stable.py — alpha-stable simulation, ECF fit, density."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.stable import stable_cdf, stable_fit_ecf, stable_pdf, stable_rvs


def test_rvs_heavy_tails() -> None:
    rng = np.random.default_rng(0)
    heavy = stable_rvs(alpha=1.2, beta=0.0, size=20000, rng=rng)
    gauss = rng.standard_normal(20000)
    # stable(alpha<2) has far more extreme observations than the Gaussian
    assert np.max(np.abs(heavy)) > 10.0 * np.max(np.abs(gauss))
    assert heavy.size == 20000
    assert np.isfinite(heavy).all()


def test_ecf_fit_recovers_alpha() -> None:
    rng = np.random.default_rng(1)
    x = stable_rvs(alpha=1.5, beta=0.0, c=1.0, size=60000, rng=rng)
    out = stable_fit_ecf(x)
    assert abs(out["alpha"] - 1.5) < 0.25
    assert out["c"] > 0


def test_ecf_fit_orders_by_tail_thickness() -> None:
    rng = np.random.default_rng(2)
    thin = stable_fit_ecf(stable_rvs(alpha=1.9, beta=0.0, size=60000, rng=rng))["alpha"]
    thick = stable_fit_ecf(stable_rvs(alpha=1.2, beta=0.0, size=60000, rng=rng))["alpha"]
    assert thin > thick


def test_pdf_cdf_wellformed() -> None:
    x = np.linspace(-6, 6, 25)
    pdf = stable_pdf(x, alpha=1.7, beta=0.0)
    cdf = stable_cdf(x, alpha=1.7, beta=0.0)
    assert (pdf >= 0).all() and np.isfinite(pdf).all()
    assert np.all(np.diff(cdf) >= -1e-9)  # monotone non-decreasing
    assert cdf[0] < 0.1 and cdf[-1] > 0.9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        stable_rvs(alpha=2.5)
    with pytest.raises(ValueError):
        stable_pdf(np.array([0.0]), alpha=1.5, beta=2.0)
    with pytest.raises(ValueError):
        stable_fit_ecf(np.arange(10.0))
