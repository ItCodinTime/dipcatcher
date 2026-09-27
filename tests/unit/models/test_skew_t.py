"""Tests for models/skew_t.py — Hansen (1994) skewed Student-t."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.skew_t import skew_t_cdf, skew_t_fit, skew_t_pdf, skew_t_ppf


def test_symmetric_when_lambda_zero() -> None:
    x = np.linspace(-5, 5, 101)
    pdf = skew_t_pdf(x, nu=6.0, lam=0.0)
    assert np.allclose(pdf, pdf[::-1], atol=1e-9)  # symmetric about 0
    assert abs(float(skew_t_cdf(np.array([0.0]), 6.0, 0.0)[0]) - 0.5) < 1e-6


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-40, 40, 8001)
    area = float(np.trapezoid(skew_t_pdf(x, nu=5.0, lam=-0.4), x))
    assert abs(area - 1.0) < 1e-3


def test_ppf_inverts_cdf() -> None:
    for p in (0.05, 0.3, 0.5, 0.8, 0.97):
        x = skew_t_ppf(p, nu=7.0, lam=0.3, mu=0.1, sigma=1.2)
        assert abs(float(skew_t_cdf(np.array([x]), 7.0, 0.3, 0.1, 1.2)[0]) - p) < 1e-6


def test_fit_recovers_negative_skew() -> None:
    rng = np.random.default_rng(0)
    u = rng.random(1500)
    x = np.array([skew_t_ppf(p, nu=6.0, lam=-0.3) for p in u])
    out = skew_t_fit(x)
    assert out["lam"] < 0.0
    assert 3.0 < out["nu"] < 30.0
    assert abs(out["mu"]) < 0.3


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        skew_t_pdf(np.array([0.0]), nu=1.5, lam=0.0)
    with pytest.raises(ValueError):
        skew_t_pdf(np.array([0.0]), nu=6.0, lam=0.0, sigma=0.0)
    with pytest.raises(ValueError):
        skew_t_ppf(1.5, nu=6.0, lam=0.0)
