"""Tests for models/skew_normal.py — Azzalini skew-normal."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, skewnorm

from quant_fund.models.skew_normal import (
    skew_normal_cdf,
    skew_normal_fit,
    skew_normal_pdf,
    skew_normal_ppf,
)


def test_alpha_zero_is_normal() -> None:
    x = np.linspace(-4, 4, 101)
    assert np.allclose(skew_normal_pdf(x, 0.0, 1.0, 0.0), norm.pdf(x))
    assert np.allclose(skew_normal_cdf(x, 0.0, 1.0, 0.0), norm.cdf(x), atol=1e-8)


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-12, 12, 4001)
    area = float(np.trapezoid(skew_normal_pdf(x, 0.5, 1.3, 3.0), x))
    assert abs(area - 1.0) < 1e-3


def test_ppf_inverts_cdf() -> None:
    for p in (0.05, 0.3, 0.5, 0.8, 0.97):
        x = skew_normal_ppf(p, xi=0.2, omega=1.1, alpha=2.0)
        assert abs(float(skew_normal_cdf(np.array([x]), 0.2, 1.1, 2.0)[0]) - p) < 1e-6


def test_fit_recovers_positive_skew() -> None:
    data = skewnorm.rvs(a=5.0, loc=0.0, scale=1.0, size=8000, random_state=0)
    out = skew_normal_fit(data)
    assert out["alpha"] > 0.5  # right-skewed
    assert out["omega"] > 0
    # MoM matches the first two moments by construction
    b = np.sqrt(2.0 / np.pi)
    delta = out["alpha"] / np.sqrt(1.0 + out["alpha"] ** 2)
    model_mean = out["xi"] + out["omega"] * b * delta
    assert abs(model_mean - float(np.mean(data))) < 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        skew_normal_pdf(np.array([0.0]), omega=0.0)
    with pytest.raises(ValueError):
        skew_normal_ppf(1.5)
    with pytest.raises(ValueError):
        skew_normal_fit(np.arange(3.0))
