"""Tests for models/tukey_gh.py — Tukey g-and-h distribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.tukey_gh import (
    tukey_gh_cdf,
    tukey_gh_fit,
    tukey_gh_pdf,
    tukey_gh_ppf,
)


def test_gh_zero_is_normal() -> None:
    for p in (0.1, 0.5, 0.9):
        assert abs(tukey_gh_ppf(p, 0.0, 1.0, 0.0, 0.0) - float(norm.ppf(p))) < 1e-9
    x = np.linspace(-3, 3, 51)
    assert np.allclose(tukey_gh_pdf(x, 0.0, 1.0, 0.0, 0.0), norm.pdf(x), atol=1e-6)
    assert np.allclose(tukey_gh_cdf(x, 0.0, 1.0, 0.0, 0.0), norm.cdf(x), atol=1e-6)


def test_ppf_monotone_and_cdf_inverse() -> None:
    ps = np.linspace(0.02, 0.98, 25)
    xs = np.array([tukey_gh_ppf(p, 0.0, 1.0, 0.3, 0.1) for p in ps])
    assert np.all(np.diff(xs) > 0)
    back = tukey_gh_cdf(xs, 0.0, 1.0, 0.3, 0.1)
    assert np.allclose(back, ps, atol=1e-6)


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-40, 40, 12001)
    area = float(np.trapezoid(tukey_gh_pdf(x, 0.0, 1.0, 0.2, 0.1), x))
    assert abs(area - 1.0) < 2e-3


def test_fit_recovers_g_and_h() -> None:
    rng = np.random.default_rng(0)
    a, b, g, h = 0.0, 1.0, 0.3, 0.15
    z = rng.standard_normal(80000)
    tail = np.exp(0.5 * h * z * z)
    x = a + b * (np.exp(g * z) - 1.0) / g * tail
    out = tukey_gh_fit(x)
    assert abs(out["g"] - g) < 0.08
    assert abs(out["h"] - h) < 0.08
    assert abs(out["b"] - b) < 0.15


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        tukey_gh_ppf(0.5, b=0.0)
    with pytest.raises(ValueError):
        tukey_gh_pdf(np.array([0.0]), h=-1.0)
    with pytest.raises(ValueError):
        tukey_gh_fit(np.arange(10.0))
