"""Tests for models/edgeworth.py — Gram-Charlier / Edgeworth density."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.edgeworth import (
    gram_charlier_cdf,
    gram_charlier_fit,
    gram_charlier_pdf,
    gram_charlier_valid,
)


def test_reduces_to_normal() -> None:
    x = np.linspace(-4, 4, 101)
    assert np.allclose(gram_charlier_pdf(x, 0.0, 1.0, 0.0, 0.0), norm.pdf(x))
    assert np.allclose(gram_charlier_cdf(x, 0.0, 1.0, 0.0, 0.0), norm.cdf(x))


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-15, 15, 6001)
    area = float(np.trapezoid(gram_charlier_pdf(x, 0.0, 2.0, 0.3, 0.8), x))
    assert abs(area - 1.0) < 1e-3


def test_validity_flag() -> None:
    assert gram_charlier_valid(0.0, 0.0)
    assert not gram_charlier_valid(3.0, 8.0)  # extreme -> negative lobes


def test_fit_matches_moments() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20000) * 1.5 + 2.0
    out = gram_charlier_fit(data)
    assert abs(out["mu"] - 2.0) < 0.05
    assert abs(out["sigma"] - 1.5) < 0.05
    assert abs(out["skew"]) < 0.1
    assert abs(out["exkurt"]) < 0.15


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        gram_charlier_pdf(np.array([0.0]), sigma=0.0)
    with pytest.raises(ValueError):
        gram_charlier_fit(np.arange(3.0))
