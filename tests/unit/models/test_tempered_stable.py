"""Tests for models/tempered_stable.py — CGMY process."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.tempered_stable import cgmy_cumulants, cgmy_pdf


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-3, 3, 601)
    pdf = cgmy_pdf(x, c=0.5, g=5.0, m=5.0, y=0.5)
    area = float(np.trapezoid(pdf, x))
    assert abs(area - 1.0) < 5e-3
    assert (pdf >= 0).all()


def test_cumulants_match_density_moments() -> None:
    c, g, m, y = 0.6, 6.0, 4.0, 0.5
    cum = cgmy_cumulants(c, g, m, y)
    x = np.linspace(-4, 4, 1201)
    pdf = cgmy_pdf(x, c, g, m, y)
    pdf = pdf / np.trapezoid(pdf, x)
    mean = float(np.trapezoid(x * pdf, x))
    var = float(np.trapezoid((x - mean) ** 2 * pdf, x))
    assert abs(mean - cum["mean"]) < 0.02
    assert abs(var - cum["var"]) < 0.02


def test_skew_sign_follows_asymmetry() -> None:
    # G < M tempers the left tail less -> negative skew; check sign convention.
    left = cgmy_cumulants(0.5, g=3.0, m=8.0, y=0.5)["skew"]
    right = cgmy_cumulants(0.5, g=8.0, m=3.0, y=0.5)["skew"]
    assert left < 0 < right


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        cgmy_cumulants(-1.0, 5.0, 5.0, 0.5)
    with pytest.raises(ValueError):
        cgmy_pdf(np.array([0.0]), 0.5, 5.0, 5.0, 1.0)  # Y == 1 disallowed
