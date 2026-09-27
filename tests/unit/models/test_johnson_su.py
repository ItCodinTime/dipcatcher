"""Tests for models/johnson_su.py — Johnson SU + Slifker-Shapiro fit."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.johnson_su import (
    johnson_su_cdf,
    johnson_su_fit,
    johnson_su_pdf,
    johnson_su_ppf,
)


def test_pdf_integrates_to_one() -> None:
    x = np.linspace(-30, 30, 8001)
    area = float(np.trapezoid(johnson_su_pdf(x, gamma=-0.5, delta=2.0, xi=0.0, lam=1.0), x))
    assert abs(area - 1.0) < 1e-3


def test_ppf_inverts_cdf() -> None:
    for p in (0.02, 0.25, 0.5, 0.75, 0.98):
        x = johnson_su_ppf(p, gamma=-0.5, delta=1.5, xi=0.2, lam=1.3)
        assert abs(float(johnson_su_cdf(np.array([x]), -0.5, 1.5, 0.2, 1.3)[0]) - p) < 1e-8


def test_fit_recovers_parameters() -> None:
    rng = np.random.default_rng(0)
    gamma, delta, xi, lam = -0.7, 1.8, 0.5, 1.2
    z = rng.standard_normal(60000)
    x = xi + lam * np.sinh((z - gamma) / delta)
    out = johnson_su_fit(x)
    assert abs(out["gamma"] - gamma) < 0.2
    assert abs(out["delta"] - delta) < 0.2
    assert abs(out["xi"] - xi) < 0.2
    assert abs(out["lam"] - lam) < 0.2


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        johnson_su_pdf(np.array([0.0]), gamma=0.0, delta=-1.0)
    with pytest.raises(ValueError):
        johnson_su_ppf(1.2, gamma=0.0, delta=1.0)
    with pytest.raises(ValueError):
        # normal data does not satisfy the SU shape criterion
        johnson_su_fit(np.random.default_rng(1).standard_normal(5000))
