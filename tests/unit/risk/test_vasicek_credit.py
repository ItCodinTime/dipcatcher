"""Tests for models/vasicek_credit.py — Vasicek ASRF loss distribution."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.vasicek_credit import (
    vasicek_es,
    vasicek_loss_cdf,
    vasicek_loss_pdf,
    vasicek_var,
)


def test_cdf_monotone_unit_interval() -> None:
    x = np.linspace(1e-4, 0.5, 200)
    cdf = vasicek_loss_cdf(x, pd=0.02, rho=0.15)
    assert np.all(np.diff(cdf) >= -1e-9)
    assert 0.0 <= cdf[0] <= cdf[-1] <= 1.0


def test_mean_loss_equals_pd() -> None:
    pd, rho = 0.03, 0.2
    x = np.linspace(1e-5, 1 - 1e-5, 200000)
    pdf = vasicek_loss_pdf(x, pd, rho)
    mean = float(np.trapezoid(x * pdf, x))
    assert abs(mean - pd) < 2e-3


def test_var_increasing_in_quantile_and_rho() -> None:
    assert vasicek_var(0.99, 0.02, 0.15) > vasicek_var(0.90, 0.02, 0.15)
    assert vasicek_var(0.999, 0.02, 0.30) > vasicek_var(0.999, 0.02, 0.10)
    assert vasicek_es(0.99, 0.02, 0.15) >= vasicek_var(0.99, 0.02, 0.15) - 1e-9


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        vasicek_loss_cdf(np.array([0.1]), pd=1.5, rho=0.1)
    with pytest.raises(ValueError):
        vasicek_var(1.5, 0.02, 0.1)
    with pytest.raises(ValueError):
        vasicek_loss_pdf(np.array([0.1]), pd=0.02, rho=0.0)
