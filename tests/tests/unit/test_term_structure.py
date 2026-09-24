"""Tests for models/term_structure.py — NS/Svensson/Diebold-Li."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.term_structure import (
    diebold_li,
    diebold_li_forecast,
    ns_fit,
    ns_loadings,
    svensson_fit,
)


def test_ns_recovers_curve() -> None:
    m = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    lam_true, beta_true = 0.7, np.array([5.0, -1.2, 0.8])
    y = ns_loadings(m, lam_true) @ beta_true
    fit = ns_fit(m, y)
    assert np.abs(np.asarray(fit["beta"]) - beta_true).max() < 0.05
    assert fit["ssr"] < 1e-8


def test_ns_fixed_lam() -> None:
    m = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    beta = np.array([4.0, -0.8, 0.5])
    y = ns_loadings(m, 0.5) @ beta + 0.01 * np.random.default_rng(0).standard_normal(5)
    fit = ns_fit(m, y, lam=0.5)
    assert np.abs(np.asarray(fit["beta"]) - beta).max() < 0.1


def test_svensson_fits_humped_curve() -> None:
    m = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
    lam_true = 0.5
    beta_true = np.array([4.5, -1.0, 2.0])
    y = ns_loadings(m, lam_true) @ beta_true + 0.05 * np.random.default_rng(1).standard_normal(9)
    fit = svensson_fit(m, y)
    assert np.asarray(fit["fitted"]).shape == (9,)
    assert float(np.abs(np.asarray(fit["residuals"])).max()) < 0.2


def test_diebold_li_factor_dynamics() -> None:
    rng = np.random.default_rng(6)
    n_dates = 200
    m = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    lam = 0.5
    # level factor follows AR(1), others near-constant
    b0 = np.cumsum(0.1 * rng.standard_normal(n_dates)) + 5.0
    b1 = np.full(n_dates, -1.0) + 0.05 * rng.standard_normal(n_dates)
    b2 = np.full(n_dates, 0.5) + 0.05 * rng.standard_normal(n_dates)
    betas = np.column_stack([b0, b1, b2])
    panel = betas @ ns_loadings(m, lam).T
    fit = diebold_li(panel, m, lam=lam)
    est = np.asarray(fit["betas"])
    assert np.abs(est - betas).max() < 1e-8
    ar = np.asarray(fit["ar_coef"])
    assert ar[0, 1] > 0.9  # level factor strongly persistent
    fc = diebold_li_forecast(fit, horizon=3)
    assert fc.shape == (3, m.size)


def test_fail_closed() -> None:
    m = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        ns_loadings(m, 0.5)  # < 3 maturities
    with pytest.raises(ValueError):
        ns_loadings(np.array([1.0, 2.0, 3.0]), -1.0)
    with pytest.raises(ValueError):
        ns_fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError):
        diebold_li(np.random.default_rng(0).standard_normal((5, 8)), np.arange(1.0, 9.0))
    with pytest.raises(ValueError):
        diebold_li_forecast(
            {"betas": np.zeros((10, 3)), "ar_coef": np.zeros((3, 2)), "loadings": np.ones((5, 3))},
            horizon=0,
        )
