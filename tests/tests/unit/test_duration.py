"""Tests for models/duration.py — Engle-Russell ACD."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.duration import acd_diagnostics, acd_fit, acd_simulate


def test_eacd_recovers_params() -> None:
    rng = np.random.default_rng(7)
    x = acd_simulate(0.05, 0.10, 0.85, 2000, rng)
    fit = acd_fit(x, dist="exp")
    assert fit["converged"] == 1.0
    assert fit["alpha"] == pytest.approx(0.10, abs=0.08)
    assert fit["beta"] == pytest.approx(0.85, abs=0.12)
    assert fit["persistence"] == pytest.approx(0.95, abs=0.15)


def test_residuals_whitened() -> None:
    rng = np.random.default_rng(3)
    x = acd_simulate(0.05, 0.15, 0.80, 3000, rng)
    fit = acd_fit(x, dist="exp")
    diag = acd_diagnostics(np.asarray(fit["resid"]))
    assert diag["mean_resid"] == pytest.approx(1.0, abs=0.08)
    # residuals should be far less autocorrelated than raw durations
    assert diag["lb_pvalue"] > 0.01


def test_raw_durations_autocorrelated() -> None:
    rng = np.random.default_rng(9)
    x = acd_simulate(0.05, 0.15, 0.80, 3000, rng)
    diag = acd_diagnostics(x)
    assert diag["lb_pvalue"] < 0.05  # clustering present pre-filter


def test_wacd_runs() -> None:
    rng = np.random.default_rng(5)
    x = acd_simulate(0.05, 0.10, 0.85, 1500, rng)
    fit = acd_fit(x, dist="weibull")
    assert "gamma" in fit
    assert fit["gamma"] > 0.05
    assert np.isfinite(np.asarray(fit["resid"])).all()


def test_simulate_fail_closed() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        acd_simulate(0.05, 0.6, 0.6, 100, rng)  # alpha+beta >= 1
    with pytest.raises(ValueError):
        acd_simulate(-0.1, 0.1, 0.8, 100, rng)
    with pytest.raises(ValueError):
        acd_simulate(0.05, 0.1, 0.8, 5, rng)


def test_fit_fail_closed() -> None:
    with pytest.raises(ValueError):
        acd_fit(np.ones(30))
    with pytest.raises(ValueError):
        acd_fit(-np.abs(np.random.default_rng(0).standard_normal(100)))
    with pytest.raises(ValueError):
        acd_fit(np.full(100, np.nan))
    with pytest.raises(ValueError):
        acd_fit(np.random.default_rng(0).exponential(1.0, 100), dist="bogus")
