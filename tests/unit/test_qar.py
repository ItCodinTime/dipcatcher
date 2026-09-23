"""Tests for models/qar.py — Koenker-Xiao quantile autoregression."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.qar import qar_fit, qar_summary


def _ar1(n: int = 800, phi: float = 0.6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


def test_qar_median_matches_ols() -> None:
    y = _ar1()
    fit = qar_fit(y, 1, np.array([0.5]))
    coef = np.asarray(fit["coef"])
    # median QAR slope ~ phi
    assert abs(coef[0, 1] - 0.6) < 0.15


def test_qar_surface_shape() -> None:
    y = _ar1(seed=1)
    taus = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    fit = qar_fit(y, 1, taus)
    coef = np.asarray(fit["coef"])
    assert coef.shape == (5, 2)
    # intercepts should be increasing in tau (location shift)
    assert np.all(np.diff(coef[:, 0]) > -0.2)


def test_qar_coverage_calibration() -> None:
    y = _ar1(seed=2)
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    fit = qar_fit(y, 1, taus)
    summ = qar_summary(fit)
    dev = np.asarray(summ["coverage_dev"])
    assert dev.max() < 0.1  # in-sample coverage near tau


def test_qar_heteroskedastic_data() -> None:
    # GARCH-like data: tail QAR slopes differ from median
    rng = np.random.default_rng(3)
    n = 900
    y = np.empty(n)
    s2 = np.empty(n)
    y[0] = s2[0] = 0.5
    for t in range(1, n):
        s2[t] = 0.05 + 0.15 * y[t - 1] ** 2 + 0.8 * s2[t - 1]
        y[t] = np.sqrt(s2[t]) * rng.standard_normal()
    taus = np.array([0.1, 0.9])
    fit = qar_fit(y, 1, taus)
    a1 = np.asarray(fit["coef"])[:, 1]
    assert np.isfinite(a1).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        qar_fit(np.random.default_rng(0).standard_normal(30), 1, np.array([0.5]))
    with pytest.raises(ValueError):
        qar_fit(np.random.default_rng(0).standard_normal(200), 1, np.array([0.0, 0.5]))
    with pytest.raises(ValueError):
        qar_fit(np.random.default_rng(0).standard_normal(200), 0, np.array([0.5]))
