"""Tests for models/threshold.py — Tong SETAR + Hansen (1999)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.threshold import (
    hansen_threshold_test,
    setar_fit,
    setar_predict,
)


def _make_setar(n: int, gamma: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    e = 0.3 * rng.standard_normal(n)
    for t in range(1, n):
        if y[t - 1] <= gamma:
            y[t] = 0.5 * y[t - 1] + e[t]
        else:
            y[t] = 1.0 - 0.3 * y[t - 1] + e[t]
    return y


def test_setar_recovers_threshold() -> None:
    y = _make_setar(600, gamma=0.5)
    fit = setar_fit(y, p=1, delay=1)
    assert fit["gamma"] == pytest.approx(0.5, abs=0.15)
    assert fit["ssr"] < fit["ssr_linear"] * 0.95
    assert fit["n_regime1"] > 100 and fit["n_regime2"] > 100


def test_setar_regime_coefs() -> None:
    y = _make_setar(800, gamma=0.0)
    fit = setar_fit(y, p=1, delay=1)
    # regime 1 (y<=0): slope ~0.5; regime 2: intercept ~1.0, slope ~-0.3
    assert fit["coef_regime1"][1] == pytest.approx(0.5, abs=0.15)
    assert fit["coef_regime2"][0] == pytest.approx(1.0, abs=0.2)


def test_hansen_rejects_on_threshold() -> None:
    y = _make_setar(400, gamma=0.4, seed=3)
    out = hansen_threshold_test(y, p=1, delay=1, n_boot=100, rng=np.random.default_rng(1))
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["pvalue"] <= 0.15


def test_hansen_no_reject_linear() -> None:
    rng = np.random.default_rng(9)
    n = 400
    y = np.zeros(n)
    e = 0.4 * rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + e[t]
    out = hansen_threshold_test(y, p=1, delay=1, n_boot=100, rng=np.random.default_rng(2))
    assert out["pvalue"] > 0.05


def test_predict_shape() -> None:
    y = _make_setar(300, gamma=0.3)
    fit = setar_fit(y, p=1, delay=1)
    fc = setar_predict(fit, y, steps=5)
    assert fc.shape == (5,)
    assert np.isfinite(fc).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        setar_fit(np.ones(20))
    with pytest.raises(ValueError):
        setar_fit(np.arange(100.0), p=0)
    y = np.linspace(0, 1, 100) + 1e-6 * np.random.default_rng(0).standard_normal(100)
    with pytest.raises(ValueError):
        hansen_threshold_test(y, n_boot=10)
    with pytest.raises(ValueError):
        setar_fit(np.full(100, np.nan))


def test_fixed_gamma_grid() -> None:
    y = _make_setar(500, gamma=0.2)
    grid = np.array([-0.5, 0.0, 0.2, 0.5])
    fit = setar_fit(y, p=1, delay=1, gamma_grid=grid, trim=0.1)
    assert fit["gamma"] == pytest.approx(0.2, abs=0.15)
