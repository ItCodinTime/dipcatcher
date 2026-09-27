"""Tests for models/arma.py — CSS, Hannan-Rissanen, AIC selection."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.arma import arma_css, arma_select, hannan_rissanen


def _ar1(n: int = 800, phi: float = 0.7, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = 0.0
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.standard_normal()
    return y


def _arma11(n: int = 1000, phi: float = 0.6, theta: float = 0.4, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rng.standard_normal()
        y[t] = phi * y[t - 1] + e[t] + theta * e[t - 1]
    return y


def test_css_ar1() -> None:
    y = _ar1()
    out = arma_css(y, 1, 0)
    assert abs(float(np.asarray(out["phi"])[0]) - 0.7) < 0.08
    assert out["sigma2"] < 1.6  # innovations var = 1 + estimation slack


def test_css_arma11() -> None:
    y = _arma11()
    out = arma_css(y, 1, 1)
    phi = float(np.asarray(out["phi"])[0])
    theta = float(np.asarray(out["theta"])[0])
    assert abs(phi - 0.6) < 0.15
    assert abs(theta - 0.4) < 0.2


def test_hannan_rissanen() -> None:
    y = _arma11(seed=2)
    out = hannan_rissanen(y, 1, 1)
    phi = float(np.asarray(out["phi"])[0])
    theta = float(np.asarray(out["theta"])[0])
    assert abs(phi - 0.6) < 0.25
    assert abs(theta - 0.4) < 0.3


def test_aic_detects_signal() -> None:
    # AIC can legitimately overfit order; check stable invariants instead:
    # AR(1) data -> selected model must keep at least one AR lag and beat
    # the (0,0) null by a wide margin; ARMA(1,1) -> p + q >= 2.
    y = _ar1(seed=3)
    sel = arma_select(y, max_p=3, max_q=2)
    assert sel["p"] >= 1.0
    assert sel["aic"] <= float(arma_css(y, 1, 0)["aic"]) + 1e-6
    ya = _arma11(seed=4)
    aic_null = ya.size * np.log(float(ya.var())) + 2.0
    sela = arma_select(ya, max_p=3, max_q=3)
    assert sela["p"] + sela["q"] >= 2.0
    assert sela["aic"] < aic_null - 50.0  # strong-signal fixture


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        arma_css(np.random.default_rng(0).standard_normal(30), 2, 1)
    with pytest.raises(ValueError):
        arma_css(np.random.default_rng(0).standard_normal(200), 0, 0)
    with pytest.raises(ValueError):
        hannan_rissanen(np.random.default_rng(0).standard_normal(200), -1, 0)
