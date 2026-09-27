"""Tests for models/ar_estimation.py — Yule-Walker, Levinson-Durbin, Burg."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ar_estimation import (
    autocovariance,
    burg,
    levinson_durbin,
    yule_walker,
)


def _ar2(phi1: float, phi2: float, n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[200:]


def test_yule_walker_recovers_ar2() -> None:
    x = _ar2(0.5, -0.3)
    out = yule_walker(x, 2)
    ar = np.asarray(out["ar"])
    assert np.allclose(ar, [0.5, -0.3], atol=0.08)
    assert float(out["sigma2"]) > 0


def test_levinson_matches_yule_walker() -> None:
    x = _ar2(0.6, -0.2, seed=1)
    r = autocovariance(x, 2)
    ld = levinson_durbin(r, 2)
    yw = yule_walker(x, 2)
    assert np.allclose(np.asarray(ld["ar"]), np.asarray(yw["ar"]), atol=1e-8)
    assert np.asarray(ld["reflection"]).shape == (2,)


def test_burg_recovers_ar2() -> None:
    x = _ar2(0.5, -0.3, seed=2)
    out = burg(x, 2)
    ar = np.asarray(out["ar"])
    assert np.allclose(ar, [0.5, -0.3], atol=0.08)
    assert np.asarray(out["reflection"]).shape == (2,)


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        yule_walker(np.arange(50.0), 0)
    with pytest.raises(ValueError):
        burg(np.arange(4.0), 2)
    with pytest.raises(ValueError):
        levinson_durbin(np.array([1.0]), 2)
