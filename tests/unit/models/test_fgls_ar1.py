"""Tests for models/fgls_ar1.py — Cochrane-Orcutt & Prais-Winsten."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fgls_ar1 import cochrane_orcutt, prais_winsten


def _ar1_error_data(rho: float, beta: tuple[float, float], n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    y = beta[0] + beta[1] * x + u
    return y, x


def test_cochrane_orcutt_recovers_params() -> None:
    y, x = _ar1_error_data(0.7, (1.0, 2.0), 3000, 0)
    out = cochrane_orcutt(y, x)
    beta = np.asarray(out["beta"])
    assert abs(beta[0] - 1.0) < 0.2
    assert abs(beta[1] - 2.0) < 0.1
    assert abs(float(out["rho"]) - 0.7) < 0.1


def test_prais_winsten_recovers_params() -> None:
    y, x = _ar1_error_data(-0.5, (0.0, 1.5), 3000, 1)
    out = prais_winsten(y, x)
    beta = np.asarray(out["beta"])
    assert abs(beta[1] - 1.5) < 0.1
    assert abs(float(out["rho"]) + 0.5) < 0.1
    assert (np.asarray(out["se"]) > 0).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        cochrane_orcutt(np.arange(5.0), np.arange(5.0))  # too few obs
    with pytest.raises(ValueError):
        prais_winsten(np.arange(200.0), np.arange(199.0))  # misaligned
