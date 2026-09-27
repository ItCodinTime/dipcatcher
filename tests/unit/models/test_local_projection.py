"""Tests for models/local_projection.py — Jorda local projections."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.local_projection import local_projection


def test_irf_recovers_ar1_geometric_decay() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    rho = 0.6
    x = rng.standard_normal(n)  # iid impulse
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + x[t] + 0.1 * rng.standard_normal()
    out = local_projection(y, x, horizons=5, control_lags=2)
    irf = out["irf"]
    assert abs(irf[0] - 1.0) < 0.1
    assert abs(irf[1] - rho) < 0.1
    assert abs(irf[2] - rho**2) < 0.12
    assert (out["se"] > 0).all()


def test_output_shapes() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    y = np.cumsum(0.1 * rng.standard_normal(500)) + x
    out = local_projection(y, x, horizons=8)
    assert out["irf"].shape == (9,)
    assert out["ci_low"].shape == (9,)
    assert np.all(out["ci_high"] >= out["ci_low"])


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        local_projection(np.arange(30.0), np.arange(30.0), horizons=5)  # too short
    with pytest.raises(ValueError):
        local_projection(np.arange(200.0), np.arange(199.0))  # misaligned
