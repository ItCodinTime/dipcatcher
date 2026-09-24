"""Tests for models/dcc.py — Engle DCC(1,1)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dcc import dcc_fit, dcc_forecast


def _dcc_data(t: int = 1200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two series with a time-varying correlation path."""
    rng = np.random.default_rng(seed)
    rho_true = np.empty(t)
    rho_true[0] = 0.2
    r = np.empty((t, 2))
    for i in range(1, t):
        # correlation drifts: regime shift at t/2
        base = 0.2 if i < t // 2 else 0.7
        rho_true[i] = 0.98 * rho_true[i - 1] + 0.02 * base
        c = np.array([[1.0, rho_true[i]], [rho_true[i], 1.0]])
        r[i] = rng.multivariate_normal(np.zeros(2), c)
    r[0] = rng.standard_normal(2)
    return r, rho_true


def test_dcc_recovers_dynamics() -> None:
    r, rho_true = _dcc_data()
    out = dcc_fit(r)
    assert out["persistence"] < 1.0
    assert out["a"] >= 0 and out["b"] >= 0
    rp = np.asarray(out["R"])
    # fitted correlation should track the true path moderately
    corr = np.corrcoef(rp[:, 0, 1], rho_true)[0, 1]
    assert corr > 0.5
    # late-period rho should be higher than early (regime shift 0.2 -> 0.7)
    assert rp[-50:, 0, 1].mean() > rp[50 : r.shape[0] // 2, 0, 1].mean()


def test_dcc_r_path_valid() -> None:
    r, _ = _dcc_data(t=800, seed=1)
    out = dcc_fit(r)
    rp = np.asarray(out["R"])
    assert np.abs(rp).max() <= 1.0 + 1e-9
    assert np.allclose(np.diagonal(rp, axis1=1, axis2=2), 1.0)
    # every R_t must be PSD
    ev = np.linalg.eigvalsh(rp[::50])
    assert ev.min() > -1e-8


def test_dcc_forecast() -> None:
    r, _ = _dcc_data(t=800, seed=2)
    out = dcc_fit(r)
    rf = dcc_forecast(out)
    assert rf.shape == (2, 2)
    assert np.abs(rf[0, 1]) <= 1.0
    assert np.isfinite(rf).all()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        dcc_fit(np.random.default_rng(0).standard_normal((50, 3)))
    with pytest.raises(ValueError):
        dcc_fit(np.random.default_rng(0).standard_normal((200, 1)))
    r, _ = _dcc_data(t=200, seed=3)
    r[10, 0] = np.nan
    with pytest.raises(ValueError):
        dcc_fit(r)
