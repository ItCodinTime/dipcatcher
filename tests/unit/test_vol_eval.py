"""Tests for metrics/vol_eval.py — MZ regression + Patton losses."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.vol_eval import (
    hmse,
    mae,
    mincer_zarnowitz,
    mse,
    mse_log,
    qlike,
    vol_loss_diff,
)


def _vol_fixture(n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """GARCH variance path + squared-return proxy."""
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    r2 = np.empty(n)
    sig2[0] = 1.0
    r2[0] = sig2[0] * rng.standard_normal() ** 2
    for t in range(1, n):
        sig2[t] = 0.05 + 0.1 * r2[t - 1] + 0.85 * sig2[t - 1]
        r2[t] = sig2[t] * rng.standard_normal() ** 2
    return np.maximum(r2, 1e-8), np.maximum(sig2, 1e-8)


def test_mz_well_calibrated() -> None:
    proxy, sig2 = _vol_fixture()
    out = mincer_zarnowitz(proxy, sig2)
    # unbiased forecast: beta ~ 1, alpha ~ 0. R2 stays small because
    # squared returns are a noisy proxy (Andersen-Bollerslev bound).
    assert abs(out["beta"] - 1.0) < 0.35
    assert abs(out["alpha"]) < 0.5
    assert 0.0 < out["r2"] < 0.3


def test_mz_detects_bias() -> None:
    proxy, sig2 = _vol_fixture(seed=1)
    out = mincer_zarnowitz(proxy, 1.8 * sig2)  # scaled-up forecast
    assert out["beta"] < 0.85  # slope should drop


def test_qlike_prefers_true_forecast() -> None:
    proxy, sig2 = _vol_fixture(seed=2)
    good = qlike(proxy, sig2).mean()
    bad = qlike(proxy, sig2 * 1.6).mean()
    assert good < bad


def test_losses_shapes_and_bounds() -> None:
    proxy, sig2 = _vol_fixture()
    for fn in (mse, qlike, mse_log, hmse, mae):
        v = fn(proxy, sig2)
        assert v.shape == proxy.shape
        assert np.isfinite(v).all()
        assert (v >= 0).all()


def test_vol_loss_diff_sign() -> None:
    proxy, sig2 = _vol_fixture(seed=3)
    out = vol_loss_diff(proxy, sig2 * 1.5, sig2)  # f1 worse -> diff > 0
    assert out["mean_diff"] > 0.0
    assert out["t"] > 0.0


def test_fail_closed() -> None:
    proxy, sig2 = _vol_fixture()
    with pytest.raises(ValueError):
        qlike(proxy, -sig2)  # negative forecast
    with pytest.raises(ValueError):
        mincer_zarnowitz(proxy[:5], sig2[:5])  # < 10 obs
    with pytest.raises(ValueError):
        mse(np.full(20, np.nan), sig2[:20])
    with pytest.raises(ValueError):
        vol_loss_diff(proxy, sig2, sig2, loss="bogus")
