"""Tests for models/stoch_vol.py — Harvey-Shephard SV QMLE."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.stoch_vol import stoch_vol_fit


def _sv_data(
    n: int = 1500, phi: float = 0.95, s_eta: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = 0.0
    for t in range(1, n):
        h[t] = phi * h[t - 1] + s_eta * rng.standard_normal()
    y = np.exp(h / 2) * rng.standard_normal(n)
    return y, h


def test_sv_recovers_phi() -> None:
    y, h = _sv_data()
    out = stoch_vol_fit(y)
    assert 0.5 < out["phi"] < 1.0
    assert out["sigma_eta"] > 0
    # filtered log-vol should correlate with the true path
    hf = np.asarray(out["h"])
    assert np.corrcoef(hf, h)[0, 1] > 0.3


def test_sv_vol_positive() -> None:
    y, _ = _sv_data(seed=1)
    out = stoch_vol_fit(y)
    vol = np.asarray(out["vol"])
    assert (vol > 0).all() and np.isfinite(vol).all()


def test_sv_stationarity() -> None:
    y, _ = _sv_data(phi=0.6, s_eta=0.4, seed=2)
    out = stoch_vol_fit(y)
    assert abs(out["phi"]) < 1.0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        stoch_vol_fit(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        stoch_vol_fit(np.full(200, np.nan))
    y = np.random.default_rng(0).standard_normal(200)
    y[10] = 0.0
    with pytest.raises(ValueError):
        stoch_vol_fit(y)
