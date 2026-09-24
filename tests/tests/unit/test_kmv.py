"""Tests for models/kmv.py — Merton-KMV distance-to-default."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from quant_fund.models.kmv import merton_kmv_solve


def _forward(v: float, sv: float, d: float, r: float, t: float) -> tuple[float, float]:
    """Forward map (V, sV) -> (E, sE) for round-trip testing."""
    sqt = sv * np.sqrt(t)
    d1 = (np.log(v / d) + (r + 0.5 * sv * sv) * t) / sqt
    d2 = d1 - sqt
    e = v * stats.norm.cdf(d1) - d * np.exp(-r * t) * stats.norm.cdf(d2)
    se = (v / e) * stats.norm.cdf(d1) * sv
    return e, se


def test_round_trip() -> None:
    v_true, sv_true, d, r, t = 120.0, 0.25, 80.0, 0.02, 1.0
    e, se = _forward(v_true, sv_true, d, r, t)
    out = merton_kmv_solve(e, se, d, r, t)
    assert abs(float(np.asarray(out["V"])[0]) - v_true) / v_true < 0.02
    assert abs(float(np.asarray(out["sigma_V"])[0]) - sv_true) < 0.03


def test_dd_monotone_in_leverage() -> None:
    # higher debt -> lower distance-to-default, higher PD
    v, sv = 100.0, 0.3
    d_lo, d_hi = 40.0, 90.0
    e_lo, se_lo = _forward(v, sv, d_lo, 0.0, 1.0)
    e_hi, se_hi = _forward(v, sv, d_hi, 0.0, 1.0)
    out_lo = merton_kmv_solve(e_lo, se_lo, d_lo, 0.0, 1.0)
    out_hi = merton_kmv_solve(e_hi, se_hi, d_hi, 0.0, 1.0)
    dd_lo = float(np.asarray(out_lo["DD"])[0])
    dd_hi = float(np.asarray(out_hi["DD"])[0])
    pd_lo = float(np.asarray(out_lo["PD"])[0])
    pd_hi = float(np.asarray(out_hi["PD"])[0])
    assert dd_lo > dd_hi
    assert pd_hi > pd_lo


def test_vectorized() -> None:
    vs = np.array([100.0, 150.0, 200.0])
    svs = np.array([0.2, 0.3, 0.35])
    ds = np.array([60.0, 90.0, 100.0])
    es = np.empty(3)
    ses = np.empty(3)
    for i in range(3):
        es[i], ses[i] = _forward(vs[i], svs[i], ds[i], 0.01, 1.0)
    out = merton_kmv_solve(es, ses, ds, 0.01, 1.0)
    v_hat = np.asarray(out["V"])
    assert np.abs(v_hat - vs).max() / vs.mean() < 0.05


def test_pd_bounds() -> None:
    e, se = _forward(150.0, 0.2, 60.0, 0.02, 1.0)
    out = merton_kmv_solve(e, se, 60.0, 0.02, 1.0)
    pd = float(np.asarray(out["PD"])[0])
    assert 0.0 <= pd <= 1.0
    assert pd < 0.1  # low leverage should be safe


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        merton_kmv_solve(-10.0, 0.3, 80.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        merton_kmv_solve(50.0, 0.0, 80.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        merton_kmv_solve(50.0, 0.3, -80.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        merton_kmv_solve(50.0, 0.3, 80.0, 0.0, -1.0)
