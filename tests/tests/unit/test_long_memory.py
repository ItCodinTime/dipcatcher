"""Tests for long-memory estimators."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.long_memory import (
    gph_estimate,
    lo_modified_rs,
    local_whittle,
    whittle_arfima,
)


def _arfima0(n: int, d: float, seed: int = 0) -> np.ndarray:
    """ARFIMA(0,d,0) via fracdiff filter on white noise."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 500)
    # y_t = (1-L)^{-d} e_t: convolution with psi_k = (k-1+d)/k recursion.
    psi = np.empty(2000)
    psi[0] = 1.0
    for k in range(1, 2000):
        psi[k] = psi[k - 1] * (k - 1.0 + d) / k
    y = np.convolve(e, psi)[: n + 500]
    return y[500:]


class TestGPH:
    def test_iid_d_near_zero(self):
        rng = np.random.default_rng(0)
        out = gph_estimate(rng.normal(size=600))
        assert abs(out["d"]) < 0.3

    def test_long_memory_detected(self):
        # GPH is a noisy semiparametric estimator — assert the estimate
        # lands in the right range and clearly exceeds the iid estimate.
        y = _arfima0(1200, d=0.40, seed=1)
        d_lm = gph_estimate(y)["d"]
        d_iid = gph_estimate(np.random.default_rng(9).normal(size=1200))["d"]
        assert d_lm > 0.15
        assert d_lm > d_iid + 0.1

    def test_failclosed(self):
        with pytest.raises(ValueError):
            gph_estimate(np.ones(200))


class TestLocalWhittle:
    def test_recovers_d(self):
        y = _arfima0(900, d=0.30, seed=2)
        out = local_whittle(y)
        assert abs(out["d"] - 0.30) < 0.25

    def test_white_noise(self):
        rng = np.random.default_rng(3)
        out = local_whittle(rng.normal(size=700))
        assert abs(out["d"]) < 0.3

    def test_failclosed(self):
        with pytest.raises(ValueError):
            local_whittle(np.ones(30))


class TestWhittleARFIMA:
    def test_direction(self):
        y_pos = _arfima0(600, d=0.3, seed=4)
        y_iid = np.random.default_rng(4).normal(size=600)
        d_pos = whittle_arfima(y_pos)["d"]
        d_iid = whittle_arfima(y_iid)["d"]
        assert d_pos > d_iid

    def test_failclosed(self):
        with pytest.raises(ValueError):
            whittle_arfima(np.full(100, np.nan))


class TestLoRS:
    def test_iid_within_band(self):
        rng = np.random.default_rng(5)
        out = lo_modified_rs(rng.normal(size=500))
        assert out["reject_short_memory"] == 0.0

    def test_persistent_series_outside_band(self):
        # Strongly trending/persistent series -> large R/S.
        rng = np.random.default_rng(6)
        y = np.cumsum(rng.normal(scale=0.3, size=600)) + np.linspace(0, 3, 600)
        out = lo_modified_rs(y)
        assert np.isfinite(out["statistic"])
        assert out["statistic"] > 0.8

    def test_failclosed(self):
        with pytest.raises(ValueError):
            lo_modified_rs(np.ones(100))
        with pytest.raises(ValueError):
            lo_modified_rs(np.random.default_rng(0).normal(size=100), q=80)
