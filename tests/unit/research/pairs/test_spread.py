"""OU half-life known answers, trailing z-score, and band state machine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.research.pairs.fixtures import ar1_spread
from quant_fund.research.pairs.spread import (
    ar1_fit,
    bands_position,
    ou_half_life,
    ou_params,
    zscore_trailing,
)


class TestHalfLife:
    def test_known_answer_ar1(self):
        # True half-life for rho=0.9: -ln2/ln0.9 ~= 6.58. OLS AR(1) is
        # downward-biased, so accept a wide honest band.
        s = ar1_spread(n=2000, rho=0.90, sigma=0.15, seed=0)
        est = ou_half_life(s)
        assert 3.0 < est < 12.0

    def test_faster_reversion_shorter_half_life(self):
        slow = ou_half_life(ar1_spread(n=2000, rho=0.95, sigma=0.1, seed=4))
        fast = ou_half_life(ar1_spread(n=2000, rho=0.80, sigma=0.1, seed=4))
        assert fast < slow

    def test_random_walk_not_reverting(self):
        rw = np.cumsum(np.random.default_rng(0).normal(size=500))
        fast = ou_half_life(ar1_spread(n=500, rho=0.9, sigma=0.15, seed=0))
        hl = ou_half_life(rw)
        # rho ~= 1 -> no finite half-life, or a horizon far beyond any
        # reverting spread (a finite RW can still yield rho < 1).
        assert hl == math.inf or hl > 10.0 * fast

    def test_ar1_fit_recovers_rho(self):
        s = ar1_spread(n=2000, rho=0.85, sigma=0.2, seed=9)
        _, rho = ar1_fit(s)
        assert rho == pytest.approx(0.85, abs=0.1)

    def test_ou_params(self):
        s = ar1_spread(n=2000, rho=0.9, sigma=0.15, seed=3)
        p = ou_params(s)
        assert p["rho"] == pytest.approx(0.9, abs=0.1)
        assert p["theta"] > 0
        assert p["half_life"] == pytest.approx(-math.log(2) / math.log(p["rho"]), rel=1e-9)

    def test_failclosed(self):
        with pytest.raises(ValueError):
            ou_half_life(np.ones(5))
        with pytest.raises(ValueError):
            ar1_fit(np.full(50, np.nan))


class TestZScore:
    def test_warmup_is_nan_and_known_answer(self):
        s = np.concatenate([np.ones(9), np.array([2.0, 3.0])])
        z = zscore_trailing(s, window=10)
        assert np.all(np.isnan(z[:9]))
        # Window at t=9 (indices 0..9): nine 1s + a 2 -> mean 1.1, sd 0.3;
        # z = 0.9/0.3 = 3.
        assert z[9] == pytest.approx(3.0, rel=1e-9)
        # Window at t=10 (indices 1..10): eight 1s + 2 + 3 -> mean 1.3,
        # sd = sqrt(0.41); z = 1.7/sqrt(0.41) ~= 2.655.
        assert z[10] == pytest.approx(1.7 / math.sqrt(0.41), rel=1e-9)

    def test_constant_spread_gives_zero(self):
        z = zscore_trailing(np.full(30, 1.5), window=10)
        assert np.all(z[9:] == 0.0)

    def test_bounded_given_bounded_spread(self):
        rng = np.random.default_rng(7)
        s = rng.uniform(-2, 2, size=200)
        z = zscore_trailing(s, window=20)
        # Exact bound for a window of n points with population std: |z| <= sqrt(n-1).
        assert np.nanmax(np.abs(z)) <= math.sqrt(19) + 1e-6

    def test_failclosed(self):
        with pytest.raises(ValueError):
            zscore_trailing(np.arange(10.0), window=3)
        with pytest.raises(ValueError):
            zscore_trailing(np.arange(10.0), window=10, min_periods=2)


class TestBands:
    def test_state_machine_known_answer(self):
        z = np.array([0.0, 2.5, 2.2, 0.4, 0.0, -2.5, -1.0, -0.4, 0.0, np.nan, 3.0])
        pos = bands_position(z, entry=2.0, exit=0.5)
        np.testing.assert_array_equal(
            pos,
            [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, -1],
        )

    def test_output_alphabet(self):
        rng = np.random.default_rng(0)
        z = rng.normal(scale=3.0, size=500)
        pos = bands_position(z, entry=1.5, exit=0.5)
        assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}

    def test_failclosed(self):
        with pytest.raises(ValueError):
            bands_position(np.zeros(10), entry=0.0)
        with pytest.raises(ValueError):
            bands_position(np.zeros(10), entry=1.0, exit=2.0)
