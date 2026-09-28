"""Strict point-in-time proof: mutating data after t changes nothing at <= t."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.pairs.fixtures import planted_pair_panel
from quant_fund.research.pairs.pit import pit_pair_signals, rolling_hedge_ratio
from quant_fund.research.pairs.spread import bands_position, zscore_trailing

_CUT = 400  # mutate everything strictly after this index


def _mutated(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a2, b2 = a.copy(), b.copy()
    # Large but finite mutations — validation must stay happy so the only
    # thing that could change outputs is an actual look-ahead bug.
    a2[_CUT + 1 :] *= 1000.0
    b2[_CUT + 1 :] *= 1e-3
    return a2, b2


@pytest.fixture()
def pair_prices():
    panel = planted_pair_panel(n_assets=4, n_dates=600, seed=8)
    prices = np.asarray(panel["prices"], dtype=float)
    return prices[:, 0], prices[:, 1]


class TestRollingHedgeNoLookahead:
    @pytest.mark.parametrize("method", ["ols", "kalman"])
    def test_mutation_after_t_bit_identical(self, pair_prices, method):
        a, b = pair_prices
        alpha0, beta0 = rolling_hedge_ratio(a, b, window=120, method=method)
        a2, b2 = _mutated(a, b)
        alpha1, beta1 = rolling_hedge_ratio(a2, b2, window=120, method=method)
        np.testing.assert_array_equal(alpha0[: _CUT + 1], alpha1[: _CUT + 1])
        np.testing.assert_array_equal(beta0[: _CUT + 1], beta1[: _CUT + 1])
        # Sanity: the mutation really did change the inputs.
        assert not np.array_equal(a, a2)


class TestSignalNoLookahead:
    @pytest.mark.parametrize("method", ["ols", "kalman"])
    def test_full_pipeline_bit_identical(self, pair_prices, method):
        a, b = pair_prices
        s0 = pit_pair_signals(a, b, window=120, z_window=60, method=method)
        a2, b2 = _mutated(a, b)
        s1 = pit_pair_signals(a2, b2, window=120, z_window=60, method=method)
        for key in ("alpha", "beta", "spread", "z", "position"):
            np.testing.assert_array_equal(s0[key][: _CUT + 1], s1[key][: _CUT + 1])


class TestPrimitivesNoLookahead:
    def test_zscore_trailing(self):
        rng = np.random.default_rng(0)
        s = rng.normal(size=300)
        z0 = zscore_trailing(s, window=30)
        s2 = s.copy()
        s2[200:] += 50.0
        z1 = zscore_trailing(s2, window=30)
        np.testing.assert_array_equal(z0[:200], z1[:200])

    def test_bands_position(self):
        rng = np.random.default_rng(1)
        z = rng.normal(scale=3.0, size=200)
        p0 = bands_position(z, entry=2.0, exit=0.5)
        z2 = z.copy()
        z2[150:] = 99.0
        p1 = bands_position(z2, entry=2.0, exit=0.5)
        np.testing.assert_array_equal(p0[:150], p1[:150])


def test_determinism(pair_prices):
    a, b = pair_prices
    s0 = pit_pair_signals(a, b, window=120, z_window=60)
    s1 = pit_pair_signals(a, b, window=120, z_window=60)
    for key in s0:
        np.testing.assert_array_equal(s0[key], s1[key])


def test_outputs_have_warmup_nan(pair_prices):
    a, b = pair_prices
    sig = pit_pair_signals(a, b, window=120, z_window=60)
    assert np.all(np.isnan(sig["beta"][:119]))
    assert np.all(np.isnan(sig["z"][:178]))  # 119 hedge + 59 z warmup
    assert np.isfinite(sig["z"][-1])
    assert sig["position"].shape == a.shape
