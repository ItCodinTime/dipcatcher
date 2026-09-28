"""Stat-arb pair invariants: no-lookahead, determinism, bounded z-score."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.research.pairs.cointegration import benjamini_hochberg, bonferroni
from quant_fund.research.pairs.hedge import kalman_hedge_ratio
from quant_fund.research.pairs.pit import pit_pair_signals
from quant_fund.research.pairs.spread import bands_position, zscore_trailing


def _bounded_spread(draw_values):
    return st.lists(draw_values, min_size=80, max_size=300).map(
        lambda xs: np.asarray(xs, dtype=float)
    )


@settings(max_examples=30, deadline=None, derandomize=True)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.5, max_value=5.0),
    window=st.integers(min_value=10, max_value=40),
)
def test_zscore_bounded_given_bounded_spread(seed: int, scale: float, window: int) -> None:
    """For a window of n points, |z| <= sqrt(n-1) exactly (population std)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-scale, scale, size=200)
    z = zscore_trailing(s, window=window)
    assert np.isnan(z[: window - 1]).all()
    assert np.nanmax(np.abs(z)) <= math.sqrt(window - 1) + 1e-6


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    cut=st.integers(min_value=100, max_value=250),
)
def test_zscore_no_lookahead(seed: int, cut: int) -> None:
    """Mutating the tail of the spread never changes z at or before the cut."""
    rng = np.random.default_rng(seed)
    s = rng.normal(size=300)
    z0 = zscore_trailing(s, window=30)
    s2 = s.copy()
    s2[cut + 1 :] += rng.normal(scale=100.0, size=299 - cut)
    z1 = zscore_trailing(s2, window=30)
    np.testing.assert_array_equal(z0[: cut + 1], z1[: cut + 1])


@settings(max_examples=15, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_pipeline_deterministic(seed: int) -> None:
    """The full PIT pipeline is a pure function of its inputs."""
    rng = np.random.default_rng(seed)
    n = 300
    base = np.cumsum(rng.normal(size=n))
    spr = np.zeros(n)
    for t in range(1, n):
        spr[t] = 0.9 * spr[t - 1] + rng.normal(scale=0.2)
    a = np.exp(base / 10 + 3)
    b = np.exp((base + spr) / 10 + 3)
    s0 = pit_pair_signals(a, b, window=80, z_window=40)
    s1 = pit_pair_signals(a, b, window=80, z_window=40)
    for key in s0:
        np.testing.assert_array_equal(s0[key], s1[key])
    assert set(np.unique(s0["position"])) <= {-1.0, 0.0, 1.0}


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    entry=st.floats(min_value=1.0, max_value=3.0),
    exit_band=st.floats(min_value=0.0, max_value=0.9),
)
def test_bands_position_alphabet(seed: int, entry: float, exit_band: float) -> None:
    """Positions stay in {-1, 0, +1} for any valid band config."""
    rng = np.random.default_rng(seed)
    z = rng.normal(scale=3.0, size=300)
    pos = bands_position(z, entry=entry, exit=exit_band)
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    # Exit band < entry band by construction of the API (validated inside).


@settings(max_examples=20, deadline=None, derandomize=True)
@given(
    m=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_bh_qvalues_wellformed(m: int, seed: int) -> None:
    """BH output: in [0,1], >= raw p, monotone nondecreasing in sorted order."""
    rng = np.random.default_rng(seed)
    p = np.sort(rng.uniform(0.0, 1.0, size=m))
    q = benjamini_hochberg(p)
    assert q.shape == (m,)
    assert np.all(q >= p - 1e-12)
    assert np.all(q <= 1.0)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    qb = bonferroni(p)
    assert np.all(qb >= q - 1e-12)  # Bonferroni is never weaker than BH


@settings(max_examples=15, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_kalman_no_lookahead(seed: int) -> None:
    """The scalar Kalman filter is forward-only: tail mutations cannot reach back."""
    rng = np.random.default_rng(seed)
    n = 300
    x = np.cumsum(rng.normal(size=n))
    y = 1.3 * x + rng.normal(scale=0.1, size=n)
    p0 = kalman_hedge_ratio(y, x, q=1e-3)
    cut = 200
    x2, y2 = x.copy(), y.copy()
    x2[cut + 1 :] *= 100.0
    y2[cut + 1 :] *= 100.0
    p1 = kalman_hedge_ratio(y2, x2, q=1e-3)
    # Calibration lives in the burn-in prefix, so the <= cut path is
    # bit-identical under tail mutation...
    np.testing.assert_array_equal(p0[: cut + 1], p1[: cut + 1])
    # ...and a prefix call reproduces the same prefix path exactly.
    p_prefix = kalman_hedge_ratio(y[: cut + 1], x[: cut + 1], q=1e-3)
    np.testing.assert_array_equal(p_prefix, p0[: cut + 1])
