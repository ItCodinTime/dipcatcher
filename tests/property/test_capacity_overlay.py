"""Property tests for the capacity-overlay lane (P5.2/P5.6)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.research.capacity_overlay import (
    SyntheticBook,
    capacity_metrics,
    vol_target_scales,
)

_returns = st.lists(
    st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
    min_size=40,
    max_size=120,
)


@given(_returns)
@settings(max_examples=30, deadline=None)
def test_scales_bounded(returns: list[float]) -> None:
    s = vol_target_scales(np.asarray(returns), lookback=10, max_leverage=2.0)
    assert np.all(s >= 0.0) and np.all(s <= 2.0)


@given(_returns)
@settings(max_examples=30, deadline=None)
def test_scales_causal_under_future_mutation(returns: list[float]) -> None:
    r = np.asarray(returns)
    s1 = vol_target_scales(r, lookback=10)
    r2 = r.copy()
    r2[-10:] = 0.5  # extreme tail mutation
    s2 = vol_target_scales(r2, lookback=10)
    # indices whose windows exclude the tail must be identical
    np.testing.assert_allclose(s1[:-10], s2[:-10])


def test_relabeling_invariance() -> None:
    """Permuting names jointly across weights+ADV preserves metrics."""
    rng = np.random.default_rng(9)
    n, t = 8, 40
    w = np.abs(rng.normal(0, 0.1, (t, n)))
    w /= w.sum(axis=1, keepdims=True)
    adv = np.exp(rng.normal(np.log(1e8), 0.5, (t, n)))
    perm = rng.permutation(n)
    b1 = SyntheticBook("a", w, adv)
    b2 = SyntheticBook("b", w[:, perm], adv[:, perm])
    m1 = capacity_metrics(b1, aum=1e7, participation_cap=0.1)
    m2 = capacity_metrics(b2, aum=1e7, participation_cap=0.1)
    for key in ("feasible", "max_participation", "days_to_trade", "impact_bps"):
        assert m1[key] == pytest.approx(m2[key]) or m1[key] == m2[key]


@given(
    st.floats(min_value=1e5, max_value=1e9, allow_nan=False),
    st.floats(min_value=1e5, max_value=1e9, allow_nan=False),
)
@settings(max_examples=25, deadline=None)
def test_participation_monotone_in_aum(a1: float, a2: float) -> None:
    rng = np.random.default_rng(4)
    n, t = 6, 30
    w = np.full((t, n), 1.0 / n)
    w[::6] *= 2
    w /= w.sum(axis=1, keepdims=True)
    adv = np.exp(rng.normal(np.log(1e8), 0.3, (t, n)))
    book = SyntheticBook("m", w, adv)
    lo, hi = min(a1, a2), max(a1, a2)
    m_lo = capacity_metrics(book, aum=lo, participation_cap=0.1)
    m_hi = capacity_metrics(book, aum=hi, participation_cap=0.1)
    assert m_hi["max_participation"] >= m_lo["max_participation"]
    assert m_hi["days_to_trade"] >= m_lo["days_to_trade"]
    assert m_hi["feasible"] <= m_lo["feasible"]
