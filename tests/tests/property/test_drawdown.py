"""Wave 41: drawdown properties (research math — not live P&L).

max_drawdown / drawdown_series are diagnostic path metrics only.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.returns import drawdown_series, max_drawdown, wealth_index


@given(
    st.lists(st.floats(0.0, 0.05, allow_nan=False, allow_infinity=False), min_size=2, max_size=40)
)
@settings(max_examples=40, deadline=None)
def test_nonneg_returns_zero_mdd(rs: list[float]) -> None:
    """Monotone non-decreasing wealth → mdd == 0."""
    assert max_drawdown(np.array(rs, dtype=float)) == 0.0


@given(
    st.lists(
        st.floats(-0.4, 0.4, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=40,
    )
)
@settings(max_examples=60, deadline=None)
def test_mdd_nonpositive_and_bounded(rs: list[float]) -> None:
    """Finite returns with r > -1 → mdd in [-1, 0]; drawdown path ≤ 0."""
    r = np.asarray(rs, dtype=float)
    # Keep wealth positive so peak division is well-defined
    r = np.clip(r, -0.9, None)
    mdd = max_drawdown(r)
    assert math.isfinite(mdd)
    assert mdd <= 0.0
    assert mdd >= -1.0 - 1e-12
    dd = drawdown_series(r)
    assert dd.size == r.size
    assert np.all(dd <= 1e-12)
    # Wealth starts compounding from 1; peak ≥ current → dd ≤ 0
    w = wealth_index(r)
    assert w.size == r.size
    assert np.all(np.isfinite(w))


@given(
    st.lists(
        st.floats(-0.3, 0.3, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=30,
    )
)
@settings(max_examples=40, deadline=None)
def test_mdd_matches_min_drawdown_series(rs: list[float]) -> None:
    r = np.clip(np.asarray(rs, dtype=float), -0.9, None)
    dd = drawdown_series(r)
    assert np.isclose(max_drawdown(r), float(np.min(dd)))


def test_empty_drawdown_edges() -> None:
    assert drawdown_series(np.array([])).size == 0
    assert max_drawdown(np.array([])) == 0.0
