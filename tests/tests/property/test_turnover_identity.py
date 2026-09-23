"""Wave 35: turnover identity properties (research math — not live P&L)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.returns import turnover


@st.composite
def equal_length_weight_pairs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    n = draw(st.integers(min_value=0, max_value=12))
    elem = st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False)
    w = draw(st.lists(elem, min_size=n, max_size=n))
    p = draw(st.lists(elem, min_size=n, max_size=n))
    return w, p


@given(equal_length_weight_pairs())
@settings(max_examples=60, deadline=None)
def test_turnover_nonnegative_and_identity(
    pair: tuple[list[float], list[float]],
) -> None:
    w, p = pair
    t = turnover(np.asarray(w, dtype=float), np.asarray(p, dtype=float))
    assert t >= 0.0
    assert math.isfinite(t)
    # Identity: identical books → zero turnover.
    assert turnover(np.asarray(w, dtype=float), np.asarray(w, dtype=float)) == 0.0
    # Empty books.
    if len(w) == 0:
        assert t == 0.0
    else:
        assert t == pytest.approx(float(np.sum(np.abs(np.asarray(w) - np.asarray(p)))))


@given(
    st.lists(st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=8)
)
@settings(max_examples=40, deadline=None)
def test_turnover_length_mismatch_raises(w: list[float]) -> None:
    bad = np.asarray(w[:-1] if len(w) > 1 else [0.0, 1.0], dtype=float)
    with pytest.raises(ValueError, match="mismatch"):
        turnover(np.asarray(w, dtype=float), bad)


def test_turnover_nonfinite_returns_nan() -> None:
    assert math.isnan(turnover(np.array([1.0, np.nan]), np.array([0.0, 0.0])))
    assert math.isnan(turnover(np.array([1.0, 0.0]), np.array([0.0, np.inf])))
