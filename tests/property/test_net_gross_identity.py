"""Wave 40: net/gross exposure identities (analytics — research diagnostic, not live P&L)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.analytics import mean_turnover, net_gross_exposure


@st.composite
def weight_vectors(draw: st.DrawFn) -> list[float]:
    n = draw(st.integers(min_value=0, max_value=16))
    elem = st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)
    return draw(st.lists(elem, min_size=n, max_size=n))


@given(weight_vectors())
@settings(max_examples=80, deadline=None)
def test_net_gross_weight_identities(w: list[float]) -> None:
    arr = np.asarray(w, dtype=float)
    out = net_gross_exposure(arr)
    finite = arr[np.isfinite(arr)]
    expected_gross = float(np.sum(np.abs(finite))) if finite.size else 0.0
    expected_net = float(np.sum(finite)) if finite.size else 0.0
    assert out["gross"] == pytest.approx(expected_gross)
    assert out["net"] == pytest.approx(expected_net)
    assert out["gross"] >= 0.0
    assert abs(out["net"]) <= out["gross"] + 1e-12
    assert out["n_long"] == int(np.sum(finite > 1e-12))
    assert out["n_short"] == int(np.sum(finite < -1e-12))
    assert out["n_long"] + out["n_short"] <= finite.size


@given(
    st.lists(st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
    st.lists(st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
)
@settings(max_examples=40, deadline=None)
def test_net_gross_series_summaries(gross: list[float], net: list[float]) -> None:
    g = np.asarray(gross, dtype=float)
    n = np.asarray(net, dtype=float)
    out = net_gross_exposure(None, gross_series=g, net_series=n)
    assert out["gross_mean"] == pytest.approx(float(np.mean(g)))
    assert out["gross_max"] == pytest.approx(float(np.max(g)))
    assert out["net_mean"] == pytest.approx(float(np.mean(n)))
    assert out["net_abs_mean"] == pytest.approx(float(np.mean(np.abs(n))))
    # Consistency: mean |net| >= |mean net|
    assert out["net_abs_mean"] + 1e-12 >= abs(out["net_mean"])


@given(
    st.lists(st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=24)
)
@settings(max_examples=40, deadline=None)
def test_mean_turnover_nonnegative_when_series_nonneg(ts: list[float]) -> None:
    """mean_turnover preserves non-negativity of a non-negative turnover series."""
    m = mean_turnover(np.asarray(ts, dtype=float))
    assert math.isfinite(m)
    assert m >= 0.0


def test_net_gross_empty_and_all_nan() -> None:
    empty = net_gross_exposure(np.array([]))
    assert empty["gross"] == 0.0 and empty["net"] == 0.0
    assert empty["n_long"] == 0 and empty["n_short"] == 0
    nan_only = net_gross_exposure(np.array([np.nan, np.nan]))
    assert nan_only["gross"] == 0.0 and nan_only["net"] == 0.0
