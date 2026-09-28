"""Property tests for run comparison: symmetry, determinism, small-n honesty."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.research.compare import compare_series

_FINITE = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_SERIES = st.lists(_FINITE, min_size=10, max_size=120)


@given(a=_SERIES, b=_SERIES, higher=st.booleans())
@settings(max_examples=30, deadline=None, derandomize=True)
def test_order_swap_flips_sign(a: list[float], b: list[float], higher: bool) -> None:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ab = compare_series("k", a, b, higher_is_better=higher, n_boot=200, seed=3)
    ba = compare_series("k", b, a, higher_is_better=higher, n_boot=200, seed=3)
    assert ab.mean_delta == pytest.approx(-ba.mean_delta)
    assert ab.dm_stat == pytest.approx(-ba.dm_stat, nan_ok=True)
    assert ab.dm_p == pytest.approx(ba.dm_p, nan_ok=True)
    if math.isfinite(ab.bootstrap_lo) and math.isfinite(ba.bootstrap_lo):
        assert ab.bootstrap_lo == pytest.approx(-ba.bootstrap_hi)
        assert ab.bootstrap_hi == pytest.approx(-ba.bootstrap_lo)
    swap_map = {"a_better": "b_better", "b_better": "a_better"}
    assert ba.verdict == swap_map.get(ab.verdict, ab.verdict)


@given(a=_SERIES, b=_SERIES)
@settings(max_examples=30, deadline=None, derandomize=True)
def test_same_seed_deterministic(a: list[float], b: list[float]) -> None:
    n = min(len(a), len(b))
    first = compare_series("k", a[:n], b[:n], n_boot=200, seed=9)
    second = compare_series("k", a[:n], b[:n], n_boot=200, seed=9)
    # to_dict maps NaN -> None so degenerate comparisons compare equal.
    assert first.to_dict() == second.to_dict()


@given(a=_SERIES)
@settings(max_examples=30, deadline=None, derandomize=True)
def test_identical_series_no_effect(a: list[float]) -> None:
    comp = compare_series("k", a, a, n_boot=200, seed=5)
    assert comp.verdict == "no_effect"
    assert comp.mean_delta == 0.0


@given(a=_SERIES, b=_SERIES, min_paired=st.integers(min_value=3, max_value=200))
@settings(max_examples=30, deadline=None, derandomize=True)
def test_small_n_always_insufficient(a: list[float], b: list[float], min_paired: int) -> None:
    n = min(len(a), len(b), min_paired - 1)
    if n < 2:
        return
    comp = compare_series("k", a[:n], b[:n], min_paired=min_paired, n_boot=50, seed=1)
    assert comp.verdict == "insufficient paired observations"
    assert comp.n_paired == n
