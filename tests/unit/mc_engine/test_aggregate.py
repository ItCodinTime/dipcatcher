"""Welford, t-digest, and P²: defined merges, bounded sketches."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.mc_engine.aggregate import P2Quantile, TDigest, Welford, merge_welford


def test_welford_matches_numpy_and_chan_merge_is_repeatable() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(loc=1.5, scale=2.0, size=500)
    acc = Welford()
    acc.add_all(values)
    assert acc.n == 500
    assert acc.mean == pytest.approx(float(values.mean()), rel=1e-12, abs=1e-12)
    assert acc.variance == pytest.approx(float(np.var(values, ddof=1)), rel=1e-10, abs=1e-10)
    cuts = [values[:100], values[100:250], values[250:]]
    states = []
    for part in cuts:
        part_acc = Welford()
        part_acc.add_all(part)
        states.append(part_acc.state())
    folded = states[0]
    for state in states[1:]:
        folded = merge_welford(folded, state)
    again = merge_welford(states[0], merge_welford(states[1], states[2]))
    # Different association; the engine always uses one order. That order is stable.
    same = states[0]
    for state in states[1:]:
        same = merge_welford(same, state)
    assert folded == same
    assert folded[0] == 500
    assert folded[1] == pytest.approx(acc.mean, rel=1e-12, abs=1e-12)
    assert again[0] == 500


def test_empty_side_of_a_merge_is_the_other_side() -> None:
    state = (3, 1.0, 2.0)
    assert merge_welford((0, 0.0, 0.0), state) == state
    assert merge_welford(state, (0, 0.0, 0.0)) == state


def test_tdigest_preserves_weight_and_merges_in_input_order() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=5_000)
    digest = TDigest(50.0)
    digest.add_all(values)
    assert digest.total_weight == pytest.approx(5000.0)
    assert digest.means.size < 20 * 50
    left = TDigest(50.0)
    right = TDigest(50.0)
    left.add_all(values[:2_000])
    right.add_all(values[2_000:])
    merged = left.merge(right)
    merged_again = left.merge(right)
    assert np.array_equal(merged.means, merged_again.means)
    assert np.array_equal(merged.weights, merged_again.weights)
    assert merged.total_weight == pytest.approx(5000.0)
    assert abs(merged.quantile(0.5) - float(np.quantile(values, 0.5))) < 0.15
    assert merged.quantile(0.1) < merged.quantile(0.9)


def test_psquare_median_of_uniform_is_near_one_half() -> None:
    rng = np.random.default_rng(2)
    estimator = P2Quantile(0.5)
    estimator.add_all(rng.random(20_000))
    assert abs(estimator.value - 0.5) < 0.03


def test_psquare_needs_five_points_and_rejects_bad_probability() -> None:
    estimator = P2Quantile(0.9)
    estimator.add_all(np.arange(4, dtype=float))
    with pytest.raises(ValueError):
        _ = estimator.value
    with pytest.raises(ValueError):
        P2Quantile(0.0)
    with pytest.raises(ValueError):
        TDigest(1.0)


@given(
    st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=30,
    )
)
@settings(max_examples=20)
def test_welford_variance_matches_numpy(values: list[float]) -> None:
    acc = Welford()
    acc.add_all(np.asarray(values, dtype=np.float64))
    assert acc.variance == pytest.approx(float(np.var(values, ddof=1)), rel=1e-8, abs=1e-8)
