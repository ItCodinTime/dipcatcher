"""Philox streams are keyed by path index, not by worker schedule."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.mc_engine.philox import philox_normals, philox_raw, philox_uniforms
from quant_fund.mc_engine.variance import draw_standard_normals, sobol_normals


def test_same_index_is_bit_identical_and_order_independent() -> None:
    indices = np.arange(8, dtype=np.int64)
    forward = philox_raw(7, indices, 16, stream_id=1)
    backward = philox_raw(7, indices[::-1], 16, stream_id=1)
    assert np.array_equal(forward, backward[::-1])
    again = philox_raw(7, indices, 16, stream_id=1)
    assert np.array_equal(forward, again)
    assert not np.array_equal(forward[0], forward[1])


def test_stream_id_and_seed_change_the_words() -> None:
    indices = np.array([3], dtype=np.int64)
    base = philox_raw(1, indices, 4, stream_id=1)
    assert not np.array_equal(base, philox_raw(1, indices, 4, stream_id=2))
    assert not np.array_equal(base, philox_raw(2, indices, 4, stream_id=1))


def test_split_indices_match_the_full_draw() -> None:
    indices = np.arange(20, dtype=np.int64)
    full = philox_normals(11, indices, 6, stream_id=1)
    left = philox_normals(11, indices[:7], 6, stream_id=1)
    right = philox_normals(11, indices[7:], 6, stream_id=1)
    assert np.array_equal(full, np.concatenate([left, right], axis=0))


def test_uniforms_lie_in_the_open_unit_interval() -> None:
    uniforms = philox_uniforms(0, np.arange(5), 9, stream_id=4)
    assert uniforms.shape == (5, 9)
    assert np.all(uniforms > 0.0)
    assert np.all(uniforms < 1.0)


def test_normals_are_standard_on_a_modest_sample() -> None:
    normals = philox_normals(3, np.arange(4000), 1, stream_id=1)[:, 0]
    assert abs(float(normals.mean())) < 0.05
    assert abs(float(normals.std(ddof=1)) - 1.0) < 0.05


def test_antithetic_pairs_cancel() -> None:
    indices = np.arange(0, 100, dtype=np.int64)
    shocks, weights = draw_standard_normals(
        indices,
        n_steps=3,
        n_factors=1,
        seed=9,
        shock_mode="antithetic",
        importance_shift=0.0,
        qmc_scramble=False,
        qmc_seed=0,
    )
    assert weights is None
    paired = shocks[:, 0, 0].reshape(-1, 2)
    assert np.all(paired[:, 0] == -paired[:, 1])


def test_sobol_chunks_match_the_full_sequence() -> None:
    full = sobol_normals(0, 16, 4, seed=5, scramble=True)
    left = sobol_normals(0, 6, 4, seed=5, scramble=True)
    right = sobol_normals(6, 10, 4, seed=5, scramble=True)
    assert np.array_equal(full, np.concatenate([left, right], axis=0))
    again = sobol_normals(6, 10, 4, seed=5, scramble=True)
    assert np.array_equal(right, again)


def test_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        philox_raw(-1, np.arange(1), 1, stream_id=1)
    with pytest.raises(ValueError):
        philox_raw(0, np.array([-1], dtype=np.int64), 1, stream_id=1)
    with pytest.raises(ValueError):
        sobol_normals(0, 1, 30_000, seed=0, scramble=False)


@given(st.integers(min_value=0, max_value=50), st.integers(min_value=0, max_value=20))
@settings(max_examples=20)
def test_repeated_index_is_stable(index: int, seed: int) -> None:
    indices = np.array([index], dtype=np.int64)
    first = philox_raw(seed, indices, 4, stream_id=1)
    second = philox_raw(seed, indices, 4, stream_id=1)
    assert np.array_equal(first, second)
