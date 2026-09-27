"""Unit pins for CSCV splits and PBO (PROOFCORE W4 §7.3, §12)."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from quant_fund.proofcore.contracts import RealityFilterError
from quant_fund.reality.cscv import cscv_pbo, cscv_splits, pbo_from_performance


def test_split_count_is_binomial() -> None:
    for s in (4, 6, 8, 16):
        splits = cscv_splits(4 * s, s_blocks=s)
        assert len(splits) == comb(s, s // 2)


def test_splits_partition_all_periods() -> None:
    n, s = 32, 8
    for is_idx, oos_idx in cscv_splits(n, s_blocks=s):
        assert is_idx.size == oos_idx.size == n // 2
        assert sorted(np.concatenate([is_idx, oos_idx]).tolist()) == list(range(n))
        assert np.all(np.diff(is_idx) > 0) and np.all(np.diff(oos_idx) > 0)


def test_splits_fail_closed_on_indivisible() -> None:
    with pytest.raises(RealityFilterError):
        cscv_splits(100, s_blocks=16)
    with pytest.raises(RealityFilterError):
        cscv_splits(16, s_blocks=5)  # odd S
    with pytest.raises(RealityFilterError):
        cscv_splits(8, s_blocks=16)  # fewer periods than blocks


def _brute_force_pbo(is_perf: np.ndarray, oos_perf: np.ndarray) -> float:
    """Dead-simple reference: fraction of combos where IS-best is below OOS median."""
    flags = []
    for c in range(is_perf.shape[0]):
        best = int(np.argmax(is_perf[c]))
        flags.append(oos_perf[c, best] < np.median(oos_perf[c]))
    return float(np.mean(flags))


def test_pbo_brute_force_parity_small() -> None:
    """S=8, n_trials=6 enumeration: reference implementation must agree."""
    rng = np.random.default_rng(3)
    n_comb = comb(8, 4)
    is_perf = rng.normal(size=(n_comb, 6))
    oos_perf = rng.normal(size=(n_comb, 6))
    out = pbo_from_performance(is_perf, oos_perf)
    assert out["n_combinations"] == n_comb
    assert out["n_dropped"] == 0
    assert out["pbo"] == pytest.approx(_brute_force_pbo(is_perf, oos_perf), abs=1e-12)
    assert len(out["logits"]) == n_comb
    assert 0.0 <= out["pbo"] <= 1.0


def test_pbo_extreme_pins() -> None:
    # IS-best always OOS-worst -> pbo == 1
    n_comb, n_tr = 10, 4
    is_perf = np.tile(np.arange(n_tr, dtype=float), (n_comb, 1))
    oos_perf = np.tile(np.arange(n_tr, 0, -1, dtype=float), (n_comb, 1))
    assert pbo_from_performance(is_perf, oos_perf)["pbo"] == 1.0
    # IS-best always OOS-best -> pbo == 0
    assert pbo_from_performance(is_perf, is_perf.copy())["pbo"] == 0.0


def test_pbo_nan_rows_dropped_and_counted() -> None:
    rng = np.random.default_rng(5)
    is_perf = rng.normal(size=(6, 4))
    oos_perf = rng.normal(size=(6, 4))
    is_perf[0, 2] = np.nan
    oos_perf[1, 0] = np.inf
    out = pbo_from_performance(is_perf, oos_perf)
    assert out["n_dropped"] == 2
    assert out["n_combinations"] == 4
    assert np.isfinite(out["pbo"])


def test_pbo_fail_closed_below_min_combinations() -> None:
    rng = np.random.default_rng(9)
    is_perf = rng.normal(size=(3, 4))
    oos_perf = rng.normal(size=(3, 4))
    out = pbo_from_performance(is_perf, oos_perf)
    assert np.isnan(out["pbo"])
    assert out["n_combinations"] == 3
    # Shape violations -> NaN, never 0.0
    assert np.isnan(pbo_from_performance(np.zeros((5, 1)), np.zeros((5, 1)))["pbo"])
    assert np.isnan(pbo_from_performance(np.zeros((5, 4)), np.zeros((4, 4)))["pbo"])


def test_cscv_pbo_end_to_end_overfitted_panel() -> None:
    """Pure-noise trials: IS-best is rightfully OOS-median-ish, pbo bounded in [0,1]."""
    rng = np.random.default_rng(17)
    panel = rng.normal(0.0, 0.01, size=(64, 6))
    out = cscv_pbo(panel, s_blocks=8)
    assert out["n_splits"] == comb(8, 4)
    assert 0.0 <= out["pbo"] <= 1.0
    assert all(np.isfinite(x) for x in out["logits"])


def test_cscv_pbo_fail_closed_shape() -> None:
    with pytest.raises(RealityFilterError):
        cscv_pbo(np.zeros((10, 6)), s_blocks=16)
    with pytest.raises(RealityFilterError):
        cscv_pbo(np.zeros((64,)), s_blocks=8)


def test_s_combinations_are_exactly_half_choices() -> None:
    """The split enumeration must equal itertools.combinations(range(S), S/2)."""
    s = 8
    n = 8 * 4
    block_len = n // s
    splits = cscv_splits(n, s_blocks=s)
    expected = list(combinations(range(s), s // 2))
    assert len(splits) == len(expected)
    for (is_idx, _), combo in zip(splits, expected, strict=True):
        first_blocks = sorted({int(i // block_len) for i in is_idx})
        assert first_blocks == sorted(combo)
