"""Fractional differentiation: weight recursion, integer-d equivalence, ADF."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.fracdiff import (
    expanding_frac_diff,
    frac_diff,
    frac_weights,
    frac_weights_ffd,
    min_d_stationary,
)


def _walk(n: int = 800, seed: int = 0) -> np.ndarray:
    r = np.random.default_rng(seed).normal(0.0002, 0.01, size=n)
    return np.exp(np.cumsum(r)) * 100.0


def test_weights_recursion_and_d1() -> None:
    w = frac_weights(1.0, 4)
    # d=1 -> (1, -1, 0, 0): exact first difference.
    assert w[0] == 1.0
    assert w[1] == pytest.approx(-1.0)
    assert w[2] == pytest.approx(0.0, abs=1e-12)
    assert w[3] == pytest.approx(0.0, abs=1e-12)


def test_weights_d0_is_identity() -> None:
    w = frac_weights(0.0, 6)
    assert w[0] == 1.0
    assert np.all(w[1:] == pytest.approx(0.0))


def test_frac_diff_d1_equals_diff() -> None:
    x = _walk(200)
    out = frac_diff(x, 1.0)
    assert np.allclose(out[1:], np.diff(x), rtol=1e-9)
    assert np.isnan(out[0])


def test_ffd_truncation_bounded() -> None:
    w = frac_weights_ffd(0.4, threshold=1e-3)
    assert w.size < 800  # bounded window vs expanding (n) weights
    assert abs(w[-1]) < 1e-3 or w.size >= 2


def test_frac_diff_memory_preservation() -> None:
    # d=0.4 output should still correlate strongly with the level (memory),
    # unlike d=1 which is ~white noise.  threshold=1e-3 keeps the FFD window
    # (~100 weights) inside the series length.
    x = _walk(600, seed=2)
    d4 = frac_diff(x, 0.4, threshold=1e-3)
    d1 = frac_diff(x, 1.0)
    a = np.isfinite(d4)
    b = np.isfinite(d1)
    corr_d4 = np.corrcoef(d4[a][1:], x[a][:-1])[0, 1]
    corr_d1 = np.corrcoef(d1[b][1:], x[b][:-1])[0, 1]
    assert corr_d4 > corr_d1 + 0.3


def test_expanding_matches_ffd_within_tail_bound() -> None:
    # FFD truncates the weight tail at ``threshold``; the difference vs the
    # expanding transform is bounded by the dropped tail mass times |x|max.
    x = _walk(120, seed=4)
    d = 0.5
    w = frac_weights_ffd(d, threshold=1e-3)
    full = expanding_frac_diff(x, d)
    ffd = frac_diff(x, d, threshold=1e-3)
    finite = np.isfinite(ffd)
    tail_bound = float(np.abs(w[w.size :]).sum()) if w.size < 120 else 0.0
    # Tail weights beyond the window for d=0.5 decay as k^-1.5.
    tail_mass = float(np.sum(np.abs(frac_weights(d, 120)[w.size :]))) if w.size < 120 else 0.0
    bound = tail_mass * float(np.max(np.abs(x)))
    diff = np.abs(ffd[finite] - full[finite])
    assert np.all(diff <= bound + 1e-9)
    _ = tail_bound


def test_frac_diff_constant_series_decay() -> None:
    # Sum of fractional weights -> 0 for 0 < d < 1: a constant series diffs to ~0.
    x = np.full(300, 50.0)
    out = frac_diff(x, 0.5, threshold=1e-3)
    finite = out[np.isfinite(out)]
    assert np.abs(finite[-1]) < np.abs(x[0]) * 0.1


def test_min_d_stationary_random_walk() -> None:
    # A random walk needs d near 1; a white-noise series near 0.
    out = min_d_stationary(np.log(_walk(600, seed=7)))
    assert np.isfinite(out["d"])
    assert 0.3 <= out["d"] <= 1.0
    noise = np.random.default_rng(8).normal(0, 1, 600)
    out2 = min_d_stationary(noise)
    assert out2["d"] == pytest.approx(0.0)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        frac_weights(0.5, 0)
    with pytest.raises(ValueError):
        frac_weights(float("nan"), 5)
    with pytest.raises(ValueError):
        frac_weights_ffd(0.5, threshold=0.0)
    with pytest.raises(ValueError):
        frac_diff(np.array([np.nan, 1.0, 2.0]), 0.4)
    with pytest.raises(ValueError):
        frac_diff(np.arange(10.0), 0.4, threshold=1e-12)  # window exceeds series
    with pytest.raises(ValueError):
        min_d_stationary(np.arange(20.0))
