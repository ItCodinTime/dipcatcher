"""P6.2 statistical-core audit: known-answer tests.

Deterministic, offline, SYNTHETIC fixtures only — research-diagnostic
machinery, never a live Sharpe / P&L claim. Each test pins either a
hand-computed value (closed form / reference implementation) or an exact
structural identity of the cited method.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from arch.bootstrap import optimal_block_length as arch_optimal_block_length

from quant_fund.metrics.inference import optimal_block_length
from quant_fund.metrics.scoring import coverage


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# --- Block-length selection (PW2004) -----------------------------------------


def test_optimal_block_length_matches_pw2004_reference() -> None:
    """Regression test for the c_i=2 stationary-bootstrap constant and the
    K_N/m_max/B_max rules: must match ``arch.bootstrap.optimal_block_length``
    (the canonical PW2004 implementation) whenever the value is >= 1."""
    for seed in range(3):
        for n, phi in [(500, 0.5), (2000, 0.2), (4000, 0.85), (4000, -0.3)]:
            x = _ar1(phi, n, seed)
            ref = float(arch_optimal_block_length(x)["stationary"].iloc[0])
            got = optimal_block_length(x)
            if ref >= 1.0:
                assert got == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_optimal_block_length_negative_dependence_not_floored_to_one() -> None:
    """Negative autocorrelation still implies dependent data: the PW2004
    optimum uses G^2, so a dominant negative rho must NOT collapse to 1.0."""
    x = _ar1(-0.3, 4000, 5)
    ref = float(arch_optimal_block_length(x)["stationary"].iloc[0])
    assert ref > 3.0  # sanity on the reference itself
    assert optimal_block_length(x) == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_optimal_block_length_edges_unchanged() -> None:
    assert optimal_block_length(np.ones(500)) == 1.0
    assert np.isnan(optimal_block_length(np.arange(5.0)))
    assert np.isnan(optimal_block_length(np.array([])))
    # Bounded by B_max = ceil(min(3 sqrt n, n/3))
    x = _ar1(0.98, 500, 9)
    assert optimal_block_length(x) <= math.ceil(min(3.0 * math.sqrt(500), 500 / 3.0))


def test_coverage_masks_nonfinite() -> None:
    """Regression: non-finite observations are masked, not counted as misses."""
    assert coverage(
        np.array([0.5, np.nan]), np.array([0.0, 0.0]), np.array([1.0, 1.0])
    ) == pytest.approx(1.0)
    assert np.isnan(coverage(np.array([np.nan]), np.array([0.0]), np.array([1.0])))
    assert coverage(
        np.array([0.0, 5.0]), np.array([-1.0, 0.0]), np.array([0.5, 2.0])
    ) == pytest.approx(0.5)
