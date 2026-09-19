"""Wave 33: Jobson–Korkie / circular_block / bootstrap_mean_ci edges.

Research diagnostics only — bootstrap/JK helpers are NOT live Sharpe or P&L claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.inference import (
    bootstrap_mean_ci,
    circular_block_indices,
    jobson_korkie_memmel,
)

# Explicit honesty stamp for overnight grind / catalog scanners.
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def test_jobson_korkie_memmel_short_n_and_equal_srs() -> None:
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, 0.5, 9, 0.2))
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, 0.5, 0, 0.0))
    theta, p = jobson_korkie_memmel(1.0, 1.0, 100, 0.5)
    assert theta == 0.0
    assert p == 1.0
    theta10, p10 = jobson_korkie_memmel(1.2, 0.8, 10, 0.3)
    assert math.isfinite(theta10) and 0.0 <= p10 <= 1.0


def test_jobson_korkie_memmel_bad_corr_and_nonfinite() -> None:
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, 0.5, 100, 1.5))
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, 0.5, 100, -1.1))
    assert all(math.isnan(x) for x in jobson_korkie_memmel(float("nan"), 0.5, 100, 0.0))
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, float("inf"), 100, 0.0))
    assert all(math.isnan(x) for x in jobson_korkie_memmel(1.0, 0.5, 100, float("nan")))
    # Boundary |corr|==1 remains defined
    th, p = jobson_korkie_memmel(1.2, 0.8, 100, 1.0)
    assert math.isfinite(th) and 0.0 <= p <= 1.0


def test_circular_block_indices_edges() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="block must be >= 1"):
        circular_block_indices(10, 0, rng)
    with pytest.raises(ValueError, match="block must be >= 1"):
        circular_block_indices(10, -3, rng)

    empty = circular_block_indices(0, 2, rng)
    assert empty.dtype == np.intp
    assert empty.size == 0

    idx = circular_block_indices(12, 5, np.random.default_rng(1))
    assert idx.shape == (12,)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < 12

    # block > n still returns length n via circular wrap
    wrap = circular_block_indices(4, 10, np.random.default_rng(2))
    assert wrap.shape == (4,)
    assert set(int(x) for x in wrap).issubset({0, 1, 2, 3})


def test_bootstrap_mean_ci_empty_short_and_nonfinite() -> None:
    lo, hi, mu = bootstrap_mean_ci(np.array([]), n_boot=50)
    assert all(math.isnan(x) for x in (lo, hi, mu))
    lo, hi, mu = bootstrap_mean_ci(np.array([1.0, 2.0, 3.0, 4.0]), n_boot=50)
    assert all(math.isnan(x) for x in (lo, hi, mu))
    # Non-finite dropped before n check → short → nan
    lo, hi, mu = bootstrap_mean_ci(np.array([1.0, np.nan, 2.0, np.inf]), n_boot=50)
    assert all(math.isnan(x) for x in (lo, hi, mu))

    x = np.arange(20, dtype=float)
    lo, hi, mu = bootstrap_mean_ci(x, n_boot=80, seed=7)
    assert lo <= mu <= hi
    assert math.isclose(mu, float(np.mean(x)))
    # Explicit block override
    lo2, hi2, mu2 = bootstrap_mean_ci(x, n_boot=80, block=3, seed=7)
    assert lo2 <= mu2 <= hi2
    assert mu2 == mu


def test_bootstrap_mean_ci_rejects_invalid_controls() -> None:
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_mean_ci(x, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_mean_ci(x, alpha=1.0)
    with pytest.raises(ValueError, match="block"):
        bootstrap_mean_ci(x, block=0)
