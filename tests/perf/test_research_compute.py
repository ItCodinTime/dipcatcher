"""SYNTHETIC wall-clock checks for research compute. Not market evidence.

These kernels sit in the ``perf_full`` tier. The CI perf job runs only the
unmarked ``tests/perf`` cases and fails when one of them has no entry in
``tests/perf/baselines/baseline.json``.

Reproduce with the same flags as the suite::

    pytest tests/perf/test_research_compute.py -q -o addopts= --benchmark-only \\
        --benchmark-min-rounds=5 --benchmark-max-time=2 \\
        --benchmark-warmup=on --benchmark-disable-gc
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.volatility import HARVol, RollingVol, ewma_variance

pytestmark = [pytest.mark.synthetic, pytest.mark.perf_full]


def test_bench_ewma_variance(benchmark: object) -> None:
    series = np.random.default_rng(0).normal(size=50_000)
    benchmark(ewma_variance, series, 0.94)


def test_bench_rolling_vol(benchmark: object) -> None:
    series = np.random.default_rng(1).normal(size=50_000)
    model = RollingVol(20)
    benchmark(model.predict_from_returns, series)


def test_bench_har_design(benchmark: object) -> None:
    series = np.abs(np.random.default_rng(2).normal(size=20_000))
    benchmark(HARVol.har_design, series)


def test_bench_ewma_variance_long(benchmark: object) -> None:
    series = np.random.default_rng(3).normal(size=200_000)
    benchmark(ewma_variance, series, 0.94)
