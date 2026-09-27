"""Synthetic throughput benchmarks.

Not part of the default testpaths. CI runs the unmarked cases; ``perf_full``
is the larger tier.

Reproduce the committed baseline (this writes the JSON the regression gate
reads)::

    uv run pytest tests/perf -q -o addopts= --benchmark-only \\
        --benchmark-min-rounds=5 --benchmark-max-time=2 \\
        --benchmark-warmup=on --benchmark-disable-gc \\
        --benchmark-json=tests/perf/baselines/baseline.json

Fast subset (GitHub CI)::

    uv run pytest tests/perf -q -m "not perf_full" -o addopts= --benchmark-only \\
        --benchmark-min-rounds=5 --benchmark-max-time=2 \\
        --benchmark-warmup=on --benchmark-disable-gc \\
        --benchmark-json=perf-ci.json
    uv run python tests/perf/check_regression.py perf-ci.json \\
        tests/perf/baselines/baseline.json --threshold 0.50

Timings are SYNTHETIC workload measurements. They are not market evidence
and not a performance claim about live trading.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from tests.perf.synthetic import (
    daily_ohlcv,
    dip_closes,
    minute_bars,
    write_dip_dir,
    write_vendor_month,
)

from fx1.bench.dip import detect_dip_events
from fx1.bench.run import run_dip_bench
from quant_fund.backtest.engine import run_backtest
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig
from quant_fund.data.adapters.hf_ohlcv_1m import (
    read_ohlcv_1m,
    resample_ohlcv,
    security_master_from_bars,
)
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.engine import (
    CROSS_SECTIONAL_COLUMNS,
    build_features,
    compute_base_features,
)

pytestmark = pytest.mark.synthetic


def _clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _backtest_inputs(n_symbols: int, n_days: int) -> tuple[pl.DataFrame, pl.DataFrame, AppConfig]:
    bars = daily_ohlcv(n_symbols, n_days).with_columns(
        (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.lit(0.02).alias("vol_20"),
    )
    weights = bars.select("event_time", "security_id").with_columns(
        pl.when(pl.col("security_id").str.ends_with("0"))
        .then(0.0)
        .otherwise(0.6 / n_symbols)
        .alias("target_weight")
    )
    config = AppConfig(
        costs=CostConfig(
            commission_bps=1.0,
            half_spread_bps=1.0,
            impact_y=0.1,
            participation_limit=0.1,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=2.0,
            max_net=1.5,
            max_name=0.25,
            max_participation=0.5,
            max_predicted_vol=1.0,
            stale_price_bars=5,
        ),
    )
    return bars, weights, config


def test_calibration_matmul(benchmark: object) -> None:
    """Machine-speed reference. Regression checks use time ratios against this."""
    rng = np.random.default_rng(0)
    left = rng.standard_normal((512, 512))
    right = rng.standard_normal((512, 512))

    def _run() -> np.ndarray:
        return left @ right

    benchmark(_run)


@pytest.mark.parametrize(
    ("n_symbols", "n_days"),
    [
        pytest.param(24, 252, id="24sym-252d"),
        pytest.param(200, 504, id="200sym-504d", marks=pytest.mark.perf_full),
    ],
)
def test_cross_section(benchmark: object, n_symbols: int, n_days: int) -> None:
    bars = daily_ohlcv(n_symbols, n_days)
    built = compute_base_features(bars, AppConfig())
    columns = [name for name in CROSS_SECTIONAL_COLUMNS if name in built.columns]
    benchmark(lambda: apply_cross_sectional(built, columns, 0.01))


@pytest.mark.parametrize(
    ("n_symbols", "n_days"),
    [
        pytest.param(24, 252, id="24sym-252d"),
        pytest.param(80, 756, id="80sym-756d", marks=pytest.mark.perf_full),
    ],
)
def test_build_features(benchmark: object, n_symbols: int, n_days: int) -> None:
    bars = daily_ohlcv(n_symbols, n_days)
    config = AppConfig()
    benchmark(lambda: build_features(bars, config))


@pytest.mark.parametrize(
    ("n_symbols", "n_sessions", "every"),
    [
        pytest.param(1, 252, "5m", id="1sym-1y-5m"),
        pytest.param(1, 252, "1d", id="1sym-1y-1d"),
        pytest.param(40, 20, "5m", id="40sym-20d-5m", marks=pytest.mark.perf_full),
    ],
)
def test_resample(benchmark: object, n_symbols: int, n_sessions: int, every: str) -> None:
    bars = minute_bars(n_symbols, n_sessions)
    benchmark(lambda: resample_ohlcv(bars, every))


@pytest.mark.parametrize(
    ("n_symbols", "n_sessions"),
    [
        pytest.param(8, 20, id="8sym-20d"),
        pytest.param(120, 40, id="120sym-40d", marks=pytest.mark.perf_full),
    ],
)
def test_security_master(benchmark: object, n_symbols: int, n_sessions: int) -> None:
    bars = minute_bars(n_symbols, n_sessions)
    benchmark(lambda: security_master_from_bars(bars, clock=_clock))


@pytest.mark.parametrize(
    ("n_symbols", "n_sessions"),
    [
        pytest.param(8, 20, id="8sym-20d"),
        pytest.param(40, 20, id="40sym-20d", marks=pytest.mark.perf_full),
    ],
)
def test_read_ohlcv_1m(
    benchmark: object,
    n_symbols: int,
    n_sessions: int,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    cache, symbols, start, end = write_vendor_month(
        tmp_path_factory.mktemp("ohlcv"), n_symbols, n_sessions
    )

    def _load() -> object:
        return read_ohlcv_1m(
            symbols=symbols,
            cache_dir=cache,
            start=start,
            end=end,
            revision="bench",
            allow_download=False,
            interval="1m",
            clock=_clock,
        )

    benchmark(_load)


@pytest.mark.parametrize(
    ("n_symbols", "n_days"),
    [
        pytest.param(8, 126, id="8sym-126d"),
        pytest.param(40, 252, id="40sym-252d", marks=pytest.mark.perf_full),
    ],
)
def test_backtest(benchmark: object, n_symbols: int, n_days: int) -> None:
    bars, weights, config = _backtest_inputs(n_symbols, n_days)
    benchmark(lambda: run_backtest(bars, weights, config, initial_nav=1_000_000.0))


@pytest.mark.parametrize(
    ("n_symbols", "n_days"),
    [
        pytest.param(40, 1000, id="40sym-1000d"),
        pytest.param(200, 2500, id="200sym-2500d", marks=pytest.mark.perf_full),
    ],
)
def test_dip_detect(benchmark: object, n_symbols: int, n_days: int) -> None:
    closes, dates = dip_closes(n_symbols, n_days)
    listed = [row.tolist() for row in closes]

    def _run() -> int:
        total = 0
        for i, series in enumerate(listed):
            total += len(detect_dip_events(series, dates, f"S{i:04d}"))
        return total

    benchmark(_run)


@pytest.mark.parametrize(
    ("n_symbols", "n_days"),
    [
        pytest.param(20, 800, id="20sym-800d"),
        pytest.param(80, 1500, id="80sym-1500d", marks=pytest.mark.perf_full),
    ],
)
def test_dip_bench(
    benchmark: object,
    n_symbols: int,
    n_days: int,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root = write_dip_dir(tmp_path_factory.mktemp("dip"), n_symbols, n_days)
    benchmark(lambda: run_dip_bench(root, threshold=0.10))
