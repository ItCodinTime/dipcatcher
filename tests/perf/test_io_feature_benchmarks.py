"""Before/after wall time for data IO and feature hot paths.

Not collected by the default pytest testpaths. SYNTHETIC timings are speed
measurements, not market evidence. The reference functions are the previous
Python loops; the library functions are the vectorized or numba paths.

Run with: ``uv run pytest tests/perf/test_io_feature_benchmarks.py -m perf_full``
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import UniverseConfig
from quant_fund.data.universe import build_membership_panel
from quant_fund.features.bars import tick_run_bars
from quant_fund.features.cycles import cycle_periodogram
from quant_fund.features.indicators import bollinger, kama
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars

pytestmark = pytest.mark.perf_full


def _prices(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n)))


def _reference_run_bars(prices: np.ndarray, sizes: np.ndarray, expected_ticks: int = 40) -> int:
    diff = np.diff(prices, prepend=prices[0])
    signs = np.sign(diff)
    for t in range(1, signs.size):
        if signs[t] == 0.0:
            signs[t] = signs[t - 1]
    signs[0] = 1.0 if signs[0] >= 0.0 else -1.0
    alpha = 0.1
    e_ticks = float(expected_ticks)
    start = 0
    e_share = 0.5
    count = 0
    for t in range(prices.size):
        is_buy = 1.0 if signs[t] > 0.0 else 0.0
        e_share = (1.0 - alpha) * e_share + alpha * is_buy
        run = max(
            float(np.sum(signs[start : t + 1] > 0.0)), float(np.sum(signs[start : t + 1] < 0.0))
        )
        if run >= max(e_ticks * max(e_share, 1.0 - e_share), 2.0):
            count += 1
            start = t + 1
    return count


def _reference_bollinger_std(close: np.ndarray, window: int = 20) -> float:
    total = 0.0
    for i in range(window - 1, close.size):
        total += float(close[i - window + 1 : i + 1].std(ddof=0))
    return total


def _reference_goertzel(values: np.ndarray) -> float:
    demeaned = values - values.mean()
    total = 0.0
    n2 = float(values.size * values.size)
    for period in range(6, values.size // 2 + 1):
        coeff = 2.0 * np.cos(2.0 * np.pi / period)
        prev = 0.0
        prev2 = 0.0
        for sample in demeaned:
            state = float(sample) + coeff * prev - prev2
            prev2 = prev
            prev = state
        total += (prev2 * prev2 + prev * prev - coeff * prev * prev2) / n2
    return total


def test_tick_run_bars_faster_than_slice_loop(benchmark: object) -> None:
    prices = _prices(40_000)
    sizes = np.ones(40_000)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: tick_run_bars(prices, sizes, 40), rounds=5, iterations=1, warmup_rounds=1
    )


def test_tick_run_reference_loop(benchmark: object) -> None:
    prices = _prices(40_000)
    sizes = np.ones(40_000)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: _reference_run_bars(prices, sizes, 40),
        rounds=3,
        iterations=1,
        warmup_rounds=0,
    )


def test_bollinger_faster_than_window_loop(benchmark: object) -> None:
    prices = _prices(40_000, seed=2)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: bollinger(prices, 20), rounds=5, iterations=1, warmup_rounds=1
    )


def test_bollinger_reference_loop(benchmark: object) -> None:
    prices = _prices(40_000, seed=2)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: _reference_bollinger_std(prices, 20),
        rounds=3,
        iterations=1,
        warmup_rounds=0,
    )


def test_kama_vectorized(benchmark: object) -> None:
    prices = _prices(40_000, seed=3)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: kama(prices, 10), rounds=5, iterations=1, warmup_rounds=1
    )


def test_periodogram_vectorized(benchmark: object) -> None:
    prices = _prices(1024, seed=4)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: cycle_periodogram(prices), rounds=5, iterations=1, warmup_rounds=1
    )


def test_periodogram_reference(benchmark: object) -> None:
    prices = _prices(1024, seed=4)
    benchmark.pedantic(  # type: ignore[attr-defined]
        lambda: _reference_goertzel(prices), rounds=2, iterations=1, warmup_rounds=0
    )


def test_membership_and_l2_panel(benchmark: object) -> None:
    rng = np.random.default_rng(5)
    n_names, n_days = 30, 80
    times = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_days)]
    security_id: list[str] = []
    close: list[float] = []
    volume: list[float] = []
    event_time: list[datetime] = []
    for i in range(n_names):
        path = 15.0 + i * 0.1 + np.cumsum(rng.normal(0.0, 0.02, size=n_days))
        vol = rng.uniform(1e5, 5e5, size=n_days)
        for t, px, sz in zip(times, path, vol, strict=True):
            security_id.append(f"S{i:02d}")
            event_time.append(t)
            close.append(float(abs(px) + 5.0))
            volume.append(float(sz))
    bars = pl.DataFrame(
        {
            "security_id": security_id,
            "event_time": event_time,
            "available_time": event_time,
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": volume,
        }
    )
    master = pl.DataFrame(
        {
            "security_id": [f"S{i:02d}" for i in range(n_names)],
            "exchange": ["XNYS"] * n_names,
            "sector": ["tech"] * n_names,
            "security_type": ["common_stock"] * n_names,
            "ticker": [f"T{i}" for i in range(n_names)],
        }
    )
    config = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=10,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )

    def _both() -> int:
        panel = build_membership_panel(bars, master, times[20:], config)
        book = synthesize_l2_from_bars(bars.head(400), depth=5, seed=7)
        return panel.height + book.height

    benchmark.pedantic(_both, rounds=3, iterations=1, warmup_rounds=1)  # type: ignore[attr-defined]
